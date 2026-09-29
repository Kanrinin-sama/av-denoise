use av_decoders::Decoder;
use av_denoise_core::{Depth, Planes};
use crossbeam_channel::{Receiver, Sender};

use crate::SceneLayout;
use crate::budget::{FramePermits, Permit};
use crate::cancel::Cancel;
use crate::planes::{planes_from_v_frame_u8, planes_from_v_frame_u16};

/// One decoded frame, staged for the worker that claims its scene.
pub(crate) struct StagedFrame {
    pub(crate) global_idx: u64,
    pub(crate) planes: Planes,
    pub(crate) permit: Permit,
}

/// One scene, offered to whichever worker is free.
///
/// `frames` closes when the scene has no more frames, which is how a
/// worker knows to flush.
pub(crate) struct SceneJob {
    pub(crate) scene_idx: u32,
    pub(crate) frames: Receiver<StagedFrame>,
}

/// What a dispatcher stages frames through.
pub(crate) struct Staging<'a> {
    pub(crate) jobs: &'a Sender<SceneJob>,
    pub(crate) permits: &'a FramePermits,
    pub(crate) closed: &'a Receiver<()>,
    pub(crate) cancel: &'a Cancel,
}

/// Reads every frame in order and offers each scene to the worker pool.
///
/// A scene's frames go into a channel of their own. Dropping that
/// channel's sender is what tells the claiming worker the scene has
/// ended.
///
/// Each staged frame holds a permit from here until the consumer has
/// taken it, which is the only bound on frames in flight. The permit
/// is taken just before the send rather than before the decode, so a
/// phantom frame never takes one and at most one decoded frame is
/// transient outside the budget.
fn stage_frames<I>(frames: I, scenes: &SceneLayout, staging: &Staging<'_>) -> Result<(), anyhow::Error>
where
    I: Iterator<Item = Result<Planes, anyhow::Error>>,
{
    let mut scene_idx = 0usize;
    let mut next_boundary = scenes.scene_starts[1];
    let mut current: Option<(usize, Sender<StagedFrame>)> = None;

    // The iterator yields emitted frames in order, so position is the
    // emitted index.
    for (g, planes) in frames.enumerate() {
        let planes = planes?;

        while g >= next_boundary && scene_idx + 1 < scenes.scene_count() {
            scene_idx += 1;
            next_boundary = scenes.scene_starts[scene_idx + 1];
        }

        if !matches!(&current, Some((idx, _)) if *idx == scene_idx) {
            let (tx, rx) = crossbeam_channel::unbounded::<StagedFrame>();

            // Dropping the previous scene's sender ends that scene, which
            // frees the worker holding it to claim this one. The queue is a
            // rendezvous, so offering the job first would deadlock whenever
            // every worker is busy.
            drop(current.take());

            staging
                .jobs
                .send(SceneJob {
                    scene_idx: scene_idx as u32,
                    frames: rx,
                })
                .map_err(|_| anyhow::anyhow!("worker pool disconnected"))?;

            current = Some((scene_idx, tx));
        }

        let permit = staging.permits.acquire(staging.closed, staging.cancel)?;

        let (_, tx) = current
            .as_ref()
            .expect("a scene sender exists after the check above");

        tx.send(StagedFrame {
            global_idx: g as u64,
            planes,
            permit,
        })
        .map_err(|_| anyhow::anyhow!("the worker holding scene {scene_idx} disconnected"))?;
    }

    Ok(())
}

/// Stages every frame `decoder` reads.
pub(crate) fn dispatch_frames(
    decoder: &mut Decoder,
    scenes: &SceneLayout,
    staging: &Staging<'_>,
) -> Result<(), anyhow::Error> {
    stage_frames(emitted_frames(decoder, scenes), scenes, staging)
}

/// Decodes the source ranges in order, reading past phantom frames.
///
/// Every frame is read, phantom or not, because the decoder walks the
/// file in order and cannot be told to skip one.
fn emitted_frames<'a>(
    decoder: &'a mut Decoder,
    scenes: &SceneLayout,
) -> impl Iterator<Item = Result<Planes, anyhow::Error>> + use<'a> {
    let layout = scenes.layout;
    let ranges = scenes.source_ranges.clone();
    let phantom = scenes.phantom.clone();
    let mut range_index = 0usize;
    let mut remaining = 0usize;
    let mut raw_index = 0usize;
    std::iter::from_fn(move || -> Option<Result<Planes, anyhow::Error>> {
        loop {
            if remaining == 0 {
                let &(start, end) = ranges.get(range_index)?;
                if let Err(error) = decoder.seek_video_frame(start) {
                    return Some(Err(error.into()));
                }
                raw_index = start;
                remaining = end - start;
                range_index += 1;
            }
            remaining -= 1;
            let current = raw_index;
            raw_index += 1;
            if phantom.contains(&current) {
                let skipped = match layout.depth {
                    Depth::Eight => decoder.read_video_frame::<u8>().map(|_| ()),
                    Depth::Ten | Depth::Twelve => decoder.read_video_frame::<u16>().map(|_| ()),
                };
                if let Err(error) = skipped {
                    return Some(Err(error.into()));
                }
                continue;
            }
            break;
        }
        match layout.depth {
            Depth::Eight => Some(
                decoder
                    .read_video_frame::<u8>()
                    .map_err(Into::into)
                    .and_then(|frame| planes_from_v_frame_u8(&frame, layout)),
            ),
            Depth::Ten | Depth::Twelve => Some(
                decoder
                    .read_video_frame::<u16>()
                    .map_err(Into::into)
                    .and_then(|frame| planes_from_v_frame_u16(&frame, layout)),
            ),
        }
    })
}
