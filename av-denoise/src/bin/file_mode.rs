use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{IsTerminal, stdout};
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread;
use std::time::Duration;

use av_decoders::{Decoder, DecoderError, Rational32};
use av_denoise::{Depth, FrameLayout, PlanarDenoiser, PlaneOptions, Planes, Subsampling, push_needs_retry};
use av_scenechange::{DetectionOptions, new_detector};
use indicatif::ProgressBar;
use v_frame::frame::Frame;
use v_frame::pixel::Pixel;
use y4m::Frame as Y4mFrame;

use crate::cli::RunOptions;
use crate::frame_index;
use crate::progress::{self, denoise_bar_visible, denoise_progress_bar};
use crate::y4m_format::subsampling_to_y4m;

/// Target ceiling for CPU-side frame buffers held in flight. Channel
/// depths shrink to stay under this when frames are large.
const FRAME_MEMORY_BUDGET_BYTES: usize = 1 << 30;

/// Channel depths used when frames are small enough to afford them.
const FRAME_CHANNEL_DEPTH_MAX: usize = 8;
const OUTPUT_CHANNEL_DEPTH_MAX: usize = 32;

/// Floors that keep the pipeline from starving no matter the frame size.
const FRAME_CHANNEL_DEPTH_MIN: usize = 2;
const OUTPUT_CHANNEL_DEPTH_MIN: usize = 4;

/// Channel depths for one run, with the frame counts they imply.
#[derive(Debug, Clone, Copy)]
struct ChannelBudget {
    frame_depth: usize,
    output_depth: usize,
}

/// Picks channel depths so the frames held in flight stay near
/// [`FRAME_MEMORY_BUDGET_BYTES`].
///
/// Small frames keep the maximum depths. Large frames scale both depths
/// down together, never below their floors.
fn channel_budget(layout: FrameLayout, workers: usize) -> ChannelBudget {
    let frame_bytes = layout.luma_bytes() + 2 * layout.chroma_bytes();
    let max_frames = workers * FRAME_CHANNEL_DEPTH_MAX + OUTPUT_CHANNEL_DEPTH_MAX;

    let affordable = FRAME_MEMORY_BUDGET_BYTES / frame_bytes.max(1);

    let (frame_depth, output_depth) = if max_frames <= affordable {
        (FRAME_CHANNEL_DEPTH_MAX, OUTPUT_CHANNEL_DEPTH_MAX)
    } else {
        let scale = affordable as f64 / max_frames as f64;
        let frame_depth =
            ((FRAME_CHANNEL_DEPTH_MAX as f64 * scale).floor() as usize).max(FRAME_CHANNEL_DEPTH_MIN);
        let output_depth =
            ((OUTPUT_CHANNEL_DEPTH_MAX as f64 * scale).floor() as usize).max(OUTPUT_CHANNEL_DEPTH_MIN);
        (frame_depth, output_depth)
    };

    ChannelBudget {
        frame_depth,
        output_depth,
    }
}

#[derive(Clone, Copy)]
struct VideoLayout {
    layout: FrameLayout,
    framerate: Rational32,
    total_frames: Option<usize>,
}

pub fn run_file(opts: &RunOptions, input: &Path, workers: usize) -> Result<(), anyhow::Error> {
    if workers == 0 {
        anyhow::bail!("--workers must be at least 1");
    }

    let mut decoder = Decoder::from_file(input)?;
    let details = *decoder.get_video_details();
    let phantom_frames = frame_index::read_index(&mut decoder)
        .map(|index| frame_index::phantom_indices(&index))
        .unwrap_or_default();
    let depth = Depth::from_bits(details.bit_depth)?;
    let video = VideoLayout {
        layout: FrameLayout {
            width: details.width as u32,
            height: details.height as u32,
            subsampling: subsampling_from_av_decoders(details.chroma_sampling)?,
            depth,
        },
        framerate: details.frame_rate,
        total_frames: details.total_frames.map(|total| total - phantom_frames.len()),
    };

    if video.total_frames == Some(0) {
        anyhow::bail!("{} holds no decodable frames", input.display());
    }

    let is_terminal = std::io::stderr().is_terminal();

    tracing::info!(
        width = video.layout.width,
        height = video.layout.height,
        subsampling = ?video.layout.subsampling,
        depth = ?video.layout.depth,
        total_frames = video.total_frames,
        workers,
        "running denoising with scene detection",
    );

    encode_scenes(
        &opts.planes,
        &mut decoder,
        video,
        &phantom_frames,
        workers,
        denoise_bar_visible(opts.progress, is_terminal),
    )
}

fn encode_scenes(
    opts: &PlaneOptions,
    decoder: &mut Decoder,
    video: VideoLayout,
    phantom_frames: &BTreeSet<usize>,
    workers: usize,
    visible: bool,
) -> Result<(), anyhow::Error> {
    let budget = channel_budget(video.layout, workers);

    let (worker_txs, worker_handles, out_rx) = spawn_workers(opts, video.layout, workers, budget);
    let (dispatch_count_tx, dispatch_count_rx) = sync_channel(1);
    let coordinator = spawn_coordinator(
        video.layout,
        video.framerate,
        out_rx,
        dispatch_count_rx,
        video.total_frames,
        visible,
    );

    let dispatch_result = match video.layout.depth {
        Depth::Eight => dispatch_frames::<u8, _>(
            decoder,
            video.layout,
            phantom_frames,
            &worker_txs,
            planes_from_v_frame_u8,
        ),
        Depth::Ten | Depth::Twelve => dispatch_frames::<u16, _>(
            decoder,
            video.layout,
            phantom_frames,
            &worker_txs,
            planes_from_v_frame_u16,
        ),
    };

    if let Ok(frame_count) = &dispatch_result {
        let _ = dispatch_count_tx.send(*frame_count);
    }
    drop(dispatch_count_tx);

    for tx in &worker_txs {
        let _ = tx.send(WorkerMsg::Eof);
    }

    drop(worker_txs);

    let mut worker_result: Result<(), anyhow::Error> = Ok(());
    for h in worker_handles {
        let result = h
            .join()
            .map_err(|e| anyhow::anyhow!("worker panicked: {e:?}"))
            .and_then(|result| result);
        if worker_result.is_ok() {
            worker_result = result;
        }
    }

    let coordinator_result = coordinator
        .join()
        .map_err(|e| anyhow::anyhow!("coordinator panicked: {e:?}"))
        .and_then(|result| result);

    dispatch_result?;
    worker_result?;
    coordinator_result
}

type WorkerJoin = thread::JoinHandle<Result<(), anyhow::Error>>;

/// Spawns `workers` worker threads.
///
/// Returns their input channels, their join handles, and the shared
/// output channel they emit denoised frames on.
fn spawn_workers(
    opts: &PlaneOptions,
    layout: FrameLayout,
    workers: usize,
    budget: ChannelBudget,
) -> (Vec<SyncSender<WorkerMsg>>, Vec<WorkerJoin>, Receiver<OutputMsg>) {
    let mut worker_txs: Vec<SyncSender<WorkerMsg>> = Vec::with_capacity(workers);
    let (out_tx, out_rx) = sync_channel::<OutputMsg>(budget.output_depth);
    let mut worker_handles: Vec<WorkerJoin> = Vec::with_capacity(workers);

    for _ in 0..workers {
        let (frame_tx, frame_rx) = sync_channel::<WorkerMsg>(budget.frame_depth);
        let opts = opts.clone();
        let out_tx = out_tx.clone();

        worker_txs.push(frame_tx);
        worker_handles.push(thread::spawn(move || run_worker(opts, layout, frame_rx, out_tx)));
    }

    // Drop the original sender so the channel closes once every worker
    // clone has terminated.
    drop(out_tx);

    (worker_txs, worker_handles, out_rx)
}

fn spawn_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: Receiver<OutputMsg>,
    dispatch_count: Receiver<usize>,
    progress_total: Option<usize>,
    visible: bool,
) -> thread::JoinHandle<Result<(), anyhow::Error>> {
    thread::spawn(move || run_coordinator(layout, framerate, rx, dispatch_count, progress_total, visible))
}

fn dispatch_frames<T, F>(
    decoder: &mut Decoder,
    layout: FrameLayout,
    phantom_frames: &BTreeSet<usize>,
    worker_txs: &[SyncSender<WorkerMsg>],
    to_planes: F,
) -> Result<usize, anyhow::Error>
where
    T: Pixel,
    F: Fn(&Frame<T>, FrameLayout) -> Planes,
{
    let options = DetectionOptions::default();
    let mut detector = new_detector::<T>(decoder, options)?;
    let detector_window = options.lookahead_distance + 2;
    let mut frames: VecDeque<Arc<Frame<T>>> = VecDeque::with_capacity(detector_window);
    let workers = worker_txs.len();
    let mut raw_frame_no = 0usize;
    let mut frame_no = 0usize;
    let mut scene_idx = 0usize;
    let mut previous_keyframe = 0usize;
    let mut eof = false;

    loop {
        let required_frames = options.lookahead_distance + 1 + usize::from(frame_no > 0);
        while frames.len() < required_frames && !eof {
            match decoder.read_video_frame::<T>() {
                Ok(frame) => {
                    if !phantom_frames.contains(&raw_frame_no) {
                        frames.push_back(Arc::new(frame));
                    }
                    raw_frame_no += 1;
                },
                Err(DecoderError::EndOfFile) => eof = true,
                Err(error) => return Err(error.into()),
            }
        }

        if frames.len() < 2 {
            if frame_no == 0
                && let Some(frame) = frames.front()
            {
                worker_txs[0]
                    .send(WorkerMsg::Frame {
                        global_idx: 0,
                        planes: to_planes(frame, layout),
                    })
                    .map_err(|_| anyhow::anyhow!("worker 0 disconnected"))?;
                frame_no = 1;
            }
            break;
        }

        let (cut, planes) = {
            let frame_set = frames.iter().take(detector_window).collect::<Vec<_>>();
            let cut = if frame_no == 0 {
                false
            } else {
                detector
                    .analyze_next_frame(&frame_set, frame_no, previous_keyframe)
                    .0
            };
            let frame_offset = usize::from(frame_no > 0);
            (cut, to_planes(frame_set[frame_offset], layout))
        };

        if cut {
            let previous_target = scene_idx % workers;
            worker_txs[previous_target]
                .send(WorkerMsg::EndScene)
                .map_err(|_| anyhow::anyhow!("worker {previous_target} disconnected"))?;
            scene_idx += 1;
            previous_keyframe = frame_no;
        }

        let target = scene_idx % workers;

        worker_txs[target]
            .send(WorkerMsg::Frame {
                global_idx: frame_no as u64,
                planes,
            })
            .map_err(|_| anyhow::anyhow!("worker {target} disconnected"))?;

        if frame_no > 0 {
            frames.pop_front();
        }

        frame_no += 1;
    }

    if frame_no > 0 {
        let target = scene_idx % workers;
        worker_txs[target]
            .send(WorkerMsg::EndScene)
            .map_err(|_| anyhow::anyhow!("worker {target} disconnected"))?;
    }

    Ok(frame_no)
}

enum WorkerMsg {
    Frame { global_idx: u64, planes: Planes },
    EndScene,
    Eof,
}

struct OutputMsg {
    global_idx: u64,
    planes: Planes,
}

fn run_worker(
    opts: PlaneOptions,
    layout: FrameLayout,
    rx: Receiver<WorkerMsg>,
    tx: SyncSender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    let mut active_scene = false;
    let mut wd: Option<PlanarDenoiser> = None;
    let mut pending: VecDeque<u64> = Default::default();

    loop {
        match rx.recv() {
            Ok(WorkerMsg::Frame { global_idx, planes }) => {
                if !active_scene {
                    if wd.is_none() {
                        wd = Some(PlanarDenoiser::create(&opts, layout)?);
                    }
                    active_scene = true;
                }

                let denoiser = wd.as_mut().expect("denoiser exists after new-scene init");
                push_with_drain(denoiser, &mut pending, global_idx, &planes, &tx)?;
            },
            Ok(WorkerMsg::EndScene) => {
                if active_scene {
                    let denoiser = wd.as_mut().expect("active scene has a denoiser");
                    flush_worker(denoiser, &mut pending, &tx)?;
                    active_scene = false;
                }
            },
            Ok(WorkerMsg::Eof) | Err(_) => {
                if active_scene {
                    let denoiser = wd.as_mut().expect("active scene has a denoiser");
                    flush_worker(denoiser, &mut pending, &tx)?;
                }
                break;
            },
        }
    }

    Ok(())
}

/// Push one frame, draining any pending output first if the queue is full.
fn push_with_drain(
    denoiser: &mut PlanarDenoiser,
    pending: &mut VecDeque<u64>,
    global_idx: u64,
    planes: &Planes,
    tx: &SyncSender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    pending.push_back(global_idx);

    if push_needs_retry(denoiser.push(planes))? {
        if let Some(out) = denoiser.recv()? {
            let g = pending
                .pop_front()
                .expect("pending has at least one entry on QueueFull recv");
            send_output(tx, g, out)?;
        }

        denoiser.push(planes)?;
    }

    Ok(())
}

fn send_output(tx: &SyncSender<OutputMsg>, global_idx: u64, planes: Planes) -> Result<(), anyhow::Error> {
    tx.send(OutputMsg { global_idx, planes })
        .map_err(|_| anyhow::anyhow!("coordinator disconnected"))
}

fn flush_worker(
    wd: &mut PlanarDenoiser,
    pending: &mut VecDeque<u64>,
    tx: &SyncSender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    let mut disconnected = false;

    wd.flush(|out| {
        if disconnected {
            return;
        }

        if let Some(g) = pending.pop_front() {
            let msg = OutputMsg {
                global_idx: g,
                planes: out,
            };
            let did_send = tx.send(msg).is_ok();
            if !did_send {
                disconnected = true;
            }
        } else {
            tracing::warn!("worker emitted flushed frame with no pending global index");
        }
    })?;

    if disconnected {
        anyhow::bail!("coordinator disconnected while flushing worker output");
    }

    Ok(())
}

fn run_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: Receiver<OutputMsg>,
    dispatch_count: Receiver<usize>,
    progress_total: Option<usize>,
    visible: bool,
) -> Result<(), anyhow::Error> {
    let stdout = stdout();
    let lock = stdout.lock();

    // No `XCOLORRANGE=` tag is emitted here. `av_decoders::VideoDetails`
    // doesn't surface the source's color range for any of its backends
    // (ffms2 included), so there's nothing to forward.
    let mut encoder = y4m::encode(
        layout.width as usize,
        layout.height as usize,
        y4m::Ratio::new((*framerate.numer()) as usize, (*framerate.denom()) as usize),
    )
    .with_colorspace(subsampling_to_y4m(layout.subsampling, layout.depth))
    .write_header(lock)?;

    let pb = progress_total.map_or_else(ProgressBar::hidden, |total| denoise_progress_bar(total, visible));

    pb.enable_steady_tick(Duration::from_millis(250));

    let result = emit_frames(&mut encoder, &rx, &dispatch_count, &pb);

    progress::finish(&pb);

    result
}

fn emit_frames<W>(
    encoder: &mut y4m::Encoder<W>,
    rx: &Receiver<OutputMsg>,
    dispatch_count: &Receiver<usize>,
    pb: &ProgressBar,
) -> Result<(), anyhow::Error>
where
    W: std::io::Write,
{
    let mut pending: BTreeMap<u64, Planes> = BTreeMap::new();
    let mut next_emit: u64 = 0;

    while let Ok(msg) = rx.recv() {
        pending.insert(msg.global_idx, msg.planes);

        while let Some(planes) = pending.remove(&next_emit) {
            let frame = Y4mFrame::new([&planes.y, &planes.u, &planes.v], None);
            encoder.write_frame(&frame)?;
            next_emit += 1;
        }

        pb.set_position(next_emit);
    }

    let dispatched = dispatch_count
        .recv()
        .map_err(|_| anyhow::anyhow!("dispatcher ended without a final frame count"))?
        as u64;

    if dispatched == 0 {
        anyhow::bail!("input holds no decodable frames");
    }

    if next_emit != dispatched {
        anyhow::bail!(
            "wrote {next_emit} frames but expected {dispatched}. Every worker disconnected \
             before the stream finished, so a frame index was likely lost"
        );
    }

    Ok(())
}

fn planes_from_v_frame_u8(frame: &v_frame::frame::Frame<u8>, layout: FrameLayout) -> Planes {
    Planes {
        y: collect_plane_u8(&frame.y_plane),
        u: frame
            .u_plane
            .as_ref()
            .map(collect_plane_u8)
            .unwrap_or_else(|| layout.neutral_chroma_plane()),
        v: frame
            .v_plane
            .as_ref()
            .map(collect_plane_u8)
            .unwrap_or_else(|| layout.neutral_chroma_plane()),
    }
}

fn planes_from_v_frame_u16(frame: &v_frame::frame::Frame<u16>, layout: FrameLayout) -> Planes {
    Planes {
        y: collect_plane_u16(&frame.y_plane),
        u: frame
            .u_plane
            .as_ref()
            .map(collect_plane_u16)
            .unwrap_or_else(|| layout.neutral_chroma_plane()),
        v: frame
            .v_plane
            .as_ref()
            .map(collect_plane_u16)
            .unwrap_or_else(|| layout.neutral_chroma_plane()),
    }
}

fn collect_plane_u8(plane: &v_frame::plane::Plane<u8>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = Vec::with_capacity(width * height);

    for y in 0..height {
        if let Some(row) = plane.row(y) {
            out.extend_from_slice(&row[..width]);
        }
    }

    out
}

fn collect_plane_u16(plane: &v_frame::plane::Plane<u16>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = vec![0; width * height * 2];

    for y in 0..height {
        if let Some(row) = plane.row(y) {
            let start = y * width * 2;
            let destination = &mut out[start..start + width * 2];

            for (&sample, bytes) in row[..width].iter().zip(destination.as_chunks_mut::<2>().0) {
                *bytes = sample.to_le_bytes();
            }
        }
    }

    out
}

fn subsampling_from_av_decoders(
    cs: v_frame::chroma::ChromaSubsampling,
) -> Result<Subsampling, anyhow::Error> {
    use v_frame::chroma::ChromaSubsampling;

    match cs {
        ChromaSubsampling::Yuv420 => Ok(Subsampling::Yuv420),
        ChromaSubsampling::Yuv422 => Ok(Subsampling::Yuv422),
        ChromaSubsampling::Yuv444 => Ok(Subsampling::Yuv444),
        other => {
            anyhow::bail!("unsupported chroma subsampling {other:?}, need 4:2:0, 4:2:2, or 4:4:4")
        },
    }
}
