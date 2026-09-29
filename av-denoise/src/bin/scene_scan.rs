use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use av_decoders::Decoder;
use av_denoise::{Depth, FrameLayout};
use av_denoise_service::{
    FrameRange,
    SceneLayout,
    emitted_boundary_to_raw,
    subsampling_from_av_decoders,
    validate_keep_frames,
};
use av_scenechange::DetectionOptions;

use crate::frame_index;
use crate::progress::{self, scene_progress_bar};
use crate::scan_chunks::{Chunk, ScanContext, THREADS_PER_CHUNK, plan_chunks, scan_chunks};
use crate::scan_source::ScanFormat;

pub fn detect(
    input: &Path,
    keep_frames: &[FrameRange],
    ffmpeg: Option<&Path>,
    visible: bool,
) -> Result<SceneLayout, anyhow::Error> {
    let index_path = std::path::PathBuf::from(format!("{}.ffindex", input.to_string_lossy()));
    let index_existed = index_path.try_exists()?;
    let mut metadata = Decoder::from_file(input)?;
    let details = *metadata.get_video_details();
    let raw_frames = details
        .total_frames
        .ok_or_else(|| anyhow::anyhow!("FFMS2 did not report a source frame count"))?;
    let entries = frame_index::read_index(&mut metadata);
    let phantom = entries
        .as_deref()
        .map(frame_index::phantom_indices)
        .unwrap_or_default();
    let index = entries.zip(frame_index::read_time_base(&mut metadata));
    drop(metadata);

    let layout = FrameLayout {
        width: details.width as u32,
        height: details.height as u32,
        subsampling: subsampling_from_av_decoders(details.chroma_sampling)?,
        depth: Depth::from_bits(details.bit_depth)?,
    };
    let source_ranges = if keep_frames.is_empty() {
        vec![(0, raw_frames)]
    } else {
        validate_keep_frames(keep_frames, raw_frames - phantom.len())?;
        keep_frames
            .iter()
            .map(|range| {
                (
                    emitted_boundary_to_raw(range.start, raw_frames, &phantom),
                    emitted_boundary_to_raw(range.end, raw_frames, &phantom),
                )
            })
            .collect()
    };
    let format = ScanFormat::new(&details);

    tracing::info!(
        width = layout.width,
        height = layout.height,
        analysis_width = format.width,
        analysis_height = format.height,
        raw_frames,
        phantom = phantom.len(),
        "running scene detection",
    );

    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get() / THREADS_PER_CHUNK)
        .max(1);
    let options = DetectionOptions::default();
    let chunks = plan_chunks(&source_ranges, workers, &options);
    let progress = scene_progress_bar(
        Some(chunks.iter().map(|c| c.head + c.end - c.start + c.tail).sum()),
        visible,
    );
    let context = ScanContext {
        input,
        ffmpeg: ffmpeg.filter(|_| {
            format.chroma == v_frame::chroma::ChromaSubsampling::Yuv420 && format.bit_depth <= 10
        }),
        format,
        frame_duration: details.frame_rate.recip(),
        options,
        index,
        phantom: &phantom,
        progress: &progress,
        fallbacks: AtomicUsize::new(0),
    };
    let cuts = scan_chunks(&context, &chunks, workers);
    progress::finish(&progress);
    if !index_existed && keep_frames.is_empty() {
        let _ = std::fs::remove_file(index_path);
    }
    let cuts = cuts?;

    tracing::info!(
        chunks = chunks.len(),
        workers,
        nvdec = context.ffmpeg.is_some(),
        ffms2_fallbacks = context.fallbacks.load(Ordering::Relaxed),
        "scene detection finished",
    );

    let (scene_starts, total_frames) = merge_scene_starts(&chunks, &cuts, &source_ranges, &phantom);

    if total_frames == 0 {
        anyhow::bail!("{} holds no decodable frames", input.display());
    }

    Ok(SceneLayout {
        layout,
        framerate: details.frame_rate,
        total_frames,
        raw_frames,
        phantom,
        scene_starts,
        source_ranges,
        keep_frames: keep_frames.to_vec(),
    })
}

/// Joins every range's cuts into one list of scene starts in emitted frame numbers,
/// returning it with the emitted frame count.
fn merge_scene_starts(
    chunks: &[Chunk],
    cuts: &[Vec<usize>],
    source_ranges: &[(usize, usize)],
    phantom: &BTreeSet<usize>,
) -> (Vec<usize>, usize) {
    let mut scene_starts = vec![0usize];
    let mut compact_offset = 0usize;
    for (range, &(raw_start, raw_end)) in source_ranges.iter().enumerate() {
        let mut local = chunks
            .iter()
            .zip(cuts)
            .filter(|(chunk, _)| chunk.range == range)
            .flat_map(|(_, cuts)| cuts.iter().map(|cut| cut - raw_start))
            .collect::<Vec<_>>();
        local.push(0);
        local.sort_unstable();
        local.dedup();
        let local_phantom = phantom
            .range(raw_start..raw_end)
            .map(|index| index - raw_start)
            .collect::<BTreeSet<_>>();
        let remapped = frame_index::remap_scene_starts(&local, &local_phantom);
        scene_starts.extend(
            remapped
                .into_iter()
                .filter(|start| *start > 0)
                .map(|start| compact_offset + start),
        );
        compact_offset += raw_end - raw_start - local_phantom.len();
        scene_starts.push(compact_offset);
    }
    scene_starts.sort_unstable();
    scene_starts.dedup();

    (scene_starts, compact_offset)
}
