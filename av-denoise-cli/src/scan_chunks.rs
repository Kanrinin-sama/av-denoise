use std::collections::{BTreeSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use av_scenechange::{DetectionOptions, Rational32, SceneChangeDetector};
use indicatif::ProgressBar;
use v_frame::pixel::Pixel;

use crate::frame_index::IndexEntry;
use crate::nvdec_source::{NvdecSource, NvdecWindow};
use crate::scan_source::{FfmsSource, ScanFormat, ScanSource};

/// Threads each concurrently analysed chunk gets for its detector and decoder.
pub const THREADS_PER_CHUNK: usize = 4;

/// Shortest chunk worth a detector of its own.
const MIN_CHUNK_FRAMES: usize = 1024;

/// One stretch of a keep range analysed on its own. Detection runs over
/// `start - head..end + tail`, and only cuts inside `start..end` are kept.
pub struct Chunk {
    pub range: usize,
    pub start: usize,
    pub end: usize,
    pub head: usize,
    pub tail: usize,
}

/// Everything a chunk needs to know about the source.
pub struct ScanContext<'a> {
    pub input: &'a Path,
    pub ffmpeg: Option<&'a Path>,
    pub format: ScanFormat,
    pub frame_duration: Rational32,
    pub options: DetectionOptions,
    pub index: Option<(Vec<IndexEntry>, f64)>,
    pub phantom: &'a BTreeSet<usize>,
    pub progress: &'a ProgressBar,
    pub fallbacks: AtomicUsize,
}

/// Splits every range into near-equal chunks sized so `workers` detectors share the frames.
pub fn plan_chunks(ranges: &[(usize, usize)], workers: usize, options: &DetectionOptions) -> Vec<Chunk> {
    let total: usize = ranges.iter().map(|(start, end)| end - start).sum();
    let target = total.div_ceil(workers).max(MIN_CHUNK_FRAMES);
    // The detector scores a frame from the lookahead after it and from the
    // score history before it, which holds five frames plus the lookahead.
    let head_margin = options.lookahead_distance + 5 + options.lookahead_distance + 2;
    let tail_margin = options.lookahead_distance + 1;
    let mut chunks = Vec::new();

    for (range, &(range_start, range_end)) in ranges.iter().enumerate() {
        let length = range_end - range_start;
        let parts = length.div_ceil(target).max(1);
        for part in 0..parts {
            let start = range_start + length * part / parts;
            let end = range_start + length * (part + 1) / parts;
            chunks.push(Chunk {
                range,
                start,
                end,
                head: head_margin.min(start - range_start),
                tail: tail_margin.min(range_end - end),
            });
        }
    }

    chunks
}

pub fn scan_chunks(
    context: &ScanContext,
    chunks: &[Chunk],
    workers: usize,
) -> Result<Vec<Vec<usize>>, anyhow::Error> {
    let next = AtomicUsize::new(0);
    let results = Mutex::new((0..chunks.len()).map(|_| None).collect::<Vec<_>>());

    std::thread::scope(|scope| {
        for _ in 0..workers.min(chunks.len()) {
            scope.spawn(|| -> Result<(), anyhow::Error> {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(THREADS_PER_CHUNK)
                    .build()?;
                loop {
                    let position = next.fetch_add(1, Ordering::Relaxed);
                    let Some(chunk) = chunks.get(position) else {
                        return Ok(());
                    };
                    let cuts = pool.install(|| scan_chunk(context, chunk));
                    let failed = cuts.is_err();
                    results.lock().expect("no worker panics holding the results")[position] = Some(cuts);
                    if failed {
                        next.store(chunks.len(), Ordering::Relaxed);
                    }
                }
            });
        }
    });

    let mut results = results
        .into_inner()
        .expect("no worker panics holding the results");
    if let Some(failed) = results.iter().position(|cuts| matches!(cuts, Some(Err(_)))) {
        return Err(results
            .swap_remove(failed)
            .expect("the failed chunk has a result")
            .expect_err("the chunk failed"));
    }
    results
        .into_iter()
        .map(|cuts| cuts.unwrap_or_else(|| Err(anyhow::anyhow!("a scene detection worker stopped early"))))
        .collect()
}

/// Returns the raw frame numbers inside the chunk that start a scene.
fn scan_chunk(context: &ScanContext, chunk: &Chunk) -> Result<Vec<usize>, anyhow::Error> {
    let feed_start = chunk.start - chunk.head;
    let feed_end = chunk.end + chunk.tail;

    if let Some(window) = nvdec_window(context, feed_start, feed_end) {
        let source = context
            .ffmpeg
            .ok_or_else(|| anyhow::anyhow!("no ffmpeg"))
            .and_then(|ffmpeg| NvdecSource::open(ffmpeg, context.input, &context.format, window));
        match source
            .and_then(|source| detect_source(context, ScanSource::Nvdec(source), feed_end - feed_start))
        {
            Ok(cuts) => return Ok(keep_chunk_cuts(chunk, cuts)),
            Err(error) => {
                context.fallbacks.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    start = chunk.start,
                    end = chunk.end,
                    "NVDEC scan failed, redoing it with ffms2: {error:#}"
                );
            },
        }
    }

    let source = FfmsSource::open(context.input, &context.format, THREADS_PER_CHUNK, feed_start)?;
    let cuts = detect_source(context, ScanSource::Ffms(source), feed_end - feed_start)?;
    Ok(keep_chunk_cuts(chunk, cuts))
}

fn keep_chunk_cuts(chunk: &Chunk, cuts: Vec<usize>) -> Vec<usize> {
    let feed_start = chunk.start - chunk.head;

    cuts.into_iter()
        .map(|cut| feed_start + cut)
        .filter(|cut| (chunk.start..chunk.end).contains(cut))
        .collect()
}

fn nvdec_window(context: &ScanContext, feed_start: usize, feed_end: usize) -> Option<NvdecWindow> {
    context.ffmpeg?;
    let (entries, seconds_per_pts) = context.index.as_ref()?;
    let expected_pts = (feed_start..feed_end)
        .filter(|raw| !context.phantom.contains(raw))
        .map(|raw| entries.get(raw).map(|entry| entry.pts))
        .collect::<Option<Vec<_>>>()?;

    Some(NvdecWindow {
        expected_pts,
        seconds_per_pts: *seconds_per_pts,
        phantom: context
            .phantom
            .range(feed_start..feed_end)
            .map(|raw| raw - feed_start)
            .collect(),
    })
}

fn detect_source(
    context: &ScanContext,
    source: ScanSource,
    count: usize,
) -> Result<Vec<usize>, anyhow::Error> {
    if context.format.bit_depth > 8 {
        detect_frames::<u16>(context, source, count)
    } else {
        detect_frames::<u8>(context, source, count)
    }
}

/// Runs the detector over `count` frames the way `av_scenechange::detect_scene_changes`
/// does, returning every cut in chunk-local frame numbers.
fn detect_frames<T: Pixel>(
    context: &ScanContext,
    mut source: ScanSource,
    count: usize,
) -> Result<Vec<usize>, anyhow::Error> {
    let format = &context.format;
    let lookahead = context.options.lookahead_distance;
    let mut detector = SceneChangeDetector::<T>::new(
        (format.width, format.height),
        format.bit_depth,
        context.frame_duration,
        format.chroma,
        lookahead,
        context.options.analysis_speed,
        0,
        u32::MAX as usize,
    );
    let mut queue = VecDeque::new();
    let mut read = 0usize;
    let mut cuts = vec![0usize];
    let mut analyzed = 0usize;

    for frameno in 0..count {
        while read < (frameno + lookahead + 1).min(count) {
            queue.push_back(Arc::new(source.read::<T>(format)?));
            read += 1;
        }
        let frame_set = queue.iter().take(lookahead + 2).collect::<Vec<_>>();
        if frame_set.len() < 2 {
            break;
        }
        if frameno > 0 {
            let previous = *cuts.last().expect("frame 0 is always a cut");
            if detector.analyze_next_frame(&frame_set, frameno, previous).0 {
                cuts.push(frameno);
            }
            queue.pop_front();
        }
        analyzed += 1;
        context.progress.inc(1);
    }
    context.progress.inc((count - analyzed) as u64);

    source.finish()?;
    Ok(cuts)
}
