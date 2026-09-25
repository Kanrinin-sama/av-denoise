use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, IsTerminal, Write, stdout};
use std::path::Path;
use std::thread;
use std::time::Duration;

use av_decoders::{Decoder, Rational32};
use av_denoise::{
    DenoisingMode,
    Depth,
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    Subsampling,
    WarmUp,
    push_needs_retry,
};
use av_scenechange::{DetectionOptions, detect_scene_changes};
use indicatif::ProgressBar;
use y4m::Frame as Y4mFrame;

use crate::cli::{FrameRange, RunOptions};
use crate::frame_index;
use crate::progress::{self, denoise_bar_visible, denoise_progress_bar, scene_progress_bar};
use crate::warm_start::{create_denoiser, finish_warm_up};
use crate::y4m_format::subsampling_to_y4m;

/// Frames in flight this run allows.
fn frame_permits(budget_bytes: u64, frame_bytes: usize, workers: usize, radius: u32) -> usize {
    // A worker emits nothing until `push` first returns QueueFull, which
    // takes `radius + MAX_PENDING + 1` pushes. Fewer permits than that
    // and its scene can never return one, so the dispatcher waits on a
    // permit the worker cannot release.
    let floor = workers * (radius as usize + av_denoise::MAX_PENDING + 2);

    frames_afforded(budget_bytes, frame_bytes).max(floor)
}

/// Frames the budget pays for at this frame size.
fn frames_afforded(budget_bytes: u64, frame_bytes: usize) -> usize {
    let per_frame = (frame_bytes as u64).max(1);

    usize::try_from(budget_bytes / per_frame).unwrap_or(usize::MAX)
}

/// Renders a byte count in the decimal units `--frame-budget` accepts.
///
/// The space `ByteSize` puts before the unit goes, so the result is one
/// shell argument a caller can paste straight back into the flag.
fn size_string(bytes: u64) -> String {
    bytesize::ByteSize::b(bytes)
        .display()
        .si()
        .to_string()
        .replace(' ', "")
}

/// Rounds a byte count up to the precision [`size_string`] prints at.
///
/// The rendering keeps one decimal place, so a raw minimum can round
/// down to a size that still fails the budget check.
fn suggested_budget(bytes: u64) -> String {
    let unit = [bytesize::GB, bytesize::MB, bytesize::KB]
        .into_iter()
        .find(|&unit| bytes >= unit)
        .unwrap_or(1);
    let step = (unit / 10).max(1);

    size_string(bytes.div_ceil(step) * step)
}

/// Frames in flight this run allows, refusing a budget below the floor.
///
/// A budget the floor has to raise serialises the pipeline, so it fails
/// here rather than running on with too few frames in flight.
fn checked_frame_permits(
    budget_bytes: u64,
    frame_bytes: usize,
    workers: usize,
    radius: u32,
) -> Result<usize, anyhow::Error> {
    let afforded = frames_afforded(budget_bytes, frame_bytes);
    let permits = frame_permits(budget_bytes, frame_bytes, workers, radius);

    if permits > afforded {
        let suggestion = suggested_budget(permits as u64 * frame_bytes as u64);

        anyhow::bail!(
            "--frame-budget {budget} affords {afforded} frames at {frame_bytes} bytes per frame, \
             but {workers} workers at temporal radius {radius} need at least {permits}. Pass at \
             least --frame-budget {suggestion}.",
            budget = size_string(budget_bytes),
        );
    }

    Ok(permits)
}

/// Builds a counting semaphore holding `count` permits.
///
/// Returns the giving end and the taking end. The coordinator holds the
/// giving end, so if it dies the dispatcher's next take fails instead of
/// blocking forever.
fn frame_permit_channel(count: usize) -> (crossbeam_channel::Sender<()>, crossbeam_channel::Receiver<()>) {
    let (give, take) = crossbeam_channel::bounded::<()>(count);

    for _ in 0..count {
        give.send(()).expect("the channel holds exactly `count` permits");
    }

    (give, take)
}

/// Scene boundaries plus the video metadata needed to build the output
/// y4m header.
#[derive(Clone)]
struct SceneLayout {
    layout: FrameLayout,
    framerate: Rational32,
    /// Frames this run emits, being `raw_frames` less the phantom entries.
    total_frames: usize,
    /// Frames the decoder hands over, phantom entries included. Every
    /// one has to be read to keep the sequential decoder in step, even
    /// though only `total_frames` of them are emitted.
    raw_frames: usize,
    /// Decoder frame numbers that carry no picture of their own. See
    /// [`crate::frame_index`].
    phantom: BTreeSet<usize>,
    /// `scene_starts[i]` is the inclusive start frame of scene `i`, in  emitted frame numbers.
    /// The final entry is `total_frames` so scene `i` covers `scene_starts[i]..scene_starts[i + 1]`.
    scene_starts: Vec<usize>,
    source_ranges: Vec<(usize, usize)>,
    keep_frames: Vec<FrameRange>,
}

const SCENE_LAYOUT_SCHEMA: u32 = 1;
const SCENE_DETECTOR_ID: &str = "av-scenechange-0.23-default";

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredSceneLayout {
    schema: u32,
    detector: String,
    width: u32,
    height: u32,
    subsampling: String,
    depth: u8,
    frame_rate_numerator: i32,
    frame_rate_denominator: i32,
    total_frames: usize,
    raw_frames: usize,
    phantom: BTreeSet<usize>,
    scene_starts: Vec<usize>,
    #[serde(default)]
    keep_frames: Vec<StoredFrameRange>,
    #[serde(default)]
    source_ranges: Vec<(usize, usize)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredFrameRange {
    start: usize,
    end: usize,
}

impl SceneLayout {
    fn scene_count(&self) -> usize {
        self.scene_starts.len() - 1
    }
}

pub fn run_file(
    opts: &RunOptions,
    input: &Path,
    workers: usize,
    frame_budget_bytes: u64,
    stored_layout: Option<&Path>,
    keep_frames: &[FrameRange],
    scene_span: Option<FrameRange>,
) -> Result<(), anyhow::Error> {
    run_file_to(
        opts,
        input,
        workers,
        frame_budget_bytes,
        stored_layout,
        keep_frames,
        scene_span,
        Box::new(stdout()),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_file_to(
    opts: &RunOptions,
    input: &Path,
    workers: usize,
    frame_budget_bytes: u64,
    stored_layout: Option<&Path>,
    keep_frames: &[FrameRange],
    scene_span: Option<FrameRange>,
    output: Box<dyn Write + Send>,
) -> Result<(), anyhow::Error> {
    if workers == 0 {
        anyhow::bail!("--workers must be at least 1");
    }

    // Scene detection finishes before a single frame is written, so its
    // bar has the terminal to itself and needs no opt-in. The denoising
    // bar shares the terminal with whatever consumes our output, so it
    // waits for --progress.
    let is_terminal = std::io::stderr().is_terminal();
    let scenes = match stored_layout {
        Some(path) => read_scene_layout(input, path, keep_frames)?,
        None if keep_frames.is_empty() => detect_scenes(input, is_terminal)?,
        None => anyhow::bail!("--keep-frames requires --scene-layout; run `scenes` first"),
    };
    let scenes = match scene_span {
        Some(span) if stored_layout.is_some() => select_scene_span(scenes, span)?,
        Some(_) => anyhow::bail!("--scene-span requires --scene-layout"),
        None => scenes,
    };

    tracing::info!(
        scene_count = scenes.scene_count(),
        total_frames = scenes.total_frames,
        workers,
        "scene detection complete",
    );

    encode_scenes(
        &opts.planes,
        input,
        &scenes,
        workers,
        denoise_bar_visible(opts.progress, is_terminal),
        frame_budget_bytes,
        output,
    )
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WindowCommand {
    Start {
        id: u64,
        slot: usize,
        start: usize,
        end: usize,
        output_path: std::path::PathBuf,
    },
    Finish,
}

#[derive(serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WindowEvent<'a> {
    Ready { version: u32, slots: usize },
    Done { id: u64, slot: usize, frames: usize },
    Error { id: u64, slot: usize, message: &'a str },
}

struct WindowRequest {
    id: u64,
    span: FrameRange,
    output_path: std::path::PathBuf,
}

pub fn run_window_service(
    opts: &RunOptions,
    input: &Path,
    workers: usize,
    frame_budget_bytes: u64,
    stored_layout: &Path,
    keep_frames: &[FrameRange],
    slots: usize,
) -> Result<(), anyhow::Error> {
    if slots == 0 {
        anyhow::bail!("--window-service-slots must be at least 1");
    }
    if workers == 0 {
        anyhow::bail!("--workers must be at least 1");
    }
    let full_layout = read_scene_layout(input, stored_layout, keep_frames)?;
    let radius = match opts.planes.mode {
        DenoisingMode::Temporal { radius } => radius,
        DenoisingMode::Spacial => 0,
    };
    let frame_bytes = full_layout.layout.luma_bytes() + 2 * full_layout.layout.chroma_bytes();
    let permits = checked_frame_permits(frame_budget_bytes, frame_bytes, workers * slots, radius)?;
    let shared_permits = frame_permit_channel(permits);
    let events = std::sync::Arc::new(std::sync::Mutex::new(std::io::BufWriter::new(stdout())));
    let busy: Vec<_> = (0..slots)
        .map(|_| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)))
        .collect();
    thread::scope(|scope| {
        let mut senders = Vec::with_capacity(slots);
        let mut handles = Vec::with_capacity(slots);
        for (slot, busy_slot) in busy.iter().enumerate().take(slots) {
            let (tx, rx) = crossbeam_channel::bounded::<WindowRequest>(1);
            senders.push(tx);
            let events = std::sync::Arc::clone(&events);
            let slot_busy = std::sync::Arc::clone(busy_slot);
            let full_layout = full_layout.clone();
            let shared_permits = &shared_permits;
            handles.push(scope.spawn(move || {
                let mut denoisers = Vec::new();
                while let Ok(request) = rx.recv() {
                    let result = (|| {
                        let scenes = select_scene_span(full_layout.clone(), request.span)?;
                        let frames = scenes.total_frames;
                        let output = std::fs::OpenOptions::new()
                            .write(true)
                            .open(&request.output_path)?;
                        encode_scenes_reusing(
                            &opts.planes,
                            input,
                            &scenes,
                            workers,
                            false,
                            frame_budget_bytes,
                            Box::new(output),
                            &mut denoisers,
                            Some(shared_permits),
                        )?;
                        Ok::<_, anyhow::Error>(frames)
                    })();
                    match result {
                        Ok(frames) => {
                            slot_busy.store(false, std::sync::atomic::Ordering::Release);
                            write_window_event(
                                &events,
                                &WindowEvent::Done {
                                    id: request.id,
                                    slot,
                                    frames,
                                },
                            )?;
                        },
                        Err(error) => {
                            let message = format!("{error:#}");
                            write_window_event(
                                &events,
                                &WindowEvent::Error {
                                    id: request.id,
                                    slot,
                                    message: &message,
                                },
                            )?;
                            std::process::exit(1);
                        },
                    }
                }
                Ok::<_, anyhow::Error>(())
            }));
        }

        write_window_event(&events, &WindowEvent::Ready { version: 1, slots })?;
        let input = std::io::stdin();
        let mut finished = false;
        for line in input.lock().lines() {
            let line =
                line.unwrap_or_else(|error| control_fatal(&format!("reading control input failed: {error}")));
            let command: WindowCommand = serde_json::from_str(&line)
                .unwrap_or_else(|error| control_fatal(&format!("invalid control message: {error}")));
            match command {
                WindowCommand::Start {
                    id,
                    slot,
                    start,
                    end,
                    output_path,
                } => {
                    if start >= end {
                        control_fatal("window start must be less than end");
                    }
                    let Some(sender) = senders.get(slot) else {
                        control_fatal(&format!("window slot {slot} is out of range"));
                    };
                    if busy[slot]
                        .compare_exchange(
                            false,
                            true,
                            std::sync::atomic::Ordering::AcqRel,
                            std::sync::atomic::Ordering::Acquire,
                        )
                        .is_err()
                    {
                        control_fatal(&format!("window slot {slot} is busy"));
                    }
                    if sender
                        .try_send(WindowRequest {
                            id,
                            span: FrameRange { start, end },
                            output_path,
                        })
                        .is_err()
                    {
                        busy[slot].store(false, std::sync::atomic::Ordering::Release);
                        control_fatal(&format!("window slot {slot} is unavailable"));
                    }
                },
                WindowCommand::Finish => {
                    if busy
                        .iter()
                        .any(|busy| busy.load(std::sync::atomic::Ordering::Acquire))
                    {
                        control_fatal("finish received while a window is active");
                    }
                    finished = true;
                    break;
                },
            }
        }
        if !finished {
            std::process::exit(1);
        }
        drop(senders);
        for handle in handles {
            handle.join().map_err(|panic| {
                anyhow::anyhow!("window worker panicked: {}", panic_message(panic.as_ref()))
            })??;
        }
        Ok(())
    })
}

fn control_fatal(message: &str) -> ! {
    tracing::error!(message, "window control protocol failed");
    std::process::exit(2)
}

fn write_window_event(
    output: &std::sync::Mutex<std::io::BufWriter<std::io::Stdout>>,
    event: &WindowEvent<'_>,
) -> Result<(), anyhow::Error> {
    let mut output = output.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    serde_json::to_writer(&mut *output, event)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn select_scene_span(mut scenes: SceneLayout, span: FrameRange) -> Result<SceneLayout, anyhow::Error> {
    if scenes.scene_starts.binary_search(&span.start).is_err()
        || scenes.scene_starts.binary_search(&span.end).is_err()
    {
        anyhow::bail!("--scene-span must match retained scene boundaries");
    }

    let mut source_ranges = Vec::new();
    let full_range = [FrameRange {
        start: 0,
        end: scenes.total_frames,
    }];
    let selected_ranges = if scenes.keep_frames.is_empty() {
        full_range.as_slice()
    } else {
        scenes.keep_frames.as_slice()
    };
    let mut compact_start = 0usize;
    for range in selected_ranges {
        let compact_end = compact_start + range.end - range.start;
        let selected_start = span.start.max(compact_start);
        let selected_end = span.end.min(compact_end);
        if selected_start < selected_end {
            let source_start = range.start + selected_start - compact_start;
            let source_end = range.start + selected_end - compact_start;
            source_ranges.push((
                emitted_boundary_to_raw(source_start, scenes.raw_frames, &scenes.phantom),
                emitted_boundary_to_raw(source_end, scenes.raw_frames, &scenes.phantom),
            ));
        }
        compact_start = compact_end;
    }

    scenes.scene_starts = scenes
        .scene_starts
        .into_iter()
        .filter(|boundary| *boundary >= span.start && *boundary <= span.end)
        .map(|boundary| boundary - span.start)
        .collect();
    scenes.total_frames = span.end - span.start;
    scenes.source_ranges = source_ranges;
    Ok(scenes)
}

pub fn write_scene_layout(
    input: &Path,
    output: &Path,
    keep_frames: &[FrameRange],
) -> Result<(), anyhow::Error> {
    let scenes = if keep_frames.is_empty() {
        detect_scenes(input, std::io::stderr().is_terminal())?
    } else {
        detect_kept_scenes(input, keep_frames, std::io::stderr().is_terminal())?
    };
    let stored = StoredSceneLayout::from_layout(&scenes);
    let bytes = serde_json::to_vec(&stored)?;
    let name = output
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("scene layout output has no file name"))?;
    let temporary = output.with_file_name(format!(".{}.{}.tmp", name.to_string_lossy(), std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    use std::io::Write;
    if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    drop(file);
    if let Err(error) = std::fs::rename(&temporary, output) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

impl StoredSceneLayout {
    fn from_layout(layout: &SceneLayout) -> Self {
        Self {
            schema: if layout.keep_frames.is_empty() {
                SCENE_LAYOUT_SCHEMA
            } else {
                2
            },
            detector: SCENE_DETECTOR_ID.into(),
            width: layout.layout.width,
            height: layout.layout.height,
            subsampling: format!("{:?}", layout.layout.subsampling),
            depth: layout.layout.depth.bits() as u8,
            frame_rate_numerator: *layout.framerate.numer(),
            frame_rate_denominator: *layout.framerate.denom(),
            total_frames: layout.total_frames,
            raw_frames: layout.raw_frames,
            phantom: layout.phantom.clone(),
            scene_starts: layout.scene_starts.clone(),
            keep_frames: layout
                .keep_frames
                .iter()
                .map(|range| StoredFrameRange {
                    start: range.start,
                    end: range.end,
                })
                .collect(),
            source_ranges: layout.source_ranges.clone(),
        }
    }
}

fn read_scene_layout(
    input: &Path,
    path: &Path,
    keep_frames: &[FrameRange],
) -> Result<SceneLayout, anyhow::Error> {
    let stored: StoredSceneLayout = serde_json::from_slice(&std::fs::read(path)?)?;
    let expected_schema = if keep_frames.is_empty() {
        SCENE_LAYOUT_SCHEMA
    } else {
        2
    };
    if stored.schema != expected_schema || stored.detector != SCENE_DETECTOR_ID {
        anyhow::bail!("scene layout schema or detector does not match this av-denoise build");
    }
    let decoder = Decoder::from_file(input)?;
    let details = *decoder.get_video_details();
    let layout = FrameLayout {
        width: details.width as u32,
        height: details.height as u32,
        subsampling: subsampling_from_av_decoders(details.chroma_sampling)?,
        depth: Depth::from_bits(details.bit_depth)?,
    };
    let actual_subsampling = format!("{:?}", layout.subsampling);
    if stored.width != layout.width
        || stored.height != layout.height
        || stored.subsampling != actual_subsampling
        || stored.depth != layout.depth.bits() as u8
        || stored.frame_rate_numerator != *details.frame_rate.numer()
        || stored.frame_rate_denominator != *details.frame_rate.denom()
        || details
            .total_frames
            .is_some_and(|frames| stored.raw_frames != frames)
    {
        anyhow::bail!("scene layout does not match the opened source video");
    }
    drop(decoder);
    let stored_keep = stored
        .keep_frames
        .iter()
        .map(|range| FrameRange {
            start: range.start,
            end: range.end,
        })
        .collect::<Vec<_>>();
    let source_total = stored
        .raw_frames
        .checked_sub(stored.phantom.len())
        .ok_or_else(|| anyhow::anyhow!("scene layout contains invalid phantom frames"))?;
    if !stored_keep.is_empty() {
        validate_keep_frames(&stored_keep, source_total)?;
    }
    if !keep_frames.is_empty() {
        validate_keep_frames(keep_frames, source_total)?;
    }
    let stored_ranges = stored_keep
        .iter()
        .map(|range| {
            (
                emitted_boundary_to_raw(range.start, stored.raw_frames, &stored.phantom),
                emitted_boundary_to_raw(range.end, stored.raw_frames, &stored.phantom),
            )
        })
        .collect::<Vec<_>>();
    let stored_keep = merge_adjacent_ranges(&stored_keep);
    let requested_keep = merge_adjacent_ranges(keep_frames);
    let expected_ranges = stored_keep
        .iter()
        .map(|range| {
            (
                emitted_boundary_to_raw(range.start, stored.raw_frames, &stored.phantom),
                emitted_boundary_to_raw(range.end, stored.raw_frames, &stored.phantom),
            )
        })
        .collect::<Vec<_>>();
    let expected_total = if stored_keep.is_empty() {
        source_total
    } else {
        stored_keep.iter().map(|range| range.end - range.start).sum()
    };
    if stored_keep != requested_keep {
        anyhow::bail!("scene layout keep frames {stored_keep:?} do not match requested {requested_keep:?}");
    }
    if !stored_keep.is_empty() && stored.source_ranges != stored_ranges {
        anyhow::bail!(
            "scene layout source ranges {:?} do not match its keep frames {stored_ranges:?}",
            stored.source_ranges
        );
    }
    if stored.total_frames != expected_total
        || stored.total_frames == 0
        || stored.scene_starts.first() != Some(&0)
        || stored.scene_starts.last() != Some(&stored.total_frames)
        || stored.scene_starts.windows(2).any(|pair| pair[0] >= pair[1])
        || stored.phantom.iter().any(|index| *index >= stored.raw_frames)
    {
        anyhow::bail!("scene layout contains invalid frame boundaries");
    }
    Ok(SceneLayout {
        layout,
        framerate: details.frame_rate,
        total_frames: stored.total_frames,
        raw_frames: stored.raw_frames,
        phantom: stored.phantom,
        scene_starts: stored.scene_starts,
        source_ranges: if stored_keep.is_empty() {
            vec![(0, stored.raw_frames)]
        } else {
            expected_ranges
        },
        keep_frames: stored_keep,
    })
}

fn merge_adjacent_ranges(ranges: &[FrameRange]) -> Vec<FrameRange> {
    let mut merged: Vec<FrameRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut()
            && previous.end == range.start
        {
            previous.end = range.end;
        } else {
            merged.push(*range);
        }
    }
    merged
}

/// Opens the input, reads its layout, and runs `av_scenechange` to
/// produce a list of scene boundaries.
///
/// Returns once the detector pass has finished and the temporary decoder
/// it used has been dropped.
fn detect_scenes(input: &Path, visible: bool) -> Result<SceneLayout, anyhow::Error> {
    let index_path = std::path::PathBuf::from(format!("{}.ffindex", input.to_string_lossy()));
    let index_existed = index_path.try_exists()?;
    let decoder = Decoder::from_file(input);
    if !index_existed {
        let _ = std::fs::remove_file(index_path);
    }
    let mut decoder = decoder?;
    let details = *decoder.get_video_details();

    let depth = Depth::from_bits(details.bit_depth)?;

    let layout = FrameLayout {
        width: details.width as u32,
        height: details.height as u32,
        subsampling: subsampling_from_av_decoders(details.chroma_sampling)?,
        depth,
    };

    tracing::info!(
        width = layout.width,
        height = layout.height,
        subsampling = ?layout.subsampling,
        depth = ?layout.depth,
        total_frames = details.total_frames,
        "running scene detection",
    );

    // Read the index before decoding anything. This only inspects the
    // metadata ffms2 already built, so it costs nothing.
    let phantom = frame_index::read_index(&mut decoder)
        .map(|index| frame_index::phantom_indices(&index))
        .unwrap_or_default();

    if !phantom.is_empty() {
        tracing::info!(
            dropped = phantom.len(),
            "the decoder reports frames that carry no picture of their own, dropping them",
        );
    }

    let pb = scene_progress_bar(details.total_frames, visible);
    let on_progress = |frames_analyzed: usize, _keyframe_count: usize| {
        pb.set_position(frames_analyzed as u64);
    };

    let detect_opts = DetectionOptions::default();
    let detection = match depth {
        Depth::Eight => detect_scene_changes::<u8>(&mut decoder, detect_opts, None, Some(&on_progress))?,
        Depth::Ten | Depth::Twelve => {
            detect_scene_changes::<u16>(&mut decoder, detect_opts, None, Some(&on_progress))?
        },
    };

    progress::finish(&pb);

    drop(decoder);

    let mut scene_starts = detection.scene_changes;

    if scene_starts.is_empty() || scene_starts[0] != 0 {
        scene_starts.insert(0, 0);
    }

    // Detection ran over every frame the decoder offers, so both the count and the boundaries
    // are in decoder frame numbers. Both move into emitted frame numbers together.
    let raw_frames = detection.frame_count;

    // The index lists every entry the container holds, while detection counts
    // what the decoder handed over. A decode that stops early leaves entries
    // above the last frame read, and those match no frame this run sees.
    let phantom: BTreeSet<usize> = phantom.into_iter().take_while(|&raw| raw < raw_frames).collect();

    scene_starts.push(raw_frames);

    let scene_starts = frame_index::remap_scene_starts(&scene_starts, &phantom);
    let total_frames = raw_frames - phantom.len();

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
        source_ranges: vec![(0, raw_frames)],
        keep_frames: Vec::new(),
    })
}

fn validate_keep_frames(ranges: &[FrameRange], total: usize) -> Result<(), anyhow::Error> {
    if ranges.is_empty()
        || ranges
            .iter()
            .any(|range| range.start >= range.end || range.end > total)
        || ranges.windows(2).any(|pair| pair[0].end > pair[1].start)
    {
        anyhow::bail!("--keep-frames ranges must be sorted, non-overlapping, and within the source");
    }
    Ok(())
}

fn emitted_boundary_to_raw(emitted: usize, raw_total: usize, phantom: &BTreeSet<usize>) -> usize {
    let mut seen = 0usize;
    for raw in 0..raw_total {
        if seen == emitted {
            return raw;
        }
        if !phantom.contains(&raw) {
            seen += 1;
        }
    }
    raw_total
}

fn detect_kept_scenes(
    input: &Path,
    keep_frames: &[FrameRange],
    visible: bool,
) -> Result<SceneLayout, anyhow::Error> {
    let mut metadata = Decoder::from_file(input)?;
    let details = *metadata.get_video_details();
    let raw_frames = details
        .total_frames
        .ok_or_else(|| anyhow::anyhow!("FFMS2 did not report a source frame count"))?;
    let phantom = frame_index::read_index(&mut metadata)
        .map(|index| frame_index::phantom_indices(&index))
        .unwrap_or_default();
    let source_total = raw_frames - phantom.len();
    validate_keep_frames(keep_frames, source_total)?;
    let depth = Depth::from_bits(details.bit_depth)?;
    let layout = FrameLayout {
        width: details.width as u32,
        height: details.height as u32,
        subsampling: subsampling_from_av_decoders(details.chroma_sampling)?,
        depth,
    };
    drop(metadata);
    let source_ranges = keep_frames
        .iter()
        .map(|range| {
            (
                emitted_boundary_to_raw(range.start, raw_frames, &phantom),
                emitted_boundary_to_raw(range.end, raw_frames, &phantom),
            )
        })
        .collect::<Vec<_>>();
    let total_frames = keep_frames.iter().map(|range| range.end - range.start).sum();
    let pb = scene_progress_bar(Some(total_frames), visible);
    let mut analyzed = 0usize;
    let mut scene_starts = vec![0usize];
    let mut compact_offset = 0usize;
    for &(raw_start, raw_end) in &source_ranges {
        let mut decoder = Decoder::from_file(input)?;
        decoder.seek_video_frame(raw_start)?;
        let raw_count = raw_end - raw_start;
        let on_progress = |frames: usize, _keyframes: usize| {
            pb.set_position((analyzed + frames) as u64);
        };
        let detection = match depth {
            Depth::Eight => detect_scene_changes::<u8>(
                &mut decoder,
                DetectionOptions::default(),
                Some(raw_count),
                Some(&on_progress),
            )?,
            Depth::Ten | Depth::Twelve => detect_scene_changes::<u16>(
                &mut decoder,
                DetectionOptions::default(),
                Some(raw_count),
                Some(&on_progress),
            )?,
        };
        let local_phantom = phantom
            .range(raw_start..raw_end)
            .map(|index| index - raw_start)
            .collect::<BTreeSet<_>>();
        let remapped = frame_index::remap_scene_starts(&detection.scene_changes, &local_phantom);
        scene_starts.extend(
            remapped
                .into_iter()
                .filter(|start| *start > 0)
                .map(|start| compact_offset + start),
        );
        let emitted_count = raw_count - local_phantom.len();
        compact_offset += emitted_count;
        scene_starts.push(compact_offset);
        analyzed += raw_count;
    }
    progress::finish(&pb);
    scene_starts.sort_unstable();
    scene_starts.dedup();
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

/// Starts the worker pool and the coordinator, reopens the decoder for
/// frame reading, then drives the dispatch loop until EOF.
///
/// Blocks until every worker and the coordinator have finished.
fn encode_scenes(
    opts: &PlaneOptions,
    input: &Path,
    scenes: &SceneLayout,
    workers: usize,
    visible: bool,
    frame_budget_bytes: u64,
    output: Box<dyn Write + Send>,
) -> Result<(), anyhow::Error> {
    let mut denoisers = Vec::new();
    encode_scenes_reusing(
        opts,
        input,
        scenes,
        workers,
        visible,
        frame_budget_bytes,
        output,
        &mut denoisers,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_scenes_reusing(
    opts: &PlaneOptions,
    input: &Path,
    scenes: &SceneLayout,
    workers: usize,
    visible: bool,
    frame_budget_bytes: u64,
    output: Box<dyn Write + Send>,
    denoisers: &mut Vec<Option<PlanarDenoiser>>,
    shared_permits: Option<&(crossbeam_channel::Sender<()>, crossbeam_channel::Receiver<()>)>,
) -> Result<(), anyhow::Error> {
    let radius = match opts.mode {
        DenoisingMode::Temporal { radius } => radius,
        DenoisingMode::Spacial => 0,
    };

    let frame_bytes = scenes.layout.luma_bytes() + 2 * scenes.layout.chroma_bytes();
    let (give, take, permits) = match shared_permits {
        Some((give, take)) => (give.clone(), take.clone(), take.capacity().unwrap_or(0)),
        None => {
            let permits = checked_frame_permits(frame_budget_bytes, frame_bytes, workers, radius)?;
            let (give, take) = frame_permit_channel(permits);
            (give, take, permits)
        },
    };

    tracing::info!(
        permits,
        frame_bytes,
        ceiling_mib = (permits * frame_bytes) / (1 << 20),
        "frame buffer budget",
    );

    denoisers.resize_with(workers, || None);
    let (job_tx, worker_handles, out_rx) = spawn_workers(opts, scenes.layout, std::mem::take(denoisers));
    let output_layout = FrameLayout {
        depth: opts.output_depth.unwrap_or(scenes.layout.depth),
        ..scenes.layout
    };
    let coordinator = spawn_coordinator(
        output_layout,
        scenes.framerate,
        out_rx,
        scenes.total_frames,
        visible,
        give,
        output,
    );

    dispatch_frames(input, scenes, &job_tx, &take)?;

    // Closing the queue is what tells the workers there are no more scenes.
    drop(job_tx);

    for h in worker_handles {
        denoisers.push(
            h.join()
                .map_err(|panic| anyhow::anyhow!("worker panicked: {}", panic_message(panic.as_ref())))??,
        );
    }

    coordinator
        .join()
        .map_err(|e| anyhow::anyhow!("coordinator panicked: {e:?}"))??;

    Ok(())
}

type WorkerJoin = thread::JoinHandle<Result<Option<PlanarDenoiser>, anyhow::Error>>;

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic payload")
}

/// Spawns `workers` worker threads over one shared scene queue.
///
/// The queue is a rendezvous, so a scene is only offered when a worker is
/// free and at most `workers` scenes are ever in flight.
///
/// Returns the queue's sender, their join handles, and the shared output
/// channel they emit denoised frames on.
fn spawn_workers(
    opts: &PlaneOptions,
    layout: FrameLayout,
    denoisers: Vec<Option<PlanarDenoiser>>,
) -> (
    crossbeam_channel::Sender<SceneJob>,
    Vec<WorkerJoin>,
    crossbeam_channel::Receiver<OutputMsg>,
) {
    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
    let (out_tx, out_rx) = crossbeam_channel::unbounded::<OutputMsg>();
    let mut worker_handles: Vec<WorkerJoin> = Vec::with_capacity(denoisers.len());

    for (worker_id, denoiser) in denoisers.into_iter().enumerate() {
        let opts = opts.clone();
        let out_tx = out_tx.clone();
        let job_rx = job_rx.clone();

        worker_handles.push(thread::spawn(move || {
            run_worker(worker_id, opts, layout, denoiser, job_rx, out_tx)
        }));
    }

    // Drop the original sender so the channel closes once every worker
    // clone has terminated.
    drop(out_tx);

    (job_tx, worker_handles, out_rx)
}

fn spawn_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: crossbeam_channel::Receiver<OutputMsg>,
    total_frames: usize,
    visible: bool,
    permits: crossbeam_channel::Sender<()>,
    output: Box<dyn Write + Send>,
) -> thread::JoinHandle<Result<(), anyhow::Error>> {
    thread::spawn(move || run_coordinator(layout, framerate, rx, total_frames, visible, permits, output))
}

/// Reads every frame in order and offers each scene to the worker pool.
///
/// A scene's frames go into a channel of their own. Dropping that
/// channel's sender is what tells the claiming worker the scene has
/// ended.
///
/// Each staged frame holds a permit from here until the coordinator has
/// written it, which is the only bound on frames in flight. The permit
/// is taken just before the send rather than before the decode, so a
/// phantom frame never takes one and at most one decoded frame is
/// transient outside the budget.
fn stage_frames<I>(
    frames: I,
    scenes: &SceneLayout,
    jobs: &crossbeam_channel::Sender<SceneJob>,
    permits: &crossbeam_channel::Receiver<()>,
) -> Result<(), anyhow::Error>
where
    I: Iterator<Item = Result<Planes, anyhow::Error>>,
{
    let mut scene_idx = 0usize;
    let mut next_boundary = scenes.scene_starts[1];
    let mut current: Option<(usize, crossbeam_channel::Sender<StagedFrame>)> = None;

    // The iterator yields frames in raw decoder order, so position is the
    // raw index. Every frame is read, phantom or not, because the decoder
    // walks the file in order and cannot be told to skip one.
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

            jobs.send(SceneJob {
                scene_idx: scene_idx as u32,
                frames: rx,
            })
            .map_err(|_| anyhow::anyhow!("worker pool disconnected"))?;

            current = Some((scene_idx, tx));
        }

        // Taken here rather than before the decode, so a phantom frame
        // never takes one. At most one decoded frame is transient outside
        // the budget.
        permits
            .recv()
            .map_err(|_| anyhow::anyhow!("the coordinator stopped before the stream finished"))?;

        let (_, tx) = current
            .as_ref()
            .expect("a scene sender exists after the check above");

        tx.send(StagedFrame {
            global_idx: g as u64,
            planes,
        })
        .map_err(|_| anyhow::anyhow!("the worker holding scene {scene_idx} disconnected"))?;
    }

    Ok(())
}

/// Opens the input and stages every frame it decodes.
fn dispatch_frames(
    input: &Path,
    scenes: &SceneLayout,
    jobs: &crossbeam_channel::Sender<SceneJob>,
    permits: &crossbeam_channel::Receiver<()>,
) -> Result<(), anyhow::Error> {
    let index_path = std::path::PathBuf::from(format!("{}.ffindex", input.to_string_lossy()));
    let index_existed = index_path.try_exists()?;
    let decoder = Decoder::from_file(input);
    if !index_existed {
        let _ = std::fs::remove_file(index_path);
    }
    let mut decoder = decoder?;
    let layout = scenes.layout;

    let ranges = scenes.source_ranges.clone();
    let phantom = scenes.phantom.clone();
    let mut range_index = 0usize;
    let mut remaining = 0usize;
    let mut raw_index = 0usize;
    let frames = std::iter::from_fn(move || -> Option<Result<Planes, anyhow::Error>> {
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
    });

    stage_frames(frames, scenes, jobs, permits)
}

/// One decoded frame, staged for the worker that claims its scene.
struct StagedFrame {
    global_idx: u64,
    planes: Planes,
}

/// One scene, offered to whichever worker is free.
///
/// `frames` closes when the scene has no more frames, which is how a
/// worker knows to flush.
struct SceneJob {
    scene_idx: u32,
    frames: crossbeam_channel::Receiver<StagedFrame>,
}

struct OutputMsg {
    global_idx: u64,
    planes: Planes,
}

fn run_worker(
    worker_id: usize,
    opts: PlaneOptions,
    layout: FrameLayout,
    mut wd: Option<PlanarDenoiser>,
    jobs: crossbeam_channel::Receiver<SceneJob>,
    tx: crossbeam_channel::Sender<OutputMsg>,
) -> Result<Option<PlanarDenoiser>, anyhow::Error> {
    // The cold-cache queue place this worker's denoiser holds, until its
    // first output frame proves the kernels are compiled and cached.
    let mut warm_up: Option<WarmUp> = None;

    while let Ok(job) = jobs.recv() {
        // Built on the first claimed scene, so a worker that never claims
        // one never compiles.
        if wd.is_none() {
            let (denoiser, place) = create_denoiser(&opts, layout)?;
            wd = Some(denoiser);
            warm_up = place;
        }

        let denoiser = wd.as_mut().expect("denoiser exists after the check above");

        tracing::debug!(worker_id, scene_idx = job.scene_idx, "worker started scene");

        // Indices of pushed-but-not-yet-emitted frames, in push order.
        let mut pending: std::collections::VecDeque<u64> = Default::default();

        // Nothing is received straight after the push.
        // `push_with_drain` handles backpressure through QueueFull
        // when the 2-deep pending pipeline fills, and `flush_worker`
        // drains the tail below. Receiving after every push would clamp
        // the pipeline back to depth 1 and put the GPU readback in the
        // critical path of the next push.
        for frame in job.frames {
            push_with_drain(
                denoiser,
                &mut warm_up,
                &mut pending,
                frame.global_idx,
                &frame.planes,
                &tx,
            )?;
        }

        // Reuse the PlanarDenoiser across scenes. Flushing here ensures
        // no temporal window spans two of them.
        flush_worker(denoiser, &mut warm_up, &mut pending, &tx)?;
    }

    Ok(wd)
}

/// Push one frame, draining any pending output first if the queue is full.
fn push_with_drain(
    denoiser: &mut PlanarDenoiser,
    warm_up: &mut Option<WarmUp>,
    pending: &mut std::collections::VecDeque<u64>,
    global_idx: u64,
    planes: &Planes,
    tx: &crossbeam_channel::Sender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    pending.push_back(global_idx);

    if push_needs_retry(denoiser.push(planes))? {
        if let Some(out) = denoiser.recv()? {
            let g = pending
                .pop_front()
                .expect("pending has at least one entry on QueueFull recv");
            send_output(tx, g, out)?;
            finish_warm_up(warm_up);
        }

        denoiser.push(planes)?;
    }

    Ok(())
}

fn send_output(
    tx: &crossbeam_channel::Sender<OutputMsg>,
    global_idx: u64,
    planes: Planes,
) -> Result<(), anyhow::Error> {
    tx.send(OutputMsg { global_idx, planes })
        .map_err(|_| anyhow::anyhow!("coordinator disconnected"))
}

fn flush_worker(
    wd: &mut PlanarDenoiser,
    warm_up: &mut Option<WarmUp>,
    pending: &mut std::collections::VecDeque<u64>,
    tx: &crossbeam_channel::Sender<OutputMsg>,
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
            if did_send {
                finish_warm_up(warm_up);
            } else {
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
    rx: crossbeam_channel::Receiver<OutputMsg>,
    total_frames: usize,
    visible: bool,
    permits: crossbeam_channel::Sender<()>,
    output: Box<dyn Write + Send>,
) -> Result<(), anyhow::Error> {
    // No `XCOLORRANGE=` tag is emitted here. `av_decoders::VideoDetails`
    // doesn't surface the source's color range for any of its backends
    // (ffms2 included), so there's nothing to forward.
    let mut encoder = y4m::encode(
        layout.width as usize,
        layout.height as usize,
        y4m::Ratio::new((*framerate.numer()) as usize, (*framerate.denom()) as usize),
    )
    .with_colorspace(subsampling_to_y4m(layout.subsampling, layout.depth))
    .write_header(output)?;

    // Counts frames written to the output, which lags the frames read by
    // the depth of the worker pipelines. Emitted frames are the honest
    // measure of progress, because the count stalls whenever whatever
    // consumes our stdout stops reading.
    let pb = denoise_progress_bar(total_frames, visible);

    // The first frame only lands once a worker has compiled its
    // kernels, which takes seconds. A steady tick draws the bar right
    // away and keeps its elapsed time moving until then.
    pb.enable_steady_tick(Duration::from_millis(250));

    let result = emit_frames(&mut encoder, &rx, total_frames as u64, &pb, &permits);

    progress::finish(&pb);

    result
}

/// Reorders worker output by frame index and writes it out, updating
/// `pb` as frames are emitted.
///
/// Returns once every frame has been written. If the workers all
/// disconnect before `total` frames have landed this errors, naming how
/// many were written and how many were expected.
fn emit_frames<W: std::io::Write>(
    encoder: &mut y4m::Encoder<W>,
    rx: &crossbeam_channel::Receiver<OutputMsg>,
    total: u64,
    pb: &ProgressBar,
    permits: &crossbeam_channel::Sender<()>,
) -> Result<(), anyhow::Error> {
    let mut pending: BTreeMap<u64, Planes> = BTreeMap::new();
    let mut next_emit: u64 = 0;

    while next_emit < total {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => break,
        };

        pending.insert(msg.global_idx, msg.planes);

        while let Some(planes) = pending.remove(&next_emit) {
            let frame = Y4mFrame::new([&planes.y, &planes.u, &planes.v], None);
            encoder.write_frame(&frame)?;
            next_emit += 1;

            // Returning the permit is what lets the decoder run further
            // ahead. The send never blocks, because permits held plus
            // permits waiting is always the channel's capacity. The
            // result is discarded because it fails once the dispatcher
            // has already errored out and dropped its receiver.
            let _ = permits.send(());
        }

        pb.set_position(next_emit);
    }

    if next_emit != total {
        anyhow::bail!(
            "wrote {next_emit} frames but expected {total}. Every worker disconnected \
             before the stream finished, so a frame index was likely lost"
        );
    }

    Ok(())
}

/// Checks each plane's byte length against the layout, failing with an error naming which plane
/// is wrong, the length found and the length expected.
fn check_plane_lens(planes: &Planes, layout: FrameLayout) -> Result<(), anyhow::Error> {
    for (name, got, expected) in [
        ("y", planes.y.len(), layout.luma_bytes()),
        ("u", planes.u.len(), layout.chroma_bytes()),
        ("v", planes.v.len(), layout.chroma_bytes()),
    ] {
        if got != expected {
            anyhow::bail!("{name} plane is {got} bytes, expected {expected} from the frame layout");
        }
    }
    Ok(())
}

fn planes_from_v_frame_u8(
    frame: &v_frame::frame::Frame<u8>,
    layout: FrameLayout,
) -> Result<Planes, anyhow::Error> {
    let y = collect_plane_u8(&frame.y_plane);
    let u = frame
        .u_plane
        .as_ref()
        .map(collect_plane_u8)
        .unwrap_or_else(|| layout.neutral_chroma_plane());
    let v = frame
        .v_plane
        .as_ref()
        .map(collect_plane_u8)
        .unwrap_or_else(|| layout.neutral_chroma_plane());

    let planes = Planes { y, u, v };
    check_plane_lens(&planes, layout)?;

    Ok(planes)
}

fn planes_from_v_frame_u16(
    frame: &v_frame::frame::Frame<u16>,
    layout: FrameLayout,
) -> Result<Planes, anyhow::Error> {
    let y = collect_plane_u16(&frame.y_plane);
    let u = frame
        .u_plane
        .as_ref()
        .map(collect_plane_u16)
        .unwrap_or_else(|| layout.neutral_chroma_plane());
    let v = frame
        .v_plane
        .as_ref()
        .map(collect_plane_u16)
        .unwrap_or_else(|| layout.neutral_chroma_plane());

    let planes = Planes { y, u, v };
    check_plane_lens(&planes, layout)?;

    Ok(planes)
}

fn collect_plane_u8(plane: &v_frame::plane::Plane<u8>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = Vec::with_capacity(width * height);

    for row in plane.rows() {
        out.extend_from_slice(row);
    }

    out
}

fn collect_plane_u16(plane: &v_frame::plane::Plane<u16>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = Vec::with_capacity(width * height * 2);

    for row in plane.rows() {
        for &s in row {
            out.extend_from_slice(&s.to_le_bytes());
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
