use std::io::{IsTerminal, Write, stdout};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use av_denoise::SceneGrain;
use av_denoise_service::{FrameRange, SceneLayout, Window, denoise_scenes};
use indicatif::ProgressBar;
use y4m::Frame as Y4mFrame;

use crate::cli::RunOptions;
use crate::grain_table::TablePath;
use crate::progress::{self, denoise_bar_visible, denoise_progress_bar};
use crate::scene_scan;
use crate::y4m_format::create_encoder;

pub fn run_file(
    opts: &RunOptions,
    input: &Path,
    workers: usize,
    frame_budget_bytes: u64,
    stored_layout: Option<&Path>,
    keep_frames: &[FrameRange],
    scene_span: Option<FrameRange>,
) -> Result<(), anyhow::Error> {
    let table = opts.grain_table.as_deref().map(TablePath::create).transpose()?;
    // Scene detection finishes before a single frame is written, so its
    // bar has the terminal to itself and needs no opt-in. The denoising
    // bar shares the terminal with whatever consumes our output, so it
    // waits for --progress.
    let is_terminal = std::io::stderr().is_terminal();
    let scenes = match stored_layout {
        Some(path) => SceneLayout::read(input, path, keep_frames)?,
        None if keep_frames.is_empty() => scene_scan::detect(input, keep_frames, None, is_terminal)?,
        None => anyhow::bail!("--keep-frames requires --scene-layout; run `scenes` first"),
    };
    let scenes = match scene_span {
        Some(span) if stored_layout.is_some() => scenes.select_span(span)?,
        Some(_) => anyhow::bail!("--scene-span requires --scene-layout"),
        None => scenes,
    };

    tracing::info!(
        scene_count = scenes.scene_count(),
        total_frames = scenes.total_frames,
        workers,
        "scene detection complete",
    );

    let window = denoise_scenes(
        opts.planes.clone(),
        input.to_path_buf(),
        scenes,
        workers,
        frame_budget_bytes,
        Arc::new(AtomicBool::new(false)),
    )?;
    let framerate = window.frame_rate();
    let (_, grain) =
        write_window_with_grain(window, stdout(), denoise_bar_visible(opts.progress, is_terminal))?;
    if let Some(table) = table {
        table.write(&grain, (*framerate.numer() as u64, *framerate.denom() as u64))?;
    }

    Ok(())
}

pub fn write_scene_layout(
    input: &Path,
    output: &Path,
    keep_frames: &[FrameRange],
    ffmpeg: Option<&Path>,
) -> Result<(), anyhow::Error> {
    scene_scan::detect(input, keep_frames, ffmpeg, std::io::stderr().is_terminal())?.write(output)
}

/// Writes a window's frames as y4m, returning how many it produced.
pub fn write_window(window: Window, output: impl Write, visible: bool) -> Result<usize, anyhow::Error> {
    write_window_with_grain(window, output, visible).map(|(frames, _)| frames)
}

fn write_window_with_grain(
    mut window: Window,
    output: impl Write,
    visible: bool,
) -> Result<(usize, Vec<SceneGrain>), anyhow::Error> {
    let layout = window.layout();
    let framerate = window.frame_rate();

    // No `XCOLORRANGE=` tag is emitted here. `av_decoders::VideoDetails`
    // doesn't surface the source's color range for any of its backends
    // (ffms2 included), so there's nothing to forward.
    let mut encoder = create_encoder(
        layout,
        y4m::Ratio::new((*framerate.numer()) as usize, (*framerate.denom()) as usize),
        None,
        Vec::new(),
        output,
    )?;

    // Counts frames written to the output, which lags the frames read by
    // the depth of the worker pipelines. Emitted frames are the honest
    // measure of progress, because the count stalls whenever whatever
    // consumes our output stops reading.
    let pb = denoise_progress_bar(Some(window.frames()), visible);

    let result = write_frames(&mut encoder, &mut window, &pb);

    progress::finish(&pb);

    result?;
    window.finish_with_grain()
}

fn write_frames<W: Write>(
    encoder: &mut y4m::Encoder<W>,
    window: &mut Window,
    pb: &ProgressBar,
) -> Result<(), anyhow::Error> {
    while let Some(planes) = window.recv() {
        let planes = planes?;
        encoder.write_frame(&Y4mFrame::new([&planes.y, &planes.u, &planes.v], None))?;
        pb.inc(1);
    }

    Ok(())
}
