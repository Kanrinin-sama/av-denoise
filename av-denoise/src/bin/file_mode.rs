use std::io::{IsTerminal, Write, stdout};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use av_denoise_service::{FrameRange, SceneLayout, Window, denoise_scenes};
use indicatif::ProgressBar;
use y4m::Frame as Y4mFrame;

use crate::cli::RunOptions;
use crate::progress::{self, denoise_bar_visible, denoise_progress_bar};
use crate::scene_scan;
use crate::y4m_format::subsampling_to_y4m;

pub fn run_file(
    opts: &RunOptions,
    input: &Path,
    workers: usize,
    frame_budget_bytes: u64,
    stored_layout: Option<&Path>,
    keep_frames: &[FrameRange],
    scene_span: Option<FrameRange>,
) -> Result<(), anyhow::Error> {
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
    write_window(window, stdout(), denoise_bar_visible(opts.progress, is_terminal))?;

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
pub fn write_window(mut window: Window, output: impl Write, visible: bool) -> Result<usize, anyhow::Error> {
    let layout = window.layout();
    let framerate = window.frame_rate();

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
    // consumes our output stops reading.
    let pb = denoise_progress_bar(window.frames(), visible);

    // The first frame only lands once a worker has compiled its
    // kernels, which takes seconds. A steady tick draws the bar right
    // away and keeps its elapsed time moving until then.
    pb.enable_steady_tick(Duration::from_millis(250));

    let result = write_frames(&mut encoder, &mut window, &pb);

    progress::finish(&pb);

    result?;
    window.finish()
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
