use std::io::{BufRead, BufWriter, Stdout, Write, stdout};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread;

use av_denoise_service::{FrameRange, ServiceConfig, Window, WindowService};

use crate::cli::RunOptions;
use crate::file_mode::write_window;

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WindowCommand {
    Start {
        id: u64,
        slot: usize,
        start: usize,
        end: usize,
        output_path: PathBuf,
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

type Events = Mutex<BufWriter<Stdout>>;

/// Serves windows from a [`WindowService`] over the stdin/stdout control
/// protocol, writing each window as y4m to the pipe its command names.
pub fn run_window_service(
    opts: &RunOptions,
    input: &Path,
    workers: usize,
    frame_budget_bytes: u64,
    stored_layout: &Path,
    keep_frames: &[FrameRange],
    slots: usize,
) -> Result<(), anyhow::Error> {
    let service = WindowService::start(
        ServiceConfig {
            source: input.to_path_buf(),
            planes: opts.planes.clone(),
            scene_layout: stored_layout.to_path_buf(),
            keep_frames: keep_frames.to_vec(),
            slots,
            workers,
            frame_budget: frame_budget_bytes,
        },
        Arc::new(AtomicBool::new(false)),
    )?;
    let events = Events::new(BufWriter::new(stdout()));
    thread::scope(|scope| {
        write_window_event(&events, &WindowEvent::Ready { version: 1, slots })?;
        let mut writers = Vec::new();
        let mut finished = false;
        for line in std::io::stdin().lock().lines() {
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
                    let window = service
                        .open(slot, start, end, Arc::new(AtomicBool::new(false)))
                        .unwrap_or_else(|error| control_fatal(&format!("{error:#}")));
                    let events = &events;
                    writers.push(scope.spawn(move || serve_window(events, id, slot, window, &output_path)));
                },
                WindowCommand::Finish => {
                    finished = true;
                    break;
                },
            }
        }
        if !finished {
            std::process::exit(1);
        }
        service
            .finish()
            .unwrap_or_else(|error| control_fatal(&format!("{error:#}")));
        for writer in writers {
            writer
                .join()
                .map_err(|_| anyhow::anyhow!("window writer panicked"))??;
        }
        Ok(())
    })
}

/// Writes one window to its pipe and reports how it went.
fn serve_window(
    events: &Events,
    id: u64,
    slot: usize,
    window: Window,
    output_path: &Path,
) -> Result<(), anyhow::Error> {
    let result = std::fs::OpenOptions::new()
        .write(true)
        .open(output_path)
        .map_err(anyhow::Error::from)
        .and_then(|output| write_window(window, output, false));
    match result {
        Ok(frames) => write_window_event(events, &WindowEvent::Done { id, slot, frames }),
        Err(error) => {
            let message = format!("{error:#}");
            write_window_event(
                events,
                &WindowEvent::Error {
                    id,
                    slot,
                    message: &message,
                },
            )?;
            std::process::exit(1);
        },
    }
}

fn control_fatal(message: &str) -> ! {
    tracing::error!(message, "window control protocol failed");
    std::process::exit(2)
}

fn write_window_event(output: &Events, event: &WindowEvent<'_>) -> Result<(), anyhow::Error> {
    let mut output = output.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    serde_json::to_writer(&mut *output, event)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}
