use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread;

use av_decoders::Decoder;
use av_denoise_core::{FrameLayout, PlaneOptions};

use crate::SceneLayout;
use crate::budget::{FramePermits, temporal_radius};
use crate::cancel::Cancel;
use crate::dispatch::{Staging, dispatch_frames};
use crate::source::{OpenedSource, open_source};
use crate::warm::create_denoiser;
use crate::window::{Window, WindowJob};
use crate::worker::{Resident, spawn_workers};

/// The decode, denoise and budget settings every window of one source
/// runs with.
pub(crate) struct Pipeline {
    planes: PlaneOptions,
    source: PathBuf,
    workers: usize,
    permits: FramePermits,
    opened: Mutex<Option<OpenedSource>>,
}

impl Pipeline {
    /// Sizes the frame budget for `streams` windows running at once.
    pub(crate) fn new(
        planes: PlaneOptions,
        source: PathBuf,
        workers: usize,
        layout: FrameLayout,
        frame_budget: u64,
        streams: usize,
    ) -> Result<Self, anyhow::Error> {
        if workers == 0 {
            anyhow::bail!("--workers must be at least 1");
        }

        let frame_bytes = layout.luma_bytes() + 2 * layout.chroma_bytes();
        let permits = FramePermits::checked(
            frame_budget,
            frame_bytes,
            workers * streams,
            temporal_radius(planes.mode),
        )?;

        tracing::info!(
            permits = permits.count(),
            frame_bytes,
            ceiling_mib = (permits.count() * frame_bytes) / (1 << 20),
            "frame buffer budget",
        );

        Ok(Self {
            planes,
            source,
            workers,
            permits,
            opened: Mutex::new(None),
        })
    }

    pub(crate) fn keep_opened(&self, decoder: Decoder) {
        *self
            .opened
            .lock()
            .expect("the opened source lock is never poisoned") = Some(OpenedSource(decoder));
    }

    fn take_decoder(&self) -> Result<Decoder, anyhow::Error> {
        let opened = self
            .opened
            .lock()
            .expect("the opened source lock is never poisoned")
            .take();
        match opened {
            Some(OpenedSource(decoder)) => Ok(decoder),
            None => open_source(&self.source),
        }
    }

    pub(crate) fn create_denoisers(
        &self,
        layout: FrameLayout,
    ) -> Result<Vec<Option<Resident>>, anyhow::Error> {
        thread::scope(|scope| {
            let builds: Vec<_> = (0..self.workers)
                .map(|_| scope.spawn(|| create_denoiser(&self.planes, layout)))
                .collect();
            builds
                .into_iter()
                .map(|build| {
                    build
                        .join()
                        .map_err(|panic| {
                            anyhow::anyhow!("denoiser build panicked: {}", panic_message(panic.as_ref()))
                        })?
                        .map(Some)
                })
                .collect()
        })
    }

    pub(crate) fn output_layout(&self, source: FrameLayout) -> FrameLayout {
        FrameLayout {
            depth: self.planes.output_depth.unwrap_or(source.depth),
            ..source
        }
    }

    /// Starts the worker pool, then drives the dispatch loop until the
    /// window's last frame is staged.
    ///
    /// Blocks until every worker has finished, handing their denoisers
    /// back for the next window.
    pub(crate) fn run(
        &self,
        job: WindowJob,
        denoisers: &mut Vec<Option<Resident>>,
    ) -> Result<(), anyhow::Error> {
        let scenes = job.scenes()?;

        denoisers.resize_with(self.workers, || None);
        let (job_tx, worker_handles) =
            spawn_workers(&self.planes, scenes.layout, std::mem::take(denoisers), job.output);

        dispatch_frames(
            self.take_decoder()?,
            &scenes,
            &Staging {
                jobs: &job_tx,
                permits: &self.permits,
                closed: &job.closed,
                cancel: &job.cancel,
            },
        )?;

        // Closing the queue is what tells the workers there are no more scenes.
        drop(job_tx);

        for h in worker_handles {
            denoisers.push(
                h.join().map_err(|panic| {
                    anyhow::anyhow!("worker panicked: {}", panic_message(panic.as_ref()))
                })??,
            );
        }

        Ok(())
    }
}

pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic payload")
}

/// Denoises every scene of `scenes` as one window with fresh denoisers.
pub fn denoise_scenes(
    planes: PlaneOptions,
    source: PathBuf,
    scenes: SceneLayout,
    workers: usize,
    frame_budget: u64,
    cancel: Arc<AtomicBool>,
) -> Result<Window, anyhow::Error> {
    let pipeline = Pipeline::new(planes, source, workers, scenes.layout, frame_budget, 1)?;
    let (window, job, done) = Window::open(
        pipeline.output_layout(scenes.layout),
        Arc::new(scenes),
        None,
        Cancel::new(vec![cancel]),
    );

    thread::spawn(move || {
        let _ = done.send(pipeline.run(job, &mut Vec::new()));
    });

    Ok(window)
}
