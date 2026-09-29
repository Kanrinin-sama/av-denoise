use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread;

use av_decoders::Decoder;
use av_denoise_core::{FrameLayout, PlaneOptions};

use crate::budget::{FramePermits, temporal_radius};
use crate::threads::release_affinity;
use crate::{CreationGuard, SceneLayout};
use crate::dispatch::{Staging, dispatch_frames};
use crate::source::{OpenedSource, SourceIndex};
use crate::warm::{create_denoiser, warm_resident};
use crate::window::{Window, WindowJob};
use crate::worker::{Resident, spawn_workers};

/// The decode, denoise and budget settings every window of one source
/// runs with.
pub(crate) struct Pipeline {
    planes: PlaneOptions,
    index: SourceIndex,
    workers: usize,
    permits: FramePermits,
    guard: CreationGuard,
    opened: Mutex<Option<OpenedSource>>,
}

impl Pipeline {
    /// Sizes the frame budget for `streams` windows running at once.
    pub(crate) fn new(
        planes: PlaneOptions,
        index: SourceIndex,
        workers: usize,
        layout: FrameLayout,
        frame_budget: u64,
        streams: usize,
        guard: CreationGuard,
    ) -> Result<Self, anyhow::Error> {
        if workers == 0 {
            anyhow::bail!("--workers must be at least 1");
        }

        let frame_bytes = layout.luma_bytes() + 2 * layout.chroma_bytes();
        let permits = FramePermits::checked(frame_budget, frame_bytes, workers, temporal_radius(planes.mode), streams)?;

        tracing::info!(
            permits = permits.count(),
            permits_per_window = permits.cap(),
            frame_bytes,
            ceiling_mib = (permits.count() * frame_bytes) / (1 << 20),
            "frame buffer budget",
        );

        Ok(Self {
            planes,
            index,
            workers,
            permits,
            guard,
            opened: Mutex::new(None),
        })
    }

    pub(crate) fn keep_opened(&self, decoder: Decoder) {
        *self
            .opened
            .lock()
            .expect("the opened source lock is never poisoned") = Some(OpenedSource(decoder));
    }

    fn decoder<'a>(&self, decoder: &'a mut Option<Decoder>) -> Result<&'a mut Decoder, anyhow::Error> {
        if decoder.is_none() {
            let opened = self
                .opened
                .lock()
                .expect("the opened source lock is never poisoned")
                .take();
            *decoder = Some(match opened {
                Some(OpenedSource(opened)) => opened,
                None => self.guard.run(|| self.index.decoder())?,
            });
        }
        Ok(decoder.as_mut().expect("the decoder was opened above"))
    }

    pub(crate) fn warmed_resident(&self, layout: FrameLayout, cancel: &AtomicBool) -> Result<Resident, anyhow::Error> {
        let mut resident = self.guard.run(|| create_denoiser(&self.planes, layout))?;
        warm_resident(&mut resident, layout, cancel)?;
        Ok(resident)
    }

    pub(crate) fn build_residents(
        &self,
        residents: &mut Vec<Option<Resident>>,
        layout: FrameLayout,
    ) -> Result<(), anyhow::Error> {
        residents.resize_with(self.workers, || None);
        thread::scope(|scope| {
            let builds: Vec<_> = residents
                .iter_mut()
                .filter(|resident| resident.is_none())
                .map(|resident| {
                    scope.spawn(move || -> Result<(), anyhow::Error> {
                        release_affinity();
                        *resident = Some(self.guard.run(|| create_denoiser(&self.planes, layout))?);
                        Ok(())
                    })
                })
                .collect();
            let mut result = Ok(());
            for build in builds {
                let built = build.join().map_err(|panic| {
                    anyhow::anyhow!("denoiser build panicked: {}", panic_message(panic.as_ref()))
                });
                if let Err(error) = built.and_then(|built| built) {
                    result = result.and(Err(error));
                }
            }
            result
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
    /// Blocks until every worker has finished, whether the window
    /// succeeded or not, handing their denoisers back for the next
    /// window. `decoder` is kept open for the next window too.
    pub(crate) fn run(
        &self,
        job: WindowJob,
        residents: &mut Vec<Option<Resident>>,
        decoder: &mut Option<Decoder>,
    ) -> Result<(), anyhow::Error> {
        let scenes = job.scenes()?;
        let permits = self.permits.window();

        residents.resize_with(self.workers, || None);
        let (job_tx, worker_handles) = spawn_workers(
            &self.planes,
            scenes.layout,
            std::mem::take(residents),
            job.output,
            &job.cancel,
            &job.abort,
            &self.guard,
        );

        let dispatched = self.decoder(decoder).and_then(|opened| {
            dispatch_frames(
                opened,
                &scenes,
                &Staging {
                    jobs: &job_tx,
                    permits: &permits,
                    closed: &job.closed,
                    cancel: &job.cancel,
                },
            )
        });
        if dispatched.is_err() {
            *decoder = None;
        }

        // Closing the queue is what tells the workers there are no more scenes.
        drop(job_tx);

        let mut failure = None;
        for handle in worker_handles {
            let resident = match handle.join() {
                Ok(Ok(resident)) => resident,
                Ok(Err(error)) => {
                    failure.get_or_insert(error);
                    None
                },
                Err(panic) => {
                    failure.get_or_insert(anyhow::anyhow!("worker panicked: {}", panic_message(panic.as_ref())));
                    None
                },
            };
            residents.push(resident);
        }

        match failure {
            Some(error) => Err(error),
            None => dispatched,
        }
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
    let index = SourceIndex::open(&source)?;
    let pipeline = Pipeline::new(planes, index, workers, scenes.layout, frame_budget, 1, CreationGuard::default())?;
    let (window, job, done) = Window::open(
        pipeline.output_layout(scenes.layout),
        Arc::new(scenes),
        None,
        vec![cancel],
    );

    thread::spawn(move || {
        release_affinity();
        let _ = done.send(pipeline.run(job, &mut Vec::new(), &mut None));
    });

    Ok(window)
}
