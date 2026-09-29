use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use av_denoise_core::FrameLayout;
use crossbeam_channel::{Receiver, Sender};

use crate::pipeline::{Pipeline, panic_message};
use crate::source::SourceIndex;
use crate::threads::release_affinity;
use crate::stored_layout::UncheckedLayout;
use crate::window::{Window, WindowJob};
use crate::worker::Resident;
use crate::{FrameRange, SceneLayout, ServiceConfig};

struct SlotRequest {
    job: WindowJob,
    done: Sender<Result<(), anyhow::Error>>,
}

struct Slot {
    requests: Sender<SlotRequest>,
    busy: Arc<AtomicBool>,
    handle: thread::JoinHandle<()>,
}

/// A resident set of window slots over one source, each keeping its
/// denoisers warm between windows.
pub struct WindowService {
    slots: Vec<Slot>,
    scenes: Arc<SceneLayout>,
    output_layout: FrameLayout,
    cancel: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl WindowService {
    pub fn start(config: ServiceConfig, cancel: Arc<AtomicBool>) -> Result<Self, anyhow::Error> {
        if config.slots == 0 {
            anyhow::bail!("--window-service-slots must be at least 1");
        }
        if config.workers == 0 {
            anyhow::bail!("--workers must be at least 1");
        }
        let unchecked = UncheckedLayout::load(&config.scene_layout, &config.keep_frames)?;
        let source_layout = unchecked.frame_layout()?;
        let guard = config.creation_guard;
        let index = guard.run(|| SourceIndex::open(&config.source))?;
        let decoder = guard.run(|| index.decoder())?;
        let scenes = unchecked.check(&decoder)?;
        let pipeline = Arc::new(Pipeline::new(
            config.planes,
            index,
            config.workers,
            source_layout,
            config.frame_budget,
            config.slots,
            guard,
        )?);
        pipeline.keep_opened(decoder);
        let mut warmed = Some(pipeline.warmed_resident(source_layout, &cancel)?);
        let slots = (0..config.slots)
            .map(|_| {
                let (requests, rx) = crossbeam_channel::bounded::<SlotRequest>(1);
                let busy = Arc::new(AtomicBool::new(false));
                let pipeline = Arc::clone(&pipeline);
                let slot_busy = Arc::clone(&busy);
                let residents = vec![warmed.take()];
                let handle = thread::spawn(move || {
                    release_affinity();
                    run_slot(&pipeline, source_layout, &slot_busy, &rx, residents)
                });
                Slot {
                    requests,
                    busy,
                    handle,
                }
            })
            .collect();
        Ok(Self {
            slots,
            output_layout: pipeline.output_layout(scenes.layout),
            scenes: Arc::new(scenes),
            cancel,
            stop: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn source_layout(&self) -> FrameLayout {
        self.scenes.layout
    }

    /// Starts denoising the retained frames `start..end` on `slot`.
    ///
    /// Both ends have to fall on scene boundaries of the stored layout.
    pub fn open(
        &self,
        slot: usize,
        start: usize,
        end: usize,
        cancel: Arc<AtomicBool>,
    ) -> Result<Window, anyhow::Error> {
        if start >= end {
            anyhow::bail!("window start must be less than end");
        }
        let Some(target) = self.slots.get(slot) else {
            anyhow::bail!("window slot {slot} is out of range");
        };
        if target
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            anyhow::bail!("window slot {slot} is busy");
        }
        let (window, job, done) = Window::open(
            self.output_layout,
            Arc::clone(&self.scenes),
            Some(FrameRange { start, end }),
            vec![Arc::clone(&self.cancel), Arc::clone(&self.stop), cancel],
        );
        if target.requests.try_send(SlotRequest { job, done }).is_err() {
            target.busy.store(false, Ordering::Release);
            anyhow::bail!("window slot {slot} is unavailable");
        }
        Ok(window)
    }

    /// Stops every slot once no window is active.
    pub fn finish(mut self) -> Result<(), anyhow::Error> {
        if self.slots.iter().any(|slot| slot.busy.load(Ordering::Acquire)) {
            anyhow::bail!("finish received while a window is active");
        }
        self.join_slots()
    }

    fn join_slots(&mut self) -> Result<(), anyhow::Error> {
        let mut result = Ok(());
        for slot in self.slots.drain(..) {
            drop(slot.requests);
            if let Err(panic) = slot.handle.join() {
                result = result.and(Err(anyhow::anyhow!(
                    "window worker panicked: {}",
                    panic_message(panic.as_ref())
                )));
            }
        }
        result
    }
}

impl Drop for WindowService {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.join_slots();
    }
}

fn run_slot(
    pipeline: &Pipeline,
    layout: FrameLayout,
    busy: &AtomicBool,
    requests: &Receiver<SlotRequest>,
    mut residents: Vec<Option<Resident>>,
) {
    if let Err(error) = pipeline.build_residents(&mut residents, layout) {
        tracing::warn!(error = format!("{error:#}"), "window slot denoisers failed to build");
    }
    let mut decoder = None;
    while let Ok(request) = requests.recv() {
        let result = pipeline
            .build_residents(&mut residents, layout)
            .and_then(|()| pipeline.run(request.job, &mut residents, &mut decoder));
        busy.store(false, Ordering::Release);
        let _ = request.done.send(result);
    }
}
