use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use av_denoise_core::FrameLayout;
use crossbeam_channel::{Receiver, Sender};

use crate::cancel::Cancel;
use crate::pipeline::{Pipeline, panic_message};
use crate::source::open_source;
use crate::stored_layout::UncheckedLayout;
use crate::window::{Window, WindowJob};
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
        let pipeline = Arc::new(Pipeline::new(
            config.planes,
            config.source.clone(),
            config.workers,
            source_layout,
            config.frame_budget,
            config.slots,
        )?);
        let slots = (0..config.slots)
            .map(|_| {
                let (requests, rx) = crossbeam_channel::bounded::<SlotRequest>(1);
                let busy = Arc::new(AtomicBool::new(false));
                let pipeline = Arc::clone(&pipeline);
                let slot_busy = Arc::clone(&busy);
                let handle = thread::spawn(move || run_slot(&pipeline, source_layout, &slot_busy, &rx));
                Slot {
                    requests,
                    busy,
                    handle,
                }
            })
            .collect();
        let decoder = open_source(&config.source)?;
        let scenes = unchecked.check(&decoder)?;
        pipeline.keep_opened(decoder);
        Ok(Self {
            slots,
            output_layout: pipeline.output_layout(scenes.layout),
            scenes: Arc::new(scenes),
            cancel,
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
            Cancel::new(vec![Arc::clone(&self.cancel), cancel]),
        );
        if target.requests.try_send(SlotRequest { job, done }).is_err() {
            target.busy.store(false, Ordering::Release);
            anyhow::bail!("window slot {slot} is unavailable");
        }
        Ok(window)
    }

    /// Stops every slot once no window is active.
    pub fn finish(self) -> Result<(), anyhow::Error> {
        if self.slots.iter().any(|slot| slot.busy.load(Ordering::Acquire)) {
            anyhow::bail!("finish received while a window is active");
        }
        for slot in self.slots {
            drop(slot.requests);
            slot.handle.join().map_err(|panic| {
                anyhow::anyhow!("window worker panicked: {}", panic_message(panic.as_ref()))
            })?;
        }
        Ok(())
    }
}

fn run_slot(pipeline: &Pipeline, layout: FrameLayout, busy: &AtomicBool, requests: &Receiver<SlotRequest>) {
    let mut denoisers = pipeline.create_denoisers(layout);
    while let Ok(request) = requests.recv() {
        let result = match &mut denoisers {
            Ok(denoisers) => pipeline.run(request.job, denoisers),
            Err(error) => Err(anyhow::anyhow!("{error:#}")),
        };
        busy.store(false, Ordering::Release);
        let _ = request.done.send(result);
    }
}
