use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use av_decoders::Rational32;
use av_denoise_core::{FrameLayout, Planes};
use crossbeam_channel::{Receiver, Sender};

use crate::budget::CANCEL_POLL;
use crate::cancel::Cancel;
use crate::worker::OutputMsg;
use crate::{FrameRange, SceneLayout};

/// The half of a window the pipeline runs.
pub(crate) struct WindowJob {
    scenes: Arc<SceneLayout>,
    span: Option<FrameRange>,
    pub(crate) output: Sender<OutputMsg>,
    pub(crate) closed: Receiver<()>,
    pub(crate) cancel: Cancel,
    pub(crate) abort: Arc<AtomicBool>,
}

impl WindowJob {
    pub(crate) fn scenes(&self) -> Result<SceneLayout, anyhow::Error> {
        let scenes = SceneLayout::clone(&self.scenes);
        match self.span {
            Some(span) => scenes.select_span(span),
            None => Ok(scenes),
        }
    }
}

/// One span of denoised frames, yielded in order at the output depth.
pub struct Window {
    layout: FrameLayout,
    frame_rate: Rational32,
    total: u64,
    next: u64,
    output: Receiver<OutputMsg>,
    reorder: BTreeMap<u64, OutputMsg>,
    done: Option<Receiver<Result<(), anyhow::Error>>>,
    failure: Option<anyhow::Error>,
    cancel: Cancel,
    stop: Arc<AtomicBool>,
    abort: Arc<AtomicBool>,
    _closed: Sender<()>,
}

impl Window {
    pub(crate) fn open(
        layout: FrameLayout,
        scenes: Arc<SceneLayout>,
        span: Option<FrameRange>,
        mut flags: Vec<Arc<AtomicBool>>,
    ) -> (Self, WindowJob, Sender<Result<(), anyhow::Error>>) {
        let stop = Arc::new(AtomicBool::new(false));
        let abort = Arc::new(AtomicBool::new(false));
        flags.extend([Arc::clone(&stop), Arc::clone(&abort)]);
        let cancel = Cancel::new(flags);
        let total = span.map_or(scenes.total_frames, |span| span.end - span.start);
        let (output_tx, output) = crossbeam_channel::unbounded();
        let (closed_tx, closed) = crossbeam_channel::bounded(0);
        let (done_tx, done) = crossbeam_channel::bounded(1);
        let window = Self {
            layout,
            frame_rate: scenes.framerate,
            total: total as u64,
            next: 0,
            output,
            reorder: BTreeMap::new(),
            done: Some(done),
            failure: None,
            cancel: cancel.clone(),
            stop,
            abort: Arc::clone(&abort),
            _closed: closed_tx,
        };
        let job = WindowJob {
            scenes,
            span,
            output: output_tx,
            closed,
            cancel,
            abort,
        };
        (window, job, done_tx)
    }

    /// Width, height, chroma layout and bit depth of every frame this
    /// window yields.
    pub fn layout(&self) -> FrameLayout {
        self.layout
    }

    pub fn frame_rate(&self) -> Rational32 {
        self.frame_rate
    }

    /// Frames this window yields when it succeeds.
    pub fn frames(&self) -> usize {
        self.total as usize
    }

    pub fn cancel(&self) {
        self.stop.store(true, Ordering::Release);
    }

    /// The next frame in order, or `None` once the window is done.
    ///
    /// An error ends the window; [`Window::finish`] reports it again.
    pub fn recv(&mut self) -> Option<Result<Planes, anyhow::Error>> {
        if self.failure.is_some() || self.next == self.total {
            return None;
        }

        loop {
            // Handing a frame out drops its permit, which is what lets
            // the decoder run further ahead.
            if let Some(msg) = self.reorder.remove(&self.next) {
                self.next += 1;
                return Some(Ok(msg.planes));
            }

            if self.cancel.is_set() {
                let error = if self.abort.load(Ordering::Acquire) {
                    self.pipeline_error()
                } else {
                    anyhow::anyhow!("the window was cancelled")
                };
                return Some(Err(self.fail(error)));
            }

            crossbeam_channel::select! {
                recv(self.output) -> msg => match msg {
                    Ok(msg) => {
                        self.reorder.insert(msg.global_idx, msg);
                    },
                    Err(_) => {
                        let error = self.pipeline_error();
                        return Some(Err(self.fail(error)));
                    },
                },
                default(CANCEL_POLL) => {},
            }
        }
    }

    fn fail(&mut self, error: anyhow::Error) -> anyhow::Error {
        let reported = anyhow::anyhow!("{error:#}");
        self.failure = Some(error);
        reported
    }

    fn wait_pipeline(&mut self) -> Result<(), anyhow::Error> {
        match self.done.take() {
            Some(done) => done
                .recv()
                .map_err(|_| anyhow::anyhow!("the window worker stopped without reporting"))?,
            None => Ok(()),
        }
    }

    /// Why the workers stopped before the last frame.
    fn pipeline_error(&mut self) -> anyhow::Error {
        match self.wait_pipeline() {
            Err(error) => error,
            Ok(()) => anyhow::anyhow!(
                "wrote {} frames but expected {}. Every worker disconnected \
                 before the stream finished, so a frame index was likely lost",
                self.next,
                self.total,
            ),
        }
    }

    /// Discards any frames not yet received, waits for the pipeline and
    /// returns how many frames the window produced.
    pub fn finish(mut self) -> Result<usize, anyhow::Error> {
        while self.recv().is_some() {}

        let pipeline = self.wait_pipeline();

        if let Some(error) = self.failure.take() {
            return Err(error);
        }

        pipeline?;

        Ok(self.next as usize)
    }
}
