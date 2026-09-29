use std::time::Duration;

use av_denoise_core::DenoisingMode;
use crossbeam_channel::{Receiver, Sender};

use crate::cancel::Cancel;

/// How often a dispatcher waiting on a permit looks at its cancel flags.
pub(crate) const CANCEL_POLL: Duration = Duration::from_millis(100);

pub(crate) fn temporal_radius(mode: DenoisingMode) -> u32 {
    match mode {
        DenoisingMode::Temporal { radius } => radius,
        DenoisingMode::Spacial => 0,
    }
}

/// Frames in flight one window needs to make progress.
fn window_floor(workers: usize, radius: u32) -> usize {
    // A worker emits nothing until `push` first returns QueueFull, which
    // takes `radius + MAX_PENDING + 1` pushes. Fewer permits than that
    // and its scene can never return one, so the dispatcher waits on a
    // permit the worker cannot release.
    workers * (radius as usize + av_denoise_core::MAX_PENDING + 2)
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

/// The frames in flight every window of one service shares, and the most
/// of them any one window may hold.
///
/// Every staged frame carries a [`Permit`] until it is handed to the
/// consumer or dropped on the way, so an aborted window gives back
/// everything it held. The cap leaves every other window its floor, so
/// a paused or stalled window cannot starve the rest.
pub(crate) struct FramePermits {
    shared: Pool,
    count: usize,
    cap: usize,
}

impl FramePermits {
    /// Frames in flight the whole budget allows across `windows` windows,
    /// refusing a budget that cannot give every window the floor.
    ///
    /// A budget the floor has to raise serialises the pipeline, so it fails
    /// here rather than running on with too few frames in flight.
    pub(crate) fn checked(
        budget_bytes: u64,
        frame_bytes: usize,
        workers: usize,
        radius: u32,
        windows: usize,
    ) -> Result<Self, anyhow::Error> {
        let afforded = frames_afforded(budget_bytes, frame_bytes);
        let floor = window_floor(workers, radius);
        let needed = floor * windows;

        if needed > afforded {
            let suggestion = suggested_budget(needed as u64 * frame_bytes as u64);

            anyhow::bail!(
                "--frame-budget {budget} affords {afforded} frames at {frame_bytes} bytes per frame                  over {windows} windows, but {workers} workers at temporal radius {radius} need at                  least {floor} per window. Pass at least --frame-budget {suggestion}.",
                budget = size_string(budget_bytes),
            );
        }

        Ok(Self {
            shared: Pool::filled(afforded),
            count: afforded,
            cap: afforded - floor * (windows - 1),
        })
    }

    pub(crate) fn count(&self) -> usize {
        self.count
    }

    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    /// The permits one window stages through, capped at [`Self::cap`].
    pub(crate) fn window(&self) -> WindowPermits<'_> {
        WindowPermits {
            own: Pool::filled(self.cap),
            shared: &self.shared,
        }
    }
}

/// One window's view of the shared permits.
pub(crate) struct WindowPermits<'a> {
    own: Pool,
    shared: &'a Pool,
}

impl WindowPermits<'_> {
    /// Waits for a permit under both the window's cap and the shared
    /// budget, failing once the window is closed or cancelled.
    pub(crate) fn acquire(&self, closed: &Receiver<()>, cancel: &Cancel) -> Result<Permit, anyhow::Error> {
        let own = self.own.acquire(closed, cancel)?;
        let shared = self.shared.acquire(closed, cancel)?;

        Ok(Permit { _own: own, _shared: shared })
    }
}

/// A counting semaphore over frames in flight.
struct Pool {
    give: Sender<()>,
    take: Receiver<()>,
}

impl Pool {
    fn filled(count: usize) -> Self {
        let (give, take) = crossbeam_channel::bounded::<()>(count);
        for _ in 0..count {
            give.send(()).expect("the channel holds exactly `count` permits");
        }

        Self { give, take }
    }

    fn acquire(&self, closed: &Receiver<()>, cancel: &Cancel) -> Result<Token, anyhow::Error> {
        loop {
            if cancel.is_set() {
                anyhow::bail!("the window was cancelled");
            }
            crossbeam_channel::select! {
                recv(self.take) -> permit => {
                    permit.map_err(|_| anyhow::anyhow!("the coordinator stopped before the stream finished"))?;
                    return Ok(Token(self.give.clone()));
                },
                recv(closed) -> _ => anyhow::bail!("the window was closed before the stream finished"),
                default(CANCEL_POLL) => {},
            }
        }
    }
}

/// One frame's claim on the budget, given back when dropped.
pub(crate) struct Permit {
    _own: Token,
    _shared: Token,
}

/// One frame's claim on a single pool, given back when dropped.
struct Token(Sender<()>);

impl Drop for Token {
    fn drop(&mut self) {
        // Never blocks, because permits held plus permits waiting is
        // always the channel's capacity.
        let _ = self.0.send(());
    }
}
