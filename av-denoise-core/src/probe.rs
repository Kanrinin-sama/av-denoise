//! Opening a backend's client without taking the process down with it.
//!
//! A build can enable a backend whose driver libraries are not
//! installed. Some backends do not report that as an error. The CUDA
//! runtime loads `libcuda` dynamically on its own worker thread and
//! panics there when the load fails, and the panic reaches the caller
//! as a second panic when cubecl unwraps the dead worker's channel.
//!
//! [`open_client`] runs that work under [`catch_unwind`], so a missing
//! driver reads as "this backend is not available here" rather than as
//! a crash. That is what makes a single binary with `cuda`, `rocm`, and
//! `vulkan` all enabled usable on a machine that has only one of them.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;

use crate::accelerate::Accelerator;

/// Backends already reported as unavailable, and the lock guarding the
/// panic hook.
///
/// The hook is process-wide, so two probes running at once would race to
/// restore each other's. Holding this for the length of a probe keeps
/// them in single file, and the list inside it keeps a backend from
/// warning again every time it is probed.
static PROBED: Mutex<Vec<Accelerator>> = Mutex::new(Vec::new());

pub(crate) fn open_client<R: Runtime>(
    accelerator: Accelerator,
    device: &R::Device,
) -> Option<ComputeClient<R>> {
    let mut probed = PROBED.lock().unwrap_or_else(|err| err.into_inner());

    let opened = quiet_panics(|| {
        let client = R::client(device);
        cubecl::future::block_on(client.sync()).map(|()| client)
    });

    match opened {
        Ok(Ok(client)) => Some(client),
        Ok(Err(err)) => {
            tracing::debug!(err = ?err, "could not use the {accelerator} runtime");
            None
        },
        Err(_) => {
            // Only the first probe of a backend says anything. A denoise
            // run probes once per denoiser it builds, and a missing
            // driver is worth one line, not one per scene.
            if !probed.contains(&accelerator) {
                probed.push(accelerator);
                tracing::warn!(
                    "the {accelerator} backend is enabled but did not start, its driver libraries are probably missing"
                );
            }
            None
        },
    }
}

/// Runs `f`, turning a panic into an `Err` and routing the panic message
/// to the debug log rather than to stderr.
///
/// A failing backend prints its own panic from its worker thread before
/// the caller ever sees one, so the hook is quietened for as long as `f`
/// runs and put back afterwards.
///
/// The hook is process-wide. Callers hold [`PROBED`] across this so two
/// probes cannot race to restore each other's.
fn quiet_panics<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|info| tracing::debug!("{info}")));
    let out = panic::catch_unwind(AssertUnwindSafe(f));
    panic::set_hook(previous);
    out
}
