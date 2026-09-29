use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use av_denoise_core::{
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    WarmUp,
    kernel_key,
    push_needs_retry,
};

/// Frames a warm-up pushes, enough to fill the temporal window and
/// compile every kernel a real run dispatches.
const WARM_FRAMES: usize = 8;

/// Points CubeCL at the kernel cache.
///
/// This has to run before the first denoiser is built, because the
/// first CubeCL client locks the global config the moment it is built.
pub fn install_kernel_cache() -> Result<Option<PathBuf>, anyhow::Error> {
    match av_denoise_core::install_compilation_cache() {
        Ok(Some(path)) => {
            tracing::info!(?path, "caching compiled kernels");
            Ok(Some(path))
        },
        Ok(None) => {
            tracing::info!(
                "kernel caching is off, every run recompiles. Unset {} to turn it back on.",
                av_denoise_core::COMPILATION_CACHE_ENV,
            );
            Ok(None)
        },
        Err(err) => Err(anyhow::Error::new(err).context("unable to install the kernel cache")),
    }
}

/// The only place a [`PlanarDenoiser`] is built.
///
/// Takes a place in the cross-process warm-up queue first, so concurrent
/// `av-denoise` processes do not each compile into a cold cache. The
/// returned place, if any, is the caller's to finish once this
/// denoiser has produced its first output frame — see [`WarmUp`] for
/// why it cannot be finished any earlier than that.
pub fn create_denoiser(
    opts: &PlaneOptions,
    layout: FrameLayout,
) -> Result<(PlanarDenoiser, Option<WarmUp>), anyhow::Error> {
    let warm_up = WarmUp::begin(kernel_key(opts, layout));
    let denoiser = PlanarDenoiser::create(opts, layout)?;

    Ok((denoiser, warm_up))
}

/// Gives up a cold-cache queue place after a frame has proven the
/// kernels it names are compiled and cached. Does nothing once already
/// finished, or if no place was taken.
pub fn finish_warm_up(warm_up: &mut Option<WarmUp>) {
    if let Some(warm_up) = warm_up.take() {
        warm_up.finish();
    }
}

/// Compiles and caches every kernel a run with `opts` on `layout`
/// frames needs, by denoising a few synthetic frames.
pub fn warm(opts: &PlaneOptions, layout: FrameLayout, cancel: &AtomicBool) -> Result<(), anyhow::Error> {
    let planes = synthetic_frame(layout);
    let (mut denoiser, mut warm_up) = create_denoiser(opts, layout)?;

    for _ in 0..WARM_FRAMES {
        if cancel.load(Ordering::Acquire) {
            anyhow::bail!("NL4D warm-up was cancelled");
        }

        if push_needs_retry(denoiser.push(&planes))? {
            if denoiser.recv()?.is_some() {
                finish_warm_up(&mut warm_up);
            }

            denoiser.push(&planes)?;
        }

        if denoiser.recv()?.is_some() {
            finish_warm_up(&mut warm_up);
        }
    }

    denoiser.flush(|_| finish_warm_up(&mut warm_up))?;

    Ok(())
}

/// Mid-grey noise at the layout's depth, so the noise estimate has
/// something to measure.
fn synthetic_frame(layout: FrameLayout) -> Planes {
    let bits = layout.depth.bits() as u32;
    let mut state = 0x9e37_79b9_u32;
    let mut plane = |bytes: usize| {
        let mid = 1_u32 << (bits - 1);
        let spread = 1_u32 << (bits - 4);
        let mut out = Vec::with_capacity(bytes);
        while out.len() < bytes {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let sample = mid - spread / 2 + state % spread;
            if bits > 8 {
                out.extend_from_slice(&(sample as u16).to_le_bytes());
            } else {
                out.push(sample as u8);
            }
        }
        out
    };

    Planes {
        y: plane(layout.luma_bytes()),
        u: plane(layout.chroma_bytes()),
        v: plane(layout.chroma_bytes()),
    }
}
