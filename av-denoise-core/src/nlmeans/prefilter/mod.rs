mod bilateral;

pub use bilateral::bilateral_radius;
pub(crate) use bilateral::inv_two_sigma_sq;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// How the reference image for each frame is produced.
///
/// NLM compares patches to decide how much two pixels look alike. Doing
/// that on a noisy image means comparing noise as well as content, so a
/// cleaner reference image can give better weights.
///
/// The pixels being averaged always come from the original input. Only
/// the weights change.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum PrefilterMode {
    /// No reference image, so patches are compared on the noisy input.
    /// This costs nothing extra.
    #[default]
    None,
    /// The caller supplies the reference frame through
    /// [`super::NlmDenoiser::push_frame_with_reference`].
    External,
    /// A quick bilateral blur run on the GPU at push time.
    Bilateral { sigma_s: f32, sigma_r: f32 },
    /// A spatial NLM pilot pass.
    ///
    /// Each frame is denoised with the windowed spatial kernel at push
    /// time and the result is kept as the reference image.
    NlmSpatial {
        /// How much of the main pass strength the pilot pass uses.
        strength_scale: f32,
    },
}

/// The measured default strength for the pilot pass, as a multiplier on
/// the main pass strength.
///
/// A calibration sweep across noise levels put the XPSNR plateau for
/// `PrefilterMode::NlmSpatial` at this value.
pub const DEFAULT_PILOT_STRENGTH_SCALE: f32 = 0.4;

impl PrefilterMode {
    /// Whether the denoiser needs to allocate the reference ring buffer.
    pub(crate) fn needs_reference_buf(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether this mode builds its reference on the GPU during
    /// `push_frame`, rather than taking one from the caller.
    pub(crate) fn is_gpu_internal(self) -> bool {
        matches!(self, Self::Bilateral { .. } | Self::NlmSpatial { .. })
    }
}

/// Parses a `--prefilter`-style string into a [`PrefilterMode`].
///
/// `"none"` or an empty string means [`PrefilterMode::None`].
///
/// `"nlm"` or `"nlm:<strength_scale>"` builds
/// [`PrefilterMode::NlmSpatial`]. `strength_scale` multiplies the main
/// pass strength for the pilot pass. Bare `"nlm"` uses
/// [`DEFAULT_PILOT_STRENGTH_SCALE`].
///
/// `"bilateral:<sigma_s>,<sigma_r>"` builds [`PrefilterMode::Bilateral`].
///
/// This never produces [`PrefilterMode::External`], since that mode has
/// no string form: it requires the caller to supply a reference frame
/// through [`super::NlmDenoiser::push_frame_with_reference`].
pub fn parse_prefilter(s: &str) -> Result<PrefilterMode, anyhow::Error> {
    if s == "none" || s.is_empty() {
        return Ok(PrefilterMode::None);
    }

    if s == "nlm" {
        return Ok(PrefilterMode::NlmSpatial {
            strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
        });
    }

    if let Some(rest) = s.strip_prefix("nlm:") {
        let strength_scale: f32 = rest
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("--prefilter nlm expects a number: nlm:<strength_scale>"))?;

        return Ok(PrefilterMode::NlmSpatial { strength_scale });
    }

    if let Some(rest) = s.strip_prefix("bilateral:") {
        let parts: Vec<&str> = rest.split(',').collect();

        if parts.len() != 2 {
            anyhow::bail!("--prefilter bilateral expects two values: bilateral:<sigma_s>,<sigma_r>");
        }

        let sigma_s: f32 = parts[0].trim().parse()?;
        let sigma_r: f32 = parts[1].trim().parse()?;

        return Ok(PrefilterMode::Bilateral { sigma_s, sigma_r });
    }

    anyhow::bail!(
        "unknown prefilter '{s}', expected `none`, `nlm[:<strength_scale>]`, or `bilateral:<sigma_s>,<sigma_r>`"
    )
}

/// The inputs one prefilter dispatch needs.
///
/// This lives only for the length of a single `push_frame`, which is
/// what makes the borrows on the denoiser's buffers sound.
pub(crate) struct PrefilterCtx<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    pub frame_count: u32,
    pub frame: u32,
    pub input_buf: &'a Handle,
    pub reference_buf: &'a Handle,
}

/// Runs the GPU prefilter for the frame that was uploaded last.
///
/// `None` and `External` do nothing here.
pub(crate) fn run_prefilter<R: Runtime>(
    mode: PrefilterMode,
    client: &ComputeClient<R>,
    ctx: &PrefilterCtx<'_>,
) -> Result<(), anyhow::Error> {
    match mode {
        PrefilterMode::None | PrefilterMode::External => Ok(()),
        // The pilot needs the full accumulator context, meaning accum,
        // weight_sum, max_weight, and h2_inv_norm, which `PrefilterCtx`
        // does not carry. `NlmDenoiser::run_nlm_spatial_pilot`
        // dispatches it directly instead of coming through here.
        PrefilterMode::NlmSpatial { .. } => Ok(()),
        PrefilterMode::Bilateral { sigma_s, sigma_r } => {
            bilateral::run_bilateral::<R>(client, ctx, sigma_s, sigma_r)
        },
    }
}
