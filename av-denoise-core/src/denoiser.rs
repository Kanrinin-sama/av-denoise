use std::collections::VecDeque;

use cubecl::Runtime;
use cubecl::prelude::ComputeClient;
use cubecl::stream_id::StreamId;

use crate::accelerate::Accelerator;
use crate::device::Device;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::{
    ChannelMode,
    Depth,
    HqParams,
    MotionCompensationMode,
    MotionSearch,
    NlmDenoiser,
    NlmParams,
    Pending,
    PrefilterMode,
    TryWait,
    hq_default_strength,
    validate_dimensions,
};
use crate::sniff::sniff_best_accelerator;

/// How a [`Denoiser`] should be set up.
///
/// Build one with `DenoiserOptions::builder()`. Every field has a
/// default, so only the parts you care about need naming.
///
/// Only the settings every algorithm reads live here. Everything else
/// belongs to whichever [`Algorithm`] variant actually uses it.
#[derive(Debug, Clone, bon::Builder)]
pub struct DenoiserOptions {
    /// Which channels of the frame to denoise.
    #[builder(default = ChannelMode::Yuv)]
    pub channel_mode: ChannelMode,
    /// Whether to clean each frame on its own or across a temporal
    /// window.
    #[builder(default = DenoisingMode::Spacial)]
    pub mode: DenoisingMode,
    /// Which algorithm to run, along with the settings only that
    /// algorithm reads.
    #[builder(default)]
    pub algorithm: Algorithm,
    /// What format denoised frames come back in.
    #[builder(default = OutputFormat::F32)]
    pub output_format: OutputFormat,
}

/// What a denoiser hands back from [`Denoiser::recv_frame`],
/// [`Denoiser::try_recv_frame`] and [`Denoiser::flush`].
///
/// The GPU only ever holds normalised `f32`. [`Depth`] is a wire concept,
/// so a denoiser that returns wire bytes is told its depth when it is
/// built rather than at each call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Normalised `f32`, one value per channel per pixel.
    F32,
    /// Wire bytes at this depth, quantised on the GPU.
    Wire { depth: Depth },
}

/// One denoised frame, in whichever format its denoiser was built for.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameOutput {
    /// Normalised `f32`, `width * height * channels` values.
    F32(Vec<f32>),
    /// Wire bytes, interleaved except for a chroma pair, which is laid
    /// out as U's whole region followed by V's.
    Wire(Vec<u8>),
}

impl FrameOutput {
    /// The `f32` frame, or `None` if this came from a wire-mode denoiser.
    pub fn into_f32(self) -> Option<Vec<f32>> {
        match self {
            Self::F32(v) => Some(v),
            Self::Wire(_) => None,
        }
    }

    /// The wire bytes, or `None` if this came from an `f32`-mode denoiser.
    pub fn into_wire(self) -> Option<Vec<u8>> {
        match self {
            Self::Wire(v) => Some(v),
            Self::F32(_) => None,
        }
    }

    /// Borrows the `f32` frame, or `None` if this came from a wire-mode
    /// denoiser.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(v),
            Self::Wire(_) => None,
        }
    }

    /// Borrows the wire bytes, or `None` if this came from an `f32`-mode
    /// denoiser.
    pub fn as_wire(&self) -> Option<&[u8]> {
        match self {
            Self::Wire(v) => Some(v),
            Self::F32(_) => None,
        }
    }
}

/// Which denoising algorithm to run.
///
/// Each variant carries its own settings, so a knob one algorithm has no
/// use for cannot be set on it.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Algorithm {
    /// The fast NLMeans path, with fixed weighting and no noise
    /// measurement.
    Nlmeans(NlmeansOptions),
    /// NLMeans with its weighting matched to the measured noise level.
    ///
    /// This also uses a different default `strength`, one that adapts to
    /// the temporal radius and the plane being denoised. See
    /// [`crate::nlmeans::hq_default_strength`].
    NlmeansHq(NlmeansHqOptions),
    /// Groups 8x8 patches across the motion-compensated temporal window
    /// itself, rather than filtering with NLM first and grouping within
    /// one frame afterward.
    ///
    /// No NLM weighting pass ever runs, so none of the NLM knobs appear
    /// on [`Nl4dOptions`].
    Nl4d(Nl4dOptions),
}

impl Default for Algorithm {
    fn default() -> Self {
        Self::Nlmeans(NlmeansOptions::default())
    }
}

/// Settings for [`Algorithm::Nlmeans`].
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct NlmeansOptions {
    /// Which reference image the NLM weights are computed against.
    ///
    /// `None`, the default, compares patches on the noisy input
    /// directly. Every other mode costs one extra GPU pass per frame.
    pub prefilter: PrefilterMode,
    /// Whether temporal denoising follows motion between frames.
    ///
    /// `None`, the default, turns motion compensation off. `Mvtools`
    /// warps temporal neighbours into line with the centre frame before
    /// the NLM weighting runs.
    ///
    /// Only has an effect when [`DenoiserOptions::mode`] is
    /// `Temporal { .. }`.
    pub motion_compensation: MotionCompensationMode,
    /// Overrides for the NLM search radius, patch radius, strength, and
    /// self-weight.
    pub tuning: NlmTuning,
}

/// Settings for [`Algorithm::NlmeansHq`].
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct NlmeansHqOptions {
    /// Everything the fast path takes, which HQ takes too.
    pub nlm: NlmeansOptions,
    /// The noise measurement and confidence weighting HQ adds on top.
    pub hq: HqParams,
}

/// Settings for [`Algorithm::Nl4d`].
///
/// nl4d runs the HQ front end only for its machinery, the frame ring,
/// the motion field, and the noise estimate. Nothing weights or averages
/// patches the NLM way, so the NLM knobs are absent here and the fields
/// below are the whole surface.
///
/// The temporal radius comes from [`DenoiserOptions::mode`], which has to
/// be `Temporal { .. }`. Motion tracking is always on, because the
/// grouping kernel reads the motion field and confidence scores it
/// produces.
///
/// `lambda_ht` has a per-plane default. `None` resolves through
/// [`nl4d_default_lambda_ht`] once the plane being denoised is known.
/// `lambda_ht_scale` then multiplies whichever value that resolves to.
///
/// Every other default comes from [`Nl4dParams::default`].
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Nl4dOptions {
    /// How motion between frames is tracked.
    pub motion: MotionSearch,
    /// A fixed noise standard deviation in `[0, 1]` units, replacing the
    /// automatic per-frame estimate.
    ///
    /// `None`, the default, measures the noise in each pushed frame and
    /// smooths it over time.
    pub sigma: Option<f32>,
    /// A multiplier applied to the measured noise level before anything
    /// reads it. Defaults to 1.0.
    ///
    /// This does nothing when `sigma` pins the noise level, because the
    /// estimator never runs in that case.
    pub sigma_scale: f32,
    /// A multiplier on the per-block mismatch threshold, which sets how
    /// much extra SAD a block tolerates before its confidence starts to
    /// fall. Defaults to 1.0.
    ///
    /// Higher values tolerate larger mismatches.
    pub thsad_scale: f32,
    /// Half-width of the refine window searched around each neighbour
    /// frame's motion-predicted position, in `1..=4`. Defaults to 2.
    pub refine: u32,
    /// Half-width of the spatial candidate window searched in the centre
    /// frame, in `1..=16`. Defaults to 9.
    pub spatial_radius: u32,
    /// Hard-threshold multiplier on the propagated coefficient sigma.
    /// Higher removes more noise and more fine detail.
    ///
    /// `None` resolves through [`nl4d_default_lambda_ht`], which returns
    /// a different value for luma than for chroma.
    pub lambda_ht: Option<f32>,
    /// A multiplier applied to the resolved `lambda_ht`. Defaults to
    /// 1.0.
    ///
    /// It scales an explicit `lambda_ht` and the calibrated per-plane
    /// default alike, so one value moves both planes together. Has to
    /// be finite and in `[0.1, 10.0]`.
    pub lambda_ht_scale: f32,
    /// The confidence floor below which a whole neighbour block is
    /// skipped rather than scored, in `[0, 1)`. Defaults to 0.05. Only
    /// affects how much compute a submit spends, never which candidates
    /// are admitted once they are scored.
    pub c_min: f32,
    /// A multiplier on the mismatch variance a poorly matched temporal
    /// member carries into the hard threshold. Defaults to 1.0.
    ///
    /// The variance grows with the square of this. The mechanism
    /// saturates well before the top of its accepted range, see
    /// [`crate::nl4d::Nl4dParams::mismatch_scale`].
    pub mismatch_scale: f32,
    /// Whether a temporal member's mismatch variance reaches the
    /// hard-threshold shrinkage at all. Defaults to `true`. See
    /// [`crate::nl4d::Nl4dParams::confidence_variance`].
    pub confidence_variance: bool,
    /// Estimates noise fresh from each frame's own window instead of
    /// smoothing it across the whole stream's history. Defaults to
    /// `false`, matching every calibrated preset.
    ///
    /// `av-denoise-vs` turns this on unconditionally, because a
    /// VapourSynth filter has to return the same pixels for a frame no
    /// matter what order frames were requested in, and history-dependent
    /// estimation breaks that guarantee under random access. See
    /// [`HqParams::windowed_noise_estimation`].
    pub windowed_noise_estimation: bool,
}

impl Default for Nl4dOptions {
    fn default() -> Self {
        let defaults = Nl4dParams::default();
        let hq = HqParams::default();
        Self {
            motion: MotionSearch::default(),
            sigma: hq.sigma_override,
            sigma_scale: hq.sigma_scale,
            thsad_scale: hq.thsad_scale,
            refine: defaults.refine,
            spatial_radius: defaults.spatial_radius,
            // Resolved per plane by `nl4d_default_lambda_ht` at
            // construction time, once the plane being denoised is
            // known.
            lambda_ht: None,
            lambda_ht_scale: 1.0,
            c_min: defaults.c_min,
            mismatch_scale: defaults.mismatch_scale,
            confidence_variance: defaults.confidence_variance,
            windowed_noise_estimation: false,
        }
    }
}

impl Nl4dOptions {
    /// The front end's HQ parameters for this configuration.
    ///
    /// `temporal_confidence` is always on, because the grouping kernel
    /// reads the confidence scores it produces. The two strength-related
    /// switches keep their defaults, since nl4d never runs a weighting
    /// pass for them to affect.
    fn to_hq_params(self) -> HqParams {
        HqParams {
            sigma_override: self.sigma,
            sigma_scale: self.sigma_scale,
            thsad_scale: self.thsad_scale,
            temporal_confidence: true,
            windowed_noise_estimation: self.windowed_noise_estimation,
            ..HqParams::default()
        }
    }
}

/// The default `lambda_ht` for nl4d's hard-threshold stage, per plane.
///
/// `lambda_ht` is how many standard deviations of estimated noise a
/// transform coefficient has to clear to survive. Raising it removes more
/// noise and more fine detail with it, so the value is a trade rather
/// than an optimum.
///
/// Luma gets 5.3, picked by eye from rendered comparisons on real grain
/// and deliberately biased toward keeping detail. Higher values remove
/// visibly more noise, but not enough to be worth what they cost in
/// texture.
///
/// `ChannelMode::Yuv` reads the luma value, on the same "a fused pass is
/// dominated by luma" assumption [`hq_default_strength`]
/// makes for its own Yuv case.
///
/// Chroma gets 4.2, picked the same way from the chroma residuals with
/// luma pinned at 5.3.
pub fn nl4d_default_lambda_ht(channels: ChannelMode) -> f32 {
    match channels {
        ChannelMode::Luma | ChannelMode::Yuv => 5.3,
        ChannelMode::Chroma => 4.2,
    }
}

/// Resolves `Nl4dOptions.lambda_ht` for one plane, falling back to
/// [`nl4d_default_lambda_ht`] when the caller left it unset, then
/// applies `lambda_ht_scale`.
///
/// The scale multiplies an explicit value and the calibrated default
/// alike, so it moves both planes together whether or not one of them
/// is pinned.
///
/// The range check lives here rather than in [`Nl4dParams`],
/// which only ever sees the product. A scale of 0 would surface there as
/// a complaint about `lambda_ht`, naming a knob the caller never set.
fn resolve_lambda_ht(opts: &Nl4dOptions, channels: ChannelMode) -> Result<f32, String> {
    if !(opts.lambda_ht_scale.is_finite() && (0.1..=10.0).contains(&opts.lambda_ht_scale)) {
        return Err(format!(
            "lambda_ht_scale must be finite and in [0.1, 10.0], got {}",
            opts.lambda_ht_scale
        ));
    }

    let lambda_ht = opts.lambda_ht.unwrap_or_else(|| nl4d_default_lambda_ht(channels));

    Ok(lambda_ht * opts.lambda_ht_scale)
}

/// Speed vs quality dial.
///
/// Each denoising family reads the same dial and fills in its own knobs
/// from it. For `nlmeans` that is [`nlmeans_variant_for`],
/// [`nlmeans_temporal_radius_for`], and [`nlmeans_search_radius_for`].
/// For `nl4d` it is [`nl4d_temporal_radius_for`] and
/// [`nl4d_spatial_radius_for`].
///
/// Both front ends parse the same names from this one type, so a preset
/// resolves to the same dials everywhere it is used.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, strum_macros::EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Preset {
    /// Fastest and lowest quality.
    Veryfast,
    /// One step up from `veryfast`.
    Fast,
    /// The default, favouring quality over speed.
    #[default]
    Base,
    /// One step down from `veryslow`.
    Slow,
    /// Slowest and highest quality.
    Veryslow,
}

/// Which nlmeans implementation a preset, or an explicit choice, selects.
#[derive(Debug, Copy, Clone, PartialEq, Eq, strum_macros::EnumString)]
#[strum(ascii_case_insensitive)]
pub enum NlmeansVariant {
    /// The fast path. Fixed weighting, no noise measurement.
    Fast,
    /// Quality focused. Calibrates its weighting to the noise level,
    /// measured automatically per frame.
    Hq,
}

/// Which [`NlmeansVariant`] a preset runs.
pub fn nlmeans_variant_for(preset: Preset) -> NlmeansVariant {
    match preset {
        Preset::Veryfast => NlmeansVariant::Fast,
        Preset::Fast | Preset::Base | Preset::Slow | Preset::Veryslow => NlmeansVariant::Hq,
    }
}

/// How many neighbouring frames on each side `nlmeans` looks at, at a
/// preset.
pub fn nlmeans_temporal_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast => 0,
        Preset::Fast => 1,
        Preset::Base => 2,
        Preset::Slow => 4,
        Preset::Veryslow => 8,
    }
}

/// How far `nlmeans` looks for similar patches inside a frame, at a
/// preset.
pub fn nlmeans_search_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast | Preset::Fast | Preset::Base => 2,
        Preset::Slow | Preset::Veryslow => 4,
    }
}

/// How far the temporal window reaches at each preset, for `nl4d`.
///
/// Unlike `nlmeans`, `veryfast` keeps a 1-frame window rather than
/// dropping to 0, because nl4d has nothing to do without neighbouring
/// frames to group against.
pub fn nl4d_temporal_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast | Preset::Fast => 1,
        Preset::Base => 2,
        Preset::Slow => 4,
        Preset::Veryslow => 8,
    }
}

/// How wide the centre frame's candidate search is at each preset, for
/// `nl4d`.
///
/// `veryfast` shares its temporal radius with `fast`, so this is what
/// separates them. The window covers `(2 * radius + 1)^2` positions, so
/// 6 searches a little over half the candidates 9 does.
///
/// Every preset from `fast` up uses the library default. Widening it
/// further at the slow end costs quadratically and has not been measured
/// to be worth it.
pub fn nl4d_spatial_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast => 6,
        Preset::Fast | Preset::Base | Preset::Slow | Preset::Veryslow => {
            Nl4dOptions::default().spatial_radius
        },
    }
}

/// Whether a frame is cleaned on its own or alongside its neighbours.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DenoisingMode {
    /// Cleans each frame using only its own pixels.
    Spacial,
    /// Cleans each frame using a window of `2 * radius + 1` frames.
    Temporal { radius: u32 },
}

/// NLM tuning knobs.
///
/// Every field is optional. Whatever is left unset falls back to the
/// library default.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct NlmTuning {
    pub search_radius: Option<u32>,
    pub patch_radius: Option<u32>,
    pub strength: Option<f32>,
    pub self_weight: Option<f32>,
}

impl DenoiserOptions {
    /// Turns this option set into the low-level [`NlmParams`] a backend
    /// denoiser is built from.
    ///
    /// For nl4d this describes the front end only, since nl4d's own
    /// grouping stage is configured from [`Nl4dOptions`] separately in
    /// [`build_engine`].
    ///
    /// Whichever default `strength` applies is folded in here. For the
    /// HQ algorithm that comes from
    /// [`crate::nlmeans::hq_default_strength`].
    ///
    /// This is public so callers building per-plane options, and tests,
    /// can read the resolved values without building a real `Denoiser`.
    #[doc(hidden)]
    pub fn to_nlm_params(&self) -> NlmParams {
        let temporal_radius = match self.mode {
            DenoisingMode::Spacial => 0,
            DenoisingMode::Temporal { radius } => radius,
        };

        match self.algorithm {
            Algorithm::Nlmeans(opts) => self.nlm_params_for(opts, None, temporal_radius),
            Algorithm::NlmeansHq(opts) => self.nlm_params_for(opts.nlm, Some(opts.hq), temporal_radius),
            // nl4d never runs a weighting pass, so `strength`,
            // `search_radius`, `patch_radius`, and `self_weight` stay at
            // their library defaults and no prefilter is built.
            Algorithm::Nl4d(opts) => NlmParams {
                channels: self.channel_mode,
                motion_compensation: opts.motion.into(),
                temporal_radius,
                hq: Some(opts.to_hq_params()),
                ..NlmParams::default()
            },
        }
    }

    /// [`Self::to_nlm_params`] for whichever of the two NLM algorithms
    /// is running, with `hq` set only for the quality one.
    fn nlm_params_for(&self, opts: NlmeansOptions, hq: Option<HqParams>, temporal_radius: u32) -> NlmParams {
        // An explicit `strength` always wins, whether it came straight
        // from `NlmTuning` or from a per-plane override the caller
        // already folded in.
        //
        // Otherwise the default depends on `auto_strength`. With it on,
        // HQ reads `strength` as a multiplier on the measured noise
        // level, so it needs its own calibrated default rather than the
        // fast path's absolute FFmpeg-style one. That calibrated default
        // also varies with the temporal radius and with the plane
        // `channel_mode` names, because each per-plane `Denoiser`
        // carries its own channel mode.
        //
        // With auto-strength off, HQ reads `strength` as an absolute
        // value just like the fast path, so it falls back to the same
        // absolute default.
        let strength = opts.tuning.strength.unwrap_or(match hq {
            Some(hq) if hq.auto_strength => hq_default_strength(self.channel_mode, temporal_radius),
            _ => NlmParams::default().strength,
        });

        let defaults = NlmParams::default();
        NlmParams {
            channels: self.channel_mode,
            prefilter: opts.prefilter,
            motion_compensation: opts.motion_compensation,
            temporal_radius,
            hq,
            strength,
            search_radius: opts.tuning.search_radius.unwrap_or(defaults.search_radius),
            patch_radius: opts.tuning.patch_radius.unwrap_or(defaults.patch_radius),
            self_weight: opts.tuning.self_weight.unwrap_or(defaults.self_weight),
        }
    }
}

/// Errors reported by the high-level [`Denoiser`].
#[derive(Debug, thiserror::Error)]
pub enum DenoiserError {
    /// An earlier denoised frame has not been collected yet, so pushing
    /// again would overwrite it in the double-buffered output slot.
    ///
    /// Call [`Denoiser::recv_frame`] or [`Denoiser::try_recv_frame`],
    /// then retry the same `push_frame` call.
    #[error("denoiser queue is full, collect the pending frame before pushing more")]
    QueueFull,
    /// An earlier call failed, so how many frames are in flight is no
    /// longer known and later output would not line up with its input.
    ///
    /// Call [`Denoiser::reset_stream`] to start a fresh stream, or drop the denoiser.
    #[error("denoiser failed earlier, reset the stream before using it again")]
    Poisoned,
    /// None of the accelerators in the priority list could be started.
    #[error("no accelerator from the priority list is available")]
    NoAcceleratorAvailable,
    /// Anything else, wrapping the internal `anyhow` errors raised by
    /// kernel dispatch and readback.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Either denoiser a `Backend` runtime arm can hold.
///
/// This keeps `Backend`'s own match arms at one line each. Without it,
/// adding a second denoiser type would multiply the runtime arms instead
/// of fanning out once here.
enum Engine<R: Runtime> {
    Nlm(Box<NlmDenoiser<R>>),
    Nl4d(Box<Nl4dDenoiser<R>>),
}

impl<R: Runtime> Engine<R> {
    fn is_nl4d(&self) -> bool {
        matches!(self, Self::Nl4d(_))
    }

    fn push_frame(&mut self, frame: &[f32]) {
        match self {
            Self::Nlm(d) => d.push_frame(frame),
            Self::Nl4d(d) => d.push_frame(frame),
        }
    }

    fn push_frame_wire(&mut self, planes: &[&[u8]], depth: Depth) {
        match self {
            Self::Nlm(d) => d.push_frame_wire(planes, depth),
            Self::Nl4d(d) => d.push_frame_wire(planes, depth),
        }
    }

    fn denoise_submit(&mut self) -> Result<Option<Pending<R>>, anyhow::Error> {
        match self {
            Self::Nlm(d) => d.denoise_submit(),
            // `Nl4dDenoiser::denoise_submit` already returns
            // `DenoiserError` rather than `anyhow::Error`, so this leans
            // on `DenoiserError`'s own `anyhow::Error` conversion instead
            // of re-wrapping it.
            Self::Nl4d(d) => d.denoise_submit().map_err(anyhow::Error::from),
        }
    }

    fn flush(&mut self, sink: impl FnMut(&FrameOutput)) -> Result<(), anyhow::Error> {
        match self {
            Self::Nlm(d) => d.flush(sink),
            Self::Nl4d(d) => d.flush(sink).map_err(anyhow::Error::from),
        }
    }

    fn reset_stream(&mut self) {
        match self {
            Self::Nlm(d) => d.reset_stream_state(),
            Self::Nl4d(d) => d.reset_stream(),
        }
    }
}

/// Builds whichever [`Engine`] `algorithm` calls for.
///
/// `Algorithm::Nl4d` carries its own grouping tuning, which is not part
/// of `NlmParams`, so it is read from `algorithm` directly rather than
/// from `params`. This is also where an unset `lambda_ht` picks up its
/// calibrated per-plane default (`resolve_lambda_ht`), the same way
/// `to_nlm_params` resolves HQ's calibrated `strength`, since this is
/// the first point construction has both `opts` and `params.channels`
/// together.
fn build_engine<R: Runtime>(
    client: &ComputeClient<R>,
    algorithm: &Algorithm,
    params: NlmParams,
    width: u32,
    height: u32,
    output_format: OutputFormat,
) -> Result<Engine<R>, DenoiserError> {
    match algorithm {
        Algorithm::Nl4d(opts) => {
            // nl4d groups patches across neighbouring frames, so there
            // is nothing for it to do without a temporal window.
            if params.temporal_radius == 0 {
                return Err(DenoiserError::Other(anyhow::anyhow!(
                    "nl4d needs a temporal window, set DenoiserOptions::mode to \
                     DenoisingMode::Temporal"
                )));
            }

            let lambda_ht = resolve_lambda_ht(opts, params.channels)
                .map_err(|e| DenoiserError::Other(anyhow::anyhow!(e)))?;
            let nl4d_params = Nl4dParams {
                temporal_radius: params.temporal_radius,
                nlm: params,
                refine: opts.refine,
                spatial_radius: opts.spatial_radius,
                lambda_ht,
                c_min: opts.c_min,
                mismatch_scale: opts.mismatch_scale,
                confidence_variance: opts.confidence_variance,
            };
            let denoiser =
                Nl4dDenoiser::with_output_format(client, nl4d_params, width, height, output_format)
                    .map_err(|e| DenoiserError::Other(anyhow::anyhow!(e)))?;
            Ok(Engine::Nl4d(Box::new(denoiser)))
        },
        Algorithm::Nlmeans(_) | Algorithm::NlmeansHq(_) => Ok(Engine::Nlm(Box::new(
            NlmDenoiser::with_output_format(client, params, width, height, output_format),
        ))),
    }
}

enum Backend {
    #[cfg(feature = "cuda")]
    Cuda(Engine<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(Engine<cubecl::hip::HipRuntime>),
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    Wgpu(Engine<cubecl::wgpu::WgpuRuntime>),
}

impl Backend {
    fn is_nl4d(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(e) => e.is_nl4d(),
            #[cfg(feature = "rocm")]
            Self::Rocm(e) => e.is_nl4d(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Self::Wgpu(e) => e.is_nl4d(),
        }
    }

}

enum BackendPending {
    #[cfg(feature = "cuda")]
    Cuda(Pending<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "rocm")]
    Rocm(Pending<cubecl::hip::HipRuntime>),
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    Wgpu(Pending<cubecl::wgpu::WgpuRuntime>),
}

impl BackendPending {
    fn wait(self) -> Result<FrameOutput, anyhow::Error> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(p) => p.wait(),
            #[cfg(feature = "rocm")]
            Self::Rocm(p) => p.wait(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Self::Wgpu(p) => p.wait(),
        }
    }

    /// Polls the readback once. `Ok(Ok(frame))` is a landed frame,
    /// `Ok(Err(self))` is a readback still in flight.
    fn try_wait(self) -> Result<Result<FrameOutput, Self>, anyhow::Error> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(p) => match p.try_wait()? {
                TryWait::Ready(frame) => Ok(Ok(frame)),
                TryWait::NotReady(p) => Ok(Err(Self::Cuda(p))),
            },
            #[cfg(feature = "rocm")]
            Self::Rocm(p) => match p.try_wait()? {
                TryWait::Ready(frame) => Ok(Ok(frame)),
                TryWait::NotReady(p) => Ok(Err(Self::Rocm(p))),
            },
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Self::Wgpu(p) => match p.try_wait()? {
                TryWait::Ready(frame) => Ok(Ok(frame)),
                TryWait::NotReady(p) => Ok(Err(Self::Wgpu(p))),
            },
        }
    }
}

/// How many readbacks the high-level [`Denoiser`] keeps in flight at
/// once.
///
/// This has to match the backend's output-handle count, which is two.
/// Going past it would reuse the oldest pending frame's output handle
/// and quietly corrupt the results.
pub const MAX_PENDING: usize = 2;

/// How many source frames a windowed operation needs behind and ahead
/// of its target frame, target frame itself not counted in either
/// number.
///
/// `reseed` needs exactly `behind + 1 + ahead` frames, oldest first,
/// with the target frame sitting at index `behind`. This is what tells
/// a caller like `reseed` how wide a window to build, and it varies by
/// algorithm because nl4d's own cross-frame accumulator needs more
/// forward context than the NLM algorithms do. See
/// [`Denoiser::window_span`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSpan {
    /// How many frames older than the target the window must include.
    pub behind: usize,
    /// How many frames newer than the target the window must include.
    pub ahead: usize,
}

impl WindowSpan {
    /// The full window size this span describes, target frame
    /// included: `behind + 1 + ahead`.
    pub fn frame_count(&self) -> usize {
        self.behind + 1 + self.ahead
    }
}

/// A stateful denoiser that cleans a stream of frames.
///
/// Push frames in order with [`push_frame`](Self::push_frame) and
/// collect the cleaned ones with [`recv_frame`](Self::recv_frame) or
/// [`try_recv_frame`](Self::try_recv_frame).
///
/// At the end of the stream call [`flush`](Self::flush) to drain
/// whatever temporal context is left.
///
/// Input frames are `f32` values in `[0, 1]`, laid out as
/// `width * height * channels`. Output comes back as a [`FrameOutput`]
/// in whichever [`OutputFormat`] the options named.
///
/// ```no_run
/// use av_denoise_core::accelerate::Accelerator;
/// use av_denoise_core::{ChannelMode, Denoiser, DenoiserOptions, DenoisingMode, Device};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let options = DenoiserOptions::builder()
///     .channel_mode(ChannelMode::Luma)
///     .mode(DenoisingMode::Temporal { radius: 2 })
///     .build();
///
/// let mut denoiser = Denoiser::create(
///     &[Accelerator::Vulkan],
///     &Device::Default,
///     1920,
///     1080,
///     options,
/// )?;
///
/// let frames: Vec<Vec<f32>> = read_my_frames();
/// let mut cleaned: Vec<Vec<f32>> = Vec::new();
///
/// for frame in &frames {
///     denoiser.push_frame(frame)?;
///
///     // Temporal denoising runs a few frames behind the input, so
///     // there is not always one ready to collect.
///     if let Some(out) = denoiser.recv_frame()? {
///         cleaned.push(out.into_f32().expect("built for f32 output"));
///     }
/// }
///
/// // Drain the frames still inside the temporal window.
/// denoiser.flush(|out| cleaned.push(out.into_f32().expect("built for f32 output")))?;
/// # Ok(())
/// # }
/// # fn read_my_frames() -> Vec<Vec<f32>> { Vec::new() }
/// ```
pub struct Denoiser {
    backend: Backend,
    pending: VecDeque<BackendPending>,
    accelerator: Accelerator,
    width: u32,
    height: u32,
    temporal_radius: u32,
    output_format: OutputFormat,
    frames_pushed: u32,
    /// Set once any call other than a `QueueFull` push has failed, so how
    /// many frames are in flight is no longer known.
    ///
    /// Every entry point refuses to run while this is set.
    /// [`Self::reset_stream`] clears it.
    poisoned: bool,
}

impl Denoiser {
    /// Tries each accelerator in `accelerators` in order and builds a
    /// denoiser on the first one that works.
    ///
    /// `device` picks a non-default device on the chosen runtime.
    ///
    /// # Thread stack size
    ///
    /// cubecl spawns its own per-device worker thread, named
    /// `DS{U,D}-…`, and runs GPU kernel codegen on it. That thread gets
    /// Rust's default stack, which is `RUST_MIN_STACK` or 2 MiB when
    /// that is unset.
    ///
    /// The windowed NLM kernels unroll their body
    /// `(2 * search_radius + 1)^2` times, so a `search_radius` of about
    /// 5 or more can overflow the 2 MiB default and abort the process.
    ///
    /// Callers using a `search_radius` above 4 should call
    /// [`crate::raise_codegen_stack_limit`] before any cubecl thread
    /// spawns, usually right at the top of `main`.
    pub fn create(
        accelerators: &[Accelerator],
        device: &Device,
        width: u32,
        height: u32,
        options: DenoiserOptions,
    ) -> Result<Self, DenoiserError> {
        Self::create_on_stream(accelerators, device, width, height, options, StreamId::current())
    }

    pub(crate) fn create_on_stream(
        accelerators: &[Accelerator],
        device: &Device,
        width: u32,
        height: u32,
        options: DenoiserOptions,
        stream_id: StreamId,
    ) -> Result<Self, DenoiserError> {
        let accelerator =
            sniff_best_accelerator(accelerators, device).ok_or(DenoiserError::NoAcceleratorAvailable)?;

        let params = options.to_nlm_params();
        params.validate()?;
        validate_dimensions(width, height)?;

        let temporal_radius = params.temporal_radius;
        let backend = build_backend(
            accelerator,
            device,
            &options.algorithm,
            params,
            width,
            height,
            options.output_format,
            stream_id,
        )?;

        Ok(Self {
            backend,
            pending: VecDeque::with_capacity(MAX_PENDING),
            accelerator,
            width,
            height,
            temporal_radius,
            output_format: options.output_format,
            frames_pushed: 0,
            poisoned: false,
        })
    }

    /// The accelerator [`sniff_best_accelerator`] picked.
    pub fn selected_accelerator(&self) -> Accelerator {
        self.accelerator
    }

    /// The width passed at construction.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The height passed at construction.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The temporal radius the resolved parameters run at.
    pub fn temporal_radius(&self) -> u32 {
        self.temporal_radius
    }

    /// The format every collected frame comes back in.
    pub fn output_format(&self) -> OutputFormat {
        self.output_format
    }

    /// How many frames behind and ahead of a target frame this
    /// denoiser needs pushed, in order, to produce that frame's
    /// output through [`PlanarDenoiser::reseed`](crate::PlanarDenoiser::reseed).
    ///
    /// Both NLM algorithms only ever need their own `2 * radius + 1`
    /// sliding window, symmetric around the target frame:
    /// `WindowSpan { behind: radius, ahead: radius }`.
    ///
    /// nl4d's cross-frame accumulator scatters every pass's
    /// contribution across the `2 * radius + 1` frames the pass
    /// reaches, and a frame's own region only starts collecting once
    /// the pass that first reaches it, the one centred `radius` frames
    /// behind it, has run. That earliest pass is itself only real once
    /// the front end's own window is full at that centre, which needs
    /// `radius` more frames behind it again. So nl4d needs the target's
    /// own `radius`-wide neighbourhood doubled on both sides:
    /// `WindowSpan { behind: 2 * radius, ahead: 2 * radius }`.
    pub fn window_span(&self) -> WindowSpan {
        let radius = self.temporal_radius as usize;
        let span = if self.backend.is_nl4d() {
            2 * radius
        } else {
            radius
        };
        WindowSpan {
            behind: span,
            ahead: span,
        }
    }

    /// Uploads one frame into the temporal window.
    ///
    /// `frame` holds `width * height * channels` `f32` values in
    /// `[0, 1]`.
    ///
    /// Once the window is full and the pipeline has room, this also
    /// starts the kernels for the next denoised frame.
    ///
    /// Up to `MAX_PENDING` outputs can be in flight at once, so the GPU
    /// runs one frame's kernels while the previous frame's readback is
    /// still travelling. At that ceiling this returns
    /// [`DenoiserError::QueueFull`], and the caller has to drain a frame
    /// with [`Self::recv_frame`] before pushing more.
    ///
    /// Any other failure poisons the denoiser, so every further call
    /// returns [`DenoiserError::Poisoned`] until [`Self::reset_stream`]
    /// clears it. `QueueFull` does not poison, since it is the documented
    /// retry signal above.
    pub fn push_frame(&mut self, frame: &[f32]) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }
        self.push_frame_inner(frame).inspect_err(|err| {
            if !matches!(err, DenoiserError::QueueFull) {
                self.poisoned = true;
            }
        })
    }

    fn push_frame_inner(&mut self, frame: &[f32]) -> Result<(), DenoiserError> {
        // After `temporal_radius` real pushes the leading-edge mirror
        // has primed the window, so the next push produces a pending
        // frame. From then on every push takes a pending slot.
        let window_full = self.frames_pushed > self.temporal_radius;
        if window_full && self.pending.len() >= MAX_PENDING {
            return Err(DenoiserError::QueueFull);
        }

        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => {
                d.push_frame(frame);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Cuda(p));
                }
            },
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => {
                d.push_frame(frame);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Rocm(p));
                }
            },
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => {
                d.push_frame(frame);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Wgpu(p));
                }
            },
        }

        self.frames_pushed = self.frames_pushed.saturating_add(1);
        Ok(())
    }

    /// Uploads one frame held as wire bytes into the temporal window.
    ///
    /// `planes` holds one `width * height` plane per channel at `depth`,
    /// which the GPU normalises and interleaves. The planes run Y, U, V
    /// for a fused frame and U, V for a chroma pair.
    ///
    /// Queueing, poisoning, and the `QueueFull` retry signal work exactly
    /// as they do for [`Self::push_frame`].
    pub fn push_frame_wire(&mut self, planes: &[&[u8]], depth: Depth) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }
        self.push_frame_wire_inner(planes, depth).inspect_err(|err| {
            if !matches!(err, DenoiserError::QueueFull) {
                self.poisoned = true;
            }
        })
    }

    fn push_frame_wire_inner(&mut self, planes: &[&[u8]], depth: Depth) -> Result<(), DenoiserError> {
        // The same window accounting `push_frame_inner` does.
        let window_full = self.frames_pushed > self.temporal_radius;
        if window_full && self.pending.len() >= MAX_PENDING {
            return Err(DenoiserError::QueueFull);
        }

        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => {
                d.push_frame_wire(planes, depth);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Cuda(p));
                }
            },
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => {
                d.push_frame_wire(planes, depth);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Rocm(p));
                }
            },
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => {
                d.push_frame_wire(planes, depth);
                if let Some(p) = d.denoise_submit()? {
                    self.pending.push_back(BackendPending::Wgpu(p));
                }
            },
        }

        self.frames_pushed = self.frames_pushed.saturating_add(1);
        Ok(())
    }

    /// Uploads one frame held as wire bytes into the temporal window
    /// without starting a denoise.
    ///
    /// The wire counterpart of [`Self::push_frame_priming`].
    pub fn push_frame_wire_priming(&mut self, planes: &[&[u8]], depth: Depth) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }
        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => d.push_frame_wire(planes, depth),
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => d.push_frame_wire(planes, depth),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => d.push_frame_wire(planes, depth),
        }

        self.frames_pushed = self.frames_pushed.saturating_add(1);
        Ok(())
    }

    /// Uploads one frame into the temporal window without starting a
    /// denoise.
    ///
    /// The ring advances exactly as it does for [`Self::push_frame`], so
    /// the window still fills, but no kernels are submitted and no
    /// output is queued. This is how a caller that can hand over a whole
    /// window at once, rather than a strictly ordered stream, fills the
    /// window in one go and lets only the last push in it submit.
    ///
    /// A failure elsewhere poisons the denoiser, so this refuses to run
    /// until [`Self::reset_stream`] clears it.
    pub fn push_frame_priming(&mut self, frame: &[f32]) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }
        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => d.push_frame(frame),
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => d.push_frame(frame),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => d.push_frame(frame),
        }

        self.frames_pushed = self.frames_pushed.saturating_add(1);
        Ok(())
    }

    /// Drops the current stream and returns to the state a fresh
    /// denoiser starts in, keeping every GPU allocation.
    ///
    /// Anything still in flight is discarded. This also clears the
    /// poison an earlier failure left, so it is the recovery path for
    /// [`DenoiserError::Poisoned`].
    pub fn reset_stream(&mut self) {
        self.pending.clear();
        self.frames_pushed = 0;
        self.poisoned = false;

        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => d.reset_stream(),
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => d.reset_stream(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => d.reset_stream(),
        }
    }

    /// Blocks until the in-flight denoise finishes and returns the
    /// cleaned frame.
    ///
    /// Returns `Ok(None)` when nothing is in flight, which happens while
    /// the temporal window is still filling up.
    ///
    /// A failure poisons the denoiser, so every further call returns
    /// [`DenoiserError::Poisoned`] until [`Self::reset_stream`] clears it.
    pub fn recv_frame(&mut self) -> Result<Option<FrameOutput>, DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }
        self.recv_frame_inner().inspect_err(|_| self.poisoned = true)
    }

    fn recv_frame_inner(&mut self) -> Result<Option<FrameOutput>, DenoiserError> {
        let Some(pending) = self.pending.pop_front() else {
            return Ok(None);
        };
        Ok(Some(pending.wait()?))
    }

    /// Polls the in-flight denoise once.
    ///
    /// Returns `Ok(None)` both when nothing is in flight and when the in-flight readback
    /// has not landed yet, so `None` alone does not tell those two cases apart.
    ///
    /// A caller that needs the frame rather than just checking on it should
    /// use [`Self::recv_frame`] instead.
    ///
    /// This only avoids blocking on the wgpu backends, meaning Vulkan and Metal.
    /// On CUDA and ROCm the readback completes synchronously on its first poll,
    /// so this call blocks until the readback lands there, the same as `recv_frame`.
    ///
    /// A failure poisons the denoiser, so every further call returns
    /// [`DenoiserError::Poisoned`] until [`Self::reset_stream`] clears it.
    pub fn try_recv_frame(&mut self) -> Result<Option<FrameOutput>, DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }
        self.try_recv_frame_inner().inspect_err(|_| self.poisoned = true)
    }

    fn try_recv_frame_inner(&mut self) -> Result<Option<FrameOutput>, DenoiserError> {
        let Some(pending) = self.pending.pop_front() else {
            return Ok(None);
        };

        match pending.try_wait()? {
            Ok(frame) => Ok(Some(frame)),
            Err(pending) => {
                self.pending.push_front(pending);
                Ok(None)
            },
        }
    }

    /// Drains the in-flight frames and the trailing temporal tail,
    /// handing each frame it produces to `sink`.
    ///
    /// The tail is padded by repeating the last pushed frame.
    ///
    /// On success the denoiser is ready for a fresh, unrelated stream of
    /// the same size and parameters. Pushing again after a flush starts
    /// a new temporal window from scratch, and flushing more than once
    /// is fine.
    ///
    /// A failure poisons the denoiser, so every further call returns
    /// [`DenoiserError::Poisoned`] until [`Self::reset_stream`] clears it.
    pub fn flush(&mut self, sink: impl FnMut(FrameOutput)) -> Result<(), DenoiserError> {
        if self.poisoned {
            return Err(DenoiserError::Poisoned);
        }
        self.flush_inner(sink).inspect_err(|_| self.poisoned = true)
    }

    fn flush_inner(&mut self, mut sink: impl FnMut(FrameOutput)) -> Result<(), DenoiserError> {
        // Drain the whole pending pipeline, up to MAX_PENDING frames,
        // before submitting the trailing-tail mirrors. This also leaves
        // every output slot free, so the tail's own readbacks cannot be
        // handed a slot a streaming readback is still reading.
        while let Some(frame) = self.recv_frame_inner()? {
            sink(frame);
        }

        // The tail frames come back through each algorithm's own
        // blocking readback, in the same format as every streaming
        // frame, so they are quantised by the same pack kernel.
        match &mut self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda(d) => d.flush(|frame| sink(frame.clone()))?,
            #[cfg(feature = "rocm")]
            Backend::Rocm(d) => d.flush(|frame| sink(frame.clone()))?,
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            Backend::Wgpu(d) => d.flush(|frame| sink(frame.clone()))?,
        }

        // The backend has already reset its own stream indices. Reset
        // the outer push counter too, so the next push re-arms the
        // window-priming check at the top of `push_frame`.
        self.frames_pushed = 0;

        Ok(())
    }

}

fn build_backend(
    accel: Accelerator,
    device: &Device,
    algorithm: &Algorithm,
    params: NlmParams,
    width: u32,
    height: u32,
    output_format: OutputFormat,
    stream_id: StreamId,
) -> Result<Backend, DenoiserError> {
    match accel {
        #[cfg(feature = "cuda")]
        Accelerator::Cuda => {
            let dev = device.to_cuda()?;
            let client = client_on_stream::<cubecl::cuda::CudaRuntime>(&dev, stream_id);
            Ok(Backend::Cuda(build_engine(
                &client,
                algorithm,
                params,
                width,
                height,
                output_format,
            )?))
        },
        #[cfg(feature = "rocm")]
        Accelerator::Rocm => {
            let dev = device.to_amd()?;
            let client = client_on_stream::<cubecl::hip::HipRuntime>(&dev, stream_id);
            Ok(Backend::Rocm(build_engine(
                &client,
                algorithm,
                params,
                width,
                height,
                output_format,
            )?))
        },
        #[cfg(feature = "vulkan")]
        Accelerator::Vulkan => {
            let dev = device.to_wgpu()?;
            let client = client_on_stream::<cubecl::wgpu::WgpuRuntime>(&dev, stream_id);
            Ok(Backend::Wgpu(build_engine(
                &client,
                algorithm,
                params,
                width,
                height,
                output_format,
            )?))
        },
        #[cfg(feature = "metal")]
        Accelerator::Metal => {
            let dev = device.to_wgpu()?;
            let client = client_on_stream::<cubecl::wgpu::WgpuRuntime>(&dev, stream_id);
            Ok(Backend::Wgpu(build_engine(
                &client,
                algorithm,
                params,
                width,
                height,
                output_format,
            )?))
        },
        // Keeps the match exhaustive on docs.rs, where `cfg(docsrs)`
        // widens the `Accelerator` enum to include variants whose
        // backend feature is not enabled. Never reached at runtime.
        #[cfg(docsrs)]
        #[expect(
            unreachable_patterns,
            reason = "the arm only keeps the match exhaustive on docs.rs"
        )]
        _ => unreachable!(),
    }
}

fn client_on_stream<R: Runtime>(device: &R::Device, stream_id: StreamId) -> ComputeClient<R> {
    let mut client = R::client(device);
    unsafe { client.set_stream(stream_id) };
    client
}
