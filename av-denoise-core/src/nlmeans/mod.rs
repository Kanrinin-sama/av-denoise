//! The non-local means denoiser that sits behind [`crate::Denoiser`].
//!
//! Non-local means cleans a pixel by finding patches elsewhere that look
//! like the patch around it, then averaging them. Similar patches get a
//! large weight and dissimilar ones get almost none, so flat areas
//! smooth out while edges survive.
//!
//! The search can reach across neighbouring frames as well as within one
//! frame, which is what the temporal radius controls.
//!
//! # Layout
//!
//! `params` holds the tuning values and the calibrated defaults, and
//! [`NlmParams`] is the single struct everything else is built from.
//!
//! [`NlmDenoiser`] owns the GPU buffers and the frame ring, and
//! `dispatch` turns one set of parameters into the sequence of kernel
//! launches that produces a frame.
//!
//! [`kernels`] holds the GPU code itself. `noise` measures how noisy a
//! frame is, [`motion`] tracks movement between frames, and
//! [`prefilter`] builds the cleaner reference image that patches are
//! compared against.

pub mod kernels;
pub mod motion;
pub mod prefilter;

mod align;
mod denoiser;
mod dispatch;
mod noise;
mod params;
mod pending;

pub(crate) use denoiser::RingView;
pub use denoiser::{GpuOutput, NlmDenoiser};
pub use motion::{MotionCompensationMode, MotionEstimation, MotionSearch};
pub use params::{
    ChannelMode,
    HqParams,
    MAX_PATCH_RADIUS,
    MAX_SEARCH_RADIUS,
    MAX_TEMPORAL_RADIUS,
    MIN_FRAME_DIM,
    NlmParams,
    hq_default_strength,
    validate_dimensions,
};
pub use pending::{Pending, TryWait};
pub(crate) use pending::{ReadbackShape, start_readback};
pub use prefilter::{DEFAULT_PILOT_STRENGTH_SCALE, PrefilterMode, parse_prefilter};

/// Cube X dimension for tile-heavy fused/separable kernels.
pub const BLOCK_X: u32 = 32;
/// Cube Y dimension for tile-heavy fused/separable kernels.
pub const BLOCK_Y: u32 = 8;

/// Cube shape for the per-pixel `nlm_accumulate` kernel, which has no
/// shared-memory tile.
///
/// On RDNA-class GPUs this shape benchmarks 10 to 25% faster than the
/// tile-heavy default. The kernel waits on memory rather than compute,
/// so the extra threads hide the load latency.
pub const BLOCK_X_THIN: u32 = 32;
pub const BLOCK_Y_THIN: u32 = 16;

/// Largest 1D grid a dispatch may ask for, set by the WebGPU and Vulkan
/// limits.
pub(crate) const MAX_GRID_1D: u32 = 65535;

/// Block size for 1D utility kernels (copy, zero).
pub(crate) const BLOCK_1D: u32 = 256;

/// Bit depth of a source's samples.
///
/// Normalisation divides by [`Depth::max_value`], so a value in
/// normalised units means the same thing at every depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    Eight,
    Ten,
    Twelve,
}

/// Returned when a source declares a bit depth the denoiser does not handle.
#[derive(Debug, thiserror::Error)]
#[error("unsupported bit depth {0}, av-denoise supports 8, 10, and 12-bit")]
pub struct UnsupportedDepthError(pub usize);

impl Depth {
    /// Maps a declared bit depth onto a [`Depth`].
    pub fn from_bits(bits: usize) -> Result<Self, UnsupportedDepthError> {
        match bits {
            8 => Ok(Depth::Eight),
            10 => Ok(Depth::Ten),
            12 => Ok(Depth::Twelve),
            other => Err(UnsupportedDepthError(other)),
        }
    }

    /// Bits per sample.
    pub fn bits(self) -> usize {
        match self {
            Depth::Eight => 8,
            Depth::Ten => 10,
            Depth::Twelve => 12,
        }
    }

    /// Bytes each sample takes up on the wire.
    ///
    /// Depths above 8 use a little-endian 16-bit word.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Depth::Eight => 1,
            Depth::Ten | Depth::Twelve => 2,
        }
    }

    /// The largest sample value this depth can hold, which is also the
    /// normalisation divisor.
    pub fn max_value(self) -> f32 {
        ((1u32 << self.bits()) - 1) as f32
    }

    /// The sample value that means neutral chroma at this depth.
    pub fn neutral_chroma(self) -> u16 {
        1 << (self.bits() - 1)
    }

    /// How [`crate::nlmeans::kernels::gpu_pack_wire`] packs samples at
    /// this depth.
    pub fn wire_pack(self) -> WirePack {
        WirePack {
            max: self.max_value(),
            samples_per_word: 4 / self.bytes_per_sample() as u32,
        }
    }
}

/// The quantisation scale and lane count `gpu_pack_wire` packs one
/// sample with.
///
/// The kernel shifts a quantised sample by `lane * (32 / samples_per_word)`
/// and never masks it, so a `max` wider than the lane holds spills bits
/// into the neighbouring sample. Both values come from one [`Depth`]
/// through [`Depth::wire_pack`], which makes that pairing impossible to
/// get wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WirePack {
    max: f32,
    samples_per_word: u32,
}

impl WirePack {
    /// The scale a normalised value is quantised against.
    pub fn max(self) -> f32 {
        self.max
    }

    /// How many samples share one `u32` word.
    pub fn samples_per_word(self) -> u32 {
        self.samples_per_word
    }
}

/// Scales native-depth samples into normalised `[0, 1]` f32.
pub fn normalize(input: &[u16], depth: Depth) -> Vec<f32> {
    let max = depth.max_value();
    input.iter().map(|&v| v as f32 / max).collect()
}

/// Reverse of [`normalize`].
///
/// Values outside `[0, 1]` are clamped, and `NaN` becomes 0.
pub fn denormalize(input: &[f32], depth: Depth) -> Vec<u16> {
    let max = depth.max_value();
    input
        .iter()
        .map(|&v| (v * max).round().clamp(0.0, max) as u16)
        .collect()
}
