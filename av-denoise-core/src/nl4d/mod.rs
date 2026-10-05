//! nl4d groups patches across several noisy frames rather than within a
//! single one.
//!
//! [`crate::collab`] groups similar 8x8 patches within one frame and
//! denoises each group jointly. This module extends that search across a
//! motion-compensated window of frames.
//!
//! Each group is a few centre-frame patches, each followed through its
//! best-matching neighbour frames along its own motion vector. Patches
//! followed through time carry independent grain, so the transform along
//! time separates it from the texture they share.

mod denoiser;
mod fruit;
pub mod grain;
pub mod harness;
pub mod kernels;
mod params;
mod regularise;
mod snapshot;

pub use denoiser::Nl4dDenoiser;
pub use params::{MAX_KAISER_BETA, Nl4dParams};
pub use snapshot::MotionSnapshot;
