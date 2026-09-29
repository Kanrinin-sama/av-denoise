use std::path::PathBuf;

use av_denoise_core::PlaneOptions;

use crate::{CreationGuard, FrameRange};

/// Everything a resident window service runs with.
///
/// `planes` carries the nl4d options, the accelerators, the device and
/// the output depth. `creation_guard` wraps every call that creates
/// threads inside the decoder or the GPU runtime.
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub source: PathBuf,
    pub planes: PlaneOptions,
    pub scene_layout: PathBuf,
    pub keep_frames: Vec<FrameRange>,
    pub slots: usize,
    pub workers: usize,
    pub frame_budget: u64,
    pub creation_guard: CreationGuard,
}
