/// Number of reference patches along one axis.
///
/// `dim` must be at least `PATCH_SIZE`, which denoiser construction
/// validates.
pub fn refs_along(dim: u32) -> u32 {
    (dim - super::PATCH_SIZE).div_ceil(super::STEP) + 1
}

/// Cubes along x for [`crate::collab::kernels::fused::collab_fused`].
///
/// That kernel gives each of its eight 8-lane groups one reference
/// patch, so a row of references needs an eighth as many cubes as
/// [`refs_along`] returns. The count rounds up, and the last cube of a
/// row runs dead groups for the references past the end.
pub fn fused_cubes_x(width: u32) -> u32 {
    refs_along(width).div_ceil(8)
}

/// Top-left pixel of reference index `i` along one axis. The last
/// reference clamps so its patch stays inside the frame.
pub fn ref_pos(i: u32, dim: u32) -> u32 {
    (i * super::STEP).min(dim - super::PATCH_SIZE)
}

/// Total reference count for a frame.
pub fn ref_count(width: u32, height: u32) -> usize {
    refs_along(width) as usize * refs_along(height) as usize
}

/// Columns and rows of a strength map.
///
/// The map holds one entry per 8x8 quarter of the noise estimator's 16x16 blocks, so a ragged
/// frame edge rounds up to a whole block.
pub fn strength_map_dims(width: u32, height: u32) -> (u32, u32) {
    let cols = 2 * width.div_ceil(16);
    let rows = 2 * height.div_ceil(16);
    (cols, rows)
}
