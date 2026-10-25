use super::Nl4dOptions;
use super::options::resolve_params;
use crate::collab::PATCH_SIZE;
use crate::collab::geometry::{ref_count, strength_map_dims};
use crate::engine::Geometry;
use crate::error::Error;
use crate::nlmeans::{BLOCK_X, BLOCK_Y, MotionEstimation, NOISE_CURVE_BINS};

const ALLOCATION_ALIGNMENT: u64 = 64 * 1024;
const ENGINE_OVERHEAD: u64 = 64 * 1024 * 1024;
const TEMPORAL_STATS_SCALARS: u64 = 37;

#[derive(Default)]
struct Reservation(u64);

impl Reservation {
    fn add(&mut self, bytes: u64, copies: u64) -> Result<(), Error> {
        let bytes = padded(bytes)?;
        let bytes = multiply(&[bytes, copies])?;
        self.0 = self.0.checked_add(bytes).ok_or_else(overflow)?;
        Ok(())
    }

    fn finish(self) -> Result<u64, Error> {
        self.0
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(ENGINE_OVERHEAD))
            .ok_or_else(overflow)
    }
}

fn overflow() -> Error {
    Error::InvalidGeometry("NL4D VRAM requirement exceeds u64".into())
}

fn multiply(factors: &[u64]) -> Result<u64, Error> {
    factors
        .iter()
        .try_fold(1u64, |value, factor| value.checked_mul(*factor))
        .ok_or_else(overflow)
}

fn padded(bytes: u64) -> Result<u64, Error> {
    bytes
        .max(1)
        .checked_next_multiple_of(ALLOCATION_ALIGNMENT)
        .ok_or_else(overflow)
}

fn pyramid_bytes(mut width: u32, mut height: u32, levels: u32) -> Result<u64, Error> {
    let mut bytes = 0u64;
    for _ in 0..levels {
        let level = padded(multiply(&[u64::from(width), u64::from(height), 4])?)?;
        bytes = bytes.checked_add(level).ok_or_else(overflow)?;
        width = (width / 2).max(1);
        height = (height / 2).max(1);
    }
    Ok(bytes)
}

pub fn vram_requirement(options: Nl4dOptions, geometry: Geometry) -> Result<u64, Error> {
    geometry.validate()?;
    if geometry.width < PATCH_SIZE || geometry.height < PATCH_SIZE {
        return Err(Error::InvalidGeometry(format!(
            "NL4D dimensions must be at least {PATCH_SIZE}x{PATCH_SIZE}"
        )));
    }
    let params = resolve_params(&options, geometry.channels)?;
    params.validate().map_err(Error::InvalidOptions)?;
    params
        .nlm
        .validate()
        .map_err(|error| Error::InvalidOptions(error.to_string()))?;
    if params.grain_export {
        return Err(Error::InvalidOptions(
            "NL4D grain export requires a bounded scene length for VRAM budgeting".into(),
        ));
    }

    let width = geometry.width;
    let height = geometry.height;
    let channels = u64::from(geometry.channels.storage_count());
    let radius = u64::from(params.temporal_radius);
    let neighbours = multiply(&[2, radius])?;
    let frames = neighbours.checked_add(1).ok_or_else(overflow)?;
    geometry.check_ring_fits(frames)?;
    let pixels = multiply(&[u64::from(width), u64::from(height)])?;
    let scalar = multiply(&[pixels, 4])?;
    let frame = multiply(&[scalar, channels])?;
    let mut reservation = Reservation::default();

    reservation.add(multiply(&[frame, frames])?, 2)?;
    if params.nlm.prefilter.needs_reference_buf() {
        reservation.add(multiply(&[frame, frames])?, 2)?;
    }
    reservation.add(frame, 3)?;
    reservation.add(scalar, 7)?;
    reservation.add(4, 2)?;
    reservation.add(multiply(&[pixels, channels, frames, 2])?, 1)?;
    let search_side = 2 * u64::from(params.nlm.search_radius) + 1;
    reservation.add(multiply(&[search_side, search_side, 4])?, 2)?;

    if params.nlm.hq.is_some_and(|hq| hq.sigma_override.is_none()) {
        let partials = multiply(&[
            u64::from(width.div_ceil(BLOCK_X)),
            u64::from(height.div_ceil(BLOCK_Y)),
            16,
        ])?;
        reservation.add(multiply(&[padded(partials)?, frames])?, 1)?;
        reservation.add(multiply(&[frames, 16])?, 1)?;
        let record = 2 * channels + TEMPORAL_STATS_SCALARS;
        let stats = multiply(&[
            u64::from(width.div_ceil(16)),
            u64::from(height.div_ceil(16)),
            record,
            4,
        ])?;
        reservation.add(multiply(&[padded(stats)?, frames])?, 1)?;
    }

    let motion = options.motion;
    let step = motion.blksize.checked_sub(motion.overlap).ok_or_else(overflow)?;
    let blocks = multiply(&[u64::from(width.div_ceil(step)), u64::from(height.div_ceil(step))])?;
    let vectors = padded(multiply(&[blocks, 8])?)?;
    let confidence = padded(multiply(&[blocks, 4])?)?;
    reservation.add(multiply(&[vectors, neighbours])?, 1)?;
    reservation.add(multiply(&[confidence, neighbours])?, 1)?;
    let pyramid = multiply(&[pyramid_bytes(width, height, motion.pyramid_levels)?, frames])?;
    reservation.add(pyramid, 1)?;
    if params.nlm.prefilter.needs_reference_buf() {
        reservation.add(pyramid, 1)?;
    }
    if matches!(
        params
            .nlm
            .motion_compensation
            .resolved_estimation(params.temporal_radius),
        Some(MotionEstimation::Chained { .. })
    ) {
        reservation.add(multiply(&[vectors, neighbours, 2])?, 1)?;
    }
    if params.field_lambda > 0.0 {
        reservation.add(multiply(&[vectors, neighbours])?, 1)?;
        reservation.add(multiply(&[confidence, neighbours])?, 1)?;
    }

    reservation.add(multiply(&[ref_count(width, height) as u64, 4])?, 1)?;
    let (map_columns, map_rows) = strength_map_dims(width, height);
    let map = multiply(&[u64::from(map_columns), u64::from(map_rows), 4])?;
    for upload in [
        multiply(&[neighbours, 4])?,
        multiply(&[channels, 4])?,
        multiply(&[NOISE_CURVE_BINS as u64, 4])?,
        map,
    ] {
        reservation.add(upload, 4)?;
    }
    reservation.add(4 * PATCH_SIZE as u64, 2)?;
    reservation.add(map, 1)?;
    reservation.add(multiply(&[frame, frames])?, 1)?;
    reservation.add(multiply(&[scalar, frames])?, 1)?;
    reservation.add(frame, 2)?;
    reservation.add(multiply(&[super::fruit::FRUIT_DITHER.len() as u64, 4])?, 1)?;

    let planes = u64::from(geometry.channels.count());
    reservation.add(geometry.input.plane_bytes(pixels), multiply(&[planes, 4])?)?;
    reservation.add(geometry.output.plane_bytes(pixels), multiply(&[planes, 4])?)?;
    reservation.finish()
}
