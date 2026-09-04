//! Measuring how noisy a source is.
//!
//! The HQ variant matches its strength to the noise level, so it needs a
//! number for that level. This module produces one per frame.
//!
//! # Two ways of looking
//!
//! The Immerkær estimate runs a small mask over each frame that cancels
//! smooth content and leaves mostly noise. It is cheap and needs only
//! one frame, but it reads grain that is correlated between neighbouring
//! pixels too low, because such grain looks partly like content to the
//! mask.
//!
//! The temporal estimate compares a frame against the one before it.
//! Where nothing moved, whatever is left over is noise, and correlated
//! grain shows up in full. It needs static content to work, so motion
//! and scene changes make it unreliable.
//!
//! The two are combined by taking whichever reads higher, which lets the
//! temporal estimate correct an Immerkær under-read without letting an
//! unreliable one drag the estimate down.
//!
//! # Two chains
//!
//! The result feeds two separate smoothed estimates.
//!
//! The median chain reads typical noise and drives the filter strength.
//!
//! The low chain reads cautiously, using lower-quartile statistics, and
//! drives the distance floor. Reading that too high scrubs fine texture,
//! so it is deliberately the more conservative of the two.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::align::StorageAlign;
use super::kernels::{
    nlm_noise_partial,
    nlm_noise_reduce,
    nlm_temporal_noise_stats,
    nlm_temporal_stats_zero,
};
use super::{BLOCK_1D, BLOCK_X, BLOCK_Y, MAX_GRID_1D};

/// The inputs one Immerkær noise estimate needs.
///
/// This lives only for the length of a single estimate call, which is
/// what makes the borrows on the denoiser's buffers sound.
pub(super) struct NoiseCtx<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    pub frame_count: u32,
    pub frame: u32,
    pub slot: u32,
    pub input_buf: &'a Handle,
    pub partials_buf: &'a Handle,
    pub results_buf: &'a Handle,
}

/// How many `f32` elements the first stage's partials buffer needs for
/// one frame.
///
/// Each block covers a tile of the frame and contributes four lanes.
pub(super) fn partials_len(width: u32, height: u32) -> usize {
    (width.div_ceil(BLOCK_X) * height.div_ceil(BLOCK_Y) * 4) as usize
}

/// The byte stride between ring slots in the partials buffer, padded up
/// to the runtime's buffer-binding alignment.
///
/// A small frame can leave [`partials_len`] short of a boundary, and
/// wgpu rejects a bind-group offset that is not a multiple of its
/// `min_storage_buffer_offset_alignment`.
///
/// This matches [`temporal_stats_slot_stride_bytes`].
pub(super) fn noise_partials_slot_stride_bytes(width: u32, height: u32, align: StorageAlign) -> u64 {
    align.pad_bytes(partials_len(width, height) as u64 * size_of::<f32>() as u64)
}

/// Runs both stages of the Immerkær noise estimate for one frame.
///
/// The per-channel totals go into the results buffer at this frame's
/// slot.
///
/// That buffer holds four values per ring slot, matching the input
/// ring's frame capacity.
pub(super) fn run_noise_estimate<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &NoiseCtx<'_>,
) -> Result<(), anyhow::Error> {
    let total_input = (ctx.frame_count * ctx.height * ctx.width * ctx.stored_ch) as usize;
    let n_partials = partials_len(ctx.width, ctx.height);
    let total_results = (ctx.frame_count * 4) as usize;
    let stored_ch = ctx.stored_ch as usize;

    unsafe {
        nlm_noise_partial::launch_unchecked::<R>(
            client,
            CubeCount::new_2d(ctx.width.div_ceil(BLOCK_X), ctx.height.div_ceil(BLOCK_Y)),
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            stored_ch,
            ArrayArg::from_raw_parts(ctx.input_buf.clone(), total_input),
            ArrayArg::from_raw_parts(ctx.partials_buf.clone(), n_partials),
            ctx.frame,
            ctx.width,
            ctx.height,
            ctx.channels,
            BLOCK_X,
            BLOCK_Y,
        );
    }

    let num_partials = (n_partials / 4) as u32;
    unsafe {
        nlm_noise_reduce::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(ctx.partials_buf.clone(), n_partials),
            ArrayArg::from_raw_parts(ctx.results_buf.clone(), total_results),
            ctx.slot,
            num_partials,
            BLOCK_1D,
        );
    }

    Ok(())
}

/// Turns the summed absolute mask responses into an Immerkær sigma.
///
/// The interior area leaves out the one-pixel border the mask cannot
/// reach.
pub(super) fn sigma_from_abs_sum(abs_sum: f32, width: u32, height: u32) -> f32 {
    let interior = ((width - 2) as f32) * ((height - 2) as f32);
    (std::f32::consts::FRAC_PI_2).sqrt() * abs_sum / (6.0 * interior)
}

/// The per-channel lower quartile of the per-block Immerkær sigmas.
///
/// It reads one slot's partials directly, in the layout the first stage
/// wrote them.
///
/// Each block's own sigma comes from the same formula
/// [`sigma_from_abs_sum`] uses, applied to however much of that block's
/// tile overlaps the frame's interior.
///
/// A block with no interior overlap is skipped rather than diluting the
/// quartile with a spurious zero.
///
/// Wherever noise is uneven across a frame, this quartile reads lower
/// than the frame-wide mean, which is the cautious estimate the low
/// chain wants.
///
/// Channels past the active count stay at 0.
pub(super) fn sigma_block_p25_from_partials(
    partials: &[f32],
    channels: u32,
    width: u32,
    height: u32,
) -> [f32; 3] {
    let cubes_x = width.div_ceil(BLOCK_X);
    let cubes_y = height.div_ceil(BLOCK_Y);
    let channels = channels as usize;

    let mut cube_sigmas: Vec<Vec<f32>> = vec![Vec::new(); channels];

    for cy in 0..cubes_y {
        let tile_y0 = cy * BLOCK_Y;
        let tile_y1 = ((cy + 1) * BLOCK_Y).min(height);
        let overlap_y0 = tile_y0.max(1);
        let overlap_y1 = tile_y1.min(height - 1);

        for cx in 0..cubes_x {
            let tile_x0 = cx * BLOCK_X;
            let tile_x1 = ((cx + 1) * BLOCK_X).min(width);
            let overlap_x0 = tile_x0.max(1);
            let overlap_x1 = tile_x1.min(width - 1);

            if overlap_x1 <= overlap_x0 || overlap_y1 <= overlap_y0 {
                continue;
            }

            let area = ((overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)) as f32;
            let cube_index = (cy * cubes_x + cx) as usize;
            let base = cube_index * 4;

            for (c, sigmas) in cube_sigmas.iter_mut().enumerate() {
                let sum = partials[base + c];
                sigmas.push(std::f32::consts::FRAC_PI_2.sqrt() * sum / (6.0 * area));
            }
        }
    }

    let mut sigma_low = [0.0f32; 3];
    for (c, sigmas) in cube_sigmas.iter_mut().enumerate() {
        if sigmas.is_empty() {
            continue;
        }
        sort_ascending(sigmas);
        sigma_low[c] = lower_quartile(sigmas);
    }
    sigma_low
}

/// The spatial block size the temporal residual statistics use, with one
/// GPU block per square of this size.
pub(super) const TEMPORAL_NOISE_BLOCK: u32 = 16;

/// How many `f32`s one block's stats record holds, being a sum and a
/// sum of squares per stored channel plus one lag-1 total.
pub(super) fn temporal_stats_record_len(stored_ch: u32) -> u32 {
    2 * stored_ch + 1
}

/// The block grid covering a frame, laid out row-major.
///
/// Ragged edges are truncated rather than padded, the same way the block
/// matcher handles its own ragged last block.
pub(super) fn temporal_stats_blocks(width: u32, height: u32) -> (u32, u32) {
    (
        width.div_ceil(TEMPORAL_NOISE_BLOCK),
        height.div_ceil(TEMPORAL_NOISE_BLOCK),
    )
}

/// Number of `f32`s in one ring slot's stats region.
pub(super) fn temporal_stats_slot_len(width: u32, height: u32, stored_ch: u32) -> usize {
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    (blocks_x * blocks_y * temporal_stats_record_len(stored_ch)) as usize
}

/// The byte stride between ring slots in the temporal-stats buffer,
/// padded up to the runtime's buffer-binding alignment.
///
/// A small frame, or a single-channel mode, can leave
/// [`temporal_stats_slot_len`] short of a boundary, and wgpu rejects a
/// bind-group offset that is not a multiple of its
/// `min_storage_buffer_offset_alignment`.
///
/// This matches `MotionCtx::confidence_bytes_per_neighbour`.
pub(super) fn temporal_stats_slot_stride_bytes(
    width: u32,
    height: u32,
    stored_ch: u32,
    align: StorageAlign,
) -> u64 {
    align.pad_bytes(temporal_stats_slot_len(width, height, stored_ch) as u64 * size_of::<f32>() as u64)
}

/// Total byte size of a `frame_count`-slot temporal-stats ring.
pub(super) fn temporal_stats_buf_bytes(
    width: u32,
    height: u32,
    stored_ch: u32,
    frame_count: u32,
    align: StorageAlign,
) -> usize {
    (temporal_stats_slot_stride_bytes(width, height, stored_ch, align) * frame_count as u64) as usize
}

/// The inputs one temporal residual statistics dispatch needs, comparing
/// the new slot against the previous one on the input ring.
pub(super) struct TemporalStatsCtx<'a> {
    pub width: u32,
    pub height: u32,
    pub stored_ch: u32,
    pub frame_count: u32,
    pub slot_new: u32,
    pub slot_prev: u32,
    pub input_buf: &'a Handle,
    pub stats_buf: &'a Handle,
    pub align: StorageAlign,
}

/// Runs the temporal residual statistics kernel for the new slot,
/// writing one record per block into that slot's padded region.
///
/// The kernel only ever addresses within its own slice, so it needs to
/// know nothing about the ring's other slots or the padding between
/// them.
pub(super) fn run_temporal_noise_stats<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &TemporalStatsCtx<'_>,
) -> Result<(), anyhow::Error> {
    let total_input = (ctx.frame_count * ctx.height * ctx.width * ctx.stored_ch) as usize;
    let (blocks_x, blocks_y) = temporal_stats_blocks(ctx.width, ctx.height);
    let slot_len = temporal_stats_slot_len(ctx.width, ctx.height, ctx.stored_ch);
    let stride = temporal_stats_slot_stride_bytes(ctx.width, ctx.height, ctx.stored_ch, ctx.align);
    let stats_slot = ctx.stats_buf.clone().offset_start((ctx.slot_new as u64) * stride);

    unsafe {
        nlm_temporal_noise_stats::launch_unchecked::<R>(
            client,
            CubeCount::new_2d(blocks_x, blocks_y),
            CubeDim::new_2d(TEMPORAL_NOISE_BLOCK, TEMPORAL_NOISE_BLOCK),
            ctx.stored_ch as usize,
            ArrayArg::from_raw_parts(ctx.input_buf.clone(), total_input),
            ArrayArg::from_raw_parts(stats_slot, slot_len),
            ctx.slot_new,
            ctx.slot_prev,
            ctx.width,
            ctx.height,
            ctx.stored_ch,
            TEMPORAL_NOISE_BLOCK,
        );
    }

    Ok(())
}

/// Fills one ring slot's temporal-stats region with zeroes.
///
/// This runs when a slot is a copy of the one before it, which happens
/// while priming a stream and during the end-of-stream flush.
///
/// The zeroes read as no static blocks with measurable noise, rather
/// than as a made-up reading of zero noise.
pub(super) fn zero_temporal_stats_slot<R: Runtime>(
    client: &ComputeClient<R>,
    stats_buf: &Handle,
    width: u32,
    height: u32,
    stored_ch: u32,
    slot: u32,
    align: StorageAlign,
) {
    let slot_len = temporal_stats_slot_len(width, height, stored_ch) as u32;
    let stride = temporal_stats_slot_stride_bytes(width, height, stored_ch, align);
    let dst = stats_buf.clone().offset_start((slot as u64) * stride);

    let grid = slot_len.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    unsafe {
        nlm_temporal_stats_zero::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(dst, slot_len as usize),
            slot_len,
            total_threads,
        );
    }
}

/// Reads exactly one ring slot's temporal-stats region back as owned
/// values.
///
/// The shared ring handle is sliced by byte offset, so the transfer only
/// covers one slot rather than the whole ring.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(super) fn read_temporal_stats_slot<R: Runtime>(
    client: &ComputeClient<R>,
    stats_buf: &Handle,
    width: u32,
    height: u32,
    stored_ch: u32,
    frame_count: u32,
    slot: u32,
    align: StorageAlign,
) -> Result<Vec<f32>, anyhow::Error> {
    let slot_len_bytes = temporal_stats_slot_len(width, height, stored_ch) as u64 * size_of::<f32>() as u64;
    let stride = temporal_stats_slot_stride_bytes(width, height, stored_ch, align);
    let total_bytes = frame_count as u64 * stride;
    let start = (slot as u64) * stride;
    let end_trim = total_bytes - start - slot_len_bytes;

    let sliced = stats_buf.clone().offset_start(start).offset_end(end_trim);
    let bytes = client
        .read_one(sliced)
        .map_err(|e| anyhow::anyhow!("temporal noise stats readback failed: {e}"))?;
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// One centre slot's aggregated temporal-residual noise measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TemporalNoiseSample {
    /// The per-channel sigma, taken as the median over static blocks, in
    /// normalised units. Entries past the active channel count stay 0.
    pub sigma: [f32; 3],
    /// The same per-channel sigma taken at the lower quartile instead of
    /// the median, in normalised units.
    ///
    /// This reads more cautiously than `sigma`, for the consumers where
    /// reading too high does more harm than reading too low. Entries
    /// past the active channel count stay 0.
    pub sigma_low: [f32; 3],
    /// How correlated the grain is between neighbouring pixels.
    ///
    /// It is the median, over the static blocks with measurable noise,
    /// of how strongly each residual matches the one beside it.
    pub rho: f32,
    /// What fraction of blocks counted as static.
    pub static_fraction: f32,
}

/// How large a block's mean residual can be and still count as static,
/// in normalised units.
///
/// A block above this is treated as moving content rather than as noise.
const STATIC_GATE: f32 = 1.5 / 255.0;
/// The smallest block sigma that still counts toward the correlation
/// median.
///
/// Below this a block carries too little signal for its correlation
/// reading to mean anything.
const RHO_SIGMA_GATE: f32 = 0.3 / 255.0;
/// The smallest fraction of static blocks a sample needs to be trusted.
///
/// Below this, motion or a scene change dominates the frame and the
/// Immerkær estimate is the only usable reading.
const STATIC_FRACTION_MIN: f32 = 0.05;
/// How far above the surviving blocks' own lower quartile a block's
/// sigma may sit before it is treated as moving texture rather than
/// noise.
///
/// # Why this is needed
///
/// A block panning across texture can average out to nearly nothing
/// over its window, which clears [`STATIC_GATE`], while its variance is
/// entirely shifted texture rather than repeatable noise. That variance
/// runs several times higher than the rest of the frame's blocks.
///
/// Real noise keeps a much narrower spread across a frame, even where
/// its magnitude genuinely varies, such as a dark region reading noisier
/// than a bright one. This threshold leaves room for that spread while
/// still catching the texture outliers.
///
/// # Why the lower quartile
///
/// The reference is the lower quartile rather than the median, so the
/// filter still works once texture makes up most of the surviving
/// blocks, as long as a genuinely static minority remains to anchor it.
///
/// The quartile is computed only over blocks whose own sigma clears
/// [`RHO_SIGMA_GATE`]. Letterbox bars and other perfectly static regions
/// read a sigma of exactly 0, and leaving them in can drag the quartile
/// itself to 0, which would reject every block carrying real noise
/// rather than just the outliers.
///
/// That exclusion only holds up while a genuinely low population remains
/// to anchor the quartile. [`aggregate_temporal_noise_stats`] covers
/// what happens when none does.
const SIGMA_OUTLIER_FACTOR: f32 = 5.0;

/// One surviving block's per-channel stats, kept just long enough to
/// work out the reference the outlier check needs before deciding which
/// blocks are really static.
struct StaticGateCandidate {
    sigmas: [f32; 3],
    sigma_ch0: f32,
    var_ch0: f32,
    mean0: f32,
    mean_lag: f32,
    n_pairs: f32,
}

/// Combines one centre slot's per-block records into a single
/// [`TemporalNoiseSample`].
///
/// `records` holds exactly one slot's region, laid out block by block as
/// [`nlm_temporal_noise_stats`] documents.
///
/// # When no sample is produced
///
/// This returns `None` in three cases.
///
/// Too few blocks counted as static, below [`STATIC_FRACTION_MIN`], so
/// motion dominates the frame and nothing here can be trusted.
///
/// No static block carried measurable noise, so the correlation median
/// would be undefined. A zero-filled duplicate slot produces exactly
/// this.
///
/// The outlier check below had no way to validate its own ceiling.
///
/// # Deciding which blocks are static
///
/// This takes two passes.
///
/// The first checks each block's mean residual against [`STATIC_GATE`].
/// A block whose average residual is near zero passes, but a block
/// panning across texture passes too, because displaced texture averages
/// toward zero over a block just as noise does.
///
/// The second pass catches what the first let through. It rejects any
/// surviving block whose sigma sits far above the surviving population's
/// own lower quartile, by more than [`SIGMA_OUTLIER_FACTOR`]. A panning
/// block's variance comes from the texture it slid across, not from
/// noise shared with the rest of the frame.
///
/// That quartile looks only at blocks clearing [`RHO_SIGMA_GATE`], so a
/// perfectly static region such as a letterbox bar cannot drag the
/// ceiling to 0 and reject every noisy block with it.
///
/// # When the ceiling cannot be trusted
///
/// Excluding those low-sigma blocks has a cost of its own. If every
/// remaining block turns out to be texture, that population sets its own
/// ceiling and lets all of its members through, because a value never
/// exceeds a multiple of itself.
///
/// Nothing in a single block's stats distinguishes texture from noise,
/// so the only check left is whether the surviving population shows any
/// internal spread.
///
/// Real per-block sigma, measured over a finite sample, always varies a
/// little from block to block, even for physically uniform noise. So
/// once the low-sigma blocks are gone, a perfectly uniform remainder is
/// the signature of a self-selected outlier population with nothing to
/// anchor it.
///
/// In that case this reports `None` rather than a confident sigma that
/// may be inflated by texture.
pub(super) fn aggregate_temporal_noise_stats(
    records: &[f32],
    channels: u32,
    stored_ch: u32,
    width: u32,
    height: u32,
) -> Option<TemporalNoiseSample> {
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let total_blocks = (blocks_x * blocks_y) as usize;
    if total_blocks == 0 {
        return None;
    }

    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let channels = channels as usize;
    let stored_ch = stored_ch as usize;

    let mut candidates = Vec::new();

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_index = (by * blocks_x + bx) as usize;
            let rec = &records[block_index * record_len..(block_index + 1) * record_len];

            let block_origin_x = bx * TEMPORAL_NOISE_BLOCK;
            let block_origin_y = by * TEMPORAL_NOISE_BLOCK;
            let block_w = TEMPORAL_NOISE_BLOCK.min(width - block_origin_x);
            let block_h = TEMPORAL_NOISE_BLOCK.min(height - block_origin_y);
            let n = (block_w * block_h) as f32;
            let n_pairs = (block_h * block_w.saturating_sub(1)) as f32;

            let mean0 = rec[0] / n;
            if mean0.abs() >= STATIC_GATE {
                continue;
            }

            let mut sigmas = [0.0f32; 3];
            let mut sigma_ch0 = 0.0f32;
            let mut var_ch0 = 0.0f32;
            for c in 0..channels {
                let mean = rec[c] / n;
                let var = (rec[stored_ch + c] / n - mean * mean).max(0.0);
                let sigma_block = var.sqrt() / std::f32::consts::SQRT_2;
                sigmas[c] = sigma_block;
                if c == 0 {
                    sigma_ch0 = sigma_block;
                    var_ch0 = var;
                }
            }

            let mean_lag = if n_pairs > 0.0 {
                rec[2 * stored_ch] / n_pairs
            } else {
                0.0
            };

            candidates.push(StaticGateCandidate {
                sigmas,
                sigma_ch0,
                var_ch0,
                mean0,
                mean_lag,
                n_pairs,
            });
        }
    }

    // A perfectly static block, such as a letterbox bar or a duplicate
    // frame, reads a sigma of 0 and clears STATIC_GATE without effort.
    // Blocks like that can dominate the surviving population while
    // carrying no measurable noise at all.
    //
    // Left in this reference set they drag the quartile toward 0, which
    // then rejects every block with real noise instead of just the
    // texture outliers the check exists for.
    //
    // Limiting the reference to blocks that themselves clear
    // RHO_SIGMA_GATE keeps the ceiling anchored to blocks that could
    // plausibly be noise.
    let mut reference_sigma_ch0: Vec<f32> = candidates
        .iter()
        .map(|c| c.sigma_ch0)
        .filter(|&sigma| sigma > RHO_SIGMA_GATE)
        .collect();
    sort_ascending(&mut reference_sigma_ch0);

    // Dropping the low-sigma blocks throws away what they told us, and
    // that has a cost. A value never exceeds a multiple of itself, so
    // if every remaining block turns out to be a texture-panning block,
    // that population sets its own ceiling and lets all of them
    // through, which is exactly the failure the ceiling exists to
    // prevent.
    //
    // Nothing available here tells texture and noise apart on its own,
    // so this asks for corroboration instead. Some blocks must have
    // been excluded, and the reference population left behind must show
    // real internal spread, meaning its lowest and highest readings
    // differ.
    //
    // Per-block sigma is measured over a finite sample, so even
    // physically uniform noise varies a little from block to block. A
    // reference population with no spread at all, sitting next to
    // excluded low-sigma blocks, is the case this cannot resolve, so it
    // reports None rather than a sigma that may be inflated by texture.
    //
    // A population where nothing was excluded, because there were no
    // low-sigma blocks to begin with, skips this check and is trusted
    // directly.
    let candidates_were_excluded = candidates.len() > reference_sigma_ch0.len();
    let reference_has_spread = match (reference_sigma_ch0.first(), reference_sigma_ch0.last()) {
        (Some(&lo), Some(&hi)) => hi > lo,
        (None, _) | (_, None) => false,
    };
    if candidates_were_excluded && !reference_has_spread {
        return None;
    }

    let sigma_ceiling = if reference_sigma_ch0.is_empty() {
        0.0
    } else {
        lower_quartile(&reference_sigma_ch0) * SIGMA_OUTLIER_FACTOR
    };

    let mut static_sigmas: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let mut rho_samples = Vec::new();
    let mut static_count = 0usize;

    for candidate in &candidates {
        if candidate.sigma_ch0 > sigma_ceiling {
            continue;
        }
        static_count += 1;

        for (c, sigmas) in static_sigmas.iter_mut().enumerate().take(channels) {
            sigmas.push(candidate.sigmas[c]);
        }

        if candidate.sigma_ch0 > RHO_SIGMA_GATE && candidate.n_pairs > 0.0 {
            let rho = (candidate.mean_lag - candidate.mean0 * candidate.mean0) / candidate.var_ch0;
            rho_samples.push(rho.clamp(0.0, 1.0));
        }
    }

    let static_fraction = static_count as f32 / total_blocks as f32;
    if static_fraction < STATIC_FRACTION_MIN || rho_samples.is_empty() {
        return None;
    }

    let mut sigma = [0.0f32; 3];
    let mut sigma_low = [0.0f32; 3];
    for (c, sigmas) in static_sigmas.iter_mut().enumerate() {
        sort_ascending(sigmas);
        sigma[c] = median(sigmas);
        sigma_low[c] = lower_quartile(sigmas);
    }
    sort_ascending(&mut rho_samples);
    let rho = median(&rho_samples);

    Some(TemporalNoiseSample {
        sigma,
        sigma_low,
        rho,
        static_fraction,
    })
}

/// Sorts `values` into ascending order in place.
///
/// Both [`median`] and [`lower_quartile`] need this first, so a caller
/// wanting both sorts once and passes the same slice to each.
fn sort_ascending(values: &mut [f32]) {
    values.sort_by(|a, b| a.partial_cmp(b).expect("noise stats are never NaN"));
}

/// The median of an already-sorted slice, averaging the two middle
/// elements when the count is even.
///
/// Callers only ever pass a non-empty slice.
fn median(values: &[f32]) -> f32 {
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// The lower quartile of an already-sorted slice.
///
/// It reads the value a quarter of the way along, interpolating between
/// the two neighbouring elements when that lands between them.
///
/// A slice of one returns that element. Callers only ever pass a
/// non-empty slice.
fn lower_quartile(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 1 {
        return values[0];
    }
    let idx = 0.25 * (n - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let t = idx - lo as f32;
        values[lo] + t * (values[hi] - values[lo])
    }
}

/// The correlation-correction table, as points sorted by correlation.
///
/// It was measured on synthetic correlated-grain sweeps against the
/// clean bench reference.
///
/// Each factor is how far the quality peak sits above the true sigma at
/// that level of grain correlation, relative to the white-noise optimum
/// at the same sigma.
///
/// At the heaviest correlation measured, both quality metrics prefer the
/// raised value. Past the last measured point the table holds flat.
const CORRELATION_FACTOR_TABLE: [(f32, f32); 4] = [(0.0, 1.0), (0.3, 1.05), (0.5, 1.25), (0.65, 1.45)];

/// The factor that turns a measured temporal sigma into an effective one
/// allowing for grain correlation.
///
/// The effective sigma is the measured one multiplied by this.
pub(super) fn correlation_factor(rho: f32) -> f32 {
    interpolate_table(&CORRELATION_FACTOR_TABLE, rho)
}

/// Reads a value from a small table of points sorted by `x`,
/// interpolating between the two nearest entries.
///
/// `x` is clamped to the table's own range first, so the result never
/// runs past the endpoints.
fn interpolate_table(table: &[(f32, f32)], x: f32) -> f32 {
    let x = x.clamp(table[0].0, table[table.len() - 1].0);

    for pair in table.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        if x <= x1 {
            if x1 == x0 {
                return y1;
            }
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }

    table[table.len() - 1].1
}

/// How much of the noise floor still applies at a given candidate
/// offset, once grain correlation is taken into account.
///
/// A nearby candidate shares some of its grain with the centre patch, so
/// only part of the white-noise floor is genuinely independent noise.
/// The further away the candidate, the less they share, and the more of
/// the floor applies.
///
/// The centre itself always returns 0, because its true distance is zero
/// and none of the floor is independent noise there.
///
/// With no measured correlation, which is also what an inert estimator
/// gives, every other offset returns 1, reproducing the flat white-noise
/// floor exactly.
pub(super) fn spatial_offset_factor(dx: i32, dy: i32, rho: f32) -> f32 {
    if dx == 0 && dy == 0 {
        return 0.0;
    }
    if rho <= 0.0 {
        return 1.0;
    }
    let d = ((dx * dx + dy * dy) as f32).sqrt();
    1.0 - (d * rho.ln()).exp()
}

/// How many `f32`s a spatial-offset table needs at a given search
/// radius.
pub(super) fn spatial_offset_lut_len(search_radius: u32) -> usize {
    let side = (2 * search_radius + 1) as usize;
    side * side
}

/// Builds the per-candidate noise-floor table for a search window, laid
/// out row-major.
///
/// Each entry is the flat noise offset scaled by how much of it applies
/// at that candidate's distance.
///
/// It is cheap enough to rebuild on every submit, reaching at most 289
/// entries at the largest supported search radius.
pub(super) fn build_spatial_offset_lut(search_radius: u32, rho: f32, noise_offset: f32) -> Vec<f32> {
    let r = search_radius as i32;
    let side = (2 * search_radius + 1) as usize;
    let mut lut = vec![0.0f32; side * side];
    for dy in -r..=r {
        for dx in -r..=r {
            let idx = ((dy + r) as usize) * side + (dx + r) as usize;
            lut[idx] = noise_offset * spatial_offset_factor(dx, dy, rho);
        }
    }
    lut
}

/// How much weight the newest frame's estimate carries when smoothing.
///
/// The sigma estimator below and the denoiser's own correlation
/// smoothing both use it.
pub(super) const EMA_ALPHA: f32 = 0.2;
/// The smallest smoothed sigma allowed, in normalised units, which works
/// out at 0.1 in 8-bit terms.
///
/// An estimate near zero would send the derived strength to infinity.
const SIGMA_FLOOR: f32 = 0.1 / 255.0;

/// The noise state for one stream.
///
/// It smooths the per-frame estimates over time, so a single busy frame
/// cannot spike the strength, and applies a floor so near-clean content
/// keeps a usable normalisation factor.
#[derive(Debug, Default)]
pub(super) struct NoiseEstimator {
    ema: Option<Vec<f32>>,
}

impl NoiseEstimator {
    /// Folds a new set of per-channel sigmas into the running estimate
    /// and returns the smoothed result.
    ///
    /// The first call sets the state directly from the sample, because
    /// there is no earlier estimate to blend with. So does every call
    /// once `windowed` is set, which drops the running estimate
    /// entirely and replaces it with this sample: the caller wants the
    /// current window's own reading, not one blended with history from
    /// frames outside it.
    ///
    /// Every element is floored at [`SIGMA_FLOOR`] either way.
    pub(super) fn update(&mut self, sigmas: &[f32], windowed: bool) -> &[f32] {
        match &mut self.ema {
            Some(ema) if !windowed => {
                for (e, &s) in ema.iter_mut().zip(sigmas.iter()) {
                    *e = (EMA_ALPHA * s + (1.0 - EMA_ALPHA) * *e).max(SIGMA_FLOOR);
                }
            },
            _ => {
                self.ema = Some(sigmas.iter().map(|&s| s.max(SIGMA_FLOOR)).collect());
            },
        }
        self.ema.as_deref().unwrap()
    }

    /// Clears the running estimate.
    ///
    /// The next [`Self::update`] then starts from its own sample rather
    /// than blending with stale state.
    pub(super) fn reset(&mut self) {
        self.ema = None;
    }

    /// The current smoothed per-channel sigma, or `None` before the
    /// first [`Self::update`] call.
    pub(super) fn current(&self) -> Option<&[f32]> {
        self.ema.as_deref()
    }
}
