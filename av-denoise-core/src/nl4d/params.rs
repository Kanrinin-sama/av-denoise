use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, MotionEstimation, NlmParams};

/// The largest [`Nl4dParams::kaiser_beta`] worth accepting.
///
/// A Kaiser window's taps fall off faster the larger `beta` is. By 8 the
/// end tap is under a fiftieth of the centre, so a patch's edge pixels
/// contribute almost nothing and the step-4 grid is left covering each
/// pixel with a handful of centres rather than a blend. Past that the
/// window stops being a taper and starts being a mask, and the smallest
/// weights fall under what the fixed-point accumulators resolve.
pub const MAX_KAISER_BETA: f32 = 8.0;

/// The most motion blocks that may cover a reference patch on one axis.
///
/// A block grid at a step below `blksize` puts several blocks over one
/// patch, and
/// [`crate::collab::kernels::fused::collab_fused`] searches all of them.
/// It unrolls its per-neighbour duplicate-rectangle arrays over the
/// square of this bound, so the bound is what caps the shader's register
/// footprint. At 4 the arrays hold 16 rectangles and the shipped
/// geometry, `blksize = 16` at `overlap = 8`, uses 2.
///
/// The step is `blksize - overlap`, so 4 admits an overlap of up to
/// three quarters of the block size.
pub const MAX_COVERING_BLOCKS: u32 = 4;

/// Tuning for [`super::Nl4dDenoiser`].
///
/// `nlm` supplies the front end that builds the frame ring, the motion
/// field, and the confidence scores the temporal grouping reads. Its own
/// `temporal_radius` is overwritten at construction time from this
/// struct's own `temporal_radius`, so it does not need to be set by the
/// caller.
#[derive(Debug, Clone)]
pub struct Nl4dParams {
    /// Machinery configuration for the front end. `hq` must be `Some`
    /// with `temporal_confidence` on, and `motion_compensation` must be
    /// active, because [`crate::nlmeans::NlmDenoiser::submit_machinery`]
    /// only builds a ring view when both are on, and this denoiser is
    /// built entirely on top of that call.
    pub nlm: NlmParams,
    /// How many frames on each side of the centre frame the temporal
    /// search reaches into. In `1..=8`.
    pub temporal_radius: u32,
    /// Half-width of the refine window searched around each neighbour
    /// frame's motion-predicted position. In `1..=4`.
    pub refine: u32,
    /// Half-width of the spatial candidate window searched in the
    /// centre frame. In `1..=16`.
    pub spatial_radius: u32,
    /// Hard-threshold multiplier on the propagated coefficient sigma.
    /// Higher shrinks more coefficients, so it removes more noise and
    /// more fine detail.
    ///
    /// Defaults to 4.158. Note that in reality luma and chroma want separately
    /// tuned values. See [nl4d_default_lambda_ht](crate::nl4d_default_lambda_ht).
    pub lambda_ht: f32,
    /// The confidence floor below which a whole neighbour block is
    /// skipped rather than scored, in `[0, 1)`. A block below the floor
    /// is never scored, and a volume left short of frames by the skip
    /// makes its group filter from the centre frame alone.
    pub c_min: f32,
    /// The `beta` of the Kaiser window each filtered patch is tapered
    /// with as it is aggregated, in `0..=8`.
    ///
    /// A pixel is covered by many patches, each of which made its own
    /// threshold decision. Tapering a patch toward its edges blends
    /// those decisions rather than letting each reach its boundary at
    /// full strength. Larger tapers harder. BM3D uses 2.0.
    ///
    /// Defaults to 2.0, BM3D's own value. `0.0` is exactly uniform
    /// aggregation, which is what this did before the window existed.
    /// See [`crate::collab::kernels::aggregate::kaiser_window`].
    pub kaiser_beta: f32,
    /// The penalty on a block's vector deviating from its
    /// neighbourhood's median, in the field regularisation pass.
    ///
    /// The pass re-scores each block's vector against the median of its
    /// neighbours, the four adjacent blocks' vectors and zero, adding
    /// this times the distance from the median, in pixels, scaled so
    /// `1.0` weighs one pixel of deviation like a 5/255 per-pixel
    /// mismatch. Defaults to `1.0`, calibrated with a `field_lambda`
    /// sweep on the `mc_accuracy` bench. The pass gains most of its
    /// accuracy by a moderate penalty and further increases add little,
    /// so `1.0` sits inside that plateau rather than at its edge. `0.0`
    /// skips the pass.
    pub field_lambda: f32,
    /// Scales the luma threshold by how noisy each brightness level is in the current frame.
    ///
    /// On by default.
    pub noise_map: bool,
    /// How much harder flat, grainy 8x8 areas are filtered, as a multiplier on the luma threshold.
    ///
    /// An area counts as flat when its texture is small next to its own grain. Only takes effect
    /// with `noise_map` on. `1.0` turns it off. Between 1.0 and 3.0, defaults to 1.75.
    pub flat_boost: f32,
    /// [Self::flat_boost] for the chroma planes, with flatness measured on the first chroma plane.
    ///
    /// Only takes effect with `noise_map` on. `1.0` turns it off. Between 1.0 and 3.0, defaults
    /// to 1.5.
    pub chroma_flat_boost: f32,
    /// How much more gently textured dark areas are filtered, as a multiplier on the luma
    /// threshold.
    ///
    /// It applies in full at or below luma 128 of 255 and fades back to 1.0 by 160. Only takes
    /// effect with `noise_map` on. `1.0` turns it off. Between 0.1 and 1.0, defaults to 0.65.
    pub shadow_soften: f32,
    /// How strongly the grain in a flat area may line up before the area counts as texture instead.
    ///
    /// Grain points every which way, while faint lines and edges share one direction. A flat area
    /// whose surroundings line up at or above this cut loses the flat boost and is filtered as
    /// texture. Lower keeps more texture, higher filters more areas as flat. Only takes effect
    /// with `noise_map` on, and only on luma. `1.0` turns it off. Between 0.0 and 1.0, defaults
    /// to 0.21.
    pub flat_texture_cut: f32,
    /// Judges each transform coefficient together with its frequency neighbours instead of alone.
    ///
    /// Faint texture spreads over several neighbouring frequencies, so it survives where each
    /// coefficient alone would fall under the threshold. On by default.
    pub pooled_threshold: bool,
    /// Whether the denoiser measures the source's film grain for an AV1 grain table.
    ///
    /// Off by default. Only a denoiser that filters luma measures. It never changes the output.
    /// Measured chunks stay on the GPU until drained, so callers drain after each flush.
    pub grain_export: bool,
}

impl Default for Nl4dParams {
    fn default() -> Self {
        Self {
            nlm: NlmParams {
                temporal_radius: 2,
                channels: ChannelMode::Yuv,
                motion_compensation: MotionCompensationMode::Mvtools {
                    blksize: 16,
                    overlap: 8,
                    search_radius: 4,
                    pyramid_levels: 2,
                    estimation: MotionEstimation::Auto,
                },
                hq: Some(HqParams::default()),
                ..NlmParams::default()
            },
            temporal_radius: 2,
            refine: 2,
            spatial_radius: 9,
            lambda_ht: 4.158,
            c_min: 0.05,
            kaiser_beta: 2.0,
            field_lambda: 1.0,
            noise_map: true,
            flat_boost: 1.75,
            chroma_flat_boost: 1.5,
            shadow_soften: 0.65,
            flat_texture_cut: 0.21,
            pooled_threshold: true,
            grain_export: false,
        }
    }
}

impl Nl4dParams {
    /// Rejects a configuration that would fail to launch, or that would
    /// hit [`crate::nlmeans::NlmDenoiser::submit_machinery`]'s own
    /// preconditions only once a real submit ran.
    pub fn validate(&self) -> Result<(), String> {
        let Some(hq) = self.nlm.hq else {
            return Err(
                "nlm.hq must be Some, the front end's noise estimate and confidence weighting \
                 are what submit_machinery builds the ring view from"
                    .to_string(),
            );
        };

        if !self.nlm.motion_compensation.is_active() {
            return Err(
                "nlm.motion_compensation must be active, the temporal grouping kernel reads \
                 the motion field submit_machinery builds from it"
                    .to_string(),
            );
        }

        if !hq.temporal_confidence {
            return Err(
                "nlm.hq.temporal_confidence must be true, submit_machinery returns an error \
                 unless both motion compensation and the confidence buffer are active"
                    .to_string(),
            );
        }

        // Only checked once the geometry itself is sound. An overlap at
        // or past blksize gives a step of 0, which `nlm.validate()`
        // rejects on its own terms below with the real fault named. Left
        // unguarded, that same case saturates the step to 1 here and
        // reports a nonsensical covering-block count instead.
        if let MotionCompensationMode::Mvtools { blksize, overlap, .. } = self.nlm.motion_compensation
            && overlap < blksize
        {
            let step = blksize - overlap;
            let covers = blksize.div_ceil(step);
            if covers > MAX_COVERING_BLOCKS {
                return Err(format!(
                    "nlm.motion_compensation blksize={blksize} at overlap={overlap} gives a step \
                     of {step}, so {covers} blocks cover a patch on each axis, past the \
                     {MAX_COVERING_BLOCKS} the temporal grouping kernel unrolls its search over. \
                     Raise the step by lowering the overlap."
                ));
            }
        }

        if !(1..=crate::collab::MAX_TEMPORAL_RADIUS).contains(&self.temporal_radius) {
            return Err(format!(
                "temporal_radius={} must be in 1..={}",
                self.temporal_radius,
                crate::collab::MAX_TEMPORAL_RADIUS,
            ));
        }

        if !(1..=4).contains(&self.refine) {
            return Err(format!("refine={} must be in 1..=4", self.refine));
        }

        if !(1..=16).contains(&self.spatial_radius) {
            return Err(format!(
                "spatial_radius={} must be in 1..=16",
                self.spatial_radius
            ));
        }

        if !(self.lambda_ht.is_finite() && self.lambda_ht > 0.0) {
            return Err(format!(
                "lambda_ht must be finite and greater than 0, got {}",
                self.lambda_ht
            ));
        }

        if !(self.c_min.is_finite() && self.c_min >= 0.0 && self.c_min < 1.0) {
            return Err(format!("c_min must be finite and in [0, 1), got {}", self.c_min));
        }

        if !(self.kaiser_beta.is_finite() && (0.0..=MAX_KAISER_BETA).contains(&self.kaiser_beta)) {
            return Err(format!(
                "kaiser_beta must be finite and in 0..={MAX_KAISER_BETA}, got {}",
                self.kaiser_beta
            ));
        }

        if !(self.field_lambda.is_finite() && self.field_lambda >= 0.0) {
            return Err(format!(
                "field_lambda must be finite and at least 0, got {}",
                self.field_lambda
            ));
        }

        for (name, value) in [
            ("flat_boost", self.flat_boost),
            ("chroma_flat_boost", self.chroma_flat_boost),
        ] {
            if !(value.is_finite() && (1.0..=3.0).contains(&value)) {
                return Err(format!("{name} must be finite and in 1.0..=3.0, got {value}"));
            }
        }

        if !(self.shadow_soften.is_finite() && (0.1..=1.0).contains(&self.shadow_soften)) {
            return Err(format!(
                "shadow_soften must be finite and in 0.1..=1.0, got {}",
                self.shadow_soften
            ));
        }

        if !(self.flat_texture_cut.is_finite() && (0.0..=1.0).contains(&self.flat_texture_cut)) {
            return Err(format!(
                "flat_texture_cut must be finite and in 0.0..=1.0, got {}",
                self.flat_texture_cut
            ));
        }

        Ok(())
    }
}
