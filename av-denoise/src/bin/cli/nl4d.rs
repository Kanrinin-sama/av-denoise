use av_denoise::{
    Algorithm,
    ChannelIntent,
    DenoisingMode,
    Depth,
    Nl4dOptions,
    PlaneOptions,
    nl4d_spatial_radius_for,
    nl4d_temporal_radius_for,
};

use super::{Args, CommonArgs, MotionArgs, Preset, RunOptions, resolve_channel_intent};

/// Flags for `nl4d`, which groups 8x8 patches across the temporal window
/// itself, rather than filtering with `nlmeans` first and grouping
/// within one frame afterward.
///
/// nl4d measures the noise level and tracks motion the way `nlmeans hq`
/// does, but never weights or averages patches the NLM way, so none of
/// the NLM knobs appear here.
#[derive(Debug, Clone, clap::Args)]
pub struct Nl4dArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// How many neighbouring frames to look at on each side when
    /// cleaning a frame.
    ///
    /// Larger values find more matches for a patch but use more memory
    /// and add latency. Between `1` and `8`.
    ///
    /// When `--input` names a file this is reset at every scene change,
    /// so raising it never causes blending across cuts.
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(long)]
    pub temporal_radius: Option<u32>,

    /// Half-width of the spatial candidate window searched in the
    /// centre frame.
    ///
    /// This is where most of the search work goes, since the window
    /// covers `(2 * radius + 1)^2` positions. Larger values find more
    /// matches but cost quadratically more. Between `1` and `16`.
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(long)]
    pub spatial_radius: Option<u32>,

    /// Half-width of the refine window searched around each neighbour
    /// frame's motion-predicted position.
    ///
    /// Raise it when motion tracking lands close but not exact.
    /// Between `1` and `4`. Library default is 2.
    #[arg(long)]
    pub refine: Option<u32>,

    /// How aggressively small transform coefficients are zeroed out.
    ///
    /// Higher removes more noise and more fine detail. The library
    /// default differs between luma and chroma.
    ///
    /// Setting this applies one value to both planes, unless
    /// `--luma-lambda-ht` or `--chroma-lambda-ht` overrides it.
    #[arg(long)]
    pub lambda_ht: Option<f32>,

    /// `--lambda-ht` override for the brightness plane only.
    ///
    /// Falls back to `--lambda-ht`, or to the per-plane library default,
    /// when not set.
    ///
    /// Ignored when luma is not being denoised, or when `--channel-mode
    /// yuv` is used.
    #[arg(long)]
    pub luma_lambda_ht: Option<f32>,

    /// `--lambda-ht` override for the colour planes only.
    ///
    /// Falls back to `--lambda-ht`, or to the per-plane library default,
    /// when not set.
    ///
    /// Ignored when chroma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long)]
    pub chroma_lambda_ht: Option<f32>,

    /// Multiplies the `--lambda-ht` in effect for each plane.
    ///
    /// `1.0` (the library default) changes nothing. Raise it to remove
    /// more noise and more fine detail with it. Lower it to keep more
    /// detail.
    ///
    /// This is the single dial for moving both planes together, since
    /// luma and chroma have different defaults. It applies on top of
    /// `--lambda-ht` and the per-plane overrides as well, so pinning one
    /// plane and scaling both works. Between `0.1` and `10.0`.
    #[arg(long)]
    pub lambda_ht_scale: Option<f32>,

    /// How much a poorly matched neighbour patch is distrusted.
    ///
    /// A patch matched in a neighbour frame is treated as a noisier view
    /// of the same content, as noisy as its own match residual says, and
    /// this scales how much noisier. `1.0` (the library default) is the
    /// shipped calibration. `0` matches `--no-confidence-variance`.
    ///
    /// The variance grows with the square of this, so `2` distrusts a
    /// bad match four times as much. The effect saturates. It saturates
    /// sooner the worse the patch matched, because the variance the
    /// mechanism derives is capped at 64 times the channel's own
    /// variance, and values above `16` are rejected because nothing up
    /// there can change a pixel.
    ///
    /// Setting this applies one value to both planes, unless
    /// `--luma-mismatch-scale` or `--chroma-mismatch-scale` overrides
    /// it.
    #[arg(long)]
    pub mismatch_scale: Option<f32>,

    /// `--mismatch-scale` override for the brightness plane only.
    ///
    /// Falls back to `--mismatch-scale`, or to the library default, when
    /// not set.
    ///
    /// Ignored when luma is not being denoised, or when `--channel-mode
    /// yuv` is used.
    #[arg(long)]
    pub luma_mismatch_scale: Option<f32>,

    /// `--mismatch-scale` override for the colour planes only.
    ///
    /// Falls back to `--mismatch-scale`, or to the library default, when
    /// not set.
    ///
    /// Ignored when chroma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long)]
    pub chroma_mismatch_scale: Option<f32>,

    /// How noisy the source is. Leave it unset for almost all uses.
    ///
    /// The noise level is measured automatically per scene when this
    /// is not set. Set it only when the automatic estimate misjudges
    /// a source and you want to pin the value.
    ///
    /// Small values mean light grain and larger values mean heavier
    /// noise. `3` is subtle grain, `6` is clearly visible grain, `12`
    /// and up is heavy noise.
    ///
    /// Always expressed on an 8-bit 0-255 scale, no matter the
    /// source's actual bit depth.
    #[arg(long)]
    pub sigma: Option<f32>,

    /// Nudges the automatically measured noise level up or down.
    ///
    /// `1.0` (the library default) keeps the measurement as-is. Raise
    /// it a little when the cleaned result still looks noisy. Lower
    /// it when detail is getting scrubbed.
    ///
    /// Has no effect when `--sigma` pins the noise level.
    #[arg(long)]
    pub sigma_scale: Option<f32>,

    /// How badly a neighbour frame may match before its patches stop
    /// being trusted.
    ///
    /// Higher values tolerate larger mismatches. Library default is 1.0.
    #[arg(long)]
    pub thsad_scale: Option<f32>,

    /// The confidence floor below which a whole neighbour block is
    /// skipped rather than scored.
    ///
    /// Between `0` and `1`, not including `1`. Library default is
    /// 0.05. Only affects how much compute a submit spends, never
    /// which candidates are admitted once they are scored.
    #[arg(long)]
    pub c_min: Option<f32>,

    /// Tapers each filtered patch toward its edges as it is blended
    /// back into the frame, with a Kaiser window of this `beta`.
    ///
    /// Between `0` and `8`. Library default is 2, BM3D's own value. A
    /// pixel is covered by many patches, each of which made its own
    /// threshold decision, and a taper blends those decisions instead of
    /// letting each one reach its patch boundary at full strength.
    /// Larger tapers harder. `0` blends every patch uniformly.
    #[arg(long)]
    pub kaiser_beta: Option<f32>,

    /// How strongly motion vectors are pulled toward their neighbours.
    ///
    /// `1.0` (the library default) is the shipped calibration and
    /// smooths the tracked field. Raise it to smooth out stray vectors
    /// on noisy or flat content, at the cost of following small
    /// objects less closely. `0` leaves the tracked field as it is.
    #[arg(long)]
    pub field_lambda: Option<f32>,

    /// Stops a poorly matched patch from being trusted less than a well
    /// matched one.
    ///
    /// On by default, the shrinkage treats a patch matched with a large
    /// match distance as a noisier observation. This flag gives every
    /// patch the same noise estimate instead.
    #[arg(long)]
    pub no_confidence_variance: bool,

    /// Estimates noise from a local window instead of a temporal EMA
    /// over stream history.
    ///
    /// Experimental measurement switch for comparing the two estimators
    /// on real footage. Not a committed public interface. Off by
    /// default, which keeps the temporal EMA every calibrated preset
    /// assumes.
    #[arg(long, hide = true)]
    pub windowed_noise_estimation: bool,

    #[command(flatten)]
    pub motion: MotionArgs,
}

impl Nl4dArgs {
    /// How far the temporal window reaches, from the explicit flag or
    /// the active `--preset`.
    ///
    /// nl4d groups patches across neighbouring frames, so a window of 0
    /// leaves it nothing to do. No preset resolves that way, so only an
    /// explicit `--temporal-radius 0` reaches the guard, and it is
    /// rejected here rather than left to fail deep inside construction.
    fn temporal_radius(&self, preset: Preset) -> Result<u32, anyhow::Error> {
        let radius = self
            .temporal_radius
            .unwrap_or_else(|| nl4d_temporal_radius_for(preset));

        if radius == 0 {
            anyhow::bail!(
                "nl4d groups patches across neighbouring frames, so --temporal-radius has to \
                 be 1 or more"
            );
        }

        Ok(radius)
    }

    /// Turns the parsed flags plus the shared globals into the options
    /// the ingest pipeline takes.
    pub fn build_options(&self, globals: &Args) -> Result<RunOptions, anyhow::Error> {
        let defaults = Nl4dOptions::default();
        let intent = resolve_channel_intent(&globals.channel_mode)?;

        // Check the raw 8-bit value here so an out-of-range `--sigma`
        // reports the number the user typed. The library re-validates
        // the same bound after the /255 normalisation, but its message
        // speaks in [0, 1] units.
        if let Some(sigma) = self.sigma
            && (!sigma.is_finite() || sigma <= 0.0 || sigma > 255.0)
        {
            anyhow::bail!("--sigma must be a finite value in (0, 255] 8-bit units (got {sigma})");
        }

        if self.sigma.is_some() && self.sigma_scale.is_some_and(|v| v != 1.0) {
            tracing::warn!("--sigma-scale has no effect when --sigma pins the noise level");
        }

        self.warn_on_dead_per_plane_flags(intent);

        Ok(RunOptions {
            planes: PlaneOptions {
                accelerators: globals.accelerators.clone(),
                device: globals.device.clone(),
                intent,
                mode: DenoisingMode::Temporal {
                    radius: self.temporal_radius(globals.preset)?,
                },
                output_depth: self
                    .common
                    .output_depth
                    .map(|bits| Depth::from_bits(bits as usize))
                    .transpose()?,
                algorithm: Algorithm::Nl4d(Nl4dOptions {
                    motion: self.motion.to_motion_search(),
                    sigma: self.sigma.map(|s| s / 255.0),
                    sigma_scale: self.sigma_scale.unwrap_or(defaults.sigma_scale),
                    thsad_scale: self.thsad_scale.unwrap_or(defaults.thsad_scale),
                    refine: self.refine.unwrap_or(defaults.refine),
                    spatial_radius: self
                        .spatial_radius
                        .unwrap_or_else(|| nl4d_spatial_radius_for(globals.preset)),
                    // Left unresolved when unset, rather than picked from
                    // `defaults` here, because the default depends on which
                    // plane is being denoised and that is not known yet.
                    // `PlaneOptions::algorithm_for` (av-denoise-core/src/frame/mod.rs)
                    // applies `--luma-`/`--chroma-lambda-ht` on top of this
                    // once the plane is known, and construction fills in the
                    // per-plane default for whatever is still unset.
                    lambda_ht: self.lambda_ht,
                    lambda_ht_scale: self.lambda_ht_scale.unwrap_or(defaults.lambda_ht_scale),
                    c_min: self.c_min.unwrap_or(defaults.c_min),
                    mismatch_scale: self.mismatch_scale.unwrap_or(defaults.mismatch_scale),
                    confidence_variance: !self.no_confidence_variance,
                    kaiser_beta: self.kaiser_beta.unwrap_or(defaults.kaiser_beta),
                    field_lambda: self.field_lambda.unwrap_or(defaults.field_lambda),
                    // The CLI keeps the temporal EMA every calibrated
                    // preset assumes by default. Only `av-denoise-vs`
                    // needs window-local estimation, for random-access
                    // determinism. `--windowed-noise-estimation` exists
                    // to measure the difference on real footage.
                    windowed_noise_estimation: self.windowed_noise_estimation,
                }),
                // nl4d has no NLM weighting pass for a strength to apply to.
                luma_strength: None,
                chroma_strength: None,
                luma_lambda_ht: self.luma_lambda_ht,
                chroma_lambda_ht: self.chroma_lambda_ht,
                luma_mismatch_scale: self.luma_mismatch_scale,
                chroma_mismatch_scale: self.chroma_mismatch_scale,
            },
            progress: globals.progress,
        })
    }

    /// Warns when a per-plane override was set but the resolved
    /// `--channel-mode` means it has no effect.
    fn warn_on_dead_per_plane_flags(&self, intent: ChannelIntent) {
        match intent {
            ChannelIntent::Luma if self.chroma_lambda_ht.is_some() => {
                tracing::warn!("--chroma-lambda-ht is ignored when chroma is not being denoised");
            },
            ChannelIntent::Chroma if self.luma_lambda_ht.is_some() => {
                tracing::warn!("--luma-lambda-ht is ignored when luma is not being denoised");
            },
            ChannelIntent::YuvFused if self.luma_lambda_ht.is_some() || self.chroma_lambda_ht.is_some() => {
                tracing::warn!(
                    "--luma-lambda-ht and --chroma-lambda-ht are ignored when --channel-mode yuv is used"
                );
            },
            _ => {},
        }

        match intent {
            ChannelIntent::Luma if self.chroma_mismatch_scale.is_some() => {
                tracing::warn!("--chroma-mismatch-scale is ignored when chroma is not being denoised");
            },
            ChannelIntent::Chroma if self.luma_mismatch_scale.is_some() => {
                tracing::warn!("--luma-mismatch-scale is ignored when luma is not being denoised");
            },
            ChannelIntent::YuvFused
                if self.luma_mismatch_scale.is_some() || self.chroma_mismatch_scale.is_some() =>
            {
                tracing::warn!(
                    "--luma-mismatch-scale and --chroma-mismatch-scale are ignored when \
                     --channel-mode yuv is used"
                );
            },
            _ => {},
        }
    }
}
