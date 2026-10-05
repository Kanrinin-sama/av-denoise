use std::path::PathBuf;

use av_denoise::{
    Algorithm, ChannelIntent, DenoisingMode, Depth, Nl4dOptions, PlaneOptions, nl4d_spatial_radius_for,
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
    /// 0.05. A block below the floor is never scored, and a volume
    /// left short of frames by the skip makes its group filter from
    /// the centre frame alone.
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

    /// Turns off the luma noise map, which scales the threshold by how noisy each brightness
    /// level is.
    #[arg(long)]
    pub no_noise_map: bool,

    /// Turns off the pooled threshold, which judges each frequency together with its neighbours
    /// so faint texture survives.
    ///
    /// With this flag and `--flat-boost 1.5`, the output matches earlier releases.
    #[arg(long)]
    pub no_pooled_threshold: bool,

    /// Filters flat, grainy areas harder, as a multiplier on the luma threshold.
    ///
    /// Between `1` and `3`. Library default is 1.75. `1` turns it off. Has no effect with
    /// `--no-noise-map`.
    #[arg(long)]
    pub flat_boost: Option<f32>,

    /// Filters flat, grainy areas of the colour planes harder, as a multiplier on the chroma
    /// threshold.
    ///
    /// Between `1` and `3`. Library default is 1.5. `1` turns it off. Has no effect with
    /// `--no-noise-map`.
    #[arg(long)]
    pub chroma_flat_boost: Option<f32>,

    /// Filters textured dark areas more gently, as a multiplier on the luma threshold.
    ///
    /// Between `0.1` and `1`. Library default is 0.65. It applies in full at or below luma 128 of
    /// 255 and fades out by 160. `1` turns it off. Has no effect with `--no-noise-map`.
    #[arg(long)]
    pub shadow_soften: Option<f32>,

    /// Stops flat areas whose surroundings line up like faint lines or edges from counting as flat.
    ///
    /// Between `0` and `1`. Library default is 0.21. Lower keeps more faint texture, higher
    /// filters more areas as flat. `1` turns it off. Luma only. Has no effect with
    /// `--no-noise-map`.
    #[arg(long)]
    pub flat_texture_cut: Option<f32>,

    /// Estimates noise from a local window instead of a temporal EMA
    /// over stream history.
    ///
    /// Experimental measurement switch for comparing the two estimators
    /// on real footage. Not a committed public interface. Off by
    /// default, which keeps the temporal EMA every calibrated preset
    /// assumes.
    #[arg(long, hide = true)]
    pub windowed_noise_estimation: bool,

    /// Writes an AV1 film grain table for the grain this run removes. Unstable.
    ///
    /// The flag and its output may change between releases. The table describes the source's
    /// luma grain, minus what the output still keeps. Add it to an AV1 encode with
    /// `grav1synth apply`. SVT-AV1's `--fgs-table` applies only the table's first entry, so it
    /// suits a single-scene table only. Its timestamps follow this run's output frames from 0,
    /// so it suits constant frame rate sources and encodes of this output.
    #[arg(long, value_name = "PATH")]
    pub unstable_export_av1_fgs: Option<PathBuf>,

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

        let exports_grain = self.unstable_export_av1_fgs.is_some();
        if exports_grain && intent == ChannelIntent::Chroma {
            anyhow::bail!("--unstable-export-av1-fgs measures luma grain, so it needs luma to be denoised");
        }

        if exports_grain {
            tracing::warn!(
                "--unstable-export-av1-fgs is unstable, its output and the flag itself may change \
                 between releases"
            );
        }

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
                    kaiser_beta: self.kaiser_beta.unwrap_or(defaults.kaiser_beta),
                    field_lambda: self.field_lambda.unwrap_or(defaults.field_lambda),
                    // The CLI keeps the temporal EMA every calibrated
                    // preset assumes by default. Only `av-denoise-vs`
                    // needs window-local estimation, for random-access
                    // determinism. `--windowed-noise-estimation` exists
                    // to measure the difference on real footage.
                    windowed_noise_estimation: self.windowed_noise_estimation,
                    noise_map: defaults.noise_map && !self.no_noise_map,
                    flat_boost: self.flat_boost.unwrap_or(defaults.flat_boost),
                    chroma_flat_boost: self.chroma_flat_boost.unwrap_or(defaults.chroma_flat_boost),
                    shadow_soften: self.shadow_soften.unwrap_or(defaults.shadow_soften),
                    flat_texture_cut: self.flat_texture_cut.unwrap_or(defaults.flat_texture_cut),
                    pooled_threshold: defaults.pooled_threshold && !self.no_pooled_threshold,
                    grain_export: exports_grain,
                }),
                // nl4d has no NLM weighting pass for a strength to apply to.
                luma_strength: None,
                chroma_strength: None,
                luma_lambda_ht: self.luma_lambda_ht,
                chroma_lambda_ht: self.chroma_lambda_ht,
            },
            progress: globals.progress,
            grain_table: self.unstable_export_av1_fgs.clone(),
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
    }
}
