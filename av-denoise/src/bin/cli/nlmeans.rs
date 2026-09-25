use av_denoise::{
    DenoisingMode,
    Depth,
    MotionCompensationMode,
    NlmTuning,
    NlmeansHqOptions,
    NlmeansOptions,
    NlmeansVariant as Variant,
    PlaneOptions,
    PrefilterMode,
    nlmeans_search_radius_for,
    nlmeans_temporal_radius_for,
    nlmeans_variant_for,
    parse_prefilter,
};

use super::{Args, CommonArgs, MotionArgs, Preset, RunOptions, resolve_channel_intent};

#[derive(Debug, Clone, clap::Args)]
pub struct NlmeansArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Which variant to run.
    ///
    /// `fast` uses fixed weighting and is the cheapest option. `hq`
    /// calibrates its weighting to the noise level, measured
    /// automatically per frame (see `--hq-sigma` to override).
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(long)]
    pub variant: Option<Variant>,

    /// Reference image used when comparing patches.
    ///
    /// Omitted (the default) means no prefilter, for both variants.
    ///
    /// `none` forces the noisy input directly (the cheapest option).
    /// This is the same as leaving the flag unset.
    ///
    /// `nlm` or `nlm:<strength_scale>` runs a windowed spatial NLM
    /// pass first and compares patches against that cleaner image.
    /// `strength_scale` multiplies the main pass strength for the
    /// pilot pass. Bare `nlm` uses the calibrated default.
    ///
    /// `bilateral:<sigma_s>,<sigma_r>` runs a quick on-GPU bilateral
    /// blur first, then compares patches against that cleaner image.
    ///
    /// `sigma_s` is the spatial blur radius in pixels, greater than 0
    /// and at most 11.0 (anything beyond this is insane.)
    ///
    /// `sigma_r` is the colour-similarity threshold, greater than 0.
    /// Values above `0` and up to `1` are typical for normalised
    /// pixel data. There is no enforced upper bound.
    ///
    /// A good starting point is `bilateral:3.0,0.02`.
    ///
    /// Prefiltering keeps more detail at the cost of one extra GPU
    /// pass per frame.
    #[arg(long)]
    pub prefilter: Option<String>,

    /// How many neighbouring frames to look at on each side when
    /// cleaning a frame.
    ///
    /// `0` means no temporal blending. Each frame is cleaned on its
    /// own.
    ///
    /// Values above `0` look at that many frames before and after
    /// the current one.
    ///
    /// Larger values give stronger cleanup but use more memory and
    /// add latency.
    ///
    /// When `--input` names a file this is reset at every scene
    /// change, so raising it never causes blending across cuts.
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(long)]
    pub temporal_radius: Option<u32>,

    /// How far away to look for similar patches inside a frame.
    ///
    /// Larger values find more matches but cost quadratically more
    /// work.
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(long)]
    pub search_radius: Option<u32>,

    /// Size of each patch being compared. The patch is
    /// `(2*patch_radius + 1)` pixels square.
    ///
    /// Larger patches preserve fine structure better but cost more
    /// GPU memory. Library default is 4.
    #[arg(long)]
    pub patch_radius: Option<u32>,

    /// Cleaning strength. Higher numbers smooth more.
    ///
    /// Must be a finite number greater than 0.
    ///
    /// The default depends on the variant. `fast` defaults to 1.2.
    /// `hq` interprets strength as a multiplier on the measured noise
    /// level. Its default is calibrated automatically, adapting to the
    /// temporal radius and to which plane (luma or chroma) is being
    /// denoised, so lower and higher radii each get their own measured
    /// value.
    ///
    /// This value applies to both planes unless `--luma-strength`
    /// or `--chroma-strength` is set.
    #[arg(long)]
    pub strength: Option<f32>,

    /// Strength override for the brightness plane only.
    ///
    /// Falls back to `--strength` (or the library default) when not
    /// set.
    ///
    /// Ignored when luma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long)]
    pub luma_strength: Option<f32>,

    /// Strength override for the colour planes only.
    ///
    /// Falls back to `--strength` (or the library default) when not
    /// set.
    ///
    /// Ignored when chroma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long)]
    pub chroma_strength: Option<f32>,

    /// How much weight to give the centre pixel itself when
    /// averaging.
    ///
    /// Library default is 1.0. Must be a finite number `>= 0`.
    ///
    /// Setting to 0 gives pure NLM (centre pixel only counts if a
    /// similar patch was found nearby).
    #[arg(long)]
    pub self_weight: Option<f32>,

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
    pub hq_sigma: Option<f32>,

    /// Treat `--strength` as an absolute value instead of a
    /// multiplier on `--hq-sigma`.
    #[arg(long)]
    pub hq_no_auto_strength: bool,

    /// Keep the expected-noise floor inside patch distances instead
    /// of subtracting it.
    #[arg(long)]
    pub hq_no_noise_floor: bool,

    /// Disable per-block temporal confidence weighting for the `hq`
    /// variant.
    ///
    /// By default HQ block-matches each temporal neighbour against the
    /// centre frame and lets a poor match suppress that neighbour's
    /// contribution, instead of blurring in occluded or changed
    /// content. Setting this applies temporal weights uniformly no
    /// matter how well a neighbour matches.
    ///
    /// Only takes effect when `--temporal-radius` is above 0.
    #[arg(long)]
    pub hq_no_temporal_confidence: bool,

    /// Multiplier on the per-block mismatch threshold temporal
    /// confidence weighting tolerates before a neighbour's contribution
    /// starts dropping.
    ///
    /// Higher values tolerate larger mismatches. Library default is
    /// 1.0. Ignored when `--hq-no-temporal-confidence` is set.
    #[arg(long)]
    pub hq_thsad_scale: Option<f32>,

    /// Nudges the automatically measured noise level up or down.
    ///
    /// `1.0` (the library default) keeps the measurement as-is. Raise
    /// it a little when the cleaned result still looks noisy. Lower
    /// it when detail is getting scrubbed.
    ///
    /// This differs from `--strength` because the noise level also
    /// sets the patch-distance noise floor and the motion-confidence
    /// floor, not just the weighting.
    ///
    /// Has no effect when `--hq-sigma` pins the noise level.
    #[arg(long)]
    pub hq_sigma_scale: Option<f32>,

    /// Turn on motion compensation for temporal denoising.
    ///
    /// When the camera or content moves between frames, the
    /// brightness at the same `(x, y)` is different content in each
    /// frame.
    ///
    /// Without help, temporal cleanup will blur moving edges.
    ///
    /// Motion compensation looks at where each block of pixels
    /// moved between frames, then shifts neighbour frames to line up
    /// with the current frame before cleaning.
    ///
    /// This keeps detail sharp on anime, fast pans, and action
    /// footage.
    ///
    /// The tracking strategy adapts automatically to `--temporal-radius`.
    ///
    /// The `--mc-*` flags only take effect with this set. Has no effect
    /// when `--temporal-radius 0`.
    #[arg(long)]
    pub motion_compensation: bool,

    /// Estimates noise from a local window instead of a temporal EMA
    /// over stream history.
    ///
    /// Experimental measurement switch for comparing the two estimators
    /// on real footage. Not a committed public interface. Off by
    /// default, which keeps the temporal EMA every calibrated preset
    /// assumes. Only applies to `--variant hq`, since `fast` never
    /// measures noise.
    #[arg(long, hide = true)]
    pub windowed_noise_estimation: bool,

    #[command(flatten)]
    pub motion: MotionArgs,
}

/// `--variant`, `--temporal-radius`, and `--search-radius` resolved
/// from either an explicit flag or the active `--preset`.
#[derive(Debug, Copy, Clone)]
pub struct ResolvedPreset {
    pub variant: Variant,
    pub temporal_radius: u32,
    pub search_radius: u32,
}

impl NlmeansArgs {
    /// Fills the unset dial-driven flags in from `preset`.
    pub fn resolve_preset(&self, preset: Preset) -> ResolvedPreset {
        ResolvedPreset {
            variant: self.variant.unwrap_or_else(|| nlmeans_variant_for(preset)),
            temporal_radius: self
                .temporal_radius
                .unwrap_or_else(|| nlmeans_temporal_radius_for(preset)),
            search_radius: self
                .search_radius
                .unwrap_or_else(|| nlmeans_search_radius_for(preset)),
        }
    }

    /// Builds the library's [`av_denoise::Algorithm`] from the resolved
    /// preset and the flags the chosen variant reads.
    ///
    /// Flags that are set but do nothing for this configuration are
    /// reported as warnings.
    pub fn resolve_algorithm(
        &self,
        resolved: ResolvedPreset,
        nlm: NlmeansOptions,
    ) -> Result<av_denoise::Algorithm, anyhow::Error> {
        let sigma_scale_is_set = self.hq_sigma_scale.is_some_and(|v| v != 1.0);

        match resolved.variant {
            Variant::Fast => {
                if self.windowed_noise_estimation {
                    anyhow::bail!(
                        "--windowed-noise-estimation has no effect on --variant fast, which \
                         never measures noise; select --variant hq instead"
                    );
                }

                if self.hq_sigma.is_some()
                    || self.hq_no_auto_strength
                    || self.hq_no_noise_floor
                    || self.hq_no_temporal_confidence
                    || self.hq_thsad_scale.is_some()
                    || sigma_scale_is_set
                {
                    tracing::warn!("--hq-* options are ignored unless --variant hq is selected");
                }
                Ok(av_denoise::Algorithm::Nlmeans(nlm))
            },
            Variant::Hq => {
                // Check the raw 8-bit value here so an out-of-range
                // `--hq-sigma` reports the number the user typed. The
                // library re-validates the same bound after the /255
                // normalisation, but its message speaks in [0, 1] units.
                if let Some(sigma) = self.hq_sigma
                    && (!sigma.is_finite() || sigma <= 0.0 || sigma > 255.0)
                {
                    anyhow::bail!("--hq-sigma must be a finite value in (0, 255] 8-bit units (got {sigma})");
                }

                if self.hq_sigma.is_some() && sigma_scale_is_set {
                    tracing::warn!("--hq-sigma-scale has no effect when --hq-sigma pins the noise level");
                }

                Ok(av_denoise::Algorithm::NlmeansHq(NlmeansHqOptions {
                    nlm,
                    hq: av_denoise::HqParams {
                        auto_strength: !self.hq_no_auto_strength,
                        noise_floor: !self.hq_no_noise_floor,
                        sigma_override: self.hq_sigma.map(|s| s / 255.0),
                        temporal_confidence: !self.hq_no_temporal_confidence,
                        thsad_scale: self.hq_thsad_scale.unwrap_or(1.0),
                        sigma_scale: self.hq_sigma_scale.unwrap_or(1.0),
                        // The CLI keeps the temporal EMA every
                        // calibrated preset assumes by default. Only
                        // `av-denoise-vs` needs window-local estimation,
                        // for random-access determinism.
                        // `--windowed-noise-estimation` exists to
                        // measure the difference on real footage.
                        windowed_noise_estimation: self.windowed_noise_estimation,
                    },
                }))
            },
        }
    }

    /// Turns the parsed flags plus the shared globals into the options
    /// the ingest pipeline takes.
    pub fn build_options(&self, globals: &Args) -> Result<RunOptions, anyhow::Error> {
        let resolved = self.resolve_preset(globals.preset);

        let mode = if resolved.temporal_radius == 0 {
            DenoisingMode::Spacial
        } else {
            DenoisingMode::Temporal {
                radius: resolved.temporal_radius,
            }
        };

        let prefilter = self
            .prefilter
            .as_deref()
            .map(parse_prefilter)
            .transpose()?
            .unwrap_or(PrefilterMode::None);
        let intent = resolve_channel_intent(&globals.channel_mode)?;

        let motion_compensation = if self.motion_compensation {
            if resolved.temporal_radius == 0 {
                tracing::warn!(
                    "--motion-compensation has no effect when --temporal-radius is 0; \
                     the spatial path doesn't use temporal neighbours"
                );
            }
            self.motion.to_motion_search().into()
        } else {
            if self.motion.any_set() {
                tracing::warn!("--mc-* options are ignored unless --motion-compensation is set");
            }
            MotionCompensationMode::None
        };

        // search_radius always has a resolved value (explicit flag or the
        // active preset), so it's always carried into the tuning override.
        let nlm = NlmeansOptions {
            prefilter,
            motion_compensation,
            tuning: NlmTuning {
                search_radius: Some(resolved.search_radius),
                patch_radius: self.patch_radius,
                strength: self.strength,
                self_weight: self.self_weight,
            },
        };

        Ok(RunOptions {
            planes: PlaneOptions {
                accelerators: globals.accelerators.clone(),
                device: globals.device.clone(),
                intent,
                mode,
                output_depth: self
                    .common
                    .output_depth
                    .map(|bits| Depth::from_bits(bits as usize))
                    .transpose()?,
                algorithm: self.resolve_algorithm(resolved, nlm)?,
                luma_strength: self.luma_strength,
                chroma_strength: self.chroma_strength,
                // `nlmeans` has no grouping stage, so these stay unset
                // here. `Nl4dArgs::build_options` fills them in afterwards.
                luma_lambda_ht: None,
                chroma_lambda_ht: None,
                luma_mismatch_scale: None,
                chroma_mismatch_scale: None,
            },
            progress: globals.progress,
        })
    }
}
