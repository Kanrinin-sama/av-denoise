use super::{ChannelIntent, FrameLayout, PlaneOptions};
use crate::backend::EngineSpec;
use av_denoise_core::ChannelMode;

pub fn nl4d_vram_requirement(options: &PlaneOptions, layout: FrameLayout) -> Result<u64, anyhow::Error> {
    options.intent.validate_for_source(layout)?;
    let (chroma_width, chroma_height) = layout.chroma_dims();
    let planes = match options.intent {
        ChannelIntent::Luma => vec![(ChannelMode::Luma, layout.width, layout.height)],
        ChannelIntent::Chroma => vec![(ChannelMode::Chroma, chroma_width, chroma_height)],
        ChannelIntent::LumaChroma => vec![
            (ChannelMode::Luma, layout.width, layout.height),
            (ChannelMode::Chroma, chroma_width, chroma_height),
        ],
        ChannelIntent::YuvFused => vec![(ChannelMode::Yuv, layout.width, layout.height)],
    };
    let mut bytes = 0u64;
    for (channels, width, height) in planes {
        let denoiser = options.denoiser_options(channels, layout.depth);
        let EngineSpec::Nl4d { options, geometry } = denoiser.algorithm.engine_spec(&denoiser, width, height)
        else {
            anyhow::bail!("VRAM requirement is only available for NL4D");
        };
        let required = av_denoise_core::nl4d::vram_requirement(options, geometry)?;
        bytes = bytes
            .checked_add(required)
            .ok_or_else(|| anyhow::anyhow!("planar NL4D VRAM requirement exceeds u64"))?;
    }
    Ok(bytes)
}
