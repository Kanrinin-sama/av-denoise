use av_decoders::Decoder;
use av_denoise_core::{Depth, FrameLayout, Planes};

/// Checks each plane's byte length against the layout, failing with an error naming which plane
/// is wrong, the length found and the length expected.
fn check_plane_lens(planes: &Planes, layout: FrameLayout) -> Result<(), anyhow::Error> {
    for (name, got, expected) in [
        ("y", planes.y.len(), layout.luma_bytes()),
        ("u", planes.u.len(), layout.chroma_bytes()),
        ("v", planes.v.len(), layout.chroma_bytes()),
    ] {
        if got != expected {
            anyhow::bail!("{name} plane is {got} bytes, expected {expected} from the frame layout");
        }
    }
    Ok(())
}

/// Decodes the next frame into `planes`, reusing their allocations.
///
/// A chroma plane the decoder leaves empty, which is how a luma-only or
/// monochrome read reports no chroma, comes back neutral at the layout's
/// depth instead.
pub(crate) fn decode_into(
    decoder: &mut Decoder,
    layout: FrameLayout,
    planes: &mut Planes,
) -> Result<(), anyhow::Error> {
    decoder.read_video_planes_into(&mut planes.y, &mut planes.u, &mut planes.v)?;

    if planes.u.is_empty() {
        fill_neutral(&mut planes.u, layout.chroma_pixels(), layout.depth);
    }
    if planes.v.is_empty() {
        fill_neutral(&mut planes.v, layout.chroma_pixels(), layout.depth);
    }

    check_plane_lens(planes, layout)?;

    Ok(())
}

fn fill_neutral(dst: &mut Vec<u8>, samples: usize, depth: Depth) {
    let neutral = depth.neutral_chroma();
    if depth.bytes_per_sample() == 1 {
        dst.resize(samples, neutral as u8);
        return;
    }
    dst.resize(samples * 2, 0);
    for chunk in dst.as_chunks_mut::<2>().0 {
        *chunk = neutral.to_le_bytes();
    }
}
