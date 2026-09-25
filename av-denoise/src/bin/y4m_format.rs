use av_denoise::{Depth, Subsampling};

/// Maps our [`Subsampling`] and [`Depth`] onto the [`y4m::Colorspace`]
/// used to read the input and write the output header.
pub fn subsampling_to_y4m(s: Subsampling, depth: Depth) -> y4m::Colorspace {
    match (s, depth) {
        (Subsampling::Yuv420, Depth::Eight) => y4m::Colorspace::C420,
        (Subsampling::Yuv420, Depth::Ten) => y4m::Colorspace::C420p10,
        (Subsampling::Yuv420, Depth::Twelve) => y4m::Colorspace::C420p12,
        (Subsampling::Yuv422, Depth::Eight) => y4m::Colorspace::C422,
        (Subsampling::Yuv422, Depth::Ten) => y4m::Colorspace::C422p10,
        (Subsampling::Yuv422, Depth::Twelve) => y4m::Colorspace::C422p12,
        (Subsampling::Yuv444, Depth::Eight) => y4m::Colorspace::C444,
        (Subsampling::Yuv444, Depth::Ten) => y4m::Colorspace::C444p10,
        (Subsampling::Yuv444, Depth::Twelve) => y4m::Colorspace::C444p12,
    }
}

/// Maps a [`y4m::Colorspace`] back onto our [`Subsampling`] and
/// [`Depth`].
///
/// Grayscale and any other unsupported colorspace are rejected with an
/// error naming what is required instead.
pub fn subsampling_from_y4m(c: y4m::Colorspace) -> Result<(Subsampling, Depth), anyhow::Error> {
    let sub = match c {
        y4m::Colorspace::C420
        | y4m::Colorspace::C420jpeg
        | y4m::Colorspace::C420paldv
        | y4m::Colorspace::C420mpeg2
        | y4m::Colorspace::C420p10
        | y4m::Colorspace::C420p12 => Subsampling::Yuv420,
        y4m::Colorspace::C422 | y4m::Colorspace::C422p10 | y4m::Colorspace::C422p12 => Subsampling::Yuv422,
        y4m::Colorspace::C444 | y4m::Colorspace::C444p10 | y4m::Colorspace::C444p12 => Subsampling::Yuv444,
        other => anyhow::bail!("unsupported y4m colorspace {other:?}, need 4:2:0, 4:2:2, or 4:4:4"),
    };

    let depth = Depth::from_bits(c.get_bit_depth())?;

    Ok((sub, depth))
}

/// Pulls the `X`-prefixed vendor extension params out of a decoded y4m
/// header's raw params bytes, `XCOLORRANGE=LIMITED` being the common one.
///
/// The leading `X` is stripped so the result can go straight to
/// [`y4m::EncoderBuilder::append_vendor_extension`], which adds the `X`
/// back when it writes the output header.
///
/// This is how whatever colorspace and range tags the source declared
/// reach the output instead of being dropped.
///
/// A token that [`y4m::VendorExtensionString`] rejects, which means one
/// containing a space, is skipped rather than failing the run.
pub fn y4m_vendor_extensions(raw_params: &[u8]) -> Vec<y4m::VendorExtensionString> {
    raw_params
        .split(|&b| b == b' ')
        .filter(|tok| tok.first() == Some(&b'X'))
        .filter_map(|tok| y4m::VendorExtensionString::new(tok[1..].to_vec()).ok())
        .collect()
}
