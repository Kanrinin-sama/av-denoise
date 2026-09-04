use std::io::{Read, Write, stdout};

use av_denoise::{FrameLayout, PlaneOptions, Planes, push_needs_retry};

use crate::warm_start::{create_denoiser, finish_warm_up};
use crate::y4m_format::{subsampling_from_y4m, subsampling_to_y4m, y4m_vendor_extensions};

/// Denoises a y4m stream frame by frame, writing y4m on stdout.
///
/// There is no scene detection here, so the temporal window slides
/// across the whole stream.
pub fn run_stream<R: Read>(opts: &PlaneOptions, reader: R) -> Result<(), anyhow::Error> {
    let mut decoder = y4m::Decoder::new(reader)?;

    let (subsampling, depth) = subsampling_from_y4m(decoder.get_colorspace())?;

    let layout = FrameLayout {
        width: decoder.get_width() as u32,
        height: decoder.get_height() as u32,
        subsampling,
        depth,
    };

    let framerate = decoder.get_framerate();
    let pixel_aspect = decoder.get_pixel_aspect();
    let colorspace = subsampling_to_y4m(layout.subsampling, layout.depth);

    let stdout = stdout();
    let mut builder = y4m::encode(layout.width as usize, layout.height as usize, framerate)
        .with_colorspace(colorspace)
        .with_pixel_aspect(pixel_aspect);
    // Forward the source's `X` extension params (e.g. `XCOLORRANGE=`)
    // verbatim instead of silently dropping them.
    for ext in y4m_vendor_extensions(decoder.get_raw_params()) {
        builder = builder.append_vendor_extension(ext);
    }
    let mut encoder = builder.write_header(stdout.lock())?;

    let (mut wd, mut warm_up) = create_denoiser(opts, layout)?;

    tracing::info!(
        accelerator = ?opts.accelerators,
        width = layout.width,
        height = layout.height,
        subsampling = ?layout.subsampling,
        depth = ?layout.depth,
        "streaming pipeline ready",
    );

    loop {
        let frame = match decoder.read_frame() {
            Ok(f) => f,
            Err(y4m::Error::EOF) => break,
            Err(e) => return Err(e.into()),
        };

        let planes = Planes {
            y: frame.get_y_plane().to_vec(),
            u: frame.get_u_plane().to_vec(),
            v: frame.get_v_plane().to_vec(),
        };

        if push_needs_retry(wd.push(&planes))? {
            if let Some(out) = wd.recv()? {
                write_planes(&mut encoder, &out)?;
                finish_warm_up(&mut warm_up);
            }

            wd.push(&planes)?;
        }

        if let Some(out) = wd.recv()? {
            write_planes(&mut encoder, &out)?;
            finish_warm_up(&mut warm_up);
        }
    }

    wd.flush(|out| {
        if let Err(e) = write_planes(&mut encoder, &out) {
            tracing::error!(error = ?e, "failed to write flushed frame");
        }
        finish_warm_up(&mut warm_up);
    })?;

    Ok(())
}

fn write_planes<W: Write>(encoder: &mut y4m::Encoder<W>, planes: &Planes) -> Result<(), anyhow::Error> {
    let frame = y4m::Frame::new([&planes.y, &planes.u, &planes.v], None);
    encoder.write_frame(&frame)?;

    Ok(())
}
