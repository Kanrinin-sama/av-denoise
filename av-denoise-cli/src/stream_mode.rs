use std::io::{IsTerminal, Read, Write, stdout};

use av_denoise::{FrameLayout, Planes, SceneGrain, push_needs_retry};
use av_denoise_service::{create_denoiser, finish_warm_up};

use crate::cli::RunOptions;
use crate::grain_table::TablePath;
use crate::progress::{self, denoise_bar_visible, denoise_progress_bar};
use crate::y4m_format::{create_encoder, subsampling_from_y4m, y4m_vendor_extensions};

/// Denoises a y4m stream frame by frame, writing y4m on stdout.
///
/// There is no scene detection here, so the temporal window slides
/// across the whole stream.
pub fn run_stream<R: Read>(opts: &RunOptions, reader: R) -> Result<(), anyhow::Error> {
    let table = opts.grain_table.as_deref().map(TablePath::create).transpose()?;
    let visible = denoise_bar_visible(opts.progress, std::io::stderr().is_terminal());
    let opts = &opts.planes;
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
    let output_depth = opts.output_depth.unwrap_or(layout.depth);

    let stdout = stdout();
    let mut encoder = create_encoder(
        FrameLayout {
            depth: output_depth,
            ..layout
        },
        framerate,
        Some(pixel_aspect),
        y4m_vendor_extensions(decoder.get_raw_params()),
        stdout.lock(),
    )?;
    let pb = denoise_progress_bar(None, visible);

    let (mut wd, mut warm_up) = create_denoiser(opts, layout)?;

    tracing::info!(
        accelerator = ?opts.accelerators,
        width = layout.width,
        height = layout.height,
        subsampling = ?layout.subsampling,
        source_depth = ?layout.depth,
        ?output_depth,
        "streaming pipeline ready",
    );

    loop {
        let frame = match decoder.read_frame() {
            Ok(f) => f,
            Err(y4m::Error::EOF) => break,
            Err(e) => return Err(e.into()),
        };

        let planes = Planes {
            device: None,
            y: frame.get_y_plane().to_vec(),
            u: frame.get_u_plane().to_vec(),
            v: frame.get_v_plane().to_vec(),
        };

        if push_needs_retry(wd.push(&planes))? {
            if let Some(out) = wd.recv()? {
                write_planes(&mut encoder, &out)?;
                pb.inc(1);
                finish_warm_up(&mut warm_up);
            }

            wd.push(&planes)?;
        }

        if let Some(out) = wd.recv()? {
            write_planes(&mut encoder, &out)?;
            pb.inc(1);
            finish_warm_up(&mut warm_up);
        }
    }

    let mut flush_error = None;
    wd.flush(|out| {
        if flush_error.is_some() {
            return;
        }
        match write_planes(&mut encoder, &out) {
            Ok(()) => pb.inc(1),
            Err(error) => flush_error = Some(error),
        }
        finish_warm_up(&mut warm_up);
    })?;
    progress::finish(&pb);
    if let Some(error) = flush_error {
        return Err(error);
    }
    if let Some(table) = table {
        let chunks = wd.drain_grain_chunks()?;
        table.write(
            &[SceneGrain {
                first_frame: 0,
                chunks,
            }],
            (framerate.num as u64, framerate.den as u64),
        )?;
    }

    Ok(())
}

fn write_planes<W: Write>(encoder: &mut y4m::Encoder<W>, planes: &Planes) -> Result<(), anyhow::Error> {
    let frame = y4m::Frame::new([&planes.y, &planes.u, &planes.v], None);
    encoder.write_frame(&frame)?;

    Ok(())
}
