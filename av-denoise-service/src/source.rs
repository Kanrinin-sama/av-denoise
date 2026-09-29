use std::path::{Path, PathBuf};

use av_decoders::Decoder;

pub(crate) fn open_source(input: &Path) -> Result<Decoder, anyhow::Error> {
    let index_path = PathBuf::from(format!("{}.ffindex", input.to_string_lossy()));
    let index_existed = index_path.try_exists()?;
    let decoder = Decoder::from_file(input);
    if !index_existed {
        let _ = std::fs::remove_file(index_path);
    }
    Ok(decoder?)
}

pub(crate) struct OpenedSource(pub(crate) Decoder);

unsafe impl Send for OpenedSource {}
