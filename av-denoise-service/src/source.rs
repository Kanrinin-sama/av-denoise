use std::path::{Path, PathBuf};
use std::sync::Arc;

use av_decoders::{Decoder, DecoderImpl, Ffms2Decoder, FfmsIndex};

pub(crate) struct SourceIndex {
    index: Arc<FfmsIndex>,
    created: Option<PathBuf>,
}

impl SourceIndex {
    pub(crate) fn open(input: &Path) -> Result<Self, anyhow::Error> {
        let index_path = PathBuf::from(format!("{}.ffindex", input.to_string_lossy()));
        let index_existed = index_path.try_exists()?;
        let index = Ffms2Decoder::index(input);
        let created = (!index_existed).then_some(index_path);
        let index = match index {
            Ok(index) => index,
            Err(error) => {
                if let Some(created) = created {
                    let _ = std::fs::remove_file(created);
                }
                return Err(error.into());
            },
        };
        Ok(Self {
            index: Arc::new(index),
            created,
        })
    }

    pub(crate) fn decoder(&self) -> Result<Decoder, anyhow::Error> {
        let decoder = Ffms2Decoder::with_index(Arc::clone(&self.index))?;
        Ok(Decoder::from_decoder_impl(DecoderImpl::Ffms2(decoder))?)
    }
}

impl Drop for SourceIndex {
    fn drop(&mut self) {
        if let Some(created) = &self.created {
            let _ = std::fs::remove_file(created);
        }
    }
}

pub(crate) struct OpenedSource(pub(crate) Decoder);

unsafe impl Send for OpenedSource {}
