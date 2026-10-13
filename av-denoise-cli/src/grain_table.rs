use std::fs::copy;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use av_denoise::{SceneGrain, build_table};
use tempfile::NamedTempFile;

pub struct TablePath {
    path: PathBuf,
    temporary: NamedTempFile,
}

impl TablePath {
    pub fn create(path: &Path) -> Result<Self, anyhow::Error> {
        let temporary = NamedTempFile::new()
            .with_context(|| format!("unable to create the grain table at {}", path.display()))?;

        Ok(Self {
            path: path.to_path_buf(),
            temporary,
        })
    }

    /// Writes the table and moves it into place.
    pub fn write(mut self, scenes: &[SceneGrain], frame_rate: (u64, u64)) -> Result<(), anyhow::Error> {
        let text = build_table(scenes, frame_rate);
        if text == "filmgrn1\n" {
            tracing::warn!("no grain could be measured, writing an empty grain table");
        }

        self.temporary
            .write_all(text.as_bytes())
            .with_context(|| format!("unable to write the grain table at {}", self.path.display()))?;
        self.temporary
            .as_file_mut()
            .sync_all()
            .with_context(|| format!("unable to write the grain table at {}", self.path.display()))?;
        copy(self.temporary.path(), &self.path)
            .with_context(|| format!("unable to move the grain table to {}", self.path.display()))?;

        Ok(())
    }
}
