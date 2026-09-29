use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;

use av_decoders::{Decoder, Rational32};
use av_denoise_core::{Depth, FrameLayout, Subsampling};

use crate::layout::{merge_adjacent_ranges, subsampling_from_av_decoders};
use crate::source::SourceIndex;
use crate::{FrameRange, SceneLayout, emitted_boundary_to_raw, validate_keep_frames};

/// Raw decoder frame ranges, end exclusive.
type SourceRanges = Vec<(usize, usize)>;

const SCENE_LAYOUT_SCHEMA: u32 = 1;
const SCENE_DETECTOR_ID: &str = "av-scenechange-0.23-default-chunked-1080p";

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredSceneLayout {
    schema: u32,
    detector: String,
    width: u32,
    height: u32,
    subsampling: String,
    depth: u8,
    frame_rate_numerator: i32,
    frame_rate_denominator: i32,
    total_frames: usize,
    raw_frames: usize,
    phantom: BTreeSet<usize>,
    scene_starts: Vec<usize>,
    #[serde(default)]
    keep_frames: Vec<StoredFrameRange>,
    #[serde(default)]
    source_ranges: Vec<(usize, usize)>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StoredFrameRange {
    start: usize,
    end: usize,
}

fn schema_for(keep_frames: &[FrameRange]) -> u32 {
    if keep_frames.is_empty() {
        SCENE_LAYOUT_SCHEMA
    } else {
        2
    }
}

impl StoredSceneLayout {
    fn from_layout(layout: &SceneLayout) -> Self {
        Self {
            schema: schema_for(&layout.keep_frames),
            detector: SCENE_DETECTOR_ID.into(),
            width: layout.layout.width,
            height: layout.layout.height,
            subsampling: format!("{:?}", layout.layout.subsampling),
            depth: layout.layout.depth.bits() as u8,
            frame_rate_numerator: *layout.framerate.numer(),
            frame_rate_denominator: *layout.framerate.denom(),
            total_frames: layout.total_frames,
            raw_frames: layout.raw_frames,
            phantom: layout.phantom.clone(),
            scene_starts: layout.scene_starts.clone(),
            keep_frames: layout
                .keep_frames
                .iter()
                .map(|range| StoredFrameRange {
                    start: range.start,
                    end: range.end,
                })
                .collect(),
            source_ranges: layout.source_ranges.clone(),
        }
    }

    fn raw_ranges(&self, ranges: &[FrameRange]) -> SourceRanges {
        ranges
            .iter()
            .map(|range| {
                (
                    emitted_boundary_to_raw(range.start, self.raw_frames, &self.phantom),
                    emitted_boundary_to_raw(range.end, self.raw_frames, &self.phantom),
                )
            })
            .collect()
    }

    /// Checks this layout was scanned from the source `decoder` reads and returns the
    /// source's frame layout and rate.
    fn check_source(&self, decoder: &Decoder) -> Result<(FrameLayout, Rational32), anyhow::Error> {
        let details = *decoder.get_video_details();
        let layout = FrameLayout {
            width: details.width as u32,
            height: details.height as u32,
            subsampling: subsampling_from_av_decoders(details.chroma_sampling)?,
            depth: Depth::from_bits(details.bit_depth)?,
        };
        let actual_subsampling = format!("{:?}", layout.subsampling);
        if self.width != layout.width
            || self.height != layout.height
            || self.subsampling != actual_subsampling
            || self.depth != layout.depth.bits() as u8
            || self.frame_rate_numerator != *details.frame_rate.numer()
            || self.frame_rate_denominator != *details.frame_rate.denom()
            || details
                .total_frames
                .is_some_and(|frames| self.raw_frames != frames)
        {
            anyhow::bail!("scene layout does not match the opened source video");
        }
        Ok((layout, details.frame_rate))
    }

    /// Checks the stored keep frames against the requested ones and
    /// returns the merged keep frames with their raw source ranges.
    fn resolve_keep_frames(
        &self,
        keep_frames: &[FrameRange],
    ) -> Result<(Vec<FrameRange>, SourceRanges), anyhow::Error> {
        let stored_keep = self
            .keep_frames
            .iter()
            .map(|range| FrameRange {
                start: range.start,
                end: range.end,
            })
            .collect::<Vec<_>>();
        let source_total = self
            .raw_frames
            .checked_sub(self.phantom.len())
            .ok_or_else(|| anyhow::anyhow!("scene layout contains invalid phantom frames"))?;
        if !stored_keep.is_empty() {
            validate_keep_frames(&stored_keep, source_total)?;
        }
        if !keep_frames.is_empty() {
            validate_keep_frames(keep_frames, source_total)?;
        }
        let stored_ranges = self.raw_ranges(&stored_keep);
        let stored_keep = merge_adjacent_ranges(&stored_keep);
        let requested_keep = merge_adjacent_ranges(keep_frames);
        let expected_ranges = self.raw_ranges(&stored_keep);
        let expected_total = if stored_keep.is_empty() {
            source_total
        } else {
            stored_keep.iter().map(|range| range.end - range.start).sum()
        };
        if stored_keep != requested_keep {
            anyhow::bail!(
                "scene layout keep frames {stored_keep:?} do not match requested {requested_keep:?}"
            );
        }
        if !stored_keep.is_empty() && self.source_ranges != stored_ranges {
            anyhow::bail!(
                "scene layout source ranges {:?} do not match its keep frames {stored_ranges:?}",
                self.source_ranges
            );
        }
        if self.total_frames != expected_total
            || self.total_frames == 0
            || self.scene_starts.first() != Some(&0)
            || self.scene_starts.last() != Some(&self.total_frames)
            || self.scene_starts.windows(2).any(|pair| pair[0] >= pair[1])
            || self.phantom.iter().any(|index| *index >= self.raw_frames)
        {
            anyhow::bail!("scene layout contains invalid frame boundaries");
        }
        let source_ranges = if stored_keep.is_empty() {
            vec![(0, self.raw_frames)]
        } else {
            expected_ranges
        };
        Ok((stored_keep, source_ranges))
    }
}

pub(crate) struct UncheckedLayout {
    stored: StoredSceneLayout,
    keep_frames: Vec<FrameRange>,
}

impl UncheckedLayout {
    pub(crate) fn load(path: &Path, keep_frames: &[FrameRange]) -> Result<Self, anyhow::Error> {
        let stored: StoredSceneLayout = serde_json::from_slice(&std::fs::read(path)?)?;
        if stored.schema != schema_for(keep_frames) || stored.detector != SCENE_DETECTOR_ID {
            anyhow::bail!("scene layout schema or detector does not match this av-denoise build");
        }
        Ok(Self {
            stored,
            keep_frames: keep_frames.to_vec(),
        })
    }

    pub(crate) fn frame_layout(&self) -> Result<FrameLayout, anyhow::Error> {
        let subsampling = [Subsampling::Yuv420, Subsampling::Yuv422, Subsampling::Yuv444]
            .into_iter()
            .find(|candidate| format!("{candidate:?}") == self.stored.subsampling)
            .ok_or_else(|| {
                anyhow::anyhow!("scene layout has unknown subsampling {}", self.stored.subsampling)
            })?;
        Ok(FrameLayout {
            width: self.stored.width,
            height: self.stored.height,
            subsampling,
            depth: Depth::from_bits(self.stored.depth as usize)?,
        })
    }

    pub(crate) fn check(self, decoder: &Decoder) -> Result<SceneLayout, anyhow::Error> {
        let stored = self.stored;
        let (layout, framerate) = stored.check_source(decoder)?;
        let (keep_frames, source_ranges) = stored.resolve_keep_frames(&self.keep_frames)?;
        Ok(SceneLayout {
            layout,
            framerate,
            total_frames: stored.total_frames,
            raw_frames: stored.raw_frames,
            phantom: stored.phantom,
            scene_starts: stored.scene_starts,
            source_ranges,
            keep_frames,
        })
    }
}

impl SceneLayout {
    /// Loads a layout `scenes` stored for `input`, checking it against
    /// the source and the requested keep frames.
    pub fn read(input: &Path, path: &Path, keep_frames: &[FrameRange]) -> Result<Self, anyhow::Error> {
        let unchecked = UncheckedLayout::load(path, keep_frames)?;
        unchecked.check(&SourceIndex::open(input)?.decoder()?)
    }

    /// Stores this layout at `output`, replacing it atomically.
    pub fn write(&self, output: &Path) -> Result<(), anyhow::Error> {
        let bytes = serde_json::to_vec(&StoredSceneLayout::from_layout(self))?;
        let name = output
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("scene layout output has no file name"))?;
        let temporary =
            output.with_file_name(format!(".{}.{}.tmp", name.to_string_lossy(), std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.into());
        }
        drop(file);
        if let Err(error) = std::fs::rename(&temporary, output) {
            let _ = std::fs::remove_file(&temporary);
            return Err(error.into());
        }
        Ok(())
    }
}
