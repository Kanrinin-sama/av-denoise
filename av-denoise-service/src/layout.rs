use std::collections::BTreeSet;

use av_decoders::Rational32;
use av_denoise_core::{FrameLayout, Subsampling};

use crate::FrameRange;

/// Scene boundaries plus the video metadata needed to build the output
/// frames.
#[derive(Clone)]
pub struct SceneLayout {
    pub layout: FrameLayout,
    pub framerate: Rational32,
    /// Frames this run emits, being `raw_frames` less the phantom entries.
    pub total_frames: usize,
    /// Frames the decoder hands over, phantom entries included. Every
    /// one has to be read to keep the sequential decoder in step, even
    /// though only `total_frames` of them are emitted.
    pub raw_frames: usize,
    /// Decoder frame numbers that carry no picture of their own.
    pub phantom: BTreeSet<usize>,
    /// `scene_starts[i]` is the inclusive start frame of scene `i`, in  emitted frame numbers.
    /// The final entry is `total_frames` so scene `i` covers `scene_starts[i]..scene_starts[i + 1]`.
    pub scene_starts: Vec<usize>,
    pub source_ranges: Vec<(usize, usize)>,
    pub keep_frames: Vec<FrameRange>,
}

impl SceneLayout {
    pub fn scene_count(&self) -> usize {
        self.scene_starts.len() - 1
    }

    pub fn select_span(mut self, span: FrameRange) -> Result<Self, anyhow::Error> {
        if self.scene_starts.binary_search(&span.start).is_err()
            || self.scene_starts.binary_search(&span.end).is_err()
        {
            anyhow::bail!("--scene-span must match retained scene boundaries");
        }

        let mut source_ranges = Vec::new();
        let full_range = [FrameRange {
            start: 0,
            end: self.total_frames,
        }];
        let selected_ranges = if self.keep_frames.is_empty() {
            full_range.as_slice()
        } else {
            self.keep_frames.as_slice()
        };
        let mut compact_start = 0usize;
        for range in selected_ranges {
            let compact_end = compact_start + range.end - range.start;
            let selected_start = span.start.max(compact_start);
            let selected_end = span.end.min(compact_end);
            if selected_start < selected_end {
                let source_start = range.start + selected_start - compact_start;
                let source_end = range.start + selected_end - compact_start;
                source_ranges.push((
                    emitted_boundary_to_raw(source_start, self.raw_frames, &self.phantom),
                    emitted_boundary_to_raw(source_end, self.raw_frames, &self.phantom),
                ));
            }
            compact_start = compact_end;
        }

        self.scene_starts = self
            .scene_starts
            .into_iter()
            .filter(|boundary| *boundary >= span.start && *boundary <= span.end)
            .map(|boundary| boundary - span.start)
            .collect();
        self.total_frames = span.end - span.start;
        self.source_ranges = source_ranges;
        Ok(self)
    }
}

pub(crate) fn merge_adjacent_ranges(ranges: &[FrameRange]) -> Vec<FrameRange> {
    let mut merged: Vec<FrameRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut()
            && previous.end == range.start
        {
            previous.end = range.end;
        } else {
            merged.push(*range);
        }
    }
    merged
}

pub fn validate_keep_frames(ranges: &[FrameRange], total: usize) -> Result<(), anyhow::Error> {
    if ranges.is_empty()
        || ranges
            .iter()
            .any(|range| range.start >= range.end || range.end > total)
        || ranges.windows(2).any(|pair| pair[0].end > pair[1].start)
    {
        anyhow::bail!("--keep-frames ranges must be sorted, non-overlapping, and within the source");
    }
    Ok(())
}

pub fn emitted_boundary_to_raw(emitted: usize, raw_total: usize, phantom: &BTreeSet<usize>) -> usize {
    let mut seen = 0usize;
    for raw in 0..raw_total {
        if seen == emitted {
            return raw;
        }
        if !phantom.contains(&raw) {
            seen += 1;
        }
    }
    raw_total
}

pub fn subsampling_from_av_decoders(
    cs: v_frame::chroma::ChromaSubsampling,
) -> Result<Subsampling, anyhow::Error> {
    use v_frame::chroma::ChromaSubsampling;

    match cs {
        ChromaSubsampling::Yuv420 => Ok(Subsampling::Yuv420),
        ChromaSubsampling::Yuv422 => Ok(Subsampling::Yuv422),
        ChromaSubsampling::Yuv444 => Ok(Subsampling::Yuv444),
        other => {
            anyhow::bail!("unsupported chroma subsampling {other:?}, need 4:2:0, 4:2:2, or 4:4:4")
        },
    }
}
