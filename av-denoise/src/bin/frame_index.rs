use std::collections::BTreeSet;

use av_decoders::Decoder;
use ffms2_sys::{FFMS_GetFrameInfo, FFMS_GetNumFrames, FFMS_GetTrackFromVideo};

/// One entry of the ffms2 video index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexEntry {
    /// Presentation timestamp, in the video track's time base.
    pub pts: i64,
    /// Whether ffms2 marked this entry as a keyframe.
    pub keyframe: bool,
}

/// Reads the video index behind `decoder`.
///
/// Returns `None` when `decoder` is not backed by ffms2, or when ffms2 declines to describe the
/// track, when this happens caller then keeps every frame.
pub fn read_index(decoder: &mut Decoder) -> Option<Vec<IndexEntry>> {
    let total = decoder.get_video_details().total_frames?;
    let source = decoder.get_ffms2_impl()?.video_source;

    // SAFETY: a live `Ffms2Decoder` holds a non-null video source, and
    // the track belongs to that source rather than to us, so it stays
    // valid for as long as the decoder does.
    let track = unsafe { FFMS_GetTrackFromVideo(source) };

    if track.is_null() {
        return None;
    }

    // SAFETY: `track` is non-null and outlives this call.
    let track_frames = unsafe { FFMS_GetNumFrames(track) };

    if track_frames < 0 {
        return None;
    }

    // `FFMS_GetFrameInfo` indexes the track's entries without checking the
    // bound, so the track's own count is what the read has to stay under.
    // The video properties describe the same track but are reported
    // separately, and a disagreement must not turn into a read past the end.
    let total = total.min(track_frames as usize);
    let mut index = Vec::with_capacity(total);

    for i in 0..total {
        // SAFETY: `track` is non-null, and `i` stays below the entry
        // count the track itself reports.
        let info = unsafe { FFMS_GetFrameInfo(track, i as i32) };

        if info.is_null() {
            return None;
        }

        // SAFETY: checked non-null just above.
        let info = unsafe { &*info };

        index.push(IndexEntry {
            pts: info.PTS,
            keyframe: info.KeyFrame != 0,
        });
    }

    Some(index)
}

/// Returns the index positions that carry no picture of their own.
pub fn phantom_indices(index: &[IndexEntry]) -> BTreeSet<usize> {
    let mut phantom = BTreeSet::new();

    if index.is_empty() {
        return phantom;
    }

    // Nothing before the first keyframe can be decoded, so ffms2
    // answers those positions with a repeat of the keyframe.
    let lead = index.iter().position(|e| e.keyframe).unwrap_or(0);

    // One leading picture is the ordinary case. More than that means the
    // index marks no keyframe for a stretch of the file, so say how much
    // is going rather than shortening the output quietly.
    if lead > 1 {
        tracing::warn!(
            dropped = lead,
            "the index marks no keyframe until entry {lead}, dropping every entry before it",
        );
    }

    phantom.extend(0..lead);

    let gaps: Vec<i64> = (lead + 1..index.len())
        .map(|i| index[i].pts.saturating_sub(index[i - 1].pts))
        .collect();

    if gaps.is_empty() {
        return phantom;
    }

    // The median gap is the clip's real frame spacing. A handful of
    // phantom entries cannot move it, however far apart they sit.
    let mut sorted = gaps.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];

    // Variable frame rate pacing puts real frames closer together than the
    // median, which is the same signature a phantom leaves. Telling the two
    // apart by timing only works on a clip that is otherwise regular, so a
    // clip that is not keeps every entry.
    let regular = gaps
        .iter()
        .filter(|&&gap| gap.saturating_sub(median).saturating_abs().saturating_mul(4) <= median)
        .count();

    if regular * 10 < gaps.len() * 9 {
        tracing::debug!(
            regular,
            gaps = gaps.len(),
            "frame spacing is too irregular to tell phantom entries from variable frame rate pacing",
        );

        return phantom;
    }

    // A phantom shares a timeline slot with the frame that follows it, landing just ahead of
    // that frame's timestamp. The entry to drop is therefore the earlier of the pair.
    // Doubling the gap rather than halving the median keeps the comparison exact,
    // which matters when a clip's time base is close enough to its frame rate that the median
    // gap is a single unit.
    for (offset, &gap) in gaps.iter().enumerate() {
        if gap.saturating_mul(2) < median {
            phantom.insert(lead + offset);
        }
    }

    phantom
}

/// Rewrites scene boundaries from ffms2 index space into the output
/// frame numbering that dropping `phantom` produces.
///
/// A boundary that lands on a dropped entry moves onto the next frame that survives,
/// which can leave it equal to its neighbour. Those collapse, because a scene cannot
/// start where the one before it does.
pub fn remap_scene_starts(starts: &[usize], phantom: &BTreeSet<usize>) -> Vec<usize> {
    let mut out: Vec<usize> = starts
        .iter()
        .map(|&raw| raw - phantom.range(..raw).count())
        .collect();

    out.dedup();
    out
}
