use std::collections::BTreeSet;

use av_decoders::Decoder;
use ffms2_sys::{FFMS_GetFrameInfo, FFMS_GetTrackFromVideo};

pub(crate) struct IndexEntry {
    pts: i64,
    keyframe: bool,
}

pub(crate) fn read_index(decoder: &mut Decoder) -> Option<Vec<IndexEntry>> {
    let total = decoder.get_video_details().total_frames?;
    let source = decoder.get_ffms2_impl()?.video_source;
    let track = unsafe { FFMS_GetTrackFromVideo(source) };

    if track.is_null() {
        return None;
    }

    let mut index = Vec::with_capacity(total);

    for frame_no in 0..total {
        let info = unsafe { FFMS_GetFrameInfo(track, frame_no as i32) };

        if info.is_null() {
            return None;
        }

        let info = unsafe { &*info };
        index.push(IndexEntry {
            pts: info.PTS,
            keyframe: info.KeyFrame != 0,
        });
    }

    Some(index)
}

pub(crate) fn phantom_indices(index: &[IndexEntry]) -> BTreeSet<usize> {
    let mut phantom = BTreeSet::new();

    if index.is_empty() {
        return phantom;
    }

    let first_keyframe = index.iter().position(|entry| entry.keyframe).unwrap_or(0);
    phantom.extend(0..first_keyframe);

    let mut gaps: Vec<i64> = (first_keyframe + 1..index.len())
        .map(|frame_no| index[frame_no].pts - index[frame_no - 1].pts)
        .collect();

    if gaps.is_empty() {
        return phantom;
    }

    gaps.sort_unstable();
    let threshold = gaps[gaps.len() / 2] / 2;

    for frame_no in first_keyframe + 1..index.len() {
        if index[frame_no].pts - index[frame_no - 1].pts < threshold {
            phantom.insert(frame_no - 1);
        }
    }

    phantom
}
