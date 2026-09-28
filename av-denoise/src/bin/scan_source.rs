use std::ffi::{CString, c_char};
use std::num::{NonZeroU8, NonZeroUsize};
use std::path::Path;

use av_decoders::VideoDetails;
use ffms2_sys::{
    FFMS_CreateVideoSource,
    FFMS_DestroyIndex,
    FFMS_DestroyVideoSource,
    FFMS_ErrorInfo,
    FFMS_Frame,
    FFMS_GetFirstIndexedTrackOfType,
    FFMS_GetFrame,
    FFMS_GetPixFmt,
    FFMS_Index,
    FFMS_ReadIndex,
    FFMS_Resizers,
    FFMS_SetOutputFormatV2,
    FFMS_TrackType,
    FFMS_VideoSource,
};
use v_frame::chroma::ChromaSubsampling;
use v_frame::frame::{Frame, FrameBuilder};
use v_frame::pixel::Pixel;

use crate::nvdec_source::NvdecSource;

/// Largest frame scene analysis runs at. Bigger sources are scaled to fit.
const ANALYSIS_BOUNDS: (usize, usize) = (1920, 1080);

/// The padding av-decoders gives every frame it hands the detector.
const LUMA_PADDING: usize = 64 + 16 + 8;

/// The picture format scene analysis sees, which is the source's own
/// format scaled to fit [`ANALYSIS_BOUNDS`].
#[derive(Clone, Copy)]
pub struct ScanFormat {
    pub width: usize,
    pub height: usize,
    pub bit_depth: usize,
    pub chroma: ChromaSubsampling,
    pub scaled: bool,
}

impl ScanFormat {
    pub fn new(details: &VideoDetails) -> Self {
        let (width, height) = (details.width, details.height);
        let (max_width, max_height) = ANALYSIS_BOUNDS;
        let (scaled_width, scaled_height) = if width <= max_width && height <= max_height {
            (width, height)
        } else if width * max_height >= height * max_width {
            (max_width, height * max_width / width)
        } else {
            (width * max_height / height, max_height)
        };

        Self {
            width: scaled_width & !1,
            height: scaled_height & !1,
            bit_depth: details.bit_depth,
            chroma: details.chroma_sampling,
            scaled: scaled_width != width || scaled_height != height,
        }
    }

    pub(crate) fn pixel_format(&self) -> String {
        let chroma = match self.chroma {
            ChromaSubsampling::Yuv422 => "422",
            ChromaSubsampling::Yuv444 => "444",
            _ => "420",
        };
        let depth = if self.bit_depth > 8 {
            format!("{}le", self.bit_depth)
        } else {
            String::new()
        };

        format!("yuv{chroma}p{depth}")
    }

    pub(crate) fn plane_sizes(&self) -> [(usize, usize); 3] {
        let (x, y) = self
            .chroma
            .subsample_ratio()
            .map_or((1, 1), |(x, y)| (x.get() as usize, y.get() as usize));
        let chroma = (self.width.div_ceil(x), self.height.div_ceil(y));

        [(self.width, self.height), chroma, chroma]
    }

    pub(crate) fn frame_bytes(&self) -> usize {
        let sample = if self.bit_depth > 8 { 2 } else { 1 };

        self.plane_sizes().iter().map(|(w, h)| w * h * sample).sum()
    }

    pub(crate) fn blank_frame<T: Pixel>(&self) -> Result<Frame<T>, anyhow::Error> {
        let frame = FrameBuilder::new(
            NonZeroUsize::new(self.width).ok_or_else(|| anyhow::anyhow!("zero analysis width"))?,
            NonZeroUsize::new(self.height).ok_or_else(|| anyhow::anyhow!("zero analysis height"))?,
            self.chroma,
            NonZeroU8::new(self.bit_depth as u8).ok_or_else(|| anyhow::anyhow!("zero bit depth"))?,
        )
        .luma_padding_left(LUMA_PADDING)
        .luma_padding_right(LUMA_PADDING)
        .luma_padding_top(LUMA_PADDING)
        .luma_padding_bottom(LUMA_PADDING)
        .build()?;

        Ok(frame)
    }
}

/// Decodes one stretch of the source for scene analysis, starting at a
/// raw ffms2 frame number and yielding every raw frame after it in order.
pub enum ScanSource {
    Ffms(FfmsSource),
    Nvdec(NvdecSource),
}

impl ScanSource {
    pub fn read<T: Pixel>(&mut self, format: &ScanFormat) -> Result<Frame<T>, anyhow::Error> {
        match self {
            Self::Ffms(source) => source.read(format),
            Self::Nvdec(source) => source.read(format),
        }
    }

    /// Confirms the source had exactly the frames it was asked for.
    pub fn finish(self) -> Result<(), anyhow::Error> {
        match self {
            Self::Ffms(_) => Ok(()),
            Self::Nvdec(source) => source.finish(),
        }
    }
}

/// ffms2's software decoder, reading the index the scan's metadata pass left on disk.
pub struct FfmsSource {
    index: *mut FFMS_Index,
    source: *mut FFMS_VideoSource,
    next: usize,
}

// SAFETY: the handles are owned by this value alone and ffms2 does not tie
// them to the thread that created them.
unsafe impl Send for FfmsSource {}

impl FfmsSource {
    pub fn open(
        input: &Path,
        format: &ScanFormat,
        threads: usize,
        start: usize,
    ) -> Result<Self, anyhow::Error> {
        let input_path = CString::new(input.to_string_lossy().as_bytes())?;
        let index_path = CString::new(format!("{}.ffindex", input.to_string_lossy()))?;
        let mut buffer = [0 as c_char; 1024];
        let mut error = FFMS_ErrorInfo {
            ErrorType: 0,
            SubType: 0,
            BufferSize: buffer.len() as i32,
            Buffer: buffer.as_mut_ptr(),
        };
        let failure = |error: &FFMS_ErrorInfo| {
            // SAFETY: ffms2 writes a NUL-terminated message into the buffer it was given.
            let message = unsafe { std::ffi::CStr::from_ptr(error.Buffer) };
            anyhow::anyhow!("ffms2: {}", message.to_string_lossy())
        };

        // SAFETY: both strings are NUL-terminated and outlive the calls, and
        // every handle is checked before use and released by `Drop`.
        unsafe {
            let index = FFMS_ReadIndex(index_path.as_ptr(), &mut error);
            if index.is_null() {
                return Err(failure(&error));
            }
            let mut opened = Self {
                index,
                source: std::ptr::null_mut(),
                next: start,
            };
            let track =
                FFMS_GetFirstIndexedTrackOfType(index, FFMS_TrackType::FFMS_TYPE_VIDEO as i32, &mut error);
            opened.source =
                FFMS_CreateVideoSource(input_path.as_ptr(), track, index, threads as i32, 1, &mut error);
            if opened.source.is_null() {
                return Err(failure(&error));
            }
            if format.scaled {
                let pixel_format = CString::new(format.pixel_format())?;
                let formats = [FFMS_GetPixFmt(pixel_format.as_ptr()), -1];
                let status = FFMS_SetOutputFormatV2(
                    opened.source,
                    formats.as_ptr(),
                    format.width as i32,
                    format.height as i32,
                    FFMS_Resizers::FFMS_RESIZER_BILINEAR as i32,
                    &mut error,
                );
                if status != 0 {
                    return Err(failure(&error));
                }
            }
            Ok(opened)
        }
    }

    fn read<T: Pixel>(&mut self, format: &ScanFormat) -> Result<Frame<T>, anyhow::Error> {
        let mut buffer = [0 as c_char; 1024];
        let mut error = FFMS_ErrorInfo {
            ErrorType: 0,
            SubType: 0,
            BufferSize: buffer.len() as i32,
            Buffer: buffer.as_mut_ptr(),
        };

        // SAFETY: the source is live, and a non-null frame stays valid until the next call.
        let decoded: &FFMS_Frame =
            unsafe { FFMS_GetFrame(self.source, self.next as i32, &mut error).as_ref() }
                .ok_or_else(|| anyhow::anyhow!("ffms2 could not decode frame {}", self.next))?;
        self.next += 1;

        let mut frame = format.blank_frame::<T>()?;
        for (plane, (index, (_, height))) in [
            Some(&mut frame.y_plane),
            frame.u_plane.as_mut(),
            frame.v_plane.as_mut(),
        ]
        .into_iter()
        .zip(format.plane_sizes().into_iter().enumerate())
        {
            let Some(plane) = plane else { continue };
            let stride = decoded.Linesize[index] as usize;
            // SAFETY: ffms2 hands over `height` rows of `stride` bytes for this plane.
            let bytes = unsafe { std::slice::from_raw_parts(decoded.Data[index], stride * height) };
            plane.copy_from_u8_slice_with_stride(
                bytes,
                NonZeroUsize::new(stride).expect("ffms2 plane stride is non-zero"),
            )?;
        }

        Ok(frame)
    }
}

impl Drop for FfmsSource {
    fn drop(&mut self) {
        // SAFETY: each handle is either null or owned by this value.
        unsafe {
            if !self.source.is_null() {
                FFMS_DestroyVideoSource(self.source);
            }
            FFMS_DestroyIndex(self.index);
        }
    }
}
