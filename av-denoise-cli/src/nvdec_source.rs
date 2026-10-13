use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread::JoinHandle;

use v_frame::frame::Frame;
use v_frame::pixel::Pixel;

use crate::scan_source::ScanFormat;

/// Hardware decode (Vulkan Video, which NVIDIA runs on NVDEC) and GPU scaling
/// through ffmpeg for one stretch of the source. Phantom positions repeat the
/// picture that follows them, as ffms2 does.
pub struct NvdecSource {
    child: Child,
    stdout: ChildStdout,
    log: Option<JoinHandle<(Vec<i64>, Vec<String>)>>,
    expected_pts: Vec<i64>,
    phantom: BTreeSet<usize>,
    position: usize,
    pending: Option<Vec<u8>>,
}

/// The stretch to decode: the index timestamps of its real pictures, in order,
/// and its phantom positions counted from the stretch start.
pub struct NvdecWindow {
    pub expected_pts: Vec<i64>,
    pub seconds_per_pts: f64,
    pub phantom: BTreeSet<usize>,
}

impl NvdecSource {
    pub fn open(
        ffmpeg: &Path,
        input: &Path,
        format: &ScanFormat,
        window: NvdecWindow,
    ) -> Result<Self, anyhow::Error> {
        let first_pts = *window
            .expected_pts
            .first()
            .ok_or_else(|| anyhow::anyhow!("the stretch holds no pictures"))?;
        let (download, output) = if format.bit_depth > 8 {
            ("p010le", "yuv420p10le")
        } else {
            ("nv12", "yuv420p")
        };
        let scale = if format.scaled {
            format!("scale_vulkan=w={}:h={},", format.width, format.height)
        } else {
            String::new()
        };
        let filter = format!(
            "select=gte(pts\\,{first_pts}),{scale}hwdownload,format={download},format={output},showinfo=checksum=0"
        );
        let seek = (first_pts as f64 * window.seconds_per_pts - 2.0).max(0.0);

        let mut command = Command::new(ffmpeg);
        command
            .current_dir(ffmpeg.parent().map_or_else(PathBuf::new, Path::to_path_buf))
            .args(["-hide_banner", "-nostats", "-loglevel", "info", "-nostdin"])
            .args([
                "-init_hw_device",
                "vulkan=gpu",
                "-hwaccel",
                "vulkan",
                "-hwaccel_device",
                "gpu",
            ])
            .args(["-hwaccel_output_format", "vulkan", "-threads", "1"])
            .args(["-noaccurate_seek", "-ss", &format!("{seek:.6}"), "-copyts", "-i"])
            .arg(input)
            .args([
                "-map",
                "0:V:0",
                "-an",
                "-sn",
                "-dn",
                "-filter_threads",
                "1",
                "-vf",
                &filter,
            ])
            .args(["-frames:v", &window.expected_pts.len().to_string()])
            .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = command.spawn()?;
        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr = child.stderr.take().expect("stderr is piped");
        let log = std::thread::spawn(move || {
            let mut pts = Vec::new();
            let mut other = Vec::new();
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                match line.contains("Parsed_showinfo").then(|| showinfo_pts(&line)) {
                    Some(Some(value)) => pts.push(value),
                    Some(None) => {},
                    None => other.push(line),
                }
            }
            (pts, other)
        });

        Ok(Self {
            child,
            stdout,
            log: Some(log),
            expected_pts: window.expected_pts,
            phantom: window.phantom,
            position: 0,
            pending: None,
        })
    }

    fn read_picture(&mut self, format: &ScanFormat) -> Result<Vec<u8>, anyhow::Error> {
        let mut bytes = vec![0u8; format.frame_bytes()];
        self.stdout
            .read_exact(&mut bytes)
            .map_err(|error| anyhow::anyhow!("ffmpeg ended before raw frame {}: {error}", self.position))?;
        Ok(bytes)
    }

    pub(crate) fn read<T: Pixel>(&mut self, format: &ScanFormat) -> Result<Frame<T>, anyhow::Error> {
        let bytes = match self.pending.take() {
            Some(bytes) => bytes,
            None => self.read_picture(format)?,
        };
        if self.phantom.contains(&self.position) {
            self.pending = Some(bytes.clone());
        }
        self.position += 1;

        let mut frame = format.blank_frame::<T>()?;
        let sample = if format.bit_depth > 8 { 2 } else { 1 };
        let mut offset = 0;
        for (plane, (width, height)) in [
            Some(&mut frame.y_plane),
            frame.u_plane.as_mut(),
            frame.v_plane.as_mut(),
        ]
        .into_iter()
        .zip(format.plane_sizes())
        {
            let Some(plane) = plane else { continue };
            let length = width * height * sample;
            plane.copy_from_u8_slice(&bytes[offset..offset + length])?;
            offset += length;
        }

        Ok(frame)
    }

    /// Confirms ffmpeg decoded exactly the pictures ffms2 indexed for the stretch.
    pub(crate) fn finish(mut self) -> Result<(), anyhow::Error> {
        let mut rest = Vec::new();
        self.stdout.read_to_end(&mut rest)?;
        let status = self.child.wait()?;
        let (pts, other) = self
            .log
            .take()
            .expect("the log reader runs until finish")
            .join()
            .map_err(|_| anyhow::anyhow!("the ffmpeg log reader panicked"))?;
        if !status.success() {
            anyhow::bail!("ffmpeg exited with {status}: {}", other.join("\n"));
        }
        if !rest.is_empty() || self.pending.is_some() || !pts.starts_with(&self.expected_pts) {
            anyhow::bail!(
                "ffmpeg decoded {} pictures, ffms2 indexed {} for this stretch, or their timestamps differ",
                pts.len(),
                self.expected_pts.len(),
            );
        }
        Ok(())
    }
}

fn showinfo_pts(line: &str) -> Option<i64> {
    let (_, after) = line.split_once(" pts:")?;
    after.split_whitespace().next()?.parse().ok()
}

impl Drop for NvdecSource {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
