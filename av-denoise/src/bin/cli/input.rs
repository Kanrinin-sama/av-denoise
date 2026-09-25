use std::fmt;
use std::io::Read;
use std::path::PathBuf;
use std::str::FromStr;

/// Where frames are read from, as named by `-i`/`--input`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    /// This process's standard input, written as `-` or `pipe:0`.
    Stdin,
    /// An inherited file descriptor, written as `pipe:N`. `N` is always
    /// 3 or above because 0, 1, and 2 are handled on their own.
    Fd(u32),
    /// A path on disk.
    File(PathBuf),
}

impl InputSource {
    /// Opens the stream this source names.
    ///
    /// Only the piped variants are readable here. A path goes through
    /// the file pipeline, which opens it with ffms2 instead.
    pub fn open_reader(&self) -> Result<Box<dyn Read>, anyhow::Error> {
        match self {
            InputSource::Stdin => Ok(Box::new(std::io::stdin().lock())),
            InputSource::Fd(fd) => open_fd(*fd),
            InputSource::File(path) => anyhow::bail!(
                "`{}` is a file path and is read through the file pipeline, not as a stream",
                path.display(),
            ),
        }
    }
}

impl FromStr for InputSource {
    type Err = String;

    /// Accepts the same input spellings as ffmpeg.
    ///
    /// - `-` and `pipe:0` are standard input
    /// - `pipe:N` for `N` of 3 or above is an inherited descriptor
    /// - anything else is a path on disk
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "-" {
            return Ok(InputSource::Stdin);
        }

        if let Some(rest) = s.strip_prefix("pipe:") {
            let fd: u32 = rest
                .parse()
                .map_err(|_| format!("pipe: expects a file descriptor number (got `{s}`)"))?;

            return match fd {
                0 => Ok(InputSource::Stdin),
                1 => Err("pipe:1 is this process's stdout, which carries the denoised y4m".to_string()),
                2 => Err("pipe:2 is this process's stderr, which carries log output".to_string()),
                n => Ok(InputSource::Fd(n)),
            };
        }

        if s.is_empty() {
            return Err("expected a file path, `-`, or `pipe:N`".to_string());
        }

        Ok(InputSource::File(PathBuf::from(s)))
    }
}

impl fmt::Display for InputSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputSource::Stdin => f.write_str("stdin"),
            InputSource::Fd(fd) => write!(f, "pipe:{fd}"),
            InputSource::File(path) => write!(f, "{}", path.display()),
        }
    }
}

/// Reopens an inherited descriptor through `/dev/fd`.
#[cfg(unix)]
fn open_fd(fd: u32) -> Result<Box<dyn Read>, anyhow::Error> {
    let path = format!("/dev/fd/{fd}");
    let file = std::fs::File::open(&path)
        .map_err(|e| anyhow::anyhow!("--input pipe:{fd} could not open {path}: {e}"))?;

    Ok(Box::new(file))
}

#[cfg(not(unix))]
fn open_fd(fd: u32) -> Result<Box<dyn Read>, anyhow::Error> {
    anyhow::bail!("--input pipe:{fd} needs a Unix platform, use `-` for stdin instead")
}
