#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FrameRange {
    pub start: usize,
    pub end: usize,
}

impl std::str::FromStr for FrameRange {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (start, end) = value
            .split_once(':')
            .ok_or_else(|| "frame range must be START:END".to_owned())?;
        let start = start.parse().map_err(|_| "frame range start is invalid")?;
        let end = end.parse().map_err(|_| "frame range end is invalid")?;
        if start >= end {
            return Err("frame range must have START < END".into());
        }
        Ok(Self { start, end })
    }
}
