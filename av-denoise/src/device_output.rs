pub use crate::gpu_share::GpuPlane;
use crate::{FrameLayout, Planes};
use std::any::Any;
use std::sync::Arc;

#[derive(Clone)]
pub struct DeviceFrame(pub Arc<dyn Any + Send + Sync>);
impl std::fmt::Debug for DeviceFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeviceFrame")
    }
}
pub trait DeviceSink: std::fmt::Debug + Send + Sync {
    fn requires_queue_lock(&self) -> bool {
        true
    }

    fn write(&self, layout: FrameLayout, planes: &[GpuPlane]) -> Result<DeviceFrame, anyhow::Error>;
}
pub(crate) fn emit(
    sink: &dyn DeviceSink,
    layout: FrameLayout,
    planes: Vec<GpuPlane>,
) -> Result<Planes, anyhow::Error> {
    Ok(Planes {
        device: Some(sink.write(layout, &planes)?),
        ..Planes::default()
    })
}
