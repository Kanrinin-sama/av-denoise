//! The wgpu device and queue behind each wgpu client this crate opens.
//!
//! CubeCL builds its wgpu device privately. Starting the server through
//! `init_setup` instead hands back the same device and queue it runs on,
//! which is what lets [`crate::upload`] copy into CubeCL-owned buffers
//! from its own persistent staging.

use crate::runtime::Runtime;
use std::sync::Mutex;
pub use wgpu;

use cubecl::prelude::*;
use cubecl::wgpu::{RuntimeOptions, WgpuDevice, WgpuRuntime};

struct Shared {
    device: WgpuDevice,
    /// Kept alive so the server identity [`device_and_queue`] compares
    /// against can never be reused by another server.
    client: Client,
    gpu: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    queue_count: u32,
}

static SHARED: Mutex<Vec<Shared>> = Mutex::new(Vec::new());

/// Starts CubeCL's server for `device` through `init_setup`, once per
/// device, keeping the wgpu device and queue it runs on.
///
/// This builds the server exactly as CubeCL's lazy default does, same
/// graphics API and same runtime options, so kernels see no difference.
/// It has to run before the first `WgpuRuntime::client` for `device`,
/// which is why [`crate::probe::load_client`] is the only way this crate
/// opens a client.
pub fn register(device: &WgpuDevice) -> Client {
    let mut shared = SHARED.lock().unwrap_or_else(|err| err.into_inner());
    if let Some(entry) = shared.iter().find(|entry| &entry.device == device) {
        return entry.client.clone();
    }

    #[cfg(feature = "vulkan")]
    let (setup, registered, queue_count) = if matches!(device.backend, cubecl::wgpu::WgpuBackend::Vulkan)
        || matches!(device.backend, cubecl::wgpu::WgpuBackend::Auto) && !cfg!(target_vendor = "apple")
    {
        let (setup, queue_count) = crate::vulkan_setup::create(device).expect("creating the shared Vulkan compute device");
        let registered = cubecl::wgpu::try_init_device_with_api::<cubecl::wgpu::Vulkan>(
            setup.clone(),
            RuntimeOptions::default(),
        )
        .expect("registering the shared Vulkan compute device");
        (setup, registered, queue_count)
    } else {
        (
            cubecl::wgpu::init_setup::<cubecl::wgpu::AutoGraphicsApi>(device, RuntimeOptions::default()),
            device.clone(),
            1,
        )
    };
    #[cfg(not(feature = "vulkan"))]
    let (setup, registered, queue_count) = (
        cubecl::wgpu::init_setup::<cubecl::wgpu::AutoGraphicsApi>(device, RuntimeOptions::default()),
        device.clone(),
        1,
    );
    shared.push(Shared {
        device: device.clone(),
        client: WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(&registered),
        gpu: setup.device,
        queue: setup.queue,
        adapter: setup.adapter,
        queue_count,
    });
    shared.last().unwrap().client.clone()
}

/// The wgpu device and queue `client`'s server runs on, or `None` for a
/// server [`register`] did not start.
///
/// Every client of one server shares that server's properties, so their
/// address identifies the server.
pub fn device_and_queue(client: &Client) -> Option<(wgpu::Device, wgpu::Queue)> {
    let shared = SHARED.lock().unwrap_or_else(|err| err.into_inner());
    shared
        .iter()
        .find(|entry| std::ptr::eq(entry.client.properties(), client.properties()))
        .map(|entry| (entry.gpu.clone(), entry.queue.clone()))
}

static QUEUE_LOCK: parking_lot::ReentrantMutex<()> = parking_lot::ReentrantMutex::new(());
pub type QueueGuard = parking_lot::ReentrantMutexGuard<'static, ()>;
pub fn queue_lock() -> QueueGuard {
    QUEUE_LOCK.lock()
}
pub fn vulkan_gpu(index: usize) -> (wgpu::Device, wgpu::Queue, wgpu::Adapter, u32) {
    let selector = WgpuDevice::new(cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(index));
    let client = register(&selector);
    let shared = SHARED.lock().unwrap_or_else(|e| e.into_inner());
    let entry = shared
        .iter()
        .find(|entry| std::ptr::eq(entry.client.properties(), client.properties()))
        .unwrap();
    (entry.gpu.clone(), entry.queue.clone(), entry.adapter.clone(), entry.queue_count)
}

#[derive(Clone, Debug)]
pub struct GpuPlane {
    pub buffer: wgpu::Buffer,
    pub offset: u64,
}
pub fn export_planes<R: Runtime>(
    client: &crate::runtime::ComputeClient<R>,
    handles: &[cubecl::server::Handle],
) -> Result<Vec<GpuPlane>, anyhow::Error> {
    use std::any::Any;
    let client = (client as &dyn Any)
        .downcast_ref::<crate::runtime::ComputeClient<WgpuRuntime<cubecl::wgpu::AutoCompiler>>>()
        .ok_or_else(|| anyhow::anyhow!("device output requires a Vulkan client"))?;
    client.flush()?;
    handles
        .iter()
        .map(|handle| {
            let resource = client.get_resource::<<WgpuRuntime as Runtime>::Server>(handle.clone())?;
            Ok(GpuPlane {
                buffer: resource.resource().buffer.clone(),
                offset: resource.resource().offset,
            })
        })
        .collect()
}
