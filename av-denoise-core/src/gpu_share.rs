//! The wgpu device and queue behind each wgpu client this crate opens.
//!
//! CubeCL builds its wgpu device privately. Starting the server through
//! `init_setup` instead hands back the same device and queue it runs on,
//! which is what lets [`crate::upload`] copy into CubeCL-owned buffers
//! from its own persistent staging.

use std::sync::Mutex;

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup};

struct Shared {
    device: WgpuDevice,
    /// Kept alive so the server identity [`device_and_queue`] compares
    /// against can never be reused by another server.
    client: ComputeClient<WgpuRuntime>,
    gpu: wgpu::Device,
    queue: wgpu::Queue,
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
pub(crate) fn register(device: &WgpuDevice) {
    let mut shared = SHARED.lock().unwrap_or_else(|err| err.into_inner());
    if shared.iter().any(|entry| &entry.device == device) {
        return;
    }

    let setup = init_setup::<AutoGraphicsApi>(device, RuntimeOptions::default());
    shared.push(Shared {
        device: device.clone(),
        client: WgpuRuntime::client(device),
        gpu: setup.device,
        queue: setup.queue,
    });
}

/// The wgpu device and queue `client`'s server runs on, or `None` for a
/// server [`register`] did not start.
///
/// Every client of one server shares that server's properties, so their
/// address identifies the server.
pub(crate) fn device_and_queue(client: &ComputeClient<WgpuRuntime>) -> Option<(wgpu::Device, wgpu::Queue)> {
    let shared = SHARED.lock().unwrap_or_else(|err| err.into_inner());
    shared
        .iter()
        .find(|entry| std::ptr::eq(entry.client.properties(), client.properties()))
        .map(|entry| (entry.gpu.clone(), entry.queue.clone()))
}
