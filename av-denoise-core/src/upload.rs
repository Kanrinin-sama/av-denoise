//! Host-to-GPU uploads into buffers that live as long as their owner.
//!
//! An [`Uploader`] owns one GPU buffer per region and refills them in
//! place. On a wgpu client [`crate::probe::load_client`] opened, the
//! bytes go through a [`StagingRing`] allocated once, so a steady stream
//! of uploads allocates nothing on the host or the device. Any other
//! client falls back to a fresh CubeCL buffer per upload.

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

#[cfg(any(feature = "vulkan", feature = "metal"))]
use crate::staging_ring::StagingRing;

/// How many uploads can be in flight before the next one waits on the
/// oldest: one per pending frame, plus the one being written.
pub(crate) const UPLOAD_RING_DEPTH: usize = crate::denoiser::MAX_PENDING + 1;

pub(crate) struct Region {
    pub(crate) handle: Handle,
    /// The most bytes one upload can write, rounded up to a whole word.
    pub(crate) capacity: usize,
}

pub(crate) struct Uploader<R: Runtime> {
    client: ComputeClient<R>,
    regions: Vec<Region>,
    #[cfg(any(feature = "vulkan", feature = "metal"))]
    ring: Option<StagingRing>,
}

impl<R: Runtime> Uploader<R> {
    /// One GPU buffer per entry of `capacities`, in bytes.
    pub(crate) fn new(client: &ComputeClient<R>, capacities: &[usize]) -> Self {
        let regions: Vec<Region> = capacities
            .iter()
            .map(|&capacity| {
                let capacity = capacity.max(1).next_multiple_of(size_of::<u32>());
                Region {
                    handle: client.empty(capacity),
                    capacity,
                }
            })
            .collect();

        Self {
            client: client.clone(),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            ring: StagingRing::for_client(client, &regions, UPLOAD_RING_DEPTH),
            regions,
        }
    }

    /// The buffer holding `region`'s latest upload.
    pub(crate) fn handle(&self, region: usize) -> &Handle {
        &self.regions[region].handle
    }

    /// The most bytes one upload can write into `region`.
    pub(crate) fn capacity(&self, region: usize) -> usize {
        self.regions[region].capacity
    }

    /// Uploads `contents[i]`'s parts back to back into region `i`, with
    /// a trailing partial word zero-filled.
    ///
    /// Launches queued before this call still see the previous contents,
    /// and launches queued after it see the new ones.
    pub(crate) fn upload(&mut self, contents: &[&[&[u8]]]) {
        #[cfg(any(feature = "vulkan", feature = "metal"))]
        if let Some(ring) = self.ring.as_mut() {
            ring.upload(&self.client, contents);
            return;
        }

        for (region, parts) in self.regions.iter_mut().zip(contents) {
            let mut bytes = parts.concat();
            bytes.resize(bytes.len().next_multiple_of(size_of::<u32>()), 0);
            region.handle = self.client.create_from_slice(&bytes);
        }
    }
}
