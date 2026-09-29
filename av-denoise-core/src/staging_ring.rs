//! Persistent host staging for uploads on the wgpu backends.
//!
//! `queue.write_buffer` creates a fresh `MAP_WRITE` buffer per call and
//! frees it after the next submit, which at frame sizes means a new
//! `vkAllocateMemory` per frame. This ring allocates its staging buffers
//! once and cycles through them instead.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use crate::gpu_share;
use crate::upload::Region;

const PENDING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

pub(crate) struct StagingRing {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Each region's destination buffer, its offset in that buffer, and
    /// where the region starts inside a staging buffer.
    targets: Vec<Target>,
    slots: Vec<Slot>,
    next: usize,
}

struct Target {
    buffer: wgpu::Buffer,
    offset: u64,
    staging_offset: usize,
}

struct Slot {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    submission: Option<wgpu::SubmissionIndex>,
}

impl StagingRing {
    /// A ring of `depth` staging buffers, each big enough for every
    /// region at once, or `None` when `client` is not a wgpu client
    /// [`gpu_share::register`] started.
    pub(crate) fn for_client<R: Runtime>(client: &ComputeClient<R>, regions: &[Region], depth: usize) -> Option<Self> {
        let client = (client as &dyn Any).downcast_ref::<ComputeClient<WgpuRuntime>>()?;
        let (device, queue) = gpu_share::device_and_queue(client)?;

        let mut staging_len = 0;
        let targets = regions
            .iter()
            .map(|region| {
                let resource = client
                    .get_resource(region.handle.clone())
                    .expect("an upload target has a wgpu resource");
                let target = Target {
                    buffer: resource.resource().buffer.clone(),
                    offset: resource.resource().offset,
                    staging_offset: staging_len,
                };
                staging_len += region.capacity;
                target
            })
            .collect();

        let slots = (0..depth)
            .map(|_| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("av-denoise upload staging"),
                    size: staging_len as u64,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                }),
                state: Arc::new(AtomicU8::new(MAPPED)),
                submission: None,
            })
            .collect();

        Some(Self {
            device,
            queue,
            targets,
            slots,
            next: 0,
        })
    }

    /// Writes each region's parts into the next staging buffer and
    /// copies them into the regions' buffers.
    ///
    /// `client` is flushed first, so every launch already queued on its
    /// stream is submitted ahead of the copy, and every launch queued
    /// after this call is submitted behind it.
    pub(crate) fn upload<R: Runtime>(&mut self, client: &ComputeClient<R>, contents: &[&[&[u8]]]) {
        let index = self.next;
        self.next = (index + 1) % self.slots.len();
        let slot = &mut self.slots[index];

        if let Some(submission) = slot.submission.take() {
            // The map callback can run on whichever thread polls the
            // device, so the wait repeats until this thread sees it.
            while slot.state.load(Ordering::Acquire) == PENDING {
                self.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(submission.clone()),
                        timeout: None,
                    })
                    .expect("waiting on an upload staging buffer failed");
            }
            assert_eq!(
                slot.state.load(Ordering::Acquire),
                MAPPED,
                "an upload staging buffer failed to map"
            );
        }

        {
            let mut view = slot.buffer.slice(..).get_mapped_range_mut();
            for (target, parts) in self.targets.iter().zip(contents) {
                let mut cursor = target.staging_offset;
                for part in *parts {
                    view.slice(cursor..cursor + part.len()).copy_from_slice(part);
                    cursor += part.len();
                }

                // A copy moves whole words, so a region ending mid-word
                // has its last word zero-filled.
                view.slice(cursor..target.staging_offset + copy_len(parts)).fill(0);
            }
        }
        slot.buffer.unmap();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("av-denoise upload"),
        });
        for (target, parts) in self.targets.iter().zip(contents) {
            let copy_len = copy_len(parts);
            if copy_len > 0 {
                encoder.copy_buffer_to_buffer(
                    &slot.buffer,
                    target.staging_offset as u64,
                    &target.buffer,
                    target.offset,
                    copy_len as u64,
                );
            }
        }

        client.flush().expect("submitting queued launches ahead of an upload failed");
        slot.submission = Some(self.queue.submit([encoder.finish()]));

        slot.state.store(PENDING, Ordering::Release);
        let state = slot.state.clone();
        slot.buffer.slice(..).map_async(wgpu::MapMode::Write, move |result| {
            state.store(if result.is_ok() { MAPPED } else { FAILED }, Ordering::Release);
        });
    }
}

/// The whole-word length a region's parts copy as.
fn copy_len(parts: &[&[u8]]) -> usize {
    parts
        .iter()
        .map(|part| part.len())
        .sum::<usize>()
        .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
}
