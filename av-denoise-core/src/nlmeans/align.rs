use cubecl::prelude::*;

/// Byte alignment every buffer binding must start on, taken from the
/// runtime the denoiser is running against.
///
/// A GPU rejects a bind group whose buffer offset is not a multiple of
/// its `min_storage_buffer_offset_alignment`. Every buffer this crate
/// slices into per-slot regions therefore pads its slot stride up to
/// this value.
///
/// Each backend reports its own figure, 32 bytes on the Vulkan adapters
/// we test against and up to 256 elsewhere, which is why the value is
/// read from the runtime rather than assumed.
///
/// It is carried as its own type rather than a bare `u64` so it cannot
/// be swapped by mistake with the width, height, or frame-count
/// arguments it travels alongside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StorageAlign(u64);

impl StorageAlign {
    /// The alignment `client`'s runtime requires.
    ///
    /// cubecl aligns every allocation it hands out to this same value,
    /// so a slot offset that is a multiple of it always lands on a
    /// boundary the backend accepts.
    pub(crate) fn from_client<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self::new(client.properties().memory.alignment)
    }

    /// A fixed alignment, for tests that have no runtime to ask.
    pub(crate) fn new(bytes: u64) -> Self {
        debug_assert!(
            bytes.is_power_of_two(),
            "storage alignment {bytes} is not a power of two"
        );
        Self(bytes.max(1))
    }

    /// `bytes` rounded up to the next aligned boundary.
    pub(crate) fn pad_bytes(self, bytes: u64) -> u64 {
        bytes.next_multiple_of(self.0)
    }

    /// A count of `T` rounded up so that many elements cover a whole
    /// number of alignment boundaries.
    ///
    /// Alignments are powers of two, so for any `T` whose size divides
    /// the alignment this lands exactly on a boundary. For a larger `T`
    /// the elements are already aligned, so the count comes back
    /// unchanged.
    pub(crate) fn pad_elems<T>(self, elems: usize) -> usize {
        let per_boundary = (self.0 as usize).div_ceil(size_of::<T>()).max(1);
        elems.next_multiple_of(per_boundary)
    }
}
