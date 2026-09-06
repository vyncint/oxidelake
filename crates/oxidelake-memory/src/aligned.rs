//! Owned, zero-initialised host allocations with explicit alignment.

use std::alloc::{self, Layout};
use std::fmt;
use std::ptr::NonNull;

use oxidelake_core::{BufferLayout, DEVICE_ALIGN, EngineError, HOST_ALIGN};

/// An owned, zero-initialised, explicitly aligned host buffer.
///
/// The allocation is at least `align` bytes and a multiple of `align`, so the
/// whole padded region can be handed to DMA engines or vector loads without
/// touching bytes outside the allocation.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
}

impl AlignedBuf {
    /// Allocates `len` zeroed bytes aligned to `align` (a non-zero power of two).
    pub fn zeroed(len: usize, align: usize) -> Result<Self, EngineError> {
        let wanted = BufferLayout::new(len, align)?;
        let size = wanted.padded_len().max(align);
        let layout = Layout::from_size_align(size, align)
            .map_err(|e| EngineError::allocation(len, align, e.to_string()))?;
        // SAFETY: `layout.size()` is at least `align >= 1`, so it is non-zero as
        // `alloc_zeroed` requires.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw)
            .ok_or_else(|| EngineError::allocation(len, align, "allocator returned null"))?;
        Ok(Self { ptr, layout, len })
    }

    /// Allocates a [`HOST_ALIGN`]-aligned buffer.
    pub fn host(len: usize) -> Result<Self, EngineError> {
        Self::zeroed(len, HOST_ALIGN)
    }

    /// Allocates a [`DEVICE_ALIGN`]-aligned buffer suitable for device transfers.
    pub fn device_staging(len: usize) -> Result<Self, EngineError> {
        Self::zeroed(len, DEVICE_ALIGN)
    }

    /// The logical length in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` when the logical length is zero.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The alignment in bytes.
    pub const fn align(&self) -> usize {
        self.layout.align()
    }

    /// The allocated (padded) size in bytes; always a multiple of [`Self::align`].
    pub const fn capacity(&self) -> usize {
        self.layout.size()
    }

    /// Read-only pointer to the first byte.
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Mutable pointer to the first byte.
    pub const fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// The logical contents as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` points to `capacity() >= len` initialised (zeroed or
        // user-written) bytes owned exclusively by `self` for the returned lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The logical contents as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_slice`; `&mut self` guarantees no other reference exists.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` was returned by `alloc_zeroed(self.layout)` in `zeroed`
        // and is released exactly once, here.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

// SAFETY: the buffer exclusively owns its allocation and exposes it only through
// `&self`/`&mut self`, so moving or sharing it across threads is sound.
unsafe impl Send for AlignedBuf {}
// SAFETY: see the `Send` impl; shared access is read-only.
unsafe impl Sync for AlignedBuf {}

impl fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBuf")
            .field("len", &self.len)
            .field("align", &self.align())
            .field("capacity", &self.capacity())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn host_buffers_are_cache_line_aligned() {
        for len in [0usize, 1, 63, 64, 65, 4096] {
            let buf = AlignedBuf::host(len).unwrap();
            assert!((buf.as_ptr() as usize).is_multiple_of(64), "len {len}");
            assert_eq!(buf.len(), len);
            assert!(buf.capacity() >= len.max(1));
            assert!(buf.capacity().is_multiple_of(64));
            assert!(buf.as_slice().iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn device_staging_buffers_are_128_aligned() {
        let buf = AlignedBuf::device_staging(1000).unwrap();
        assert!((buf.as_ptr() as usize).is_multiple_of(128));
        assert_eq!(buf.capacity(), 1024);
        assert_eq!(buf.align(), 128);
    }

    #[test]
    fn writes_are_visible_through_slices() {
        let mut buf = AlignedBuf::host(16).unwrap();
        buf.as_mut_slice().copy_from_slice(&[7u8; 16]);
        assert_eq!(buf.as_slice(), &[7u8; 16]);
    }

    #[test]
    fn rejects_invalid_alignment() {
        assert!(AlignedBuf::zeroed(8, 24).is_err());
    }
}
