//! Unified-memory buffers through Metal (feature `metal`, macOS only).
//!
//! `MTLResourceStorageModeShared` buffers are one physical allocation visible
//! to both the CPU and the GPU, so there is no transfer stage at all.

use std::fmt;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use oxidelake_core::{EngineError, HOST_ALIGN};

/// A zeroed Metal buffer in shared storage mode.
pub struct MetalSharedBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
}

impl MetalSharedBuffer {
    /// Allocates `len` zeroed bytes on `device` (at least one byte is allocated).
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, len: usize) -> Result<Self, EngineError> {
        let mut me = Self::new_uninit(device, len)?;
        me.as_mut_slice().fill(0);
        Ok(me)
    }

    /// Allocates `len` bytes on `device` without clearing them (at least one
    /// byte is allocated). Metal defines no initial contents, so callers must
    /// write every byte they later read — kernel outputs that cover the whole
    /// buffer, or host-filled argument buffers.
    pub fn new_uninit(
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
    ) -> Result<Self, EngineError> {
        let buffer = device
            .newBufferWithLength_options(len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                EngineError::allocation(len, HOST_ALIGN, "MTLDevice returned no buffer")
            })?;
        Ok(Self { buffer, len })
    }

    /// Copies `bytes` into a new shared buffer (one copy — the unavoidable
    /// host→shared move; Arrow allocations are not page-aligned, which
    /// `newBufferWithBytesNoCopy` would require).
    pub fn from_bytes(
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: &[u8],
    ) -> Result<Self, EngineError> {
        let mut me = Self::new_uninit(device, bytes.len())?;
        me.as_mut_slice().copy_from_slice(bytes);
        Ok(me)
    }

    /// Logical length in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` when the buffer holds no bytes.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The shared contents as seen by the CPU.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `contents()` points to `length() >= len` bytes of shared
        // storage owned by `self.buffer`, alive for the returned lifetime.
        unsafe {
            std::slice::from_raw_parts(self.buffer.contents().as_ptr().cast::<u8>(), self.len)
        }
    }

    /// The shared contents, mutably. Callers must not have GPU work in flight
    /// on this buffer.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as in `as_slice`; `&mut self` guarantees exclusive CPU access.
        unsafe {
            std::slice::from_raw_parts_mut(self.buffer.contents().as_ptr().cast::<u8>(), self.len)
        }
    }

    /// The Metal buffer object, for binding to compute encoders.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

// SAFETY: MTLBuffer objects are documented as thread-safe and this wrapper owns
// its buffer exclusively; CPU-side access goes through &self/&mut self.
unsafe impl Send for MetalSharedBuffer {}
// SAFETY: see the `Send` impl.
unsafe impl Sync for MetalSharedBuffer {}
// A panic while a reference is live cannot leave the buffer in a broken state:
// it holds no CPU-side invariants beyond the retained Metal object. Needed so
// an `Arc<MetalSharedBuffer>` can back an Arrow `Buffer` (zero-copy download).
impl std::panic::RefUnwindSafe for MetalSharedBuffer {}

impl fmt::Debug for MetalSharedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalSharedBuffer")
            .field("len", &self.len)
            .field("capacity", &self.buffer.length())
            .finish()
    }
}
