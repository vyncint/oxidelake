//! Page-locked host memory through the CUDA driver (feature `cuda`).

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaContext, DriverError, PinnedHostSlice};
use oxidelake_core::{BackendKind, EngineError, HOST_ALIGN};

use crate::host::{HostAlloc, PinnedProvider};

fn driver_error(err: DriverError) -> EngineError {
    EngineError::device(BackendKind::Cuda, err.to_string())
}

/// A zeroed, page-locked host allocation owned by a CUDA context.
pub struct PinnedBuf {
    slice: PinnedHostSlice<u8>,
    len: usize,
    ctx: Arc<CudaContext>,
}

impl PinnedBuf {
    /// Allocates `len` zeroed pinned bytes (at least one byte is allocated).
    pub fn alloc(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, EngineError> {
        // SAFETY: the memory is uninitialised until the `fill(0)` below, and
        // nothing observes it before that.
        let mut slice = unsafe { ctx.alloc_pinned::<u8>(len.max(1)) }
            .map_err(|e| EngineError::allocation(len, HOST_ALIGN, e.to_string()))?;
        slice.as_mut_slice().map_err(driver_error)?.fill(0);
        Ok(Self {
            slice,
            len,
            ctx: Arc::clone(ctx),
        })
    }

    /// Logical length in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` when the buffer holds no bytes.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The contents, after waiting for device work targeting this buffer.
    pub fn as_slice(&self) -> Result<&[u8], EngineError> {
        Ok(&self.slice.as_slice().map_err(driver_error)?[..self.len])
    }

    /// Mutable contents, after waiting for device work targeting this buffer.
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8], EngineError> {
        let len = self.len;
        Ok(&mut self.slice.as_mut_slice().map_err(driver_error)?[..len])
    }

    /// The underlying cudarc slice, for `memcpy_htod`.
    pub const fn raw(&self) -> &PinnedHostSlice<u8> {
        &self.slice
    }

    /// The underlying cudarc slice, mutably, for `memcpy_dtoh`.
    pub const fn raw_mut(&mut self) -> &mut PinnedHostSlice<u8> {
        &mut self.slice
    }

    /// The owning context.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl fmt::Debug for PinnedBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedBuf")
            .field("len", &self.len)
            .field("device", &self.ctx.ordinal())
            .finish()
    }
}

/// [`PinnedProvider`] backed by a CUDA context.
#[derive(Debug)]
pub struct CudaPinnedProvider {
    ctx: Arc<CudaContext>,
}

impl CudaPinnedProvider {
    /// Creates a provider for `ctx`.
    pub fn new(ctx: Arc<CudaContext>) -> Self {
        Self { ctx }
    }
}

impl PinnedProvider for CudaPinnedProvider {
    fn alloc_pinned(&self, len: usize) -> Result<HostAlloc, EngineError> {
        Ok(HostAlloc::from_pinned(PinnedBuf::alloc(&self.ctx, len)?))
    }
}
