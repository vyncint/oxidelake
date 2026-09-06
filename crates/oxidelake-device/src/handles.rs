//! Opaque, backend-tagged handles. Their internals are enums over the enabled
//! backends; every handle frees its resource on `Drop` through the owning
//! backend's RAII types.

use std::fmt;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use oxidelake_core::BackendKind;
use oxidelake_memory::AlignedBuf;

/// A host buffer whose memory class (pinned/pageable) is observable.
pub use oxidelake_memory::HostAlloc as HostBuffer;

pub(crate) enum DeviceBufferInner {
    Host(AlignedBuf),
    #[cfg(feature = "cuda")]
    Cuda(cudarc::driver::CudaSlice<u8>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(oxidelake_memory::metal::MetalSharedBuffer),
}

/// A device-memory allocation.
pub struct DeviceBuffer {
    backend: BackendKind,
    len: usize,
    pub(crate) inner: DeviceBufferInner,
}

impl DeviceBuffer {
    pub(crate) const fn new(backend: BackendKind, len: usize, inner: DeviceBufferInner) -> Self {
        Self {
            backend,
            len,
            inner,
        }
    }

    /// The backend that owns the allocation.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Logical length in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` when the buffer holds no bytes.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The host allocation behind a CPU-backend buffer.
    pub fn as_host(&self) -> Option<&AlignedBuf> {
        match &self.inner {
            DeviceBufferInner::Host(b) => Some(b),
            #[cfg(feature = "cuda")]
            DeviceBufferInner::Cuda(_) => None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            DeviceBufferInner::Metal(_) => None,
        }
    }

    pub(crate) fn as_host_mut(&mut self) -> Option<&mut AlignedBuf> {
        match &mut self.inner {
            DeviceBufferInner::Host(b) => Some(b),
            #[cfg(feature = "cuda")]
            DeviceBufferInner::Cuda(_) => None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            DeviceBufferInner::Metal(_) => None,
        }
    }
}

impl fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("backend", &self.backend)
            .field("len", &self.len)
            .finish()
    }
}

pub(crate) enum StreamInner {
    Host,
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Arc<cudarc::driver::CudaStream>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::metal::MetalQueue),
}

/// An ordered execution stream (CUDA stream, Metal command queue, or the
/// calling thread for the CPU backend).
pub struct DeviceStream {
    backend: BackendKind,
    pub(crate) inner: StreamInner,
}

impl DeviceStream {
    pub(crate) const fn new(backend: BackendKind, inner: StreamInner) -> Self {
        Self { backend, inner }
    }

    /// The backend that owns the stream.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }
}

impl fmt::Debug for DeviceStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceStream")
            .field("backend", &self.backend)
            .finish()
    }
}

pub(crate) enum BatchInner {
    Host(RecordBatch),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::CudaBatch),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(crate::metal::MetalBatch),
}

/// A record batch resident on a backend.
pub struct DeviceBatch {
    backend: BackendKind,
    schema: SchemaRef,
    num_rows: usize,
    pub(crate) inner: BatchInner,
}

impl DeviceBatch {
    /// Wraps a host batch (the CPU backend's representation).
    pub fn from_host(batch: RecordBatch) -> Self {
        Self {
            backend: BackendKind::CpuSimd,
            schema: batch.schema(),
            num_rows: batch.num_rows(),
            inner: BatchInner::Host(batch),
        }
    }

    #[cfg_attr(
        not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))),
        allow(dead_code)
    )]
    pub(crate) const fn new(
        backend: BackendKind,
        schema: SchemaRef,
        num_rows: usize,
        inner: BatchInner,
    ) -> Self {
        Self {
            backend,
            schema,
            num_rows,
            inner,
        }
    }

    /// The backend the batch is resident on.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// The batch schema.
    pub const fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Number of rows.
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The host batch, when host-resident.
    pub fn as_host(&self) -> Option<&RecordBatch> {
        match &self.inner {
            BatchInner::Host(b) => Some(b),
            #[cfg(feature = "cuda")]
            BatchInner::Cuda(_) => None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            BatchInner::Metal(_) => None,
        }
    }

    /// Consumes the handle, returning the host batch when host-resident.
    pub fn into_host(self) -> Option<RecordBatch> {
        match self.inner {
            BatchInner::Host(b) => Some(b),
            #[cfg(feature = "cuda")]
            BatchInner::Cuda(_) => None,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            BatchInner::Metal(_) => None,
        }
    }
}

impl fmt::Debug for DeviceBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBatch")
            .field("backend", &self.backend)
            .field("num_rows", &self.num_rows)
            .field("columns", &self.schema.fields().len())
            .finish()
    }
}
