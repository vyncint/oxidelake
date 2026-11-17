//! Opaque, backend-tagged handles. Their internals are enums over the enabled
//! backends; every handle frees its resource on `Drop` through the owning
//! backend's RAII types.

use std::fmt;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use oxidelake_core::BackendKind;
use oxidelake_memory::AlignedBuf;

use crate::cpu::kernels::{JoinMatches, JoinTable};

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

pub(crate) enum JoinTableInner {
    Host(JoinTable),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::ops::CudaJoinTable),
}

/// A hash join's build side prepared on a backend
/// ([`crate::GpuBackend::prepare_join_build`]): the backend's index of the
/// build rows, made once and probed by every batch of one operator, and
/// freed when the operator drops it (#45).
pub struct DeviceJoinTable {
    backend: BackendKind,
    build_rows: usize,
    uploaded_bytes: u64,
    pub(crate) inner: JoinTableInner,
}

impl DeviceJoinTable {
    /// Wraps a host index over a build side of `build_rows` rows (the CPU
    /// backend's representation).
    pub fn from_host(table: JoinTable, build_rows: usize) -> Self {
        Self {
            backend: BackendKind::CpuSimd,
            build_rows,
            uploaded_bytes: 0,
            inner: JoinTableInner::Host(table),
        }
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) const fn new(
        backend: BackendKind,
        build_rows: usize,
        uploaded_bytes: u64,
        inner: JoinTableInner,
    ) -> Self {
        Self {
            backend,
            build_rows,
            uploaded_bytes,
            inner,
        }
    }

    /// The backend the index is resident on.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Rows of the build side it indexes.
    pub const fn build_rows(&self) -> usize {
        self.build_rows
    }

    /// Bytes copied host → device to prepare it; zero for a host index.
    pub const fn uploaded_bytes(&self) -> u64 {
        self.uploaded_bytes
    }

    /// The host index, when host-resident.
    pub fn as_host(&self) -> Option<&JoinTable> {
        match &self.inner {
            JoinTableInner::Host(table) => Some(table),
            #[cfg(feature = "cuda")]
            JoinTableInner::Cuda(_) => None,
        }
    }
}

impl fmt::Debug for DeviceJoinTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceJoinTable")
            .field("backend", &self.backend)
            .field("build_rows", &self.build_rows)
            .field("uploaded_bytes", &self.uploaded_bytes)
            .finish()
    }
}

pub(crate) enum JoinMatchesInner {
    Host(JoinMatches),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::ops::CudaJoinMatches),
}

/// The matches of one probe batch against a [`DeviceJoinTable`]
/// ([`crate::GpuBackend::probe_join`]): a sequence in probe-row order, and in
/// build order within a probe row, that [`crate::GpuBackend::hash_join`]
/// gathers in ranges (#95).
pub struct DeviceJoinMatches {
    backend: BackendKind,
    probe_rows: usize,
    len: u64,
    pub(crate) inner: JoinMatchesInner,
}

impl DeviceJoinMatches {
    /// Wraps host matches (the CPU backend's representation).
    pub fn from_host(matches: JoinMatches) -> Self {
        Self {
            backend: BackendKind::CpuSimd,
            probe_rows: matches.probe_rows(),
            len: matches.len(),
            inner: JoinMatchesInner::Host(matches),
        }
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) const fn new(
        backend: BackendKind,
        probe_rows: usize,
        len: u64,
        inner: JoinMatchesInner,
    ) -> Self {
        Self {
            backend,
            probe_rows,
            len,
            inner,
        }
    }

    /// The backend the matches are resident on.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Probe rows looked up.
    pub const fn probe_rows(&self) -> usize {
        self.probe_rows
    }

    /// Total matches: the rows the probe batch joins to.
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// `true` when no probe row matched.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The host matches, when host-resident.
    pub fn as_host(&self) -> Option<&JoinMatches> {
        match &self.inner {
            JoinMatchesInner::Host(matches) => Some(matches),
            #[cfg(feature = "cuda")]
            JoinMatchesInner::Cuda(_) => None,
        }
    }
}

impl fmt::Debug for DeviceJoinMatches {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceJoinMatches")
            .field("backend", &self.backend)
            .field("probe_rows", &self.probe_rows)
            .field("len", &self.len)
            .finish()
    }
}
