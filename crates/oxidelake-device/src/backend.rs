//! The object-safe hardware abstraction.

use std::fmt;

use arrow::array::RecordBatch;
use oxidelake_core::params::Predicate;
use oxidelake_core::{BackendKind, DeviceId, EngineError};

use crate::args::{AggregateArgs, FilterProjectArgs, HashJoinArgs, VectorDistanceArgs};
use crate::handles::{DeviceBatch, DeviceBuffer, DeviceStream, HostBuffer};

/// Memory capacity report for one backend device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Total memory addressable by the backend, in bytes.
    pub total_bytes: u64,
    /// Memory currently available for allocation, in bytes.
    pub free_bytes: u64,
    /// Whether page-locked (pinned) host allocations are available.
    pub pinned_supported: bool,
}

impl MemoryInfo {
    /// Bytes currently in use (`total - free`, saturating).
    pub const fn used_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.free_bytes)
    }
}

/// A hardware backend: memory management, transfers and the four operator
/// kernels of the bounded v1 coverage (docs/SPEC.md §2.3).
///
/// The trait is deliberately **object-safe** (no associated types; handles are
/// opaque structs) so the detector can hand out `Arc<dyn GpuBackend>`, and
/// **synchronous**: asynchrony lives on device streams and at the operator
/// layer. Every method that cannot serve a request returns
/// [`EngineError::Unsupported`], never a panic, so callers can fall back to
/// the CPU reference path.
pub trait GpuBackend: Send + Sync + fmt::Debug {
    /// Which backend this is.
    fn kind(&self) -> BackendKind;

    /// The device this backend drives.
    fn device_id(&self) -> DeviceId;

    /// Memory capacity of the device.
    fn memory_info(&self) -> Result<MemoryInfo, EngineError>;

    /// Allocates `bytes` of zeroed device memory (128-byte aligned).
    fn alloc_device(&self, bytes: usize) -> Result<DeviceBuffer, EngineError>;

    /// Allocates `bytes` of host memory, page-locked when the backend supports
    /// it; the buffer's `class()` reports which.
    fn alloc_pinned_host(&self, bytes: usize) -> Result<HostBuffer, EngineError>;

    /// Creates an execution stream (command queue) for ordering work.
    fn create_stream(&self) -> Result<DeviceStream, EngineError>;

    /// Copies host → device on `stream`.
    fn copy_h2d(
        &self,
        stream: &DeviceStream,
        src: &HostBuffer,
        dst: &mut DeviceBuffer,
    ) -> Result<(), EngineError>;

    /// Copies device → host on `stream`.
    fn copy_d2h(
        &self,
        stream: &DeviceStream,
        src: &DeviceBuffer,
        dst: &mut HostBuffer,
    ) -> Result<(), EngineError>;

    /// Blocks until all work queued on `stream` has completed.
    fn synchronize(&self, stream: &DeviceStream) -> Result<(), EngineError>;

    /// Moves a host batch onto the device (a clone for the CPU backend).
    fn upload(
        &self,
        stream: &DeviceStream,
        batch: &RecordBatch,
    ) -> Result<DeviceBatch, EngineError>;

    /// Brings a device batch back to host memory.
    fn download(
        &self,
        stream: &DeviceStream,
        batch: &DeviceBatch,
    ) -> Result<RecordBatch, EngineError>;

    /// Fused filter + projection.
    fn filter_project(
        &self,
        stream: &DeviceStream,
        args: FilterProjectArgs<'_>,
    ) -> Result<DeviceBatch, EngineError>;

    /// Inner hash join on one `Int64` key.
    fn hash_join(
        &self,
        stream: &DeviceStream,
        args: HashJoinArgs<'_>,
    ) -> Result<DeviceBatch, EngineError>;

    /// Grouped aggregation.
    fn aggregate(
        &self,
        stream: &DeviceStream,
        args: AggregateArgs<'_>,
    ) -> Result<DeviceBatch, EngineError>;

    /// Vector distance.
    fn vector_distance(
        &self,
        stream: &DeviceStream,
        args: VectorDistanceArgs<'_>,
    ) -> Result<DeviceBatch, EngineError>;

    /// `true` when [`Self::filter_project`] can evaluate `predicate` on the
    /// device. Operators consult this *before* uploading anything, so a
    /// backend that would only answer `Unsupported` costs no transfer.
    fn supports_predicate(&self, predicate: &Predicate) -> bool {
        let _ = predicate;
        true
    }

    /// `true` when [`Self::hash_join`] runs on the device (see
    /// [`Self::supports_predicate`]).
    fn supports_hash_join(&self) -> bool {
        true
    }

    /// `true` when [`Self::aggregate`] runs on the device (see
    /// [`Self::supports_predicate`]).
    fn supports_aggregate(&self) -> bool {
        true
    }
}
