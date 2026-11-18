//! The object-safe hardware abstraction.

use std::fmt;

use arrow::array::RecordBatch;
use oxidelake_core::params::Predicate;
use oxidelake_core::{BackendKind, DeviceId, EngineError};

use crate::args::{
    AggregateArgs, FilterProjectArgs, HashJoinArgs, JoinBuildArgs, JoinProbeArgs,
    VectorDistanceArgs,
};
use crate::handles::{
    DeviceBatch, DeviceBuffer, DeviceJoinMatches, DeviceJoinTable, DeviceStream, HostBuffer,
};

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

    /// Inner hash join on one `Int64` key: matches `args.first..args.first +
    /// args.rows` of a probe ([`Self::probe_join`]), as the probe side's
    /// columns followed by the build side's, gathered at each match's probe
    /// row and build row.
    ///
    /// An operator calls this once per output chunk of at most a batch, so
    /// the product of a key that repeats on both sides is never materialised
    /// in one piece (#95). A join runs as [`Self::prepare_join_build`] once,
    /// then [`Self::probe_join`] and one or more `hash_join` calls per probe
    /// batch.
    fn hash_join(
        &self,
        stream: &DeviceStream,
        args: HashJoinArgs<'_>,
    ) -> Result<DeviceBatch, EngineError>;

    /// Prepares a hash join's build side once, for every probe batch of one
    /// operator (#45): typically an index of the build rows in device memory,
    /// freed when the operator drops it.
    ///
    /// The default declines with [`EngineError::Unsupported`], so the join
    /// takes the CPU reference path — counted as a fallback — rather than
    /// handing a backend arguments it does not read. A backend that runs
    /// joins implements this, [`Self::probe_join`] and [`Self::hash_join`]
    /// together.
    fn prepare_join_build(
        &self,
        stream: &DeviceStream,
        args: JoinBuildArgs<'_>,
    ) -> Result<DeviceJoinTable, EngineError> {
        let _ = (stream, args);
        Err(EngineError::unsupported(
            "backend.prepare_join_build",
            format!(
                "backend `{}` does not prepare hash-join build sides",
                self.kind()
            ),
        ))
    }

    /// Probes one batch against a prepared build side: which build rows each
    /// probe row's key matches, kept for [`Self::hash_join`] to gather in
    /// ranges. Declines by default, like [`Self::prepare_join_build`].
    ///
    /// The matches must be numbered as the CPU reference numbers them
    /// ([`crate::cpu::kernels::JoinMatches`]): in probe-row order, and in
    /// build order within a probe row. An operator relies on it to finish a
    /// probe batch on the CPU from any match when a range is declined,
    /// without repeating or losing a row.
    fn probe_join(
        &self,
        stream: &DeviceStream,
        args: JoinProbeArgs<'_>,
    ) -> Result<DeviceJoinMatches, EngineError> {
        let _ = (stream, args);
        Err(EngineError::unsupported(
            "backend.probe_join",
            format!("backend `{}` does not probe hash joins", self.kind()),
        ))
    }

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
