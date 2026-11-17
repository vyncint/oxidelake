//! Argument bundles for the operator entry points of [`crate::GpuBackend`].

use oxidelake_core::params::{AggregateSpec, DistanceMetric, Predicate};

use crate::cpu::kernels::JoinTable;
use crate::{DeviceBatch, DeviceJoinMatches, DeviceJoinTable};

/// Fused filter + projection.
#[derive(Debug, Clone, Copy)]
pub struct FilterProjectArgs<'a> {
    /// Input batch (device-resident for GPU backends, host-resident for the CPU).
    pub input: &'a DeviceBatch,
    /// Predicate in the bounded v1 grammar.
    pub predicate: &'a Predicate,
    /// Output column indices, in output order.
    pub projection: &'a [usize],
}

/// A hash join's build side, for [`crate::GpuBackend::prepare_join_build`].
#[derive(Debug, Clone, Copy)]
pub struct JoinBuildArgs<'a> {
    /// The build side, resident on the backend: the rows `index` numbers.
    pub build: &'a DeviceBatch,
    /// Key column index in `build`.
    pub key: usize,
    /// The host's index of the same rows: each distinct non-`NULL` key's run
    /// in a row array grouped by key, in build order. The operator already
    /// holds it for the CPU path; a backend uploads it instead of hashing the
    /// build side again, and then emits a key's rows in the order the CPU
    /// reference does.
    pub index: &'a JoinTable,
}

/// One probe batch, for [`crate::GpuBackend::probe_join`].
#[derive(Debug, Clone, Copy)]
pub struct JoinProbeArgs<'a> {
    /// The probe side, resident on the backend.
    pub probe: &'a DeviceBatch,
    /// Key column index in `probe`.
    pub key: usize,
    /// The build side, prepared by [`crate::GpuBackend::prepare_join_build`].
    pub table: &'a DeviceJoinTable,
}

/// One chunk of an inner hash join's output, for
/// [`crate::GpuBackend::hash_join`]: matches `first..first + rows` of a
/// probe.
#[derive(Debug, Clone, Copy)]
pub struct HashJoinArgs<'a> {
    /// Probe side; its columns come first in the output.
    pub left: &'a DeviceBatch,
    /// Build side, as given to [`crate::GpuBackend::prepare_join_build`].
    pub right: &'a DeviceBatch,
    /// The prepared build side.
    pub table: &'a DeviceJoinTable,
    /// `left`'s matches, from [`crate::GpuBackend::probe_join`].
    pub matches: &'a DeviceJoinMatches,
    /// The first match to emit.
    pub first: u64,
    /// How many matches to emit; `first + rows` is at most `matches.len()`.
    pub rows: usize,
}

/// Grouped aggregation.
#[derive(Debug, Clone, Copy)]
pub struct AggregateArgs<'a> {
    /// Input batch.
    pub input: &'a DeviceBatch,
    /// Group key and aggregate functions.
    pub spec: &'a AggregateSpec,
}

/// Distance between a query vector and a `FixedSizeList<Float32>` column.
#[derive(Debug, Clone, Copy)]
pub struct VectorDistanceArgs<'a> {
    /// Input batch.
    pub input: &'a DeviceBatch,
    /// Index of the vector column.
    pub column: usize,
    /// The query vector (length must equal the list size).
    pub query: &'a [f32],
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Name of the appended `Float32` output column.
    pub output_name: &'a str,
}
