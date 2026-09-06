//! Argument bundles for the operator entry points of [`crate::GpuBackend`].

use oxidelake_core::params::{AggregateSpec, DistanceMetric, Predicate};

use crate::DeviceBatch;

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

/// Inner hash join on one `Int64` key per side.
#[derive(Debug, Clone, Copy)]
pub struct HashJoinArgs<'a> {
    /// Probe side; its columns come first in the output.
    pub left: &'a DeviceBatch,
    /// Build side.
    pub right: &'a DeviceBatch,
    /// Key column index in `left`.
    pub left_key: usize,
    /// Key column index in `right`.
    pub right_key: usize,
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
