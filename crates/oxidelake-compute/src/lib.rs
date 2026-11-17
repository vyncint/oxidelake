//! Physical operators (docs/SPEC.md §2.3): the `Gpu*Exec` DataFusion
//! `ExecutionPlan`s, the backend-agnostic [`GpuOperator`] that runs each batch
//! on the local backend with CPU fallback, and the conformance suite (in
//! `tests/`) that pins every operator to the stock DataFusion semantics.
//!
//! * [`GpuFilterExec`] — fused filter + projection
//! * [`GpuHashJoinExec`] — inner hash join on one `Int64` key
//! * [`GpuAggregateExec`] — `SUM/COUNT/MIN/MAX` grouped by one `Int64` key
//! * [`GpuVectorDistanceExec`] — L2 / cosine distance column
//!
//! Every exec records the *planned* backend (its `EXPLAIN` tag) but selects the
//! *real* backend at `execute()` time from the local hardware detector, so a
//! plan shipped to a machine without a GPU still runs correctly.
//!
//! Dependency direction: depends on `oxidelake-core`, `oxidelake-memory`, `oxidelake-device`.

pub mod backend;
pub mod display;
pub mod exec;
pub mod operator;
pub mod udf;

/// Which optional capabilities of this build change what a plan can say, as
/// a bitmask.
///
/// A plan is only executable by an executor that can run every node in it,
/// and the features decide what a planner will put there — `predict` adds a
/// UDF the planner will happily reference and an executor without it cannot
/// resolve. The plan codec folds this into its fingerprint so the mismatch
/// is refused when the plan is decoded rather than discovered when it runs
/// (#42), and a cluster scheduler refuses a worker whose fingerprint is not
/// its own when the worker registers (#92).
///
/// Bit 0 `predict`. Bits 1 and 2 were `cuda` and `metal` until 0.3.0 and are
/// retired, always zero: those features change where an operator runs, not
/// what a plan says (`GpuNode` and the placement rule's output are the same
/// with or without them), and counting them refused the heterogeneous
/// cluster this project describes, a default-build scheduler with CUDA
/// workers (#92). Append, never renumber: an old executor reading a new mask
/// must still disagree with it, which it does as long as the bits it knows
/// keep their places.
pub const FEATURE_MASK: u32 = cfg!(feature = "predict") as u32;

pub use backend::{init_local_backend, local_backend};
pub use exec::{
    GpuAggregateExec, GpuFilterExec, GpuHashJoinExec, GpuVectorDistanceExec,
    aggregate_output_schema,
};
pub use operator::{GpuOperator, JoinBuild, JoinProbe};
pub use oxidelake_core::params;
pub use oxidelake_core::params::{
    AggregateFunction, AggregateSpec, Comparison, DistanceMetric, Literal, Predicate,
    gpu_eligible_scalar, vector_dimension,
};
#[cfg(feature = "predict")]
pub mod predict;
#[cfg(feature = "predict")]
pub use predict::{ACTIVATION_KEY, Activation, Model, ModelSpec, PREDICT, predict_udf};

pub use udf::{
    COSINE_DISTANCE, L2_DISTANCE, cosine_distance_udf, distance_metric_for, distance_udf_name,
    l2_distance_udf, literal_query, oxide_udfs, query_literal,
};
