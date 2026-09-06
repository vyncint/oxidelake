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

pub use backend::local_backend;
pub use exec::{
    GpuAggregateExec, GpuFilterExec, GpuHashJoinExec, GpuVectorDistanceExec,
    aggregate_output_schema,
};
pub use operator::{GpuOperator, JoinBuild};
pub use oxidelake_core::params;
pub use oxidelake_core::params::{
    AggregateFunction, AggregateSpec, Comparison, DistanceMetric, Literal, Predicate,
    gpu_eligible_scalar, vector_dimension,
};
#[cfg(feature = "predict")]
pub mod predict;
#[cfg(feature = "predict")]
pub use predict::{Model, ModelSpec, PREDICT, predict_udf};

pub use udf::{
    COSINE_DISTANCE, L2_DISTANCE, cosine_distance_udf, distance_metric_for, distance_udf_name,
    l2_distance_udf, literal_query, oxide_udfs, query_literal,
};
