//! Planning (docs/SPEC.md §2.5): the [`HardwarePlacementRule`] — a DataFusion
//! `PhysicalOptimizerRule` that rewrites eligible `FilterExec` /
//! `ProjectionExec` / `HashJoinExec` / `AggregateExec` nodes into `Gpu*Exec`
//! nodes — and the [`OxidePhysicalCodec`], a `PhysicalExtensionCodec` that lets
//! Ballista ship those nodes to executors.
//!
//! Placement is decided against a *target* backend: the local hardware in
//! embedded mode, the declared cluster capability (`OXIDE_CLUSTER_BACKEND`) on
//! the Ballista scheduler. `EXPLAIN` shows the target as a tag, e.g.
//! `GpuFilterExec[cuda]`; every node still selects its real backend at
//! execute time and falls back to the CPU path when the device is absent.
//!
//! Dependency direction: depends on `oxidelake-core`, `oxidelake-compute` and
//! `datafusion-proto` (the single `datafusion-*` sub-crate exception, ADR-0009).

pub mod codec;
pub mod rule;
pub mod target;

pub use codec::OxidePhysicalCodec;
pub use rule::{HardwarePlacementRule, VectorDistancePlacementRule, physical_optimizer_rules};
pub use target::{CLUSTER_TARGET_ENV, cluster_target_from_env, resolve_cluster_target};
