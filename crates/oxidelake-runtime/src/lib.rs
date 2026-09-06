//! Runtime (docs/SPEC.md §2.6): [`OxideSession`] for embedded and cluster
//! execution, and the thin wrappers that run Apache DataFusion Ballista's
//! scheduler and executors with OxideLake's placement rule, plan codec and
//! function registry installed (see [`cluster`]).
//!
//! Dependency direction: depends on every other OxideLake library crate and on
//! the `ballista*` crates.

pub mod cluster;
pub mod dashboard;
pub mod session;

pub use oxidelake_compute::udf;
pub use session::{OxideSession, SessionMode};
