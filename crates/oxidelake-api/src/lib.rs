//! Public API facade (docs/SPEC.md §2.8): the fluent DataFrame API and SQL entry
//! points over the runtime — [`OxideSession`] for sessions (embedded or
//! cluster), [`OxideFrame`] for queries, and the `l2_distance` /
//! `cosine_distance` SQL UDFs registered on every session.
//!
//! ```no_run
//! use oxidelake_api::prelude::*;
//!
//! # async fn demo() -> Result<(), EngineError> {
//! let session = OxideSession::local()?;
//! session.register_parquet("t", "data/").await?;
//! let top = session
//!     .table("t")
//!     .await?
//!     .filter(col("k").gt_eq(lit(2)))?
//!     .vector_distance("emb", &[0.0; 8], DistanceMetric::L2, "d")?
//!     .sort(vec![col("d").sort(true, false)])?
//!     .limit(10)?
//!     .collect()
//!     .await?;
//! # let _ = top; Ok(())
//! # }
//! ```
//!
//! Dependency direction: depends on `oxidelake-runtime` (and `oxidelake-core`).

mod frame;

pub use frame::{OxideFrame, OxideSessionExt};
pub use oxidelake_core::params::DistanceMetric;
pub use oxidelake_core::{BackendKind, EngineError};
pub use oxidelake_runtime::{OxideSession, SessionMode};

/// Commonly used imports: the session and frame types plus the DataFusion
/// expression builders the verbs take.
pub mod prelude {
    pub use datafusion::functions_aggregate::expr_fn::{avg, count, max, min, sum};
    pub use datafusion::prelude::{Expr, col, lit};

    pub use crate::{
        BackendKind, DistanceMetric, EngineError, OxideFrame, OxideSession, OxideSessionExt,
        SessionMode,
    };
}
