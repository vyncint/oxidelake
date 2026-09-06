//! The user-facing session: one query API over embedded and cluster execution.

use std::sync::Arc;

use ballista::prelude::SessionContextExt;
use ballista_core::extension::SessionConfigExt;
use datafusion::dataframe::DataFrame;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::displayable;
use datafusion::prelude::{SessionConfig, SessionContext};
use oxidelake_compute::{local_backend, oxide_udfs};
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_core::{BackendKind, EngineError};
use oxidelake_planner::{HardwarePlacementRule, physical_optimizer_rules};
use oxidelake_storage::{
    default_object_store, register_local_store, register_parquet_table, with_gpu_batch_size,
    with_pruning,
};

use crate::cluster;

/// Where a session executes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionMode {
    /// In-process DataFusion with the placement rule targeting the local backend.
    Embedded {
        /// The backend detected on this machine.
        target: BackendKind,
    },
    /// A Ballista cluster reached through its scheduler URL (`df://host:port`).
    Cluster {
        /// The scheduler URL.
        scheduler_url: String,
    },
}

/// An OxideLake session.
pub struct OxideSession {
    ctx: SessionContext,
    mode: SessionMode,
    telemetry: Arc<TelemetryHub>,
}

impl OxideSession {
    /// Creates an embedded session: Parquet pruning on, the local object store
    /// registered, the SQL UDFs, and the placement rule targeting the detected
    /// backend.
    pub fn local() -> Result<Self, EngineError> {
        Self::local_with_target(local_backend()?.kind())
    }

    /// An embedded session whose placement rule targets `target` instead of
    /// the detected backend. Placement is a planning decision: operators still
    /// select the real local backend at `execute()` time and fall back to the
    /// CPU reference, so planning for an absent GPU is safe — it is exactly
    /// what cluster executors do with the scheduler's plans.
    pub fn local_with_target(target: BackendKind) -> Result<Self, EngineError> {
        let telemetry = TelemetryHub::new();
        let rule = HardwarePlacementRule::new(target).with_telemetry(Arc::clone(&telemetry));
        let mut config = with_pruning(SessionConfig::new());
        if target.is_gpu() {
            config = with_gpu_batch_size(config);
        }
        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_config(config)
            .with_physical_optimizer_rules(physical_optimizer_rules(rule))
            .build();
        let ctx = SessionContext::new_with_state(state);
        register_local_store(&ctx, default_object_store());
        for udf in oxide_udfs() {
            ctx.register_udf(udf.as_ref().clone());
        }
        Ok(Self {
            ctx,
            mode: SessionMode::Embedded { target },
            telemetry,
        })
    }

    /// Connects to a Ballista scheduler (`df://host:port`). The session carries
    /// OxideLake's plan codec so `Gpu*Exec` nodes survive the trip to executors,
    /// and the SQL UDFs so queries plan client-side; placement itself happens
    /// on the scheduler.
    pub async fn connect(scheduler_url: &str) -> Result<Self, EngineError> {
        let config = with_pruning(SessionConfig::new_with_ballista())
            .with_ballista_physical_extension_codec(cluster::oxide_codec());
        let state = SessionStateBuilder::new()
            .with_default_features()
            .with_config(config)
            .build();
        let ctx = SessionContext::remote_with_state(scheduler_url, state).await?;
        for udf in oxide_udfs() {
            ctx.register_udf(udf.as_ref().clone());
        }
        Ok(Self {
            ctx,
            mode: SessionMode::Cluster {
                scheduler_url: scheduler_url.to_owned(),
            },
            telemetry: TelemetryHub::new(),
        })
    }

    /// The execution mode.
    pub fn mode(&self) -> &SessionMode {
        &self.mode
    }

    /// The underlying DataFusion context.
    pub fn ctx(&self) -> &SessionContext {
        &self.ctx
    }

    /// The telemetry hub for this session (local process only).
    pub fn telemetry(&self) -> &Arc<TelemetryHub> {
        &self.telemetry
    }

    /// Plans a SQL statement into a lazily executed [`DataFrame`].
    pub async fn sql(&self, query: &str) -> Result<DataFrame, EngineError> {
        Ok(self.ctx.sql(query).await?)
    }

    /// Registers a Parquet file or directory as `name`.
    pub async fn register_parquet(&self, name: &str, path: &str) -> Result<(), EngineError> {
        register_parquet_table(&self.ctx, name, path).await
    }

    /// The indented physical plan for `query`, with placement tags in embedded
    /// mode. In cluster mode this is the client-side plan (`DistributedQueryExec`);
    /// the scheduler's plan is what carries the tags there.
    pub async fn explain(&self, query: &str) -> Result<String, EngineError> {
        let plan = self.ctx.sql(query).await?.create_physical_plan().await?;
        Ok(displayable(plan.as_ref()).indent(true).to_string())
    }
}

impl std::fmt::Debug for OxideSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OxideSession")
            .field("mode", &self.mode)
            .field("session_id", &self.ctx.session_id())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embedded_session_runs_sql_and_reports_mode() {
        let session = OxideSession::local().unwrap();
        assert!(matches!(session.mode(), SessionMode::Embedded { .. }));
        let batches = session
            .sql("SELECT 1 + 1 AS two")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        let text = session.explain("SELECT 1 + 1 AS two").await.unwrap();
        assert!(text.contains("ProjectionExec"), "{text}");
    }
}
