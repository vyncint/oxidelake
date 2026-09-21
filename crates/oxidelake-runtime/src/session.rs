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
#[non_exhaustive]
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

/// Knobs a session is built with.
///
/// Every field has a default that matches the no-argument constructors, so
/// `SessionOptions::default()` and [`OxideSession::local`] agree. The struct
/// is `#[non_exhaustive]`: a knob added later is a minor release, not a
/// broken build for anyone who used the builder methods.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SessionOptions {
    /// Plan for this backend instead of the detected one (embedded only).
    pub target: Option<BackendKind>,
    /// Rows per record batch. `None` leaves DataFusion's default (8192), or
    /// [`oxidelake_storage::GPU_BATCH_SIZE`] when the placement target is a GPU.
    pub batch_size: Option<usize>,
}

impl SessionOptions {
    /// Defaults: detected backend, default batch size.
    pub fn new() -> Self {
        Self::default()
    }

    /// Plans for `target` rather than the detected backend.
    pub fn with_target(mut self, target: BackendKind) -> Self {
        self.target = Some(target);
        self
    }

    /// Sets the rows per record batch, overriding the CPU and GPU defaults.
    pub fn with_batch_size(mut self, rows: usize) -> Self {
        self.batch_size = Some(rows);
        self
    }

    /// Applies the batch size to `config`, or the GPU default when the
    /// placement target is a device and no size was asked for.
    ///
    /// A batch size of zero is rejected here rather than passed on: DataFusion
    /// would take it and then produce no rows at all, which reads as an empty
    /// result rather than as a bad flag.
    fn apply(
        &self,
        config: SessionConfig,
        target: Option<BackendKind>,
    ) -> Result<SessionConfig, EngineError> {
        match self.batch_size {
            Some(0) => Err(EngineError::plan(
                "batch size must be at least 1 row (0 would make every query return nothing)",
            )),
            Some(rows) => Ok(config.with_batch_size(rows)),
            None if target.is_some_and(BackendKind::is_gpu) => Ok(with_gpu_batch_size(config)),
            None => Ok(config),
        }
    }
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
        Self::local_with_options(&SessionOptions::new())
    }

    /// An embedded session whose placement rule targets `target` instead of
    /// the detected backend. Placement is a planning decision: operators still
    /// select the real local backend at `execute()` time and fall back to the
    /// CPU reference, so planning for an absent GPU is safe — it is exactly
    /// what cluster executors do with the scheduler's plans.
    pub fn local_with_target(target: BackendKind) -> Result<Self, EngineError> {
        Self::local_with_options(&SessionOptions::new().with_target(target))
    }

    /// An embedded session built with `options`.
    pub fn local_with_options(options: &SessionOptions) -> Result<Self, EngineError> {
        let target = match options.target {
            Some(target) => target,
            None => local_backend()?.kind(),
        };
        let telemetry = TelemetryHub::new();
        let rule = HardwarePlacementRule::new(target).with_telemetry(Arc::clone(&telemetry));
        let config = options.apply(with_pruning(SessionConfig::new()), Some(target))?;
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
        Self::connect_with_options(scheduler_url, &SessionOptions::new()).await
    }

    /// Connects to a Ballista scheduler with `options`. `target` is ignored:
    /// on a cluster the scheduler's `OXIDE_CLUSTER_BACKEND` decides placement,
    /// so a client-side target would be a knob that quietly does nothing.
    pub async fn connect_with_options(
        scheduler_url: &str,
        options: &SessionOptions,
    ) -> Result<Self, EngineError> {
        let config = with_pruning(SessionConfig::new_with_ballista())
            .with_ballista_physical_extension_codec(cluster::oxide_codec());
        let config = options.apply(config, None)?;
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

    fn batch_size(session: &OxideSession) -> usize {
        session.ctx().state().config().batch_size()
    }

    /// `--batch-size` has to reach DataFusion's configuration: a flag that is
    /// parsed and then dropped would pass every test that only compares query
    /// results, because batch boundaries do not change them.
    #[test]
    fn the_batch_size_option_reaches_the_session_configuration() {
        let session =
            OxideSession::local_with_options(&SessionOptions::new().with_batch_size(7)).unwrap();
        assert_eq!(batch_size(&session), 7);
    }

    /// Without an explicit size, a GPU-targeted session keeps the larger
    /// default that amortises the host↔device round trip, and a CPU one keeps
    /// DataFusion's. An explicit size overrides both.
    #[test]
    fn the_target_picks_the_default_batch_size_and_the_option_overrides_it() {
        let gpu = OxideSession::local_with_target(BackendKind::Cuda).unwrap();
        assert_eq!(batch_size(&gpu), oxidelake_storage::GPU_BATCH_SIZE);

        let cpu = OxideSession::local_with_target(BackendKind::CpuSimd).unwrap();
        assert_eq!(batch_size(&cpu), SessionConfig::new().batch_size());

        let forced = OxideSession::local_with_options(
            &SessionOptions::new()
                .with_target(BackendKind::Cuda)
                .with_batch_size(1_024),
        )
        .unwrap();
        assert_eq!(batch_size(&forced), 1_024);
    }

    #[test]
    fn a_zero_batch_size_is_refused() {
        let err = OxideSession::local_with_options(&SessionOptions::new().with_batch_size(0))
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least 1 row"), "{err}");
    }

    /// The no-argument constructors and `SessionOptions::default()` are the
    /// same session, so the builder cannot drift away from the plain one.
    #[test]
    fn the_default_options_are_the_plain_constructor() {
        let plain = OxideSession::local().unwrap();
        let built = OxideSession::local_with_options(&SessionOptions::default()).unwrap();
        assert_eq!(plain.mode(), built.mode());
        assert_eq!(batch_size(&plain), batch_size(&built));
    }
}
