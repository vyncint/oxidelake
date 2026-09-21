//! The user-facing session: one query API over embedded and cluster execution.

use std::sync::Arc;
use std::time::Instant;

use ballista::prelude::SessionContextExt;
use ballista_core::extension::SessionConfigExt;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
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

impl std::fmt::Display for SessionMode {
    /// `embedded/cuda` or `cluster/df://host:port` — one field in a log line
    /// rather than two, because the second is only meaningful given the first.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionMode::Embedded { target } => write!(f, "embedded/{target}"),
            SessionMode::Cluster { scheduler_url } => write!(f, "cluster/{scheduler_url}"),
        }
    }
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

    /// The telemetry hub for this session.
    ///
    /// **Embedded sessions only.** A cluster session's operators run on
    /// executors, in other processes; this hub is created for the shape of
    /// the type and stays empty, so a snapshot of it is not "no work
    /// happened" but "the work happened somewhere else" (#33). Each worker
    /// reports into its own process-wide hub, which `oxide-worker
    /// --metrics-port` exposes as Prometheus text.
    pub fn telemetry(&self) -> &Arc<TelemetryHub> {
        &self.telemetry
    }

    /// Plans a SQL statement into a lazily executed [`DataFrame`].
    ///
    /// Nothing runs here, so nothing is logged here: see [`Self::collect`]
    /// for the one-line-per-query record.
    pub async fn sql(&self, query: &str) -> Result<DataFrame, EngineError> {
        Ok(self.ctx.sql(query).await?)
    }

    /// Runs `query` to completion and logs one `INFO` line describing it
    /// (#33): the backend it was planned for, the rows it produced, how long
    /// it took, and how many batches fell back to the CPU reference.
    ///
    /// This is the line an operator reads to answer "is the GPU being used
    /// and how long did the query take" without attaching a dashboard. It is
    /// on `collect` rather than on [`Self::sql`] because a `DataFrame` has
    /// not run yet: a line logged at planning time could only report the
    /// plan, and the interesting half is what the plan then did.
    /// The schema comes back beside the batches because an empty result has
    /// no batch to take it from, and a caller rendering CSV still has to
    /// print the header.
    pub async fn collect(&self, query: &str) -> Result<(SchemaRef, Vec<RecordBatch>), EngineError> {
        let started = Instant::now();
        let frame = self.sql(query).await?;
        let schema = SchemaRef::from(frame.schema().clone());
        let batches = frame.collect().await?;
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let operators = self.telemetry.snapshot().operators;
        let fallback_batches: u64 = operators.iter().map(|o| o.fallback_batches).sum();
        let accelerated = operators.iter().filter(|o| o.backend.is_gpu()).count();
        tracing::info!(
            mode = %self.mode,
            rows,
            batches = batches.len(),
            elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
            gpu_operators = accelerated,
            fallback_batches,
            "query finished"
        );
        Ok((schema, batches))
    }

    /// Registers a Parquet file or directory as `name`.
    pub async fn register_parquet(&self, name: &str, path: &str) -> Result<(), EngineError> {
        register_parquet_table(&self.ctx, name, path).await
    }

    /// The indented physical plan for `query`, with placement tags in embedded
    /// mode. In cluster mode this is the client-side plan (`DistributedQueryExec`);
    /// the scheduler's plan is what carries the tags there.
    pub async fn explain(&self, query: &str) -> Result<String, EngineError> {
        self.telemetry.clear_skips();
        let plan = self.ctx.sql(query).await?.create_physical_plan().await?;
        let mut text = displayable(plan.as_ref()).indent(true).to_string();
        text.push_str(&self.placement_notes());
        Ok(text)
    }

    /// The placement rule's reasons for leaving nodes on the CPU (#32).
    ///
    /// A node the rule skipped is an ordinary DataFusion operator in the
    /// plan above, indistinguishable from one that was never eligible. The
    /// reason used to live only in a `debug!` line, which is no use to
    /// someone reading a plan, so it is printed under it.
    ///
    /// Empty in cluster mode: the rule runs on the scheduler there, and this
    /// process only planned the `DistributedQueryExec` wrapper.
    fn placement_notes(&self) -> String {
        let skips = self.telemetry.skips();
        if skips.is_empty() {
            return String::new();
        }
        let mut out = String::from("\nplacement notes (target ");
        match &self.mode {
            SessionMode::Embedded { target } => out.push_str(target.as_str()),
            SessionMode::Cluster { .. } => out.push_str("on the scheduler"),
        }
        out.push_str("):\n");
        for skip in skips {
            out.push_str(&format!("  {}: {}\n", skip.node, skip.reason));
        }
        out
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
