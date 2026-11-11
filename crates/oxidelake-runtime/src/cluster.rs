//! Cluster mode on Apache DataFusion Ballista (ADR-0013).
//!
//! OxideLake does not write a scheduler. This module builds the pieces Ballista
//! lets us plug in — a session builder that installs the placement rule for the
//! declared cluster capability, a physical codec that ships `Gpu*Exec` nodes,
//! and the executor function registry — and wires them into Ballista's
//! scheduler and executor processes, plus an in-process standalone cluster for
//! tests and the CLI.

use std::any::Any;
use std::net::SocketAddr;
use std::sync::Arc;

use ballista_core::extension::{
    SessionConfigExt, ballista_aggregate_functions, ballista_scalar_functions,
    ballista_window_functions,
};
use ballista_core::registry::BallistaFunctionRegistry;
use ballista_core::serde::protobuf::scheduler_grpc_client::SchedulerGrpcClient;
use ballista_core::serde::{
    BallistaCodec, BallistaLogicalExtensionCodec, BallistaPhysicalExtensionCodec,
};
use ballista_core::{ConfigProducer, RuntimeProducer};
use ballista_executor::executor_process::{ExecutorProcessConfig, start_executor_process};
use ballista_executor::new_standalone_executor_from_builder;
use ballista_scheduler::cluster::BallistaCluster;
use ballista_scheduler::config::SchedulerConfig;
use ballista_scheduler::scheduler_process::start_server;
use ballista_scheduler::scheduler_server::SessionBuilder;
use ballista_scheduler::standalone::new_standalone_scheduler_with_builder;
use datafusion::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion::error::DataFusionError;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionState;
use datafusion::prelude::SessionConfig;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use oxidelake_compute::oxide_udfs;
use oxidelake_core::{BackendKind, EngineError};
use oxidelake_planner::{HardwarePlacementRule, OxidePhysicalCodec, physical_optimizer_rules};
use oxidelake_storage::{
    BoundedReadStore, ParquetScanGuard, default_object_store, with_gpu_batch_size, with_pruning,
};

fn ballista_error(err: impl std::fmt::Display) -> EngineError {
    EngineError::execution(format!("ballista: {err}"))
}

/// The physical codec every OxideLake process installs: our `Gpu*Exec`
/// encoding on top of Ballista's own codec (shuffle nodes).
pub fn oxide_codec() -> Arc<dyn PhysicalExtensionCodec> {
    Arc::new(OxidePhysicalCodec::new(Arc::new(
        BallistaPhysicalExtensionCodec::default(),
    )))
}

/// Ballista's combined logical + physical codec with [`oxide_codec`] as the physical half.
pub fn ballista_codec() -> BallistaCodec {
    BallistaCodec::new(
        Arc::new(BallistaLogicalExtensionCodec::default()),
        oxide_codec(),
    )
}

/// Session configuration for every process: Ballista defaults, Parquet pruning
/// on, and the OxideLake codec.
pub fn session_config() -> SessionConfig {
    with_pruning(SessionConfig::new_with_ballista())
        .with_ballista_physical_extension_codec(oxide_codec())
        .with_option_extension(ClientOptions::default())
}

/// What an OxideLake cluster client tells the scheduler beside DataFusion's
/// own settings (#93).
///
/// Ballista ships a client's whole configuration to the scheduler as
/// key/value pairs, and an extension registered at both ends travels with it
/// ([`session_config`] registers this one everywhere). The scheduler needs a
/// fact DataFusion's settings cannot carry: whether the batch size was a
/// choice. A GPU-targeted cluster raises an unchosen size to
/// [`GPU_BATCH_SIZE`](oxidelake_storage::GPU_BATCH_SIZE), and DataFusion's
/// default of 8192 is also a size someone can ask for, so the value alone
/// cannot say which it is.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ClientOptions {
    /// The client set its batch size (`--batch-size`,
    /// `SessionOptions::with_batch_size`).
    pub(crate) explicit_batch_size: bool,
}

impl ClientOptions {
    const EXPLICIT_BATCH_SIZE: &'static str = "explicit_batch_size";
}

impl ExtensionOptions for ClientOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> datafusion::error::Result<()> {
        if key != Self::EXPLICIT_BATCH_SIZE {
            return Err(DataFusionError::Configuration(format!(
                "no OxideLake setting `{}.{key}`",
                Self::PREFIX
            )));
        }
        self.explicit_batch_size = value.parse().map_err(|_| {
            DataFusionError::Configuration(format!(
                "`{}.{key}` takes true or false, not `{value}`",
                Self::PREFIX
            ))
        })?;
        Ok(())
    }

    // Prefixed, unlike the entries `extensions_options!` generates: Ballista
    // sends `entries()` to the scheduler as they are, and the scheduler's
    // `ConfigOptions::set` finds the extension by that prefix. An unprefixed
    // key arrives without a namespace and is dropped at `debug` level, so the
    // marker would silently never arrive.
    fn entries(&self) -> Vec<ConfigEntry> {
        vec![ConfigEntry {
            key: format!("{}.{}", Self::PREFIX, Self::EXPLICIT_BATCH_SIZE),
            value: Some(self.explicit_batch_size.to_string()),
            description: "The client chose its batch size, so a GPU-targeted \
                          cluster keeps it instead of raising it to the GPU default",
        }]
    }
}

impl ConfigExtension for ClientOptions {
    const PREFIX: &'static str = "oxidelake";
}

/// Whether `config`'s batch size is one the client asked for (#93): said so
/// explicitly, or set to anything but DataFusion's default (a `SET
/// datafusion.execution.batch_size`, or a client that predates the marker).
fn client_chose_batch_size(config: &SessionConfig) -> bool {
    config
        .options()
        .extensions
        .get::<ClientOptions>()
        .is_some_and(|client| client.explicit_batch_size)
        || config.batch_size() != SessionConfig::new().batch_size()
}

/// Ballista `ConfigProducer` yielding [`session_config`].
pub fn config_producer() -> ConfigProducer {
    Arc::new(session_config)
}

/// Builds the scheduler-side session state: Ballista's defaults plus the
/// placement rule targeting the declared cluster capability. This is where
/// `Gpu*Exec` nodes enter a distributed plan.
///
/// It is also where a cluster checks the Parquet files a query reads (#109).
/// Executors scan with DataFusion's own reader, so [`ParquetScanGuard`]
/// checks, while the scheduler plans, the metadata planning has just read
/// into this session's cache, and the local store is bounded like an
/// embedded session's, so that planning read cannot reserve memory for a
/// range a corrupt footer declares past the end of the file.
/// `config` is the client's (Ballista sends it with every query). A GPU
/// target raises its batch size to the GPU default unless the client chose
/// one, which it keeps (#93).
pub fn build_session_state(
    config: SessionConfig,
    target: BackendKind,
) -> datafusion::error::Result<SessionState> {
    let mut scalar_functions = ballista_scalar_functions();
    scalar_functions.extend(oxide_udfs());
    let config = if target.is_gpu() && !client_chose_batch_size(&config) {
        with_gpu_batch_size(config)
    } else {
        config
    };
    let runtime = Arc::new(RuntimeEnvBuilder::new().build()?);
    runtime.register_object_store(
        ObjectStoreUrl::local_filesystem().as_ref(),
        Arc::new(BoundedReadStore::new(default_object_store())),
    );
    let guard = ParquetScanGuard::new()
        .with_metadata_cache(runtime.cache_manager.get_file_metadata_cache());
    Ok(SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_scalar_functions(scalar_functions)
        .with_aggregate_functions(ballista_aggregate_functions())
        .with_window_functions(ballista_window_functions())
        .with_physical_optimizer_rules(physical_optimizer_rules(HardwarePlacementRule::new(target)))
        .with_physical_optimizer_rule(Arc::new(guard))
        .build())
}

/// Ballista `SessionBuilder` wrapping [`build_session_state`].
pub fn session_builder(target: BackendKind) -> SessionBuilder {
    Arc::new(move |config: SessionConfig| build_session_state(config, target))
}

/// Ballista `RuntimeProducer` for executors.
pub fn runtime_producer() -> RuntimeProducer {
    Arc::new(|_config: &SessionConfig| Ok(Arc::new(RuntimeEnvBuilder::new().build()?)))
}

/// Functions registered on every executor: Ballista's defaults plus
/// OxideLake's SQL UDFs (`l2_distance`, `cosine_distance`), so a distance
/// projection that was not lowered to a `Gpu*Exec` still runs on workers.
pub fn function_registry() -> BallistaFunctionRegistry {
    let mut registry = BallistaFunctionRegistry::default();
    for udf in oxide_udfs() {
        registry.scalar_functions.insert(udf.name().to_owned(), udf);
    }
    registry
}

/// Scheduler configuration with OxideLake's hooks installed.
pub fn scheduler_config(bind_host: &str, port: u16, target: BackendKind) -> SchedulerConfig {
    SchedulerConfig {
        bind_host: bind_host.to_owned(),
        bind_port: port,
        override_session_builder: Some(session_builder(target)),
        override_config_producer: Some(config_producer()),
        override_physical_codec: Some(oxide_codec()),
        ..SchedulerConfig::default()
    }
}

/// Runs a Ballista scheduler until it exits.
pub async fn run_scheduler(config: SchedulerConfig) -> Result<(), EngineError> {
    let address: SocketAddr = format!("{}:{}", config.bind_host, config.bind_port)
        .parse()
        .map_err(|e| EngineError::plan(format!("invalid scheduler bind address: {e}")))?;
    let config = Arc::new(config);
    let cluster = BallistaCluster::new_from_config(&config)
        .await
        .map_err(ballista_error)?;
    start_server(cluster, address, config)
        .await
        .map_err(ballista_error)
}

/// Executor configuration with OxideLake's hooks installed.
pub fn executor_config(
    scheduler_host: &str,
    scheduler_port: u16,
    port: u16,
    grpc_port: u16,
    concurrent_tasks: Option<usize>,
    work_dir: Option<String>,
) -> ExecutorProcessConfig {
    let defaults = ExecutorProcessConfig::default();
    ExecutorProcessConfig {
        scheduler_host: scheduler_host.to_owned(),
        scheduler_port,
        port,
        grpc_port,
        // Ballista 55 renamed `concurrent_tasks` to `vcores`: still the number
        // of tasks this executor runs at once, and `0` still means every
        // available core (`start_executor_process` resolves it).
        vcores: concurrent_tasks.unwrap_or(defaults.vcores),
        work_dir,
        override_physical_codec: Some(oxide_codec()),
        override_function_registry: Some(Arc::new(function_registry())),
        override_config_producer: Some(config_producer()),
        ..defaults
    }
}

/// Runs a Ballista executor until it exits.
pub async fn run_executor(config: ExecutorProcessConfig) -> Result<(), EngineError> {
    start_executor_process(Arc::new(config))
        .await
        .map_err(ballista_error)
}

/// Starts an in-process scheduler plus `executors` executors on ephemeral
/// localhost ports and returns the scheduler address. Everything runs on the
/// current tokio runtime and stops with it.
pub async fn start_standalone(
    target: BackendKind,
    executors: usize,
    concurrent_tasks: usize,
) -> Result<SocketAddr, EngineError> {
    let addr = new_standalone_scheduler_with_builder(
        session_builder(target),
        config_producer(),
        ballista_codec(),
    )
    .await
    .map_err(ballista_error)?;
    for _ in 0..executors.max(1) {
        let client = SchedulerGrpcClient::connect(format!("http://{addr}"))
            .await
            .map_err(ballista_error)?;
        new_standalone_executor_from_builder(
            client,
            concurrent_tasks.max(1),
            config_producer(),
            runtime_producer(),
            ballista_codec(),
            function_registry(),
        )
        .await
        .map_err(ballista_error)?;
    }
    tracing::info!(%addr, executors, "standalone Ballista cluster started");
    Ok(addr)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A cluster's Parquet checks live on the scheduler (#109): its executors
    /// scan with DataFusion's own reader, so if the scheduler's session lost
    /// the guard, its metadata cache or the bounded store, nothing in the
    /// cluster would check a file at all.
    #[test]
    fn the_scheduler_session_checks_parquet_files_and_bounds_reads() {
        let state = build_session_state(session_config(), BackendKind::CpuSimd).unwrap();
        let guards: Vec<String> = state
            .physical_optimizers()
            .iter()
            .filter(|rule| rule.name() == "ParquetScanGuard")
            .map(|rule| format!("{rule:?}"))
            .collect();
        assert_eq!(guards.len(), 1, "{guards:?}");
        assert!(
            guards[0].contains("checks_metadata_cache: true"),
            "{}",
            guards[0]
        );
        let store = state
            .runtime_env()
            .object_store(ObjectStoreUrl::local_filesystem())
            .unwrap();
        assert!(
            store.to_string().starts_with("BoundedReadStore("),
            "{store}"
        );
    use ballista_core::extension::SessionConfigHelperExt;
    use oxidelake_storage::GPU_BATCH_SIZE;

    /// The config the scheduler hands its session builder for a client's
    /// query: its own producer's, updated from the client's settings the way
    /// Ballista's `create_context` does it.
    fn as_the_scheduler_sees(client: &SessionConfig) -> SessionConfig {
        session_config().update_from_key_value_pair(&client.to_key_value_pairs())
    }

    fn client(explicit: bool, batch_size: usize) -> SessionConfig {
        session_config()
            .with_option_extension(ClientOptions {
                explicit_batch_size: explicit,
            })
            .with_batch_size(batch_size)
    }

    fn planned_batch_size(config: SessionConfig, target: BackendKind) -> usize {
        build_session_state(config, target)
            .unwrap()
            .config()
            .batch_size()
    }

    /// The marker survives Ballista's key/value trip to the scheduler. A key
    /// without its `oxidelake.` prefix would be dropped there at `debug`
    /// level, and every other assertion about the marker would then be
    /// testing a value the scheduler never receives.
    #[test]
    fn the_explicit_batch_size_marker_reaches_the_scheduler() {
        let seen = as_the_scheduler_sees(&client(true, 7));
        let options = seen.options().extensions.get::<ClientOptions>().unwrap();
        assert!(options.explicit_batch_size);
        assert_eq!(seen.batch_size(), 7);

        let seen = as_the_scheduler_sees(&client(false, 7));
        let options = seen.options().extensions.get::<ClientOptions>().unwrap();
        assert!(!options.explicit_batch_size);
    }

    /// A GPU cluster raises only a batch size nobody chose (#93).
    #[test]
    fn a_gpu_cluster_keeps_a_batch_size_the_client_chose() {
        let default = SessionConfig::new().batch_size();
        // Nobody chose: the GPU default.
        let seen = as_the_scheduler_sees(&client(false, default));
        assert_eq!(planned_batch_size(seen, BackendKind::Cuda), GPU_BATCH_SIZE);
        // Chosen, and equal to DataFusion's default: only the marker says so.
        let seen = as_the_scheduler_sees(&client(true, default));
        assert_eq!(planned_batch_size(seen, BackendKind::Cuda), default);
        // Not the default (`SET`, or a client from before the marker).
        let seen = as_the_scheduler_sees(&client(false, 7));
        assert_eq!(planned_batch_size(seen, BackendKind::Metal), 7);
        // A CPU cluster never raises it.
        let seen = as_the_scheduler_sees(&client(false, default));
        assert_eq!(planned_batch_size(seen, BackendKind::CpuSimd), default);
    }

    #[test]
    fn a_malformed_marker_is_refused_by_name() {
        let mut options = ClientOptions::default();
        let err = options.set("explicit_batch_size", "yes").unwrap_err();
        assert!(
            err.to_string().contains("oxidelake.explicit_batch_size"),
            "{err}"
        );
        let err = options.set("batch", "true").unwrap_err();
        assert!(err.to_string().contains("oxidelake.batch"), "{err}");
    }
}
