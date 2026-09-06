//! Cluster mode on Apache DataFusion Ballista (ADR-0013).
//!
//! OxideLake does not write a scheduler. This module builds the pieces Ballista
//! lets us plug in — a session builder that installs the placement rule for the
//! declared cluster capability, a physical codec that ships `Gpu*Exec` nodes,
//! and the executor function registry — and wires them into Ballista's
//! scheduler and executor processes, plus an in-process standalone cluster for
//! tests and the CLI.

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
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionState;
use datafusion::prelude::SessionConfig;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use oxidelake_compute::oxide_udfs;
use oxidelake_core::{BackendKind, EngineError};
use oxidelake_planner::{HardwarePlacementRule, OxidePhysicalCodec, physical_optimizer_rules};
use oxidelake_storage::{with_gpu_batch_size, with_pruning};

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
}

/// Ballista `ConfigProducer` yielding [`session_config`].
pub fn config_producer() -> ConfigProducer {
    Arc::new(session_config)
}

/// Builds the scheduler-side session state: Ballista's defaults plus the
/// placement rule targeting the declared cluster capability. This is where
/// `Gpu*Exec` nodes enter a distributed plan.
pub fn build_session_state(
    config: SessionConfig,
    target: BackendKind,
) -> datafusion::error::Result<SessionState> {
    let mut scalar_functions = ballista_scalar_functions();
    scalar_functions.extend(oxide_udfs());
    let config = if target.is_gpu() {
        with_gpu_batch_size(config)
    } else {
        config
    };
    Ok(SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(Arc::new(RuntimeEnvBuilder::new().build()?))
        .with_scalar_functions(scalar_functions)
        .with_aggregate_functions(ballista_aggregate_functions())
        .with_window_functions(ballista_window_functions())
        .with_physical_optimizer_rules(physical_optimizer_rules(HardwarePlacementRule::new(target)))
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
        concurrent_tasks: concurrent_tasks.unwrap_or(defaults.concurrent_tasks),
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
