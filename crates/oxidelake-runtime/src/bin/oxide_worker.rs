//! `oxide-worker` — a Ballista executor that can run OxideLake's `Gpu*Exec` operators.

use std::time::Duration;

use clap::Parser;
use oxidelake_core::BackendKind;
use oxidelake_runtime::cluster;
use oxidelake_runtime::shutdown::{ShutdownSignal, Stopped};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "oxide-worker",
    version,
    about = "OxideLake worker (Apache DataFusion Ballista executor)"
)]
struct Cli {
    /// Scheduler host.
    #[arg(long, default_value = "localhost")]
    scheduler_host: String,
    /// Scheduler port.
    #[arg(long, default_value_t = 50050)]
    scheduler_port: u16,
    /// Arrow Flight port for shuffle data.
    #[arg(long, default_value_t = 50051)]
    port: u16,
    /// gRPC control port.
    #[arg(long, default_value_t = 50052)]
    grpc_port: u16,
    /// Concurrent tasks (default: available parallelism).
    #[arg(long)]
    concurrent_tasks: Option<usize>,
    /// Directory for shuffle files (default: a temporary directory).
    #[arg(long)]
    work_dir: Option<String>,
    /// Backend this worker executes on (cpu, cuda or metal), overriding
    /// `OXIDE_BACKEND`. A backend this machine cannot provide is a startup
    /// error, never a silent fall back to the CPU — a worker that reports
    /// GPU capacity it does not have poisons the whole cluster's placement.
    #[arg(long, value_name = "BACKEND")]
    backend: Option<BackendKind>,
    /// Serve Prometheus metrics on this port of `--metrics-host` (requires
    /// the `metrics` feature). Unauthenticated: bind it on a private
    /// interface, as the Ballista ports already are.
    #[arg(long, value_name = "PORT")]
    metrics_port: Option<u16>,
    /// Address the metrics endpoint binds to.
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    metrics_host: String,
    /// Seconds between this worker's heartbeats. The scheduler presumes a
    /// worker lost after its `--worker-timeout` (180 s by default) without
    /// one, so keep this well below that.
    #[arg(long, default_value_t = 60, value_name = "SECS", value_parser = clap::value_parser!(u64).range(1..))]
    heartbeat_interval: u64,
    /// On SIGTERM or SIGINT, take no new tasks and give the running ones this
    /// long to finish, then exit 0; the scheduler runs them again elsewhere.
    /// A second signal exits at once.
    #[arg(long, default_value_t = 20, value_name = "SECS")]
    drain_timeout: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        // Colour only when a person is reading. Logs are usually redirected
        // to a file or a collector, where escape codes are noise that also
        // breaks a grep for `field=value` (#33).
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .init();
    let cli = Cli::parse();
    // Before the worker joins, so a stop request during startup is held for
    // the shutdown below rather than killing the process.
    let signal = ShutdownSignal::listen()?;
    // Selected before anything runs: operators read the memoised
    // `local_backend()`, so a choice made after the first task would be
    // accepted and then ignored.
    let backend = match cli.backend {
        Some(kind) => oxidelake_compute::init_local_backend(Some(kind.as_str()))?,
        None => oxidelake_compute::local_backend()?,
    };
    tracing::info!(
        backend = %backend.kind(),
        build = %oxidelake_planner::codec::build_description(),
        "worker backend"
    );
    serve_metrics(cli.metrics_port, &cli.metrics_host).await?;
    let mut config = cluster::executor_config(
        &cli.scheduler_host,
        cli.scheduler_port,
        cli.port,
        cli.grpc_port,
        cli.concurrent_tasks,
        cli.work_dir,
    );
    config.executor_heartbeat_interval_seconds = cli.heartbeat_interval;
    let stopped =
        cluster::run_executor_with_shutdown(config, signal, Duration::from_secs(cli.drain_timeout))
            .await?;
    if stopped == Stopped::Abandoned {
        // Returning would drop the runtime, which waits for the tasks still
        // running: the very wait the drain timeout bounds.
        std::process::exit(0);
    }
    Ok(())
}

/// Starts the Prometheus endpoint when asked for, over the process-wide hub
/// the plan codec reports into (#33).
#[cfg(feature = "metrics")]
async fn serve_metrics(port: Option<u16>, host: &str) -> anyhow::Result<()> {
    use std::net::ToSocketAddrs;

    let Some(port) = port else { return Ok(()) };
    let addr = (host, port)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| anyhow::anyhow!("--metrics-host '{host}' resolved to no address"))?;
    let hub = std::sync::Arc::clone(oxidelake_core::telemetry::TelemetryHub::global());
    let (bound, server) = oxidelake_runtime::metrics::serve(addr, hub).await?;
    tracing::info!(%bound, path = oxidelake_runtime::metrics::METRICS_PATH, "serving metrics");
    tokio::spawn(server);
    Ok(())
}

/// Without the feature the flag is still parsed, and refused rather than
/// ignored: a deployment that asked for metrics and silently got none is the
/// failure this whole issue is about.
#[cfg(not(feature = "metrics"))]
async fn serve_metrics(port: Option<u16>, _host: &str) -> anyhow::Result<()> {
    match port {
        None => Ok(()),
        Some(_) => anyhow::bail!(
            "--metrics-port needs the `metrics` feature; rebuild with \
             `cargo build -p oxidelake-runtime --features metrics`"
        ),
    }
}
