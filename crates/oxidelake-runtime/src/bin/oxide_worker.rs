//! `oxide-worker` — a Ballista executor that can run OxideLake's `Gpu*Exec` operators.

use clap::Parser;
use oxidelake_core::BackendKind;
use oxidelake_runtime::cluster;
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
    // Selected before anything runs: operators read the memoised
    // `local_backend()`, so a choice made after the first task would be
    // accepted and then ignored.
    let backend = match cli.backend {
        Some(kind) => oxidelake_compute::init_local_backend(Some(kind.as_str()))?,
        None => oxidelake_compute::local_backend()?,
    };
    tracing::info!(backend = %backend.kind(), "worker backend");
    serve_metrics(cli.metrics_port, &cli.metrics_host).await?;
    cluster::run_executor(cluster::executor_config(
        &cli.scheduler_host,
        cli.scheduler_port,
        cli.port,
        cli.grpc_port,
        cli.concurrent_tasks,
        cli.work_dir,
    ))
    .await?;
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
