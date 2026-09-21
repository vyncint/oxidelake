//! `oxide-scheduler` — a Ballista scheduler with OxideLake's placement rule and plan codec.

use clap::Parser;
use oxidelake_core::BackendKind;
use oxidelake_runtime::cluster;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "oxide-scheduler",
    version,
    about = "OxideLake scheduler (Apache DataFusion Ballista)"
)]
struct Cli {
    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1")]
    bind_host: String,
    /// gRPC port.
    #[arg(long, default_value_t = 50050)]
    port: u16,
    /// Declared cluster capability used for placement: cpu, cuda or metal.
    /// Defaults to OXIDE_CLUSTER_BACKEND, then cpu.
    #[arg(long)]
    cluster_backend: Option<String>,
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
    let target: BackendKind = match cli.cluster_backend {
        Some(name) => name.parse()?,
        None => oxidelake_planner::cluster_target_from_env()?,
    };
    tracing::info!(%target, "placement target for this cluster");
    serve_metrics(cli.metrics_port, &cli.metrics_host).await?;
    cluster::run_scheduler(cluster::scheduler_config(&cli.bind_host, cli.port, target)).await?;
    Ok(())
}

/// Starts the Prometheus endpoint when asked for (#33).
///
/// A scheduler plans but does not execute, so its hub holds the counters of
/// whatever ran in *this* process — today, nothing. The endpoint is here so a
/// deployment can scrape every OxideLake process the same way rather than
/// special-casing one of them, and it says so by serving an empty set instead
/// of refusing.
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
