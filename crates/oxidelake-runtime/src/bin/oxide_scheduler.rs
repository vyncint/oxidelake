//! `oxide-scheduler` — a Ballista scheduler with OxideLake's placement rule and plan codec.

use std::time::Duration;

use ballista_scheduler::config::SchedulerConfig;
use clap::Parser;
use oxidelake_core::BackendKind;
use oxidelake_runtime::cluster;
use oxidelake_runtime::exposure::{check_listen_address, warn_unless_loopback};
use oxidelake_runtime::shutdown::ShutdownSignal;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "oxide-scheduler",
    version,
    about = "OxideLake scheduler (Apache DataFusion Ballista)"
)]
struct Cli {
    /// Address to bind: loopback or private (10/8, 172.16/12, 192.168/16,
    /// fc00::/7). The port has no authentication or encryption, so a public
    /// address, or every address (0.0.0.0, ::), needs
    /// `--allow-insecure-bind`.
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
    /// Address the metrics endpoint binds to, held to the same rule as
    /// `--bind-host`.
    #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
    metrics_host: String,
    /// Listen on a public address, or on every address, although the ports
    /// have no authentication or encryption: say so only when a firewall or
    /// an authenticating proxy already decides who can connect (SECURITY.md).
    #[arg(long)]
    allow_insecure_bind: bool,
    /// Times one task may fail before its query fails; 1 means no retry
    /// (Ballista's `task_max_failures`). A task lost with its worker is run
    /// again elsewhere without counting.
    #[arg(long, default_value_t = 4, value_name = "N", value_parser = at_least_one)]
    task_max_failures: usize,
    /// Times a stage may fail because a shuffle fetch failed, its input lost
    /// with a worker, before its query fails (Ballista's
    /// `stage_max_failures`). The default, 1, fails the query at the first
    /// such failure, naming the worker; a worker that is stopped or goes
    /// silent fails its queries whatever this says (#148). Raising it turns
    /// on Ballista 55's re-run of the stage, which can return duplicated rows
    /// with exit 0 or never finish (fixed upstream by
    /// apache/datafusion-ballista#2677, unreleased).
    #[arg(
        long,
        default_value_t = cluster::DEFAULT_STAGE_MAX_FAILURES,
        value_name = "N",
        value_parser = at_least_one
    )]
    stage_max_failures: usize,
    /// Seconds a worker may go without a heartbeat before it is presumed lost
    /// and its tasks run elsewhere. Until then a query holding tasks on a dead
    /// worker waits. Keep it well above the workers' `--heartbeat-interval`.
    #[arg(
        long,
        default_value_t = cluster::DEFAULT_WORKER_TIMEOUT.as_secs(),
        value_name = "SECS",
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    worker_timeout: u64,
    /// On SIGTERM or SIGINT, refuse new queries and give the running ones
    /// this long to finish, then exit 0. A second signal exits at once.
    #[arg(long, default_value_t = 20, value_name = "SECS")]
    drain_timeout: u64,
}

/// A retry limit of zero would mean what one means, so it is refused rather
/// than given a second spelling.
fn at_least_one(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(0) => Err("must be at least 1 (1 means no retry)".to_owned()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

/// The scheduler `cli` asks for, planning for `target`.
fn config(cli: &Cli, target: BackendKind) -> SchedulerConfig {
    let config = cluster::with_worker_timeout(
        cluster::scheduler_config(&cli.bind_host, cli.port, target),
        Duration::from_secs(cli.worker_timeout),
    );
    SchedulerConfig {
        task_max_failures: cli.task_max_failures,
        stage_max_failures: cli.stage_max_failures,
        ..config
    }
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
    // Before anything is served, so a stop request during startup is held
    // for the shutdown below rather than killing the process.
    let signal = ShutdownSignal::listen()?;
    // Before anything listens: the ports are unauthenticated, so where they
    // listen is the whole of the security boundary (#30).
    let exposure = check_listen_address("--bind-host", &cli.bind_host, cli.allow_insecure_bind)?;
    if cli.metrics_port.is_some() {
        check_listen_address("--metrics-host", &cli.metrics_host, cli.allow_insecure_bind)?;
    }
    warn_unless_loopback("the scheduler", &cli.bind_host, exposure);
    let target: BackendKind = match &cli.cluster_backend {
        Some(name) => name.parse()?,
        None => oxidelake_planner::cluster_target_from_env()?,
    };
    tracing::info!(%target, "placement target for this cluster");
    // Workers are admitted only if they are this build (#92), so the log
    // says which build that is.
    tracing::info!(
        build = %oxidelake_planner::codec::build_description(),
        "admitting workers of this build only"
    );
    serve_metrics(cli.metrics_port, &cli.metrics_host).await?;
    cluster::run_scheduler_with_shutdown(
        config(&cli, target),
        signal,
        Duration::from_secs(cli.drain_timeout),
    )
    .await?;
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("oxide-scheduler").chain(args.iter().copied()))
    }

    /// The flags reach the configuration Ballista's scheduler runs with (#31).
    #[test]
    fn the_retry_and_timeout_flags_reach_the_scheduler() {
        let cli = parse(&[
            "--task-max-failures",
            "2",
            "--stage-max-failures",
            "1",
            "--worker-timeout",
            "9",
        ])
        .unwrap();
        let config = config(&cli, BackendKind::CpuSimd);
        assert_eq!(config.task_max_failures, 2);
        assert_eq!(config.stage_max_failures, 1);
        assert_eq!(config.executor_timeout_seconds, 9);
        assert_eq!(config.expire_dead_executor_interval_seconds, 3);
        assert_eq!(config.no_executors_grace_period_seconds, 9);
    }

    /// Unset, they are Ballista's own defaults, save one: a stage is not run
    /// again after a lost worker's shuffle output could not be fetched
    /// (#148), where Ballista would try 4 times.
    #[test]
    fn the_defaults_are_ballistas_but_a_lost_worker_is_not_retried() {
        let ballista = SchedulerConfig::default();
        let config = config(&parse(&[]).unwrap(), BackendKind::CpuSimd);
        assert_eq!(config.task_max_failures, ballista.task_max_failures);
        assert_eq!(ballista.stage_max_failures, 4, "Ballista's default moved");
        assert_eq!(config.stage_max_failures, 1);
        assert_eq!(
            config.executor_timeout_seconds,
            ballista.executor_timeout_seconds
        );
        assert_eq!(
            config.expire_dead_executor_interval_seconds,
            ballista.expire_dead_executor_interval_seconds
        );
        assert_eq!(
            config.no_executors_grace_period_seconds,
            ballista.no_executors_grace_period_seconds
        );
    }

    #[test]
    fn a_zero_retry_limit_or_timeout_is_refused() {
        for flag in ["--task-max-failures", "--stage-max-failures"] {
            let err = parse(&[flag, "0"]).err().expect(flag).to_string();
            assert!(err.contains("1 means no retry"), "{err}");
        }
        assert!(parse(&["--worker-timeout", "0"]).is_err());
    }
}
