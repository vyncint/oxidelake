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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
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
