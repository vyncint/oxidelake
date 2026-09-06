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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    let target: BackendKind = match cli.cluster_backend {
        Some(name) => name.parse()?,
        None => oxidelake_planner::cluster_target_from_env()?,
    };
    tracing::info!(%target, "placement target for this cluster");
    cluster::run_scheduler(cluster::scheduler_config(&cli.bind_host, cli.port, target)).await?;
    Ok(())
}
