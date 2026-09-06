//! `oxide` — the OxideLake command-line interface.

use clap::{Parser, Subcommand};
use oxidelake_core::BackendKind;
use oxidelake_runtime::{OxideSession, dashboard};
use oxidelake_storage::{Compression, demo_write_options, write_demo_table};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "oxide",
    version,
    about = "OxideLake: GPU-accelerated, Arrow-native query engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Write the deterministic demo table (id, k, v, s, emb) as Parquet.
    GenData {
        /// Rows to generate.
        #[arg(long, default_value_t = 1_000_000)]
        rows: u64,
        /// Output directory (created if needed); writes `<out>/t.parquet`.
        #[arg(long)]
        out: String,
        /// PRNG seed; the same seed always writes the same file.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Rows per Parquet row group.
        #[arg(long, default_value_t = 65_536)]
        row_group_rows: usize,
        /// Compression codec: none, lz4 or zstd.
        #[arg(long, default_value = "zstd")]
        compression: Compression,
    },
    /// Run a SQL statement and print the result.
    Sql {
        /// The SQL statement to execute.
        #[arg(short, long)]
        query: String,
        /// Register a Parquet file or directory as a table before running (`name=path`, repeatable).
        #[arg(short, long = "table", value_name = "NAME=PATH")]
        tables: Vec<String>,
        /// Run on a Ballista cluster instead of in-process (`df://host:port`).
        #[arg(long, value_name = "URL")]
        cluster: Option<String>,
        /// Plan for this backend (cpu, cuda or metal) instead of the detected
        /// one. Execution still uses the hardware that is present — operators
        /// planned for an absent GPU take the CPU path. Embedded mode only.
        #[arg(long, value_name = "BACKEND")]
        target: Option<BackendKind>,
    },
    /// Print the physical plan for a SQL statement (with placement tags in embedded mode).
    Explain {
        /// The SQL statement to plan.
        #[arg(short, long)]
        query: String,
        /// Register a Parquet file or directory as a table before planning (`name=path`, repeatable).
        #[arg(short, long = "table", value_name = "NAME=PATH")]
        tables: Vec<String>,
        /// Plan for this backend (cpu, cuda or metal) instead of the detected one.
        #[arg(long, value_name = "BACKEND")]
        target: Option<BackendKind>,
    },
    /// Open the terminal dashboard: run a query and inspect its plan,
    /// telemetry and column profiles — or a synthetic demo without a query.
    Tui {
        /// A SQL statement to execute and inspect.
        #[arg(short, long)]
        query: Option<String>,
        /// Register a Parquet file or directory as a table first (`name=path`, repeatable).
        #[arg(short, long = "table", value_name = "NAME=PATH")]
        tables: Vec<String>,
        /// Plan for this backend (cpu, cuda or metal) instead of the detected one.
        #[arg(long, value_name = "BACKEND")]
        target: Option<BackendKind>,
    },
}

fn parse_tables(tables: &[String]) -> anyhow::Result<Vec<(&str, &str)>> {
    tables
        .iter()
        .map(|spec| {
            spec.split_once('=')
                .ok_or_else(|| anyhow::anyhow!("--table expects NAME=PATH, got '{spec}'"))
        })
        .collect()
}

async fn session(
    cluster: Option<&str>,
    target: Option<BackendKind>,
    tables: &[String],
) -> anyhow::Result<OxideSession> {
    let session = match (cluster, target) {
        (Some(_), Some(_)) => anyhow::bail!(
            "--target picks the embedded placement target; on a cluster the \
             scheduler's OXIDE_CLUSTER_BACKEND decides placement"
        ),
        (Some(url), None) => OxideSession::connect(url).await?,
        (None, Some(target)) => OxideSession::local_with_target(target)?,
        (None, None) => OxideSession::local()?,
    };
    for (name, path) in parse_tables(tables)? {
        session.register_parquet(name, path).await?;
    }
    Ok(session)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::GenData {
            rows,
            out,
            seed,
            row_group_rows,
            compression,
        } => {
            let options = demo_write_options(row_group_rows, compression);
            let table = write_demo_table(std::path::Path::new(&out), rows, seed, &options)?;
            println!(
                "wrote {} rows to {} (seed {seed}, {row_group_rows} rows per row group, {})",
                table.rows,
                table.path.display(),
                compression.as_str(),
            );
        }
        Command::Sql {
            query,
            tables,
            cluster,
            target,
        } => {
            let session = session(cluster.as_deref(), target, &tables).await?;
            session.sql(&query).await?.show().await?;
        }
        Command::Explain {
            query,
            tables,
            target,
        } => {
            let session = session(None, target, &tables).await?;
            print!("{}", session.explain(&query).await?);
        }
        Command::Tui {
            query,
            tables,
            target,
        } => {
            let model = match &query {
                Some(sql) => {
                    let session = session(None, target, &tables).await?;
                    let names: Vec<String> = parse_tables(&tables)?
                        .into_iter()
                        .map(|(name, _)| name.to_owned())
                        .collect();
                    dashboard::query_dashboard(&session, sql, &names).await?
                }
                None => oxidelake_tui::demo_model(),
            };
            oxidelake_tui::run_terminal(model, None, |_| {})?;
        }
    }
    Ok(())
}
