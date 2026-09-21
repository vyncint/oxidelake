//! `oxide` — the OxideLake command-line interface.

use std::io::{self, Write};

use clap::{Parser, Subcommand};
use oxidelake_core::BackendKind;
use oxidelake_runtime::{OutputFormat, OxideSession, SessionOptions, dashboard};
use oxidelake_storage::{Compression, demo_write_options, write_demo_table};
use tracing_subscriber::EnvFilter;

/// Writes `text` to stdout, treating a closed reader as success (#34).
///
/// `println!` panics when the pipe it is writing to has gone — which is
/// what `oxide sql … | head -1` does the moment `head` has its line. A
/// query tool whose primary use is a shell pipeline must not treat the
/// most ordinary thing in a pipeline as a crash: SECURITY.md's "no panics"
/// bullet and ADR-0008 both say so. The write is made by hand so the
/// `BrokenPipe` is a value to match on rather than a panic inside the
/// formatting machinery.
fn write_out(text: &str) -> Result<(), io::Error> {
    let mut stdout = io::stdout().lock();
    match stdout
        .write_all(text.as_bytes())
        .and_then(|()| stdout.flush())
    {
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        other => other,
    }
}

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
        /// Rows per record batch, overriding the CPU default (8192) and the
        /// GPU-target default (65536). Batch boundaries never change results.
        #[arg(long, value_name = "ROWS")]
        batch_size: Option<usize>,
        /// How to print the result: table, json or csv.
        #[arg(long, value_name = "FORMAT", default_value = "table")]
        output: OutputFormat,
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
        /// Rows per record batch (see `oxide sql --batch-size`), so the plan
        /// is built under the same session configuration as the `oxide sql`
        /// run it explains.
        #[arg(long, value_name = "ROWS")]
        batch_size: Option<usize>,
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
        /// Rows per record batch (see `oxide sql --batch-size`).
        #[arg(long, value_name = "ROWS")]
        batch_size: Option<usize>,
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
    batch_size: Option<usize>,
    tables: &[String],
) -> anyhow::Result<OxideSession> {
    let mut options = SessionOptions::new();
    if let Some(rows) = batch_size {
        options = options.with_batch_size(rows);
    }
    let session = match (cluster, target) {
        (Some(_), Some(_)) => anyhow::bail!(
            "--target picks the embedded placement target; on a cluster the \
             scheduler's OXIDE_CLUSTER_BACKEND decides placement"
        ),
        (Some(url), None) => OxideSession::connect_with_options(url, &options).await?,
        (None, target) => {
            if let Some(target) = target {
                options = options.with_target(target);
            }
            OxideSession::local_with_options(&options)?
        }
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
        // Colour only when a person is reading. Logs are usually redirected
        // to a file or a collector, where escape codes are noise that also
        // breaks a grep for `field=value` (#33).
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
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
            write_out(&format!(
                "wrote {} rows to {} (seed {seed}, {row_group_rows} rows per row group, {})\n",
                table.rows,
                table.path.display(),
                compression.as_str(),
            ))?;
        }
        Command::Sql {
            query,
            tables,
            cluster,
            target,
            batch_size,
            output,
        } => {
            let session = session(cluster.as_deref(), target, batch_size, &tables).await?;
            // Not `DataFrame::show()`: that is `println!` inside DataFusion,
            // so a closed pipe panics there and the CLI never sees it (#34).
            // `session.collect` rather than the DataFrame's: it logs the one
            // INFO line per query that `RUST_LOG=info` is for (#33), and
            // hands back the schema an empty result still has to print.
            let (schema, batches) = session.collect(&query).await?;
            write_out(&oxidelake_runtime::output::render(
                &schema, &batches, output,
            )?)?;
        }
        Command::Explain {
            query,
            tables,
            target,
            batch_size,
        } => {
            let session = session(None, target, batch_size, &tables).await?;
            write_out(&session.explain(&query).await?)?;
        }
        Command::Tui {
            query,
            tables,
            target,
            batch_size,
        } => {
            if !oxidelake_tui::is_interactive_terminal() {
                eprintln!("{}", oxidelake_tui::NON_INTERACTIVE_TERMINAL_MESSAGE);
                std::process::exit(2);
            }
            let model = match &query {
                Some(sql) => {
                    let session = session(None, target, batch_size, &tables).await?;
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
