//! Corrupt Parquet comes back as `EngineError::Format`: never a panic, an
//! abort, or an allocation the size of whatever the file declares.
//!
//! Every case runs in a child process under an address-space cap. The
//! failure guarded against includes allocations sized by a corrupt length
//! field: uncapped, a regression would try to take the machine's memory
//! instead of failing the test. Under the cap the allocation fails, the
//! child aborts, and the parent reports which case did it. RLIMIT_AS is only
//! enforced on Linux, so the file is Linux-only rather than uncapped
//! anywhere.

#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use oxidelake_core::EngineError;
use oxidelake_storage::{
    Compression, ParquetWriteOptions, classify_error, pruning_session, register_parquet_table,
    write_parquet,
};

/// Prefix of every line the child reports, so the parent can find them in
/// the test harness's own output.
const RECORD: &str = "@@";

/// The variable that turns [`child`] from a no-op into one case.
const CASE_VAR: &str = "OXIDELAKE_CORRUPT_PARQUET_CASE";
const DIR_VAR: &str = "OXIDELAKE_CORRUPT_PARQUET_DIR";

/// 4 GB of address space, the cap the 0.3.0 review ran its probes under.
const ADDRESS_SPACE_CAP: u64 = 4_000_000_000;

/// What a child may use at most, in KiB of peak RSS: the file is ~100 KB, so
/// anything near this means a declared length was believed.
const RSS_LIMIT_KIB: i64 = 512 * 1024;

const ROWS: i64 = 10_000;

/// The file `oxide gen-data` would write for one Int64 column: OxideLake's
/// writer settings, page statistics on, so the page index is present.
fn write_valid(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from_iter_values(0..ROWS))],
    )
    .unwrap();
    let opts = ParquetWriteOptions {
        compression: Compression::None,
        ..Default::default()
    };
    write_parquet(path, schema, &[batch], &opts).unwrap();
}

/// Builds the bytes of one case from a fresh valid file.
fn corrupt(case: &str, dir: &Path) -> Vec<u8> {
    let base = dir.join("base.parquet");
    write_valid(&base);
    let file = std::fs::read(&base).unwrap();
    match case {
        "ok" => file,
        // The footer is intact; the first page header is garbage, so the
        // file registers and only the scan can find out (#118).
        "bad-page" => {
            let mut out = file;
            out[4..36].fill(0xFF);
            out
        }
        other => panic!("unknown case {other}"),
    }
}

fn outcome(result: &Result<(), EngineError>) -> String {
    match result {
        Ok(()) => "Ok".into(),
        Err(EngineError::Format(m)) => format!("Format {m}"),
        Err(EngineError::Io(e)) => format!("Io {e}"),
        Err(other) => format!("Other {other}"),
    }
}

async fn query(ctx: &SessionContext, sql: &str) -> Result<(), EngineError> {
    let batches = ctx
        .sql(sql)
        .await
        .map_err(classify_error)?
        .collect()
        .await
        .map_err(classify_error)?;
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    println!("{RECORD}ROWS {rows}");
    Ok(())
}

fn max_rss_kib() -> i64 {
    // SAFETY: getrusage writes into the zeroed struct it is given.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage.ru_maxrss
    }
}

/// The three query shapes that reach a file differently: a full aggregate
/// (scan tasks spawned per partition), a `LIMIT` (read on the caller's
/// thread), and a range the page index can narrow.
async fn query_all(ctx: &SessionContext) {
    for (key, sql) in [
        ("SUM", "SELECT SUM(k) FROM t"),
        ("LIMIT", "SELECT k FROM t LIMIT 3"),
        ("RANGE", "SELECT k FROM t WHERE k BETWEEN 600 AND 700"),
    ] {
        println!("{RECORD}{key} {}", outcome(&query(ctx, sql).await));
    }
}

/// The child half: a no-op in a normal test run, one case when the parent
/// re-executes this binary with [`CASE_VAR`] set.
///
/// `register:<case>` registers the case's file and, if registration lets it
/// through, queries it.
#[test]
fn child() {
    let Ok(spec) = std::env::var(CASE_VAR) else {
        return;
    };
    let dir = PathBuf::from(std::env::var(DIR_VAR).unwrap());
    let (when, case) = spec.split_once(':').unwrap();
    let bytes = corrupt(case, &dir);
    let path = dir.join("t.parquet");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let ctx = pruning_session();
        match when {
            "register" => {
                std::fs::write(&path, &bytes).unwrap();
                let registered = register_parquet_table(&ctx, "t", path.to_str().unwrap()).await;
                println!("{RECORD}REGISTER {}", outcome(&registered));
                if registered.is_ok() {
                    query_all(&ctx).await;
                }
            }
            other => panic!("unknown phase {other}"),
        }
    });
    println!("{RECORD}MAXRSS_KIB {}", max_rss_kib());
}

struct Run {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

impl Run {
    /// The value the child reported under `key`. The harness prints
    /// "test child ... " before the child's first line, so a record is
    /// found by its marker rather than at the start of a line.
    fn line(&self, key: &str) -> Option<&str> {
        let marker = format!("{RECORD}{key} ");
        self.stdout
            .lines()
            .find_map(|l| l.split_once(marker.as_str()).map(|(_, rest)| rest))
    }

    fn explain(&self) -> String {
        let stderr: Vec<&str> = self.stderr.lines().collect();
        let tail = stderr[stderr.len().saturating_sub(15)..].join("\n");
        format!(
            "status {:?}\n--- stdout\n{}\n--- stderr (tail)\n{tail}",
            self.status, self.stdout
        )
    }
}

/// Runs one case in a capped child and waits (bounded) for it.
fn run_capped(spec: &str) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["child", "--exact", "--nocapture", "--test-threads=1"])
        .env(CASE_VAR, spec)
        .env(DIR_VAR, dir.path())
        // One malloc arena per thread reserves 64 MiB of address space each;
        // under the cap that is space the case under test cannot use.
        .env("MALLOC_ARENA_MAX", "2")
        .env("RUST_BACKTRACE", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // SAFETY: setrlimit is async-signal-safe and affects only the child.
    unsafe {
        command.pre_exec(|| {
            let cap = libc::rlimit {
                rlim_cur: ADDRESS_SPACE_CAP,
                rlim_max: ADDRESS_SPACE_CAP,
            };
            if libc::setrlimit(libc::RLIMIT_AS, &cap) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command.spawn().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let out = std::thread::spawn(move || {
        let mut s = String::new();
        stdout.read_to_string(&mut s).unwrap();
        s
    });
    let err = std::thread::spawn(move || {
        let mut s = String::new();
        stderr.read_to_string(&mut s).unwrap();
        s
    });
    // Generous: a loaded machine is slow, and a hung child is a failure, not
    // something to wait on forever.
    let deadline = Instant::now() + Duration::from_secs(300);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            break child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    Run {
        status,
        stdout: out.join().unwrap(),
        stderr: err.join().unwrap(),
    }
}

/// The child exited on its own (no panic, no abort) inside the memory limit.
fn assert_clean(case: &str, run: &Run) {
    assert!(
        run.status.success(),
        "{case}: the child did not exit cleanly\n{}",
        run.explain()
    );
    let rss: i64 = run
        .line("MAXRSS_KIB")
        .unwrap_or_else(|| panic!("{case}: no MAXRSS line\n{}", run.explain()))
        .parse()
        .unwrap();
    assert!(
        rss < RSS_LIMIT_KIB,
        "{case}: peak RSS {rss} KiB for a ~100 KB file\n{}",
        run.explain()
    );
}

fn assert_format(case: &str, run: &Run, key: &str, needles: &[&str]) {
    let line = run
        .line(key)
        .unwrap_or_else(|| panic!("{case}: no {key} line\n{}", run.explain()));
    assert!(
        line.starts_with("Format "),
        "{case}: {key} was not a Format error: {line}\n{}",
        run.explain()
    );
    for needle in needles {
        assert!(
            line.contains(needle),
            "{case}: {key} does not name {needle:?}: {line}"
        );
    }
}

/// The harness itself: a valid file registers and queries in the child.
#[test]
fn a_valid_file_registers_and_scans_in_the_capped_child() {
    let run = run_capped("register:ok");
    assert_clean("ok", &run);
    for key in ["REGISTER", "SUM", "LIMIT", "RANGE"] {
        assert_eq!(run.line(key), Some("Ok"), "{key}\n{}", run.explain());
    }
}

/// Corruption the footer cannot reveal is found by the scan, and comes back
/// classified (#118): before, two of the three query shapes returned
/// `EngineError::DataFusion(..)` because `classify_error` did not look
/// inside the wrapper a repartitioned scan puts around its error.
#[test]
fn a_corrupt_page_found_by_the_scan_is_a_format_error() {
    let run = run_capped("register:bad-page");
    assert_clean("bad-page", &run);
    assert_eq!(run.line("REGISTER"), Some("Ok"), "{}", run.explain());
    for key in ["SUM", "LIMIT", "RANGE"] {
        assert_format("bad-page", &run, key, &["struct field type"]);
    }
}
