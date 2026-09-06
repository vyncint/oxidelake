//! End-to-end CLI tests (docs/SPEC.md Phase 7): `oxide gen-data` writes Parquet;
//! `oxide sql` prints results that match an *independent* read of that Parquet
//! (the `parquet` crate directly — no DataFusion, no placement rule);
//! `oxide explain --target` shows placement tags; and the same query on a real
//! spawned `oxide-scheduler` + `oxide-worker` cluster prints exactly the
//! embedded output.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs::File;
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command as StdCommand, Stdio};
use std::time::{Duration, Instant};

use arrow::array::{Array, Float64Array, Int64Array};
use assert_cmd::Command;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const QUERY: &str = "SELECT k, SUM(v) AS sv FROM t GROUP BY k ORDER BY k";

fn oxide() -> Command {
    Command::cargo_bin("oxide").unwrap()
}

fn gen_data(dir: &Path, rows: u64) {
    let assert = oxide()
        .args([
            "gen-data",
            "--rows",
            &rows.to_string(),
            "--out",
            dir.to_str().unwrap(),
            "--row-group-rows",
            "4096",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains(&format!("wrote {rows} rows")), "{stdout}");
}

/// `k -> SUM(v)` computed straight off the Parquet file with the `parquet`
/// crate — the ground truth the engine's output must match.
fn expected_sums(dir: &Path) -> BTreeMap<Option<i64>, f64> {
    let file = File::open(dir.join("t.parquet")).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let mut sums: BTreeMap<Option<i64>, f64> = BTreeMap::new();
    for batch in reader {
        let batch = batch.unwrap();
        let k = batch
            .column(batch.schema().index_of("k").unwrap())
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .clone();
        let v = batch
            .column(batch.schema().index_of("v").unwrap())
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .clone();
        for row in 0..batch.num_rows() {
            if v.is_null(row) {
                continue;
            }
            let key = (!k.is_null(row)).then(|| k.value(row));
            *sums.entry(key).or_insert(0.0) += v.value(row);
        }
    }
    sums
}

/// Parses `| k | sv |` rows out of the CLI's pretty-printed table.
fn parse_table(stdout: &str) -> BTreeMap<Option<i64>, f64> {
    let mut rows = BTreeMap::new();
    for line in stdout.lines() {
        let cells: Vec<&str> = line
            .strip_prefix('|')
            .and_then(|l| l.strip_suffix('|'))
            .map(|l| l.split('|').map(str::trim).collect())
            .unwrap_or_default();
        if cells.len() != 2 || cells[0] == "k" {
            continue;
        }
        let key = (!cells[0].is_empty()).then(|| cells[0].parse::<i64>().unwrap());
        let sum = cells[1].parse::<f64>().unwrap();
        rows.insert(key, sum);
    }
    rows
}

#[test]
fn gen_data_sql_and_explain_are_correct_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 20_000);

    // `oxide sql` output matches the independent Parquet read exactly
    // (v is generated in multiples of 0.25, so the sums are exact in f64).
    let table_arg = format!("t={}", dir.path().display());
    let assert = oxide()
        .args(["sql", "-q", QUERY, "--table", &table_arg])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let printed = parse_table(&stdout);
    let expected = expected_sums(dir.path());
    assert_eq!(printed.len(), expected.len(), "{stdout}");
    for (key, sum) in &expected {
        let got = printed
            .get(key)
            .unwrap_or_else(|| panic!("missing {key:?}"));
        assert_eq!(got, sum, "group {key:?}\n{stdout}");
    }

    // The same session semantics hold when the plan is GPU-targeted.
    let assert = oxide()
        .args([
            "sql", "-q", QUERY, "--table", &table_arg, "--target", "cuda",
        ])
        .assert()
        .success();
    let gpu_stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert_eq!(parse_table(&gpu_stdout), expected);

    // `oxide explain --target cuda` shows placement tags.
    let assert = oxide()
        .args([
            "explain",
            "-q",
            "SELECT k, SUM(v) FROM t WHERE k >= 2 AND v < 4.0 GROUP BY k",
            "--table",
            &table_arg,
            "--target",
            "cuda",
        ])
        .assert()
        .success();
    let plan = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(plan.contains("GpuAggregateExec[cuda]"), "{plan}");
    assert!(plan.contains("GpuFilterExec[cuda]"), "{plan}");
    // …and the vector operator lowers from plain SQL.
    let assert = oxide()
        .args([
            "explain",
            "-q",
            "SELECT *, l2_distance(emb, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) AS d FROM t",
            "--table",
            &table_arg,
            "--target",
            "metal",
        ])
        .assert()
        .success();
    let plan = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(
        plan.contains("GpuVectorDistanceExec[metal]: l2(emb) AS d, dim=8"),
        "{plan}"
    );
}

/// Kills the spawned cluster processes even when the test panics.
struct ClusterGuard {
    scheduler: Child,
    worker: Option<Child>,
}

impl Drop for ClusterGuard {
    fn drop(&mut self) {
        if let Some(worker) = &mut self.worker {
            let _ = worker.kill();
            let _ = worker.wait();
        }
        let _ = self.scheduler.kill();
        let _ = self.scheduler.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn wait_for_port(port: u16, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("{what} did not open port {port} within 30s");
}

#[test]
fn cluster_sql_matches_embedded_sql() {
    let dir = tempfile::tempdir().unwrap();
    // Enough rows that the Parquet file crosses DataFusion's 10 MiB
    // `repartition_file_min_size` and the scan is byte-range-split into a
    // multi-partition `DataSourceExec` — the shape that exposed the missing
    // exchange under `GpuAggregateExec` in distributed plans. Small files
    // produce single-partition scans and hide that entire failure class.
    gen_data(dir.path(), 600_000);
    let table_arg = format!("t={}", dir.path().display());

    let embedded = oxide()
        .args(["sql", "-q", QUERY, "--table", &table_arg])
        .assert()
        .success();
    let embedded_stdout = String::from_utf8(embedded.get_output().stdout.clone()).unwrap();

    // A real cluster: `oxide-scheduler` (placement target cuda, so `Gpu*Exec`
    // nodes travel through the plan codec) plus one `oxide-worker`.
    let scheduler_port = free_port();
    let scheduler = StdCommand::new(env!("CARGO_BIN_EXE_oxide-scheduler"))
        .args(["--port", &scheduler_port.to_string()])
        .env("OXIDE_CLUSTER_BACKEND", "cuda")
        .stdout(Stdio::null())
        .stderr(File::create(dir.path().join("scheduler.log")).unwrap())
        .spawn()
        .unwrap();
    let mut cluster = ClusterGuard {
        scheduler,
        worker: None,
    };
    wait_for_port(scheduler_port, "oxide-scheduler");

    let (flight_port, grpc_port) = (free_port(), free_port());
    cluster.worker = Some(
        StdCommand::new(env!("CARGO_BIN_EXE_oxide-worker"))
            .args([
                "--scheduler-host",
                "127.0.0.1",
                "--scheduler-port",
                &scheduler_port.to_string(),
                "--port",
                &flight_port.to_string(),
                "--grpc-port",
                &grpc_port.to_string(),
                "--concurrent-tasks",
                "2",
                "--work-dir",
                dir.path().join("shuffle").to_str().unwrap(),
            ])
            .stdout(Stdio::null())
            .stderr(File::create(dir.path().join("worker.log")).unwrap())
            .spawn()
            .unwrap(),
    );
    wait_for_port(grpc_port, "oxide-worker");

    // The worker registers with the scheduler asynchronously; retry the query
    // until the cluster serves it (bounded).
    let url = format!("df://127.0.0.1:{scheduler_port}");
    let deadline = Instant::now() + Duration::from_secs(60);
    let cluster_stdout = loop {
        let output = oxide()
            .args(["sql", "-q", QUERY, "--table", &table_arg, "--cluster", &url])
            .timeout(Duration::from_secs(30))
            .output()
            .unwrap();
        if output.status.success() {
            break String::from_utf8(output.stdout).unwrap();
        }
        assert!(
            Instant::now() < deadline,
            "cluster query kept failing:\n{}\nscheduler log:\n{}\nworker log:\n{}",
            String::from_utf8_lossy(&output.stderr),
            std::fs::read_to_string(dir.path().join("scheduler.log")).unwrap_or_default(),
            std::fs::read_to_string(dir.path().join("worker.log")).unwrap_or_default(),
        );
        std::thread::sleep(Duration::from_millis(500));
    };
    assert_eq!(cluster_stdout, embedded_stdout);
}
