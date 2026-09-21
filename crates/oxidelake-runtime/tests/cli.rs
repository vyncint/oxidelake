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

#[test]
fn tui_rejects_non_interactive_terminal() {
    oxide()
        .arg("tui")
        .assert()
        .code(2)
        .stderr("oxide tui needs an interactive terminal; use oxide sql or oxide explain for non-interactive output\n");
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

/// A closed reader is the most ordinary thing in a shell pipeline, and it
/// used to be exit 101 and a panic from inside DataFusion's `show()` (#34).
///
/// `assert_cmd` cannot express this: it captures stdout itself and never
/// closes it early. So the pipeline is built by hand — `oxide sql` writing
/// into `head -1`, which takes its line and goes — and both ends are checked.
/// The query is ordered and wide enough that the output is many lines, so
/// `head` is certainly gone long before the writer is done.
#[test]
fn sql_into_a_closed_pipe_exits_zero_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    // Enough rows that the rendered table is far larger than a pipe buffer,
    // so the writer is certainly still writing when `head` takes its line and
    // goes. 5,000 rows fit and the bug does not reproduce at that size --
    // measured, after a first version of this test passed against the panic.
    gen_data(dir.path(), 50_000);

    let bin = assert_cmd::cargo::cargo_bin("oxide");
    let mut sql = StdCommand::new(&bin)
        .args(["sql", "-q", "SELECT id, k, v FROM t ORDER BY id", "--table"])
        .arg(format!("t={}", dir.path().display()))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let head = StdCommand::new("head")
        .arg("-1")
        .stdin(Stdio::from(sql.stdout.take().unwrap()))
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let head_out = head.wait_with_output().unwrap();
    let sql_out = sql.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&sql_out.stderr);

    assert!(
        !stderr.contains("panicked"),
        "a closed pipe is not a crash:\n{stderr}"
    );
    assert_eq!(
        sql_out.status.code(),
        Some(0),
        "a closed pipe is a clean exit; stderr:\n{stderr}"
    );
    assert_eq!(
        String::from_utf8_lossy(&head_out.stdout).lines().count(),
        1,
        "head took its one line"
    );
}

/// `explain` and `gen-data` write to stdout through the same helper, and the
/// issue asked for both to be checked rather than assumed (#34).
#[test]
fn explain_into_a_closed_pipe_exits_zero_without_panicking() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 1_000);

    // `explain` output is short and fits a pipe buffer whatever the data, so
    // this cannot reproduce #34 the way the `sql` case does. It is here to
    // hold the other write path to the same exit code, not as a regression
    // test for the panic.
    let bin = assert_cmd::cargo::cargo_bin("oxide");
    let mut explain = StdCommand::new(&bin)
        .args(["explain", "-q", QUERY, "--table"])
        .arg(format!("t={}", dir.path().display()))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let head = StdCommand::new("head")
        .arg("-1")
        .stdin(Stdio::from(explain.stdout.take().unwrap()))
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    head.wait_with_output().unwrap();
    let out = explain.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(out.status.code(), Some(0), "stderr:\n{stderr}");
}

// --- The operational flags (#49): --batch-size, --output, --backend ---------

/// `k -> n` parsed out of the CSV form, with the header checked.
fn parse_csv(stdout: &str) -> BTreeMap<Option<i64>, i64> {
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("k,n"), "csv header\n{stdout}");
    lines
        .map(|line| {
            let (k, n) = line.split_once(',').unwrap_or_else(|| panic!("{line:?}"));
            let key = (!k.is_empty()).then(|| k.parse::<i64>().unwrap());
            (key, n.parse::<i64>().unwrap())
        })
        .collect()
}

/// `k -> n` parsed out of the JSON form. A null `k` is *absent* from the
/// object rather than `null`, which is arrow-json's encoding and is why this
/// reads the key rather than expecting it.
fn parse_json(stdout: &str) -> BTreeMap<Option<i64>, i64> {
    let trimmed = stdout.trim();
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or_else(|| panic!("not a JSON array: {trimmed}"));
    if inner.is_empty() {
        return BTreeMap::new();
    }
    inner
        .split("},{")
        .map(|object| {
            let object = object.trim_matches(['{', '}']);
            let mut row = (None, None);
            for field in object.split(',') {
                let (name, value) = field.split_once(':').unwrap();
                match name.trim_matches('"') {
                    "k" => row.0 = Some(value.parse::<i64>().unwrap()),
                    "n" => row.1 = Some(value.parse::<i64>().unwrap()),
                    other => panic!("unexpected field {other}"),
                }
            }
            (row.0, row.1.expect("every row has a count"))
        })
        .collect()
}

const COUNT_QUERY: &str = "SELECT k, COUNT(v) AS n FROM t GROUP BY k ORDER BY k";

/// The format is a rendering choice made after execution, so all three must
/// carry the same rows — and `--batch-size` must not change them either. Both
/// claims are in the flags' own `--help` text; this is what makes them true.
#[test]
fn output_formats_and_batch_sizes_agree_on_the_same_rows() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 20_000);
    let table_arg = format!("t={}", dir.path().display());

    let run = |args: &[&str]| {
        let assert = oxide()
            .args(["sql", "-q", COUNT_QUERY, "--table", &table_arg])
            .args(args)
            .assert()
            .success();
        String::from_utf8(assert.get_output().stdout.clone()).unwrap()
    };

    let csv = parse_csv(&run(&["--output", "csv"]));
    assert!(!csv.is_empty());
    assert_eq!(parse_json(&run(&["--output", "json"])), csv);

    // The table form is the default, and naming it explicitly changes nothing.
    let table = run(&[]);
    assert_eq!(run(&["--output", "table"]), table);
    assert!(table.starts_with('+'), "{table}");

    // A batch size far below and far above the row count: same answer.
    for rows in ["1", "7", "1000000"] {
        assert_eq!(
            parse_csv(&run(&["--output", "csv", "--batch-size", rows])),
            csv,
            "--batch-size {rows}"
        );
    }
}

/// An empty result still says what the columns were, so a script can tell
/// "no rows" from "the query failed".
#[test]
fn an_empty_result_keeps_its_csv_header_and_json_array() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 4_096);
    let table_arg = format!("t={}", dir.path().display());
    let run = |format: &str| {
        let assert = oxide()
            .args([
                "sql",
                "-q",
                "SELECT k, COUNT(v) AS n FROM t WHERE k = -1 GROUP BY k",
                "--table",
                &table_arg,
                "--output",
                format,
            ])
            .assert()
            .success();
        String::from_utf8(assert.get_output().stdout.clone()).unwrap()
    };
    assert_eq!(run("csv"), "k,n\n");
    assert_eq!(run("json"), "[]\n");
}

/// Zero rows per batch would be accepted by DataFusion and then return
/// nothing, which reads as an empty result rather than as a bad flag.
#[test]
fn a_zero_batch_size_is_refused_rather_than_returning_nothing() {
    let assert = oxide()
        .args(["sql", "-q", "SELECT 1", "--batch-size", "0"])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(stderr.contains("at least 1 row"), "{stderr}");
}

#[test]
fn an_unknown_output_format_names_the_ones_that_exist() {
    let assert = oxide()
        .args(["sql", "-q", "SELECT 1", "--output", "yaml"])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(stderr.contains("table, json, csv"), "{stderr}");
}

/// The batch-size and output flags are documented knobs, so they have to be
/// discoverable from `--help` — that is where the README table points.
#[test]
fn the_flags_appear_in_help() {
    let assert = oxide().args(["sql", "--help"]).assert().success();
    let help = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    for flag in ["--batch-size", "--output", "--target", "--cluster"] {
        assert!(help.contains(flag), "oxide sql --help lacks {flag}\n{help}");
    }

    let assert = Command::cargo_bin("oxide-worker")
        .unwrap()
        .arg("--help")
        .assert()
        .success();
    let help = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(help.contains("--backend"), "{help}");
    assert!(help.contains("OXIDE_BACKEND"), "{help}");
}

/// A worker asked for a backend this machine cannot provide must fail at
/// startup rather than join the cluster on the CPU: the scheduler would then
/// place GPU operators on a worker that has no GPU.
///
/// Guarded by the real availability, because on a machine that *does* have the
/// device the worker would start and then block waiting for a scheduler.
#[test]
fn a_worker_refuses_a_backend_this_machine_does_not_have() {
    let availability = oxidelake_device::HardwareDetector::availability();
    let absent = if !availability.cuda {
        "cuda"
    } else if !availability.metal {
        "metal"
    } else {
        return;
    };
    let assert = Command::cargo_bin("oxide-worker")
        .unwrap()
        .args(["--backend", absent])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(stderr.contains(absent), "{stderr}");
    assert!(stderr.contains("explicitly requested"), "{stderr}");
}

#[test]
fn a_worker_rejects_an_unknown_backend_name() {
    let assert = Command::cargo_bin("oxide-worker")
        .unwrap()
        .args(["--backend", "tpu"])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(stderr.contains("cpu, cuda, metal"), "{stderr}");
}
