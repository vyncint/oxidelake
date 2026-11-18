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

/// A port on 127.0.0.1 that nothing listens on, for a process this test is
/// about to start.
///
/// Not port 0's choice: the kernel hands out source ports for outgoing
/// connections from that same range (`ip_local_port_range`, 32768-60999 on
/// Linux, 49152 and up on macOS), so a port taken from there and released
/// was at times reused by some connection on a busy machine before the
/// worker meant to bind it had started, and the worker died with "Address
/// already in use" (twice under `make test`, 2026-11-18). Below that range
/// no one hands ports out but this function, which gives each call in this
/// binary a different one, from an offset in the process id so that two
/// test binaries running at once seldom meet, and checks it is free.
fn free_port() -> u16 {
    use std::sync::atomic::{AtomicU32, Ordering};
    const LOW: u32 = 20_000;
    const SPAN: u32 = 12_000;
    static NEXT: AtomicU32 = AtomicU32::new(u32::MAX);
    let _ = NEXT.compare_exchange(
        u32::MAX,
        std::process::id() % SPAN,
        Ordering::SeqCst,
        Ordering::SeqCst,
    );
    for _ in 0..SPAN {
        let port = u16::try_from(LOW + NEXT.fetch_add(1, Ordering::SeqCst) % SPAN).unwrap();
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    panic!("no free port in {LOW}..{}", LOW + SPAN)
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

    // Planned for a GPU, the aggregate is fed by every one of those
    // partitions (#29, #90): its partial half reads the split scan with no
    // merge in between, and its final half reads the hash exchange over the
    // same partitions. It used to sit alone over a `CoalescePartitionsExec`.
    let explain = oxide()
        .args([
            "explain", "-q", QUERY, "--table", &table_arg, "--target", "cuda",
        ])
        .assert()
        .success();
    let plan = String::from_utf8(explain.get_output().stdout.clone()).unwrap();
    let lines: Vec<&str> = plan.lines().map(str::trim_start).collect();
    let partial = lines
        .iter()
        .position(|l| l.starts_with("GpuAggregateExec[cuda]") && l.contains("aggr=[SUM(v)]"))
        .unwrap_or_else(|| panic!("no partial GPU aggregate in\n{plan}"));
    let scan_partitions: usize = lines
        .get(partial + 1)
        .and_then(|l| l.strip_prefix("DataSourceExec: file_groups={"))
        .and_then(|l| l.split_once(' '))
        .and_then(|(n, _)| n.parse().ok())
        .unwrap_or_else(|| panic!("the partial aggregate does not read the scan:\n{plan}"));
    // DataFusion splits the scan into one partition per core.
    if std::thread::available_parallelism().map_or(1, usize::from) > 1 {
        assert!(scan_partitions > 1, "{plan}");
        let exchange = lines[partial - 1];
        assert!(
            exchange.starts_with("RepartitionExec: partitioning=Hash([k@0]")
                && exchange.ends_with(&format!("input_partitions={scan_partitions}")),
            "{plan}"
        );
        assert!(
            lines[partial - 2].starts_with("GpuAggregateExec[cuda]"),
            "{plan}"
        );
    }

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

/// The reproduction from #119: `gen-data` over an existing table with a
/// flag it refuses used to empty `t.parquet` first (211,115 bytes to 0).
#[test]
fn a_refused_gen_data_leaves_the_existing_table_intact() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 1_000);
    let table = dir.path().join("t.parquet");
    let before = std::fs::read(&table).unwrap();
    let assert = oxide()
        .args([
            "gen-data",
            "--rows",
            "2000",
            "--out",
            dir.path().to_str().unwrap(),
            "--row-group-rows",
            "0",
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(
        stderr.contains("row_group_rows must be positive"),
        "{stderr}"
    );
    let after = std::fs::read(&table).unwrap();
    assert!(
        after == before,
        "t.parquet is {} bytes after the refused run, was {}",
        after.len(),
        before.len()
    );
    let names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names, ["t.parquet"], "nothing is left beside the table");
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

/// `oxide explain` names why a node stayed on the CPU (#32).
///
/// The plan above the notes shows an ordinary `AggregateExec`, which is
/// exactly what a node that was never eligible looks like; the difference is
/// only visible because the rule writes its reason down.
#[test]
fn explain_names_why_a_node_stayed_on_the_cpu() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 4_096);
    let table_arg = format!("t={}", dir.path().display());

    let assert = oxide()
        .args([
            "explain",
            "-q",
            "SELECT k, AVG(v) FROM t GROUP BY k",
            "--table",
            &table_arg,
            "--target",
            "cuda",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains("placement notes (target cuda)"), "{stdout}");
    assert!(stdout.contains("AggregateExec:"), "{stdout}");
    assert!(stdout.contains("sum, count, min, max"), "{stdout}");

    // A plan that lowers completely has nothing to explain.
    let assert = oxide()
        .args([
            "explain",
            "-q",
            "SELECT k, SUM(v) FROM t WHERE k >= 2 GROUP BY k",
            "--table",
            &table_arg,
            "--target",
            "cuda",
        ])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(stdout.contains("GpuAggregateExec[cuda]"), "{stdout}");
    assert!(!stdout.contains("placement notes"), "{stdout}");
}

/// `oxide explain`'s notes are accurate for a `!=` filter and present for a
/// join the device cannot take (#94). The first listed `!=` among the
/// comparisons the kernels implement; the second printed no notes section,
/// as if the rule had never looked at the join.
#[test]
fn explain_notes_a_not_equal_filter_and_a_join_left_on_the_cpu() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 4_096);
    let table_arg = format!("t={}", dir.path().display());
    let explain = |query: &str| {
        let assert = oxide()
            .args([
                "explain", "-q", query, "--table", &table_arg, "--target", "cuda",
            ])
            .assert()
            .success();
        String::from_utf8(assert.get_output().stdout.clone()).unwrap()
    };

    let stdout = explain("SELECT id FROM t WHERE k != 3");
    assert!(
        stdout.contains(
            "placement notes (target cuda):\n  FilterExec: `!=` is not one of the \
             comparisons the device kernels implement (=, <, <=, >, >=)\n"
        ),
        "{stdout}"
    );

    let stdout = explain("SELECT a.id, b.s FROM t a LEFT JOIN t b ON a.id = b.id");
    assert!(
        stdout.contains("placement notes (target cuda):"),
        "{stdout}"
    );
    assert!(
        stdout.contains("join has no device kernel; the device join is inner only"),
        "{stdout}"
    );
}

// --- observability (#33) ----------------------------------------------------

/// `RUST_LOG=info oxide sql …` prints one line per query with the mode, the
/// rows, the time and how much of it fell back to the CPU.
#[test]
fn a_query_logs_one_line_with_rows_and_elapsed() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 4_096);
    let table_arg = format!("t={}", dir.path().display());

    let assert = oxide()
        .env("RUST_LOG", "oxidelake_runtime=info")
        .args([
            "sql",
            "-q",
            "SELECT k, SUM(v) FROM t WHERE k >= 2 GROUP BY k",
            "--table",
            &table_arg,
            "--target",
            "cuda",
        ])
        .assert()
        .success();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(stderr.contains("query finished"), "{stderr}");
    assert!(stderr.contains("mode=embedded/cuda"), "{stderr}");
    assert!(stderr.contains("rows="), "{stderr}");
    assert!(stderr.contains("elapsed_ms="), "{stderr}");
    assert!(stderr.contains("fallback_batches="), "{stderr}");

    // The log goes to stderr, so it never mixes into `--output json`.
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    assert!(!stdout.contains("query finished"), "{stdout}");
}

/// Without the `metrics` feature the flag is refused, not ignored. A worker
/// that was asked for metrics and quietly served none is the failure the
/// endpoint exists to prevent.
#[test]
#[cfg_attr(feature = "metrics", ignore = "the flag is honoured in this build")]
fn a_metrics_port_without_the_feature_is_refused() {
    let assert = Command::cargo_bin("oxide-worker")
        .unwrap()
        .args(["--metrics-port", "19999"])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
    assert!(
        stderr.contains("--metrics-port needs the `metrics` feature"),
        "{stderr}"
    );
    assert!(stderr.contains("--features metrics"), "{stderr}");
}

// --- real cluster processes (#92) ------------------------------------------

/// A scheduler and the workers a test starts, every one killed and reaped on
/// drop so a failing assertion never leaves a process listening behind it.
/// Everything binds 127.0.0.1 on a free port, and every process writes its
/// log to `<dir>/<name>.log` for the failure message.
struct TestCluster {
    dir: std::path::PathBuf,
    scheduler_port: u16,
    processes: Vec<(String, Child)>,
    /// Each worker's Arrow Flight port, which is how a failed shuffle fetch
    /// names the worker it could not reach.
    flight_ports: Vec<(String, u16)>,
}

impl TestCluster {
    /// Starts `oxide-scheduler` planning for `backend`, with `args`.
    fn start(dir: &Path, backend: &str, args: &[&str]) -> Self {
        let scheduler_port = free_port();
        let mut cluster = Self {
            dir: dir.to_owned(),
            scheduler_port,
            processes: Vec::new(),
            flight_ports: Vec::new(),
        };
        let scheduler = StdCommand::new(env!("CARGO_BIN_EXE_oxide-scheduler"))
            .args([
                "--bind-host",
                "127.0.0.1",
                "--port",
                &scheduler_port.to_string(),
            ])
            .args(args)
            .env("OXIDE_CLUSTER_BACKEND", backend)
            .env("RUST_LOG", "info")
            .stdout(Stdio::null())
            .stderr(cluster.log_file("scheduler"))
            .spawn()
            .unwrap();
        cluster.processes.push(("scheduler".to_owned(), scheduler));
        wait_for_port(scheduler_port, "oxide-scheduler");
        cluster
    }

    fn log_file(&self, name: &str) -> File {
        File::create(self.dir.join(format!("{name}.log"))).unwrap()
    }

    fn log(&self, name: &str) -> String {
        std::fs::read_to_string(self.dir.join(format!("{name}.log"))).unwrap_or_default()
    }

    fn logs(&self) -> String {
        self.processes
            .iter()
            .map(|(name, _)| format!("--- {name}.log\n{}", self.log(name)))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn url(&self) -> String {
        format!("df://127.0.0.1:{}", self.scheduler_port)
    }

    /// Starts the `oxide-worker` at `bin` (this build's, or another build's)
    /// as `name`, and returns its index in `processes`. It is not waited
    /// for: a refused worker never opens its ports for long.
    fn spawn_worker(&mut self, name: &str, bin: &Path, args: &[&str]) -> usize {
        let work_dir = self.dir.join(format!("{name}-shuffle"));
        let flight_port = free_port();
        self.flight_ports.push((name.to_owned(), flight_port));
        let worker = StdCommand::new(bin)
            .args([
                "--scheduler-host",
                "127.0.0.1",
                "--scheduler-port",
                &self.scheduler_port.to_string(),
                "--port",
                &flight_port.to_string(),
                "--grpc-port",
                &free_port().to_string(),
                "--concurrent-tasks",
                "2",
                "--work-dir",
                work_dir.to_str().unwrap(),
            ])
            .args(args)
            .env("RUST_LOG", "info")
            .stdout(Stdio::null())
            .stderr(self.log_file(name))
            .spawn()
            .unwrap();
        self.processes.push((name.to_owned(), worker));
        self.processes.len() - 1
    }

    /// Waits until the scheduler's `/readyz` (Ballista's probe, on the
    /// scheduler port) reports `executors` live workers.
    fn wait_ready(&self, executors: usize) {
        let want = format!("ready: {executors} executors registered");
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let body = http_get(self.scheduler_port, "/readyz").unwrap_or_default();
            if body.contains(&want) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "never {want:?}; last /readyz: {body:?}\n{}",
                self.logs()
            );
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// `oxide sql --cluster`, bounded so a hang fails the test rather than
    /// the suite.
    fn sql(&self, query: &str, table_arg: &str) -> std::process::Output {
        oxide()
            .args([
                "sql",
                "-q",
                query,
                "--table",
                table_arg,
                "--cluster",
                &self.url(),
            ])
            .timeout(Duration::from_secs(120))
            .output()
            .unwrap()
    }
}

impl TestCluster {
    /// Sends `signal` (`TERM`, `KILL`, …) to process `index`.
    fn signal(&self, index: usize, signal: &str) {
        let pid = self.processes[index].1.id().to_string();
        let sent = StdCommand::new("kill")
            .args([&format!("-{signal}"), &pid])
            .status()
            .unwrap();
        assert!(sent.success(), "kill -{signal} {pid}");
    }

    /// Waits until `name`'s log has a line `matches` accepts, up to `limit`.
    fn wait_log(&self, name: &str, what: &str, limit: Duration, matches: impl Fn(&str) -> bool) {
        let deadline = Instant::now() + limit;
        while !self.log(name).lines().any(&matches) {
            assert!(
                Instant::now() < deadline,
                "{name} never logged {what} within {limit:?}\n{}",
                self.logs()
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn flight_port(&self, name: &str) -> u16 {
        self.flight_ports
            .iter()
            .find_map(|(worker, port)| (worker == name).then_some(*port))
            .unwrap()
    }

    /// `oxide sql --cluster` in the background, writing `<dir>/<name>.out`
    /// and `<dir>/<name>.log`, for a query that is to be interrupted.
    fn spawn_sql(&self, name: &str, query: &str, table_arg: &str) -> Child {
        StdCommand::new(assert_cmd::cargo::cargo_bin("oxide"))
            .args([
                "sql",
                "-q",
                query,
                "--table",
                table_arg,
                "--cluster",
                &self.url(),
            ])
            .stdout(File::create(self.dir.join(format!("{name}.out"))).unwrap())
            .stderr(self.log_file(name))
            .spawn()
            .unwrap()
    }
}

/// Reaps a background query (or kills it after `limit`), and returns whether
/// it succeeded with its stdout and stderr.
fn finish_sql(
    cluster: &TestCluster,
    name: &str,
    mut child: Child,
    limit: Duration,
) -> (bool, String, String) {
    let status = wait_exit(&mut child, limit);
    if status.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    let stdout =
        std::fs::read_to_string(cluster.dir.join(format!("{name}.out"))).unwrap_or_default();
    let stderr = cluster.log(name);
    let status = status.unwrap_or_else(|| {
        panic!(
            "`{name}` was still running after {limit:?}\n{stderr}\n{}",
            cluster.logs()
        )
    });
    (status.success(), stdout, stderr)
}

impl Drop for TestCluster {
    fn drop(&mut self) {
        for (_, process) in &mut self.processes {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

/// A plain HTTP/1.1 GET on 127.0.0.1, enough to read a probe's body.
fn http_get(port: u16, path: &str) -> Option<String> {
    use std::io::{Read, Write};
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    Some(response)
}

/// Waits for `child` to exit, up to `limit`.
fn wait_exit(child: &mut Child, limit: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().unwrap() {
            return Some(status);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    None
}

/// A worker from another build is refused when it registers, and a query on
/// a cluster left with no worker it can use fails at once and says why
/// (#92).
///
/// The worker is this build's own executor with another fingerprint in its
/// id — what a worker built with other features or from another release
/// presents. The cluster plans for `cpu`, so no `Gpu*Exec` payload exists
/// to check: the refusal no longer depends on what a plan contains.
/// `make test-mixed-build` runs the same refusal with a real
/// `--features predict` binary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_worker_from_another_build_is_refused_and_the_query_says_why() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 4_096);
    let table_arg = format!("t={}", dir.path().display());
    let cluster = TestCluster::start(dir.path(), "cpu", &[]);

    let mut config = oxidelake_runtime::cluster::executor_config(
        "127.0.0.1",
        cluster.scheduler_port,
        free_port(),
        free_port(),
        Some(1),
        Some(dir.path().join("forged").to_string_lossy().into_owned()),
    );
    let ours = format!("{:08x}", oxidelake_planner::codec::PLAN_FINGERPRINT);
    let theirs = format!("{:08x}", !oxidelake_planner::codec::PLAN_FINGERPRINT);
    assert!(config.executor_id.contains(&ours), "{}", config.executor_id);
    config.executor_id = config.executor_id.replacen(&ours, &theirs, 1);

    let refused = tokio::time::timeout(
        Duration::from_secs(60),
        oxidelake_runtime::cluster::run_executor(config),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "a worker from another build was not refused\n{}",
            cluster.logs()
        )
    })
    .expect_err("registration is refused")
    .to_string();
    assert!(refused.contains("refused"), "{refused}");
    assert!(refused.contains(&format!("0x{theirs}")), "{refused}");
    assert!(refused.contains(&format!("0x{ours}")), "{refused}");

    let started = Instant::now();
    let output = cluster.sql(QUERY, &table_arg);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stderr}");
    assert!(
        stderr.contains("no worker it can use") && stderr.contains(&format!("0x{theirs}")),
        "the query names the refusal: {stderr}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "failed, but only after {:?}",
        started.elapsed()
    );
}

/// `oxide-worker` built another way, for the two-binary tests below. Only
/// `make test-mixed-build` builds them, so the tests are `#[ignore]`d and a
/// run without the variable fails loudly instead of passing vacuously.
fn other_build(var: &str) -> std::path::PathBuf {
    std::env::var_os(var)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| panic!("{var} is not set: run `make test-mixed-build`"))
}

/// A worker built with `--features cuda` runs the GPU-placed plans of a
/// default-build scheduler (#92): `cuda` changes where an operator runs,
/// not what a plan says, which is the heterogeneous cluster the README
/// describes. Until 0.3.0 the feature was in the fingerprint, so the worker
/// refused every `Gpu*Exec` payload and each such query failed.
#[test]
#[ignore = "needs a second build of oxide-worker: make test-mixed-build"]
fn mixed_build_a_cuda_worker_runs_a_default_schedulers_gpu_plans() {
    let worker = other_build("OXIDE_TEST_CUDA_WORKER");
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 20_000);
    let table_arg = format!("t={}", dir.path().display());
    let embedded = oxide()
        .args(["sql", "-q", QUERY, "--table", &table_arg])
        .assert()
        .success();
    let embedded_stdout = String::from_utf8(embedded.get_output().stdout.clone()).unwrap();

    let mut cluster = TestCluster::start(dir.path(), "cuda", &[]);
    cluster.spawn_worker("cuda-worker", &worker, &[]);
    cluster.wait_ready(1);
    let output = cluster.sql(QUERY, &table_arg);
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        cluster.logs()
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), embedded_stdout);
}

/// A worker built with `--features predict` is refused by a default-build
/// scheduler, exits non-zero with both fingerprints named, and a query
/// then fails within seconds saying why (#42, #92). Before 0.3.0 the worker
/// joined: a `cpu` cluster never compared fingerprints, and a GPU one only
/// when a task's plan was decoded, which failed (or, under Ballista 54,
/// hung) the query with the reason in the worker's log.
#[test]
#[ignore = "needs a second build of oxide-worker: make test-mixed-build"]
fn mixed_build_a_predict_worker_is_refused_by_a_default_scheduler() {
    let worker = other_build("OXIDE_TEST_PREDICT_WORKER");
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 4_096);
    let table_arg = format!("t={}", dir.path().display());

    let mut cluster = TestCluster::start(dir.path(), "cpu", &[]);
    let index = cluster.spawn_worker("predict-worker", &worker, &[]);
    let status = wait_exit(&mut cluster.processes[index].1, Duration::from_secs(60))
        .unwrap_or_else(|| panic!("the predict worker joined\n{}", cluster.logs()));
    assert!(!status.success(), "{}", cluster.logs());
    let log = cluster.log("predict-worker");
    assert!(
        log.contains("refused") && log.contains("feature mask 0x1"),
        "{log}"
    );

    let started = Instant::now();
    let output = cluster.sql(QUERY, &table_arg);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stderr}");
    assert!(stderr.contains("no worker it can use"), "{stderr}");
    assert!(started.elapsed() < Duration::from_secs(30));
}

// --- resilience: worker loss, retry limits, SIGTERM (#31) -------------------

/// A query slow enough to stop a process in the middle of: a self-join that
/// fans out on `k` (about 6,000 rows a key), whose `id` bound sets the work.
/// Ballista runs it in four stages, two scans, the join ([`JOIN_STAGE`]) and
/// the final aggregate, and the join's 16 tasks run in waves on two workers
/// of two tasks each, so a worker stopped as its first join task starts
/// leaves join tasks still to run that read its shuffle output.
fn slow_join(rows: u64) -> String {
    format!(
        "SELECT a.k, COUNT(*) AS n, SUM(a.v + b.v) AS sv FROM t a JOIN t b ON a.k = b.k \
         WHERE a.id < {rows} GROUP BY a.k ORDER BY a.k"
    )
}

const JOIN_STAGE: usize = 3;

/// Whether `line` is Ballista's executor starting a task of `stage` (its
/// first attempt): `Execute task  : [TID <job>/<stage>.<attempt>/<partition>.<attempt>]`.
fn starts_stage(stage: usize) -> impl Fn(&str) -> bool {
    let needle = format!("/{stage}.0/");
    move |line| line.contains("Execute task") && line.contains(&needle)
}

fn embedded_sql(query: &str, table_arg: &str) -> String {
    let assert = oxide()
        .args(["sql", "-q", query, "--table", table_arg])
        .assert()
        .success();
    String::from_utf8(assert.get_output().stdout.clone()).unwrap()
}

fn this_worker() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_oxide-worker"))
}

/// The knobs are documented in the README tables, so they must be
/// discoverable from `--help` (#31).
#[test]
fn the_resilience_flags_appear_in_help() {
    for (bin, flags) in [
        (
            "oxide-scheduler",
            &[
                "--task-max-failures",
                "--stage-max-failures",
                "--worker-timeout",
                "--drain-timeout",
            ][..],
        ),
        (
            "oxide-worker",
            &["--heartbeat-interval", "--drain-timeout"][..],
        ),
    ] {
        let assert = Command::cargo_bin(bin)
            .unwrap()
            .arg("--help")
            .assert()
            .success();
        let help = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
        for flag in flags {
            assert!(help.contains(flag), "{bin} --help lacks {flag}\n{help}");
        }
    }
}

/// How a worker was lost, which decides what its query's error can say.
#[derive(Clone, Copy, PartialEq)]
enum Lost {
    /// Stopped with `SIGTERM`: it keeps serving its shuffle data while it
    /// drains, and OxideLake fails the query naming it.
    Stopped,
    /// Killed: a fetch from it fails and the error names its address,
    /// unless it died in the middle of sending a shuffle block to a task
    /// still running, when Ballista reports that broken stream instead
    /// (`IpcError ... error reading a body from connection`). That error
    /// carries no address, but it fails the stage at once, with no
    /// rollback, so the query still fails cleanly (1 run in about 8 of the
    /// silent-worker test, 2026-11-18).
    Killed,
}

/// What a client sees when a worker is lost in the middle of its query:
/// the query fails, prints no rows, and the error says why (#31, #148).
/// Then the next query runs on the workers left, with the right rows.
fn assert_the_query_fails_naming(
    cluster: &TestCluster,
    worker: &str,
    how: Lost,
    query: Child,
    then: &str,
) {
    let (succeeded, stdout, stderr) = finish_sql(cluster, "query", query, Duration::from_secs(120));
    assert!(!succeeded, "{stderr}\n{}", cluster.logs());
    assert_eq!(stdout, "", "no rows are printed for a failed query");
    let port = format!("127.0.0.1:{}", cluster.flight_port(worker));
    let broken_stream = how == Lost::Killed && stderr.contains("IpcError");
    assert!(
        stderr.contains(&port) || broken_stream,
        "the error names the lost worker, {port}:\n{stderr}"
    );
    let table_arg = format!("t={}", cluster.dir.display());
    let output = cluster.sql(QUERY, &table_arg);
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        cluster.logs()
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), then);
}

/// A worker killed in the middle of a query (#31): with the scheduler's
/// defaults the query fails within seconds naming the worker it lost,
/// rather than running the lost work again (#148).
///
/// The join tasks still to run cannot fetch the dead worker's shuffle
/// output; Ballista retries each fetch three times, three seconds apart,
/// then fails the stage, and the default `--stage-max-failures 1` makes
/// that the query's end. Ballista's 180-second wait on the dead worker's
/// heartbeat is far longer than that, so it never decides the outcome here.
#[test]
#[cfg(unix)]
fn a_query_that_loses_a_worker_fails_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 600_000);
    let table_arg = format!("t={}", dir.path().display());
    let after = embedded_sql(QUERY, &table_arg);

    let mut cluster = TestCluster::start(dir.path(), "cpu", &[]);
    cluster.spawn_worker("w1", &this_worker(), &[]);
    let lost = cluster.spawn_worker("w2", &this_worker(), &[]);
    cluster.wait_ready(2);

    let query = cluster.spawn_sql("query", &slow_join(200_000), &table_arg);
    cluster.wait_log(
        "w2",
        "a join task",
        Duration::from_secs(120),
        starts_stage(JOIN_STAGE),
    );
    cluster.signal(lost, "KILL");
    assert_the_query_fails_naming(&cluster, "w2", Lost::Killed, query, &after);
}

/// A worker stopped with `SIGTERM` in the middle of a query (#31, #148):
/// the worker exits 0, and the query fails at once naming it. Ballista
/// would have rolled the query back and run the lost work again, which on
/// 55.0.0 hung in both runs tried; OxideLake fails the query before that
/// rollback can start.
#[test]
#[cfg(unix)]
fn a_query_whose_worker_stops_fails_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 600_000);
    let table_arg = format!("t={}", dir.path().display());
    let after = embedded_sql(QUERY, &table_arg);

    let mut cluster = TestCluster::start(dir.path(), "cpu", &[]);
    cluster.spawn_worker("w1", &this_worker(), &[]);
    let stopped = cluster.spawn_worker("w2", &this_worker(), &[]);
    cluster.wait_ready(2);

    let query = cluster.spawn_sql("query", &slow_join(200_000), &table_arg);
    cluster.wait_log(
        "w2",
        "a join task",
        Duration::from_secs(120),
        starts_stage(JOIN_STAGE),
    );
    cluster.signal(stopped, "TERM");
    let status = wait_exit(&mut cluster.processes[stopped].1, Duration::from_secs(60))
        .unwrap_or_else(|| panic!("the stopped worker did not exit\n{}", cluster.logs()));
    assert_eq!(status.code(), Some(0), "{}", cluster.logs());
    assert_the_query_fails_naming(&cluster, "w2", Lost::Stopped, query, &after);
    let stderr = cluster.log("query");
    assert!(
        stderr.contains("was lost while this query ran") && stderr.contains("#148"),
        "OxideLake's reason, not Ballista's \"Cancelled\":\n{stderr}"
    );
}

/// A worker that goes silent in the middle of a query (#31, #148): once
/// `--worker-timeout` passes without its heartbeat the query fails naming
/// it, here before any shuffle fetch from it has run out of retries. With
/// Ballista's recovery the same run returned duplicated rows.
#[test]
#[cfg(unix)]
fn a_query_whose_worker_goes_silent_fails_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 600_000);
    let table_arg = format!("t={}", dir.path().display());
    let after = embedded_sql(QUERY, &table_arg);

    let mut cluster = TestCluster::start(dir.path(), "cpu", &["--worker-timeout", "5"]);
    cluster.spawn_worker("w1", &this_worker(), &["--heartbeat-interval", "1"]);
    let lost = cluster.spawn_worker("w2", &this_worker(), &["--heartbeat-interval", "1"]);
    cluster.wait_ready(2);

    let query = cluster.spawn_sql("query", &slow_join(200_000), &table_arg);
    cluster.wait_log(
        "w2",
        "a join task",
        Duration::from_secs(120),
        starts_stage(JOIN_STAGE),
    );
    cluster.signal(lost, "KILL");
    assert_the_query_fails_naming(&cluster, "w2", Lost::Killed, query, &after);
}

/// What #31 asked for, and what Ballista 55.0.0 does not do: with stage
/// retries raised to Ballista's own default of 4, a query that loses a
/// worker should come back with exactly the rows it would have had. It
/// does not, which is why OxideLake's default is 1 (#148): the rolled-back
/// stage keeps its partial outputs, fixed upstream by
/// apache/datafusion-ballista#2677 (merged after 55.0.0, unreleased), and
/// the tasks the rollback cancels never return their slots, so a query can
/// wait for capacity that does not come back (see
/// apache/datafusion-ballista#2539 and #2698). Measured with this setup
/// (`kill -9` as the worker starts its first join task): 12 runs in 15
/// returned the 100 groups plus 13 or 14 of them a second time, exit 0,
/// 7 of 8 from a shell loop and 5 of 7 of this test. Run by name to reproduce; it
/// fails when either happens, which is not every run. Remove the `ignore`
/// once the chain is on a Ballista with the fix.
#[test]
#[cfg(unix)]
#[ignore = "Ballista 55 can recover a query that lost a worker into duplicated rows (#148)"]
fn a_query_that_loses_a_worker_comes_back_with_the_right_rows() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 600_000);
    let table_arg = format!("t={}", dir.path().display());
    let query = slow_join(200_000);
    let expected = embedded_sql(&query, &table_arg);

    let mut cluster = TestCluster::start(dir.path(), "cpu", &["--stage-max-failures", "4"]);
    cluster.spawn_worker("w1", &this_worker(), &[]);
    let lost = cluster.spawn_worker("w2", &this_worker(), &[]);
    cluster.wait_ready(2);
    let running = cluster.spawn_sql("query", &query, &table_arg);
    cluster.wait_log(
        "w2",
        "a join task",
        Duration::from_secs(120),
        starts_stage(JOIN_STAGE),
    );
    cluster.signal(lost, "KILL");
    let (succeeded, stdout, stderr) =
        finish_sql(&cluster, "query", running, Duration::from_secs(300));
    assert!(succeeded, "{stderr}");
    assert_eq!(stdout, expected, "{}", cluster.logs());
}

/// `SIGTERM` stops a worker within its `--drain-timeout`, with exit 0 (#31).
/// An idle worker stops at once, and the scheduler stops counting it. A
/// worker running join tasks far longer than its timeout stops when the
/// timeout passes, leaving them to the scheduler; without the bound it
/// waited for them, however long they took.
#[test]
#[cfg(unix)]
fn sigterm_stops_a_worker_within_its_drain_timeout_with_exit_zero() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 600_000);
    let table_arg = format!("t={}", dir.path().display());
    let mut cluster = TestCluster::start(dir.path(), "cpu", &[]);
    let idle = cluster.spawn_worker("idle", &this_worker(), &[]);
    let busy = cluster.spawn_worker("busy", &this_worker(), &["--drain-timeout", "1"]);
    cluster.wait_ready(2);

    let started = Instant::now();
    cluster.signal(idle, "TERM");
    let status = wait_exit(&mut cluster.processes[idle].1, Duration::from_secs(20))
        .unwrap_or_else(|| panic!("an idle worker ignored SIGTERM\n{}", cluster.logs()));
    assert_eq!(status.code(), Some(0), "{}", cluster.logs());
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    cluster.wait_ready(1);

    // Every row joins about 6,000 others: each of the 16 join tasks is a
    // sixteenth of 3.6 billion rows, about five seconds on the machine this
    // was written on, so a task outlasts the one-second bound with room.
    let query = cluster.spawn_sql("query", &slow_join(600_000), &table_arg);
    cluster.wait_log(
        "busy",
        "a join task",
        Duration::from_secs(120),
        starts_stage(JOIN_STAGE),
    );
    let started = Instant::now();
    cluster.signal(busy, "TERM");
    let status =
        wait_exit(&mut cluster.processes[busy].1, Duration::from_secs(30)).unwrap_or_else(|| {
            panic!(
                "a busy worker outlived its drain timeout\n{}",
                cluster.logs()
            )
        });
    let took = started.elapsed();
    assert_eq!(status.code(), Some(0), "{}", cluster.logs());
    assert!(
        took >= Duration::from_secs(1) && took < Duration::from_secs(10),
        "stopped after {took:?}\n{}",
        cluster.log("busy")
    );
    assert!(
        cluster.log("busy").contains("drain timeout passed"),
        "{}",
        cluster.log("busy")
    );
    // A worker draining in the middle of a query must not lead that query
    // into Ballista's rollback (#148): the query fails at once, naming the
    // worker, rather than waiting for one that is never coming back.
    let (succeeded, stdout, stderr) = finish_sql(&cluster, "query", query, Duration::from_secs(60));
    assert!(!succeeded, "{stderr}");
    assert_eq!(stdout, "", "no rows are printed for a failed query");
    let port = format!("127.0.0.1:{}", cluster.flight_port("busy"));
    assert!(
        stderr.contains("was lost while this query ran") && stderr.contains(&port),
        "{stderr}"
    );
}

/// `SIGTERM` stops a scheduler gracefully (#31): a query sent from then on
/// is refused saying why, the query already running finishes with the right
/// rows, and the scheduler then exits 0. It used to die of the signal
/// (status 143) and take the running query with it.
#[test]
#[cfg(unix)]
fn sigterm_drains_a_scheduler_and_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    gen_data(dir.path(), 600_000);
    let table_arg = format!("t={}", dir.path().display());
    let query = slow_join(200_000);
    let expected = embedded_sql(&query, &table_arg);

    let mut cluster = TestCluster::start(dir.path(), "cpu", &["--drain-timeout", "120"]);
    cluster.spawn_worker("w1", &this_worker(), &[]);
    cluster.spawn_worker("w2", &this_worker(), &[]);
    cluster.wait_ready(2);
    let running = cluster.spawn_sql("running", &query, &table_arg);
    cluster.wait_log(
        "scheduler",
        "the query's job",
        Duration::from_secs(120),
        |line| line.contains("Job submitted"),
    );
    cluster.signal(0, "TERM");
    cluster.wait_log(
        "scheduler",
        "that it is shutting down",
        Duration::from_secs(20),
        |line| line.contains("shutting down") && line.contains("running=1"),
    );

    let refused = cluster.sql("SELECT COUNT(*) FROM t", &table_arg);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "{stderr}");
    assert!(stderr.contains("shutting down"), "{stderr}");

    let (succeeded, stdout, stderr) =
        finish_sql(&cluster, "running", running, Duration::from_secs(120));
    assert!(succeeded, "{stderr}\n{}", cluster.logs());
    assert_eq!(stdout, expected);
    let status = wait_exit(&mut cluster.processes[0].1, Duration::from_secs(30))
        .unwrap_or_else(|| panic!("the scheduler did not exit\n{}", cluster.logs()));
    assert_eq!(status.code(), Some(0), "{}", cluster.logs());
    assert!(
        cluster
            .log("scheduler")
            .contains("every running query finished"),
        "{}",
        cluster.log("scheduler")
    );
}

/// `--worker-timeout` and `--heartbeat-interval` reach the processes (#31):
/// a worker killed while idle stops counting once the timeout passes, and
/// the live one, heartbeating more often than that, keeps counting. With
/// Ballista's 60-second heartbeat the live one would be presumed lost too;
/// with its 180-second timeout the dead one would still count. The timeout
/// is five heartbeats, not two: Ballista keeps heartbeat times in whole
/// seconds, and this test shares the machine with joins.
#[test]
#[cfg(unix)]
fn a_silent_worker_stops_counting_after_the_worker_timeout() {
    let dir = tempfile::tempdir().unwrap();
    let mut cluster = TestCluster::start(dir.path(), "cpu", &["--worker-timeout", "5"]);
    cluster.spawn_worker("w1", &this_worker(), &["--heartbeat-interval", "1"]);
    let lost = cluster.spawn_worker("w2", &this_worker(), &["--heartbeat-interval", "1"]);
    cluster.wait_ready(2);
    let killed = Instant::now();
    cluster.signal(lost, "KILL");
    cluster.wait_ready(1);
    assert!(
        killed.elapsed() < Duration::from_secs(20),
        "{:?}",
        killed.elapsed()
    );
    // Twice the timeout later, the live worker still counts.
    std::thread::sleep(Duration::from_secs(10));
    let body = http_get(cluster.scheduler_port, "/readyz").unwrap_or_default();
    assert!(
        body.contains("ready: 1 executors registered"),
        "{body}\n{}",
        cluster.logs()
    );
}

// --- where cluster processes may listen (#30) -------------------------------

/// A scheduler or worker asked to listen on a public address, or on every
/// address, without `--allow-insecure-bind` refuses before it listens, and
/// says what to do (#30). The ports are unauthenticated, so this is the
/// whole of the security boundary until OxideLake has TLS and auth.
///
/// Port 1 keeps a regression from exposing anything while this runs: an
/// unprivileged process cannot bind it, so a scheduler that skipped the
/// check would fail on the bind, with another message, instead of
/// listening on every interface.
#[test]
fn a_public_or_wildcard_listener_is_refused_without_the_flag() {
    let refusals = [
        (
            "oxide-scheduler",
            &["--bind-host", "0.0.0.0", "--port", "1"][..],
        ),
        ("oxide-scheduler", &["--bind-host", "::", "--port", "1"][..]),
        (
            "oxide-scheduler",
            &["--bind-host", "203.0.113.7", "--port", "1"][..],
        ),
        (
            "oxide-scheduler",
            &[
                "--port",
                "1",
                "--metrics-port",
                "1",
                "--metrics-host",
                "0.0.0.0",
            ][..],
        ),
        (
            "oxide-worker",
            &["--bind-host", "0.0.0.0", "--port", "1", "--grpc-port", "1"][..],
        ),
        (
            "oxide-worker",
            &["--metrics-port", "1", "--metrics-host", "203.0.113.7"][..],
        ),
    ];
    for (bin, args) in refusals {
        let assert = Command::cargo_bin(bin)
            .unwrap()
            .args(args)
            .timeout(Duration::from_secs(30))
            .assert()
            .failure();
        let stderr = String::from_utf8(assert.get_output().stderr.clone()).unwrap();
        assert!(
            stderr.contains("--allow-insecure-bind") && stderr.contains("no authentication"),
            "{bin} {args:?}: {stderr}"
        );
    }
    for bin in ["oxide-scheduler", "oxide-worker"] {
        let assert = Command::cargo_bin(bin)
            .unwrap()
            .arg("--help")
            .assert()
            .success();
        let help = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
        assert!(help.contains("--allow-insecure-bind"), "{bin}\n{help}");
        assert!(help.contains("--bind-host"), "{bin}\n{help}");
    }
}
