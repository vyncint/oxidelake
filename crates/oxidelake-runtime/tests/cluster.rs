//! Cluster equivalence (docs/SPEC.md §2.6): an in-process Ballista scheduler plus
//! two executors, planned with `OXIDE_CLUSTER_BACKEND`-style target `cuda` so
//! `Gpu*Exec` nodes travel through the codec (the executors have no GPU and take
//! the CPU path), must return exactly what embedded mode returns.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::physical_expr::Partitioning;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties, displayable};
use datafusion::prelude::SessionContext;
use oxidelake_core::BackendKind;
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_runtime::cluster::{build_session_state, session_config, start_standalone};
use oxidelake_runtime::{OxideSession, SessionMode};
use oxidelake_storage::{Compression, ParquetWriteOptions, write_parquet};

fn write_dataset(dir: &Path) -> (String, String) {
    let t_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
        Field::new("s", DataType::Utf8, false),
    ]));
    let rows = 5_000usize;
    let k: Vec<Option<i64>> = (0..rows)
        .map(|i| (i % 11 != 0).then_some((i % 9) as i64))
        .collect();
    let v: Vec<Option<f64>> = (0..rows)
        .map(|i| (i % 13 != 0).then_some((i % 40) as f64 / 4.0))
        .collect();
    let s: Vec<String> = (0..rows).map(|i| format!("s{}", i % 7)).collect();
    let t = RecordBatch::try_new(
        Arc::clone(&t_schema),
        vec![
            Arc::new(Int64Array::from(k)),
            Arc::new(Float64Array::from(v)),
            Arc::new(StringArray::from(s)),
        ],
    )
    .unwrap();
    let r_schema = Arc::new(Schema::new(vec![
        Field::new("rk", DataType::Int64, true),
        Field::new("name", DataType::Utf8, false),
    ]));
    let r = RecordBatch::try_new(
        r_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![
                Some(0),
                Some(2),
                Some(2),
                None,
                Some(8),
                Some(42),
            ])),
            Arc::new(StringArray::from(vec![
                "zero", "two-a", "two-b", "null", "eight", "none",
            ])),
        ],
    )
    .unwrap();
    let opts = ParquetWriteOptions {
        row_group_rows: 1_000,
        compression: Compression::None,
        ..Default::default()
    };
    let t_path = dir.join("t.parquet");
    let r_path = dir.join("r.parquet");
    write_parquet(&t_path, t_schema, &[t], &opts).unwrap();
    write_parquet(&r_path, r_schema, &[r], &opts).unwrap();
    (
        t_path.to_string_lossy().into_owned(),
        r_path.to_string_lossy().into_owned(),
    )
}

fn sorted_rows(batches: &[RecordBatch]) -> Vec<String> {
    let opts = FormatOptions::default().with_null("NULL");
    let mut rows = Vec::new();
    for b in batches {
        let fs: Vec<ArrayFormatter<'_>> = b
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), &opts).unwrap())
            .collect();
        for i in 0..b.num_rows() {
            rows.push(
                fs.iter()
                    .map(|f| f.value(i).to_string())
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
    }
    rows.sort();
    rows
}

const QUERIES: [&str; 5] = [
    "SELECT k, sum(v), count(v), min(v), max(v) FROM t GROUP BY k",
    "SELECT t.k, t.s, r.name FROM t JOIN r ON t.k = r.rk",
    "SELECT s, k FROM t WHERE k >= 2 AND v < 4.0",
    "SELECT r.name, count(*) FROM t JOIN r ON t.k = r.rk WHERE t.v > 1.0 GROUP BY r.name",
    // Every 11th `t.k` is NULL, and a NULL is never `NOT IN` a non-empty
    // set, so those rows must not come back (#91).
    NOT_IN_NULL_PROBE,
];

/// `NOT IN` over a subquery with no NULLs, probed by a column with NULLs.
const NOT_IN_NULL_PROBE: &str =
    "SELECT k, s FROM t WHERE k NOT IN (SELECT rk FROM r WHERE rk IS NOT NULL)";
/// `NOT IN` over a subquery that contains a NULL: no row can qualify.
const NOT_IN_NULL_BUILD: &str = "SELECT k, s FROM t WHERE k NOT IN (SELECT rk FROM r)";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn distributed_results_equal_embedded_results() {
    let dir = tempfile::tempdir().unwrap();
    let (t_path, r_path) = write_dataset(dir.path());

    let addr = start_standalone(BackendKind::Cuda, 2, 2).await.unwrap();
    let cluster = OxideSession::connect(&format!("df://{addr}"))
        .await
        .unwrap();
    assert!(matches!(cluster.mode(), SessionMode::Cluster { .. }));
    cluster.register_parquet("t", &t_path).await.unwrap();
    cluster.register_parquet("r", &r_path).await.unwrap();

    let embedded = OxideSession::local().unwrap();
    embedded.register_parquet("t", &t_path).await.unwrap();
    embedded.register_parquet("r", &r_path).await.unwrap();

    for sql in QUERIES {
        let distributed = sorted_rows(&cluster.sql(sql).await.unwrap().collect().await.unwrap());
        let local = sorted_rows(&embedded.sql(sql).await.unwrap().collect().await.unwrap());
        assert!(!local.is_empty(), "{sql}");
        assert_eq!(distributed, local, "{sql}");
    }
}

/// `NOT IN (subquery)` keeps SQL's three-valued semantics on a cluster (#91).
///
/// No GPU operator is involved, so the cluster plans for `cpu`. Under
/// Ballista 54 the cluster session's `prefer_hash_join = false` made
/// DataFusion plan this as a `SortMergeJoinExec: LeftAnti`, which is not
/// null-aware: the NULL-key rows of `t` came back, where embedded mode
/// (stock DataFusion) returns none of them. The second query is the other
/// half of the semantics: one NULL in the subquery and nothing qualifies.
///
/// Two upstream changes each fix it on the 55 chain: DataFusion 55's planner
/// never gives a null-aware anti join to the merge join, and Ballista 55's
/// client rewrites `NOT IN` into a plain anti join and a count before one is
/// needed. This pins the answer, not either mechanism.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn not_in_a_subquery_is_null_aware_on_a_cluster() {
    let dir = tempfile::tempdir().unwrap();
    let (t_path, r_path) = write_dataset(dir.path());

    let addr = start_standalone(BackendKind::CpuSimd, 2, 2).await.unwrap();
    let cluster = OxideSession::connect(&format!("df://{addr}"))
        .await
        .unwrap();
    cluster.register_parquet("t", &t_path).await.unwrap();
    cluster.register_parquet("r", &r_path).await.unwrap();
    let embedded = OxideSession::local().unwrap();
    embedded.register_parquet("t", &t_path).await.unwrap();
    embedded.register_parquet("r", &r_path).await.unwrap();

    let distributed = rows_of(&cluster, NOT_IN_NULL_PROBE).await;
    let null_keys = distributed
        .iter()
        .filter(|r| r.starts_with("NULL|"))
        .count();
    assert_eq!(null_keys, 0, "a NULL key is never NOT IN a non-empty set");
    let local = rows_of(&embedded, NOT_IN_NULL_PROBE).await;
    assert!(!local.is_empty());
    assert_eq!(distributed, local);

    assert_eq!(
        rows_of(&cluster, NOT_IN_NULL_BUILD).await,
        Vec::<String>::new()
    );
    assert_eq!(
        rows_of(&embedded, NOT_IN_NULL_BUILD).await,
        Vec::<String>::new()
    );
}

async fn rows_of(session: &OxideSession, sql: &str) -> Vec<String> {
    sorted_rows(&session.sql(sql).await.unwrap().collect().await.unwrap())
}

/// The plan the scheduler builds for the cluster (its session builder with
/// target `cuda`) carries the placement tags; embedded mode on this GPU-less
/// machine does not.
#[tokio::test]
async fn scheduler_plans_carry_placement_tags() {
    let dir = tempfile::tempdir().unwrap();
    let (t_path, r_path) = write_dataset(dir.path());
    let state = build_session_state(session_config(), BackendKind::Cuda).unwrap();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_parquet("t", &t_path, Default::default())
        .await
        .unwrap();
    ctx.register_parquet("r", &r_path, Default::default())
        .await
        .unwrap();
    for (sql, tag) in [
        (QUERIES[0], "GpuAggregateExec[cuda]"),
        (QUERIES[1], "GpuHashJoinExec[cuda]"),
        (QUERIES[2], "GpuFilterExec[cuda]"),
    ] {
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let text = displayable(plan.as_ref()).indent(true).to_string();
        assert!(text.contains(tag), "{sql}\n{text}");
    }

    let embedded = OxideSession::local().unwrap();
    embedded.register_parquet("t", &t_path).await.unwrap();
    let text = embedded.explain(QUERIES[0]).await.unwrap();
    if let SessionMode::Embedded { target } = embedded.mode()
        && !target.is_gpu()
    {
        assert!(!text.contains("Gpu"), "{text}");
    }
}

/// The first node of `plan`, depth first, that satisfies `matches`.
fn find(
    plan: &Arc<dyn ExecutionPlan>,
    matches: &dyn Fn(&dyn ExecutionPlan) -> bool,
) -> Option<Arc<dyn ExecutionPlan>> {
    if matches(plan.as_ref()) {
        return Some(Arc::clone(plan));
    }
    plan.children().into_iter().find_map(|c| find(c, matches))
}

/// Operators above an aggregate that were planned on its hash partitions: a
/// `UNION ALL` interleaves two aggregates, and a partitioned join pairs
/// their partitions. The lowered aggregate declared one partition, so the
/// scheduler failed to plan both (#90).
const ON_AGGREGATE_PARTITIONS: [&str; 2] = [
    "SELECT k, s FROM (SELECT k, sum(v) AS s FROM t GROUP BY k) \
     UNION ALL SELECT k, s FROM (SELECT k, min(v) AS s FROM t GROUP BY k)",
    "SELECT a.k, a.s, b.m FROM (SELECT k, sum(v) AS s FROM t GROUP BY k) a \
     LEFT JOIN (SELECT k, avg(v) AS m FROM t GROUP BY k) b ON a.k = b.k",
];

/// Joins planned `Partitioned` however small their inputs, as for tables
/// too large to collect into one partition. Ballista ships the client's
/// settings with each job, so this reaches the scheduler's planner.
const PARTITIONED_JOINS: [&str; 2] = [
    "SET datafusion.optimizer.hash_join_single_partition_threshold = 0",
    "SET datafusion.optimizer.hash_join_single_partition_threshold_rows = 0",
];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn aggregates_keep_their_partitioning_on_a_cluster() {
    let dir = tempfile::tempdir().unwrap();
    let (t_path, _) = write_dataset(dir.path());

    // The scheduler's plans have the shapes that used to break, lowered.
    let ctx = SessionContext::new_with_state(
        build_session_state(session_config(), BackendKind::Cuda).unwrap(),
    );
    ctx.register_parquet("t", &t_path, Default::default())
        .await
        .unwrap();
    for set in PARTITIONED_JOINS {
        ctx.sql(set).await.unwrap();
    }
    for sql in ON_AGGREGATE_PARTITIONS {
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let text = displayable(plan.as_ref()).indent(true).to_string();
        assert!(text.contains("GpuAggregateExec[cuda]"), "{sql}\n{text}");
        // The parent that relies on the aggregate's partitions: the union's
        // interleave, or a join (sort-merge under Ballista's settings) whose
        // two inputs are hash-partitioned alike.
        let parent = find(&plan, &|n| {
            n.name() == "InterleaveExec" || n.name().ends_with("JoinExec")
        })
        .unwrap_or_else(|| panic!("{sql}\n{text}"));
        let partitioning: Vec<_> = parent
            .children()
            .iter()
            .map(|c| c.output_partitioning().clone())
            .collect();
        assert!(
            partitioning
                .iter()
                .all(|p| matches!(p, Partitioning::Hash(_, n) if *n > 1 && *n == partitioning[0].partition_count())),
            "{sql}: {partitioning:?}\n{text}"
        );
    }

    // And through the scheduler and two executors they equal embedded mode.
    let addr = start_standalone(BackendKind::Cuda, 2, 2).await.unwrap();
    let cluster = OxideSession::connect(&format!("df://{addr}"))
        .await
        .unwrap();
    cluster.register_parquet("t", &t_path).await.unwrap();
    for set in PARTITIONED_JOINS {
        cluster.sql(set).await.unwrap();
    }
    let embedded = OxideSession::local().unwrap();
    embedded.register_parquet("t", &t_path).await.unwrap();
    for sql in ON_AGGREGATE_PARTITIONS {
        let distributed = sorted_rows(&cluster.sql(sql).await.unwrap().collect().await.unwrap());
        let local = sorted_rows(&embedded.sql(sql).await.unwrap().collect().await.unwrap());
        assert!(!local.is_empty(), "{sql}");
        assert_eq!(distributed, local, "{sql}");
    }
}

#[test]
fn binaries_print_help() {
    for bin in [
        env!("CARGO_BIN_EXE_oxide-scheduler"),
        env!("CARGO_BIN_EXE_oxide-worker"),
        env!("CARGO_BIN_EXE_oxide"),
    ] {
        let out = std::process::Command::new(bin)
            .arg("--help")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{bin} --help failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("Usage"),
            "{bin}"
        );
    }
}

/// The next operator id the process-wide hub will hand out.
fn next_operator_id() -> usize {
    TelemetryHub::global()
        .snapshot()
        .operators
        .iter()
        .map(|o| o.id + 1)
        .max()
        .unwrap_or(0)
}

/// How many `name` operators registered on the process-wide hub from id
/// `since` on, and the rows they produced. The codec attaches that hub to
/// every `Gpu*Exec` an executor decodes, so this is what the executors of
/// the in-process cluster actually ran.
fn ran_since(since: usize, name: &str) -> (usize, u64) {
    TelemetryHub::global()
        .snapshot()
        .operators
        .iter()
        .filter(|o| o.id >= since && o.name == name)
        .fold((0, 0), |(count, rows), o| (count + 1, rows + o.rows_out))
}

/// The placement rule's own cases on a cluster that plans with it, against
/// embedded mode.
///
/// Ballista 55's adaptive planner, on by default, rebuilds the scheduler's
/// session with its own physical optimizer rules and never runs the
/// placement rule, so these queries use the static planner, which does.
/// Each query also checks, through the process-wide telemetry hub, that the
/// lowered node it is about ran on the executors.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_static_planner_runs_lowered_nodes_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let (t_path, r_path) = write_dataset(dir.path());
    let addr = start_standalone(BackendKind::Cuda, 2, 2).await.unwrap();
    let cluster = OxideSession::connect(&format!("df://{addr}"))
        .await
        .unwrap();
    cluster
        .sql("SET ballista.planner.adaptive.enabled = false")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    cluster.register_parquet("t", &t_path).await.unwrap();
    cluster.register_parquet("r", &r_path).await.unwrap();
    let embedded = OxideSession::local().unwrap();
    embedded.register_parquet("t", &t_path).await.unwrap();
    embedded.register_parquet("r", &r_path).await.unwrap();

    // Which rows a LIMIT keeps is unspecified, so only the count is compared.
    // Ballista does not round-robin a scan, so the single-file `t` is one
    // partition and the limit lives only in the filter's `fetch` (#86).
    let sql = "SELECT s, k FROM t WHERE k >= 2 LIMIT 3";
    let since = next_operator_id();
    let batches = cluster.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        3,
        "{sql}"
    );
    assert!(
        ran_since(since, "GpuFilterExec").0 > 0,
        "{sql}: no GpuFilterExec ran"
    );

    // COUNT(*) reads no column, so the filter and the join under it have
    // nothing to project; the filter failed planning on the scheduler (#87).
    // The lowered node ran and produced exactly the rows counted.
    for (sql, lowered) in [
        ("SELECT count(*) FROM t WHERE k >= 3", "GpuFilterExec"),
        (
            "SELECT count(*) FROM t JOIN r ON t.k = r.rk",
            "GpuHashJoinExec",
        ),
    ] {
        let since = next_operator_id();
        let batches = cluster.sql(sql).await.unwrap().collect().await.unwrap();
        let local = embedded.sql(sql).await.unwrap().collect().await.unwrap();
        assert_eq!(sorted_rows(&batches), sorted_rows(&local), "{sql}");
        let counted = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        let (ran, rows) = ran_since(since, lowered);
        assert!(
            ran > 0 && i64::try_from(rows) == Ok(counted),
            "{sql}: {ran} {lowered} produced {rows} rows, {counted} counted"
        );
    }

    // The scheduler plans these joins as sort-merge joins (Ballista's
    // sessions do not prefer hash joins), and an ORDER BY that matches the
    // merge join's own key order (ascending, NULLs first) loses its sort.
    // Lowering the merge join must keep that order (#88): it came back
    // unsorted.
    for sql in [
        "SELECT t.k, r.name FROM t JOIN r ON t.k = r.rk ORDER BY t.k NULLS FIRST",
        "SELECT r.rk, t.s FROM t JOIN r ON t.k = r.rk ORDER BY r.rk NULLS FIRST",
    ] {
        let since = next_operator_id();
        let batches = cluster.sql(sql).await.unwrap().collect().await.unwrap();
        let keys: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert!(keys.len() > 1000, "{sql}: {} rows", keys.len());
        assert!(keys.is_sorted(), "{sql}: not in key order: {keys:?}");
        let local = sorted_rows(&embedded.sql(sql).await.unwrap().collect().await.unwrap());
        assert_eq!(sorted_rows(&batches), local, "{sql}");
        assert!(
            ran_since(since, "GpuHashJoinExec").0 > 0,
            "{sql}: no GpuHashJoinExec ran"
        );
    }
}
