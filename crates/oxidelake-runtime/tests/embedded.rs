//! Embedded GPU-target equivalence: the same queries over the same Parquet
//! return identical rows whether the placement rule targets CPU (no rewrites),
//! CUDA or Metal. On a machine without the targeted device the `Gpu*Exec`
//! nodes take their per-batch CPU fallback; with `--features metal` on macOS
//! the Metal target actually executes on the GPU. This is the net that catches
//! multi-partition execution bugs the (differently partitioned) Ballista plans
//! hide — the shared build-side collection fix came from exactly this setup.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use oxidelake_core::BackendKind;
use oxidelake_runtime::OxideSession;
use oxidelake_storage::{Compression, demo_write_options, write_demo_table};

const QUERIES: [&str; 7] = [
    "SELECT k, sum(v), count(v), min(v), max(v) FROM t GROUP BY k",
    "SELECT a.id, b.s FROM (SELECT id, k FROM t WHERE id < 100) a \
     JOIN (SELECT k, s FROM t WHERE id >= 900) b ON a.k = b.k",
    "SELECT s, k FROM t WHERE k >= 2 AND v < 4.0",
    "SELECT b.s, count(*) FROM (SELECT id, k FROM t WHERE v > 1.0) a \
     JOIN (SELECT k, s FROM t WHERE id < 500) b ON a.k = b.k GROUP BY b.s",
    "SELECT id, l2_distance(emb, [1.0, 0.0, -1.0, 2.0, 0.5, 0.0, 1.5, -0.5]) AS d \
     FROM t WHERE k >= 50 ORDER BY d, id LIMIT 20",
    // COUNT(*) reads no column, so the filter and the join have nothing to
    // project; both used to fail to plan on a GPU target (#87).
    "SELECT count(*) FROM t WHERE k >= 2",
    "SELECT count(*) FROM t a JOIN t b ON a.k = b.k",
];

fn sorted_rows(batches: &[RecordBatch]) -> Vec<String> {
    let opts = FormatOptions::default().with_null("NULL");
    let mut rows = Vec::new();
    for batch in batches {
        let formatters: Vec<ArrayFormatter<'_>> = batch
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), &opts).unwrap())
            .collect();
        for row in 0..batch.num_rows() {
            rows.push(
                formatters
                    .iter()
                    .map(|f| f.value(row).to_string())
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
    }
    rows.sort();
    rows
}

/// A LIMIT returns its row count on every target (#86). The demo table is one
/// small file, so the scan is one partition and DataFusion leaves the limit
/// only as the `fetch` of the filter or join it pushed it into; the lowered
/// node used to drop it and return every matching row.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_limit_returns_its_row_count_on_every_target() {
    let dir = tempfile::tempdir().unwrap();
    let options = demo_write_options(512, Compression::None);
    write_demo_table(dir.path(), 4000, 11, &options).unwrap();
    let path = dir.path().to_str().unwrap();

    for target in [BackendKind::CpuSimd, BackendKind::Cuda, BackendKind::Metal] {
        let session = OxideSession::local_with_target(target).unwrap();
        session.register_parquet("t", path).await.unwrap();
        for (sql, limit) in [
            ("SELECT id, k FROM t WHERE k >= 2 LIMIT 7", 7),
            ("SELECT a.id, b.s FROM t a JOIN t b ON a.k = b.k LIMIT 5", 5),
        ] {
            let plan = session.explain(sql).await.unwrap();
            if target != BackendKind::CpuSimd {
                assert!(plan.contains("Gpu"), "no placement for {sql}:\n{plan}");
            }
            let rows = sorted_rows(&session.sql(sql).await.unwrap().collect().await.unwrap());
            assert_eq!(rows.len(), limit, "{target}: {sql}\n{plan}");
        }
    }
}

/// A WHERE with more comparisons than the device filter takes (64) returns
/// the CPU answer on every target, and `explain` says why the filter stayed
/// on the CPU (#133); it used to fail the query on a GPU target.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_predicate_over_the_device_cap_returns_the_cpu_answer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("w.parquet");
    let columns = (0..65)
        .map(|j| format!("CAST(value + {j} AS BIGINT) AS c{j}"))
        .collect::<Vec<_>>()
        .join(", ");
    OxideSession::local_with_target(BackendKind::CpuSimd)
        .unwrap()
        .sql(&format!(
            "COPY (SELECT {columns} FROM generate_series(0, 999)) TO '{}' STORED AS PARQUET",
            path.display()
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let query = |comparisons: usize| {
        let predicate = (0..comparisons)
            .map(|j| format!("c{j} >= {}", j + 3))
            .collect::<Vec<_>>()
            .join(" AND ");
        format!("SELECT count(c0), min(c0) FROM w WHERE {predicate}")
    };

    for target in [BackendKind::CpuSimd, BackendKind::Cuda, BackendKind::Metal] {
        let session = OxideSession::local_with_target(target).unwrap();
        session
            .register_parquet("w", path.to_str().unwrap())
            .await
            .unwrap();
        let rows = sorted_rows(
            &session
                .sql(&query(65))
                .await
                .unwrap()
                .collect()
                .await
                .unwrap(),
        );
        assert_eq!(rows, vec!["997|3".to_owned()], "{target}");
        if target.is_gpu() {
            let plan = session.explain(&query(65)).await.unwrap();
            assert!(
                plan.contains(
                    "  FilterExec: GpuFilterExec predicate has 65 comparisons; at most 64 \
                     are supported\n"
                ),
                "{target}:\n{plan}"
            );
            let plan = session.explain(&query(64)).await.unwrap();
            assert!(
                plan.contains("GpuFilterExec"),
                "{target}: 64 still lower:\n{plan}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gpu_targets_return_cpu_results() {
    let dir = tempfile::tempdir().unwrap();
    let options = demo_write_options(512, Compression::None);
    write_demo_table(dir.path(), 4000, 11, &options).unwrap();
    let path = dir.path().to_str().unwrap();

    let mut reference: Vec<Vec<String>> = Vec::new();
    for target in [BackendKind::CpuSimd, BackendKind::Cuda, BackendKind::Metal] {
        let session = OxideSession::local_with_target(target).unwrap();
        session.register_parquet("t", path).await.unwrap();
        for (i, sql) in QUERIES.iter().enumerate() {
            if target != BackendKind::CpuSimd {
                let plan = session.explain(sql).await.unwrap();
                assert!(plan.contains("Gpu"), "no placement for {sql}:\n{plan}");
            }
            let rows = sorted_rows(&session.sql(sql).await.unwrap().collect().await.unwrap());
            assert!(!rows.is_empty(), "{sql}");
            match target {
                BackendKind::CpuSimd => reference.push(rows),
                _ => assert_eq!(rows, reference[i], "{target}: {sql}"),
            }
        }
    }
}

/// The `Int64` values of the first column, in output order.
fn first_column(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let column = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(column.null_count(), 0);
            column.values().to_vec()
        })
        .collect()
}

/// Under `prefer_hash_join = false`, Ballista's session default, DataFusion
/// plans a sort-merge join and leaves out the sort an ORDER BY on its key
/// needs, because the merge join's output already has that order; a merge
/// join or a window above reads the order the same way. The lowered join
/// returned the ORDER BY unsorted on a GPU target, lost rows in the LEFT
/// join above it and failed the window (#88). Every target returns the CPU
/// target's rows now, in its key order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_join_keeps_its_order_on_every_target() {
    let dir = tempfile::tempdir().unwrap();
    let options = demo_write_options(512, Compression::None);
    write_demo_table(dir.path(), 4000, 11, &options).unwrap();
    let path = dir.path().to_str().unwrap();
    let a = "(SELECT id, k FROM t WHERE id < 200)";
    let b = "(SELECT id, k FROM t WHERE id >= 3800)";
    // `k` is nullable, so only NULLS FIRST is the merge join's own order
    // and lets DataFusion drop the sort.
    let queries = [
        format!(
            "SELECT a.k, a.id, b.id FROM {a} a JOIN {b} b ON a.k = b.k ORDER BY a.k NULLS FIRST"
        ),
        format!(
            "SELECT a.k, b.id, c.id FROM {a} a JOIN {b} b ON a.k = b.k \
             LEFT JOIN {b} c ON a.k = c.k"
        ),
        format!(
            "SELECT a.k, b.id, count(*) OVER (PARTITION BY b.k) \
             FROM {a} a JOIN {b} b ON a.k = b.k"
        ),
    ];

    let mut reference: Vec<(Vec<i64>, Vec<String>)> = Vec::new();
    for target in [BackendKind::CpuSimd, BackendKind::Cuda, BackendKind::Metal] {
        let session = OxideSession::local_with_target(target).unwrap();
        session
            .sql("SET datafusion.optimizer.prefer_hash_join = false")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        session.register_parquet("t", path).await.unwrap();
        for (i, sql) in queries.iter().enumerate() {
            let plan = session.explain(sql).await.unwrap();
            let lowered = plan.contains("GpuHashJoinExec");
            assert!(
                plan.contains("SortMergeJoinExec") || lowered,
                "{target}: no merge join to cover:\n{plan}"
            );
            assert_eq!(lowered, target.is_gpu(), "{target}: {sql}\n{plan}");
            let batches = session
                .sql(sql)
                .await
                .unwrap()
                .collect()
                .await
                .unwrap_or_else(|e| panic!("{target}: {sql}: {e}\n{plan}"));
            let keys = first_column(&batches);
            let rows = sorted_rows(&batches);
            assert!(rows.len() > 100, "{target}: {sql}: {} rows", rows.len());
            if i == 0 {
                assert!(
                    keys.is_sorted(),
                    "{target}: not in key order: {keys:?}\n{plan}"
                );
            }
            match target {
                BackendKind::CpuSimd => reference.push((keys, rows)),
                _ => {
                    let (want_keys, want_rows) = &reference[i];
                    if i == 0 {
                        assert_eq!(&keys, want_keys, "{target}: {sql}");
                    }
                    assert_eq!(&rows, want_rows, "{target}: {sql}");
                }
            }
        }
    }
}
