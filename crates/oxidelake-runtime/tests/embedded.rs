//! Embedded GPU-target equivalence: the same queries over the same Parquet
//! return identical rows whether the placement rule targets CPU (no rewrites),
//! CUDA or Metal. On a machine without the targeted device the `Gpu*Exec`
//! nodes take their per-batch CPU fallback; with `--features metal` on macOS
//! the Metal target actually executes on the GPU. This is the net that catches
//! multi-partition execution bugs the (differently partitioned) Ballista plans
//! hide — the shared build-side collection fix came from exactly this setup.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use arrow::array::RecordBatch;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use oxidelake_core::BackendKind;
use oxidelake_runtime::OxideSession;
use oxidelake_storage::{Compression, demo_write_options, write_demo_table};

const QUERIES: [&str; 5] = [
    "SELECT k, sum(v), count(v), min(v), max(v) FROM t GROUP BY k",
    "SELECT a.id, b.s FROM (SELECT id, k FROM t WHERE id < 100) a \
     JOIN (SELECT k, s FROM t WHERE id >= 900) b ON a.k = b.k",
    "SELECT s, k FROM t WHERE k >= 2 AND v < 4.0",
    "SELECT b.s, count(*) FROM (SELECT id, k FROM t WHERE v > 1.0) a \
     JOIN (SELECT k, s FROM t WHERE id < 500) b ON a.k = b.k GROUP BY b.s",
    "SELECT id, l2_distance(emb, [1.0, 0.0, -1.0, 2.0, 0.5, 0.0, 1.5, -0.5]) AS d \
     FROM t WHERE k >= 50 ORDER BY d, id LIMIT 20",
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
