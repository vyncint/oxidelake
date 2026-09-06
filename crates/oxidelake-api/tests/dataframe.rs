//! The fluent DataFrame API against SQL over the generated demo table: every
//! verb produces the same rows as the equivalent SQL, and a GPU-target session
//! plans `Gpu*Exec` nodes for the fluent pipeline too.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use oxidelake_api::prelude::*;
use oxidelake_storage::{Compression, demo_write_options, write_demo_table};
use tempfile::TempDir;

fn demo_dir(rows: u64) -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    let options = demo_write_options(2048, Compression::None);
    write_demo_table(dir.path(), rows, 7, &options).unwrap();
    dir
}

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

async fn session_over_demo(dir: &TempDir, target: BackendKind) -> OxideSession {
    let session = OxideSession::local_with_target(target).unwrap();
    session
        .register_parquet("t", dir.path().to_str().unwrap())
        .await
        .unwrap();
    session
}

#[tokio::test]
async fn filter_select_aggregate_match_sql() {
    let dir = demo_dir(4000);
    let session = session_over_demo(&dir, BackendKind::Cuda).await;
    let frame_rows = sorted_rows(
        &session
            .table("t")
            .await
            .unwrap()
            .filter(col("k").gt_eq(lit(2)))
            .unwrap()
            .aggregate(vec![col("k")], vec![sum(col("v")), count(col("v"))])
            .unwrap()
            .collect()
            .await
            .unwrap(),
    );
    let sql_rows = sorted_rows(
        &session
            .sql("SELECT k, sum(v), count(v) FROM t WHERE k >= 2 GROUP BY k")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
    );
    assert!(!frame_rows.is_empty());
    assert_eq!(frame_rows, sql_rows);
}

#[tokio::test]
async fn join_and_select_match_sql() {
    let dir = demo_dir(2000);
    let session = session_over_demo(&dir, BackendKind::Cuda).await;
    // Self-join on the low-cardinality key, projected small to keep it cheap.
    let left = session
        .table("t")
        .await
        .unwrap()
        .filter(col("id").lt(lit(50)))
        .unwrap()
        .select(&["id", "k"])
        .unwrap()
        .alias("a")
        .unwrap();
    let right = session
        .table("t")
        .await
        .unwrap()
        .filter(col("id").gt_eq(lit(1950)))
        .unwrap()
        .select(&["k", "s"])
        .unwrap()
        .alias("b")
        .unwrap();
    let frame_rows = sorted_rows(
        &left
            .join(right, "k", "k")
            .unwrap()
            .select(&["id", "s"])
            .unwrap()
            .collect()
            .await
            .unwrap(),
    );
    let sql_rows = sorted_rows(
        &session
            .sql(
                "SELECT a.id, b.s FROM \
                 (SELECT id, k FROM t WHERE id < 50) a JOIN \
                 (SELECT k, s FROM t WHERE id >= 1950) b ON a.k = b.k",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
    );
    assert!(!frame_rows.is_empty());
    assert_eq!(frame_rows, sql_rows);
}

#[tokio::test]
async fn vector_distance_matches_sql_and_lowers_on_gpu_targets() {
    let dir = demo_dir(500);
    let session = session_over_demo(&dir, BackendKind::Metal).await;
    let query = [0.5f32, -1.0, 2.0, 0.0, 1.0, -2.0, 0.25, 3.0];
    let frame = session
        .table("t")
        .await
        .unwrap()
        .vector_distance("emb", &query, DistanceMetric::L2, "d")
        .unwrap();
    let plan = frame.explain().await.unwrap();
    assert!(
        plan.contains("GpuVectorDistanceExec[metal]: l2(emb) AS d, dim=8"),
        "{plan}"
    );
    let frame_rows = sorted_rows(
        &frame
            .sort(vec![col("d").sort(true, false)])
            .unwrap()
            .limit(5)
            .unwrap()
            .select(&["id", "d"])
            .unwrap()
            .collect()
            .await
            .unwrap(),
    );
    let sql_rows = sorted_rows(
        &session
            .sql(
                "SELECT id, l2_distance(emb, [0.5, -1.0, 2.0, 0.0, 1.0, -2.0, 0.25, 3.0]) AS d \
                 FROM t ORDER BY d LIMIT 5",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
    );
    assert_eq!(frame_rows.len(), 5);
    assert_eq!(frame_rows, sql_rows);
}

#[tokio::test]
async fn read_parquet_and_explain_show_placement() {
    let dir = demo_dir(1000);
    let session = session_over_demo(&dir, BackendKind::Cuda).await;
    let frame = session
        .read_parquet(dir.path().to_str().unwrap())
        .await
        .unwrap()
        .filter(col("k").gt_eq(lit(2)).and(col("v").lt(lit(4.0))))
        .unwrap()
        .select(&["k", "v"])
        .unwrap();
    let plan = frame.explain().await.unwrap();
    assert!(plan.contains("GpuFilterExec[cuda]"), "{plan}");
    let schema = frame.schema();
    assert_eq!(schema.fields().len(), 2);
    let rows = frame.collect().await.unwrap();
    assert!(rows.iter().map(RecordBatch::num_rows).sum::<usize>() > 0);
}
