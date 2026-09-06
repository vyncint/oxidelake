//! Proof that DataFusion prunes the Parquet files OxideLake writes: row groups
//! skipped by statistics and by Bloom filters show up in the scan metrics, and
//! the pruned results equal the run with pruning disabled. Also pins that
//! corrupt files surface as `EngineError::Format`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::pretty::pretty_format_batches;
use datafusion::physical_plan::collect;
use datafusion::prelude::{SessionConfig, SessionContext};
use oxidelake_core::EngineError;
use oxidelake_storage::{
    Compression, ParquetWriteOptions, PruningSummary, classify_error, register_parquet_table,
    scan_pruning_metrics, with_pruning, with_pruning_disabled, write_parquet,
};

const ROWS: usize = 200_000;
const ROW_GROUP: usize = 2_000; // 100 row groups
const CHUNK: usize = 10_000;

fn lcg(state: &mut u64) -> i64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) & 0x7fff_ffff_ffff) as i64
}

/// `k` sorted (one value per row group → statistics prune), `id` random and
/// high-cardinality (only Bloom filters can prune), `v` arbitrary.
fn dataset(dir: &Path) -> (PathBuf, i64, usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("id", DataType::Int64, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let mut state = 0x5eed_u64;
    let mut ids = Vec::with_capacity(ROWS);
    let mut batches = Vec::new();
    for start in (0..ROWS).step_by(CHUNK) {
        let k: Vec<i64> = (start..start + CHUNK)
            .map(|i| (i / ROW_GROUP) as i64)
            .collect();
        let id: Vec<i64> = (0..CHUNK).map(|_| lcg(&mut state)).collect();
        ids.extend_from_slice(&id);
        let v: Vec<f64> = (start..start + CHUNK)
            .map(|i| (i % 97) as f64 / 4.0)
            .collect();
        batches.push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(k)),
                    Arc::new(Int64Array::from(id)),
                    Arc::new(Float64Array::from(v)),
                ],
            )
            .unwrap(),
        );
    }
    let probe = ids[123_456];
    let occurrences = ids.iter().filter(|&&x| x == probe).count();
    let path = dir.join("t.parquet");
    let opts = ParquetWriteOptions {
        row_group_rows: ROW_GROUP,
        compression: Compression::None,
        ..Default::default()
    }
    .with_bloom_filter("id");
    assert_eq!(
        write_parquet(&path, schema, &batches, &opts).unwrap(),
        ROWS as u64
    );
    (path, probe, occurrences)
}

async fn run(config: SessionConfig, path: &Path, sql: &str) -> (String, PruningSummary) {
    let ctx = SessionContext::new_with_config(config);
    register_parquet_table(&ctx, "t", path.to_str().unwrap())
        .await
        .unwrap();
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    (
        pretty_format_batches(&batches).unwrap().to_string(),
        scan_pruning_metrics(plan.as_ref()),
    )
}

#[tokio::test]
async fn statistics_prune_row_groups_and_results_match() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _, _) = dataset(dir.path());
    let sql = "SELECT count(*) AS n, sum(v) AS total FROM t WHERE k = 42";
    let (pruned, s) = run(with_pruning(SessionConfig::new()), &path, sql).await;
    let (full, s0) = run(with_pruning_disabled(SessionConfig::new()), &path, sql).await;
    assert_eq!(pruned, full);
    assert!(pruned.contains("2000"), "{pruned}");
    assert!(s.row_groups_pruned_statistics >= 90, "{s:?}");
    assert!(s.row_groups_matched_statistics >= 1, "{s:?}");
    assert_eq!(s0.row_groups_pruned(), 0, "{s0:?}");
}

#[tokio::test]
async fn bloom_filters_prune_row_groups_when_statistics_cannot() {
    let dir = tempfile::tempdir().unwrap();
    let (path, probe, occurrences) = dataset(dir.path());
    let sql = format!("SELECT count(*) AS n FROM t WHERE id = {probe}");
    let (pruned, s) = run(with_pruning(SessionConfig::new()), &path, &sql).await;
    let (full, _) = run(with_pruning_disabled(SessionConfig::new()), &path, &sql).await;
    assert_eq!(pruned, full);
    assert!(pruned.contains(&format!("| {occurrences} ")), "{pruned}");
    assert!(s.row_groups_pruned_bloom_filter >= 90, "{s:?}");
    assert!(
        s.row_groups_pruned_statistics < 10,
        "random ids should defeat min/max pruning: {s:?}"
    );
}

#[tokio::test]
async fn corrupt_parquet_is_a_format_error() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _, _) = dataset(dir.path());
    let len = std::fs::metadata(&path).unwrap().len();
    let truncated = dir.path().join("truncated.parquet");
    std::fs::copy(&path, &truncated).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&truncated)
        .unwrap()
        .set_len(len / 2)
        .unwrap();
    let ctx = SessionContext::new();
    let err = register_parquet_table(&ctx, "bad", truncated.to_str().unwrap())
        .await
        .unwrap_err();
    assert!(matches!(err, EngineError::Format(_)), "{err}");

    let garbage = dir.path().join("garbage.parquet");
    std::fs::write(&garbage, b"definitely not a parquet file, just bytes").unwrap();
    let err = register_parquet_table(&ctx, "bad2", garbage.to_str().unwrap())
        .await
        .unwrap_err();
    assert!(matches!(err, EngineError::Format(_)), "{err}");

    let err = classify_error(datafusion::error::DataFusionError::IoError(
        std::io::Error::other("disk"),
    ));
    assert!(matches!(err, EngineError::Io(_)));
}
