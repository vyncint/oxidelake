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
use datafusion::physical_plan::displayable;
use datafusion::prelude::SessionContext;
use oxidelake_core::BackendKind;
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

const QUERIES: [&str; 4] = [
    "SELECT k, sum(v), count(v), min(v), max(v) FROM t GROUP BY k",
    "SELECT t.k, t.s, r.name FROM t JOIN r ON t.k = r.rk",
    "SELECT s, k FROM t WHERE k >= 2 AND v < 4.0",
    "SELECT r.name, count(*) FROM t JOIN r ON t.k = r.rk WHERE t.v > 1.0 GROUP BY r.name",
];

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
