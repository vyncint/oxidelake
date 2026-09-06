//! Conformance suite (docs/SPEC.md §2.3): every `Gpu*Exec`, running on the
//! `CpuBackend`, must produce the same rows as the stock DataFusion operators
//! over seeded random batches with nulls, empty inputs and odd row counts.
//!
//! The same assertions run against CUDA/Metal when a device is present: set
//! `OXIDE_BACKEND=cuda` (or `metal`) and run with `--ignored` — see
//! `gpu_backend_matches_datafusion_when_available`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::SessionContext;
use oxidelake_compute::{
    AggregateFunction, AggregateSpec, Comparison, DistanceMetric, GpuAggregateExec, GpuFilterExec,
    GpuHashJoinExec, GpuOperator, GpuVectorDistanceExec, Literal, Predicate,
};
use oxidelake_core::BackendKind;
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_device::{CpuBackend, GpuBackend, HardwareDetector};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const DIM: usize = 4;
const SIZES: [usize; 5] = [0, 1, 7, 64, 1000];
const SEEDS: [u64; 3] = [1, 7, 42];

/// `k Int64?`, `v Float64?` (multiples of 0.25 so sums are exact), `s Utf8`, `vec FixedSizeList<Float32, DIM>?`.
fn gen_left(seed: u64, rows: usize) -> RecordBatch {
    let mut rng = StdRng::seed_from_u64(seed);
    let k: Vec<Option<i64>> = (0..rows)
        .map(|_| (!rng.random_bool(0.1)).then(|| rng.random_range(0..10i64)))
        .collect();
    let v: Vec<Option<f64>> = (0..rows)
        .map(|_| (!rng.random_bool(0.1)).then(|| f64::from(rng.random_range(0..40i32)) / 4.0))
        .collect();
    let s: Vec<String> = (0..rows).map(|i| format!("s{}", i % 13)).collect();
    let values: Vec<f32> = (0..rows * DIM)
        .map(|_| rng.random_range(-5.0f32..5.0))
        .collect();
    let valid: Vec<bool> = (0..rows).map(|_| !rng.random_bool(0.1)).collect();
    let vectors = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        DIM as i32,
        Arc::new(Float32Array::from(values)),
        Some(NullBuffer::from(valid)),
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
        Field::new("s", DataType::Utf8, false),
        Field::new("vec", vectors.data_type().clone(), true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(k)),
            Arc::new(Float64Array::from(v)),
            Arc::new(StringArray::from(s)),
            Arc::new(vectors),
        ],
    )
    .unwrap()
}

/// `rk Int64?`, `name Utf8`.
fn gen_right(seed: u64, rows: usize) -> RecordBatch {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xdead_beef);
    let rk: Vec<Option<i64>> = (0..rows)
        .map(|_| (!rng.random_bool(0.1)).then(|| rng.random_range(0..10i64)))
        .collect();
    let name: Vec<String> = (0..rows).map(|i| format!("n{i}")).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rk", DataType::Int64, true),
        Field::new("name", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(rk)),
            Arc::new(StringArray::from(name)),
        ],
    )
    .unwrap()
}

/// Every row rendered as `a|b|c`, sorted, so results compare as multisets and
/// column names are ignored.
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
            let cells: Vec<String> = formatters
                .iter()
                .map(|f| f.value(row).to_string())
                .collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    rows
}

async fn physical_plan(ctx: &SessionContext, batch: RecordBatch) -> Arc<dyn ExecutionPlan> {
    ctx.read_batch(batch)
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap()
}

async fn sql_rows(ctx: &SessionContext, sql: &str) -> Vec<String> {
    sorted_rows(&ctx.sql(sql).await.unwrap().collect().await.unwrap())
}

async fn exec_rows(ctx: &SessionContext, plan: Arc<dyn ExecutionPlan>) -> Vec<String> {
    sorted_rows(&collect(plan, ctx.task_ctx()).await.unwrap())
}

fn cpu() -> Arc<dyn GpuBackend> {
    Arc::new(CpuBackend::new())
}

async fn check_filter(backend: Arc<dyn GpuBackend>, seed: u64, rows: usize) {
    let ctx = SessionContext::new();
    let batch = gen_left(seed, rows);
    ctx.register_batch("t", batch.clone()).unwrap();
    let expected = sql_rows(&ctx, "SELECT s, k FROM t WHERE k >= 2 AND v < 4.0").await;
    let predicate = Predicate::and(
        Predicate::compare(0, Comparison::GtEq, Literal::Int64(2)),
        Predicate::compare(1, Comparison::Lt, Literal::Float64(4.0)),
    );
    let exec = GpuFilterExec::try_new(
        physical_plan(&ctx, batch).await,
        predicate,
        vec![2, 0],
        backend.kind(),
    )
    .unwrap()
    .with_backend(backend);
    let actual = exec_rows(&ctx, Arc::new(exec)).await;
    assert_eq!(actual, expected, "filter seed={seed} rows={rows}");
}

async fn check_join(backend: Arc<dyn GpuBackend>, seed: u64, rows: usize) {
    let ctx = SessionContext::new();
    let left = gen_left(seed, rows);
    let right = gen_right(seed, rows / 2 + 1);
    ctx.register_batch("l", left.clone()).unwrap();
    ctx.register_batch("r", right.clone()).unwrap();
    let expected = sql_rows(
        &ctx,
        "SELECT l.k, l.v, l.s, l.vec, r.rk, r.name FROM l INNER JOIN r ON l.k = r.rk",
    )
    .await;
    let exec = GpuHashJoinExec::try_new(
        physical_plan(&ctx, left).await,
        physical_plan(&ctx, right).await,
        0,
        0,
        backend.kind(),
    )
    .unwrap()
    .with_backend(backend);
    let actual = exec_rows(&ctx, Arc::new(exec)).await;
    assert_eq!(actual, expected, "join seed={seed} rows={rows}");
}

async fn check_aggregate(backend: Arc<dyn GpuBackend>, seed: u64, rows: usize) {
    let ctx = SessionContext::new();
    let batch = gen_left(seed, rows);
    ctx.register_batch("t", batch.clone()).unwrap();
    let expected = sql_rows(
        &ctx,
        "SELECT k, SUM(v), COUNT(v), MIN(v), MAX(k) FROM t GROUP BY k",
    )
    .await;
    let spec = AggregateSpec {
        group_by: 0,
        aggregates: vec![
            (AggregateFunction::Sum, 1),
            (AggregateFunction::Count, 1),
            (AggregateFunction::Min, 1),
            (AggregateFunction::Max, 0),
        ],
    };
    let exec = GpuAggregateExec::try_new(physical_plan(&ctx, batch).await, spec, backend.kind())
        .unwrap()
        .with_backend(backend);
    let actual = exec_rows(&ctx, Arc::new(exec)).await;
    assert_eq!(actual, expected, "aggregate seed={seed} rows={rows}");
}

fn reference_distances(
    batch: &RecordBatch,
    query: &[f32],
    metric: DistanceMetric,
) -> Vec<Option<f64>> {
    let list = batch
        .column(3)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    (0..batch.num_rows())
        .map(|row| {
            if list.is_null(row) {
                return None;
            }
            let start = list.value_offset(row) as usize;
            let v: Vec<f64> = (0..DIM)
                .map(|i| f64::from(values.value(start + i)))
                .collect();
            let q: Vec<f64> = query.iter().map(|&x| f64::from(x)).collect();
            Some(match metric {
                DistanceMetric::L2 => v
                    .iter()
                    .zip(&q)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt(),
                DistanceMetric::Cosine => {
                    let dot: f64 = v.iter().zip(&q).map(|(a, b)| a * b).sum();
                    let na = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                    let nb = q.iter().map(|b| b * b).sum::<f64>().sqrt();
                    if na == 0.0 || nb == 0.0 {
                        f64::NAN
                    } else {
                        1.0 - dot / (na * nb)
                    }
                }
            })
        })
        .collect()
}

async fn check_vector(
    backend: Arc<dyn GpuBackend>,
    seed: u64,
    rows: usize,
    metric: DistanceMetric,
) {
    let ctx = SessionContext::new();
    let batch = gen_left(seed, rows);
    let query = [0.5f32, -1.0, 2.0, 0.25];
    let expected = reference_distances(&batch, &query, metric);
    let exec = GpuVectorDistanceExec::try_new(
        physical_plan(&ctx, batch).await,
        3,
        query.to_vec(),
        metric,
        "dist",
        backend.kind(),
    )
    .unwrap()
    .with_backend(backend);
    let out = collect(Arc::new(exec), ctx.task_ctx()).await.unwrap();
    let mut actual: Vec<Option<f32>> = Vec::new();
    for b in &out {
        assert_eq!(b.schema().field(4).name(), "dist");
        let d = b.column(4).as_any().downcast_ref::<Float32Array>().unwrap();
        actual.extend(d.iter());
    }
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        match (a, e) {
            (None, None) => {}
            (Some(a), Some(e)) => assert!(
                (f64::from(*a) - e).abs() < 1e-4 || (a.is_nan() && e.is_nan()),
                "row {i}: got {a}, expected {e} (seed={seed} rows={rows} {metric:?})"
            ),
            other => panic!("row {i}: null mismatch {other:?}"),
        }
    }
}

async fn run_all(backend: Arc<dyn GpuBackend>) {
    for &seed in &SEEDS {
        for &rows in &SIZES {
            check_filter(Arc::clone(&backend), seed, rows).await;
            check_join(Arc::clone(&backend), seed, rows).await;
            check_aggregate(Arc::clone(&backend), seed, rows).await;
            check_vector(Arc::clone(&backend), seed, rows, DistanceMetric::L2).await;
            check_vector(Arc::clone(&backend), seed, rows, DistanceMetric::Cosine).await;
        }
    }
}

#[tokio::test]
async fn cpu_backend_matches_datafusion() {
    run_all(cpu()).await;
}

/// Same suite on the detected GPU backend. Ignored by default because the
/// development machine has no device; run on a GPU box with
/// `OXIDE_BACKEND=cuda cargo test -p oxidelake-compute --features cuda -- --ignored`.
#[tokio::test]
#[ignore = "requires a GPU backend (CUDA or Metal) on this machine"]
async fn gpu_backend_matches_datafusion_when_available() {
    let backend = HardwareDetector::select().unwrap();
    assert!(
        backend.kind().is_gpu(),
        "no GPU backend detected: {:?}",
        backend.kind()
    );
    run_all(Arc::clone(&backend)).await;
    // Equal results alone would also be produced by the CPU fallback. Prove the
    // device actually ran: an operator with telemetry moves bytes to and from
    // the device for an Int64 filter (supported by every GPU backend) even
    // though the batch carries a Utf8 column, which stays on the host.
    let hub = TelemetryHub::new();
    let stats = hub.register_operator("GpuFilterExec", backend.kind());
    let operator = GpuOperator::new(backend, Some(Arc::clone(&stats))).unwrap();
    let batch = gen_left(3, 500);
    let predicate = Predicate::compare(0, Comparison::GtEq, Literal::Int64(4));
    let out = operator
        .filter_project(&batch, &predicate, &[2, 0])
        .unwrap();
    let snapshot = stats.snapshot();
    assert!(
        snapshot.bytes_h2d > 0,
        "no bytes were uploaded: {snapshot:?}"
    );
    assert!(
        snapshot.bytes_d2h > 0,
        "no bytes were downloaded: {snapshot:?}"
    );
    assert_eq!(out.schema().field(0).name(), "s");
    assert_eq!(snapshot.rows_out, out.num_rows() as u64);
}

/// The CPU backend never moves bytes anywhere — the transfer counters are a
/// reliable witness of device execution in the test above.
#[test]
fn cpu_backend_reports_no_transfers() {
    let hub = TelemetryHub::new();
    let stats = hub.register_operator("GpuFilterExec", BackendKind::CpuSimd);
    let operator = GpuOperator::new(cpu(), Some(Arc::clone(&stats))).unwrap();
    let batch = gen_left(3, 100);
    let predicate = Predicate::compare(0, Comparison::GtEq, Literal::Int64(4));
    let out = operator
        .filter_project(&batch, &predicate, &[2, 0])
        .unwrap();
    let snapshot = stats.snapshot();
    assert_eq!((snapshot.bytes_h2d, snapshot.bytes_d2h), (0, 0));
    assert_eq!(snapshot.rows_in, 100);
    assert_eq!(snapshot.rows_out, out.num_rows() as u64);
}

#[tokio::test]
async fn explain_shows_placement_tags() {
    let ctx = SessionContext::new();
    let batch = gen_left(1, 8);
    let scan = physical_plan(&ctx, batch.clone()).await;
    let filter: Arc<dyn ExecutionPlan> = Arc::new(
        GpuFilterExec::try_new(
            Arc::clone(&scan),
            Predicate::compare(0, Comparison::Gt, Literal::Int64(1)),
            vec![0, 1],
            BackendKind::Cuda,
        )
        .unwrap(),
    );
    let text = displayable(filter.as_ref()).indent(true).to_string();
    assert!(
        text.contains("GpuFilterExec[cuda]: k > 1, projection=[k, v]"),
        "{text}"
    );

    let agg = GpuAggregateExec::try_new(
        Arc::clone(&scan),
        AggregateSpec {
            group_by: 0,
            aggregates: vec![(AggregateFunction::Sum, 1)],
        },
        BackendKind::Metal,
    )
    .unwrap();
    let text = displayable(&agg).indent(true).to_string();
    assert!(
        text.contains("GpuAggregateExec[metal]: group_by=[k], aggr=[SUM(v)]"),
        "{text}"
    );
    assert_eq!(agg.schema().field(1).name(), "SUM(v)");

    let join = GpuHashJoinExec::try_new(
        Arc::clone(&scan),
        physical_plan(&ctx, gen_right(1, 4)).await,
        0,
        0,
        BackendKind::CpuSimd,
    )
    .unwrap();
    let text = displayable(&join).indent(true).to_string();
    assert!(
        text.contains("GpuHashJoinExec[cpu]: join_type=Inner, on=[k = rk]"),
        "{text}"
    );

    let vec = GpuVectorDistanceExec::try_new(
        scan,
        3,
        vec![0.0; DIM],
        DistanceMetric::Cosine,
        "d",
        BackendKind::Cuda,
    )
    .unwrap();
    let text = displayable(&vec).indent(true).to_string();
    assert!(
        text.contains("GpuVectorDistanceExec[cuda]: cosine(vec) AS d, dim=4"),
        "{text}"
    );
}

#[tokio::test]
async fn invalid_plans_are_rejected_at_construction() {
    let ctx = SessionContext::new();
    let scan = physical_plan(&ctx, gen_left(1, 4)).await;
    // literal type mismatch: v is Float64
    assert!(
        GpuFilterExec::try_new(
            Arc::clone(&scan),
            Predicate::compare(1, Comparison::Eq, Literal::Int64(1)),
            vec![0],
            BackendKind::CpuSimd
        )
        .is_err()
    );
    // string key
    assert!(
        GpuHashJoinExec::try_new(
            Arc::clone(&scan),
            Arc::clone(&scan),
            2,
            0,
            BackendKind::CpuSimd
        )
        .is_err()
    );
    // string aggregate input
    assert!(
        GpuAggregateExec::try_new(
            Arc::clone(&scan),
            AggregateSpec {
                group_by: 0,
                aggregates: vec![(AggregateFunction::Sum, 2)]
            },
            BackendKind::CpuSimd
        )
        .is_err()
    );
    // wrong query dimension
    assert!(
        GpuVectorDistanceExec::try_new(
            scan,
            3,
            vec![1.0; 3],
            DistanceMetric::L2,
            "d",
            BackendKind::CpuSimd
        )
        .is_err()
    );
}

#[test]
fn row_rendering_is_stable() {
    let batch = gen_left(3, 3);
    let rows = sorted_rows(&[batch.clone(), RecordBatch::new_empty(batch.schema())]);
    assert_eq!(rows.len(), 3);
    let cols: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None]));
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let b = RecordBatch::try_new(schema, vec![cols]).unwrap();
    assert_eq!(sorted_rows(&[b]), vec!["1".to_owned(), "NULL".to_owned()]);
}

/// The build side must be collected once and shared across probe partitions.
/// A shared-state build child — `RepartitionExec` hands each row to exactly
/// one consumer — would otherwise give every probe partition a different
/// fragment of the build side and silently drop join matches. Found on the
/// first machine where embedded GPU placement executed multi-partition plans.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_collects_repartitioned_build_side_once() {
    use datafusion::physical_expr::expressions::Column;
    use datafusion::physical_plan::Partitioning;
    use datafusion::physical_plan::repartition::RepartitionExec;

    let ctx = SessionContext::new();
    let left = gen_left(7, 500).project(&[0, 2]).unwrap(); // k, s
    let right = gen_right(7, 40);
    ctx.register_batch("t", left.clone()).unwrap();
    ctx.register_batch("r", right.clone()).unwrap();
    let expected = sql_rows(&ctx, "SELECT t.k, t.s, r.name FROM t JOIN r ON t.k = r.rk").await;
    assert!(!expected.is_empty());

    let probe = Arc::new(
        RepartitionExec::try_new(
            physical_plan(&ctx, left).await,
            Partitioning::Hash(vec![Arc::new(Column::new("k", 0))], 4),
        )
        .unwrap(),
    );
    let build = Arc::new(
        RepartitionExec::try_new(
            physical_plan(&ctx, right).await,
            Partitioning::Hash(vec![Arc::new(Column::new("rk", 0))], 4),
        )
        .unwrap(),
    );
    // Joined columns are k, s, rk, name; keep k, s, name.
    let join = GpuHashJoinExec::try_new(probe, build, 0, 0, BackendKind::CpuSimd)
        .unwrap()
        .with_projection(Some(vec![0, 1, 3]))
        .unwrap()
        .with_backend(cpu());
    assert_eq!(exec_rows(&ctx, Arc::new(join)).await, expected);
}
