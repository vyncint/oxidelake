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
use datafusion::prelude::{SessionConfig, SessionContext};
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

/// `batch` with its `Int64` column `column` reduced modulo `keys`, NULLs kept.
fn skew(batch: &RecordBatch, column: usize, keys: i64) -> RecordBatch {
    let mut columns = batch.columns().to_vec();
    columns[column] =
        arrow::compute::kernels::numeric::rem(&columns[column], &Int64Array::new_scalar(keys))
            .unwrap();
    RecordBatch::try_new(batch.schema(), columns).unwrap()
}

/// A probe batch whose matches outnumber the session's batch size comes out
/// in batches of at most that size, with stock DataFusion's rows (#95). Three
/// keys on each side make every probe row match about a third of the build
/// side, so chunk boundaries fall inside one probe row's matches as well as
/// between rows; `CHUNK` is prime so they do not line up with either.
async fn check_join_chunked(backend: Arc<dyn GpuBackend>, seed: u64) {
    const CHUNK: usize = 61;
    let stock = SessionContext::new();
    let left = skew(&gen_left(seed, 300), 0, 3);
    let right = skew(&gen_right(seed, 200), 0, 3);
    stock.register_batch("l", left.clone()).unwrap();
    stock.register_batch("r", right.clone()).unwrap();
    let expected = sql_rows(
        &stock,
        "SELECT l.k, l.v, l.s, l.vec, r.rk, r.name FROM l INNER JOIN r ON l.k = r.rk",
    )
    .await;
    assert!(expected.len() > 50 * CHUNK, "{} rows", expected.len());

    let ctx = SessionContext::new_with_config(SessionConfig::new().with_batch_size(CHUNK));
    let exec = GpuHashJoinExec::try_new(
        physical_plan(&ctx, left).await,
        physical_plan(&ctx, right).await,
        0,
        0,
        backend.kind(),
    )
    .unwrap()
    .with_backend(backend);
    let batches = collect(Arc::new(exec), ctx.task_ctx()).await.unwrap();
    let largest = batches.iter().map(RecordBatch::num_rows).max().unwrap_or(0);
    assert!(
        largest <= CHUNK,
        "seed={seed}: a {largest}-row batch from a join whose session batch size is {CHUNK}"
    );
    assert_eq!(sorted_rows(&batches), expected, "chunked join seed={seed}");
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

// --- aggregate edge cases (#100, #46) ----------------------------------------
//
// Each fixture below exists to pin one semantic, so it is generated on its
// own: the filter and join checks above must not change meaning because an
// aggregate fixture grew a row.

/// `k Int64?`, `x Int64`: every group's `SUM(x)` leaves the Int64 range. Group
/// 1 is #100's reproduction (`i64::MAX - 1` twice wraps to -4), group 3 wraps
/// downwards, group 4 overflows on the way and ends back in range, and the
/// `NULL`-key group overflows too. Stock DataFusion wraps every one of them
/// (`add_wrapping` in each `SUM` accumulator), and it is the reference.
fn gen_sum_overflow() -> RecordBatch {
    let rows = [
        (Some(1), i64::MAX - 1),
        (Some(2), 5),
        (Some(1), i64::MAX - 1),
        (Some(3), i64::MIN),
        (Some(4), i64::MAX),
        (Some(3), -1),
        (None, i64::MAX),
        (Some(4), 1),
        (None, i64::MAX),
        (Some(4), -2),
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("x", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|(_, x)| *x).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// `SUM(Int64)` wraps on overflow on every backend, as stock DataFusion does
/// (#100; #46 asked for the opposite on a premise that turned out inverted).
async fn check_sum_overflow(backend: Arc<dyn GpuBackend>) {
    let ctx = SessionContext::new();
    let batch = gen_sum_overflow();
    ctx.register_batch("o", batch.clone()).unwrap();
    let expected = sql_rows(&ctx, "SELECT k, SUM(x), COUNT(x) FROM o GROUP BY k").await;
    // Pin the oracle as well as the operator: if DataFusion ever starts
    // erroring here, this suite should say so rather than follow it.
    assert_eq!(
        expected,
        [
            "1|-4|2",
            "2|5|1",
            "3|9223372036854775807|2",
            "4|9223372036854775806|3",
            "NULL|-2|2",
        ],
        "stock DataFusion no longer wraps SUM(Int64)"
    );
    let spec = AggregateSpec {
        group_by: 0,
        aggregates: vec![(AggregateFunction::Sum, 1), (AggregateFunction::Count, 1)],
    };
    let exec = GpuAggregateExec::try_new(physical_plan(&ctx, batch).await, spec, backend.kind())
        .unwrap()
        .with_backend(backend);
    let actual = exec_rows(&ctx, Arc::new(exec)).await;
    assert_eq!(actual, expected, "SUM(Int64) overflow");
}

#[tokio::test]
async fn cpu_sum_overflow_wraps_like_datafusion() {
    check_sum_overflow(cpu()).await;
}

/// The aggregate edge cases on the detected GPU backend. Ignored by default
/// because the development machine has no device; on a CUDA machine run
/// `OXIDE_BACKEND=cuda cargo test -p oxidelake-compute --features cuda --test conformance -- --ignored`.
/// Equal answers would also come from the CPU fallback, so a backend that
/// aggregates on its device must also show that it did: bytes uploaded and
/// no batch fallen back. Metal declares aggregates unsupported; there the
/// checks pin the fallback's answers.
#[tokio::test]
#[ignore = "requires a GPU backend (CUDA or Metal) on this machine"]
async fn gpu_aggregate_edges_match_the_reference_when_available() {
    let backend = HardwareDetector::select().unwrap();
    assert!(
        backend.kind().is_gpu(),
        "no GPU backend detected: {:?}",
        backend.kind()
    );
    check_sum_overflow(Arc::clone(&backend)).await;
    check_float_min_max(Arc::clone(&backend)).await;
    if backend.supports_aggregate() {
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuAggregateExec", backend.kind());
        let operator = GpuOperator::new(Arc::clone(&backend), Some(Arc::clone(&stats))).unwrap();
        let batch = gen_sum_overflow();
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![(AggregateFunction::Sum, 1)],
        };
        let device = operator.aggregate(&batch, &spec).unwrap();
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.fallback_batches, 0, "{snapshot:?}");
        assert!(snapshot.bytes_h2d > 0, "{snapshot:?}");
        let reference = GpuOperator::new(cpu(), None)
            .unwrap()
            .aggregate(&batch, &spec)
            .unwrap();
        assert_eq!(sorted_rows(&[device]), sorted_rows(&[reference]));
    }
}

// --- MIN/MAX(Float64) in IEEE 754 totalOrder (#101) -----------------------------
//
// Stock DataFusion is not the oracle here: its grouped float MIN/MAX lets a
// NaN reset the running value (MIN of {1.0, NaN, 3.0} is 3.0) and leaks its
// starting values (MAX of {-inf} is f64::MIN), so its answer depends on row
// order. The answers below are Arrow's order — `f64::total_cmp`, the order
// its comparison and sort kernels (and so the CPU filter path) already use:
// -NaN < -inf < … < -0.0 < 0.0 < … < inf < NaN.

/// A group key and one of its values.
type FloatRow = (Option<i64>, f64);

/// A group, its values, and the `(MIN, MAX)` it must come out with.
type FloatGroup = (Option<i64>, Vec<f64>, (f64, f64));

fn special_float_groups() -> Vec<FloatGroup> {
    let nan = f64::NAN;
    vec![
        // #101's table, row by row.
        (Some(1), vec![1.0, nan, 3.0], (1.0, nan)),
        (Some(2), vec![nan, nan], (nan, nan)),
        (Some(3), vec![0.0, -0.0], (-0.0, 0.0)),
        (Some(4), vec![-0.0, 0.0], (-0.0, 0.0)),
        (Some(5), vec![f64::INFINITY, nan], (f64::INFINITY, nan)),
        // The values DataFusion's starting values would replace.
        (
            Some(6),
            vec![f64::NEG_INFINITY],
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
        ),
        (Some(7), vec![f64::INFINITY], (f64::INFINITY, f64::INFINITY)),
        // A NaN with the sign bit set — what 0.0 / 0.0 produces on x86 — is the
        // least value of the order, not the greatest.
        (Some(8), vec![1.0, -nan], (-nan, 1.0)),
        (None, vec![nan, 2.0], (2.0, nan)),
    ]
}

/// The groups as two batches, the first value of every group in one and the
/// rest in the other; `reversed` swaps the batches. A result that depends on
/// which batch comes first is the bug this section exists for.
fn special_float_batches(reversed: bool) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Float64, false),
    ]));
    let (mut first, mut rest): (Vec<FloatRow>, Vec<FloatRow>) = (Vec::new(), Vec::new());
    for (k, values, _) in special_float_groups() {
        for (i, v) in values.into_iter().enumerate() {
            if i == 0 { &mut first } else { &mut rest }.push((k, v));
        }
    }
    let batch = |rows: Vec<FloatRow>| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(
                    rows.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    };
    let (a, b) = (batch(first), batch(rest));
    if reversed { vec![b, a] } else { vec![a, b] }
}

/// MIN/MAX(Float64) per group, compared bit for bit, so `-0.0` is not `0.0`
/// and a NaN's sign is checked.
async fn check_float_min_max(backend: Arc<dyn GpuBackend>) {
    for reversed in [false, true] {
        let ctx = SessionContext::new();
        let batches = special_float_batches(reversed);
        let scan = ctx
            .read_batches(batches)
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![(AggregateFunction::Min, 1), (AggregateFunction::Max, 1)],
        };
        let exec = GpuAggregateExec::try_new(scan, spec, backend.kind())
            .unwrap()
            .with_backend(Arc::clone(&backend));
        let out = collect(Arc::new(exec), ctx.task_ctx()).await.unwrap();
        let mut actual = std::collections::BTreeMap::new();
        for b in &out {
            let keys = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let col = |i: usize| b.column(i).as_any().downcast_ref::<Float64Array>().unwrap();
            let (mins, maxs) = (col(1), col(2));
            for row in 0..b.num_rows() {
                let key = keys.is_valid(row).then(|| keys.value(row));
                let pair = (mins.value(row).to_bits(), maxs.value(row).to_bits());
                assert!(actual.insert(key, pair).is_none(), "group {key:?} twice");
            }
        }
        let expected: std::collections::BTreeMap<_, _> = special_float_groups()
            .into_iter()
            .map(|(k, _, (min, max))| (k, (min.to_bits(), max.to_bits())))
            .collect();
        let show = |m: &std::collections::BTreeMap<Option<i64>, (u64, u64)>| {
            m.iter()
                .map(|(k, (lo, hi))| {
                    format!("{k:?}: {}/{}", f64::from_bits(*lo), f64::from_bits(*hi))
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            actual,
            expected,
            "MIN/MAX(Float64), reversed={reversed}\n got: {:?}\nwant: {:?}",
            show(&actual),
            show(&expected)
        );
    }
}

#[tokio::test]
async fn cpu_float_min_max_follow_total_order() {
    check_float_min_max(cpu()).await;
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
        check_join_chunked(Arc::clone(&backend), seed).await;
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

/// NULL elements inside vectors through the device transfer format (#105).
///
/// The format carries one validity bit per row, so a NULL element inside a
/// valid row used to come back from a CUDA or Metal filter, or a CUDA join,
/// as a valid `0.0`. The CPU backend never moves bytes, which is why the
/// suite above could not see it: `TransferSim` can.
mod vector_null_elements {
    use arrow::array::AsArray;
    use arrow::datatypes::Float32Type;
    use oxidelake_compute::JoinBuild;
    use oxidelake_core::{DeviceId, EngineError};
    use oxidelake_device::cpu::kernels;
    use oxidelake_device::{
        AggregateArgs, DeviceBatch, DeviceBuffer, DeviceJoinMatches, DeviceJoinTable, DeviceStream,
        FilterProjectArgs, HashJoinArgs, HostBuffer, JoinBuildArgs, JoinProbeArgs, MemoryInfo,
        VectorDistanceArgs, columns,
    };

    use super::*;

    /// A backend that claims to be CUDA and moves every batch through the
    /// device byte format — `columns::extract_batch` on upload,
    /// `columns::rebuild_batch` on download, what `CudaBackend` and
    /// `MetalBackend` do around their copies — while the kernels run on the
    /// CPU reference. On a machine with no GPU it is the closest thing to a
    /// device: whatever the format cannot carry comes back as a wrong row.
    #[derive(Debug)]
    struct TransferSim(CpuBackend);

    impl TransferSim {
        fn shared() -> Arc<dyn GpuBackend> {
            Arc::new(Self(CpuBackend::new()))
        }

        fn through_the_format(batch: &RecordBatch) -> Result<RecordBatch, EngineError> {
            columns::rebuild_batch(&batch.schema(), columns::extract_batch(batch)?)
        }
    }

    impl GpuBackend for TransferSim {
        fn kind(&self) -> BackendKind {
            BackendKind::Cuda
        }
        fn device_id(&self) -> DeviceId {
            DeviceId::new(BackendKind::Cuda, 0)
        }
        fn memory_info(&self) -> Result<MemoryInfo, EngineError> {
            self.0.memory_info()
        }
        fn alloc_device(&self, bytes: usize) -> Result<DeviceBuffer, EngineError> {
            self.0.alloc_device(bytes)
        }
        fn alloc_pinned_host(&self, bytes: usize) -> Result<HostBuffer, EngineError> {
            self.0.alloc_pinned_host(bytes)
        }
        fn create_stream(&self) -> Result<DeviceStream, EngineError> {
            self.0.create_stream()
        }
        fn copy_h2d(
            &self,
            stream: &DeviceStream,
            src: &HostBuffer,
            dst: &mut DeviceBuffer,
        ) -> Result<(), EngineError> {
            self.0.copy_h2d(stream, src, dst)
        }
        fn copy_d2h(
            &self,
            stream: &DeviceStream,
            src: &DeviceBuffer,
            dst: &mut HostBuffer,
        ) -> Result<(), EngineError> {
            self.0.copy_d2h(stream, src, dst)
        }
        fn synchronize(&self, stream: &DeviceStream) -> Result<(), EngineError> {
            self.0.synchronize(stream)
        }
        fn upload(
            &self,
            stream: &DeviceStream,
            batch: &RecordBatch,
        ) -> Result<DeviceBatch, EngineError> {
            self.0.upload(stream, &Self::through_the_format(batch)?)
        }
        fn download(
            &self,
            stream: &DeviceStream,
            batch: &DeviceBatch,
        ) -> Result<RecordBatch, EngineError> {
            Self::through_the_format(&self.0.download(stream, batch)?)
        }
        fn filter_project(
            &self,
            stream: &DeviceStream,
            args: FilterProjectArgs<'_>,
        ) -> Result<DeviceBatch, EngineError> {
            self.0.filter_project(stream, args)
        }
        fn hash_join(
            &self,
            stream: &DeviceStream,
            args: HashJoinArgs<'_>,
        ) -> Result<DeviceBatch, EngineError> {
            self.0.hash_join(stream, args)
        }
        fn prepare_join_build(
            &self,
            stream: &DeviceStream,
            args: JoinBuildArgs<'_>,
        ) -> Result<DeviceJoinTable, EngineError> {
            self.0.prepare_join_build(stream, args)
        }
        fn probe_join(
            &self,
            stream: &DeviceStream,
            args: JoinProbeArgs<'_>,
        ) -> Result<DeviceJoinMatches, EngineError> {
            self.0.probe_join(stream, args)
        }
        fn aggregate(
            &self,
            stream: &DeviceStream,
            args: AggregateArgs<'_>,
        ) -> Result<DeviceBatch, EngineError> {
            self.0.aggregate(stream, args)
        }
        fn vector_distance(
            &self,
            stream: &DeviceStream,
            args: VectorDistanceArgs<'_>,
        ) -> Result<DeviceBatch, EngineError> {
            self.0.vector_distance(stream, args)
        }
    }

    /// `id Int64`, `k Int64?`, `vec FixedSizeList<Float32, DIM>?` with a
    /// nullable child. Every NULL row holds NULL elements, the way a
    /// builder's `append_null` leaves it; with `holes`, about one valid row in
    /// five holds a NULL element too — the shape the format cannot carry.
    fn gen_null_elements(seed: u64, rows: usize, holes: bool) -> RecordBatch {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x0e1e_3e17);
        let k: Vec<Option<i64>> = (0..rows)
            .map(|_| (!rng.random_bool(0.1)).then(|| rng.random_range(0..10i64)))
            .collect();
        let valid: Vec<bool> = (0..rows).map(|_| !rng.random_bool(0.15)).collect();
        let mut elements: Vec<Option<f32>> = Vec::with_capacity(rows * DIM);
        for &row_valid in &valid {
            let hole =
                (holes && row_valid && rng.random_bool(0.2)).then(|| rng.random_range(0..DIM));
            for e in 0..DIM {
                elements
                    .push((row_valid && hole != Some(e)).then(|| rng.random_range(-5.0f32..5.0)));
            }
        }
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            DIM as i32,
            Arc::new(Float32Array::from(elements)),
            Some(NullBuffer::from(valid)),
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("k", DataType::Int64, true),
            Field::new("vec", vectors.data_type().clone(), true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(0..rows as i64)),
                Arc::new(Int64Array::from(k)),
                Arc::new(vectors),
            ],
        )
        .unwrap()
    }

    async fn check_filter(backend: Arc<dyn GpuBackend>, seed: u64, rows: usize) {
        let ctx = SessionContext::new();
        let batch = gen_null_elements(seed, rows, true);
        ctx.register_batch("t", batch.clone()).unwrap();
        let expected = sql_rows(&ctx, "SELECT id, vec FROM t WHERE k >= 3").await;
        let exec = GpuFilterExec::try_new(
            physical_plan(&ctx, batch).await,
            Predicate::compare(1, Comparison::GtEq, Literal::Int64(3)),
            vec![0, 2],
            backend.kind(),
        )
        .unwrap()
        .with_backend(backend);
        let actual = exec_rows(&ctx, Arc::new(exec)).await;
        assert_eq!(actual, expected, "filter seed={seed} rows={rows}");
    }

    /// The vector column on the probe side, then on the build side.
    async fn check_join(backend: Arc<dyn GpuBackend>, seed: u64, rows: usize) {
        let ctx = SessionContext::new();
        let vectors = gen_null_elements(seed, rows, true);
        let names = gen_right(seed, rows / 2 + 1);
        ctx.register_batch("v", vectors.clone()).unwrap();
        ctx.register_batch("r", names.clone()).unwrap();
        let expected = sql_rows(
            &ctx,
            "SELECT v.id, v.k, v.vec, r.rk, r.name FROM v INNER JOIN r ON v.k = r.rk",
        )
        .await;
        let exec = GpuHashJoinExec::try_new(
            physical_plan(&ctx, vectors.clone()).await,
            physical_plan(&ctx, names.clone()).await,
            1,
            0,
            backend.kind(),
        )
        .unwrap()
        .with_backend(Arc::clone(&backend));
        let actual = exec_rows(&ctx, Arc::new(exec)).await;
        assert_eq!(actual, expected, "probe-side join seed={seed} rows={rows}");

        let expected = sql_rows(
            &ctx,
            "SELECT r.rk, r.name, v.id, v.k, v.vec FROM r INNER JOIN v ON r.rk = v.k",
        )
        .await;
        let exec = GpuHashJoinExec::try_new(
            physical_plan(&ctx, names).await,
            physical_plan(&ctx, vectors).await,
            0,
            1,
            backend.kind(),
        )
        .unwrap()
        .with_backend(backend);
        let actual = exec_rows(&ctx, Arc::new(exec)).await;
        assert_eq!(actual, expected, "build-side join seed={seed} rows={rows}");
    }

    /// Runs `run` on a `GpuOperator` over `backend` and returns what it
    /// produced with the operator's `(fallback_batches, bytes_h2d)`: the
    /// witness of which path it took.
    fn witnessed<T>(
        backend: Arc<dyn GpuBackend>,
        run: impl FnOnce(&GpuOperator) -> T,
    ) -> (T, u64, u64) {
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuOperator", backend.kind());
        let operator = GpuOperator::new(backend, Some(Arc::clone(&stats))).unwrap();
        let out = run(&operator);
        let snapshot = stats.snapshot();
        (out, snapshot.fallback_batches, snapshot.bytes_h2d)
    }

    /// One filter batch through a `GpuOperator` on `backend`, checked against
    /// the CPU reference; returns its `(fallback_batches, bytes_h2d)`.
    pub(super) fn filter_once(
        backend: Arc<dyn GpuBackend>,
        batch: &RecordBatch,
        predicate: &Predicate,
        projection: &[usize],
    ) -> (u64, u64) {
        let (out, fallbacks, h2d) = witnessed(backend, |op| {
            op.filter_project(batch, predicate, projection).unwrap()
        });
        assert_eq!(
            out,
            kernels::filter_project(batch, predicate, projection).unwrap(),
            "device path and CPU reference disagree"
        );
        (fallbacks, h2d)
    }

    /// One probe batch joined against `build`, checked against the CPU
    /// reference; returns its `(fallback_batches, bytes_h2d)`. The output
    /// comes out in pieces of at most `CHUNK` rows (#95), so a column kept on
    /// the host is gathered at each piece's row ids, not once.
    fn join_once(
        backend: Arc<dyn GpuBackend>,
        probe: &RecordBatch,
        probe_key: usize,
        build: &RecordBatch,
        build_key: usize,
    ) -> (u64, u64) {
        const CHUNK: usize = 61;
        let build = JoinBuild::new(build.clone(), build_key).unwrap();
        let (pieces, fallbacks, h2d) = witnessed(backend, |op| {
            let mut probed = op.hash_join(probe, &build, probe_key, None, CHUNK).unwrap();
            let mut pieces = Vec::new();
            while let Some(piece) = op.next_join_chunk(&mut probed, &build).unwrap() {
                pieces.push(piece);
            }
            pieces
        });
        assert!(pieces.len() > 1, "{} piece(s)", pieces.len());
        let reference =
            kernels::hash_join_probe(probe, probe_key, build.batch(), build.table(), None).unwrap();
        assert_eq!(
            sorted_rows(&pieces),
            sorted_rows(&[reference]),
            "device path and CPU reference disagree"
        );
        (fallbacks, h2d)
    }

    /// Equal rows prove nothing if every batch quietly took the CPU path. A
    /// filter or a join runs on the device whether or not a valid row holds a
    /// NULL element: such a column is gathered on the host, not uploaded —
    /// `extract`, and so `TransferSim`, would refuse it and count a fallback.
    /// A vector distance has to read the column on the device, so over such
    /// a batch it is the one that runs on the CPU reference, counted.
    fn assert_device_path_witnessed(backend: &Arc<dyn GpuBackend>) {
        let predicate = Predicate::compare(1, Comparison::GtEq, Literal::Int64(3));
        let names = gen_right(5, 40);
        let query = [0.5f32, -1.0, 2.0, 0.25];
        for holes in [false, true] {
            let batch = gen_null_elements(5, 300, holes);
            let vectors = batch.column(2);
            assert!(vectors.null_count() > 0, "the fixture has NULL rows");
            assert_eq!(
                columns::row_with_null_element(vectors.as_ref())
                    .unwrap()
                    .is_some(),
                holes,
                "the fixture's valid rows hold NULL elements"
            );
            let (fallbacks, h2d) = filter_once(Arc::clone(backend), &batch, &predicate, &[0, 2]);
            assert_eq!(fallbacks, 0, "filter fell back (holes: {holes})");
            assert!(h2d > 0, "filter uploaded nothing (holes: {holes})");
            if backend.supports_hash_join() {
                let probe = join_once(Arc::clone(backend), &batch, 1, &names, 0);
                let build = join_once(Arc::clone(backend), &names, 0, &batch, 1);
                for (side, (fallbacks, h2d)) in [("probe", probe), ("build", build)] {
                    assert_eq!(fallbacks, 0, "{side}-side join fell back (holes: {holes})");
                    assert!(h2d > 0, "{side}-side join uploaded nothing");
                }
            }
            let (out, fallbacks, _) = witnessed(Arc::clone(backend), |op| {
                op.vector_distance(&batch, 2, &query, DistanceMetric::L2, "d")
                    .unwrap()
            });
            let expected = u64::from(holes);
            assert_eq!(fallbacks, expected, "vector distance (holes: {holes})");
            let reference =
                kernels::vector_distance(&batch, 2, &query, DistanceMetric::L2, "d").unwrap();
            let got = out.column(3).as_primitive::<Float32Type>();
            let want = reference.column(3).as_primitive::<Float32Type>();
            assert_eq!(got.nulls(), want.nulls(), "vector distance NULLs");
            for (row, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                if let (Some(g), Some(w)) = (g, w) {
                    assert!((g - w).abs() < 1e-4, "row {row}: {g} vs {w}");
                }
            }
        }
    }

    #[tokio::test]
    async fn null_vector_elements_survive_the_device_format() {
        let backend = TransferSim::shared();
        for &seed in &SEEDS {
            for &rows in &SIZES {
                check_filter(Arc::clone(&backend), seed, rows).await;
                check_join(Arc::clone(&backend), seed, rows).await;
            }
        }
        assert_device_path_witnessed(&backend);
    }

    /// The same on a real device. Run on a GPU box with
    /// `OXIDE_BACKEND=cuda cargo test -p oxidelake-compute --features cuda -- --ignored`.
    #[tokio::test]
    #[ignore = "requires a GPU backend (CUDA or Metal) on this machine"]
    async fn gpu_backend_keeps_null_vector_elements_when_available() {
        let backend = HardwareDetector::select().unwrap();
        assert!(
            backend.kind().is_gpu(),
            "no GPU backend detected: {:?}",
            backend.kind()
        );
        for &seed in &SEEDS {
            for &rows in &SIZES {
                check_filter(Arc::clone(&backend), seed, rows).await;
                check_join(Arc::clone(&backend), seed, rows).await;
            }
        }
        assert_device_path_witnessed(&backend);
    }
}
