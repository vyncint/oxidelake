//! Timings for the device hash join (#45, #108). Needs a GPU.
//!
//! The CUDA join used to hash the build side into one slot per build *row*,
//! again for every probe batch: a key repeated D times cost O(D²) `atomicCAS`
//! attempts per batch, and a probe row whose hash landed in its run walked
//! O(D) slots. Each case below is a shape that cost exposed, checked against
//! the CPU reference (rows and a checksum a lost or doubled match changes)
//! and timed on the device.
//!
//! The file uses only APIs older releases already have, so the same file
//! measures both sides of the change on one machine:
//!
//! ```text
//! OXIDE_BACKEND=cuda cargo test --release -p oxidelake-compute --features cuda \
//!     --test join_timing -- --ignored --nocapture
//! ```
//!
//! then copy it onto the commit before the change (08a9147) and run the same
//! command there.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{SessionConfig, SessionContext};
use oxidelake_compute::GpuHashJoinExec;
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_device::{CpuBackend, GpuBackend, HardwareDetector};

/// The batch size of a GPU-targeted session (`oxidelake_storage::GPU_BATCH_SIZE`).
const BATCH: usize = 65_536;

/// `k Int64` from `keys`, `x Int64` = the row number.
fn side(keys: Vec<i64>) -> RecordBatch {
    let n = i64::try_from(keys.len()).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("x", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Int64Array::from_iter_values(0..n)),
        ],
    )
    .unwrap()
}

/// One partition of `BATCH`-row batches, as a scan hands them over.
fn source(batch: &RecordBatch) -> Arc<dyn ExecutionPlan> {
    let schema: SchemaRef = batch.schema();
    let parts: Vec<RecordBatch> = (0..batch.num_rows())
        .step_by(BATCH)
        .map(|offset| batch.slice(offset, BATCH.min(batch.num_rows() - offset)))
        .collect();
    MemorySourceConfig::try_new_exec(&[parts], schema, None).unwrap()
}

fn column(batch: &RecordBatch, i: usize) -> &Int64Array {
    batch
        .column(i)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
}

/// Rows of `probe ⋈ build` on `k`, and the sum of probe x + build x over
/// them, by counting rather than joining.
fn expected(probe: &RecordBatch, build: &RecordBatch) -> (u64, i128) {
    let mut per_key: HashMap<i64, (u64, i128)> = HashMap::new();
    for (k, x) in column(build, 0)
        .values()
        .iter()
        .zip(column(build, 1).values())
    {
        let e = per_key.entry(*k).or_default();
        e.0 += 1;
        e.1 += i128::from(*x);
    }
    let (mut rows, mut sum) = (0u64, 0i128);
    for (k, x) in column(probe, 0)
        .values()
        .iter()
        .zip(column(probe, 1).values())
    {
        if let Some(&(count, xs)) = per_key.get(k) {
            rows += count;
            sum += i128::from(*x) * i128::from(count) + xs;
        }
    }
    (rows, sum)
}

/// Runs the join on `backend`: rows, checksum, milliseconds, CPU-fallback batches.
async fn run(
    backend: Arc<dyn GpuBackend>,
    probe: &RecordBatch,
    build: &RecordBatch,
    ctx: Arc<TaskContext>,
) -> (u64, i128, f64, u64) {
    let hub = TelemetryHub::new();
    let join = GpuHashJoinExec::try_new(source(probe), source(build), 0, 0, backend.kind())
        .unwrap()
        // probe x, build x
        .with_projection(Some(vec![1, 3]))
        .unwrap()
        .with_backend(backend)
        .with_telemetry(Arc::clone(&hub));
    let started = Instant::now();
    let out = collect(Arc::new(join), ctx).await.unwrap();
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let (mut rows, mut sum) = (0u64, 0i128);
    for batch in &out {
        rows += batch.num_rows() as u64;
        for (a, b) in column(batch, 0)
            .values()
            .iter()
            .zip(column(batch, 1).values())
        {
            sum += i128::from(*a) + i128::from(*b);
        }
    }
    let fallbacks = hub
        .snapshot()
        .operators
        .iter()
        .map(|o| o.fallback_batches)
        .sum();
    (rows, sum, ms, fallbacks)
}

#[tokio::test]
#[ignore = "requires a GPU backend; prints timings for #45/#108"]
async fn device_hash_join_timings() {
    let gpu = HardwareDetector::select().unwrap();
    assert!(
        gpu.kind().is_gpu(),
        "no GPU backend detected: {:?}",
        gpu.kind()
    );
    // Metal declines the join up front (it has no join kernel), so on a Mac
    // every batch would run on the CPU and there is no device time to take.
    // The on-device suite runs every ignored test there; say so and stop.
    if !gpu.supports_hash_join() {
        println!(
            "skipped: the {:?} backend has no device hash join, so there is nothing to time",
            gpu.kind()
        );
        return;
    }
    let cpu: Arc<dyn GpuBackend> = Arc::new(CpuBackend::new());
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_batch_size(BATCH)).task_ctx();

    let n = 100_000i64;
    let probe_rows = 65_536i64;
    let cases: Vec<(&str, Vec<i64>, Vec<i64>)> = vec![
        (
            "conformance-sized, 10 keys",
            (0..1_000).map(|i| i % 10).collect(),
            (0..501).map(|j| j % 10).collect(),
        ),
        (
            "unique build keys, every probe hits",
            (0..probe_rows).collect(),
            (0..n).collect(),
        ),
        (
            "half the build on key 0, probes absent",
            (0..probe_rows).map(|i| i + 1_000_000).collect(),
            (0..n).map(|j| if j % 2 == 0 { 0 } else { j }).collect(),
        ),
        (
            "one build key, probes absent",
            (0..probe_rows).map(|i| i + 1_000_000).collect(),
            vec![0; 100_000],
        ),
        (
            "one build key, 16 probes hit (1.6M rows)",
            (0..probe_rows)
                .map(|i| if i % 4_096 == 0 { 0 } else { i + 1_000_000 })
                .collect(),
            vec![0; 100_000],
        ),
    ];
    println!(
        "{:<42} {:>7} {:>7} {:>9} {:>11} {:>11}",
        "case", "probe", "build", "matches", "cpu ms", "device ms"
    );
    for (name, probe_keys, build_keys) in cases {
        let (probe, build) = (side(probe_keys), side(build_keys));
        let (rows, sum) = expected(&probe, &build);
        let (cpu_rows, cpu_sum, cpu_ms, _) =
            run(Arc::clone(&cpu), &probe, &build, Arc::clone(&ctx)).await;
        assert_eq!((cpu_rows, cpu_sum), (rows, sum), "{name}: CPU reference");
        let (dev_rows, dev_sum, dev_ms, fallbacks) =
            run(Arc::clone(&gpu), &probe, &build, Arc::clone(&ctx)).await;
        assert_eq!((dev_rows, dev_sum), (rows, sum), "{name}: device");
        assert_eq!(
            fallbacks, 0,
            "{name}: a batch fell back to the CPU, so the time is not the device's"
        );
        println!(
            "{name:<42} {:>7} {:>7} {rows:>9} {cpu_ms:>11.1} {dev_ms:>11.1}",
            probe.num_rows(),
            build.num_rows()
        );
    }
}
