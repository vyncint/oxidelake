//! Timings for the device aggregate (#29, #45). Needs a GPU.
//!
//! A `GROUP BY k` over 10 million rows with 100 keys, the case #45 names. The
//! aggregate used to take each partition whole in one device call, with a
//! group table of two slots per *row* that the host read back entire, so its
//! device→host traffic followed the input whatever the cardinality. It now
//! takes one batch per call (#29), reduces into one entry per *group*, and
//! reads back only those (#45). The result is checked against the CPU
//! reference and timed on the device.
//!
//! The file uses only APIs older releases already have, so the same file
//! measures both sides of the change on one machine:
//!
//! ```text
//! OXIDE_BACKEND=cuda cargo test --release -p oxidelake-compute --features cuda \
//!     --test aggregate_timing -- --ignored --nocapture
//! ```
//!
//! then copy it onto the commit before the change (08a9147) and run the same
//! command there.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{SessionConfig, SessionContext};
use oxidelake_compute::{AggregateFunction, AggregateSpec, GpuAggregateExec};
use oxidelake_core::telemetry::{OperatorSnapshot, TelemetryHub};
use oxidelake_device::{CpuBackend, GpuBackend, HardwareDetector};

const ROWS: usize = 10_000_000;
const KEYS: u64 = 100;
/// The batch size of a GPU-targeted session (`oxidelake_storage::GPU_BATCH_SIZE`).
const BATCH: usize = 65_536;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `k Int64?` (100 keys, 1 in 50 `NULL`), `x Int64`, `v Float64` (multiples
/// of 0.25, so a float sum is exact in any order), in `BATCH`-row batches.
fn table() -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("x", DataType::Int64, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let mut state = 45u64;
    let mut batches = Vec::with_capacity(ROWS.div_ceil(BATCH));
    let mut left = ROWS;
    while left > 0 {
        let rows = left.min(BATCH);
        left -= rows;
        let (mut k, mut x, mut v) = (
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
        );
        for _ in 0..rows {
            let r = splitmix64(&mut state);
            k.push((!r.is_multiple_of(50)).then_some(((r >> 8) % KEYS) as i64));
            x.push((r >> 16) as i64 % 1_000);
            v.push(f64::from(((r >> 32) % 400) as u32) / 4.0);
        }
        batches.push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(k)),
                    Arc::new(Int64Array::from(x)),
                    Arc::new(arrow::array::Float64Array::from(v)),
                ],
            )
            .unwrap(),
        );
    }
    batches
}

fn spec() -> AggregateSpec {
    AggregateSpec {
        group_by: 0,
        aggregates: vec![
            (AggregateFunction::Sum, 1),
            (AggregateFunction::Count, 1),
            (AggregateFunction::Min, 2),
            (AggregateFunction::Max, 2),
            (AggregateFunction::Sum, 2),
        ],
    }
}

/// Runs the aggregate on `backend` over one partition of `batches`: the
/// output, milliseconds, and the operator's counters.
async fn run(
    backend: Arc<dyn GpuBackend>,
    batches: &[RecordBatch],
) -> (Vec<RecordBatch>, f64, OperatorSnapshot) {
    // A GPU session's batch size: the memory source re-chunks to it.
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_batch_size(BATCH));
    let schema = batches[0].schema();
    let source: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_exec(&[batches.to_vec()], schema, None).unwrap();
    let hub = TelemetryHub::new();
    let exec = GpuAggregateExec::try_new(source, spec(), backend.kind())
        .unwrap()
        .with_backend(backend)
        .with_telemetry(Arc::clone(&hub));
    let start = Instant::now();
    let out = collect(Arc::new(exec), ctx.task_ctx()).await.unwrap();
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let snapshot = hub.snapshot().operators.into_iter().next().unwrap();
    (out, ms, snapshot)
}

#[tokio::test]
#[ignore = "needs a GPU: times the device aggregate"]
async fn ten_million_rows_one_hundred_keys() {
    let device = HardwareDetector::select().unwrap();
    assert!(
        device.kind().is_gpu(),
        "no GPU backend detected: {:?}",
        device.kind()
    );
    // Metal declines the aggregate up front (it has no aggregate kernel), so
    // on a Mac every batch would run on the CPU and there is no device time
    // to take. The on-device suite runs every ignored test there; say so and
    // stop, as the join timing probe does.
    if !device.supports_aggregate() {
        println!(
            "skipped: the {:?} backend has no device aggregate, so there is nothing to time",
            device.kind()
        );
        return;
    }
    let batches = table();
    let (reference, cpu_ms, _) = run(Arc::new(CpuBackend::new()), &batches).await;
    // Compiles the kernels (NVRTC, cached per backend) outside the timing.
    run(Arc::clone(&device), &batches[..1]).await;
    let mut best = f64::INFINITY;
    for round in 0..3 {
        let (out, ms, snapshot) = run(Arc::clone(&device), &batches).await;
        assert_eq!(out, reference, "round {round}");
        assert_eq!(snapshot.fallback_batches, 0, "{snapshot:?}");
        println!(
            "device round {round}: {ms:.1} ms, {} calls, {} rows in, {} groups out, \
             {} B up, {} B down (the down counter sees the result batches only)",
            snapshot.batches,
            snapshot.rows_in,
            snapshot.rows_out,
            snapshot.bytes_h2d,
            snapshot.bytes_d2h
        );
        best = best.min(ms);
    }
    println!(
        "{ROWS} rows, {KEYS} keys and NULL: CPU reference {cpu_ms:.1} ms, device best of 3 {best:.1} ms"
    );
}
