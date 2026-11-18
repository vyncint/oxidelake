//! `Gpu*Exec::operator_stats` (#123): an exec instance reads back its own hub
//! entry. Two operators that share a name are two entries, and a reader that
//! looked them up by name — the dashboard did — added them together.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::SessionContext;
use oxidelake_compute::{Comparison, GpuFilterExec, Literal, Predicate};
use oxidelake_core::BackendKind;
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_device::CpuBackend;

/// `k` = 0, 1, …, 99 in one batch.
async fn scan(ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from_iter_values(0..100))]).unwrap();
    ctx.read_batch(batch)
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap()
}

fn filter(input: &Arc<dyn ExecutionPlan>, op: Comparison, value: i64) -> GpuFilterExec {
    GpuFilterExec::try_new(
        Arc::clone(input),
        Predicate::compare(0, op, Literal::Int64(value)),
        vec![0],
        BackendKind::Cuda,
    )
    .unwrap()
    .with_backend(Arc::new(CpuBackend::new()))
}

#[tokio::test]
async fn each_instance_reads_its_own_counters() {
    let ctx = SessionContext::new();
    let input = scan(&ctx).await;
    let hub = TelemetryHub::new();
    let low = Arc::new(filter(&input, Comparison::Lt, 4).with_telemetry(Arc::clone(&hub)));
    let high = Arc::new(filter(&input, Comparison::GtEq, 50).with_telemetry(Arc::clone(&hub)));
    assert_eq!(low.operator_stats(), None, "nothing has executed yet");

    for exec in [&low, &high] {
        let plan: Arc<dyn ExecutionPlan> = Arc::clone(exec) as _;
        collect(plan, ctx.task_ctx()).await.unwrap();
    }
    let (low, high) = (
        low.operator_stats().unwrap(),
        high.operator_stats().unwrap(),
    );
    assert_eq!(low.name, "GpuFilterExec");
    assert_eq!(high.name, "GpuFilterExec");
    assert_ne!(low.id, high.id, "one hub entry per instance");
    assert_eq!((low.rows_in, low.rows_out), (100, 4), "k < 4: {low:?}");
    assert_eq!(
        (high.rows_in, high.rows_out),
        (100, 50),
        "k >= 50: {high:?}"
    );
    assert_eq!(hub.snapshot().operators.len(), 2);
}

#[tokio::test]
async fn an_exec_without_a_hub_has_no_counters_to_read() {
    let ctx = SessionContext::new();
    let input = scan(&ctx).await;
    let exec: Arc<dyn ExecutionPlan> = Arc::new(filter(&input, Comparison::Lt, 4));
    collect(Arc::clone(&exec), ctx.task_ctx()).await.unwrap();
    let exec = exec.downcast_ref::<GpuFilterExec>().unwrap();
    assert_eq!(exec.operator_stats(), None);
}
