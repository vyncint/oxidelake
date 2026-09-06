//! Placement-rule tests with a mocked "GPU present" target: eligible nodes are
//! rewritten, ineligible ones are untouched, `EXPLAIN` shows the tags, and
//! results equal a plain DataFusion session (execution falls back to the CPU
//! reference on this GPU-less machine). Also round-trips every node through
//! `OxidePhysicalCodec`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::SessionContext;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use oxidelake_core::BackendKind;
use oxidelake_planner::{HardwarePlacementRule, OxidePhysicalCodec, physical_optimizer_rules};

fn batches_t() -> Vec<Vec<RecordBatch>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
        Field::new("s", DataType::Utf8, false),
    ]));
    let mk = |ks: Vec<Option<i64>>, vs: Vec<Option<f64>>| {
        let n = ks.len();
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ks)),
                Arc::new(Float64Array::from(vs)),
                Arc::new(StringArray::from(
                    (0..n).map(|i| format!("s{i}")).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    };
    vec![
        vec![mk(
            vec![Some(1), Some(2), None, Some(2)],
            vec![Some(0.5), Some(1.5), Some(2.5), None],
        )],
        vec![mk(
            vec![Some(5), Some(2), Some(7)],
            vec![Some(3.5), Some(4.5), Some(-1.0)],
        )],
    ]
}

fn batches_r() -> Vec<Vec<RecordBatch>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("rk", DataType::Int64, true),
        Field::new("name", DataType::Utf8, false),
    ]));
    vec![vec![
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(2), Some(2), None, Some(7)])),
                Arc::new(StringArray::from(vec!["two-a", "two-b", "null", "seven"])),
            ],
        )
        .unwrap(),
    ]]
}

/// `id Int64`, `emb FixedSizeList<Float32, 2>?`.
fn batches_e() -> Vec<Vec<RecordBatch>> {
    let emb = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        2,
        Arc::new(Float32Array::from(vec![
            1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 2.0,
        ])),
        Some(vec![true, true, false, true].into()),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("emb", emb.data_type().clone(), true),
    ]));
    vec![vec![
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![0, 1, 2, 3])), Arc::new(emb)],
        )
        .unwrap(),
    ]]
}

fn register(ctx: &SessionContext) {
    for udf in oxidelake_compute::oxide_udfs() {
        ctx.register_udf(udf.as_ref().clone());
    }
    let e = batches_e();
    ctx.register_table(
        "e",
        Arc::new(MemTable::try_new(e[0][0].schema(), e).unwrap()),
    )
    .unwrap();
    let t = batches_t();
    let r = batches_r();
    ctx.register_table(
        "t",
        Arc::new(MemTable::try_new(t[0][0].schema(), t).unwrap()),
    )
    .unwrap();
    ctx.register_table(
        "r",
        Arc::new(MemTable::try_new(r[0][0].schema(), r).unwrap()),
    )
    .unwrap();
}

fn ctx_with_rule(target: BackendKind) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_physical_optimizer_rules(physical_optimizer_rules(HardwarePlacementRule::new(target)))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register(&ctx);
    ctx
}

fn plain_ctx() -> SessionContext {
    let ctx = SessionContext::new();
    register(&ctx);
    ctx
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

async fn plan_and_text(ctx: &SessionContext, sql: &str) -> (Arc<dyn ExecutionPlan>, String) {
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let text = displayable(plan.as_ref()).indent(true).to_string();
    (plan, text)
}

async fn assert_same_results(
    sql: &str,
    target: BackendKind,
    expect_tag: Option<&str>,
) -> Arc<dyn ExecutionPlan> {
    let gpu_ctx = ctx_with_rule(target);
    let (plan, text) = plan_and_text(&gpu_ctx, sql).await;
    match expect_tag {
        Some(tag) => assert!(text.contains(tag), "expected {tag} in plan:\n{text}"),
        None => assert!(
            !text.contains("Gpu"),
            "expected no Gpu nodes in plan:\n{text}"
        ),
    }
    let actual = sorted_rows(
        &collect(Arc::clone(&plan), gpu_ctx.task_ctx())
            .await
            .unwrap(),
    );
    let expected = sorted_rows(&plain_ctx().sql(sql).await.unwrap().collect().await.unwrap());
    assert_eq!(actual, expected, "{sql}");
    plan
}

#[tokio::test]
async fn filter_is_rewritten_and_fused_with_projection() {
    let plan = assert_same_results(
        "SELECT s, k FROM t WHERE k >= 2 AND v < 4.0",
        BackendKind::Cuda,
        Some("GpuFilterExec[cuda]: k >= 2 AND v < 4.0, projection=[s, k]"),
    )
    .await;
    let text = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        !text.contains("FilterExec:"),
        "stock FilterExec should be gone:\n{text}"
    );
    // literal on the left, flipped comparison
    assert_same_results(
        "SELECT k FROM t WHERE 2 < k",
        BackendKind::Metal,
        Some("GpuFilterExec[metal]: k > 2"),
    )
    .await;
}

#[tokio::test]
async fn aggregate_is_rewritten_with_planned_names() {
    let plan = assert_same_results(
        "SELECT k, sum(v), count(v), min(v), max(k) FROM t GROUP BY k",
        BackendKind::Cuda,
        Some("GpuAggregateExec[cuda]: group_by=[k], aggr=[SUM(v), COUNT(v), MIN(v), MAX(k)]"),
    )
    .await;
    let schema = plan.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        vec!["k", "sum(t.v)", "count(t.v)", "min(t.v)", "max(t.k)"]
    );
    let text = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        !text.contains("AggregateExec: mode"),
        "stock AggregateExec should be gone:\n{text}"
    );
}

#[tokio::test]
async fn join_is_rewritten() {
    assert_same_results(
        "SELECT t.k, t.s, r.name FROM t JOIN r ON t.k = r.rk",
        BackendKind::Cuda,
        Some("GpuHashJoinExec[cuda]: join_type=Inner, on=["),
    )
    .await;
}

#[tokio::test]
async fn ineligible_nodes_and_cpu_target_are_untouched() {
    assert_same_results("SELECT k FROM t WHERE s = 's1'", BackendKind::Cuda, None).await;
    assert_same_results(
        "SELECT r.name FROM t JOIN r ON t.s = r.name",
        BackendKind::Cuda,
        None,
    )
    .await;
    assert_same_results(
        "SELECT s, count(*) FROM t GROUP BY s",
        BackendKind::Cuda,
        None,
    )
    .await;
    assert_same_results(
        "SELECT k, sum(v) FROM t GROUP BY k",
        BackendKind::CpuSimd,
        None,
    )
    .await;
    assert_same_results(
        "SELECT k FROM t WHERE k >= 2 OR v < 1.0",
        BackendKind::Cuda,
        None,
    )
    .await;
}

#[tokio::test]
async fn codec_round_trips_every_node() {
    let ctx = ctx_with_rule(BackendKind::Cuda);
    let codec = OxidePhysicalCodec::default();
    for sql in [
        "SELECT s, k FROM t WHERE k >= 2 AND v < 4.0",
        "SELECT k, sum(v), count(v) FROM t GROUP BY k",
        "SELECT t.k, r.name FROM t JOIN r ON t.k = r.rk",
        "SELECT id, emb, l2_distance(emb, [1.0, 0.0]) AS d FROM e",
    ] {
        let (plan, text) = plan_and_text(&ctx, sql).await;
        let mut seen = 0;
        // Walk the tree and round-trip each Gpu node against its own children.
        let mut stack = vec![plan];
        while let Some(node) = stack.pop() {
            if OxidePhysicalCodec::describe(node.as_ref()).is_some() {
                let mut buf = Vec::new();
                codec.try_encode(Arc::clone(&node), &mut buf).unwrap();
                assert!(buf.starts_with(b"OXGP"));
                let children: Vec<Arc<dyn ExecutionPlan>> =
                    node.children().into_iter().cloned().collect();
                let back = codec.try_decode(&buf, &children, &ctx.task_ctx()).unwrap();
                assert_eq!(
                    displayable(back.as_ref()).one_line().to_string(),
                    displayable(node.as_ref()).one_line().to_string()
                );
                assert_eq!(back.schema(), node.schema());
                seen += 1;
            }
            stack.extend(node.children().into_iter().cloned());
        }
        assert!(seen >= 1, "no Gpu node in plan for {sql}:\n{text}");
    }
    // Non-Oxide payloads and corrupt magic payloads are handled.
    assert!(
        codec
            .try_decode(b"OXGP\x09garbage", &[], &ctx.task_ctx())
            .is_err()
    );
}

#[tokio::test]
async fn vector_distance_projection_is_rewritten() {
    let plan = assert_same_results(
        "SELECT id, emb, l2_distance(emb, [1.0, 0.0]) AS d FROM e",
        BackendKind::Cuda,
        Some("GpuVectorDistanceExec[cuda]: l2(emb) AS d, dim=2"),
    )
    .await;
    let text = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        !text.contains("ProjectionExec"),
        "the projection should be gone:\n{text}"
    );
    // Reversed argument order (the metric is symmetric), cosine, Metal target.
    assert_same_results(
        "SELECT id, emb, cosine_distance([0.0, 1.0], emb) AS c FROM e",
        BackendKind::Metal,
        Some("GpuVectorDistanceExec[metal]: cosine(emb) AS c, dim=2"),
    )
    .await;
}

#[tokio::test]
async fn ineligible_distance_projections_stay_on_cpu() {
    // Dropping the passthrough columns, putting the call first, a non-literal
    // query, and a CPU target: all stay stock ProjectionExec (results equal).
    for (sql, target) in [
        (
            "SELECT l2_distance(emb, [1.0, 0.0]) AS d FROM e",
            BackendKind::Cuda,
        ),
        (
            "SELECT l2_distance(emb, [1.0, 0.0]) AS d, id FROM e",
            BackendKind::Cuda,
        ),
        (
            "SELECT id, emb, l2_distance(emb, emb) AS d FROM e",
            BackendKind::Cuda,
        ),
        (
            "SELECT id, emb, l2_distance(emb, [1.0, 0.0]) AS d FROM e",
            BackendKind::CpuSimd,
        ),
    ] {
        assert_same_results(sql, target, None).await;
    }
}
