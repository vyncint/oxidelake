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
use datafusion::config::ConfigOptions;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_proto::physical_plan::{DefaultPhysicalProtoConverter, PhysicalExtensionCodec};
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

/// `0..100` in an order that is neither sorted nor reversed, so an operator
/// that only happens to preserve its input's order cannot pass for sorted.
fn scrambled(i: i64) -> i64 {
    (i * 37) % 100
}

/// `k Int64`, `v Float64`: 100 rows in one partition and one batch, which
/// DataFusion never splits, however many partitions the session targets.
fn batches_n() -> Vec<Vec<RecordBatch>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let k: Vec<i64> = (0..100).map(scrambled).collect();
    let v: Vec<f64> = k.iter().map(|&k| k as f64 / 2.0).collect();
    vec![vec![
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(k)),
                Arc::new(Float64Array::from(v)),
            ],
        )
        .unwrap(),
    ]]
}

/// `bk Int64`, `y Int64`: every key of `n` four times, one partition.
fn batches_d() -> Vec<Vec<RecordBatch>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bk", DataType::Int64, false),
        Field::new("y", DataType::Int64, false),
    ]));
    vec![vec![
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..400).map(|i| scrambled(i % 100)),
                )),
                Arc::new(Int64Array::from_iter_values(0..400)),
            ],
        )
        .unwrap(),
    ]]
}

/// `c0..c64 Int64`: 65 columns, one more than the device filter takes
/// comparisons (`MAX_PREDICATE_LEAVES`); row `i` holds `i + j` in `cj`.
fn batches_w() -> Vec<Vec<RecordBatch>> {
    let width: i64 = 65;
    let schema = Arc::new(Schema::new(
        (0..width)
            .map(|j| Field::new(format!("c{j}"), DataType::Int64, false))
            .collect::<Vec<_>>(),
    ));
    let columns: Vec<Arc<dyn Array>> = (0..width)
        .map(|j| Arc::new(Int64Array::from_iter_values((0..10).map(|i| i + j))) as _)
        .collect();
    vec![vec![RecordBatch::try_new(schema, columns).unwrap()]]
}

/// The rows of `batches` cut into `parts` partitions, in order.
fn partitioned(batches: Vec<Vec<RecordBatch>>, parts: usize) -> Vec<Vec<RecordBatch>> {
    let all =
        arrow::compute::concat_batches(&batches[0][0].schema(), batches.iter().flatten()).unwrap();
    let step = all.num_rows().div_ceil(parts);
    (0..parts)
        .map(|p| {
            let start = p * step;
            vec![all.slice(start, step.min(all.num_rows() - start))]
        })
        .collect()
}

fn register(ctx: &SessionContext) {
    for udf in oxidelake_compute::oxide_udfs() {
        ctx.register_udf(udf.as_ref().clone());
    }
    for (name, batches) in [
        ("n", batches_n()),
        ("d", batches_d()),
        ("w", batches_w()),
        // `n` and `d` split across partitions, so a join over them is
        // hash-repartitioned the way a cluster's is.
        ("np", partitioned(batches_n(), 3)),
        ("dp", partitioned(batches_d(), 2)),
    ] {
        ctx.register_table(
            name,
            Arc::new(MemTable::try_new(batches[0][0].schema(), batches).unwrap()),
        )
        .unwrap();
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
    ctx_with_rule_and(target, SessionConfig::new())
}

fn ctx_with_rule_and(target: BackendKind, config: SessionConfig) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_physical_optimizer_rules(physical_optimizer_rules(HardwarePlacementRule::new(target)))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register(&ctx);
    ctx
}

fn plain_ctx() -> SessionContext {
    plain_ctx_with(SessionConfig::new())
}

fn plain_ctx_with(config: SessionConfig) -> SessionContext {
    let ctx = SessionContext::new_with_config(config);
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

/// DataFusion's `LimitPushdown` hands a LIMIT to the `FilterExec` or
/// `HashJoinExec` under it as `fetch` and removes the limit node whenever that
/// node runs in one partition, so the node's `fetch` is the query's only
/// limit. The lowered plan keeps it (#86); it used to return every matching
/// row. Two ways a node ends up in one partition: `target_partitions = 1`, or
/// an input smaller than one batch, which DataFusion never splits (GPU
/// sessions use 65 536-row batches, so that is every small table).
#[tokio::test]
async fn a_limit_pushed_into_a_lowered_node_is_kept() {
    let filter = "SELECT k, v FROM n WHERE k > 3 LIMIT 5";
    let join = "SELECT n.k, d.y FROM n JOIN d ON n.k = d.bk LIMIT 3";
    // (target_partitions, query, lowered node, limit, whether the stock
    // plan's root is that node, i.e. its fetch is the only limit there is)
    for (target_partitions, sql, lowered, limit, only_limit) in [
        (1, filter, "GpuFilterExec", 5, true),
        (4, filter, "GpuFilterExec", 5, true),
        (1, join, "GpuHashJoinExec", 3, true),
        (4, join, "GpuHashJoinExec", 3, true),
        // Two partitions: each gets the fetch and a global limit above
        // merges them; the lowered plan must keep the per-partition half.
        (
            4,
            "SELECT k, v FROM t WHERE k >= 2 LIMIT 2",
            "GpuFilterExec",
            2,
            false,
        ),
    ] {
        let config = || SessionConfig::new().with_target_partitions(target_partitions);
        let context = format!("target_partitions={target_partitions}: {sql}");
        let (_, stock) = plan_and_text(&plain_ctx_with(config()), sql).await;
        let fetch = format!("fetch={limit}");
        let carrier = stock.lines().position(|l| {
            let l = l.trim_start();
            (l.starts_with("FilterExec:") || l.starts_with("HashJoinExec:")) && l.contains(&fetch)
        });
        assert!(
            carrier.is_some() && (carrier == Some(0)) == only_limit,
            "{context}: the stock plan no longer carries the limit the way this \
             test assumes, so it no longer covers #86:\n{stock}"
        );

        let ctx = ctx_with_rule_and(BackendKind::Cuda, config());
        let (plan, text) = plan_and_text(&ctx, sql).await;
        let rows = sorted_rows(&collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap());
        assert_eq!(rows.len(), limit, "{context}:\n{text}");
        let above = parent_of(&plan, lowered);
        assert_eq!(
            above.as_ref().and_then(|p| p.fetch()),
            Some(limit),
            "{context}: the node directly above {lowered} carries the fetch:\n{text}"
        );
        // Which rows a LIMIT keeps is unspecified; each must still be a row
        // of the query without it.
        let unlimited = sql.replace(&format!(" LIMIT {limit}"), "");
        let all = sorted_rows(
            &plain_ctx_with(config())
                .sql(&unlimited)
                .await
                .unwrap()
                .collect()
                .await
                .unwrap(),
        );
        assert!(
            all.len() > limit,
            "{context}: the fixture must exceed the limit"
        );
        for row in &rows {
            assert!(
                all.contains(row),
                "{context}: `{row}` is not one of its rows"
            );
        }
    }
}

/// The parent of the first node called `name`, depth first.
fn parent_of(plan: &Arc<dyn ExecutionPlan>, name: &str) -> Option<Arc<dyn ExecutionPlan>> {
    plan.children().into_iter().find_map(|child| {
        if child.name() == name {
            Some(Arc::clone(plan))
        } else {
            parent_of(child, name)
        }
    })
}

/// The device filter takes at most `MAX_PREDICATE_LEAVES` (64) comparisons,
/// a cap that guards plans decoded off the network. A longer WHERE used to
/// fail the whole query on a GPU target; it stays on the CPU with a note now
/// (#133), and 64 comparisons still lower.
#[tokio::test]
async fn a_predicate_over_the_device_cap_stays_on_the_cpu_with_a_note() {
    let query = |comparisons: i64| {
        let predicate = (0..comparisons)
            .map(|j| format!("c{j} >= {}", j + 3))
            .collect::<Vec<_>>()
            .join(" AND ");
        format!("SELECT c0, c64 FROM w WHERE {predicate}")
    };
    let over = query(65);
    let plan = assert_same_results(&over, BackendKind::Cuda, None).await;
    let rows: usize = collect(plan, plain_ctx().task_ctx())
        .await
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 7, "the comparison must not be vacuous");
    assert_eq!(
        placement_notes(&over).await,
        vec![
            "FilterExec: GpuFilterExec predicate has 65 comparisons; at most 64 are supported"
                .to_owned()
        ]
    );
    assert_same_results(&query(64), BackendKind::Cuda, Some("GpuFilterExec[cuda]")).await;
}

/// `COUNT(*)` reads no column, so DataFusion gives the filter or join under it
/// an empty projection. The rule handed that to the `Gpu*Exec` constructor,
/// which refuses it, and the whole query failed to plan on every GPU target
/// (#87). The node is lowered now with one column it already has on the
/// device, and an empty projection above drops it again.
#[tokio::test]
async fn count_star_over_a_lowered_filter_or_join_plans_and_counts() {
    for (sql, lowered) in [
        (
            "SELECT count(*) FROM t WHERE k >= 2",
            "GpuFilterExec[cuda]: k >= 2, projection=[k]",
        ),
        (
            "SELECT count(*) FROM n WHERE k > 3",
            "GpuFilterExec[cuda]: k > 3, projection=[k]",
        ),
        (
            "SELECT count(*) FROM t JOIN r ON t.k = r.rk",
            "GpuHashJoinExec[cuda]",
        ),
        (
            "SELECT count(*) FROM n JOIN d ON n.k = d.bk",
            "GpuHashJoinExec[cuda]",
        ),
    ] {
        // The shape this pins: the stock node has nothing to project.
        let (_, stock) = plan_and_text(&plain_ctx(), sql).await;
        assert!(
            stock.lines().any(|l| {
                let l = l.trim_start();
                (l.starts_with("FilterExec:") || l.starts_with("HashJoinExec:"))
                    && l.contains("projection=[]")
            }),
            "{sql}: the stock node no longer has an empty projection, so this no \
             longer covers #87:\n{stock}"
        );
        let plan = assert_same_results(sql, BackendKind::Cuda, Some(lowered)).await;
        let text = displayable(plan.as_ref()).indent(true).to_string();
        let above = parent_of(
            &plan,
            if sql.contains("JOIN") {
                "GpuHashJoinExec"
            } else {
                "GpuFilterExec"
            },
        );
        assert_eq!(
            above.map(|p| (p.name().to_owned(), p.schema().fields().len())),
            Some(("ProjectionExec".to_owned(), 0)),
            "{sql}: an empty projection drops the column the device kept:\n{text}"
        );
    }
}

/// An empty projection above a filter the rule already lowered stays where
/// it is: the device filter returns at least one column, and folding the
/// projection into it failed the query the same way (#87).
#[tokio::test]
async fn an_empty_projection_over_a_lowered_filter_stays() {
    let ctx = ctx_with_rule(BackendKind::Cuda);
    let (lowered, text) = plan_and_text(&ctx, "SELECT k FROM t WHERE k >= 2").await;
    assert_eq!(lowered.name(), "GpuFilterExec", "{text}");
    let counted: Arc<dyn ExecutionPlan> =
        Arc::new(ProjectionExec::try_new(Vec::<ProjectionExpr>::new(), lowered).unwrap());
    let plan = HardwarePlacementRule::new(BackendKind::Cuda)
        .optimize(counted, &ConfigOptions::default())
        .unwrap();
    assert_eq!(plan.name(), "ProjectionExec");
    let rows: usize = collect(plan, ctx.task_ctx())
        .await
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 5);
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
                codec
                    .try_encode(
                        Arc::clone(&node),
                        &mut buf,
                        &DefaultPhysicalProtoConverter {},
                    )
                    .unwrap();
                assert!(buf.starts_with(b"OXGP"));
                let children: Vec<Arc<dyn ExecutionPlan>> =
                    node.children().into_iter().cloned().collect();
                let back = codec
                    .try_decode(
                        &buf,
                        &children,
                        &ctx.task_ctx(),
                        &DefaultPhysicalProtoConverter {},
                    )
                    .unwrap();
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
            .try_decode(
                b"OXGP\x09garbage",
                &[],
                &ctx.task_ctx(),
                &DefaultPhysicalProtoConverter {},
            )
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

// --- placement notes (#32) --------------------------------------------------

/// Plans `sql` on a Cuda target with a telemetry hub attached and returns the
/// `node: reason` notes the rule recorded.
async fn placement_notes(sql: &str) -> Vec<String> {
    placement_notes_with(SessionConfig::new(), sql).await
}

/// [`placement_notes`] in a session configured with `config`.
async fn placement_notes_with(config: SessionConfig, sql: &str) -> Vec<String> {
    let hub = oxidelake_core::telemetry::TelemetryHub::new();
    let rule = HardwarePlacementRule::new(BackendKind::Cuda).with_telemetry(Arc::clone(&hub));
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_physical_optimizer_rules(physical_optimizer_rules(rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register(&ctx);
    hub.clear_skips();
    ctx.sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    hub.skips()
        .into_iter()
        .map(|s| format!("{}: {}", s.node, s.reason))
        .collect()
}

/// A node that stayed on the CPU says why, and the reason names the thing the
/// reader has to change. Without this the plan shows an ordinary
/// `AggregateExec` and the reason lives only in a `debug!` line.
#[tokio::test]
async fn an_ineligible_shape_records_why_it_stayed_on_the_cpu() {
    let notes = placement_notes("SELECT k, avg(v) FROM t GROUP BY k").await;
    assert!(
        notes.iter().any(|n| n.starts_with("AggregateExec:")
            && n.contains("avg")
            && n.contains("sum, count, min, max")),
        "{notes:?}"
    );

    let notes = placement_notes("SELECT s, sum(v) FROM t GROUP BY s").await;
    assert!(
        notes
            .iter()
            .any(|n| n.contains("grouping key `s`") && n.contains("Int64")),
        "{notes:?}"
    );

    let notes = placement_notes("SELECT k FROM t WHERE k >= 2 OR v < 1.0").await;
    assert!(
        notes.iter().any(|n| n.starts_with("FilterExec:")),
        "{notes:?}"
    );

    // A string comparison: the literal is not one the kernels hold, and the
    // note says so rather than repeating "ineligible".
    let notes = placement_notes("SELECT k FROM t WHERE s = 's1'").await;
    assert!(
        notes
            .iter()
            .any(|n| n.starts_with("FilterExec:") && n.contains("Int64 or Float64 literal")),
        "{notes:?}"
    );
}

/// A plan the rule lowered completely has nothing to explain. A note that
/// appeared for an accelerated query would teach readers to ignore the
/// section.
#[tokio::test]
async fn a_fully_lowered_plan_records_no_notes() {
    assert_eq!(
        placement_notes("SELECT k, sum(v) FROM t WHERE k >= 2 GROUP BY k").await,
        Vec::<String>::new()
    );
}

/// `!=` has no device comparison; the note used to list it among the ones
/// the kernels implement (#94).
#[tokio::test]
async fn the_not_equal_note_names_only_comparisons_that_exist() {
    assert_eq!(
        placement_notes("SELECT k FROM t WHERE k != 2").await,
        vec![
            "FilterExec: `!=` is not one of the comparisons the device kernels \
             implement (=, <, <=, >, >=)"
                .to_owned()
        ]
    );
}

/// A join the device cannot take says why (#94). It used to record nothing,
/// which reads as if the rule never looked at it. Every early return of both
/// join lowerings is covered: the hash join DataFusion plans by default and
/// the merge join it plans when hash joins are not preferred (Ballista).
#[tokio::test]
async fn a_join_left_on_the_cpu_records_why() {
    for (config, node) in [
        (SessionConfig::new(), "HashJoinExec"),
        (merge_join_config(), "SortMergeJoinExec"),
    ] {
        for (sql, says) in [
            (
                "SELECT n.k, d.y FROM n LEFT JOIN d ON n.k = d.bk",
                "join has no device kernel; the device join is inner only",
            ),
            (
                "SELECT n.k, d.y FROM n JOIN d ON n.k = d.bk AND n.v < d.y",
                "a condition besides its key equality",
            ),
            (
                "SELECT t.k FROM t JOIN r ON t.k = r.rk AND t.s = r.name",
                "2 key pairs; the device join takes exactly one",
            ),
            (
                "SELECT r.name FROM t JOIN r ON t.s = r.name",
                "is Utf8; the device join takes Int64 keys",
            ),
            (
                "SELECT t.k FROM t JOIN r ON t.k IS NOT DISTINCT FROM r.rk",
                "matches NULL keys to each other",
            ),
        ] {
            let notes = placement_notes_with(config.clone(), sql).await;
            assert!(
                notes
                    .iter()
                    .any(|n| n.starts_with(&format!("{node}: ")) && n.contains(says)),
                "{node}: {sql}: {notes:?}"
            );
        }
    }
}

// --- merge joins (#88) -------------------------------------------------------

/// Ballista's session default, `prefer_hash_join = false`, under which
/// DataFusion plans `SortMergeJoinExec`; four partitions so the merge join
/// runs partitioned the way it does on a cluster.
fn merge_join_config() -> SessionConfig {
    SessionConfig::new()
        .with_target_partitions(4)
        .set_bool("datafusion.optimizer.prefer_hash_join", false)
}

/// The `Int64` values of `column`, in output order.
fn column_values(batches: &[RecordBatch], column: usize) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            b.column(column)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect()
}

/// Whether `plan` (indented text) has an inner merge join with no sort
/// anywhere above it: what is above reads the merge join's order as it is.
fn merge_join_order_is_relied_on(plan: &str) -> bool {
    let join = plan.lines().position(|l| {
        l.trim_start()
            .starts_with("SortMergeJoinExec: join_type=Inner")
    });
    join.is_some_and(|at| {
        !plan
            .lines()
            .take(at)
            .any(|l| l.trim_start().starts_with("SortExec"))
    })
}

/// A merge join's output is sorted on its keys, and DataFusion plans what is
/// above it on that: it drops the sort an ORDER BY on either key needs, a
/// merge join above merges it as it arrives, and a window partitioned by
/// either key reads its partitions off it. Lowering the merge join into an
/// unordered hash join broke all three (#88): unsorted ORDER BY output, rows
/// lost from a LEFT join, and an internal assertion from the window. The
/// lowered join is re-sorted now, and its rows and order equal stock
/// DataFusion's.
#[tokio::test]
async fn a_lowered_merge_join_keeps_the_order_above_it_relies_on() {
    let shapes = [
        (
            "SELECT n.k, d.y FROM {n} n JOIN {d} d ON n.k = d.bk ORDER BY n.k",
            Some(0),
        ),
        (
            "SELECT d.bk, d.y FROM {n} n JOIN {d} d ON n.k = d.bk ORDER BY d.bk",
            Some(0),
        ),
        // Every key has four rows, so the first 20 are a whole set of keys
        // and which rows a LIMIT keeps is not up to the plan.
        (
            "SELECT n.k, d.y FROM {n} n JOIN {d} d ON n.k = d.bk ORDER BY n.k LIMIT 20",
            Some(0),
        ),
        (
            "SELECT n.k, d.y, x.y AS y2 FROM {n} n JOIN {d} d ON n.k = d.bk \
             LEFT JOIN {d} x ON n.k = x.bk",
            None,
        ),
        (
            "SELECT n.k, d.y, count(*) OVER (PARTITION BY n.k) AS c \
             FROM {n} n JOIN {d} d ON n.k = d.bk",
            None,
        ),
        (
            "SELECT n.k, d.y, count(*) OVER (PARTITION BY d.bk) AS c \
             FROM {n} n JOIN {d} d ON n.k = d.bk",
            None,
        ),
    ];
    // One partition each, and several, which DataFusion hash-repartitions
    // under the merge join the way a cluster does.
    let queries = [("n", "d"), ("np", "dp")].into_iter().flat_map(|(n, d)| {
        shapes.iter().map(move |(shape, ordered_by)| {
            (shape.replace("{n}", n).replace("{d}", d), *ordered_by)
        })
    });
    for (sql, ordered_by) in queries {
        let sql = sql.as_str();
        let (_, stock) = plan_and_text(&plain_ctx_with(merge_join_config()), sql).await;
        assert!(
            merge_join_order_is_relied_on(&stock),
            "{sql}: no inner merge join whose order is relied on above, so this \
             no longer covers #88:\n{stock}"
        );
        let expected = plain_ctx_with(merge_join_config())
            .sql(sql)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let ctx = ctx_with_rule_and(BackendKind::Cuda, merge_join_config());
        let (plan, text) = plan_and_text(&ctx, sql).await;
        assert!(
            text.contains("GpuHashJoinExec[cuda]"),
            "{sql}: the merge join is still lowered:\n{text}"
        );
        let actual = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}\n{text}"));
        if let Some(column) = ordered_by {
            let keys = column_values(&actual, column);
            assert!(
                keys.is_sorted(),
                "{sql}: not in key order: {keys:?}\n{text}"
            );
            assert_eq!(keys, column_values(&expected, column), "{sql}");
        }
        assert_eq!(
            sorted_rows(&actual),
            sorted_rows(&expected),
            "{sql}\n{text}"
        );
        assert_eq!(
            parent_of(&plan, "GpuHashJoinExec").map(|p| p.name().to_owned()),
            Some("SortExec".to_owned()),
            "{sql}: the merge join's order is restored by one sort over the device join:\n{text}"
        );
    }
}

/// A merge join can promise more than one order. Over a `row_number()`
/// window its output is ordered by the key and by the row number, and a
/// window partitioned by the row number reads its partitions off it. A sort
/// over the device join promises only what it sorts on, so lowering that
/// join failed the window with DataFusion's internal "All partition by
/// columns should have an ordering" assertion. Such a merge join stays on
/// the CPU, and the note names the order it would lose (#88).
#[tokio::test]
async fn a_merge_join_whose_order_a_sort_cannot_restore_stays_on_the_cpu() {
    let sql = "SELECT l.k, l.rn, d.y, count(*) OVER (PARTITION BY l.rn) AS c \
               FROM (SELECT k, row_number() OVER (ORDER BY k) AS rn FROM n) l \
               JOIN d ON l.k = d.bk";
    let (_, stock) = plan_and_text(&plain_ctx_with(merge_join_config()), sql).await;
    assert!(
        merge_join_order_is_relied_on(&stock) && stock.contains("PARTITION BY [l.rn]"),
        "the window no longer reads the row-number order off the merge join, so \
         this no longer covers the case:\n{stock}"
    );
    let ctx = ctx_with_rule_and(BackendKind::Cuda, merge_join_config());
    let (plan, text) = plan_and_text(&ctx, sql).await;
    let actual = collect(plan, ctx.task_ctx())
        .await
        .unwrap_or_else(|e| panic!("{e}\n{text}"));
    let expected = plain_ctx_with(merge_join_config())
        .sql(sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(sorted_rows(&actual), sorted_rows(&expected), "{text}");
    assert!(
        text.contains("SortMergeJoinExec: join_type=Inner") && !text.contains("GpuHashJoinExec"),
        "the merge join stays:\n{text}"
    );
    let notes = placement_notes_with(merge_join_config(), sql).await;
    assert!(
        notes
            .iter()
            .any(|n| n.starts_with(
                "SortMergeJoinExec: the merge join's output is also ordered by `rn@1 ASC"
            ) && n.ends_with("an order a sort over the device join would not promise")),
        "{notes:?}"
    );
}

/// A lowered join promises what any inner equi-join does: its two keys hold
/// the same value on every row. DataFusion sorts a hash join's output on one
/// key for a window partitioned by the other; the device join promised no
/// such equality, so that window failed over it with the same internal
/// assertion. This is DataFusion's default join, so it failed embedded on
/// any GPU target, not only under Ballista's merge joins (found with #88).
#[tokio::test]
async fn a_window_by_the_other_key_runs_over_a_lowered_hash_join() {
    let sql = "SELECT n.k, d.y, count(*) OVER (PARTITION BY d.bk) AS c \
               FROM n JOIN d ON n.k = d.bk";
    for target_partitions in [1, 4] {
        let config = || SessionConfig::new().with_target_partitions(target_partitions);
        let context = format!("target_partitions={target_partitions}");
        let (_, stock) = plan_and_text(&plain_ctx_with(config()), sql).await;
        assert!(
            stock.contains("HashJoinExec: mode=") && stock.contains("SortExec: expr=[k@0"),
            "{context}: the window over d.bk no longer sorts on n.k, so this no \
             longer covers the case:\n{stock}"
        );
        let ctx = ctx_with_rule_and(BackendKind::Cuda, config());
        let (plan, text) = plan_and_text(&ctx, sql).await;
        assert!(text.contains("GpuHashJoinExec[cuda]"), "{context}:\n{text}");
        let actual = collect(plan, ctx.task_ctx())
            .await
            .unwrap_or_else(|e| panic!("{context}: {e}\n{text}"));
        let expected = plain_ctx_with(config())
            .sql(sql)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(sorted_rows(&actual), sorted_rows(&expected), "{context}");
    }
}
