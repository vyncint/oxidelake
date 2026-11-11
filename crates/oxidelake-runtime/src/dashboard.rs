//! Building a [`DashboardModel`] from a live embedded session: the executed
//! plan as a DAG with placement tags, each operator's own counters as it
//! reported them to the session's
//! [`TelemetryHub`](oxidelake_core::telemetry::TelemetryHub), and per-column
//! profiles of the registered tables. This is what `oxide tui --query …`
//! renders; without a query the CLI falls back to the synthetic demo fixture.

use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, new_empty_array};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::common::TableReference;
use datafusion::functions_aggregate::min_max::{MaxAccumulator, MinAccumulator};
use datafusion::logical_expr::Accumulator;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use oxidelake_compute::{GpuAggregateExec, GpuFilterExec, GpuHashJoinExec, GpuVectorDistanceExec};
use oxidelake_core::telemetry::{OperatorSnapshot, PlanNodeSummary};
use oxidelake_core::{BackendKind, EngineError};
use oxidelake_tui::{ColumnProfile, DashboardModel};

use crate::session::OxideSession;

/// The backend a physical node was planned for, and the counters it
/// reported: the tag and the hub entry of our `Gpu*Exec` nodes, CPU and
/// nothing for every stock DataFusion node, which does not report.
fn node_placement(node: &dyn ExecutionPlan) -> (BackendKind, Option<OperatorSnapshot>) {
    if let Some(exec) = node.downcast_ref::<GpuFilterExec>() {
        (exec.target(), exec.operator_stats())
    } else if let Some(exec) = node.downcast_ref::<GpuHashJoinExec>() {
        (exec.target(), exec.operator_stats())
    } else if let Some(exec) = node.downcast_ref::<GpuAggregateExec>() {
        (exec.target(), exec.operator_stats())
    } else if let Some(exec) = node.downcast_ref::<GpuVectorDistanceExec>() {
        (exec.target(), exec.operator_stats())
    } else {
        (BackendKind::CpuSimd, None)
    }
}

/// One dashboard row per plan node, pre-order, beside that node's counters
/// (the Inspector pairs plan row *i* with operator *i*).
///
/// The counters are read from the node itself. The hub registers one entry
/// per exec instance, but an entry carries only a name, and they used to be
/// merged by it — so every operator showed the sum of all the operators that
/// shared its name, a self-join's two filters each reporting both (#123).
fn plan_rows(plan: &Arc<dyn ExecutionPlan>) -> Vec<(PlanNodeSummary, OperatorSnapshot)> {
    fn walk(
        node: &Arc<dyn ExecutionPlan>,
        depth: usize,
        out: &mut Vec<(PlanNodeSummary, OperatorSnapshot)>,
    ) {
        let line = displayable(node.as_ref()).one_line().to_string();
        let detail = line
            .split_once(": ")
            .map_or(String::new(), |(_, rest)| rest.trim().to_owned());
        let name = node.name().to_owned();
        let (backend, recorded) = node_placement(node.as_ref());
        let mut counters = recorded.unwrap_or_else(|| OperatorSnapshot {
            id: 0,
            name: name.clone(),
            backend,
            rows_in: 0,
            rows_out: 0,
            batches: 0,
            elapsed_ns: 0,
            bytes_h2d: 0,
            bytes_d2h: 0,
            memory_bytes: 0,
            fallback_batches: 0,
        });
        counters.id = out.len();
        out.push((
            PlanNodeSummary {
                depth,
                name,
                backend,
                detail,
            },
            counters,
        ));
        for child in node.children() {
            walk(child, depth + 1, out);
        }
    }
    let mut out = Vec::new();
    walk(plan, 0, &mut out);
    out
}

/// The physical plan as the dashboard's DAG rows (pre-order, with depths).
pub fn plan_summary(plan: &Arc<dyn ExecutionPlan>) -> Vec<PlanNodeSummary> {
    plan_rows(plan).into_iter().map(|(node, _)| node).collect()
}

fn render_cell(batch: &RecordBatch, column: usize) -> String {
    let opts = FormatOptions::default().with_null("");
    batch
        .columns()
        .get(column)
        .and_then(|c| ArrayFormatter::try_new(c.as_ref(), &opts).ok())
        .map_or(String::new(), |f| f.value(0).to_string())
}

/// Whether DataFusion's `min` and `max` can order `data_type` (#121).
///
/// Asked of the accumulators `min(col)` and `max(col)` run, over an empty
/// column of the type, rather than decided from a list kept here. The list
/// this replaced named `Int64`, `Float64` and `Utf8`, and DataFusion reads
/// every Parquet string as `Utf8View`, so the demo table's own `s` never had a
/// min or a max. The functions' signatures cannot answer it either: their
/// `coerce_types` accept every type, and one they cannot order (a `Map`)
/// fails only at execution, which would take the whole profile query down.
fn min_max_applies(data_type: &DataType) -> bool {
    let empty = new_empty_array(data_type);
    let min = MinAccumulator::try_new(data_type)
        .and_then(|mut acc| acc.update_batch(std::slice::from_ref(&empty)));
    let max = MaxAccumulator::try_new(data_type).and_then(|mut acc| acc.update_batch(&[empty]));
    min.is_ok() && max.is_ok()
}

/// Where one column's aggregates sit in the profile query's single row.
struct Slots {
    non_null: usize,
    min_max: Option<usize>,
    quantiles: Option<usize>,
}

/// Profiles every column of `table`: null count for all of them, min and max
/// wherever DataFusion can order the type, and P25/P50/P99 for numeric
/// columns — one aggregation query per table.
pub async fn profile_table(
    session: &OxideSession,
    table: &str,
) -> Result<Vec<ColumnProfile>, EngineError> {
    let schema = session
        .ctx()
        .table_provider(table)
        .await
        .map_err(EngineError::from)?
        .schema();
    let mut selects = vec!["count(*)".to_owned()];
    // Per column: count, then min and max, then three quantiles, each only
    // where it applies; the slots say where to read them back.
    let mut layout = Vec::new();
    for field in schema.fields() {
        let name = field.name();
        let quoted = format!("\"{}\"", name.replace('"', "\"\""));
        let non_null = selects.len();
        selects.push(format!("count({quoted})"));
        let min_max = min_max_applies(field.data_type()).then(|| {
            selects.push(format!("min({quoted})"));
            selects.push(format!("max({quoted})"));
            non_null + 1
        });
        // `approx_percentile_cont` coerces any numeric type to Float64:
        // integers of every width, floats and decimals.
        let quantiles = field.data_type().is_numeric().then(|| {
            let at = selects.len();
            for q in ["0.25", "0.5", "0.99"] {
                selects.push(format!("approx_percentile_cont({quoted}, {q})"));
            }
            at
        });
        layout.push(Slots {
            non_null,
            min_max,
            quantiles,
        });
    }
    // The name comes from the CLI (`--table NAME=PATH`), so it is quoted: a
    // name containing `"` cannot escape the query. But it is quoted as
    // *registration resolved it*, not as typed. Registering `Trips` parses it
    // as a SQL identifier and stores `trips`, and quoting the raw string asked
    // for a case-sensitive `"Trips"` that does not exist (#120). Resolving it
    // the same way and quoting each part where it needs it (doubling any `"`)
    // names the table `table_provider` found above.
    let quoted_table = TableReference::from(table).to_quoted_string();
    let sql = format!("SELECT {} FROM {quoted_table}", selects.join(", "));
    let batches = session.sql(&sql).await?.collect().await?;
    let row = batches
        .iter()
        .find(|b| b.num_rows() > 0)
        .ok_or_else(|| EngineError::execution("profile query returned no rows"))?;
    let total = render_cell(row, 0).parse::<u64>().unwrap_or(0);
    let mut profiles = Vec::with_capacity(schema.fields().len());
    for (field, slots) in schema.fields().iter().zip(layout) {
        let non_null = render_cell(row, slots.non_null).parse::<u64>().unwrap_or(0);
        let (min, max) = slots.min_max.map_or_else(
            || (String::new(), String::new()),
            |at| (render_cell(row, at), render_cell(row, at + 1)),
        );
        let quantile = |i: usize| {
            slots
                .quantiles
                .map_or_else(String::new, |at| render_cell(row, at + i))
        };
        profiles.push(ColumnProfile {
            name: field.name().clone(),
            data_type: field.data_type().to_string(),
            min,
            max,
            null_count: total.saturating_sub(non_null),
            p25: quantile(0),
            p50: quantile(1),
            p99: quantile(2),
        });
    }
    Ok(profiles)
}

/// Runs `sql` on `session` to completion and returns the dashboard model for
/// it: the executed plan with placement tags, each operator's counters from
/// the run, and column profiles for `tables`.
pub async fn query_dashboard(
    session: &OxideSession,
    sql: &str,
    tables: &[String],
) -> Result<DashboardModel, EngineError> {
    let plan = session
        .sql(sql)
        .await?
        .create_physical_plan()
        .await
        .map_err(EngineError::from)?;
    collect(Arc::clone(&plan), session.ctx().task_ctx())
        .await
        .map_err(EngineError::from)?;
    let (nodes, operators): (Vec<_>, Vec<_>) = plan_rows(&plan).into_iter().unzip();
    session.telemetry().set_plan(nodes.clone());
    let mut telemetry = session.telemetry().snapshot();
    telemetry.operators = operators;
    telemetry.plan = nodes;
    let mut profiles = Vec::new();
    for table in tables {
        profiles.extend(profile_table(session, table).await?);
    }
    Ok(DashboardModel {
        telemetry,
        profiles,
        title: sql.to_owned(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{Field, Schema};

    use super::*;

    fn session_with_table() -> OxideSession {
        let session = OxideSession::local_with_target(BackendKind::Cuda).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(2)])),
                Arc::new(Float64Array::from(vec![
                    Some(0.5),
                    Some(1.5),
                    Some(2.5),
                    None,
                ])),
            ],
        )
        .unwrap();
        session.ctx().register_batch("t", batch).unwrap();
        session
    }

    #[tokio::test]
    async fn dashboard_reports_plan_tags_and_live_counters() {
        let session = session_with_table();
        let model = query_dashboard(
            &session,
            "SELECT k, sum(v) FROM t WHERE k >= 1 GROUP BY k",
            &["t".to_owned()],
        )
        .await
        .unwrap();
        let names: Vec<&str> = model.plan().iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"GpuAggregateExec"), "{names:?}");
        assert!(names.contains(&"GpuFilterExec"), "{names:?}");
        let agg_row = model
            .plan()
            .iter()
            .position(|n| n.name == "GpuAggregateExec")
            .unwrap();
        assert_eq!(model.plan()[agg_row].backend, BackendKind::Cuda);
        // The run recorded real batches through the telemetry hub.
        let op = model.operator(agg_row).unwrap();
        assert_eq!(op.name, "GpuAggregateExec");
        assert!(op.rows_in > 0 && op.batches > 0, "{op:?}");
        // Profiles: k and v with correct null counts.
        assert_eq!(model.profiles.len(), 2);
        assert_eq!(model.profiles[0].name, "k");
        assert_eq!(model.profiles[0].null_count, 1);
        assert_eq!(model.profiles[0].min, "1");
        assert_eq!(model.profiles[0].max, "2");
        assert_eq!(model.profiles[1].null_count, 1);
        assert!(!model.profiles[1].p50.is_empty());
    }

    /// A Parquet demo table registered the way `oxide tui --table NAME=PATH`
    /// registers it.
    async fn session_with_parquet(name: &str) -> (OxideSession, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        oxidelake_storage::write_demo_table(
            dir.path(),
            300,
            7,
            &oxidelake_storage::demo_write_options(128, oxidelake_storage::Compression::None),
        )
        .unwrap();
        let session = OxideSession::local_with_target(BackendKind::CpuSimd).unwrap();
        session
            .register_parquet(name, dir.path().to_str().unwrap())
            .await
            .unwrap();
        (session, dir)
    }

    /// `--table Trips=PATH` registers `trips`: DataFusion folds an unquoted
    /// name to lower case. The profile query used to quote the name as typed,
    /// and `"Trips"` is case-sensitive, so the dashboard failed with `table
    /// 'datafusion.public.Trips' not found` after the user's own query had
    /// already run (#120).
    ///
    /// The other two names keep the quoting the 2026-09-02 audit (S6) added:
    /// one that is no SQL identifier at all, and one carrying a `"` that
    /// would end a naively quoted name and let the rest run as SQL.
    #[tokio::test]
    async fn a_table_named_in_upper_case_is_profiled() {
        for (name, from) in [
            ("Trips", "Trips"),
            ("trip data", "\"trip data\""),
            ("a\"b", "\"a\"\"b\""),
        ] {
            let (session, _dir) = session_with_parquet(name).await;
            let model = query_dashboard(
                &session,
                &format!("SELECT k, sum(v) FROM {from} GROUP BY k"),
                &[name.to_owned()],
            )
            .await
            .unwrap_or_else(|e| panic!("--table {name}=…: {e}"));
            let columns: Vec<&str> = model.profiles.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(columns, ["id", "k", "v", "s", "emb"], "{name}");
            assert_eq!(
                model.profiles[0].max, "299",
                "{name}: {:?}",
                model.profiles[0]
            );
        }
    }

    /// Describe used to profile min and max for `Int64`, `Float64` and `Utf8`
    /// only, and DataFusion reads every Parquet string as `Utf8View` — so the
    /// demo table's own `s`, read the way `oxide tui --table` reads it, showed
    /// none, nor did any other integer width, date or list (#121).
    #[tokio::test]
    async fn every_column_min_and_max_can_order_gets_them() {
        let (session, _dir) = session_with_parquet("t").await;
        let profiles = profile_table(&session, "t").await.unwrap();
        let profile = |name: &str| profiles.iter().find(|p| p.name == name).unwrap();
        let s = profile("s");
        assert_eq!(
            s.data_type, "Utf8View",
            "the type Parquet strings arrive as"
        );
        assert!(s.min.starts_with('s') && s.max.starts_with('s'), "{s:?}");
        assert!(s.min < s.max, "{s:?}");
        assert!(s.p50.is_empty(), "a string has no quantiles: {s:?}");
        let emb = profile("emb");
        assert!(emb.data_type.starts_with("FixedSizeList"), "{emb:?}");
        assert!(
            emb.min.starts_with('[') && emb.max.starts_with('['),
            "{emb:?}"
        );
    }

    /// The other type families, and one that `min` and `max` cannot order: a
    /// `Map` column. Their `coerce_types` accept it and the accumulator
    /// refuses it at execution, which would fail the whole profile query —
    /// so it has to be left out, not tried.
    #[tokio::test]
    async fn the_other_type_families_are_profiled_and_a_map_is_left_out() {
        use arrow::array::{
            ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Int32Array,
            LargeStringArray, MapBuilder, StringBuilder, TimestampMillisecondArray, UInt32Array,
        };
        let mut map = MapBuilder::new(None, StringBuilder::new(), Int32Array::builder(4));
        for (key, value) in [("a", 1), ("b", 2), ("c", 3)] {
            map.keys().append_value(key);
            map.values().append_value(value);
            map.append(true).unwrap();
        }
        map.append(false).unwrap();
        let columns: Vec<(&str, ArrayRef)> = vec![
            (
                "i32",
                Arc::new(Int32Array::from(vec![Some(-3), Some(7), None, Some(1)])),
            ),
            ("u32", Arc::new(UInt32Array::from(vec![4, 9, 2, 6]))),
            (
                "f32",
                Arc::new(Float32Array::from(vec![0.5, -1.5, 2.0, 1.0])),
            ),
            (
                "dec",
                Arc::new(
                    Decimal128Array::from(vec![1250, -300, 999, 1])
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
            ),
            (
                "day",
                Arc::new(Date32Array::from(vec![19_000, 19_500, 18_000, 19_001])),
            ),
            (
                "ts",
                Arc::new(TimestampMillisecondArray::from(vec![0, 86_400_000, 1, 2])),
            ),
            (
                "flag",
                Arc::new(BooleanArray::from(vec![true, false, true, true])),
            ),
            (
                "big",
                Arc::new(LargeStringArray::from(vec!["m", "z", "a", "q"])),
            ),
            ("map", Arc::new(map.finish())),
        ];
        let batch = RecordBatch::try_from_iter(columns).unwrap();
        let session = OxideSession::local_with_target(BackendKind::CpuSimd).unwrap();
        session.ctx().register_batch("u", batch).unwrap();

        let profiles = profile_table(&session, "u").await.unwrap();
        let found: Vec<(&str, &str, &str, u64)> = profiles
            .iter()
            .map(|p| {
                (
                    p.name.as_str(),
                    p.min.as_str(),
                    p.max.as_str(),
                    p.null_count,
                )
            })
            .collect();
        assert_eq!(
            found,
            [
                ("i32", "-3", "7", 1),
                ("u32", "2", "9", 0),
                ("f32", "-1.5", "2.0", 0),
                ("dec", "-3.00", "12.50", 0),
                ("day", "2019-04-14", "2023-05-23", 0),
                ("ts", "1970-01-01T00:00:00", "1970-01-02T00:00:00", 0),
                ("flag", "false", "true", 0),
                ("big", "a", "z", 0),
                ("map", "", "", 1),
            ]
        );
        for p in &profiles {
            let numeric = ["i32", "u32", "f32", "dec"].contains(&p.name.as_str());
            assert_eq!(!p.p50.is_empty(), numeric, "quantiles iff numeric: {p:?}");
        }
    }

    /// Two operators of one name are two operators (#123). The Inspector used
    /// to merge the hub's counters by name and show every `GpuFilterExec` the
    /// sum of both: here, both filters as reading the table twice and
    /// emitting both filters' rows.
    #[tokio::test]
    async fn two_operators_with_one_name_keep_their_own_counters() {
        let dir = tempfile::tempdir().unwrap();
        oxidelake_storage::write_demo_table(
            dir.path(),
            3000,
            7,
            &oxidelake_storage::demo_write_options(1024, oxidelake_storage::Compression::None),
        )
        .unwrap();
        let path = dir.path().to_str().unwrap();
        let count = |sql: &'static str| async move {
            let truth = OxideSession::local_with_target(BackendKind::CpuSimd).unwrap();
            truth.register_parquet("t", path).await.unwrap();
            let batches = truth.sql(sql).await.unwrap().collect().await.unwrap();
            render_cell(&batches[0], 0).parse::<u64>().unwrap()
        };
        let (low, high, rows) = (
            count("SELECT count(*) FROM t WHERE k < 4").await,
            count("SELECT count(*) FROM t WHERE k >= 50").await,
            count("SELECT count(*) FROM t").await,
        );
        assert_ne!(
            low, high,
            "the two filters must be told apart by their counts"
        );

        let session = OxideSession::local_with_target(BackendKind::Cuda).unwrap();
        session.register_parquet("t", path).await.unwrap();
        let model = query_dashboard(
            &session,
            "SELECT a.id, b.v FROM (SELECT id, k FROM t WHERE k < 4) a \
             JOIN (SELECT k, v FROM t WHERE k >= 50) b ON a.id = b.k",
            &[],
        )
        .await
        .unwrap();
        let filter = |predicate: &str| {
            let row = model
                .plan()
                .iter()
                .position(|n| n.name == "GpuFilterExec" && n.detail.contains(predicate))
                .unwrap_or_else(|| panic!("no GpuFilterExec on `{predicate}`: {:?}", model.plan()));
            model.operator(row).unwrap().clone()
        };
        let (lt, ge) = (filter("< 4"), filter(">= 50"));
        assert_eq!((lt.rows_in, lt.rows_out), (rows, low), "k < 4: {lt:?}");
        assert_eq!((ge.rows_in, ge.rows_out), (rows, high), "k >= 50: {ge:?}");
        assert!(lt.batches > 0 && ge.batches > 0);

        // Every exec type is read the same way: the join above, and a vector
        // distance in a second query whose operators are new hub entries. The
        // projection keeps `emb`: the lowering only takes the shape that
        // appends the distance to its input (planner tests, placement.rs).
        let joined = count(
            "SELECT count(*) FROM (SELECT id, k FROM t WHERE k < 4) a \
             JOIN (SELECT k, v FROM t WHERE k >= 50) b ON a.id = b.k",
        )
        .await;
        let operator = |model: &DashboardModel, name: &str| {
            let row = model
                .plan()
                .iter()
                .position(|n| n.name == name)
                .unwrap_or_else(|| panic!("no {name}: {:?}", model.plan()));
            model.operator(row).unwrap().clone()
        };
        let join = operator(&model, "GpuHashJoinExec");
        assert_eq!(join.rows_out, joined, "{join:?}");
        let model = query_dashboard(
            &session,
            "SELECT id, emb, l2_distance(emb, [1.0, 0.0, -1.0, 2.0, 0.5, 0.0, 1.5, -0.5]) AS d FROM t",
            &[],
        )
        .await
        .unwrap();
        let distance = operator(&model, "GpuVectorDistanceExec");
        assert_eq!(
            (distance.rows_in, distance.rows_out),
            (rows, rows),
            "{distance:?}"
        );
    }
}
