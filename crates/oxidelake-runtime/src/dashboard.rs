//! Building a [`DashboardModel`] from a live embedded session: the executed
//! plan as a DAG with placement tags, per-operator counters aggregated from
//! the session's [`TelemetryHub`](oxidelake_core::telemetry::TelemetryHub), and
//! per-column profiles of the registered
//! tables. This is what `oxide tui --query …` renders; without a query the CLI
//! falls back to the synthetic demo fixture.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use oxidelake_compute::{GpuAggregateExec, GpuFilterExec, GpuHashJoinExec, GpuVectorDistanceExec};
use oxidelake_core::telemetry::{OperatorSnapshot, PlanNodeSummary};
use oxidelake_core::{BackendKind, EngineError};
use oxidelake_tui::{ColumnProfile, DashboardModel};

use crate::session::OxideSession;

/// The backend a physical node was planned for: the tag on our `Gpu*Exec`
/// nodes, CPU for every stock DataFusion node.
fn node_backend(node: &dyn ExecutionPlan) -> BackendKind {
    if let Some(exec) = node.downcast_ref::<GpuFilterExec>() {
        exec.target()
    } else if let Some(exec) = node.downcast_ref::<GpuHashJoinExec>() {
        exec.target()
    } else if let Some(exec) = node.downcast_ref::<GpuAggregateExec>() {
        exec.target()
    } else if let Some(exec) = node.downcast_ref::<GpuVectorDistanceExec>() {
        exec.target()
    } else {
        BackendKind::CpuSimd
    }
}

/// The physical plan as the dashboard's DAG rows (pre-order, with depths).
pub fn plan_summary(plan: &Arc<dyn ExecutionPlan>) -> Vec<PlanNodeSummary> {
    fn walk(node: &Arc<dyn ExecutionPlan>, depth: usize, out: &mut Vec<PlanNodeSummary>) {
        let line = displayable(node.as_ref()).one_line().to_string();
        let detail = line
            .split_once(": ")
            .map_or(String::new(), |(_, rest)| rest.trim().to_owned());
        out.push(PlanNodeSummary {
            depth,
            name: node.name().to_owned(),
            backend: node_backend(node.as_ref()),
            detail,
        });
        for child in node.children() {
            walk(child, depth + 1, out);
        }
    }
    let mut out = Vec::new();
    walk(plan, 0, &mut out);
    out
}

/// Aggregates the hub's per-partition operator counters by operator name and
/// aligns them with `plan` (the Inspector panel pairs plan row *i* with
/// operator *i*). Stock DataFusion nodes do not report into the hub and show
/// zeros.
fn operators_for_plan(
    plan: &[PlanNodeSummary],
    recorded: &[OperatorSnapshot],
) -> Vec<OperatorSnapshot> {
    let mut by_name: HashMap<&str, OperatorSnapshot> = HashMap::new();
    for op in recorded {
        let entry = by_name
            .entry(op.name.as_str())
            .or_insert_with(|| OperatorSnapshot {
                id: 0,
                name: op.name.clone(),
                backend: op.backend,
                rows_in: 0,
                rows_out: 0,
                batches: 0,
                elapsed_ns: 0,
                bytes_h2d: 0,
                bytes_d2h: 0,
                memory_bytes: 0,
            });
        entry.rows_in += op.rows_in;
        entry.rows_out += op.rows_out;
        entry.batches += op.batches;
        entry.elapsed_ns += op.elapsed_ns;
        entry.bytes_h2d += op.bytes_h2d;
        entry.bytes_d2h += op.bytes_d2h;
        entry.memory_bytes = entry.memory_bytes.max(op.memory_bytes);
    }
    plan.iter()
        .enumerate()
        .map(|(id, node)| {
            let mut op =
                by_name
                    .get(node.name.as_str())
                    .cloned()
                    .unwrap_or_else(|| OperatorSnapshot {
                        id: 0,
                        name: node.name.clone(),
                        backend: node.backend,
                        rows_in: 0,
                        rows_out: 0,
                        batches: 0,
                        elapsed_ns: 0,
                        bytes_h2d: 0,
                        bytes_d2h: 0,
                        memory_bytes: 0,
                    });
            op.id = id;
            op.backend = node.backend;
            op
        })
        .collect()
}

fn render_cell(batch: &RecordBatch, column: usize) -> String {
    let opts = FormatOptions::default().with_null("");
    batch
        .columns()
        .get(column)
        .and_then(|c| ArrayFormatter::try_new(c.as_ref(), &opts).ok())
        .map_or(String::new(), |f| f.value(0).to_string())
}

/// Profiles every column of `table`: min, max, null count, and P25/P50/P99
/// (numeric columns only) — one aggregation query per table.
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
    // Per column: [count, min?, max?, p25?, p50?, p99?] — track each column's
    // slot layout so the result row can be unpacked positionally.
    let mut layout = Vec::new();
    for field in schema.fields() {
        let name = field.name();
        let quoted = format!("\"{}\"", name.replace('"', "\"\""));
        let numeric = matches!(field.data_type(), DataType::Int64 | DataType::Float64);
        let orderable = numeric || matches!(field.data_type(), DataType::Utf8);
        let first = selects.len();
        selects.push(format!("count({quoted})"));
        if orderable {
            selects.push(format!("min({quoted})"));
            selects.push(format!("max({quoted})"));
        }
        if numeric {
            for q in ["0.25", "0.5", "0.99"] {
                selects.push(format!("approx_percentile_cont({quoted}, {q})"));
            }
        }
        layout.push((first, orderable, numeric));
    }
    // Identifiers come from the CLI (`--table NAME=PATH`); quote them like the
    // column names above so a name containing `"` cannot escape the query.
    let quoted_table = format!("\"{}\"", table.replace('"', "\"\""));
    let sql = format!("SELECT {} FROM {quoted_table}", selects.join(", "));
    let batches = session.sql(&sql).await?.collect().await?;
    let row = batches
        .iter()
        .find(|b| b.num_rows() > 0)
        .ok_or_else(|| EngineError::execution("profile query returned no rows"))?;
    let total = render_cell(row, 0).parse::<u64>().unwrap_or(0);
    let mut profiles = Vec::with_capacity(schema.fields().len());
    for (field, (first, orderable, numeric)) in schema.fields().iter().zip(layout) {
        let non_null = render_cell(row, first).parse::<u64>().unwrap_or(0);
        let (min, max) = if orderable {
            (render_cell(row, first + 1), render_cell(row, first + 2))
        } else {
            (String::new(), String::new())
        };
        let quantile = |i: usize| {
            if numeric {
                render_cell(row, first + 3 + i)
            } else {
                String::new()
            }
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
/// it: the executed plan with placement tags, the aggregated telemetry the
/// run produced, and column profiles for `tables`.
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
    let nodes = plan_summary(&plan);
    session.telemetry().set_plan(nodes.clone());
    let mut telemetry = session.telemetry().snapshot();
    telemetry.operators = operators_for_plan(&nodes, &telemetry.operators);
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
}
