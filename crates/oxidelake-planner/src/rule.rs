//! `HardwarePlacementRule`: rewrites eligible physical nodes into `Gpu*Exec`.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{JoinType, NullEquality, ScalarValue};
use datafusion::error::Result;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::{HashJoinExec, SortMergeJoinExec};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use oxidelake_compute::{
    AggregateFunction, AggregateSpec, Comparison, GpuAggregateExec, GpuFilterExec, GpuHashJoinExec,
    GpuVectorDistanceExec, Literal as Lit, Predicate, distance_metric_for, literal_query,
    local_backend,
};
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_core::{BackendKind, EngineError};

use crate::target::cluster_target_from_env;

/// Rewrites eligible nodes into `Gpu*Exec` when the target backend is a GPU.
#[derive(Debug, Clone)]
pub struct HardwarePlacementRule {
    target: BackendKind,
    telemetry: Option<Arc<TelemetryHub>>,
}

impl HardwarePlacementRule {
    /// A rule targeting `target`. `CpuSimd` disables all rewrites.
    pub const fn new(target: BackendKind) -> Self {
        Self {
            target,
            telemetry: None,
        }
    }

    /// Every `Gpu*Exec` this rule creates reports per-batch statistics into
    /// `telemetry` (used by embedded sessions to feed the dashboard).
    #[must_use]
    pub fn with_telemetry(mut self, telemetry: Arc<TelemetryHub>) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    /// A rule targeting the backend detected on this machine (embedded mode).
    pub fn local() -> Result<Self> {
        Ok(Self::new(local_backend()?.kind()))
    }

    /// A rule targeting the declared cluster capability (`OXIDE_CLUSTER_BACKEND`).
    pub fn from_env() -> std::result::Result<Self, EngineError> {
        Ok(Self::new(cluster_target_from_env()?))
    }

    /// The target backend.
    pub const fn target(&self) -> BackendKind {
        self.target
    }

    /// Attaches the rule's telemetry hub to a freshly built exec, when set.
    fn instrument<T>(&self, exec: T, with_telemetry: impl FnOnce(T, Arc<TelemetryHub>) -> T) -> T {
        match &self.telemetry {
            Some(hub) => with_telemetry(exec, Arc::clone(hub)),
            None => exec,
        }
    }

    fn rewrite(&self, node: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        if let Some(filter) = node.downcast_ref::<FilterExec>()
            && let Some(gpu) = self.lower_filter(filter)?
        {
            return Ok(Transformed::yes(Arc::new(gpu)));
        }
        if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
            if let Some(gpu) = self.fuse_projection(projection)? {
                return Ok(Transformed::yes(Arc::new(gpu)));
            }
            if let Some(gpu) = self.lower_vector_distance(projection) {
                return Ok(Transformed::yes(Arc::new(gpu)));
            }
        }
        if let Some(join) = node.downcast_ref::<HashJoinExec>()
            && let Some(gpu) = self.lower_join(join)?
        {
            return Ok(Transformed::yes(Arc::new(gpu)));
        }
        if let Some(join) = node.downcast_ref::<SortMergeJoinExec>()
            && let Some(gpu) = self.lower_sort_merge_join(join)?
        {
            return Ok(Transformed::yes(Arc::new(gpu)));
        }
        if let Some(agg) = node.downcast_ref::<AggregateExec>()
            && let Some(gpu) = self.lower_aggregate(agg)?
        {
            return Ok(Transformed::yes(Arc::new(gpu)));
        }
        Ok(Transformed::no(node))
    }

    fn lower_filter(&self, filter: &FilterExec) -> Result<Option<GpuFilterExec>> {
        let input = filter.input();
        let schema = input.schema();
        let Some(predicate) = lower_predicate(filter.predicate(), &schema) else {
            return Ok(None);
        };
        let projection: Vec<usize> = match filter.projection() {
            Some(p) => p.iter().copied().collect(),
            None => (0..schema.fields().len()).collect(),
        };
        let exec = GpuFilterExec::try_new(Arc::clone(input), predicate, projection, self.target)?;
        if exec.schema() != filter.schema() {
            tracing::debug!("skipping FilterExec: fused schema differs from the original");
            return Ok(None);
        }
        Ok(Some(self.instrument(exec, GpuFilterExec::with_telemetry)))
    }

    /// `ProjectionExec` of plain columns over an already-lowered `GpuFilterExec`
    /// folds into the filter's projection when no column is renamed.
    fn fuse_projection(&self, projection: &ProjectionExec) -> Result<Option<GpuFilterExec>> {
        let Some(filter) = projection.input().downcast_ref::<GpuFilterExec>() else {
            return Ok(None);
        };
        let filter_schema = projection.input().schema();
        let mut fused = Vec::with_capacity(projection.expr().len());
        for expr in projection.expr() {
            let Some(column) = as_any(&expr.expr).downcast_ref::<Column>() else {
                return Ok(None);
            };
            let field = filter_schema.field(column.index());
            if field.name() != &expr.alias {
                return Ok(None);
            }
            fused.push(filter.projection()[column.index()]);
        }
        let exec = GpuFilterExec::try_new(
            Arc::clone(filter.input()),
            filter.predicate().clone(),
            fused,
            self.target,
        )?;
        if exec.schema() != projection.schema() {
            return Ok(None);
        }
        Ok(Some(self.instrument(exec, GpuFilterExec::with_telemetry)))
    }

    /// A `ProjectionExec` that passes every input column through unchanged and
    /// appends one `l2_distance` / `cosine_distance` call over a column and a
    /// query-vector literal — the shape `SELECT *, l2_distance(emb, …) AS d …`
    /// and the DataFrame `vector_distance` verb both plan — lowers to
    /// [`GpuVectorDistanceExec`]. Any other shape (reordered or dropped
    /// columns, a non-literal query, a wrong column type) stays on the CPU.
    fn lower_vector_distance(&self, projection: &ProjectionExec) -> Option<GpuVectorDistanceExec> {
        let input = projection.input();
        let input_schema = input.schema();
        let (last, passthrough) = projection.expr().split_last()?;
        if passthrough.len() != input_schema.fields().len() {
            return None;
        }
        for (position, expr) in passthrough.iter().enumerate() {
            let column = as_any(&expr.expr).downcast_ref::<Column>()?;
            if column.index() != position || input_schema.field(position).name() != &expr.alias {
                return None;
            }
        }
        let call = as_any(&last.expr).downcast_ref::<ScalarFunctionExpr>()?;
        let metric = distance_metric_for(call.fun().name())?;
        let [a, b] = call.args() else {
            return None;
        };
        let (column, literal) = match (
            as_any(a).downcast_ref::<Column>(),
            as_any(b).downcast_ref::<Column>(),
        ) {
            (Some(column), None) => (column, as_any(b).downcast_ref::<Literal>()?),
            (None, Some(column)) => (column, as_any(a).downcast_ref::<Literal>()?),
            _ => return None,
        };
        let query = literal_query(literal.value())?;
        let exec = match GpuVectorDistanceExec::try_new(
            Arc::clone(input),
            column.index(),
            query,
            metric,
            last.alias.clone(),
            self.target,
        ) {
            Ok(exec) => exec,
            Err(err) => {
                tracing::debug!(error = %err, "skipping ProjectionExec: distance call not representable");
                return None;
            }
        };
        (exec.schema() == projection.schema())
            .then(|| self.instrument(exec, GpuVectorDistanceExec::with_telemetry))
    }

    /// DataFusion builds on its *left* input and probes with the *right*;
    /// `GpuHashJoinExec` streams its left (probe) input and collects its right
    /// (build) input. The sides are therefore swapped, and DataFusion's
    /// `left ++ right` column order (plus any projection pushed into the join)
    /// is re-expressed as a projection over our `probe ++ build` order, so the
    /// output schema is identical.
    fn lower_join(&self, join: &HashJoinExec) -> Result<Option<GpuHashJoinExec>> {
        if *join.join_type() != JoinType::Inner
            || join.filter().is_some()
            || join.null_equality() != NullEquality::NullEqualsNothing
            || join.on().len() != 1
        {
            return Ok(None);
        }
        let (build, probe) = (join.left(), join.right());
        let (bs, ps) = (build.schema(), probe.schema());
        let (b, p) = &join.on()[0];
        let (Some(build_key), Some(probe_key)) = (column_index(b), column_index(p)) else {
            return Ok(None);
        };
        if bs.field(build_key).data_type() != &DataType::Int64
            || ps.field(probe_key).data_type() != &DataType::Int64
        {
            return Ok(None);
        }
        let (bl, pl) = (bs.fields().len(), ps.fields().len());
        // DataFusion's full output order is build ++ probe.
        let full: Vec<_> = bs
            .fields()
            .iter()
            .chain(ps.fields().iter())
            .cloned()
            .collect();
        let df_projection: Vec<usize> = if join.contains_projection() {
            let mut indices = Vec::with_capacity(join.schema().fields().len());
            for field in join.schema().fields() {
                let matches: Vec<usize> = full
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| *f == field)
                    .map(|(i, _)| i)
                    .collect();
                match matches.as_slice() {
                    [only] => indices.push(*only),
                    _ => {
                        tracing::debug!(field = %field.name(), "skipping HashJoinExec: projected field is ambiguous");
                        return Ok(None);
                    }
                }
            }
            indices
        } else {
            (0..bl + pl).collect()
        };
        // Re-express over our probe ++ build order.
        let ours: Vec<usize> = df_projection
            .iter()
            .map(|&i| if i < bl { pl + i } else { i - bl })
            .collect();
        let exec = GpuHashJoinExec::try_new(
            Arc::clone(probe),
            Arc::clone(build),
            probe_key,
            build_key,
            self.target,
        )?
        .with_projection(Some(ours))?;
        if exec.schema() != join.schema() {
            tracing::debug!("skipping HashJoinExec: rewritten schema differs from the original");
            return Ok(None);
        }
        Ok(Some(self.instrument(exec, GpuHashJoinExec::with_telemetry)))
    }

    /// `SortMergeJoinExec` (what DataFusion plans when hash joins are not
    /// preferred) has the same inner-join semantics; its output order is
    /// `left ++ right`, so no side swap is needed. The `SortExec`s DataFusion
    /// inserted for the merge are dropped when they carry no `LIMIT`.
    fn lower_sort_merge_join(&self, join: &SortMergeJoinExec) -> Result<Option<GpuHashJoinExec>> {
        if join.join_type() != JoinType::Inner
            || join.filter().is_some()
            || join.null_equality() != NullEquality::NullEqualsNothing
            || join.on().len() != 1
        {
            return Ok(None);
        }
        let (left, right) = (strip_sort(join.left()), strip_sort(join.right()));
        let (ls, rs) = (left.schema(), right.schema());
        if join.schema().fields().len() != ls.fields().len() + rs.fields().len() {
            return Ok(None);
        }
        let (l, r) = &join.on()[0];
        let (Some(lk), Some(rk)) = (column_index(l), column_index(r)) else {
            return Ok(None);
        };
        if ls.field(lk).data_type() != &DataType::Int64
            || rs.field(rk).data_type() != &DataType::Int64
        {
            return Ok(None);
        }
        let exec = GpuHashJoinExec::try_new(left, right, lk, rk, self.target)?;
        if exec.schema() != join.schema() {
            tracing::debug!(
                "skipping SortMergeJoinExec: rewritten schema differs from the original"
            );
            return Ok(None);
        }
        Ok(Some(self.instrument(exec, GpuHashJoinExec::with_telemetry)))
    }

    fn lower_aggregate(&self, agg: &AggregateExec) -> Result<Option<GpuAggregateExec>> {
        let (source, spec) = match agg.mode() {
            AggregateMode::Single | AggregateMode::SinglePartitioned => {
                let Some(spec) = lower_aggregate_spec(agg) else {
                    return Ok(None);
                };
                (Arc::clone(agg.input()), spec)
            }
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                // Look through the exchange between the final and partial halves.
                let mut child = Arc::clone(agg.input());
                loop {
                    if child.downcast_ref::<RepartitionExec>().is_some()
                        || child.downcast_ref::<CoalescePartitionsExec>().is_some()
                    {
                        let next = Arc::clone(child.children()[0]);
                        child = next;
                    } else {
                        break;
                    }
                }
                let Some(partial) = child.downcast_ref::<AggregateExec>() else {
                    return Ok(None);
                };
                if *partial.mode() != AggregateMode::Partial
                    || partial.aggr_expr().len() != agg.aggr_expr().len()
                    || partial.group_expr().expr().len() != agg.group_expr().expr().len()
                {
                    return Ok(None);
                }
                let Some(spec) = lower_aggregate_spec(partial) else {
                    return Ok(None);
                };
                (Arc::clone(partial.input()), spec)
            }
            _ => return Ok(None),
        };
        // A multi-partition source keeps one `CoalescePartitionsExec` under the
        // exec. Embedded mode does not need it (the exec collects every input
        // partition itself), but Ballista's planner cuts shuffle stages exactly
        // at exchange nodes — without one, a distributed plan runs the whole
        // scan + aggregate fragment as one stage and loses every partition but
        // the task's own. Found running the README's 1M-row quickstart against
        // a real cluster: files over 10 MiB are byte-range-split into
        // multi-partition scans, which no smaller test had produced.
        let source = if source.output_partitioning().partition_count() > 1 {
            Arc::new(CoalescePartitionsExec::new(source)) as Arc<dyn ExecutionPlan>
        } else {
            source
        };
        let exec = GpuAggregateExec::try_new(source, spec, self.target)?;
        match exec.with_output_schema(agg.schema()) {
            Ok(exec) => Ok(Some(
                self.instrument(exec, GpuAggregateExec::with_telemetry),
            )),
            Err(err) => {
                tracing::debug!(error = %err, "skipping AggregateExec: output schema not representable");
                Ok(None)
            }
        }
    }
}

/// Drops a `SortExec` that only exists to feed a merge join (no `LIMIT`).
fn strip_sort(plan: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    match plan.downcast_ref::<SortExec>() {
        Some(sort) if sort.fetch().is_none() => Arc::clone(sort.input()),
        _ => Arc::clone(plan),
    }
}

fn as_any(expr: &Arc<dyn PhysicalExpr>) -> &dyn Any {
    expr.as_ref()
}

fn column_index(expr: &Arc<dyn PhysicalExpr>) -> Option<usize> {
    as_any(expr).downcast_ref::<Column>().map(Column::index)
}

fn flip(op: Comparison) -> Comparison {
    match op {
        Comparison::Eq => Comparison::Eq,
        Comparison::Lt => Comparison::Gt,
        Comparison::LtEq => Comparison::GtEq,
        Comparison::Gt => Comparison::Lt,
        Comparison::GtEq => Comparison::LtEq,
    }
}

fn comparison(op: &Operator) -> Option<Comparison> {
    Some(match op {
        Operator::Eq => Comparison::Eq,
        Operator::Lt => Comparison::Lt,
        Operator::LtEq => Comparison::LtEq,
        Operator::Gt => Comparison::Gt,
        Operator::GtEq => Comparison::GtEq,
        _ => return None,
    })
}

fn literal(expr: &Arc<dyn PhysicalExpr>) -> Option<Lit> {
    match as_any(expr).downcast_ref::<Literal>()?.value() {
        ScalarValue::Int64(Some(v)) => Some(Lit::Int64(v.to_owned())),
        ScalarValue::Float64(Some(v)) => Some(Lit::Float64(v.to_owned())),
        _ => None,
    }
}

/// Lowers a DataFusion predicate into the bounded v1 grammar, or `None` when
/// any part of it is outside that grammar.
pub fn lower_predicate(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Option<Predicate> {
    let binary = as_any(expr).downcast_ref::<BinaryExpr>()?;
    if *binary.op() == Operator::And {
        let l = lower_predicate(binary.left(), schema)?;
        let r = lower_predicate(binary.right(), schema)?;
        return Some(Predicate::and(l, r));
    }
    let op = comparison(binary.op())?;
    let (column, lit, op) = match (column_index(binary.left()), literal(binary.right())) {
        (Some(c), Some(l)) => (c, l, op),
        _ => match (literal(binary.left()), column_index(binary.right())) {
            (Some(l), Some(c)) => (c, l, flip(op)),
            _ => return None,
        },
    };
    let dt = schema.fields().get(column)?.data_type();
    if *dt != lit.data_type() {
        return None;
    }
    Some(Predicate::compare(column, op, lit))
}

/// Lowers an `AggregateExec`'s grouping and aggregates into an [`AggregateSpec`].
pub fn lower_aggregate_spec(agg: &AggregateExec) -> Option<AggregateSpec> {
    let group = agg.group_expr();
    if !group.is_single() || group.expr().len() != 1 || !group.null_expr().is_empty() {
        return None;
    }
    let input_schema = agg.input().schema();
    let group_by = column_index(&group.expr()[0].0)?;
    if input_schema.field(group_by).data_type() != &DataType::Int64 {
        return None;
    }
    if agg.filter_expr().iter().any(Option::is_some) {
        return None;
    }
    let mut aggregates = Vec::with_capacity(agg.aggr_expr().len());
    for expr in agg.aggr_expr() {
        let func = match expr.fun().name() {
            "sum" => AggregateFunction::Sum,
            "count" => AggregateFunction::Count,
            "min" => AggregateFunction::Min,
            "max" => AggregateFunction::Max,
            _ => return None,
        };
        if expr.is_distinct() {
            return None;
        }
        let args = expr.expressions();
        if args.len() != 1 {
            return None;
        }
        let column = column_index(&args[0])?;
        if !matches!(
            input_schema.field(column).data_type(),
            DataType::Int64 | DataType::Float64
        ) {
            return None;
        }
        aggregates.push((func, column));
    }
    Some(AggregateSpec {
        group_by,
        aggregates,
    })
}

impl PhysicalOptimizerRule for HardwarePlacementRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !self.target.is_gpu() {
            return Ok(plan);
        }
        plan.transform_up(|node| self.rewrite(node)).map(|t| t.data)
    }

    fn name(&self) -> &str {
        "HardwarePlacementRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// The vector-distance half of the placement rule on its own. It must run
/// *before* DataFusion's `ProjectionPushdown`, which otherwise folds an
/// eligible distance projection into `DataSourceExec` itself where no rule can
/// lower it any more; every other lowering keeps running at the end of the
/// pipeline (after the exchanges and join orders it matches are final). See
/// [`physical_optimizer_rules`].
#[derive(Debug, Clone)]
pub struct VectorDistancePlacementRule(HardwarePlacementRule);

impl VectorDistancePlacementRule {
    /// Wraps `rule`, keeping its target and telemetry hub.
    pub fn new(rule: HardwarePlacementRule) -> Self {
        Self(rule)
    }
}

impl PhysicalOptimizerRule for VectorDistancePlacementRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !self.0.target().is_gpu() {
            return Ok(plan);
        }
        plan.transform_up(|node| {
            if let Some(projection) = node.downcast_ref::<ProjectionExec>()
                && let Some(gpu) = self.0.lower_vector_distance(projection)
            {
                return Ok(Transformed::yes(Arc::new(gpu) as Arc<dyn ExecutionPlan>));
            }
            Ok(Transformed::no(node))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "VectorDistancePlacementRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// DataFusion's default physical optimizers with OxideLake's placement
/// installed at the two positions it needs: [`VectorDistancePlacementRule`]
/// right before `ProjectionPushdown` and the full [`HardwarePlacementRule`] at
/// the end. Install with `SessionStateBuilder::with_physical_optimizer_rules`.
pub fn physical_optimizer_rules(
    rule: HardwarePlacementRule,
) -> Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> {
    let mut rules = datafusion::physical_optimizer::optimizer::PhysicalOptimizer::default().rules;
    let before_pushdown = rules
        .iter()
        .position(|r| r.name() == "ProjectionPushdown")
        .unwrap_or(rules.len());
    rules.insert(
        before_pushdown,
        Arc::new(VectorDistancePlacementRule::new(rule.clone())),
    );
    rules.push(Arc::new(rule));
    rules
}
