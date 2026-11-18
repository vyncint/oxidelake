//! `HardwarePlacementRule`: rewrites eligible physical nodes into `Gpu*Exec`.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{JoinType, NullEquality, ScalarValue};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_expr::{PhysicalExpr, PhysicalExprRef, ScalarFunctionExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::joins::{HashJoinExec, SortMergeJoinExec};
use datafusion::physical_plan::limit::LocalLimitExec;
use datafusion::physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
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

    /// Records why a node stayed on the CPU, in the log and — when a hub is
    /// attached — where `oxide explain` can print it (#32).
    ///
    /// A skipped node looks exactly like one that was never eligible, so
    /// without this the reason existed only in a `debug!` line nobody was
    /// going to turn on.
    fn skip(&self, node: &str, reason: impl std::fmt::Display) -> Option<()> {
        let reason = reason.to_string();
        tracing::debug!(node, reason, "leaving node on the CPU");
        if let Some(hub) = &self.telemetry {
            hub.record_skip(node, reason);
        }
        None
    }

    /// Attaches the rule's telemetry hub to a freshly built exec, when set.
    fn instrument<T>(&self, exec: T, with_telemetry: impl FnOnce(T, Arc<TelemetryHub>) -> T) -> T {
        match &self.telemetry {
            Some(hub) => with_telemetry(exec, Arc::clone(hub)),
            None => exec,
        }
    }

    fn rewrite(&self, node: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        Ok(match self.lower(&node)? {
            Some(lowered) => Transformed::yes(keep_fetch(node.fetch(), lowered)),
            None => Transformed::no(node),
        })
    }

    /// The replacement for `node`, or `None` when it stays as it is.
    fn lower(&self, node: &Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(filter) = node.downcast_ref::<FilterExec>()
            && let Some(lowered) = self.lower_filter(filter)?
        {
            return Ok(Some(lowered));
        }
        if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
            if let Some(gpu) = self.fuse_projection(projection) {
                return Ok(Some(Arc::new(gpu)));
            }
            if let Some(gpu) = self.lower_vector_distance(projection) {
                return Ok(Some(Arc::new(gpu)));
            }
        }
        if let Some(join) = node.downcast_ref::<HashJoinExec>()
            && let Some(lowered) = self.lower_join(join)?
        {
            return Ok(Some(lowered));
        }
        if let Some(join) = node.downcast_ref::<SortMergeJoinExec>()
            && let Some(lowered) = self.lower_sort_merge_join(join)?
        {
            return Ok(Some(lowered));
        }
        if let Some(agg) = node.downcast_ref::<AggregateExec>()
            && let Some(gpu) = self.lower_aggregate(agg)?
        {
            return Ok(Some(Arc::new(gpu)));
        }
        Ok(None)
    }

    fn lower_filter(&self, filter: &FilterExec) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let input = filter.input();
        let schema = input.schema();
        let predicate = match try_lower_predicate(filter.predicate(), &schema) {
            Ok(predicate) => predicate,
            Err(why) => {
                self.skip("FilterExec", why);
                return Ok(None);
            }
        };
        let projection: Vec<usize> = match filter.projection() {
            Some(p) => p.iter().copied().collect(),
            None => (0..schema.fields().len()).collect(),
        };
        // COUNT(*) reads no column, so DataFusion leaves the filter an empty
        // projection, and the device filter returns at least one column
        // (#87). It keeps the first column the predicate reads, which is on
        // the device for the comparison anyway, and an empty projection
        // above drops it again, leaving only the row count.
        let count_only = projection.is_empty();
        let projection = if count_only {
            predicate.columns().into_iter().take(1).collect()
        } else {
            projection
        };
        // The constructor's checks exist for plans decoded off the network
        // (the predicate's leaf cap is audit S5's); at planning time a node
        // they refuse stays on the CPU rather than failing the query (#133).
        let exec =
            match GpuFilterExec::try_new(Arc::clone(input), predicate, projection, self.target) {
                Ok(exec) => exec,
                Err(err) => {
                    self.skip("FilterExec", refusal(&err));
                    return Ok(None);
                }
            };
        let exec: Arc<dyn ExecutionPlan> =
            Arc::new(self.instrument(exec, GpuFilterExec::with_telemetry));
        let lowered = if count_only {
            match self.row_count_only("FilterExec", exec) {
                Some(lowered) => lowered,
                None => return Ok(None),
            }
        } else {
            exec
        };
        if lowered.schema() != filter.schema() {
            self.skip("FilterExec", "the fused schema differs from the original");
            return Ok(None);
        }
        Ok(Some(lowered))
    }

    /// `exec` under an empty projection, which keeps its row count and none
    /// of its columns: what a COUNT(*) above reads (#87). `None`, with a
    /// note, if DataFusion refuses the projection.
    fn row_count_only(
        &self,
        node: &str,
        exec: Arc<dyn ExecutionPlan>,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        match ProjectionExec::try_new(Vec::<ProjectionExpr>::new(), exec) {
            Ok(projection) => Some(Arc::new(projection)),
            Err(err) => {
                self.skip(node, refusal(&err));
                None
            }
        }
    }

    /// `ProjectionExec` of plain columns over an already-lowered `GpuFilterExec`
    /// folds into the filter's projection when no column is renamed. An
    /// empty projection (COUNT(*)) stays where it is: the device filter
    /// returns at least one column (#87).
    fn fuse_projection(&self, projection: &ProjectionExec) -> Option<GpuFilterExec> {
        let filter = projection.input().downcast_ref::<GpuFilterExec>()?;
        if projection.expr().is_empty() {
            return None;
        }
        let filter_schema = projection.input().schema();
        let mut fused = Vec::with_capacity(projection.expr().len());
        for expr in projection.expr() {
            let column = as_any(&expr.expr).downcast_ref::<Column>()?;
            let field = filter_schema.field(column.index());
            if field.name() != &expr.alias {
                return None;
            }
            fused.push(filter.projection()[column.index()]);
        }
        let exec = GpuFilterExec::try_new(
            Arc::clone(filter.input()),
            filter.predicate().clone(),
            fused,
            self.target,
        )
        .ok()?;
        (exec.schema() == projection.schema())
            .then(|| self.instrument(exec, GpuFilterExec::with_telemetry))
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
                self.skip(
                    "ProjectionExec",
                    format_args!("the distance call is not representable: {err}"),
                );
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
    fn lower_join(&self, join: &HashJoinExec) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(why) = join_shape_refusal(
            *join.join_type(),
            join.filter().is_some(),
            join.null_equality(),
            join.on(),
        ) {
            self.skip("HashJoinExec", why);
            return Ok(None);
        }
        let (build, probe) = (join.left(), join.right());
        let (bs, ps) = (build.schema(), probe.schema());
        let (build_key, probe_key) = match join_key_columns(join.on(), &bs, &ps) {
            Ok(keys) => keys,
            Err(why) => {
                self.skip("HashJoinExec", why);
                return Ok(None);
            }
        };
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
                        self.skip(
                            "HashJoinExec",
                            format_args!("the projected field `{}` is ambiguous", field.name()),
                        );
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
        // COUNT(*) over a join reads no column either (#87): keep the probe
        // key, which the device join reads anyway, and drop it again above.
        let count_only = ours.is_empty();
        let ours = if count_only { vec![probe_key] } else { ours };
        let exec = match GpuHashJoinExec::try_new(
            Arc::clone(probe),
            Arc::clone(build),
            probe_key,
            build_key,
            self.target,
        )
        .and_then(|exec| exec.with_projection(Some(ours)))
        {
            Ok(exec) => exec,
            Err(err) => {
                self.skip("HashJoinExec", refusal(&err));
                return Ok(None);
            }
        };
        let exec: Arc<dyn ExecutionPlan> =
            Arc::new(self.instrument(exec, GpuHashJoinExec::with_telemetry));
        let lowered = if count_only {
            match self.row_count_only("HashJoinExec", exec) {
                Some(lowered) => lowered,
                None => return Ok(None),
            }
        } else {
            exec
        };
        if lowered.schema() != join.schema() {
            self.skip(
                "HashJoinExec",
                "the rewritten schema differs from the original",
            );
            return Ok(None);
        }
        Ok(Some(lowered))
    }

    /// `SortMergeJoinExec` (what DataFusion plans when hash joins are not
    /// preferred, as in every Ballista session) has the same inner-join
    /// semantics; its output order is `left ++ right`, so no side swap is
    /// needed. The `SortExec`s DataFusion inserted for the merge are dropped
    /// when they carry no `LIMIT`.
    ///
    /// The merge join's output is sorted on its keys, and by the time this
    /// rule runs DataFusion has planned what is above it on that: an ORDER BY
    /// on a key lost its sort, a merge join above reads it as sorted, a
    /// window partitioned by a key finds its partitions in it. The device
    /// join promises no order, so lowering it alone returned unsorted rows,
    /// lost rows in the merge join above, and failed the window (#88). A
    /// `SortExec` over the device join, partition by partition, restores the
    /// order: the sort the merge join paid on its inputs, paid once on its
    /// output. The lowering is kept only if the sorted join promises every
    /// ordering the merge join did, since the operators above rebuild their
    /// own properties from it when the rule swaps it in.
    fn lower_sort_merge_join(
        &self,
        join: &SortMergeJoinExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(why) = join_shape_refusal(
            join.join_type(),
            join.filter().is_some(),
            join.null_equality(),
            join.on(),
        ) {
            self.skip("SortMergeJoinExec", why);
            return Ok(None);
        }
        let (left, right) = (strip_sort(join.left()), strip_sort(join.right()));
        let (ls, rs) = (left.schema(), right.schema());
        if join.schema().fields().len() != ls.fields().len() + rs.fields().len() {
            self.skip(
                "SortMergeJoinExec",
                "the merge join's output is not its two inputs side by side",
            );
            return Ok(None);
        }
        let (lk, rk) = match join_key_columns(join.on(), &ls, &rs) {
            Ok(keys) => keys,
            Err(why) => {
                self.skip("SortMergeJoinExec", why);
                return Ok(None);
            }
        };
        let exec = match GpuHashJoinExec::try_new(left, right, lk, rk, self.target) {
            Ok(exec) => exec,
            Err(err) => {
                self.skip("SortMergeJoinExec", refusal(&err));
                return Ok(None);
            }
        };
        if exec.schema() != join.schema() {
            self.skip(
                "SortMergeJoinExec",
                "the rewritten schema differs from the original",
            );
            return Ok(None);
        }
        let exec: Arc<dyn ExecutionPlan> =
            Arc::new(self.instrument(exec, GpuHashJoinExec::with_telemetry));
        let merge = join.properties();
        let lowered: Arc<dyn ExecutionPlan> = match merge.output_ordering() {
            Some(ordering) => {
                Arc::new(SortExec::new(ordering.clone(), exec).with_preserve_partitioning(true))
            }
            None => exec,
        };
        // The sort promises its own ordering and nothing else, so a merge
        // join with a second, independent order (a `row_number()` below it,
        // say) would lose it, and an operator above that reads it fails.
        let promised = lowered.equivalence_properties();
        for ordering in merge.equivalence_properties().oeq_class().iter() {
            if !promised.ordering_satisfy(ordering.clone())? {
                self.skip(
                    "SortMergeJoinExec",
                    format_args!(
                        "the merge join's output is also ordered by `{ordering}`, an order \
                         a sort over the device join would not promise"
                    ),
                );
                return Ok(None);
            }
        }
        Ok(Some(lowered))
    }

    /// Lowers one `AggregateExec` where it stands, keeping its input.
    ///
    /// The two halves of a two-phase aggregate are lowered separately and
    /// DataFusion's exchange between them stays (#90). Each half aggregates
    /// per input partition and declares the partitioning the node it replaces
    /// declared, so the operators above — a partitioned join, the
    /// `InterleaveExec` of a `UNION ALL` — get the N hash partitions they were
    /// planned on, and Ballista still cuts its stages at the exchanges, which
    /// is what keeps a distributed plan from reading one scan partition of
    /// N. The rule used to fold the pair and the exchange into one
    /// single-partition `GpuAggregateExec` over a `CoalescePartitionsExec`.
    fn lower_aggregate(&self, agg: &AggregateExec) -> Result<Option<GpuAggregateExec>> {
        let spec = match agg.mode() {
            // Rows in: a whole aggregate, or the partial half of one.
            AggregateMode::Partial | AggregateMode::Single | AggregateMode::SinglePartitioned => {
                match try_lower_aggregate_spec(agg) {
                    Ok(spec) => spec,
                    Err(why) => {
                        self.skip("AggregateExec", why);
                        return Ok(None);
                    }
                }
            }
            // Partial results in. Lowered only over a partial half that was
            // lowered itself — the same aggregates, so the same verdict — and
            // nothing to say when it was not: that half's note says why.
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                let Some(partial) = lowered_partial(agg.input()) else {
                    return Ok(None);
                };
                match try_merge_spec(agg, partial.spec()) {
                    Ok(spec) => spec,
                    Err(why) => {
                        self.skip("AggregateExec", why);
                        return Ok(None);
                    }
                }
            }
            _ => return Ok(None),
        };
        // `GpuAggregateExec` emits each partition's groups by ascending key,
        // `NULL` last. An aggregate whose input is sorted on its key runs in an
        // ordered mode instead: it emits groups in the input's order as each
        // one ends, and DataFusion plans the operators above on that order —
        // an `ORDER BY k DESC` over it needs no sort of its own. Lowered, such
        // a query came back ascending (#89). DataFusion's ordered aggregate
        // also streams with one group open, where the kernel holds them all.
        // The test is the order promised, not the input order mode: a
        // constant key runs in the ordered mode too, but promises no order
        // (there is one group), and is lowered.
        if let Some(ordering) = agg.properties().output_ordering() {
            self.skip(
                "AggregateExec",
                format_args!(
                    "its input is sorted on the grouping key, so DataFusion aggregates it in \
                     order and the operators above rely on its output order [{ordering}]; \
                     the device kernel emits groups by ascending key with NULL last"
                ),
            );
            return Ok(None);
        }
        let exec =
            GpuAggregateExec::try_new_partitioned(Arc::clone(agg.input()), spec, self.target)?;
        match exec.with_output_schema(agg.schema()) {
            Ok(exec) => Ok(Some(
                self.instrument(exec, GpuAggregateExec::with_telemetry),
            )),
            Err(err) => {
                self.skip(
                    "AggregateExec",
                    format_args!("the output schema is not representable: {err}"),
                );
                Ok(None)
            }
        }
    }
}

/// Re-applies the `fetch` of the node a lowering replaced (#86).
///
/// DataFusion's `LimitPushdown` hands a LIMIT to the operator under it as
/// that operator's `fetch`, and deletes the limit node when the operator runs
/// in one partition, so the replaced node's `fetch` can be the only limit
/// the query has. No `Gpu*Exec` has a fetch of its own. A `LocalLimitExec`
/// makes the same cut the replaced node made — at most `fetch` rows from each
/// partition — and leaves any global limit DataFusion kept above (the
/// multi-partition case) in place; as a stock node it also crosses the
/// cluster codec without a change to the wire format.
fn keep_fetch(fetch: Option<usize>, lowered: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    match fetch {
        Some(fetch) if lowered.fetch() != Some(fetch) => {
            Arc::new(LocalLimitExec::new(lowered, fetch))
        }
        _ => lowered,
    }
}

/// A constructor's refusal as a placement note: its own words, without
/// DataFusion's `Error during planning: ` prefix.
fn refusal(err: &DataFusionError) -> String {
    match err {
        DataFusionError::Plan(reason) => reason.clone(),
        other => other.to_string(),
    }
}

/// The lowered partial half under a final aggregate, looking through the
/// nodes DataFusion puts between the halves, or `None` when that half stayed
/// on the CPU. Only nodes that pass every row and column through are looked
/// through, so what is found is the final's input as it was planned.
fn lowered_partial(input: &Arc<dyn ExecutionPlan>) -> Option<&GpuAggregateExec> {
    let mut node = input;
    loop {
        if let Some(partial) = node.downcast_ref::<GpuAggregateExec>() {
            return Some(partial);
        }
        let passes_rows = node.fetch().is_none()
            && (node.downcast_ref::<RepartitionExec>().is_some()
                || node.downcast_ref::<CoalescePartitionsExec>().is_some()
                || node.downcast_ref::<SortExec>().is_some()
                || node.downcast_ref::<SortPreservingMergeExec>().is_some());
        if !passes_rows {
            return None;
        }
        node = node.children().into_iter().next()?;
    }
}

/// The final half of a two-phase aggregate whose partial half was lowered
/// with `partial`, as an aggregate over the partial results: the key is
/// column 0 and aggregate `i`'s partial value column `1 + i` (one state field
/// each for `SUM/COUNT/MIN/MAX`). The `SUM`, `MIN` or `MAX` of partial
/// `SUM`s, `MIN`s or `MAX`s is the function again, and partial counts add up,
/// so a `COUNT` becomes a `SUM` here — on the device as on the CPU reference.
///
/// DataFusion only plans a final half over its own partial half, so the
/// check below is the cheap insurance that what was found under the
/// exchanges is that half: merging the wrong partials would not fail, it
/// would compute.
fn try_merge_spec(
    agg: &AggregateExec,
    partial: &AggregateSpec,
) -> std::result::Result<AggregateSpec, String> {
    let group = agg.group_expr();
    let merges: Vec<Option<AggregateFunction>> = agg
        .aggr_expr()
        .iter()
        .map(|e| device_function(e.fun().name()).filter(|_| !e.is_distinct()))
        .collect();
    let lowered: Vec<Option<AggregateFunction>> =
        partial.aggregates.iter().map(|(f, _)| Some(*f)).collect();
    if merges != lowered
        || !group.is_single()
        || group.expr().len() != 1
        || column_index(&group.expr()[0].0) != Some(0)
        || agg.filter_expr().iter().any(Option::is_some)
    {
        return Err(format!(
            "the final half ({}) does not merge the partial half lowered below it ({})",
            agg.aggr_expr()
                .iter()
                .map(|e| e.fun().name())
                .collect::<Vec<_>>()
                .join(", "),
            partial
                .aggregates
                .iter()
                .map(|(f, _)| f.name())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let aggregates = partial
        .aggregates
        .iter()
        .enumerate()
        .map(|(i, (func, _))| {
            let merge = match func {
                AggregateFunction::Count => AggregateFunction::Sum,
                other => *other,
            };
            (merge, 1 + i)
        })
        .collect();
    Ok(AggregateSpec {
        group_by: 0,
        aggregates,
    })
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

/// The DataFusion comparisons the device filter implements, with the kernel
/// comparison each lowers to. The lowering and the note that names what it
/// accepts both read this table, so the note cannot list an operator the
/// lowering refuses (it listed `!=`, #94).
const COMPARISONS: [(Operator, Comparison); 5] = [
    (Operator::Eq, Comparison::Eq),
    (Operator::Lt, Comparison::Lt),
    (Operator::LtEq, Comparison::LtEq),
    (Operator::Gt, Comparison::Gt),
    (Operator::GtEq, Comparison::GtEq),
];

fn comparison(op: &Operator) -> Option<Comparison> {
    COMPARISONS
        .iter()
        .find(|(implemented, _)| implemented == op)
        .map(|&(_, kernel)| kernel)
}

/// Why the device join cannot take a join of this shape, whatever its keys
/// are, or `None` when it can. Each early return of the join lowerings says
/// which one it was (#94): a skipped join otherwise looks exactly like one
/// the rule never looked at.
fn join_shape_refusal(
    join_type: JoinType,
    has_filter: bool,
    null_equality: NullEquality,
    on: &[(PhysicalExprRef, PhysicalExprRef)],
) -> Option<String> {
    if join_type != JoinType::Inner {
        return Some(format!(
            "a `{join_type}` join has no device kernel; the device join is inner only"
        ));
    }
    if has_filter {
        return Some(
            "the join has a condition besides its key equality, which the device \
             join cannot evaluate"
                .to_owned(),
        );
    }
    if null_equality != NullEquality::NullEqualsNothing {
        return Some(
            "the join matches NULL keys to each other (IS NOT DISTINCT FROM); the \
             device join never matches a NULL key"
                .to_owned(),
        );
    }
    if on.len() != 1 {
        return Some(format!(
            "the join has {} key pairs; the device join takes exactly one",
            on.len()
        ));
    }
    None
}

/// The column indices of a one-key join's left and right keys, or why the
/// device join cannot take them.
fn join_key_columns(
    on: &[(PhysicalExprRef, PhysicalExprRef)],
    left: &Schema,
    right: &Schema,
) -> std::result::Result<(usize, usize), String> {
    let [(l, r)] = on else {
        return Err(format!(
            "the join has {} key pairs; the device join takes exactly one",
            on.len()
        ));
    };
    let mut keys = [0; 2];
    for (slot, (expr, schema)) in keys.iter_mut().zip([(l, left), (r, right)]) {
        let Some(index) = column_index(expr) else {
            return Err(format!(
                "the join key `{expr}` is an expression; the device join takes plain columns"
            ));
        };
        let Some(field) = schema.fields().get(index) else {
            return Err(format!(
                "join key column {index} is out of range for its input"
            ));
        };
        if field.data_type() != &DataType::Int64 {
            return Err(format!(
                "the join key `{}` is {}; the device join takes Int64 keys",
                field.name(),
                field.data_type()
            ));
        }
        *slot = index;
    }
    Ok((keys[0], keys[1]))
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
    try_lower_predicate(expr, schema).ok()
}

/// [`lower_predicate`], keeping the reason so it can be shown (#32).
///
/// The public function stays `Option`-shaped; this is the same walk with the
/// refusal named, so the two cannot disagree about what is representable.
fn try_lower_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> std::result::Result<Predicate, String> {
    let Some(binary) = as_any(expr).downcast_ref::<BinaryExpr>() else {
        return Err(format!(
            "`{expr}` is not a binary comparison; the v1 grammar is \
             column <op> literal, conjoined with AND"
        ));
    };
    if *binary.op() == Operator::And {
        let l = try_lower_predicate(binary.left(), schema)?;
        let r = try_lower_predicate(binary.right(), schema)?;
        return Ok(Predicate::and(l, r));
    }
    let Some(op) = comparison(binary.op()) else {
        let implemented: Vec<&str> = COMPARISONS
            .iter()
            .map(|(_, kernel)| kernel.symbol())
            .collect();
        return Err(format!(
            "`{}` is not one of the comparisons the device kernels implement ({})",
            binary.op(),
            implemented.join(", ")
        ));
    };
    let (column, lit, op) = match (column_index(binary.left()), literal(binary.right())) {
        (Some(c), Some(l)) => (c, l, op),
        _ => match (literal(binary.left()), column_index(binary.right())) {
            (Some(l), Some(c)) => (c, l, flip(op)),
            _ => {
                return Err(format!(
                    "`{binary}` does not compare a column against an Int64 or \
                     Float64 literal"
                ));
            }
        },
    };
    let Some(field) = schema.fields().get(column) else {
        return Err(format!(
            "column {column} is out of range for the input schema"
        ));
    };
    if *field.data_type() != lit.data_type() {
        return Err(format!(
            "`{}` is {} and the literal is {}; the kernels compare a column \
             against a literal of its own type",
            field.name(),
            field.data_type(),
            lit.data_type()
        ));
    }
    Ok(Predicate::compare(column, op, lit))
}

/// Lowers an `AggregateExec`'s grouping and aggregates into an [`AggregateSpec`].
pub fn lower_aggregate_spec(agg: &AggregateExec) -> Option<AggregateSpec> {
    try_lower_aggregate_spec(agg).ok()
}

/// The device kernel for DataFusion's aggregate function `name`, if any.
fn device_function(name: &str) -> Option<AggregateFunction> {
    Some(match name {
        "sum" => AggregateFunction::Sum,
        "count" => AggregateFunction::Count,
        "min" => AggregateFunction::Min,
        "max" => AggregateFunction::Max,
        _ => return None,
    })
}

/// [`lower_aggregate_spec`], keeping the reason so it can be shown (#32).
fn try_lower_aggregate_spec(agg: &AggregateExec) -> std::result::Result<AggregateSpec, String> {
    let group = agg.group_expr();
    if !group.is_single() || group.expr().len() != 1 || !group.null_expr().is_empty() {
        return Err(format!(
            "the device kernel groups by exactly one column; this groups by {}",
            group.expr().len()
        ));
    }
    let input_schema = agg.input().schema();
    let Some(group_by) = column_index(&group.expr()[0].0) else {
        return Err("the grouping key is an expression, not a plain column".to_owned());
    };
    let group_field = input_schema.field(group_by);
    if group_field.data_type() != &DataType::Int64 {
        return Err(format!(
            "the grouping key `{}` is {}; the device kernel groups on Int64",
            group_field.name(),
            group_field.data_type()
        ));
    }
    if agg.filter_expr().iter().any(Option::is_some) {
        return Err("a FILTER clause on an aggregate has no device kernel".to_owned());
    }
    let mut aggregates = Vec::with_capacity(agg.aggr_expr().len());
    for expr in agg.aggr_expr() {
        let Some(func) = device_function(expr.fun().name()) else {
            return Err(format!(
                "`{}` is not one of the aggregates the device kernels \
                 implement (sum, count, min, max)",
                expr.fun().name()
            ));
        };
        if expr.is_distinct() {
            return Err(format!(
                "DISTINCT is not implemented for `{}`",
                expr.fun().name()
            ));
        }
        let args = expr.expressions();
        if args.len() != 1 {
            return Err(format!(
                "`{}` takes {} arguments; the device kernels take one",
                expr.fun().name(),
                args.len()
            ));
        }
        let Some(column) = column_index(&args[0]) else {
            return Err(format!(
                "`{}` aggregates an expression, not a plain column",
                expr.fun().name()
            ));
        };
        let field = input_schema.field(column);
        if !matches!(field.data_type(), DataType::Int64 | DataType::Float64) {
            return Err(format!(
                "`{}` is {}; the device kernels aggregate Int64 and Float64",
                field.name(),
                field.data_type()
            ));
        }
        aggregates.push((func, column));
    }
    Ok(AggregateSpec {
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
