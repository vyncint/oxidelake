//! `GpuHashJoinExec`: inner hash join on one `Int64` key per side.

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::common::JoinType;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::equivalence::join_equivalence_properties;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionMapping;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream, collect,
};
use futures::future::{BoxFuture, FutureExt, Shared};
use futures::{StreamExt, TryStreamExt};
use oxidelake_core::BackendKind;
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_device::GpuBackend;

use super::{ExecConfig, field_type, plan_err};
use crate::operator::{GpuOperator, JoinBuild, JoinProbe};

/// The collected and hashed build side, shared by every probe partition.
/// Collecting once matters for correctness, not just cost: a shared-state
/// build child (e.g. a `RepartitionExec`) delivers each row to one consumer
/// only, so a per-partition collect would hand every probe partition a
/// different fragment of the build side. Hashing once (see [`JoinBuild`])
/// keeps the probe loop O(probe rows) instead of O(probe batches × build rows).
type SharedBuild = Shared<BoxFuture<'static, std::result::Result<Arc<JoinBuild>, SharedError>>>;
type SharedError = Arc<DataFusionError>;

/// Lazily starts the build-side collection on first use; later partitions
/// await the same future. A fresh exec (from [`ExecutionPlan::with_new_children`]
/// or [`ExecutionPlan::reset_state`]) starts empty.
#[derive(Default)]
struct BuildOnce(Mutex<Option<SharedBuild>>);

impl BuildOnce {
    fn get_or_init(&self, init: impl FnOnce() -> SharedBuild) -> SharedBuild {
        let mut slot = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        slot.get_or_insert_with(init).clone()
    }
}

impl fmt::Debug for BuildOnce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let started = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some();
        f.debug_tuple("BuildOnce").field(&started).finish()
    }
}

/// One partition's probe side on its way through the join: the probe batch
/// being emitted, piece by piece, and what the next one needs.
struct ProbeStream {
    left: SendableRecordBatchStream,
    operator: GpuOperator,
    build: Arc<JoinBuild>,
    left_key: usize,
    projection: Option<Vec<usize>>,
    batch_size: usize,
    span: tracing::Span,
    baseline: BaselineMetrics,
    probe: Option<JoinProbe>,
}

impl ProbeStream {
    /// Starts joining `batch`; its output comes from [`Self::next_piece`].
    fn start(&mut self, batch: &RecordBatch) -> Result<()> {
        let _entered = self.span.enter();
        let _timer = self.baseline.elapsed_compute().timer();
        self.probe = Some(self.operator.hash_join(
            batch,
            &self.build,
            self.left_key,
            self.projection.as_deref(),
            self.batch_size,
        )?);
        Ok(())
    }

    /// The current probe batch's next piece, or `None` when it has none left
    /// (or there is no probe batch yet).
    fn next_piece(&mut self) -> Result<Option<RecordBatch>> {
        let Some(probe) = self.probe.as_mut() else {
            return Ok(None);
        };
        let _entered = self.span.enter();
        let _timer = self.baseline.elapsed_compute().timer();
        match self.operator.next_join_chunk(probe, &self.build)? {
            Some(out) => {
                self.baseline.record_output(out.num_rows());
                Ok(Some(out))
            }
            None => {
                self.probe = None;
                Ok(None)
            }
        }
    }
}

/// Inner hash join. The right child is the build side and is collected in
/// full once, when the first partition starts; each left (probe) batch
/// streams through, its output in batches of at most the session's batch
/// size however often its keys repeat. Output columns are the left columns
/// followed by the right columns, or the [`Self::with_projection`] subset of
/// them.
#[derive(Debug)]
pub struct GpuHashJoinExec {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    left_key: usize,
    right_key: usize,
    projection: Option<Vec<usize>>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    config: ExecConfig,
    build: BuildOnce,
}

fn joined_schema(left: &dyn ExecutionPlan, right: &dyn ExecutionPlan) -> SchemaRef {
    let fields: Vec<_> = left
        .schema()
        .fields()
        .iter()
        .chain(right.schema().fields().iter())
        .cloned()
        .collect();
    Arc::new(Schema::new(fields))
}

/// The output schema and what the join promises the operators above it:
/// both inputs' equalities and constants and its two keys equal to each
/// other (an inner equi-join in which NULL matches nothing, so on every
/// output row they hold the same value), derived by DataFusion's own
/// `join_equivalence_properties` and carried through `projection`; no
/// ordering, since the probe order is not promised; and the probe side's
/// partitioning.
///
/// A parent rebuilds its own properties from these when the placement rule
/// swaps this node in for a stock join, so an equality left out here is one
/// no operator above can plan on any more. Without the key equality, a
/// window partitioned by one key over a sort on the other, which DataFusion
/// plans over a hash join and over a re-sorted merge join alike, failed with
/// an internal "All partition by columns should have an ordering" assertion
/// (#88).
fn join_properties(
    left: &Arc<dyn ExecutionPlan>,
    right: &Arc<dyn ExecutionPlan>,
    (left_key, right_key): (usize, usize),
    projection: Option<&[usize]>,
) -> Result<(SchemaRef, Arc<PlanProperties>)> {
    let full = joined_schema(left.as_ref(), right.as_ref());
    let key = |plan: &Arc<dyn ExecutionPlan>, index: usize| -> Result<Arc<dyn PhysicalExpr>> {
        let schema = plan.schema();
        let field = schema.fields().get(index).ok_or_else(|| {
            DataFusionError::Plan(format!(
                "GpuHashJoinExec: key column {index} out of range for {schema}"
            ))
        })?;
        Ok(Arc::new(Column::new(field.name(), index)))
    };
    let on = [(key(left, left_key)?, key(right, right_key)?)];
    let mut equivalences = join_equivalence_properties(
        left.equivalence_properties().clone(),
        right.equivalence_properties().clone(),
        &JoinType::Inner,
        Arc::clone(&full),
        &[false, false],
        None,
        &on,
    )?;
    let schema = match projection {
        Some(indices) => {
            let projected = Arc::new(full.project(indices)?);
            let mapping = ProjectionMapping::from_indices(indices, &full)?;
            equivalences = equivalences.project(&mapping, Arc::clone(&projected));
            projected
        }
        None => full,
    };
    let properties = PlanProperties::new(
        equivalences,
        left.output_partitioning().clone(),
        EmissionType::Incremental,
        Boundedness::Bounded,
    );
    Ok((schema, Arc::new(properties)))
}

impl GpuHashJoinExec {
    /// Builds the exec, validating that both keys are `Int64` columns.
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        left_key: usize,
        right_key: usize,
        target: BackendKind,
    ) -> Result<Self> {
        let (ls, rs) = (left.schema(), right.schema());
        for (schema, key, side) in [(&ls, left_key, "left"), (&rs, right_key, "right")] {
            let dt = field_type(schema, key, "GpuHashJoinExec")?;
            if *dt != DataType::Int64 {
                return plan_err(format!(
                    "GpuHashJoinExec: {side} key column {key} has type {dt:?}; v1 supports Int64"
                ));
            }
        }
        let (schema, properties) = join_properties(&left, &right, (left_key, right_key), None)?;
        Ok(Self {
            left,
            right,
            left_key,
            right_key,
            projection: None,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            config: ExecConfig::new(target),
            build: BuildOnce::default(),
        })
    }

    /// Selects and orders output columns by index into the joined
    /// `left ++ right` row; `None` keeps every column.
    pub fn with_projection(mut self, projection: Option<Vec<usize>>) -> Result<Self> {
        if projection.as_ref().is_some_and(Vec::is_empty) {
            return plan_err("GpuHashJoinExec: projection must select at least one column");
        }
        let (schema, properties) = join_properties(
            &self.left,
            &self.right,
            (self.left_key, self.right_key),
            projection.as_deref(),
        )?;
        self.properties = properties;
        self.schema = schema;
        self.projection = projection;
        Ok(self)
    }

    /// Runs on an explicit backend instead of the process-local one.
    pub fn with_backend(mut self, backend: Arc<dyn GpuBackend>) -> Self {
        self.config.backend = Some(backend);
        self
    }

    /// Reports per-batch statistics into `telemetry`.
    pub fn with_telemetry(mut self, telemetry: Arc<TelemetryHub>) -> Self {
        self.config.telemetry = Some(telemetry);
        self
    }

    /// Probe-side key column index.
    pub fn left_key(&self) -> usize {
        self.left_key
    }

    /// Build-side key column index.
    pub fn right_key(&self) -> usize {
        self.right_key
    }

    /// The output projection over `left ++ right`, if any.
    pub fn projection(&self) -> Option<&[usize]> {
        self.projection.as_deref()
    }

    /// The backend this node was planned for.
    pub fn target(&self) -> BackendKind {
        self.config.target
    }

    /// The probe (left) input.
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// The build (right) input.
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }
}

impl DisplayAs for GpuHashJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = |schema: SchemaRef, i: usize| {
            schema
                .fields()
                .get(i)
                .map_or_else(|| format!("col{i}"), |fl| fl.name().clone())
        };
        let on = format!(
            "{} = {}",
            name(self.left.schema(), self.left_key),
            name(self.right.schema(), self.right_key)
        );
        let projection = match &self.projection {
            Some(_) => format!(
                ", projection=[{}]",
                self.schema
                    .fields()
                    .iter()
                    .map(|fl| fl.name().clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            None => String::new(),
        };
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "GpuHashJoinExec[{}]: join_type=Inner, on=[{on}]{projection}",
                self.config.target
            ),
            DisplayFormatType::TreeRender => write!(f, "backend={}\non={on}", self.config.target),
        }
    }
}

impl ExecutionPlan for GpuHashJoinExec {
    fn name(&self) -> &str {
        "GpuHashJoinExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    // Shallow by contract, and there is nothing to visit: this node's
    // predicate, keys and aggregates are OxideLake's own plan vocabulary
    // (`oxidelake_core::params`), not DataFusion `PhysicalExpr`s, and they are
    // evaluated by the operator layer rather than by expression evaluation.
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [left, right] = <[Arc<dyn ExecutionPlan>; 2]>::try_from(children).map_err(|c| {
            DataFusionError::Plan(format!(
                "GpuHashJoinExec expects 2 children, got {}",
                c.len()
            ))
        })?;
        let mut exec = Self::try_new(
            left,
            right,
            self.left_key,
            self.right_key,
            self.config.target,
        )?
        .with_projection(self.projection.clone())?;
        exec.config = self.config.clone();
        Ok(Arc::new(exec))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let left = self.left.execute(partition, Arc::clone(&context))?;
        let operator = self.config.operator("GpuHashJoinExec")?;
        let span = self
            .config
            .span("GpuHashJoinExec", partition, operator.backend().kind());
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let (left_key, right_key) = (self.left_key, self.right_key);
        let projection = self.projection.clone();
        let batch_size = context.session_config().batch_size();
        let build_fut = self.build.get_or_init(|| {
            let right = Arc::clone(&self.right);
            let right_schema = right.schema();
            async move {
                let batches = collect(right, context).await.map_err(Arc::new)?;
                let build = concat_batches(&right_schema, &batches)
                    .map_err(|e| Arc::new(DataFusionError::from(e)))?;
                JoinBuild::new(build, right_key)
                    .map(Arc::new)
                    .map_err(|e| Arc::new(DataFusionError::from(e)))
            }
            .boxed()
            .shared()
        });
        let stream = futures::stream::once(async move {
            let build = build_fut.await.map_err(DataFusionError::Shared)?;
            let state = ProbeStream {
                left,
                operator,
                build,
                left_key,
                projection,
                batch_size,
                span,
                baseline,
                probe: None,
            };
            // One piece per poll, so a probe batch whose matches outnumber
            // the batch size is never held whole (#95).
            Ok::<_, DataFusionError>(futures::stream::try_unfold(state, |mut state| async move {
                loop {
                    if let Some(out) = state.next_piece()? {
                        return Ok(Some((out, state)));
                    }
                    match state.left.next().await {
                        Some(batch) => state.start(&batch?)?,
                        None => return Ok(None),
                    }
                }
            }))
        })
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
