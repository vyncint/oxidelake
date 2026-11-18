//! `GpuAggregateExec`: `SUM/COUNT/MIN/MAX` grouped by one `Int64` key.

use std::fmt;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionMapping;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::TryStreamExt;
use oxidelake_core::BackendKind;
use oxidelake_core::params::{AggregateFunction, AggregateSpec};
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_device::GpuBackend;

use super::{ExecConfig, field_type, plan_err, plan_properties};
use crate::display;

/// Output schema of an aggregation: the `Int64` key followed by one column per
/// aggregate, named `FUNC(column)`. `SUM(Int64)` is `Int64`, `SUM(Float64)` is
/// `Float64`, `COUNT` is `Int64`, `MIN`/`MAX` keep the input type; all nullable.
pub fn aggregate_output_schema(input: &Schema, spec: &AggregateSpec) -> Result<SchemaRef> {
    let key_type = field_type(input, spec.group_by, "GpuAggregateExec group key")?;
    if *key_type != DataType::Int64 {
        return plan_err(format!(
            "GpuAggregateExec: group key column {} has type {key_type:?}; v1 supports Int64",
            spec.group_by
        ));
    }
    let mut fields = vec![Field::new(
        input.field(spec.group_by).name(),
        DataType::Int64,
        true,
    )];
    for (func, idx) in &spec.aggregates {
        let dt = field_type(input, *idx, "GpuAggregateExec aggregate input")?;
        if !matches!(dt, DataType::Int64 | DataType::Float64) {
            return plan_err(format!(
                "GpuAggregateExec: aggregate input column {idx} has type {dt:?}; v1 supports Int64 and Float64"
            ));
        }
        let out_type = match func {
            AggregateFunction::Count => DataType::Int64,
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => dt.clone(),
        };
        fields.push(Field::new(
            format!("{}({})", func.name(), input.field(*idx).name()),
            out_type,
            true,
        ));
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// Grouped aggregation, one input partition at a time.
///
/// Each input partition is aggregated on its own and becomes the output
/// partition with the same number: one batch, sorted by key with the `NULL`
/// group last, emitted when that partition's input ends. That is a complete
/// aggregate when the input is a single partition or is hash-partitioned on
/// the key, and otherwise the partial half of a two-phase aggregate (#90).
/// [`Self::try_new`] builds a complete aggregate of any input;
/// [`Self::try_new_partitioned`] the per-partition one.
#[derive(Debug)]
pub struct GpuAggregateExec {
    input: Arc<dyn ExecutionPlan>,
    spec: AggregateSpec,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    config: ExecConfig,
}

impl GpuAggregateExec {
    /// A complete aggregate of `input`, validating the key and aggregate
    /// input types. A multi-partition input is merged first, by a
    /// `CoalescePartitionsExec` that [`Self::input`] then returns: the merge
    /// is a plan node rather than something `execute` does, so a Ballista
    /// stage boundary lands on it as on any other exchange.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        spec: AggregateSpec,
        target: BackendKind,
    ) -> Result<Self> {
        let input = if input.output_partitioning().partition_count() > 1 {
            Arc::new(CoalescePartitionsExec::new(input)) as Arc<dyn ExecutionPlan>
        } else {
            input
        };
        Self::try_new_partitioned(input, spec, target)
    }

    /// Aggregates each input partition independently, validating the key and
    /// aggregate input types (see the type's documentation for when that is
    /// a complete aggregate).
    pub fn try_new_partitioned(
        input: Arc<dyn ExecutionPlan>,
        spec: AggregateSpec,
        target: BackendKind,
    ) -> Result<Self> {
        let schema = aggregate_output_schema(&input.schema(), &spec)?;
        let properties = properties(&input, spec.group_by, &schema)?;
        Ok(Self {
            input,
            spec,
            schema,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
            config: ExecConfig::new(target),
        })
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

    /// Uses the planner's output schema (field names and nullability) instead
    /// of the default `FUNC(column)` names, so a rewritten `AggregateExec`
    /// keeps its exact schema. Types must match the spec's output types.
    pub fn with_output_schema(mut self, schema: SchemaRef) -> Result<Self> {
        if schema.fields().len() != self.schema.fields().len() {
            return plan_err(format!(
                "GpuAggregateExec: planned schema has {} fields, the spec yields {}",
                schema.fields().len(),
                self.schema.fields().len()
            ));
        }
        for (planned, ours) in schema.fields().iter().zip(self.schema.fields()) {
            if planned.data_type() != ours.data_type() {
                return plan_err(format!(
                    "GpuAggregateExec: planned field {} has type {:?}, the spec yields {:?}",
                    planned.name(),
                    planned.data_type(),
                    ours.data_type()
                ));
            }
        }
        self.properties = properties(&self.input, self.spec.group_by, &schema)?;
        self.schema = schema;
        Ok(self)
    }

    /// The aggregation spec.
    pub fn spec(&self) -> &AggregateSpec {
        &self.spec
    }

    /// The backend this node was planned for.
    pub fn target(&self) -> BackendKind {
        self.config.target
    }

    /// The input plan.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

/// The input's partitioning with the key renamed to the output column it
/// becomes, which is what DataFusion's `AggregateExec` declares for the same
/// input: an aggregate the placement rule substitutes hands the operators
/// above it the partitioning they were planned on.
fn properties(
    input: &Arc<dyn ExecutionPlan>,
    group_by: usize,
    schema: &SchemaRef,
) -> Result<Arc<PlanProperties>> {
    let input_schema = input.schema();
    let key: Arc<dyn PhysicalExpr> =
        Arc::new(Column::new(field_name(&input_schema, group_by)?, group_by));
    let mapping =
        ProjectionMapping::try_new([(key, field_name(schema, 0)?.to_owned())], &input_schema)?;
    let partitioning = input
        .output_partitioning()
        .project(&mapping, input.equivalence_properties());
    Ok(plan_properties(
        Arc::clone(schema),
        partitioning,
        EmissionType::Final,
    ))
}

fn field_name(schema: &Schema, index: usize) -> Result<&str> {
    schema
        .fields()
        .get(index)
        .map(|f| f.name().as_str())
        .ok_or_else(|| {
            DataFusionError::Plan(format!(
                "GpuAggregateExec: column index {index} out of range for {schema}"
            ))
        })
}

impl DisplayAs for GpuAggregateExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = display::aggregate(&self.spec, &self.input.schema());
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "GpuAggregateExec[{}]: {detail}", self.config.target)
            }
            DisplayFormatType::TreeRender => write!(f, "backend={}\n{detail}", self.config.target),
        }
    }
}

impl ExecutionPlan for GpuAggregateExec {
    fn name(&self) -> &str {
        "GpuAggregateExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
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

    // Per-partition semantics whatever built this node: `try_new`'s merge,
    // when there was one, is already the child being replaced.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [input] = <[Arc<dyn ExecutionPlan>; 1]>::try_from(children).map_err(|c| {
            DataFusionError::Plan(format!("GpuAggregateExec expects 1 child, got {}", c.len()))
        })?;
        let mut exec = Self::try_new_partitioned(input, self.spec.clone(), self.config.target)?
            .with_output_schema(Arc::clone(&self.schema))?;
        exec.config = self.config.clone();
        Ok(Arc::new(exec))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let input_schema = self.input.schema();
        let operator = self.config.operator("GpuAggregateExec")?;
        let span = self
            .config
            .span("GpuAggregateExec", partition, operator.backend().kind());
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let spec = self.spec.clone();
        let schema = Arc::clone(&self.schema);
        let stream = futures::stream::once(async move {
            let batches: Vec<RecordBatch> = input.try_collect().await?;
            let all = concat_batches(&input_schema, &batches)?;
            // Entered after the input is collected: the await is above it.
            let _entered = span.enter();
            let _timer = baseline.elapsed_compute().timer();
            let out = operator.aggregate(&all, &spec)?;
            let out = RecordBatch::try_new(Arc::clone(&schema), out.columns().to_vec())?;
            baseline.record_output(out.num_rows());
            Ok::<_, DataFusionError>(out)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}
