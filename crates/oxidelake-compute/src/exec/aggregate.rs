//! `GpuAggregateExec`: `SUM/COUNT/MIN/MAX` grouped by one `Int64` key.

use std::fmt;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::Partitioning;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream, collect,
};
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

/// Grouped aggregation. Consumes every input partition, then emits one batch
/// sorted by key (`NULL` group last) from a single output partition.
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
    /// Builds the exec, validating the key and aggregate input types.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        spec: AggregateSpec,
        target: BackendKind,
    ) -> Result<Self> {
        let schema = aggregate_output_schema(&input.schema(), &spec)?;
        let properties = plan_properties(
            Arc::clone(&schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
        );
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
        self.properties = plan_properties(
            Arc::clone(&schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
        );
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

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [input] = <[Arc<dyn ExecutionPlan>; 1]>::try_from(children).map_err(|c| {
            DataFusionError::Plan(format!("GpuAggregateExec expects 1 child, got {}", c.len()))
        })?;
        let mut exec = Self::try_new(input, self.spec.clone(), self.config.target)?
            .with_output_schema(Arc::clone(&self.schema))?;
        exec.config = self.config.clone();
        Ok(Arc::new(exec))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "GpuAggregateExec has a single output partition, got request for {partition}"
            )));
        }
        let input = Arc::clone(&self.input);
        let input_schema = input.schema();
        let operator = self.config.operator("GpuAggregateExec")?;
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let spec = self.spec.clone();
        let schema = Arc::clone(&self.schema);
        let stream = futures::stream::once(async move {
            let batches = collect(input, context).await?;
            let all = concat_batches(&input_schema, &batches)?;
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
