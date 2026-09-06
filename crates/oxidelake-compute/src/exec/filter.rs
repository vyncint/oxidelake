//! `GpuFilterExec`: fused filter + projection.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use futures::StreamExt;
use oxidelake_core::BackendKind;
use oxidelake_core::params::Predicate;
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_device::GpuBackend;

use super::{ExecConfig, plan_err, plan_properties, validate_predicate};
use crate::display;

/// Fused filter + projection over the bounded v1 predicate grammar.
#[derive(Debug)]
pub struct GpuFilterExec {
    input: Arc<dyn ExecutionPlan>,
    predicate: Predicate,
    projection: Vec<usize>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    config: ExecConfig,
}

impl GpuFilterExec {
    /// Builds the exec, validating the predicate and projection against the input schema.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        predicate: Predicate,
        projection: Vec<usize>,
        target: BackendKind,
    ) -> Result<Self> {
        let input_schema = input.schema();
        validate_predicate(&input_schema, &predicate)?;
        if projection.is_empty() {
            return plan_err("GpuFilterExec: projection must select at least one column");
        }
        let schema = Arc::new(input_schema.project(&projection)?);
        let properties = plan_properties(
            Arc::clone(&schema),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
        );
        Ok(Self {
            input,
            predicate,
            projection,
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

    /// The predicate.
    pub fn predicate(&self) -> &Predicate {
        &self.predicate
    }

    /// The projected input column indices.
    pub fn projection(&self) -> &[usize] {
        &self.projection
    }

    /// The backend this node was planned for.
    pub fn target(&self) -> BackendKind {
        self.config.target
    }

    /// The input plan.
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    fn rebuild(&self, input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        let mut exec = Self::try_new(
            input,
            self.predicate.clone(),
            self.projection.clone(),
            self.config.target,
        )?;
        exec.config = self.config.clone();
        Ok(exec)
    }
}

impl DisplayAs for GpuFilterExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let input_schema = self.input.schema();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "GpuFilterExec[{}]: {}, projection=[{}]",
                self.config.target,
                display::predicate(&self.predicate, &input_schema),
                display::projection(&self.projection, &input_schema)
            ),
            DisplayFormatType::TreeRender => write!(
                f,
                "backend={}\npredicate={}",
                self.config.target,
                display::predicate(&self.predicate, &input_schema)
            ),
        }
    }
}

impl ExecutionPlan for GpuFilterExec {
    fn name(&self) -> &str {
        "GpuFilterExec"
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
            datafusion::error::DataFusionError::Plan(format!(
                "GpuFilterExec expects 1 child, got {}",
                c.len()
            ))
        })?;
        Ok(Arc::new(self.rebuild(input)?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let operator = self.config.operator("GpuFilterExec")?;
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let predicate = self.predicate.clone();
        let projection = self.projection.clone();
        let stream = input.map(move |batch| {
            let batch = batch?;
            let _timer = baseline.elapsed_compute().timer();
            let out = operator.filter_project(&batch, &predicate, &projection)?;
            baseline.record_output(out.num_rows());
            Ok(out)
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
