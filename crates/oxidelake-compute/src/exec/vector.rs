//! `GpuVectorDistanceExec`: appends an L2 or cosine distance column.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
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
use oxidelake_core::params::{DistanceMetric, vector_dimension};
use oxidelake_core::telemetry::TelemetryHub;
use oxidelake_device::GpuBackend;

use super::{ExecConfig, field_type, plan_err, plan_properties};

/// Distance between a query vector and a `FixedSizeList<Float32>` column,
/// appended as a nullable `Float32` column.
#[derive(Debug)]
pub struct GpuVectorDistanceExec {
    input: Arc<dyn ExecutionPlan>,
    column: usize,
    query: Arc<Vec<f32>>,
    metric: DistanceMetric,
    output_name: String,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    config: ExecConfig,
}

impl GpuVectorDistanceExec {
    /// Builds the exec, validating the vector column and query dimension.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        column: usize,
        query: Vec<f32>,
        metric: DistanceMetric,
        output_name: impl Into<String>,
        target: BackendKind,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let dt = field_type(&input_schema, column, "GpuVectorDistanceExec")?;
        let dim = vector_dimension(dt)
            .ok_or_else(|| datafusion::error::DataFusionError::Plan(format!(
                "GpuVectorDistanceExec: column {column} has type {dt:?}; expected FixedSizeList<Float32>"
            )))?;
        if query.len() != dim {
            return plan_err(format!(
                "GpuVectorDistanceExec: query has {} dimensions but the column has {dim}",
                query.len()
            ));
        }
        let output_name = output_name.into();
        let mut fields: Vec<_> = input_schema.fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(&output_name, DataType::Float32, true)));
        let schema = Arc::new(Schema::new(fields));
        let properties = plan_properties(
            Arc::clone(&schema),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
        );
        Ok(Self {
            input,
            column,
            query: Arc::new(query),
            metric,
            output_name,
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

    /// The vector column index.
    pub fn column(&self) -> usize {
        self.column
    }

    /// The query vector.
    pub fn query(&self) -> &[f32] {
        &self.query
    }

    /// The metric.
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// The appended column's name.
    pub fn output_name(&self) -> &str {
        &self.output_name
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

impl DisplayAs for GpuVectorDistanceExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self
            .input
            .schema()
            .fields()
            .get(self.column)
            .map_or_else(|| format!("col{}", self.column), |fl| fl.name().clone());
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "GpuVectorDistanceExec[{}]: {}({}) AS {}, dim={}",
                self.config.target,
                self.metric.name(),
                name,
                self.output_name,
                self.query.len()
            ),
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "backend={}\nmetric={}\ncolumn={name}",
                    self.config.target,
                    self.metric.name()
                )
            }
        }
    }
}

impl ExecutionPlan for GpuVectorDistanceExec {
    fn name(&self) -> &str {
        "GpuVectorDistanceExec"
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
                "GpuVectorDistanceExec expects 1 child, got {}",
                c.len()
            ))
        })?;
        let mut exec = Self::try_new(
            input,
            self.column,
            self.query.as_ref().clone(),
            self.metric,
            self.output_name.clone(),
            self.config.target,
        )?;
        exec.config = self.config.clone();
        Ok(Arc::new(exec))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let operator = self.config.operator("GpuVectorDistanceExec")?;
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let column = self.column;
        let query = Arc::clone(&self.query);
        let metric = self.metric;
        let output_name = self.output_name.clone();
        let stream = input.map(move |batch| {
            let batch = batch?;
            let _timer = baseline.elapsed_compute().timer();
            let out = operator.vector_distance(&batch, column, &query, metric, &output_name)?;
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
