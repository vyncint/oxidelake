//! `GpuHashJoinExec`: inner hash join on one `Int64` key per side.

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::EmissionType;
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

use super::{ExecConfig, field_type, plan_err, plan_properties};
use crate::operator::JoinBuild;

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

/// Inner hash join. The right child is the build side and is collected in
/// full once, when the first partition starts; each left (probe) batch
/// streams through. Output columns are the left columns followed by the right
/// columns, or the [`Self::with_projection`] subset of them.
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
        let schema = joined_schema(left.as_ref(), right.as_ref());
        let properties = plan_properties(
            Arc::clone(&schema),
            left.output_partitioning().clone(),
            EmissionType::Incremental,
        );
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
        let full = joined_schema(self.left.as_ref(), self.right.as_ref());
        let schema = match &projection {
            Some(indices) if indices.is_empty() => {
                return plan_err("GpuHashJoinExec: projection must select at least one column");
            }
            Some(indices) => Arc::new(full.project(indices)?),
            None => full,
        };
        self.properties = plan_properties(
            Arc::clone(&schema),
            self.left.output_partitioning().clone(),
            EmissionType::Incremental,
        );
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
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let (left_key, right_key) = (self.left_key, self.right_key);
        let projection = self.projection.clone();
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
            Ok::<_, DataFusionError>(left.map(move |batch| {
                let batch = batch?;
                let _timer = baseline.elapsed_compute().timer();
                let out = operator.hash_join(&batch, &build, left_key, projection.as_deref())?;
                baseline.record_output(out.num_rows());
                Ok(out)
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
