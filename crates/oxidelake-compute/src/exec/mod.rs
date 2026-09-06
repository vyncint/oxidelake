//! The four `Gpu*Exec` DataFusion `ExecutionPlan`s and their shared plumbing.

mod aggregate;
mod filter;
mod join;
mod vector;

use std::sync::{Arc, OnceLock};

use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::PlanProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use oxidelake_core::BackendKind;
use oxidelake_core::params::{Literal, Predicate};
use oxidelake_core::telemetry::{OperatorStats, TelemetryHub};
use oxidelake_device::GpuBackend;

pub use aggregate::{GpuAggregateExec, aggregate_output_schema};
pub use filter::GpuFilterExec;
pub use join::GpuHashJoinExec;
pub use vector::GpuVectorDistanceExec;

use crate::backend::local_backend;
use crate::operator::GpuOperator;

/// Most comparison leaves a predicate may carry. Plans arrive over the network
/// in cluster mode; bounding the tree keeps a hostile or runaway payload from
/// exhausting the stack in the recursive evaluators.
pub const MAX_PREDICATE_LEAVES: usize = 64;

/// Settings shared by every exec: the planned target (the `EXPLAIN` tag), an
/// optional explicit backend (tests, embedded sessions) and optional telemetry.
#[derive(Debug, Clone)]
pub(crate) struct ExecConfig {
    pub target: BackendKind,
    pub backend: Option<Arc<dyn GpuBackend>>,
    pub telemetry: Option<Arc<TelemetryHub>>,
    /// The hub registration for this exec instance, made on first execution
    /// and shared by every partition (and by nodes rebuilt from this one), so
    /// a query registers one entry per operator rather than one per partition.
    stats: OnceLock<Arc<OperatorStats>>,
}

impl ExecConfig {
    pub(crate) fn new(target: BackendKind) -> Self {
        Self {
            target,
            backend: None,
            telemetry: None,
            stats: OnceLock::new(),
        }
    }

    /// Builds the per-partition operator: explicit backend if set, otherwise
    /// the process-local one chosen by the hardware detector.
    pub(crate) fn operator(&self, name: &str) -> Result<GpuOperator> {
        let backend = match &self.backend {
            Some(b) => Arc::clone(b),
            None => local_backend()?,
        };
        let stats: Option<Arc<OperatorStats>> = self.telemetry.as_ref().map(|hub| {
            Arc::clone(
                self.stats
                    .get_or_init(|| hub.register_operator(name, self.target)),
            )
        });
        Ok(GpuOperator::new(backend, stats)?)
    }
}

pub(crate) fn plan_properties(
    schema: SchemaRef,
    partitioning: Partitioning,
    emission: EmissionType,
) -> Arc<PlanProperties> {
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(schema),
        partitioning,
        emission,
        Boundedness::Bounded,
    ))
}

pub(crate) fn plan_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(DataFusionError::Plan(msg.into()))
}

pub(crate) fn field_type<'a>(schema: &'a Schema, index: usize, what: &str) -> Result<&'a DataType> {
    schema
        .fields()
        .get(index)
        .map(|f| f.data_type())
        .ok_or_else(|| {
            DataFusionError::Plan(format!(
                "{what}: column index {index} out of range for {schema}"
            ))
        })
}

/// Validates that a predicate only references existing columns whose type
/// matches its literal (Int64 or Float64) and stays within
/// [`MAX_PREDICATE_LEAVES`].
pub(crate) fn validate_predicate(schema: &Schema, predicate: &Predicate) -> Result<()> {
    let leaves = predicate.leaf_count();
    if leaves > MAX_PREDICATE_LEAVES {
        return plan_err(format!(
            "GpuFilterExec predicate has {leaves} comparisons; at most {MAX_PREDICATE_LEAVES} are supported"
        ));
    }
    validate_predicate_leaves(schema, predicate)
}

fn validate_predicate_leaves(schema: &Schema, predicate: &Predicate) -> Result<()> {
    match predicate {
        Predicate::Compare {
            column, literal, ..
        } => {
            let dt = field_type(schema, *column, "GpuFilterExec predicate")?;
            let expected = match literal {
                Literal::Int64(_) => DataType::Int64,
                Literal::Float64(_) => DataType::Float64,
            };
            if *dt != expected {
                return plan_err(format!(
                    "GpuFilterExec predicate: column {column} has type {dt:?} but the literal is {expected:?}"
                ));
            }
            Ok(())
        }
        Predicate::And(l, r) => {
            validate_predicate_leaves(schema, l)?;
            validate_predicate_leaves(schema, r)
        }
    }
}
