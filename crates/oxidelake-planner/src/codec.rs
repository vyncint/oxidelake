//! `OxidePhysicalCodec`: (de)serializes `Gpu*Exec` nodes for Ballista.
//!
//! Payload layout: `b"OXGP"` magic, one version byte, then a `postcard`
//! encoding of [`GpuNode`]. Children are encoded by DataFusion itself and
//! arrive as `inputs` on decode. Anything that is not a `Gpu*Exec` is delegated
//! to the wrapped codec (Ballista's own codec in cluster mode, DataFusion's
//! default otherwise), so shuffle nodes keep working.

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::physical_plan::{DefaultPhysicalExtensionCodec, PhysicalExtensionCodec};
use oxidelake_compute::{
    AggregateSpec, DistanceMetric, GpuAggregateExec, GpuFilterExec, GpuHashJoinExec,
    GpuVectorDistanceExec, Predicate,
};
use oxidelake_core::BackendKind;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"OXGP";
const VERSION: u8 = 1;

/// Serialized form of one `Gpu*Exec` node (children excluded).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GpuNode {
    /// `GpuFilterExec`
    Filter {
        /// Planned backend tag.
        target: BackendKind,
        /// Predicate.
        predicate: Predicate,
        /// Output column indices.
        projection: Vec<usize>,
    },
    /// `GpuHashJoinExec`
    HashJoin {
        /// Planned backend tag.
        target: BackendKind,
        /// Probe-side key.
        left_key: usize,
        /// Build-side key.
        right_key: usize,
        /// Output projection over `left ++ right`.
        projection: Option<Vec<usize>>,
    },
    /// `GpuAggregateExec`
    Aggregate {
        /// Planned backend tag.
        target: BackendKind,
        /// Grouping and aggregates.
        spec: AggregateSpec,
        /// Output field names and nullability, as planned.
        output: Vec<(String, bool)>,
    },
    /// `GpuVectorDistanceExec`
    VectorDistance {
        /// Planned backend tag.
        target: BackendKind,
        /// Vector column index.
        column: usize,
        /// Query vector.
        query: Vec<f32>,
        /// Metric.
        metric: DistanceMetric,
        /// Output column name.
        output_name: String,
    },
}

/// The codec.
pub struct OxidePhysicalCodec {
    inner: Arc<dyn PhysicalExtensionCodec>,
}

impl Default for OxidePhysicalCodec {
    fn default() -> Self {
        Self::new(Arc::new(DefaultPhysicalExtensionCodec {}))
    }
}

impl fmt::Debug for OxidePhysicalCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OxidePhysicalCodec")
            .field("inner", &self.inner)
            .finish()
    }
}

fn child<'a>(
    inputs: &'a [Arc<dyn ExecutionPlan>],
    index: usize,
    what: &str,
) -> Result<&'a Arc<dyn ExecutionPlan>> {
    inputs.get(index).ok_or_else(|| {
        DataFusionError::Internal(format!(
            "{what}: missing child {index} (got {})",
            inputs.len()
        ))
    })
}

impl OxidePhysicalCodec {
    /// Wraps `inner`, which handles every non-OxideLake node.
    pub fn new(inner: Arc<dyn PhysicalExtensionCodec>) -> Self {
        Self { inner }
    }

    /// Serialized form of `node`, or `None` if it is not a `Gpu*Exec`.
    pub fn describe(node: &dyn ExecutionPlan) -> Option<GpuNode> {
        if let Some(f) = node.downcast_ref::<GpuFilterExec>() {
            return Some(GpuNode::Filter {
                target: f.target(),
                predicate: f.predicate().clone(),
                projection: f.projection().to_vec(),
            });
        }
        if let Some(j) = node.downcast_ref::<GpuHashJoinExec>() {
            return Some(GpuNode::HashJoin {
                target: j.target(),
                left_key: j.left_key(),
                right_key: j.right_key(),
                projection: j.projection().map(<[usize]>::to_vec),
            });
        }
        if let Some(a) = node.downcast_ref::<GpuAggregateExec>() {
            return Some(GpuNode::Aggregate {
                target: a.target(),
                spec: a.spec().clone(),
                output: a
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| (f.name().clone(), f.is_nullable()))
                    .collect(),
            });
        }
        if let Some(v) = node.downcast_ref::<GpuVectorDistanceExec>() {
            return Some(GpuNode::VectorDistance {
                target: v.target(),
                column: v.column(),
                query: v.query().to_vec(),
                metric: v.metric(),
                output_name: v.output_name().to_owned(),
            });
        }
        None
    }

    /// Rebuilds a node from its serialized form and children.
    pub fn build(
        node: GpuNode,
        inputs: &[Arc<dyn ExecutionPlan>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(match node {
            GpuNode::Filter {
                target,
                predicate,
                projection,
            } => Arc::new(GpuFilterExec::try_new(
                Arc::clone(child(inputs, 0, "GpuFilterExec")?),
                predicate,
                projection,
                target,
            )?),
            GpuNode::HashJoin {
                target,
                left_key,
                right_key,
                projection,
            } => Arc::new(
                GpuHashJoinExec::try_new(
                    Arc::clone(child(inputs, 0, "GpuHashJoinExec")?),
                    Arc::clone(child(inputs, 1, "GpuHashJoinExec")?),
                    left_key,
                    right_key,
                    target,
                )?
                .with_projection(projection)?,
            ),
            GpuNode::Aggregate {
                target,
                spec,
                output,
            } => {
                let exec = GpuAggregateExec::try_new(
                    Arc::clone(child(inputs, 0, "GpuAggregateExec")?),
                    spec,
                    target,
                )?;
                if output.len() != exec.schema().fields().len() {
                    return Err(DataFusionError::Internal(format!(
                        "GpuAggregateExec: planned {} output fields, spec yields {}",
                        output.len(),
                        exec.schema().fields().len()
                    )));
                }
                let fields: Vec<Field> = exec
                    .schema()
                    .fields()
                    .iter()
                    .zip(output)
                    .map(|(f, (name, nullable))| Field::new(name, f.data_type().clone(), nullable))
                    .collect();
                Arc::new(exec.with_output_schema(Arc::new(Schema::new(fields)))?)
            }
            GpuNode::VectorDistance {
                target,
                column,
                query,
                metric,
                output_name,
            } => Arc::new(GpuVectorDistanceExec::try_new(
                Arc::clone(child(inputs, 0, "GpuVectorDistanceExec")?),
                column,
                query,
                metric,
                output_name,
                target,
            )?),
        })
    }
}

impl PhysicalExtensionCodec for OxidePhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if buf.len() > MAGIC.len() + 1 && &buf[..MAGIC.len()] == MAGIC {
            let version = buf[MAGIC.len()];
            if version != VERSION {
                return Err(DataFusionError::Internal(format!(
                    "OxidePhysicalCodec: unsupported payload version {version} (expected {VERSION})"
                )));
            }
            let node: GpuNode = postcard::from_bytes(&buf[MAGIC.len() + 1..]).map_err(|e| {
                DataFusionError::Internal(format!("OxidePhysicalCodec: corrupt payload: {e}"))
            })?;
            return Self::build(node, inputs);
        }
        self.inner.try_decode(buf, inputs, ctx)
    }

    fn try_encode(&self, node: Arc<dyn ExecutionPlan>, buf: &mut Vec<u8>) -> Result<()> {
        match Self::describe(node.as_ref()) {
            Some(gpu) => {
                buf.extend_from_slice(MAGIC);
                buf.push(VERSION);
                let payload = postcard::to_allocvec(&gpu).map_err(|e| {
                    DataFusionError::Internal(format!("OxidePhysicalCodec: encode failed: {e}"))
                })?;
                buf.extend_from_slice(&payload);
                Ok(())
            }
            None => self.inner.try_encode(node, buf),
        }
    }
}
