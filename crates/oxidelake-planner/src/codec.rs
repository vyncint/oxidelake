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
use datafusion_proto::physical_plan::{
    DefaultPhysicalExtensionCodec, PhysicalExtensionCodec, PhysicalProtoConverterExtension,
};
use oxidelake_compute::{
    AggregateSpec, DistanceMetric, GpuAggregateExec, GpuFilterExec, GpuHashJoinExec,
    GpuVectorDistanceExec, Predicate,
};
use oxidelake_core::BackendKind;
use oxidelake_core::telemetry::TelemetryHub;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 4] = b"OXGP";
/// Bumped to 2 by the fingerprint: the header grew four bytes (#42).
const VERSION: u8 = 2;

/// What the two ends of a plan must agree on, beyond the version byte.
///
/// `postcard` is not self-describing. A field added to or reordered within
/// `GpuNode` decodes into a *different* field of the same width without
/// complaint, so a mixed-build fleet does not fail — it computes the wrong
/// answer from plausible-looking parameters. The version byte only helps if
/// somebody remembers to bump it; this is derived, so it changes whether or
/// not anybody remembered.
///
/// It folds in the crate version (every release is a new fingerprint, which
/// is blunt but never wrong) and the compute layer's plan-changing features
/// ([`oxidelake_compute::FEATURE_MASK`]: a planner with `predict` emits plans
/// referencing a UDF an executor without it cannot resolve).
///
/// A cluster compares it twice: a scheduler refuses a worker whose
/// fingerprint is not its own when the worker registers (#92), so the
/// mismatch is a deployment error with a name rather than a failed query, and
/// this codec refuses a payload from another build when it decodes one.
pub const PLAN_FINGERPRINT: u32 = fingerprint();

const fn fnv1a(mut hash: u32, bytes: &[u8]) -> u32 {
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        i += 1;
    }
    hash
}

const fn fingerprint() -> u32 {
    let h = fnv1a(0x811c_9dc5, env!("CARGO_PKG_VERSION").as_bytes());
    let h = fnv1a(h, &[VERSION]);
    fnv1a(h, &oxidelake_compute::FEATURE_MASK.to_le_bytes())
}

/// `MAGIC` + version byte + fingerprint.
const HEADER: usize = MAGIC.len() + 1 + 4;

/// This build in the terms its fingerprint is made of, for a message that
/// has to say which of two builds is which: `fingerprint 0x… (oxidelake X,
/// plan codec vN, feature mask 0x…)`.
pub fn build_description() -> String {
    format!(
        "fingerprint {PLAN_FINGERPRINT:#010x} (oxidelake {}, plan codec v{VERSION}, \
         feature mask {:#x})",
        env!("CARGO_PKG_VERSION"),
        oxidelake_compute::FEATURE_MASK
    )
}

/// Refuses a plan this build cannot be trusted to read, before `postcard`
/// turns it into confident nonsense (#42).
///
/// Different fingerprints mean a different crate version or a different
/// feature set at the other end, and either can change what the bytes after
/// the header mean. Extracted so the refusal is testable without a
/// `TaskContext`: the failure it guards against is the one nobody sets up by
/// accident.
fn check_fingerprint(tag: &[u8]) -> Result<()> {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(tag);
    let theirs = u32::from_le_bytes(bytes);
    if theirs == PLAN_FINGERPRINT {
        return Ok(());
    }
    Err(DataFusionError::Internal(format!(
        "OxidePhysicalCodec: plan was encoded by a build with fingerprint {theirs:#010x}; \
         this build is {}. The two ends of the cluster are not the same version, or \
         disagree on `predict`. The plan is refused rather than decoded into parameters \
         that would only look right.",
        build_description()
    )))
}

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
    ///
    /// Every node is attached to the process-wide [`TelemetryHub`]. A cluster
    /// executor never runs the placement rule — the plan arrives here already
    /// lowered — so this is the only place its operators can be given
    /// somewhere to report, and without it a worker's `/metrics` would
    /// describe no work at all (#33).
    pub fn build(
        node: GpuNode,
        inputs: &[Arc<dyn ExecutionPlan>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let hub = TelemetryHub::global();
        Ok(match node {
            GpuNode::Filter {
                target,
                predicate,
                projection,
            } => Arc::new(
                GpuFilterExec::try_new(
                    Arc::clone(child(inputs, 0, "GpuFilterExec")?),
                    predicate,
                    projection,
                    target,
                )?
                .with_telemetry(Arc::clone(hub)),
            ),
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
                .with_projection(projection)?
                .with_telemetry(Arc::clone(hub)),
            ),
            GpuNode::Aggregate {
                target,
                spec,
                output,
            } => {
                // Per partition, never `try_new`: the decoded child is the
                // plan as encoded, merge included when there was one, and a
                // second merge would turn a stage's N tasks into one.
                let exec = GpuAggregateExec::try_new_partitioned(
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
                Arc::new(
                    exec.with_output_schema(Arc::new(Schema::new(fields)))?
                        .with_telemetry(Arc::clone(hub)),
                )
            }
            GpuNode::VectorDistance {
                target,
                column,
                query,
                metric,
                output_name,
            } => Arc::new(
                GpuVectorDistanceExec::try_new(
                    Arc::clone(child(inputs, 0, "GpuVectorDistanceExec")?),
                    column,
                    query,
                    metric,
                    output_name,
                    target,
                )?
                .with_telemetry(Arc::clone(hub)),
            ),
        })
    }
}

impl PhysicalExtensionCodec for OxidePhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if buf.len() > HEADER && &buf[..MAGIC.len()] == MAGIC {
            let version = buf[MAGIC.len()];
            if version != VERSION {
                return Err(DataFusionError::Internal(format!(
                    "OxidePhysicalCodec: unsupported payload version {version} (expected {VERSION})"
                )));
            }
            check_fingerprint(&buf[MAGIC.len() + 1..HEADER])?;
            let node: GpuNode = postcard::from_bytes(&buf[HEADER..]).map_err(|e| {
                DataFusionError::Internal(format!("OxidePhysicalCodec: corrupt payload: {e}"))
            })?;
            return Self::build(node, inputs);
        }
        self.inner.try_decode(buf, inputs, ctx, proto_converter)
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
        proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<()> {
        match Self::describe(node.as_ref()) {
            Some(gpu) => {
                buf.extend_from_slice(MAGIC);
                buf.push(VERSION);
                buf.extend_from_slice(&PLAN_FINGERPRINT.to_le_bytes());
                let payload = postcard::to_allocvec(&gpu).map_err(|e| {
                    DataFusionError::Internal(format!("OxidePhysicalCodec: encode failed: {e}"))
                })?;
                buf.extend_from_slice(&payload);
                Ok(())
            }
            None => self.inner.try_encode(node, buf, proto_converter),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The fingerprint is derived, not declared, so it cannot be forgotten
    /// the way the version byte can (#42).
    #[test]
    fn the_fingerprint_depends_on_the_version_and_the_feature_set() {
        // Recomputed here the way a different build would, rather than
        // asserting a literal: a literal would have to be updated on every
        // release and would then be testing that somebody updated it.
        let expected = {
            let h = fnv1a(0x811c_9dc5, env!("CARGO_PKG_VERSION").as_bytes());
            let h = fnv1a(h, &[VERSION]);
            fnv1a(h, &oxidelake_compute::FEATURE_MASK.to_le_bytes())
        };
        assert_eq!(PLAN_FINGERPRINT, expected);

        // A different crate version and a different feature set each move it.
        let other_version = fnv1a(
            fnv1a(fnv1a(0x811c_9dc5, b"0.0.0-not-this-build"), &[VERSION]),
            &oxidelake_compute::FEATURE_MASK.to_le_bytes(),
        );
        assert_ne!(
            PLAN_FINGERPRINT, other_version,
            "the crate version is in it"
        );

        let other_features = fnv1a(
            fnv1a(
                fnv1a(0x811c_9dc5, env!("CARGO_PKG_VERSION").as_bytes()),
                &[VERSION],
            ),
            &(oxidelake_compute::FEATURE_MASK ^ 0b1).to_le_bytes(),
        );
        assert_ne!(
            PLAN_FINGERPRINT, other_features,
            "flipping the `predict` bit must move it: that is the mismatch \
             this exists to catch"
        );
    }

    /// The refusal names both fingerprints and says what it means, because
    /// the person reading it is looking at a cluster that half works.
    #[test]
    fn a_foreign_fingerprint_is_refused_and_named() {
        check_fingerprint(&PLAN_FINGERPRINT.to_le_bytes()).expect("our own plan decodes");

        let foreign = PLAN_FINGERPRINT ^ 0xdead_beef;
        let err = check_fingerprint(&foreign.to_le_bytes())
            .expect_err("a plan from another build is refused");
        let text = err.to_string();
        assert!(text.contains(&format!("{foreign:#010x}")), "{text}");
        assert!(
            text.contains(&format!("{PLAN_FINGERPRINT:#010x}")),
            "{text}"
        );
        assert!(
            text.contains("not the same version") && text.contains("`predict`"),
            "the message says which two things could differ: {text}"
        );
        assert!(text.contains(&build_description()), "{text}");
    }

    /// The description names every input of the fingerprint, so two builds'
    /// descriptions side by side say which input differs (#92).
    #[test]
    fn the_build_description_names_what_the_fingerprint_is_made_of() {
        let text = build_description();
        assert!(
            text.contains(&format!("{PLAN_FINGERPRINT:#010x}")),
            "{text}"
        );
        assert!(text.contains(env!("CARGO_PKG_VERSION")), "{text}");
        assert!(text.contains(&format!("plan codec v{VERSION}")), "{text}");
        assert!(
            text.contains(&format!(
                "feature mask {:#x}",
                oxidelake_compute::FEATURE_MASK
            )),
            "{text}"
        );
    }

    /// The header is exactly what `try_encode` writes, so the offsets the
    /// decoder slices at are not two independent guesses.
    #[test]
    fn the_header_is_magic_version_and_fingerprint() {
        assert_eq!(HEADER, MAGIC.len() + 1 + 4);
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.extend_from_slice(&PLAN_FINGERPRINT.to_le_bytes());
        assert_eq!(buf.len(), HEADER);
        assert_eq!(&buf[..MAGIC.len()], MAGIC);
        assert_eq!(buf[MAGIC.len()], VERSION);
        check_fingerprint(&buf[MAGIC.len() + 1..HEADER]).expect("its own header verifies");
    }
}
