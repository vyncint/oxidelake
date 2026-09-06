//! [`GpuOperator`]: runs one operator's batches on a backend with per-batch
//! CPU fallback and telemetry.
//!
//! The operator layer owns the *column plumbing* around a device kernel so the
//! backends only ever see the columns a kernel reads:
//!
//! * only the columns an operator needs are uploaded — a filter's predicate
//!   and projected columns, a join's keys and projected columns, an
//!   aggregate's key and inputs, a vector distance's single vector column;
//! * projected columns of types the device cannot hold (`Utf8`, …) never
//!   leave the host: the device also compacts a synthetic row-id column and
//!   the operator gathers those columns with Arrow `take` afterwards, so a
//!   string column in the projection no longer forces a whole batch onto the
//!   CPU path;
//! * a backend is asked whether it supports an operation *before* anything is
//!   uploaded ([`GpuBackend::supports_predicate`] and friends);
//! * a hash join's build side is hashed once ([`JoinBuild`]) and uploaded once
//!   per operator, then probed by every batch.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use oxidelake_core::EngineError;
use oxidelake_core::params::{AggregateSpec, DistanceMetric, Predicate, vector_dimension};
use oxidelake_core::telemetry::OperatorStats;
use oxidelake_device::cpu::kernels::{self, JoinTable};
use oxidelake_device::{
    AggregateArgs, DeviceBatch, DeviceStream, FilterProjectArgs, GpuBackend, HashJoinArgs,
    VectorDistanceArgs,
};

/// Name of the synthetic row-id column appended to device sub-batches whose
/// compaction result must be applied to host-resident columns.
const ROW_ID: &str = "__oxide_row";

/// `true` for the column types a device can hold in v1.
fn device_eligible(dt: &DataType) -> bool {
    matches!(dt, DataType::Int64 | DataType::Float64 | DataType::Float32)
        || vector_dimension(dt).is_some()
}

fn push_unique(cols: &mut Vec<usize>, index: usize) {
    if !cols.contains(&index) {
        cols.push(index);
    }
}

fn field(schema: &Schema, index: usize) -> Result<&FieldRef, EngineError> {
    schema.fields().get(index).ok_or_else(|| {
        EngineError::plan(format!(
            "column index {index} out of range for a batch with {} columns",
            schema.fields().len()
        ))
    })
}

/// The columns of one input that travel to the device, in sub-batch order,
/// plus an optional trailing row-id column.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SidePlan {
    device_cols: Vec<usize>,
    row_ids: bool,
}

impl SidePlan {
    fn position(&self, original: usize) -> Option<usize> {
        self.device_cols.iter().position(|&c| c == original)
    }

    fn row_id_position(&self) -> Option<usize> {
        self.row_ids.then_some(self.device_cols.len())
    }

    /// The sub-batch: the planned columns, then `0..n` row ids when requested.
    fn sub_batch(&self, batch: &RecordBatch) -> Result<RecordBatch, EngineError> {
        let schema = batch.schema();
        let mut fields: Vec<FieldRef> = Vec::with_capacity(self.device_cols.len() + 1);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.device_cols.len() + 1);
        for &index in &self.device_cols {
            fields.push(Arc::clone(field(&schema, index)?));
            columns.push(Arc::clone(batch.column(index)));
        }
        if self.row_ids {
            let n = i64::try_from(batch.num_rows())
                .map_err(|_| EngineError::execution("batch exceeds i64::MAX rows"))?;
            fields.push(Arc::new(Field::new(ROW_ID, DataType::Int64, false)));
            columns.push(Arc::new(Int64Array::from_iter_values(0..n)));
        }
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }
}

fn remap_predicate(predicate: &Predicate, plan: &SidePlan) -> Result<Predicate, EngineError> {
    Ok(match predicate {
        Predicate::Compare {
            column,
            op,
            literal,
        } => Predicate::compare(
            plan.position(*column).ok_or_else(|| {
                EngineError::plan(format!(
                    "predicate column {column} missing from device plan"
                ))
            })?,
            *op,
            *literal,
        ),
        Predicate::And(l, r) => {
            Predicate::and(remap_predicate(l, plan)?, remap_predicate(r, plan)?)
        }
    })
}

/// Gathers `column` at the row ids the device compaction produced.
fn take_rows(column: &ArrayRef, row_ids: &ArrayRef) -> Result<ArrayRef, EngineError> {
    Ok(take(column.as_ref(), row_ids.as_ref(), None)?)
}

/// The build side of a hash join, prepared once per join: the collected
/// batch plus its CPU hash table. Shared by every probe partition.
#[derive(Debug)]
pub struct JoinBuild {
    batch: RecordBatch,
    key: usize,
    table: JoinTable,
}

impl JoinBuild {
    /// Hashes `batch` on its `Int64` column `key`.
    pub fn new(batch: RecordBatch, key: usize) -> Result<Self, EngineError> {
        let table = JoinTable::build(&batch, key)?;
        Ok(Self { batch, key, table })
    }

    /// The build batch.
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// The key column index.
    pub fn key(&self) -> usize {
        self.key
    }

    /// The prebuilt table.
    pub fn table(&self) -> &JoinTable {
        &self.table
    }
}

/// The build side as uploaded for one operator, with the plan it was cut by.
struct DeviceBuild {
    plan: SidePlan,
    batch: DeviceBatch,
}

/// Executes operator kernels for one `Gpu*Exec` instance and partition.
///
/// GPU backends get two streams used round-robin; the CPU backend runs the
/// reference kernels directly. Any [`EngineError::Unsupported`] from a GPU
/// path falls back to the CPU reference for that batch.
pub struct GpuOperator {
    backend: Arc<dyn GpuBackend>,
    streams: Vec<DeviceStream>,
    next_stream: AtomicUsize,
    stats: Option<Arc<OperatorStats>>,
    device_build: OnceLock<Arc<DeviceBuild>>,
}

impl std::fmt::Debug for GpuOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuOperator")
            .field("backend", &self.backend.kind())
            .field("streams", &self.streams.len())
            .finish()
    }
}

/// What a device attempt produced: the downloaded batch and the bytes moved
/// host→device / device→host, or nothing when the backend declined.
type DeviceOutcome = Option<(RecordBatch, (usize, usize))>;

impl GpuOperator {
    /// Creates an operator on `backend`, optionally reporting into `stats`.
    pub fn new(
        backend: Arc<dyn GpuBackend>,
        stats: Option<Arc<OperatorStats>>,
    ) -> Result<Self, EngineError> {
        let streams = if backend.kind().is_gpu() {
            vec![backend.create_stream()?, backend.create_stream()?]
        } else {
            vec![backend.create_stream()?]
        };
        Ok(Self {
            backend,
            streams,
            next_stream: AtomicUsize::new(0),
            stats,
            device_build: OnceLock::new(),
        })
    }

    /// The backend batches run on.
    pub fn backend(&self) -> &Arc<dyn GpuBackend> {
        &self.backend
    }

    fn stream(&self) -> &DeviceStream {
        let i = self.next_stream.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        &self.streams[i]
    }

    fn record(
        &self,
        rows_in: usize,
        out: &RecordBatch,
        start: Instant,
        moved: Option<(usize, usize)>,
    ) {
        if let Some(stats) = &self.stats {
            stats.record_batch(rows_in as u64, out.num_rows() as u64, start.elapsed());
            if let Some((h2d, d2h)) = moved {
                stats.record_transfer(h2d as u64, d2h as u64);
            }
        }
    }

    /// Runs `gpu` on a GPU backend; `Ok(None)` when the backend answered
    /// `Unsupported` (the caller then takes the CPU reference path).
    fn try_device<G>(&self, gpu: G) -> Result<DeviceOutcome, EngineError>
    where
        G: FnOnce(&DeviceStream) -> Result<(RecordBatch, usize), EngineError>,
    {
        let stream = self.stream();
        match gpu(stream) {
            Ok((batch, h2d)) => {
                let d2h = batch.get_array_memory_size();
                Ok(Some((batch, (h2d, d2h))))
            }
            Err(err) if err.is_unsupported() => {
                tracing::debug!(backend = %self.backend.kind(), error = %err, "GPU path unsupported for this batch; using the CPU reference");
                Ok(None)
            }
            Err(err) => Err(err),
        }
    }

    /// Uploads `sub`, runs `kernel` on it and downloads the result.
    fn round_trip(
        &self,
        stream: &DeviceStream,
        sub: &RecordBatch,
        kernel: impl FnOnce(&DeviceStream, &DeviceBatch) -> Result<DeviceBatch, EngineError>,
    ) -> Result<(RecordBatch, usize), EngineError> {
        let dev = self.backend.upload(stream, sub)?;
        let out = kernel(stream, &dev)?;
        let host = self.backend.download(stream, &out)?;
        self.backend.synchronize(stream)?;
        Ok((host, sub.get_array_memory_size()))
    }

    /// Fused filter + projection of one batch.
    pub fn filter_project(
        &self,
        batch: &RecordBatch,
        predicate: &Predicate,
        projection: &[usize],
    ) -> Result<RecordBatch, EngineError> {
        let start = Instant::now();
        let device_ok = self.backend.kind().is_gpu() && self.backend.supports_predicate(predicate);
        let outcome = if device_ok {
            self.filter_project_on_device(batch, predicate, projection)?
        } else {
            None
        };
        let (out, moved) = match outcome {
            Some((out, moved)) => (out, Some(moved)),
            None => (kernels::filter_project(batch, predicate, projection)?, None),
        };
        self.record(batch.num_rows(), &out, start, moved);
        Ok(out)
    }

    fn filter_project_on_device(
        &self,
        batch: &RecordBatch,
        predicate: &Predicate,
        projection: &[usize],
    ) -> Result<DeviceOutcome, EngineError> {
        let schema = batch.schema();
        let eligible: Vec<bool> = projection
            .iter()
            .map(|&i| field(&schema, i).map(|f| device_eligible(f.data_type())))
            .collect::<Result<_, _>>()?;
        let mut device_cols = predicate.columns();
        device_cols.dedup();
        let mut unique = Vec::new();
        for c in device_cols {
            push_unique(&mut unique, c);
        }
        for (&i, &ok) in projection.iter().zip(&eligible) {
            if ok {
                push_unique(&mut unique, i);
            }
        }
        let plan = SidePlan {
            device_cols: unique,
            row_ids: eligible.iter().any(|&ok| !ok),
        };
        let sub = plan.sub_batch(batch)?;
        let remapped = remap_predicate(predicate, &plan)?;
        // Device projection: the eligible projected columns in output order,
        // then the row ids (if any host column must be gathered afterwards).
        let mut dev_projection: Vec<usize> = Vec::with_capacity(projection.len() + 1);
        for (&i, &ok) in projection.iter().zip(&eligible) {
            if ok {
                dev_projection.push(plan.position(i).ok_or_else(|| {
                    EngineError::plan(format!("projected column {i} missing from device plan"))
                })?);
            }
        }
        if let Some(pos) = plan.row_id_position() {
            dev_projection.push(pos);
        }
        let Some((dev_out, moved)) = self.try_device(|stream| {
            self.round_trip(stream, &sub, |stream, dev| {
                self.backend.filter_project(
                    stream,
                    FilterProjectArgs {
                        input: dev,
                        predicate: &remapped,
                        projection: &dev_projection,
                    },
                )
            })
        })?
        else {
            return Ok(None);
        };
        // Reassemble in projection order: device outputs for eligible columns,
        // host gathers (at the compacted row ids) for the rest.
        let row_ids = plan
            .row_ids
            .then(|| Arc::clone(dev_out.column(dev_out.num_columns() - 1)));
        let mut next_device = 0usize;
        let mut fields = Vec::with_capacity(projection.len());
        let mut columns = Vec::with_capacity(projection.len());
        for (&i, &ok) in projection.iter().zip(&eligible) {
            fields.push(Arc::clone(field(&schema, i)?));
            if ok {
                columns.push(Arc::clone(dev_out.column(next_device)));
                next_device += 1;
            } else {
                let ids = row_ids
                    .as_ref()
                    .ok_or_else(|| EngineError::execution("row ids missing for host gather"))?;
                columns.push(take_rows(batch.column(i), ids)?);
            }
        }
        let out = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        Ok(Some((out, moved)))
    }

    /// Inner hash join of one probe batch against a prepared build side. The
    /// output is `left ++ build` columns, or `projection` (indices into that
    /// concatenation) in projection order.
    pub fn hash_join(
        &self,
        left: &RecordBatch,
        build: &JoinBuild,
        left_key: usize,
        projection: Option<&[usize]>,
    ) -> Result<RecordBatch, EngineError> {
        let start = Instant::now();
        let device_ok = self.backend.kind().is_gpu() && self.backend.supports_hash_join();
        let outcome = if device_ok {
            self.hash_join_on_device(left, build, left_key, projection)?
        } else {
            None
        };
        let (out, moved) = match outcome {
            Some((out, moved)) => (out, Some(moved)),
            None => (
                kernels::hash_join_probe(left, left_key, build.batch(), build.table(), projection)?,
                None,
            ),
        };
        self.record(left.num_rows(), &out, start, moved);
        Ok(out)
    }

    fn side_plan(
        schema: &Schema,
        key: usize,
        selected: impl Iterator<Item = usize>,
    ) -> Result<SidePlan, EngineError> {
        let mut device_cols = vec![key];
        let mut row_ids = false;
        for index in selected {
            if device_eligible(field(schema, index)?.data_type()) {
                push_unique(&mut device_cols, index);
            } else {
                row_ids = true;
            }
        }
        Ok(SidePlan {
            device_cols,
            row_ids,
        })
    }

    /// The build side on the device, uploaded on first use and reused by every
    /// probe batch of this operator (a fresh operator per partition, so at most
    /// one upload per partition instead of one per batch).
    fn device_build(
        &self,
        stream: &DeviceStream,
        build: &JoinBuild,
        plan: &SidePlan,
    ) -> Result<Arc<DeviceBuild>, EngineError> {
        if let Some(cached) = self.device_build.get()
            && cached.plan == *plan
        {
            return Ok(Arc::clone(cached));
        }
        let sub = plan.sub_batch(build.batch())?;
        let batch = self.backend.upload(stream, &sub)?;
        // Later batches may run on the operator's other stream: make the
        // upload visible to every stream before anyone probes it.
        self.backend.synchronize(stream)?;
        let fresh = Arc::new(DeviceBuild {
            plan: plan.clone(),
            batch,
        });
        Ok(Arc::clone(self.device_build.get_or_init(|| fresh)))
    }

    fn hash_join_on_device(
        &self,
        left: &RecordBatch,
        build: &JoinBuild,
        left_key: usize,
        projection: Option<&[usize]>,
    ) -> Result<DeviceOutcome, EngineError> {
        let (ls, rs) = (left.schema(), build.batch().schema());
        let width = ls.fields().len();
        let selected: Vec<usize> = match projection {
            Some(p) => p.to_vec(),
            None => (0..width + rs.fields().len()).collect(),
        };
        for &index in &selected {
            if index >= width + rs.fields().len() {
                return Err(EngineError::plan(format!(
                    "join projection index {index} out of range"
                )));
            }
        }
        let left_plan = Self::side_plan(
            &ls,
            left_key,
            selected.iter().copied().filter(|&i| i < width),
        )?;
        let right_plan = Self::side_plan(
            &rs,
            build.key(),
            selected
                .iter()
                .copied()
                .filter(|&i| i >= width)
                .map(|i| i - width),
        )?;
        let left_sub = left_plan.sub_batch(left)?;
        let left_key_pos = left_plan.position(left_key).unwrap_or(0);
        let right_key_pos = right_plan.position(build.key()).unwrap_or(0);
        let Some((dev_out, moved)) = self.try_device(|stream| {
            let dev_build = self.device_build(stream, build, &right_plan)?;
            self.round_trip(stream, &left_sub, |stream, dev_left| {
                self.backend.hash_join(
                    stream,
                    HashJoinArgs {
                        left: dev_left,
                        right: &dev_build.batch,
                        left_key: left_key_pos,
                        right_key: right_key_pos,
                    },
                )
            })
        })?
        else {
            return Ok(None);
        };
        // Device output = left sub-batch columns ++ right sub-batch columns
        // (each with its row-id column last when present).
        let left_width = left_plan.device_cols.len() + usize::from(left_plan.row_ids);
        let left_ids = left_plan
            .row_id_position()
            .map(|pos| Arc::clone(dev_out.column(pos)));
        let right_ids = right_plan
            .row_id_position()
            .map(|pos| Arc::clone(dev_out.column(left_width + pos)));
        let mut fields = Vec::with_capacity(selected.len());
        let mut columns = Vec::with_capacity(selected.len());
        for &index in &selected {
            let (schema, batch, plan, ids, offset, local) = if index < width {
                (&ls, left, &left_plan, &left_ids, 0, index)
            } else {
                (
                    &rs,
                    build.batch(),
                    &right_plan,
                    &right_ids,
                    left_width,
                    index - width,
                )
            };
            fields.push(Arc::clone(field(schema, local)?));
            match plan.position(local) {
                Some(pos) => columns.push(Arc::clone(dev_out.column(offset + pos))),
                None => {
                    let ids = ids
                        .as_ref()
                        .ok_or_else(|| EngineError::execution("row ids missing for host gather"))?;
                    columns.push(take_rows(batch.column(local), ids)?);
                }
            }
        }
        let out = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        Ok(Some((out, moved)))
    }

    /// Grouped aggregation of one (complete) input batch.
    pub fn aggregate(
        &self,
        batch: &RecordBatch,
        spec: &AggregateSpec,
    ) -> Result<RecordBatch, EngineError> {
        let start = Instant::now();
        let device_ok = self.backend.kind().is_gpu() && self.backend.supports_aggregate();
        let outcome = if device_ok {
            self.aggregate_on_device(batch, spec)?
        } else {
            None
        };
        let (out, moved) = match outcome {
            Some((out, moved)) => (out, Some(moved)),
            None => (kernels::aggregate(batch, spec)?, None),
        };
        self.record(batch.num_rows(), &out, start, moved);
        Ok(out)
    }

    fn aggregate_on_device(
        &self,
        batch: &RecordBatch,
        spec: &AggregateSpec,
    ) -> Result<DeviceOutcome, EngineError> {
        // Only the key and the aggregate inputs travel; the kernel's output
        // names come from the sub-batch fields, which are the input's fields.
        let mut device_cols = vec![spec.group_by];
        for (_, index) in &spec.aggregates {
            push_unique(&mut device_cols, *index);
        }
        let plan = SidePlan {
            device_cols,
            row_ids: false,
        };
        let sub = plan.sub_batch(batch)?;
        let remapped = AggregateSpec {
            group_by: plan.position(spec.group_by).unwrap_or(0),
            aggregates: spec
                .aggregates
                .iter()
                .map(|(func, index)| (*func, plan.position(*index).unwrap_or(0)))
                .collect(),
        };
        self.try_device(|stream| {
            self.round_trip(stream, &sub, |stream, dev| {
                self.backend.aggregate(
                    stream,
                    AggregateArgs {
                        input: dev,
                        spec: &remapped,
                    },
                )
            })
        })
    }

    /// Appends a distance column to one batch.
    pub fn vector_distance(
        &self,
        batch: &RecordBatch,
        column: usize,
        query: &[f32],
        metric: DistanceMetric,
        output_name: &str,
    ) -> Result<RecordBatch, EngineError> {
        let start = Instant::now();
        let outcome = if self.backend.kind().is_gpu() {
            self.vector_distance_on_device(batch, column, query, metric, output_name)?
        } else {
            None
        };
        let (out, moved) = match outcome {
            Some((out, moved)) => (out, Some(moved)),
            None => (
                kernels::vector_distance(batch, column, query, metric, output_name)?,
                None,
            ),
        };
        self.record(batch.num_rows(), &out, start, moved);
        Ok(out)
    }

    fn vector_distance_on_device(
        &self,
        batch: &RecordBatch,
        column: usize,
        query: &[f32],
        metric: DistanceMetric,
        output_name: &str,
    ) -> Result<DeviceOutcome, EngineError> {
        // Only the vector column travels; every other column is passed through
        // on the host and the distance column is appended to the input batch.
        let plan = SidePlan {
            device_cols: vec![column],
            row_ids: false,
        };
        let sub = plan.sub_batch(batch)?;
        let Some((dev_out, moved)) = self.try_device(|stream| {
            self.round_trip(stream, &sub, |stream, dev| {
                self.backend.vector_distance(
                    stream,
                    VectorDistanceArgs {
                        input: dev,
                        column: 0,
                        query,
                        metric,
                        output_name,
                    },
                )
            })
        })?
        else {
            return Ok(None);
        };
        let distance = Arc::clone(dev_out.column(dev_out.num_columns() - 1));
        let mut fields: Vec<FieldRef> = batch.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(output_name, DataType::Float32, true)));
        let mut columns = batch.columns().to_vec();
        columns.push(distance);
        let out = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        Ok(Some((out, moved)))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use arrow::array::StringArray;

    use super::*;

    fn batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("s", DataType::Utf8, false),
            Field::new("v", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3), Some(4)])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
                Arc::new(arrow::array::Float64Array::from(vec![
                    Some(0.5),
                    Some(1.5),
                    None,
                    Some(3.5),
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn side_plan_cuts_eligible_columns_and_appends_row_ids() {
        let b = batch();
        let plan = SidePlan {
            device_cols: vec![2, 0],
            row_ids: true,
        };
        let sub = plan.sub_batch(&b).unwrap();
        assert_eq!(sub.num_columns(), 3);
        assert_eq!(sub.schema().field(0).name(), "v");
        assert_eq!(sub.schema().field(1).name(), "k");
        assert_eq!(sub.schema().field(2).name(), ROW_ID);
        assert_eq!(plan.position(0), Some(1));
        assert_eq!(plan.row_id_position(), Some(2));
        let ids = sub.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(ids.values(), &[0, 1, 2, 3]);
    }

    #[test]
    fn predicates_are_remapped_to_sub_batch_positions() {
        let plan = SidePlan {
            device_cols: vec![2, 0],
            row_ids: false,
        };
        let p = Predicate::and(
            Predicate::compare(
                0,
                oxidelake_core::params::Comparison::Gt,
                oxidelake_core::params::Literal::Int64(1),
            ),
            Predicate::compare(
                2,
                oxidelake_core::params::Comparison::Lt,
                oxidelake_core::params::Literal::Float64(3.0),
            ),
        );
        let remapped = remap_predicate(&p, &plan).unwrap();
        assert_eq!(remapped.columns(), vec![1, 0]);
        assert!(
            remap_predicate(
                &Predicate::compare(
                    1,
                    oxidelake_core::params::Comparison::Eq,
                    oxidelake_core::params::Literal::Int64(0)
                ),
                &plan
            )
            .is_err()
        );
    }

    #[test]
    fn join_build_hashes_once() {
        let b = batch();
        let build = JoinBuild::new(b.clone(), 0).unwrap();
        assert_eq!(build.table().rows(), 3);
        assert_eq!(build.table().distinct_keys(), 3);
        assert_eq!(build.table().matches(3), &[2]);
        assert!(build.table().matches(2).is_empty());
        assert_eq!(build.key(), 0);
        assert_eq!(build.batch().num_rows(), 4);
    }
}
