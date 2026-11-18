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
//!   CPU path. A vector column with a NULL element inside a valid row stays
//!   there too, since the transfer format cannot carry that NULL (#105);
//! * a backend is asked whether it supports an operation *before* anything is
//!   uploaded ([`GpuBackend::supports_predicate`] and friends);
//! * a hash join's build side is hashed once ([`JoinBuild`]), then uploaded
//!   and indexed on the device once per operator
//!   ([`GpuBackend::prepare_join_build`]) and probed by every batch;
//! * a hash join's output leaves in batches of a bounded size
//!   ([`JoinProbe`]), however often its keys repeat.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
    AggregateArgs, DeviceBatch, DeviceJoinMatches, DeviceJoinTable, DeviceStream,
    FilterProjectArgs, GpuBackend, HashJoinArgs, JoinBuildArgs, JoinProbeArgs, VectorDistanceArgs,
    columns,
};

/// Name of the synthetic row-id column appended to device sub-batches whose
/// compaction result must be applied to host-resident columns.
const ROW_ID: &str = "__oxide_row";

/// `true` for the column types a device can hold in v1.
fn device_eligible(dt: &DataType) -> bool {
    matches!(dt, DataType::Int64 | DataType::Float64 | DataType::Float32)
        || vector_dimension(dt).is_some()
}

/// Whether a projected `column` is uploaded, rather than kept on the host and
/// gathered at the row ids the device selected.
///
/// The type decides first. A vector column of a device type stays on the host
/// as well when a valid row holds a NULL element: the transfer format has one
/// validity bit per row, so a device filter or join would hand the element
/// back as a valid number (#105). Gathered on the host it keeps its NULL, and
/// the batch still runs on the device instead of falling back whole.
fn travels(column: &ArrayRef) -> Result<bool, EngineError> {
    Ok(device_eligible(column.data_type())
        && oxidelake_device::columns::row_with_null_element(column.as_ref())?.is_none())
}

/// [`travels`] for column `index` of `batch`, which must exist.
fn column_travels(batch: &RecordBatch, index: usize) -> Result<bool, EngineError> {
    field(&batch.schema(), index)?;
    travels(batch.column(index))
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

/// The bytes a transfer of `batch` moves: each column's values and validity
/// as [`columns::extract`] cuts them for the device, which is what an upload
/// copies and a download brings back.
///
/// Not `get_array_memory_size`: that is the capacity of the buffers behind
/// each array, so a ten-row slice of a 200 000-row batch reported the whole
/// parent's 3.2 MB (#103). A column no device holds (`Utf8`, …) never
/// travels — the operator gathers those on the host — and counts nothing.
fn transfer_bytes(batch: &RecordBatch) -> u64 {
    batch
        .columns()
        .iter()
        .map(|column| columns::extract(column).map_or(0, |bytes| bytes.total_bytes()))
        .sum()
}

/// The build side of a hash join, prepared once per join: the collected
/// batch plus its CPU hash table. Shared by every probe partition.
#[derive(Debug)]
pub struct JoinBuild {
    batch: RecordBatch,
    key: usize,
    table: JoinTable,
    /// Set once the build rows are in some operator's `rows_in` (#103). The
    /// build side is collected once per join and shared by every probe
    /// partition, so exactly one recorded batch — the first, whichever
    /// partition produces it — carries them. A join whose probe side yields
    /// no batch at all records none, and so reports no build rows either.
    rows_counted: AtomicBool,
    /// Per column, whether it [`travels`] to the device. Every probe batch's
    /// [`JoinPlan`] asks; the build side never changes and is uploaded once
    /// per operator (#45), so it is checked once here rather than once per
    /// probe batch — for a nullable vector column, a pass over its rows.
    uploadable: Vec<bool>,
}

impl JoinBuild {
    /// Hashes `batch` on its `Int64` column `key`.
    pub fn new(batch: RecordBatch, key: usize) -> Result<Self, EngineError> {
        let table = JoinTable::build(&batch, key)?;
        let uploadable = batch
            .columns()
            .iter()
            .map(travels)
            .collect::<Result<_, _>>()?;
        Ok(Self {
            batch,
            key,
            table,
            rows_counted: AtomicBool::new(false),
            uploadable,
        })
    }

    /// Whether build column `index` [`travels`] to the device.
    fn uploadable(&self, index: usize) -> Result<bool, EngineError> {
        self.uploadable.get(index).copied().ok_or_else(|| {
            EngineError::plan(format!(
                "column index {index} out of range for a build side with {} columns",
                self.uploadable.len()
            ))
        })
    }

    /// The build rows for `rows_in` the first time it is asked, then zero.
    fn uncounted_rows(&self) -> usize {
        if self.rows_counted.swap(true, Ordering::Relaxed) {
            0
        } else {
            self.batch.num_rows()
        }
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

/// The build side as uploaded and indexed for one operator, with the plan it
/// was cut by.
struct DeviceBuild {
    plan: SidePlan,
    batch: DeviceBatch,
    table: DeviceJoinTable,
}

/// What became of this operator's attempt to put the build side on the
/// device.
enum BuildState {
    Ready(Arc<DeviceBuild>),
    /// The backend declined. Kept so that a backend which cannot prepare a
    /// join — one that leaves `prepare_join_build` at its default, say — is
    /// handed the build side once rather than once per probe batch.
    Declined {
        feature: &'static str,
        detail: String,
    },
}

/// The columns a join's device path moves from each side, and where each
/// output column comes from.
struct JoinPlan {
    /// Output columns, as indices into `left ++ build`.
    selected: Vec<usize>,
    /// Columns of the probe side.
    width: usize,
    left: SidePlan,
    right: SidePlan,
}

impl JoinPlan {
    /// The plan for one probe batch `left` against `build`. Which columns
    /// travel depends on what they hold, not only their type (#105): the
    /// probe batch is checked here, each time; the build side was checked
    /// once, when its [`JoinBuild`] was made.
    fn new(
        left: &RecordBatch,
        build: &JoinBuild,
        left_key: usize,
        projection: Option<&[usize]>,
    ) -> Result<Self, EngineError> {
        let width = left.num_columns();
        let all = width + build.batch().num_columns();
        let selected: Vec<usize> = match projection {
            Some(p) => p.to_vec(),
            None => (0..all).collect(),
        };
        if let Some(&index) = selected.iter().find(|&&index| index >= all) {
            return Err(EngineError::plan(format!(
                "join projection index {index} out of range"
            )));
        }
        let left_plan = GpuOperator::side_plan(
            left_key,
            selected.iter().copied().filter(|&i| i < width),
            |index| column_travels(left, index),
        )?;
        let right_plan = GpuOperator::side_plan(
            build.key(),
            selected
                .iter()
                .copied()
                .filter(|&i| i >= width)
                .map(|i| i - width),
            |index| build.uploadable(index),
        )?;
        Ok(Self {
            selected,
            width,
            left: left_plan,
            right: right_plan,
        })
    }

    /// The output rows of one downloaded device piece, whose columns are the
    /// left sub-batch's then the right's (each with its row-id column last
    /// when present). A column the device does not hold is gathered on the
    /// host at the row ids.
    fn assemble(
        &self,
        dev_out: &RecordBatch,
        left: &RecordBatch,
        build: &RecordBatch,
    ) -> Result<RecordBatch, EngineError> {
        let (ls, rs) = (left.schema(), build.schema());
        let left_width = self.left.device_cols.len() + usize::from(self.left.row_ids);
        let left_ids = self
            .left
            .row_id_position()
            .map(|pos| Arc::clone(dev_out.column(pos)));
        let right_ids = self
            .right
            .row_id_position()
            .map(|pos| Arc::clone(dev_out.column(left_width + pos)));
        let mut fields = Vec::with_capacity(self.selected.len());
        let mut columns = Vec::with_capacity(self.selected.len());
        for &index in &self.selected {
            let (schema, batch, plan, ids, offset, local) = if index < self.width {
                (&ls, left, &self.left, &left_ids, 0, index)
            } else {
                (
                    &rs,
                    build,
                    &self.right,
                    &right_ids,
                    left_width,
                    index - self.width,
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
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }
}

/// One probe batch of a hash join, on its way out in pieces of a bounded
/// size (#95): the path [`GpuOperator::hash_join`] chose for it and how much
/// of its output has been emitted. Drained with
/// [`GpuOperator::next_join_chunk`].
pub struct JoinProbe {
    left: RecordBatch,
    left_key: usize,
    projection: Option<Vec<usize>>,
    batch_size: usize,
    /// Until the first piece is out: when the probe started, and the rows it
    /// reports in `rows_in`.
    first: Option<(Instant, usize)>,
    path: ProbePath,
}

enum ProbePath {
    /// Probed on the device, where its matches stay; each piece is gathered
    /// there and downloaded.
    Device(Box<DeviceProbe>),
    /// The CPU reference, gathering from match `next` on; `why` is the reason
    /// each piece is counted as a fallback.
    Host {
        matches: kernels::JoinMatches,
        next: u64,
        why: EngineError,
    },
}

/// A probe batch on the device: its sub-batch and its matches, resident on
/// the stream that probed them, and the next match to gather.
struct DeviceProbe {
    stream: usize,
    build: Arc<DeviceBuild>,
    left: DeviceBatch,
    matches: DeviceJoinMatches,
    plan: JoinPlan,
    next: u64,
}

/// What probing one batch on the device produced.
enum Probed {
    Device(Box<DeviceProbe>),
    /// The backend answered `Unsupported` on the way; the batch takes the
    /// CPU reference path.
    Declined(EngineError),
}

impl std::fmt::Debug for JoinProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("JoinProbe");
        s.field("probe_rows", &self.left.num_rows())
            .field("batch_size", &self.batch_size);
        match &self.path {
            ProbePath::Device(device) => s
                .field("device_matches", &device.matches.len())
                .field("device_emitted", &device.next),
            ProbePath::Host { matches, next, .. } => s
                .field("cpu_matches", &matches.len())
                .field("cpu_emitted", next),
        };
        s.finish()
    }
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
    device_build: OnceLock<BuildState>,
    /// One `warn!` per operator instance, not one per batch: a fallback that
    /// happens at all usually happens for every batch, and a log line per
    /// batch would bury the fact it is reporting.
    warned_fallback: AtomicBool,
}

impl std::fmt::Debug for GpuOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuOperator")
            .field("backend", &self.backend.kind())
            .field("streams", &self.streams.len())
            .finish()
    }
}

/// What a device attempt produced.
///
/// The declining case carries *why* rather than being a bare `None` (#32): a
/// fallback is counted and warned about at the point the decision is made,
/// and a counter that cannot say what it counted is not much better than the
/// `debug!` it replaces.
enum DeviceOutcome {
    /// It ran on the device: the downloaded batch and the bytes moved
    /// host→device / device→host.
    Ran(RecordBatch, (u64, u64)),
    /// The device path was declined; the CPU reference runs instead.
    Declined(EngineError),
}

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
            warned_fallback: AtomicBool::new(false),
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

    fn record(&self, rows_in: usize, out: &RecordBatch, start: Instant, moved: Option<(u64, u64)>) {
        if let Some(stats) = &self.stats {
            stats.record_batch(rows_in as u64, out.num_rows() as u64, start.elapsed());
            if let Some((h2d, d2h)) = moved {
                stats.record_transfer(h2d, d2h);
            }
        }
    }

    /// Runs `gpu` on a GPU backend; [`DeviceOutcome::Declined`] when the
    /// backend answered `Unsupported` (the caller then takes the CPU
    /// reference path).
    fn try_device<G>(&self, gpu: G) -> Result<DeviceOutcome, EngineError>
    where
        G: FnOnce(&DeviceStream) -> Result<(RecordBatch, u64), EngineError>,
    {
        let stream = self.stream();
        match gpu(stream) {
            Ok((batch, h2d)) => {
                let d2h = transfer_bytes(&batch);
                Ok(DeviceOutcome::Ran(batch, (h2d, d2h)))
            }
            Err(err) if err.is_unsupported() => Ok(DeviceOutcome::Declined(err)),
            Err(err) => Err(err),
        }
    }

    /// Why this operator has no device path for an operation, or `None` when
    /// it has one.
    ///
    /// `supported` is the backend's own answer for the specific shape (this
    /// predicate, a hash join, …), asked before anything is uploaded.
    fn no_device_path(&self, supported: bool, what: &str) -> Option<EngineError> {
        let kind = self.backend.kind();
        if !kind.is_gpu() {
            Some(EngineError::unsupported(
                "backend.device",
                format!(
                    "the backend selected on this machine is `{kind}`, so an \
                     operator placed on a device runs the CPU reference"
                ),
            ))
        } else if !supported {
            Some(EngineError::unsupported(
                "backend.operation",
                format!("backend `{kind}` does not support {what}"),
            ))
        } else {
            None
        }
    }

    /// Counts a batch that took the CPU reference path, and says so once.
    fn fell_back(&self, why: &EngineError) {
        if let Some(stats) = &self.stats {
            stats.record_fallback();
        }
        if !self.warned_fallback.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                backend = %self.backend.kind(),
                reason = %why,
                "operator placed on a device is running the CPU reference; \
                 results are correct and unaccelerated"
            );
        }
        tracing::debug!(backend = %self.backend.kind(), reason = %why, "batch on the CPU reference");
    }

    /// Uploads `sub`, runs `kernel` on it and downloads the result.
    fn round_trip(
        &self,
        stream: &DeviceStream,
        sub: &RecordBatch,
        kernel: impl FnOnce(&DeviceStream, &DeviceBatch) -> Result<DeviceBatch, EngineError>,
    ) -> Result<(RecordBatch, u64), EngineError> {
        let dev = self.backend.upload(stream, sub)?;
        let out = kernel(stream, &dev)?;
        let host = self.backend.download(stream, &out)?;
        self.backend.synchronize(stream)?;
        Ok((host, transfer_bytes(sub)))
    }

    /// Fused filter + projection of one batch.
    pub fn filter_project(
        &self,
        batch: &RecordBatch,
        predicate: &Predicate,
        projection: &[usize],
    ) -> Result<RecordBatch, EngineError> {
        let start = Instant::now();
        let outcome = match self
            .no_device_path(self.backend.supports_predicate(predicate), "this predicate")
        {
            Some(why) => DeviceOutcome::Declined(why),
            None => self.filter_project_on_device(batch, predicate, projection)?,
        };
        let (out, moved) = match outcome {
            DeviceOutcome::Ran(out, moved) => (out, Some(moved)),
            DeviceOutcome::Declined(why) => {
                self.fell_back(&why);
                (kernels::filter_project(batch, predicate, projection)?, None)
            }
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
            .map(|&i| column_travels(batch, i))
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
        let (dev_out, moved) = match self.try_device(|stream| {
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
        })? {
            DeviceOutcome::Ran(batch, moved) => (batch, moved),
            declined => return Ok(declined),
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
        Ok(DeviceOutcome::Ran(out, moved))
    }

    /// Inner hash join of one probe batch against a prepared build side: looks
    /// the batch's keys up and chooses its path. The output comes from
    /// [`Self::next_join_chunk`] in batches of at most `batch_size` rows, as
    /// `left ++ build` columns or `projection` (indices into that
    /// concatenation) in projection order.
    ///
    /// Bounded because a key that repeats on both sides joins to the product
    /// of its two counts: 65 536 probe rows against 4 000 build rows of one
    /// key are 262 million output rows, which one batch held in full until
    /// the process aborted (#95).
    pub fn hash_join(
        &self,
        left: &RecordBatch,
        build: &JoinBuild,
        left_key: usize,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> Result<JoinProbe, EngineError> {
        let start = Instant::now();
        let probed = match self.no_device_path(self.backend.supports_hash_join(), "hash joins") {
            Some(why) => Probed::Declined(why),
            None => self.probe_on_device(left, build, left_key, projection)?,
        };
        let path = match probed {
            Probed::Device(device) => ProbePath::Device(device),
            Probed::Declined(why) => ProbePath::Host {
                matches: kernels::JoinMatches::probe(left, left_key, build.table())?,
                next: 0,
                why,
            },
        };
        Ok(JoinProbe {
            left: left.clone(),
            left_key,
            projection: projection.map(<[usize]>::to_vec),
            batch_size: batch_size.max(1),
            first: Some((start, left.num_rows() + build.uncounted_rows())),
            path,
        })
    }

    /// The next piece of `probe`'s output, at most its batch size, or `None`
    /// once all of it is out. A probe batch always yields at least one piece
    /// (empty when nothing matched), which is the one that reports its rows
    /// in `rows_in`.
    pub fn next_join_chunk(
        &self,
        probe: &mut JoinProbe,
        build: &JoinBuild,
    ) -> Result<Option<RecordBatch>, EngineError> {
        let started = Instant::now();
        let (out, moved) = loop {
            let (total, next) = match &probe.path {
                ProbePath::Device(device) => (device.matches.len(), device.next),
                ProbePath::Host { matches, next, .. } => (matches.len(), *next),
            };
            let remaining = total - next;
            if remaining == 0 && probe.first.is_none() {
                return Ok(None);
            }
            // At most the batch size, so it fits a `usize`.
            let rows = remaining.min(probe.batch_size as u64) as usize;
            match &mut probe.path {
                ProbePath::Device(device) => {
                    match self.gather_on_device(device, &probe.left, build, rows) {
                        Ok((out, d2h)) => {
                            device.next += rows as u64;
                            break (out, Some((0, d2h)));
                        }
                        // A backend numbers a probe's matches as the CPU
                        // reference does (`GpuBackend::probe_join`), so the
                        // rest of the batch continues there from the same
                        // match, neither repeating nor losing a row.
                        Err(why) if why.is_unsupported() => {
                            probe.path = ProbePath::Host {
                                matches: kernels::JoinMatches::probe(
                                    &probe.left,
                                    probe.left_key,
                                    build.table(),
                                )?,
                                next,
                                why,
                            };
                        }
                        Err(err) => return Err(err),
                    }
                }
                ProbePath::Host { matches, next, why } => {
                    self.fell_back(why);
                    let out = kernels::hash_join_gather(
                        &probe.left,
                        build.batch(),
                        build.table(),
                        matches,
                        *next,
                        rows,
                        probe.projection.as_deref(),
                    )?;
                    *next += rows as u64;
                    break (out, None);
                }
            }
        };
        // The first piece carries the probe itself in its time, and the rows.
        let (start, rows_in) = probe.first.take().unwrap_or((started, 0));
        self.record(rows_in, &out, start, moved);
        Ok(Some(out))
    }

    /// One join side's plan: its key, then each `selected` column that
    /// `travels`; any other selected column is gathered on the host at row
    /// ids.
    fn side_plan(
        key: usize,
        selected: impl Iterator<Item = usize>,
        travels: impl Fn(usize) -> Result<bool, EngineError>,
    ) -> Result<SidePlan, EngineError> {
        let mut device_cols = vec![key];
        let mut row_ids = false;
        for index in selected {
            if travels(index)? {
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

    fn record_transfer(&self, h2d: u64, d2h: u64) {
        if let Some(stats) = &self.stats {
            stats.record_transfer(h2d, d2h);
        }
    }

    /// The build side on the device: uploaded and indexed on first use and
    /// reused by every probe batch of this operator (a fresh operator per
    /// partition, so one upload per partition instead of one per batch). A
    /// backend that declines is not asked again.
    fn device_build(
        &self,
        stream: &DeviceStream,
        build: &JoinBuild,
        plan: &SidePlan,
    ) -> Result<Arc<DeviceBuild>, EngineError> {
        match self.device_build.get() {
            Some(BuildState::Ready(cached)) if cached.plan == *plan => {
                return Ok(Arc::clone(cached));
            }
            Some(BuildState::Declined { feature, detail }) => {
                return Err(EngineError::unsupported(feature, detail.clone()));
            }
            _ => {}
        }
        match self.prepare_device_build(stream, build, plan) {
            Ok(fresh) => {
                // `set` fails only if a build with another plan is cached,
                // which one exec's fixed projection never produces; that
                // batch then uses its own upload without caching it.
                let _ = self.device_build.set(BuildState::Ready(Arc::clone(&fresh)));
                Ok(fresh)
            }
            Err(EngineError::Unsupported { feature, detail }) => {
                let _ = self.device_build.set(BuildState::Declined {
                    feature,
                    detail: detail.clone(),
                });
                Err(EngineError::unsupported(feature, detail))
            }
            Err(other) => Err(other),
        }
    }

    fn prepare_device_build(
        &self,
        stream: &DeviceStream,
        build: &JoinBuild,
        plan: &SidePlan,
    ) -> Result<Arc<DeviceBuild>, EngineError> {
        let sub = plan.sub_batch(build.batch())?;
        let batch = self.backend.upload(stream, &sub)?;
        // Counted where it happens, once per operator: the per-batch
        // counters only see the probe side (#103).
        self.record_transfer(transfer_bytes(&sub), 0);
        let key = plan
            .position(build.key())
            .ok_or_else(|| EngineError::plan("build key missing from the device plan"))?;
        let table = self.backend.prepare_join_build(
            stream,
            JoinBuildArgs {
                build: &batch,
                key,
                index: build.table(),
            },
        )?;
        self.record_transfer(table.uploaded_bytes(), 0);
        // Later batches may run on the operator's other stream: make the
        // upload and the index visible to every stream before anyone probes
        // them.
        self.backend.synchronize(stream)?;
        Ok(Arc::new(DeviceBuild {
            plan: plan.clone(),
            batch,
            table,
        }))
    }

    /// Uploads one probe batch's columns and probes them against the build
    /// side on the device. [`Probed::Declined`] when the backend answered
    /// `Unsupported` on the way: the batch then takes the CPU reference path.
    fn probe_on_device(
        &self,
        left: &RecordBatch,
        build: &JoinBuild,
        left_key: usize,
        projection: Option<&[usize]>,
    ) -> Result<Probed, EngineError> {
        let plan = JoinPlan::new(left, build, left_key, projection)?;
        let left_sub = plan.left.sub_batch(left)?;
        let key = plan
            .left
            .position(left_key)
            .ok_or_else(|| EngineError::plan("probe key missing from the device plan"))?;
        // Every piece of this batch is gathered on the stream that holds its
        // matches.
        let stream = self.next_stream.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        let on = &self.streams[stream];
        let probed = self
            .device_build(on, build, &plan.right)
            .and_then(|dev_build| {
                let dev_left = self.backend.upload(on, &left_sub)?;
                // Counted where it happens, like the build side's.
                self.record_transfer(transfer_bytes(&left_sub), 0);
                let matches = self.backend.probe_join(
                    on,
                    JoinProbeArgs {
                        probe: &dev_left,
                        key,
                        table: &dev_build.table,
                    },
                )?;
                Ok((dev_build, dev_left, matches))
            });
        match probed {
            Ok((build, left, matches)) => Ok(Probed::Device(Box::new(DeviceProbe {
                stream,
                build,
                left,
                matches,
                plan,
                next: 0,
            }))),
            Err(why) if why.is_unsupported() => Ok(Probed::Declined(why)),
            Err(err) => Err(err),
        }
    }

    /// Matches `device.next..+rows` of a probe batch gathered on the device,
    /// downloaded and assembled, with the bytes the download moved.
    fn gather_on_device(
        &self,
        device: &DeviceProbe,
        left: &RecordBatch,
        build: &JoinBuild,
        rows: usize,
    ) -> Result<(RecordBatch, u64), EngineError> {
        let stream = &self.streams[device.stream];
        let out = self.backend.hash_join(
            stream,
            HashJoinArgs {
                left: &device.left,
                right: &device.build.batch,
                table: &device.build.table,
                matches: &device.matches,
                first: device.next,
                rows,
            },
        )?;
        let host = self.backend.download(stream, &out)?;
        self.backend.synchronize(stream)?;
        let d2h = transfer_bytes(&host);
        Ok((device.plan.assemble(&host, left, build.batch())?, d2h))
    }

    /// Grouped aggregation of one (complete) input batch.
    pub fn aggregate(
        &self,
        batch: &RecordBatch,
        spec: &AggregateSpec,
    ) -> Result<RecordBatch, EngineError> {
        let start = Instant::now();
        let outcome = match self.no_device_path(self.backend.supports_aggregate(), "aggregation") {
            Some(why) => DeviceOutcome::Declined(why),
            None => self.aggregate_on_device(batch, spec)?,
        };
        let (out, moved) = match outcome {
            DeviceOutcome::Ran(out, moved) => (out, Some(moved)),
            DeviceOutcome::Declined(why) => {
                self.fell_back(&why);
                (kernels::aggregate(batch, spec)?, None)
            }
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
        let outcome = match self.no_device_path(true, "vector distance") {
            Some(why) => DeviceOutcome::Declined(why),
            None => self.vector_distance_on_device(batch, column, query, metric, output_name)?,
        };
        let (out, moved) = match outcome {
            DeviceOutcome::Ran(out, moved) => (out, Some(moved)),
            DeviceOutcome::Declined(why) => {
                self.fell_back(&why);
                (
                    kernels::vector_distance(batch, column, query, metric, output_name)?,
                    None,
                )
            }
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
        let (dev_out, moved) = match self.try_device(|stream| {
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
        })? {
            DeviceOutcome::Ran(batch, moved) => (batch, moved),
            declined => return Ok(declined),
        };
        let distance = Arc::clone(dev_out.column(dev_out.num_columns() - 1));
        let mut fields: Vec<FieldRef> = batch.schema().fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(output_name, DataType::Float32, true)));
        let mut columns = batch.columns().to_vec();
        columns.push(distance);
        let out = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        Ok(DeviceOutcome::Ran(out, moved))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use arrow::array::StringArray;
    use oxidelake_core::telemetry::TelemetryHub;
    use oxidelake_core::{BackendKind, DeviceId};
    use oxidelake_device::{CpuBackend, DeviceBuffer, HostBuffer, MemoryInfo};

    use super::*;

    /// A backend that *claims* to be a GPU and runs the CPU kernels.
    ///
    /// This is the only way to test the fallback counter (#32) on a machine
    /// with no device: the counter's whole purpose is to distinguish a plan
    /// placed on a device from the work that actually ran there, and both
    /// halves of that need a backend that reports `cuda` while declining some
    /// shapes. `float_predicates` is the knob the Metal backend really has —
    /// it has no f64, so a `Float64` comparison is declined before anything
    /// is uploaded. `declines_prepare` declines `prepare_join_build` the way
    /// the trait's default does, as a backend written before it existed
    /// would; `gathers_before_declining` declines every join piece after
    /// that many. The counters say what the operator asked of the device.
    #[derive(Debug, Default)]
    struct FakeGpu {
        inner: CpuBackend,
        float_predicates: bool,
        declines_prepare: bool,
        gathers_before_declining: Option<usize>,
        uploads: AtomicUsize,
        prepared: AtomicUsize,
        probes: AtomicUsize,
        joins: AtomicUsize,
    }

    impl FakeGpu {
        fn new(float_predicates: bool) -> Arc<Self> {
            Arc::new(Self {
                float_predicates,
                ..Self::default()
            })
        }

        fn shared(float_predicates: bool) -> Arc<dyn GpuBackend> {
            Self::new(float_predicates)
        }
    }

    /// Exhaustive on purpose: `Predicate` is plan vocabulary, so a variant
    /// added to it must be classified here rather than falling into a `_`
    /// arm that would quietly report "no float" for a float predicate.
    fn mentions_float(predicate: &Predicate) -> bool {
        use oxidelake_core::params::Literal;
        match predicate {
            Predicate::Compare { literal, .. } => matches!(literal, Literal::Float64(_)),
            Predicate::And(a, b) => mentions_float(a) || mentions_float(b),
        }
    }

    impl GpuBackend for FakeGpu {
        fn kind(&self) -> BackendKind {
            BackendKind::Cuda
        }
        fn device_id(&self) -> DeviceId {
            DeviceId::new(BackendKind::Cuda, 0)
        }
        fn memory_info(&self) -> Result<MemoryInfo, EngineError> {
            self.inner.memory_info()
        }
        fn alloc_device(&self, bytes: usize) -> Result<DeviceBuffer, EngineError> {
            self.inner.alloc_device(bytes)
        }
        fn alloc_pinned_host(&self, bytes: usize) -> Result<HostBuffer, EngineError> {
            self.inner.alloc_pinned_host(bytes)
        }
        fn create_stream(&self) -> Result<DeviceStream, EngineError> {
            self.inner.create_stream()
        }
        fn copy_h2d(
            &self,
            stream: &DeviceStream,
            src: &HostBuffer,
            dst: &mut DeviceBuffer,
        ) -> Result<(), EngineError> {
            self.inner.copy_h2d(stream, src, dst)
        }
        fn copy_d2h(
            &self,
            stream: &DeviceStream,
            src: &DeviceBuffer,
            dst: &mut HostBuffer,
        ) -> Result<(), EngineError> {
            self.inner.copy_d2h(stream, src, dst)
        }
        fn synchronize(&self, stream: &DeviceStream) -> Result<(), EngineError> {
            self.inner.synchronize(stream)
        }
        fn upload(
            &self,
            stream: &DeviceStream,
            batch: &RecordBatch,
        ) -> Result<DeviceBatch, EngineError> {
            self.uploads.fetch_add(1, Ordering::Relaxed);
            self.inner.upload(stream, batch)
        }
        fn download(
            &self,
            stream: &DeviceStream,
            batch: &DeviceBatch,
        ) -> Result<RecordBatch, EngineError> {
            self.inner.download(stream, batch)
        }
        fn filter_project(
            &self,
            stream: &DeviceStream,
            args: FilterProjectArgs<'_>,
        ) -> Result<DeviceBatch, EngineError> {
            self.inner.filter_project(stream, args)
        }
        fn hash_join(
            &self,
            stream: &DeviceStream,
            args: HashJoinArgs<'_>,
        ) -> Result<DeviceBatch, EngineError> {
            let done = self.joins.fetch_add(1, Ordering::Relaxed);
            if self
                .gathers_before_declining
                .is_some_and(|limit| done >= limit)
            {
                return Err(EngineError::unsupported(
                    "fake.hash_join",
                    "this fake declines join pieces after a few",
                ));
            }
            self.inner.hash_join(stream, args)
        }
        fn prepare_join_build(
            &self,
            stream: &DeviceStream,
            args: JoinBuildArgs<'_>,
        ) -> Result<DeviceJoinTable, EngineError> {
            if self.declines_prepare {
                return Err(EngineError::unsupported(
                    "backend.prepare_join_build",
                    "this fake leaves it at the trait's default",
                ));
            }
            self.prepared.fetch_add(1, Ordering::Relaxed);
            self.inner.prepare_join_build(stream, args)
        }
        fn probe_join(
            &self,
            stream: &DeviceStream,
            args: JoinProbeArgs<'_>,
        ) -> Result<DeviceJoinMatches, EngineError> {
            self.probes.fetch_add(1, Ordering::Relaxed);
            self.inner.probe_join(stream, args)
        }
        fn aggregate(
            &self,
            stream: &DeviceStream,
            args: AggregateArgs<'_>,
        ) -> Result<DeviceBatch, EngineError> {
            self.inner.aggregate(stream, args)
        }
        fn vector_distance(
            &self,
            stream: &DeviceStream,
            args: VectorDistanceArgs<'_>,
        ) -> Result<DeviceBatch, EngineError> {
            self.inner.vector_distance(stream, args)
        }
        fn supports_predicate(&self, predicate: &Predicate) -> bool {
            self.float_predicates || !mentions_float(predicate)
        }
    }

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

    /// #105: a vector column whose valid rows hold a NULL element is planned
    /// like a `Utf8` column — kept off the device and gathered at row ids —
    /// on either side of a join, while the same column with NULL elements
    /// only under its NULL row is uploaded.
    #[test]
    fn a_vector_column_with_null_elements_in_valid_rows_stays_on_the_host() {
        use arrow::array::{Array, FixedSizeListArray, Float32Array};
        use arrow::buffer::NullBuffer;

        let with = |second: Option<f32>| {
            let vectors = FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                2,
                Arc::new(Float32Array::from(vec![
                    Some(1.0),
                    second,
                    None,
                    None,
                    Some(5.0),
                    Some(6.0),
                ])),
                Some(NullBuffer::from(vec![true, false, true])),
            )
            .unwrap();
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int64, false),
                    Field::new("vec", vectors.data_type().clone(), true),
                ])),
                vec![Arc::new(Int64Array::from(vec![1, 2, 3])), Arc::new(vectors)],
            )
            .unwrap()
        };
        let sides = |probe: &RecordBatch, build: &RecordBatch| {
            let build = JoinBuild::new(build.clone(), 0).unwrap();
            let plan = JoinPlan::new(probe, &build, 0, None).unwrap();
            (plan.left, plan.right)
        };
        let uploaded = SidePlan {
            device_cols: vec![0, 1],
            row_ids: false,
        };
        let gathered = SidePlan {
            device_cols: vec![0],
            row_ids: true,
        };
        let (whole, holed) = (with(Some(2.0)), with(None));
        assert_eq!(sides(&whole, &whole), (uploaded.clone(), uploaded.clone()));
        assert_eq!(sides(&holed, &whole), (gathered.clone(), uploaded.clone()));
        assert_eq!(sides(&whole, &holed), (uploaded, gathered));
    }

    // --- the CPU-fallback counter (#32) ------------------------------------

    fn int_predicate() -> Predicate {
        Predicate::compare(
            0,
            oxidelake_core::params::Comparison::Gt,
            oxidelake_core::params::Literal::Int64(1),
        )
    }

    fn float_predicate() -> Predicate {
        Predicate::compare(
            2,
            oxidelake_core::params::Comparison::Lt,
            oxidelake_core::params::Literal::Float64(3.0),
        )
    }

    fn run_filter(backend: Arc<dyn GpuBackend>, predicate: &Predicate) -> (u64, u64, RecordBatch) {
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuFilterExec", backend.kind());
        let op = GpuOperator::new(backend, Some(Arc::clone(&stats))).unwrap();
        let b = batch();
        let out = op.filter_project(&b, predicate, &[0, 2]).unwrap();
        let snapshot = stats.snapshot();
        (snapshot.batches, snapshot.fallback_batches, out)
    }

    /// A predicate the backend accepts runs on the device: no fallback.
    #[test]
    fn a_supported_predicate_counts_no_fallback() {
        let (batches, fallbacks, _) = run_filter(FakeGpu::shared(true), &float_predicate());
        assert_eq!(batches, 1);
        assert_eq!(fallbacks, 0);
    }

    /// A predicate the backend declines is counted, and the answer is still
    /// right — which is exactly why the counter has to exist: nothing else
    /// about the query changes.
    #[test]
    fn a_declined_predicate_counts_a_fallback_and_still_answers() {
        let strict = FakeGpu::shared(false);
        let lenient = FakeGpu::shared(true);
        let (batches, fallbacks, declined) = run_filter(strict, &float_predicate());
        assert_eq!(batches, 1);
        assert_eq!(fallbacks, 1, "a declined predicate is a counted fallback");
        let (_, accepted_fallbacks, accepted) = run_filter(lenient, &float_predicate());
        assert_eq!(accepted_fallbacks, 0);
        assert_eq!(declined, accepted, "the two paths must agree on the rows");

        // The same backend with an Int64 predicate has nothing to decline.
        let (_, int_fallbacks, _) = run_filter(FakeGpu::shared(false), &int_predicate());
        assert_eq!(int_fallbacks, 0);
    }

    /// The shape #32 is about: an operator placed on a device on a machine
    /// that has none. Every batch is a fallback, and the snapshot says so in
    /// one call.
    #[test]
    fn a_cpu_machine_running_a_gpu_plan_falls_back_entirely() {
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuFilterExec", BackendKind::Cuda);
        let op = GpuOperator::new(Arc::new(CpuBackend::new()), Some(Arc::clone(&stats))).unwrap();
        let b = batch();
        for _ in 0..3 {
            op.filter_project(&b, &int_predicate(), &[0, 2]).unwrap();
        }
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.batches, 3);
        assert_eq!(snapshot.fallback_batches, 3);
        assert!(snapshot.fell_back_entirely());
    }

    // --- what the counters say moved (#103) --------------------------------

    /// `{p}k Int64?` cycling through 50 keys with every 7th row (from row 3)
    /// `NULL`, `{p}x Int64`, `{p}s Utf8`.
    fn keyed(rows: usize, p: &str) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(format!("{p}k"), DataType::Int64, true),
            Field::new(format!("{p}x"), DataType::Int64, false),
            Field::new(format!("{p}s"), DataType::Utf8, false),
        ]));
        let rows = i64::try_from(rows).unwrap();
        let k: Int64Array = (0..rows).map(|r| (r % 7 != 3).then_some(r % 50)).collect();
        let x: Int64Array = (0..rows).collect();
        let s: StringArray = (0..rows).map(|r| Some(format!("{p}{}", r % 4))).collect();
        RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(x), Arc::new(s)]).unwrap()
    }

    /// Every piece of one probe batch's output, joined on key column 0.
    fn join_all(
        op: &GpuOperator,
        probe: &RecordBatch,
        build: &JoinBuild,
        projection: Option<&[usize]>,
        batch_size: usize,
    ) -> Vec<RecordBatch> {
        let mut joined = op
            .hash_join(probe, build, 0, projection, batch_size)
            .unwrap();
        let mut pieces = Vec::new();
        while let Some(piece) = op.next_join_chunk(&mut joined, build).unwrap() {
            pieces.push(piece);
        }
        pieces
    }

    /// A join placed on a device reports its build side: the upload once per
    /// operator however many probe batches follow, and the build rows in
    /// `rows_in` once. A probe batch that is a slice counts the bytes the
    /// slice moves, not its parent's buffers. Both offsets are probed because
    /// a validity bitmap that does not start on a byte boundary is re-packed
    /// on the way to the device.
    #[test]
    fn join_counters_include_the_build_side_and_only_the_slices_bytes() {
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuHashJoinExec", BackendKind::Cuda);
        let op = GpuOperator::new(FakeGpu::shared(true), Some(Arc::clone(&stats))).unwrap();
        let build = JoinBuild::new(keyed(200, "r"), 0).unwrap();
        let parent = keyed(2_000, "l");
        // lx, ls, rx, rs: the Utf8 columns stay on the host, so each side
        // also uploads a row-id column for the host gather.
        let projection = [1, 2, 4, 5];
        let mut out_rows = 0;
        for offset in [1_000, 1_010] {
            let probe = parent.slice(offset, 10);
            let out = join_all(&op, &probe, &build, Some(&projection), 1_000);
            assert_eq!(out.len(), 1);
            assert!(out[0].num_rows() > 0);
            out_rows += out[0].num_rows() as u64;
        }
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.fallback_batches, 0, "{snapshot:?}");
        assert_eq!(
            snapshot.rows_in,
            2 * 10 + 200,
            "both probe slices, and the build rows once: {snapshot:?}"
        );
        // Build sub-batch: rk (values, 25 validity bytes), rx, row ids.
        let build_bytes = 200 * 8 + 25 + 200 * 8 + 200 * 8;
        // Each probe slice: lk (values, 2 validity bytes), lx, row ids.
        let probe_bytes = 10 * 8 + 2 + 10 * 8 + 10 * 8;
        assert_eq!(
            snapshot.bytes_h2d,
            build_bytes + 2 * probe_bytes,
            "{snapshot:?}"
        );
        // Back: lk, lx, left row id, rk, rx, right row id per output row. A
        // matched key is never NULL, so no validity bitmap comes back.
        assert_eq!(snapshot.bytes_d2h, 6 * 8 * out_rows, "{snapshot:?}");
    }

    // --- a join's output leaves in pieces of a bounded size (#95) ----------

    /// A probe batch whose matches outnumber the batch size leaves in pieces
    /// of at most that size, together exactly the one-batch reference in its
    /// order. 61 is prime, so the cuts fall inside a probe row's matches as
    /// well as between rows. Each piece is a batch in the counters; the probe
    /// rows (and the build rows, once) are in the first piece's `rows_in`, and
    /// a probe batch that matches nothing still yields one empty piece to
    /// carry them.
    #[test]
    fn a_skewed_probe_batch_leaves_in_pieces_of_the_batch_size() {
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuHashJoinExec", BackendKind::Cuda);
        let op = GpuOperator::new(Arc::new(CpuBackend::new()), Some(Arc::clone(&stats))).unwrap();
        let build = JoinBuild::new(keyed(200, "r"), 0).unwrap();
        let probe = keyed(2_000, "l");
        let reference =
            kernels::hash_join_probe(&probe, 0, build.batch(), build.table(), None).unwrap();
        let total = reference.num_rows();
        assert!(total > 50 * 61, "{total} rows");

        let pieces = join_all(&op, &probe, &build, None, 61);
        assert_eq!(pieces.len(), total.div_ceil(61));
        let largest = pieces.iter().map(RecordBatch::num_rows).max().unwrap();
        assert_eq!(largest, 61);
        let joined = arrow::compute::concat_batches(&reference.schema(), &pieces).unwrap();
        assert_eq!(joined, reference);

        // Row 3's key is NULL, so it matches nothing.
        let nothing = join_all(&op, &probe.slice(3, 1), &build, None, 61);
        assert_eq!(nothing.len(), 1);
        assert_eq!(nothing[0].num_rows(), 0);

        let snapshot = stats.snapshot();
        let batches = pieces.len() as u64 + 1;
        assert_eq!(snapshot.batches, batches, "{snapshot:?}");
        assert_eq!(snapshot.fallback_batches, batches, "{snapshot:?}");
        assert_eq!(snapshot.rows_in, 2_000 + 200 + 1, "{snapshot:?}");
        assert_eq!(snapshot.rows_out, total as u64, "{snapshot:?}");
    }

    fn fake(configure: impl FnOnce(&mut FakeGpu)) -> Arc<FakeGpu> {
        let mut fake = FakeGpu {
            float_predicates: true,
            ..FakeGpu::default()
        };
        configure(&mut fake);
        Arc::new(fake)
    }

    fn joined(pieces: &[RecordBatch]) -> RecordBatch {
        arrow::compute::concat_batches(&pieces[0].schema(), pieces).unwrap()
    }

    /// The device path gathers a skewed probe batch in pieces too (#95): one
    /// probe on the device, then one gather per piece of at most the batch
    /// size, together exactly the CPU reference in its order (#45).
    #[test]
    fn a_device_gathers_a_skewed_probe_batch_in_pieces() {
        let fake = fake(|_| {});
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuHashJoinExec", BackendKind::Cuda);
        let backend = Arc::clone(&fake) as Arc<dyn GpuBackend>;
        let op = GpuOperator::new(backend, Some(Arc::clone(&stats))).unwrap();
        let build = JoinBuild::new(keyed(200, "r"), 0).unwrap();
        let probe = keyed(2_000, "l");
        let reference =
            kernels::hash_join_probe(&probe, 0, build.batch(), build.table(), None).unwrap();

        let pieces = join_all(&op, &probe, &build, None, 61);
        assert_eq!(pieces.len(), reference.num_rows().div_ceil(61));
        assert!(pieces.iter().all(|piece| piece.num_rows() <= 61));
        assert_eq!(joined(&pieces), reference);
        assert_eq!(fake.prepared.load(Ordering::Relaxed), 1);
        assert_eq!(fake.probes.load(Ordering::Relaxed), 1);
        assert_eq!(fake.joins.load(Ordering::Relaxed), pieces.len());
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.fallback_batches, 0, "{snapshot:?}");
        assert_eq!(snapshot.batches, pieces.len() as u64, "{snapshot:?}");
    }

    /// The build side is uploaded and indexed once per operator, however
    /// many probe batches follow (#45), and the device path returns the CPU
    /// reference's rows in the CPU reference's order.
    #[test]
    fn a_join_prepares_its_build_side_once_per_operator() {
        let fake = fake(|_| {});
        let op = GpuOperator::new(Arc::clone(&fake) as Arc<dyn GpuBackend>, None).unwrap();
        let build = JoinBuild::new(keyed(200, "r"), 0).unwrap();
        let parent = keyed(2_000, "l");
        for offset in [0, 500, 1_000] {
            let probe = parent.slice(offset, 100);
            let device = join_all(&op, &probe, &build, None, 65_536);
            let reference =
                kernels::hash_join_probe(&probe, 0, build.batch(), build.table(), None).unwrap();
            assert!(reference.num_rows() > 0);
            assert_eq!(device, vec![reference], "probe rows {offset}..+100");
        }
        assert_eq!(fake.prepared.load(Ordering::Relaxed), 1);
        assert_eq!(fake.probes.load(Ordering::Relaxed), 3);
        assert_eq!(
            fake.uploads.load(Ordering::Relaxed),
            1 + 3,
            "the build side once, then each probe batch"
        );
    }

    /// A backend that cannot prepare a join runs every probe batch on the
    /// CPU reference, counted, and is handed the build side once rather than
    /// once per batch.
    #[test]
    fn a_backend_that_cannot_prepare_a_join_is_asked_once() {
        let fake = fake(|fake| fake.declines_prepare = true);
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuHashJoinExec", BackendKind::Cuda);
        let backend = Arc::clone(&fake) as Arc<dyn GpuBackend>;
        let op = GpuOperator::new(backend, Some(Arc::clone(&stats))).unwrap();
        let build = JoinBuild::new(keyed(200, "r"), 0).unwrap();
        let probe = keyed(100, "l");
        let reference =
            kernels::hash_join_probe(&probe, 0, build.batch(), build.table(), None).unwrap();
        for _ in 0..3 {
            assert_eq!(
                join_all(&op, &probe, &build, None, 65_536),
                vec![reference.clone()]
            );
        }
        assert_eq!(stats.snapshot().fallback_batches, 3);
        assert_eq!(fake.uploads.load(Ordering::Relaxed), 1);
        assert_eq!(fake.probes.load(Ordering::Relaxed), 0);
    }

    /// A probe batch whose device piece is declined finishes on the CPU
    /// reference from the same match: both paths number a probe's matches
    /// the same way, so no row is repeated or lost at the switch.
    #[test]
    fn a_probe_batch_finishes_on_the_cpu_when_the_device_declines_a_piece() {
        let fake = fake(|fake| fake.gathers_before_declining = Some(2));
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuHashJoinExec", BackendKind::Cuda);
        let backend = Arc::clone(&fake) as Arc<dyn GpuBackend>;
        let op = GpuOperator::new(backend, Some(Arc::clone(&stats))).unwrap();
        let build = JoinBuild::new(keyed(200, "r"), 0).unwrap();
        let probe = keyed(2_000, "l");
        let reference =
            kernels::hash_join_probe(&probe, 0, build.batch(), build.table(), None).unwrap();

        let pieces = join_all(&op, &probe, &build, Some(&[1, 2, 4, 5]), 61);
        assert!(pieces.iter().all(|piece| piece.num_rows() <= 61));
        assert_eq!(joined(&pieces), reference.project(&[1, 2, 4, 5]).unwrap());
        let snapshot = stats.snapshot();
        assert_eq!(
            snapshot.fallback_batches,
            pieces.len() as u64 - 2,
            "two pieces on the device, the rest on the CPU: {snapshot:?}"
        );
    }

    /// The reviewer's case for every operator: a filter over ten rows of a
    /// 200 000-row batch moves ten rows, and used to report the parent's
    /// 3.2 MB.
    #[test]
    fn a_sliced_batch_counts_only_the_bytes_it_moves() {
        let hub = TelemetryHub::new();
        let stats = hub.register_operator("GpuFilterExec", BackendKind::Cuda);
        let op = GpuOperator::new(FakeGpu::shared(true), Some(Arc::clone(&stats))).unwrap();
        let slice = keyed(200_000, "t").slice(1_000, 10);
        let all = Predicate::compare(
            1,
            oxidelake_core::params::Comparison::Gt,
            oxidelake_core::params::Literal::Int64(0),
        );
        let out = op.filter_project(&slice, &all, &[0, 1]).unwrap();
        assert_eq!(out.num_rows(), 10);
        let snapshot = stats.snapshot();
        // tk (values, 2 validity bytes) and tx, each way: row 1004 is NULL in
        // tk and survives the filter, so its bitmap comes back too.
        assert_eq!(snapshot.bytes_h2d, 10 * 8 + 2 + 10 * 8, "{snapshot:?}");
        assert_eq!(snapshot.bytes_d2h, 10 * 8 + 2 + 10 * 8, "{snapshot:?}");
    }

    /// Aggregation and vector distance count too, not just the filter.
    #[test]
    fn every_operator_counts_its_fallbacks() {
        let hub = TelemetryHub::new();
        let b = batch();

        let stats = hub.register_operator("GpuAggregateExec", BackendKind::Cuda);
        let op = GpuOperator::new(Arc::new(CpuBackend::new()), Some(Arc::clone(&stats))).unwrap();
        op.aggregate(
            &b,
            &oxidelake_core::params::AggregateSpec {
                group_by: 0,
                aggregates: vec![(oxidelake_core::params::AggregateFunction::Count, 2)],
            },
        )
        .unwrap();
        assert_eq!(stats.snapshot().fallback_batches, 1);
    }
}
