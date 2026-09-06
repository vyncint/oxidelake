//! The Metal backend (feature `metal`, macOS only).
//!
//! Buffers are `MTLResourceStorageModeShared`, so CPU and GPU share one
//! allocation: uploads are one memcpy into shared memory and downloads are
//! zero-copy — the Arrow output buffers alias the Metal allocation, which stays
//! alive through the `Arc` handed to Arrow. MSL sources are compiled at runtime
//! with `newLibraryWithSource`, cached per library name; compute pipeline
//! states are cached per kernel, and a small pool of command queues is shared
//! by every stream. Each operator encodes as many dispatches as possible into
//! one command buffer and waits once per phase (twice per filter batch: the
//! host computes the exclusive scan between mask and scatter).
//!
//! Filter + projection and vector distance dispatch the MSL kernels in
//! `kernels/metal/` (embedded with `include_str!`); hash join and aggregation
//! stay on the CPU under Metal in v1, and Float64 comparisons too (Metal has
//! no double on most devices) — [`GpuBackend::supports_predicate`] and friends
//! say so up front so the operator layer never uploads for them.
//!
//! This module cannot be compiled on Linux; it is verified on macOS (see
//! `STATUS.md`).

use std::collections::HashMap;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow::alloc::Allocation;
use arrow::array::RecordBatch;
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLSize,
};
use oxidelake_core::params::{Comparison, DistanceMetric, Literal, Predicate, vector_dimension};
use oxidelake_core::{BackendKind, DeviceId, EngineError};
use oxidelake_memory::HostAlloc;
use oxidelake_memory::metal::MetalSharedBuffer;

use crate::args::{AggregateArgs, FilterProjectArgs, HashJoinArgs, VectorDistanceArgs};
use crate::backend::{GpuBackend, MemoryInfo};
use crate::columns::{self, ColumnBytes};
use crate::handles::{
    BatchInner, DeviceBatch, DeviceBuffer, DeviceBufferInner, DeviceStream, HostBuffer, StreamInner,
};

/// MSL source embedded at build time; compiled at runtime.
pub const FILTER_PROJECT_SRC: &str = include_str!("../kernels/metal/filter_project.metal");
/// MSL source embedded at build time; compiled at runtime.
pub const VECTOR_DISTANCE_SRC: &str = include_str!("../kernels/metal/vector_distance.metal");

/// Threads per threadgroup for the row-parallel kernels.
const TG: usize = 256;
/// Command queues shared (round-robin) by every stream the backend hands out.
/// Metal caps live queues per device; one per partition per query would hit it.
const QUEUE_POOL: usize = 4;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

fn ns_error(err: Retained<NSError>) -> EngineError {
    EngineError::device(BackendKind::Metal, err.localizedDescription().to_string())
}

fn foreign(what: &str, backend: BackendKind) -> EngineError {
    EngineError::device(
        BackendKind::Metal,
        format!("{what} belongs to the {backend} backend, not the Metal backend"),
    )
}

/// A Metal command queue (the Metal stream). Streams share the backend's pool.
pub struct MetalQueue(pub(crate) Retained<ProtocolObject<dyn MTLCommandQueue>>);

// SAFETY: MTLCommandQueue is thread-safe under Metal's threading model and the
// wrapper hands out only shared references.
unsafe impl Send for MetalQueue {}
// SAFETY: see the `Send` impl.
unsafe impl Sync for MetalQueue {}

/// One column in shared memory. Buffers are shared (`Arc`) so pass-through
/// columns are not copied and downloaded arrays can alias them.
#[derive(Clone)]
pub struct MetalColumn {
    data_type: DataType,
    len: usize,
    child_len: usize,
    values_len: usize,
    values: Arc<MetalSharedBuffer>,
    validity: Option<Arc<MetalSharedBuffer>>,
}

/// A record batch in shared memory.
pub struct MetalBatch {
    num_rows: usize,
    columns: Vec<MetalColumn>,
}

impl MetalBatch {
    /// Bytes held by all columns.
    pub fn device_bytes(&self) -> u64 {
        self.columns
            .iter()
            .map(|c| c.values.len() + c.validity.as_ref().map_or(0, |v| v.len()))
            .map(|n| u64::try_from(n).unwrap_or(u64::MAX))
            .sum()
    }
}

impl fmt::Debug for MetalBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBatch")
            .field("columns", &self.columns.len())
            .field("device_bytes", &self.device_bytes())
            .finish()
    }
}

/// The Metal backend for the system default device.
pub struct MetalBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    device_id: DeviceId,
    libraries: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLLibrary>>>>,
    pipelines: Mutex<HashMap<(String, String), Pipeline>>,
    queues: Vec<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
    next_queue: AtomicUsize,
    /// Bound in the validity slot of columns without a bitmap (Metal buffer
    /// arguments cannot be null); the kernels never read it.
    dummy: MetalSharedBuffer,
}

// SAFETY: MTLDevice, MTLLibrary, MTLComputePipelineState and MTLCommandQueue
// objects are thread-safe; all mutable state is behind mutexes or atomics.
unsafe impl Send for MetalBackend {}
// SAFETY: see the `Send` impl.
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    /// `true` when a Metal device exists.
    pub fn is_available() -> bool {
        MTLCreateSystemDefaultDevice().is_some()
    }

    /// Opens the system default device.
    pub fn new() -> Result<Self, EngineError> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| {
            EngineError::device(BackendKind::Metal, "no Metal device is available")
        })?;
        let queues = (0..QUEUE_POOL)
            .map(|_| {
                device.newCommandQueue().ok_or_else(|| {
                    EngineError::device(BackendKind::Metal, "could not create a command queue")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dummy = MetalSharedBuffer::new(&device, 1)?;
        Ok(Self {
            device,
            device_id: DeviceId::new(BackendKind::Metal, 0),
            libraries: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            queues,
            next_queue: AtomicUsize::new(0),
            dummy,
        })
    }

    /// The Metal device.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Compiles `source` (MSL) once per `name` and returns the library.
    pub fn library(
        &self,
        name: &str,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, EngineError> {
        let mut cache = self
            .libraries
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(lib) = cache.get(name) {
            return Ok(lib.clone());
        }
        let lib = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            .map_err(ns_error)?;
        cache.insert(name.to_owned(), lib.clone());
        tracing::debug!(library = name, "compiled Metal shader library from source");
        Ok(lib)
    }

    /// The compute pipeline for `function` in library `name`, compiling the
    /// library and building the pipeline state on first use only.
    pub fn pipeline(
        &self,
        name: &str,
        source: &str,
        function: &str,
    ) -> Result<Pipeline, EngineError> {
        let key = (name.to_owned(), function.to_owned());
        if let Some(pso) = self
            .pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
        {
            return Ok(pso.clone());
        }
        let library = self.library(name, source)?;
        let func = library
            .newFunctionWithName(&NSString::from_str(function))
            .ok_or_else(|| {
                EngineError::device(
                    BackendKind::Metal,
                    format!("kernel '{function}' not found in library '{name}'"),
                )
            })?;
        let pso = self
            .device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(ns_error)?;
        self.pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, pso.clone());
        Ok(pso)
    }

    fn queue<'a>(&self, stream: &'a DeviceStream) -> Result<&'a MetalQueue, EngineError> {
        match &stream.inner {
            StreamInner::Metal(q) => Ok(q),
            StreamInner::Host => Err(foreign("stream", stream.backend())),
            #[cfg(feature = "cuda")]
            StreamInner::Cuda(_) => Err(foreign("stream", stream.backend())),
        }
    }

    /// An uninitialised shared buffer for a kernel output or a host-filled
    /// argument: every byte is written before it is read.
    fn alloc(&self, bytes: usize) -> Result<MetalSharedBuffer, EngineError> {
        MetalSharedBuffer::new_uninit(&self.device, bytes)
    }

    fn upload_column(&self, col: &ColumnBytes) -> Result<MetalColumn, EngineError> {
        Ok(MetalColumn {
            data_type: col.data_type.clone(),
            len: col.len,
            child_len: col.child_len,
            values_len: col.values.len(),
            values: Arc::new(MetalSharedBuffer::from_bytes(&self.device, &col.values)?),
            validity: match &col.validity {
                Some(bits) => Some(Arc::new(MetalSharedBuffer::from_bytes(&self.device, bits)?)),
                None => None,
            },
        })
    }

    /// An Arrow buffer aliasing the first `len` bytes of a shared buffer. The
    /// `Arc` keeps the Metal allocation alive for as long as Arrow holds it.
    fn view(buffer: &Arc<MetalSharedBuffer>, len: usize) -> Buffer {
        let len = len.min(buffer.len());
        match NonNull::new(buffer.as_slice().as_ptr().cast_mut()) {
            // SAFETY: `ptr` addresses `len <= buffer.len()` bytes of shared
            // storage owned by the `Arc<MetalSharedBuffer>` handed to the
            // `Buffer`, which therefore outlives every view. Downloads happen
            // after `waitUntilCompleted`, and no kernel writes a buffer after
            // the pass that produced it, so the bytes are immutable from here.
            Some(ptr) => unsafe {
                Buffer::from_custom_allocation(ptr, len, Arc::clone(buffer) as Arc<dyn Allocation>)
            },
            None => Buffer::from(buffer.as_slice()[..len].to_vec()),
        }
    }

    fn download_column(col: &MetalColumn) -> ColumnBytes {
        ColumnBytes {
            data_type: col.data_type.clone(),
            len: col.len,
            values: Self::view(&col.values, col.values_len),
            validity: col
                .validity
                .as_ref()
                .map(|v| Self::view(v, col.len.div_ceil(8))),
            child_len: col.child_len,
        }
    }
}

impl fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device_id", &self.device_id)
            .field("name", &self.device.name().to_string())
            .field("queues", &self.queues.len())
            .finish()
    }
}

/// A scalar kernel argument passed with `setBytes`.
enum Scalar {
    U32(u32),
    I32(i32),
    I64(i64),
}

impl Scalar {
    fn bytes(&self) -> (NonNull<std::ffi::c_void>, usize) {
        match self {
            Scalar::U32(v) => (NonNull::from(v).cast(), std::mem::size_of::<u32>()),
            Scalar::I32(v) => (NonNull::from(v).cast(), std::mem::size_of::<i32>()),
            Scalar::I64(v) => (NonNull::from(v).cast(), std::mem::size_of::<i64>()),
        }
    }
}

/// One kernel argument slot.
enum Arg<'a> {
    Buffer(&'a ProtocolObject<dyn MTLBuffer>),
    Scalar(Scalar),
}

/// One command buffer with one compute encoder: dispatches are encoded in
/// order (Metal tracks the shared buffers, so a later dispatch sees an earlier
/// one's writes) and the GPU is waited for exactly once, in [`Pass::finish`].
struct Pass {
    cmd: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

impl Pass {
    fn begin(queue: &MetalQueue) -> Result<Self, EngineError> {
        let cmd = queue.0.commandBuffer().ok_or_else(|| {
            EngineError::device(BackendKind::Metal, "could not create a command buffer")
        })?;
        let enc = cmd.computeCommandEncoder().ok_or_else(|| {
            EngineError::device(BackendKind::Metal, "could not create a compute encoder")
        })?;
        Ok(Self { cmd, enc })
    }

    /// Encodes one dispatch of `groups` threadgroups of [`TG`] threads. Every
    /// buffer argument must stay alive until [`Self::finish`] returns.
    fn dispatch(&self, pipeline: &Pipeline, args: &[Arg<'_>], groups: usize) {
        self.enc.setComputePipelineState(pipeline);
        for (index, arg) in args.iter().enumerate() {
            match arg {
                // SAFETY: the caller keeps `buffer` alive until the pass has
                // completed (`finish` waits before returning).
                Arg::Buffer(buffer) => unsafe {
                    self.enc.setBuffer_offset_atIndex(Some(buffer), 0, index)
                },
                Arg::Scalar(scalar) => {
                    let (ptr, len) = scalar.bytes();
                    // SAFETY: `ptr` points to `len` initialised bytes; `setBytes`
                    // copies them into the command buffer before returning.
                    unsafe { self.enc.setBytes_length_atIndex(ptr, len, index) }
                }
            }
        }
        let threadgroups = MTLSize {
            width: groups.max(1),
            height: 1,
            depth: 1,
        };
        let per_group = MTLSize {
            width: TG,
            height: 1,
            depth: 1,
        };
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, per_group);
    }

    /// Commits the pass and waits for the GPU; a command-buffer error (GPU
    /// fault, timeout) surfaces as a device error instead of stale output.
    fn finish(self) -> Result<(), EngineError> {
        self.enc.endEncoding();
        self.cmd.commit();
        self.cmd.waitUntilCompleted();
        match self.cmd.error() {
            Some(err) => Err(ns_error(err)),
            None => Ok(()),
        }
    }
}

fn metal_batch(batch: &DeviceBatch) -> Result<&MetalBatch, EngineError> {
    match &batch.inner {
        BatchInner::Metal(b) => Ok(b),
        BatchInner::Host(_) => Err(foreign("batch", batch.backend())),
        #[cfg(feature = "cuda")]
        BatchInner::Cuda(_) => Err(foreign("batch", batch.backend())),
    }
}

fn column(batch: &MetalBatch, index: usize) -> Result<&MetalColumn, EngineError> {
    batch.columns.get(index).ok_or_else(|| {
        EngineError::plan(format!(
            "column index {index} out of range for a batch with {} columns",
            batch.columns.len()
        ))
    })
}

fn u32_len(n: usize, what: &str) -> Result<u32, EngineError> {
    u32::try_from(n).map_err(|_| EngineError::execution(format!("{what} exceeds u32::MAX rows")))
}

fn exclusive_scan(counts: &[u32]) -> Result<(Vec<u32>, usize), EngineError> {
    let mut offsets = Vec::with_capacity(counts.len());
    let mut total: u32 = 0;
    for &c in counts {
        offsets.push(total);
        total = total
            .checked_add(c)
            .ok_or_else(|| EngineError::execution("compaction result exceeds u32::MAX rows"))?;
    }
    Ok((offsets, total as usize))
}

/// Metal v1 compares `Int64` only: every leaf must carry an `Int64` literal.
fn all_int64_leaves(predicate: &Predicate) -> bool {
    match predicate {
        Predicate::Compare { literal, .. } => matches!(literal, Literal::Int64(_)),
        Predicate::And(l, r) => all_int64_leaves(l) && all_int64_leaves(r),
    }
}

impl MetalBackend {
    fn validity_args<'a>(&'a self, col: &'a MetalColumn) -> [Arg<'a>; 2] {
        match &col.validity {
            Some(bits) => [Arg::Buffer(bits.raw()), Arg::Scalar(Scalar::U32(1))],
            None => [Arg::Buffer(self.dummy.raw()), Arg::Scalar(Scalar::U32(0))],
        }
    }

    /// Encodes the mask of `predicate` into `pass`. Every intermediate mask is
    /// pushed onto `keep` (they must outlive the pass); the result is the index
    /// of the final mask in `keep`.
    fn mask(
        &self,
        pass: &Pass,
        batch: &MetalBatch,
        predicate: &Predicate,
        n: u32,
        keep: &mut Vec<MetalSharedBuffer>,
    ) -> Result<usize, EngineError> {
        let groups = (n as usize).div_ceil(TG);
        match predicate {
            Predicate::Compare {
                column: idx,
                op,
                literal,
            } => {
                let col = column(batch, *idx)?;
                let literal = match (&col.data_type, literal) {
                    (DataType::Int64, Literal::Int64(v)) => *v,
                    (dt, lit) => {
                        return Err(EngineError::unsupported(
                            "metal.filter",
                            format!(
                                "column type {dt:?} vs literal {:?}: Metal v1 compares Int64 only",
                                lit.data_type()
                            ),
                        ));
                    }
                };
                let op_code: i32 = match op {
                    Comparison::Eq => 0,
                    Comparison::Lt => 1,
                    Comparison::LtEq => 2,
                    Comparison::Gt => 3,
                    Comparison::GtEq => 4,
                };
                let out = self.alloc(n as usize)?;
                let pipeline =
                    self.pipeline("filter_project", FILTER_PROJECT_SRC, "oxide_compare_i64")?;
                let [v0, v1] = self.validity_args(col);
                pass.dispatch(
                    &pipeline,
                    &[
                        Arg::Buffer(col.values.raw()),
                        v0,
                        v1,
                        Arg::Scalar(Scalar::I32(op_code)),
                        Arg::Scalar(Scalar::I64(literal)),
                        Arg::Buffer(out.raw()),
                        Arg::Scalar(Scalar::U32(n)),
                    ],
                    groups,
                );
                keep.push(out);
                Ok(keep.len() - 1)
            }
            Predicate::And(left, right) => {
                let l = self.mask(pass, batch, left, n, keep)?;
                let r = self.mask(pass, batch, right, n, keep)?;
                let out = self.alloc(n as usize)?;
                let pipeline =
                    self.pipeline("filter_project", FILTER_PROJECT_SRC, "oxide_mask_and")?;
                pass.dispatch(
                    &pipeline,
                    &[
                        Arg::Buffer(keep[l].raw()),
                        Arg::Buffer(keep[r].raw()),
                        Arg::Buffer(out.raw()),
                        Arg::Scalar(Scalar::U32(n)),
                    ],
                    groups,
                );
                keep.push(out);
                Ok(keep.len() - 1)
            }
        }
    }

    /// Encodes the gather of `col` at `indices` (`total` rows) into `pass`.
    fn gather(
        &self,
        pass: &Pass,
        col: &MetalColumn,
        indices: &MetalSharedBuffer,
        total: usize,
    ) -> Result<MetalColumn, EngineError> {
        let width = columns::row_width(&col.data_type)?;
        let total_u32 = u32_len(total, "gather output")?;
        let values = self.alloc(total * width)?;
        if total > 0 {
            let pipeline =
                self.pipeline("filter_project", FILTER_PROJECT_SRC, "oxide_gather_fixed")?;
            pass.dispatch(
                &pipeline,
                &[
                    Arg::Buffer(col.values.raw()),
                    Arg::Scalar(Scalar::U32(u32_len(width, "row width")?)),
                    Arg::Buffer(indices.raw()),
                    Arg::Scalar(Scalar::U32(total_u32)),
                    Arg::Buffer(values.raw()),
                ],
                total.div_ceil(TG),
            );
        }
        let validity = match &col.validity {
            Some(src) => {
                let bytes = total.div_ceil(8);
                let dst = self.alloc(bytes)?;
                if total > 0 {
                    let pipeline = self.pipeline(
                        "filter_project",
                        FILTER_PROJECT_SRC,
                        "oxide_gather_validity",
                    )?;
                    pass.dispatch(
                        &pipeline,
                        &[
                            Arg::Buffer(src.raw()),
                            Arg::Scalar(Scalar::U32(1)),
                            Arg::Buffer(indices.raw()),
                            Arg::Scalar(Scalar::U32(total_u32)),
                            Arg::Buffer(dst.raw()),
                        ],
                        bytes.div_ceil(TG),
                    );
                }
                Some(Arc::new(dst))
            }
            None => None,
        };
        let dim = vector_dimension(&col.data_type).unwrap_or(0);
        Ok(MetalColumn {
            data_type: col.data_type.clone(),
            len: total,
            child_len: total * dim,
            values_len: total * width,
            values: Arc::new(values),
            validity,
        })
    }

    fn dispatch_filter_project(
        &self,
        queue: &MetalQueue,
        args: FilterProjectArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        let input = metal_batch(args.input)?;
        let schema = args.input.schema();
        let n = u32_len(input.num_rows, "filter input")?;
        let fields = args
            .projection
            .iter()
            .map(|&i| {
                schema
                    .fields()
                    .get(i)
                    .cloned()
                    .ok_or_else(|| EngineError::plan(format!("column index {i} out of range")))
            })
            .collect::<Result<Vec<FieldRef>, _>>()?;
        for &i in args.projection {
            column(input, i)?;
        }
        let groups = (n as usize).div_ceil(TG).max(1);

        // Phase 1: comparison masks, their conjunction, and per-threadgroup counts.
        let mut masks = Vec::new();
        let counts = self.alloc(groups * 4)?;
        let phase1 = Pass::begin(queue)?;
        let mask = self.mask(&phase1, input, args.predicate, n, &mut masks)?;
        let pipeline = self.pipeline("filter_project", FILTER_PROJECT_SRC, "oxide_block_count")?;
        phase1.dispatch(
            &pipeline,
            &[
                Arg::Buffer(masks[mask].raw()),
                Arg::Scalar(Scalar::U32(n)),
                Arg::Buffer(counts.raw()),
            ],
            groups,
        );
        phase1.finish()?;

        // Host: exclusive scan of the threadgroup counts.
        let host_counts: Vec<u32> = counts
            .as_slice()
            .as_chunks::<4>()
            .0
            .iter()
            .take(groups)
            .map(|c| u32::from_ne_bytes(*c))
            .collect();
        let (offsets, total) = exclusive_scan(&host_counts)?;
        let mut offsets_buf = self.alloc(offsets.len() * 4)?;
        for (dst, off) in offsets_buf
            .as_mut_slice()
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&offsets)
        {
            *dst = off.to_ne_bytes();
        }

        // Phase 2: scatter the selected row indices, then gather every projected column.
        let indices = self.alloc(total * 4)?;
        let phase2 = Pass::begin(queue)?;
        if total > 0 {
            let pipeline = self.pipeline(
                "filter_project",
                FILTER_PROJECT_SRC,
                "oxide_scatter_indices",
            )?;
            phase2.dispatch(
                &pipeline,
                &[
                    Arg::Buffer(masks[mask].raw()),
                    Arg::Scalar(Scalar::U32(n)),
                    Arg::Buffer(offsets_buf.raw()),
                    Arg::Buffer(indices.raw()),
                ],
                groups,
            );
        }
        let columns = args
            .projection
            .iter()
            .map(|&i| self.gather(&phase2, column(input, i)?, &indices, total))
            .collect::<Result<Vec<_>, _>>()?;
        phase2.finish()?;
        drop((masks, counts, offsets_buf, indices));

        Ok(DeviceBatch::new(
            BackendKind::Metal,
            Arc::new(Schema::new(fields)),
            total,
            BatchInner::Metal(MetalBatch {
                num_rows: total,
                columns,
            }),
        ))
    }

    fn dispatch_vector_distance(
        &self,
        queue: &MetalQueue,
        args: VectorDistanceArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        let input = metal_batch(args.input)?;
        let schema = args.input.schema();
        let col = column(input, args.column)?;
        let dim = vector_dimension(&col.data_type).ok_or_else(|| {
            EngineError::unsupported(
                "metal.vector_distance",
                format!(
                    "column {} has type {:?}; expected FixedSizeList<Float32>",
                    args.column, col.data_type
                ),
            )
        })?;
        if args.query.len() != dim {
            return Err(EngineError::plan(format!(
                "query vector has {} dimensions but the column has {dim}",
                args.query.len()
            )));
        }
        let n = u32_len(input.num_rows, "vector input")?;
        let mut query = self.alloc(dim * 4)?;
        for (dst, q) in query
            .as_mut_slice()
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(args.query)
        {
            *dst = q.to_ne_bytes();
        }
        let out = self.alloc(input.num_rows * 4)?;
        if n > 0 {
            let metric: i32 = match args.metric {
                DistanceMetric::L2 => 0,
                DistanceMetric::Cosine => 1,
            };
            let pipeline =
                self.pipeline("vector_distance", VECTOR_DISTANCE_SRC, "oxide_vec_distance")?;
            let [v0, v1] = self.validity_args(col);
            let pass = Pass::begin(queue)?;
            // One SIMD-group (32 threads) per row, 8 rows per 256-thread group.
            pass.dispatch(
                &pipeline,
                &[
                    Arg::Buffer(col.values.raw()),
                    v0,
                    v1,
                    Arg::Scalar(Scalar::U32(n)),
                    Arg::Scalar(Scalar::U32(u32_len(dim, "vector dimension")?)),
                    Arg::Buffer(query.raw()),
                    Arg::Scalar(Scalar::I32(metric)),
                    Arg::Buffer(out.raw()),
                ],
                input.num_rows.div_ceil(TG / 32),
            );
            pass.finish()?;
        }
        let mut columns = input.columns.clone();
        columns.push(MetalColumn {
            data_type: DataType::Float32,
            len: input.num_rows,
            child_len: 0,
            values_len: input.num_rows * 4,
            values: Arc::new(out),
            validity: col.validity.clone(),
        });
        let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
        fields.push(Arc::new(Field::new(
            args.output_name,
            DataType::Float32,
            true,
        )));
        Ok(DeviceBatch::new(
            BackendKind::Metal,
            Arc::new(Schema::new(fields)),
            input.num_rows,
            BatchInner::Metal(MetalBatch {
                num_rows: input.num_rows,
                columns,
            }),
        ))
    }
}

impl GpuBackend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn memory_info(&self) -> Result<MemoryInfo, EngineError> {
        let total_bytes = self.device.recommendedMaxWorkingSetSize();
        let used = u64::try_from(self.device.currentAllocatedSize()).unwrap_or(u64::MAX);
        Ok(MemoryInfo {
            total_bytes,
            free_bytes: total_bytes.saturating_sub(used),
            pinned_supported: false,
        })
    }

    fn alloc_device(&self, bytes: usize) -> Result<DeviceBuffer, EngineError> {
        let buffer = MetalSharedBuffer::new(&self.device, bytes)?;
        Ok(DeviceBuffer::new(
            BackendKind::Metal,
            bytes,
            DeviceBufferInner::Metal(buffer),
        ))
    }

    fn alloc_pinned_host(&self, bytes: usize) -> Result<HostBuffer, EngineError> {
        // Unified memory: every shared buffer is already host-visible, so there
        // is no separate pinned class to report.
        HostAlloc::pageable(bytes)
    }

    fn create_stream(&self) -> Result<DeviceStream, EngineError> {
        let index = self.next_queue.fetch_add(1, Ordering::Relaxed) % self.queues.len();
        let queue = self.queues.get(index).cloned().ok_or_else(|| {
            EngineError::device(BackendKind::Metal, "command queue pool is empty")
        })?;
        Ok(DeviceStream::new(
            BackendKind::Metal,
            StreamInner::Metal(MetalQueue(queue)),
        ))
    }

    fn copy_h2d(
        &self,
        stream: &DeviceStream,
        src: &HostBuffer,
        dst: &mut DeviceBuffer,
    ) -> Result<(), EngineError> {
        self.queue(stream)?;
        if src.len() > dst.len() {
            return Err(EngineError::execution(format!(
                "copy_h2d: destination holds {} bytes, source has {}",
                dst.len(),
                src.len()
            )));
        }
        let backend = dst.backend();
        let DeviceBufferInner::Metal(buffer) = &mut dst.inner else {
            return Err(foreign("buffer", backend));
        };
        let src = src.as_slice()?;
        buffer.as_mut_slice()[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(
        &self,
        stream: &DeviceStream,
        src: &DeviceBuffer,
        dst: &mut HostBuffer,
    ) -> Result<(), EngineError> {
        self.queue(stream)?;
        if src.len() > dst.len() {
            return Err(EngineError::execution(format!(
                "copy_d2h: destination holds {} bytes, source has {}",
                dst.len(),
                src.len()
            )));
        }
        let DeviceBufferInner::Metal(buffer) = &src.inner else {
            return Err(foreign("buffer", src.backend()));
        };
        dst.as_mut_slice()?[..src.len()].copy_from_slice(&buffer.as_slice()[..src.len()]);
        Ok(())
    }

    fn synchronize(&self, stream: &DeviceStream) -> Result<(), EngineError> {
        // Every pass waits for completion before returning.
        self.queue(stream).map(|_| ())
    }

    fn upload(
        &self,
        stream: &DeviceStream,
        batch: &RecordBatch,
    ) -> Result<DeviceBatch, EngineError> {
        self.queue(stream)?;
        let extracted = columns::extract_batch(batch)?;
        let columns = extracted
            .iter()
            .map(|c| self.upload_column(c))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DeviceBatch::new(
            BackendKind::Metal,
            batch.schema(),
            batch.num_rows(),
            BatchInner::Metal(MetalBatch {
                num_rows: batch.num_rows(),
                columns,
            }),
        ))
    }

    fn download(
        &self,
        stream: &DeviceStream,
        batch: &DeviceBatch,
    ) -> Result<RecordBatch, EngineError> {
        self.queue(stream)?;
        match &batch.inner {
            BatchInner::Metal(resident) => {
                let cols: Vec<ColumnBytes> =
                    resident.columns.iter().map(Self::download_column).collect();
                columns::rebuild_batch(batch.schema(), cols)
            }
            BatchInner::Host(host) => Ok(host.clone()),
            #[cfg(feature = "cuda")]
            BatchInner::Cuda(_) => Err(foreign("batch", batch.backend())),
        }
    }

    fn filter_project(
        &self,
        stream: &DeviceStream,
        args: FilterProjectArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        self.dispatch_filter_project(self.queue(stream)?, args)
    }

    fn hash_join(
        &self,
        _stream: &DeviceStream,
        _args: HashJoinArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        Err(EngineError::unsupported(
            "metal.hash_join",
            "not implemented on Metal in v1; the CPU path is used",
        ))
    }

    fn aggregate(
        &self,
        _stream: &DeviceStream,
        _args: AggregateArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        Err(EngineError::unsupported(
            "metal.aggregate",
            "not implemented on Metal in v1; the CPU path is used",
        ))
    }

    fn vector_distance(
        &self,
        stream: &DeviceStream,
        args: VectorDistanceArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        self.dispatch_vector_distance(self.queue(stream)?, args)
    }

    fn supports_predicate(&self, predicate: &Predicate) -> bool {
        all_int64_leaves(predicate)
    }

    fn supports_hash_join(&self) -> bool {
        false
    }

    fn supports_aggregate(&self) -> bool {
        false
    }
}
