//! Operator implementations for the CUDA backend: launch orchestration for the
//! kernels in `kernels/cuda/`.
//!
//! Every function takes device-resident inputs, returns device-resident (or,
//! for the small aggregate result, host-resident) outputs, and validates types
//! and lengths *before* launching so cudarc's internal asserts can never fire.
//! Small prefix scans (block counts, per-row match counts) round-trip through
//! the host; everything else stays on the device.
//!
//! Divergence from the CPU reference, by design: device `SUM` over `Int64`
//! wraps on overflow where the CPU path reports an error.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use cudarc::driver::{
    CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits,
};
use oxidelake_core::params::{
    AggregateFunction, Comparison, DistanceMetric, Literal, Predicate, vector_dimension,
};
use oxidelake_core::{BackendKind, EngineError};

use super::{CudaBackend, CudaBatch, CudaColumn, driver_error, foreign};
use crate::args::{AggregateArgs, FilterProjectArgs, HashJoinArgs, VectorDistanceArgs};
use crate::columns;
use crate::handles::{BatchInner, DeviceBatch};

/// Threads per block for the one-thread-per-row kernels.
const BLOCK: u32 = 256;
/// Warps per block for the vector-distance kernel (one warp per row).
const VEC_WARPS_PER_BLOCK: u32 = 8;
/// Largest query vector staged in dynamic shared memory (48 KiB).
const MAX_SHARED_QUERY_BYTES: usize = 48 * 1024;

/// Kernel source embedded at build time; compiled at runtime with NVRTC.
pub const FILTER_PROJECT_SRC: &str = include_str!("../../kernels/cuda/filter_project.cu");
/// Kernel source embedded at build time; compiled at runtime with NVRTC.
pub const HASH_JOIN_SRC: &str = include_str!("../../kernels/cuda/hash_join.cu");
/// Kernel source embedded at build time; compiled at runtime with NVRTC.
pub const AGGREGATION_SRC: &str = include_str!("../../kernels/cuda/aggregation.cu");
/// Kernel source embedded at build time; compiled at runtime with NVRTC.
pub const VECTOR_DISTANCE_SRC: &str = include_str!("../../kernels/cuda/vector_distance.cu");

/// A null device pointer; kernels treat a null validity pointer as "all valid".
static NULL_PTR: u64 = 0;

fn u32_len(n: usize, what: &str) -> Result<u32, EngineError> {
    u32::try_from(n).map_err(|_| EngineError::execution(format!("{what} exceeds u32::MAX rows")))
}

fn grid_for(n: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn cuda_batch(batch: &DeviceBatch) -> Result<&CudaBatch, EngineError> {
    match &batch.inner {
        BatchInner::Cuda(b) => Ok(b),
        BatchInner::Host(_) => Err(foreign("batch", batch.backend())),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        BatchInner::Metal(_) => Err(foreign("batch", batch.backend())),
    }
}

fn column(batch: &CudaBatch, index: usize) -> Result<&CudaColumn, EngineError> {
    batch.columns.get(index).ok_or_else(|| {
        EngineError::plan(format!(
            "column index {index} out of range for a batch with {} columns",
            batch.columns.len()
        ))
    })
}

fn field(schema: &SchemaRef, index: usize) -> Result<FieldRef, EngineError> {
    schema.fields().get(index).cloned().ok_or_else(|| {
        EngineError::plan(format!(
            "column index {index} out of range for schema {schema}"
        ))
    })
}

fn alloc<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    n: usize,
) -> Result<CudaSlice<T>, EngineError> {
    stream.alloc_zeros::<T>(n.max(1)).map_err(driver_error)
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

fn require_int64(col: &CudaColumn, role: &str) -> Result<(), EngineError> {
    if col.data_type == DataType::Int64 {
        Ok(())
    } else {
        Err(EngineError::unsupported(
            "cuda.int64_column",
            format!("{role} has type {:?}; v1 supports Int64", col.data_type),
        ))
    }
}

/// Pushes the validity pointer (or null) as a kernel argument.
fn push_validity<'a>(
    launch: &mut cudarc::driver::LaunchArgs<'a>,
    validity: &'a Option<Arc<CudaSlice<u8>>>,
) {
    match validity {
        Some(bits) => {
            launch.arg(&**bits);
        }
        None => {
            launch.arg(&NULL_PTR);
        }
    }
}

// ---------------------------------------------------------------------------
// filter + project
// ---------------------------------------------------------------------------

fn leaf_mask(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    col: &CudaColumn,
    op: Comparison,
    literal: Literal,
    n: u32,
) -> Result<CudaSlice<u8>, EngineError> {
    if col.data_type != literal.data_type() {
        return Err(EngineError::unsupported(
            "cuda.filter",
            format!(
                "column has type {:?} but the literal is {:?}",
                col.data_type,
                literal.data_type()
            ),
        ));
    }
    let op_code: i32 = match op {
        Comparison::Eq => 0,
        Comparison::Lt => 1,
        Comparison::LtEq => 2,
        Comparison::Gt => 3,
        Comparison::GtEq => 4,
    };
    let mut mask = alloc::<u8>(stream, n as usize)?;
    match literal {
        Literal::Int64(v) => {
            let func =
                backend.function("filter_project", FILTER_PROJECT_SRC, "oxide_compare_i64")?;
            let mut launch = stream.launch_builder(&func);
            launch.arg(&*col.values);
            push_validity(&mut launch, &col.validity);
            launch.arg(&op_code);
            launch.arg(&v);
            launch.arg(&mut mask);
            launch.arg(&n);
            // SAFETY: arguments match `oxide_compare_i64`; every buffer holds >= n elements.
            unsafe { launch.launch(grid_for(n)) }.map_err(driver_error)?;
        }
        Literal::Float64(v) => {
            let func =
                backend.function("filter_project", FILTER_PROJECT_SRC, "oxide_compare_f64")?;
            let mut launch = stream.launch_builder(&func);
            launch.arg(&*col.values);
            push_validity(&mut launch, &col.validity);
            launch.arg(&op_code);
            launch.arg(&v);
            launch.arg(&mut mask);
            launch.arg(&n);
            // SAFETY: arguments match `oxide_compare_f64`; every buffer holds >= n elements.
            unsafe { launch.launch(grid_for(n)) }.map_err(driver_error)?;
        }
    }
    Ok(mask)
}

fn evaluate_mask(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    batch: &CudaBatch,
    predicate: &Predicate,
    n: u32,
) -> Result<CudaSlice<u8>, EngineError> {
    match predicate {
        Predicate::Compare {
            column: idx,
            op,
            literal,
        } => leaf_mask(backend, stream, column(batch, *idx)?, *op, *literal, n),
        Predicate::And(left, right) => {
            let l = evaluate_mask(backend, stream, batch, left, n)?;
            let r = evaluate_mask(backend, stream, batch, right, n)?;
            let mut out = alloc::<u8>(stream, n as usize)?;
            let func = backend.function("filter_project", FILTER_PROJECT_SRC, "oxide_mask_and")?;
            let mut launch = stream.launch_builder(&func);
            launch.arg(&l);
            launch.arg(&r);
            launch.arg(&mut out);
            launch.arg(&n);
            // SAFETY: arguments match `oxide_mask_and`; all three masks hold >= n bytes.
            unsafe { launch.launch(grid_for(n)) }.map_err(driver_error)?;
            Ok(out)
        }
    }
}

/// Stable stream compaction of a byte mask into selected row indices.
fn compact(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    mask: &CudaSlice<u8>,
    n: u32,
) -> Result<(CudaSlice<u32>, usize), EngineError> {
    let blocks = n.div_ceil(BLOCK).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut counts = alloc::<u32>(stream, blocks as usize)?;
    {
        let func = backend.function("filter_project", FILTER_PROJECT_SRC, "oxide_block_count")?;
        let mut launch = stream.launch_builder(&func);
        launch.arg(mask);
        launch.arg(&n);
        launch.arg(&mut counts);
        // SAFETY: arguments match `oxide_block_count`; `counts` holds one slot per block.
        unsafe { launch.launch(cfg) }.map_err(driver_error)?;
    }
    let mut host_counts = stream.clone_dtoh(&counts).map_err(driver_error)?;
    host_counts.truncate(blocks as usize);
    let (offsets, total) = exclusive_scan(&host_counts)?;
    let offsets_dev = stream.clone_htod(&offsets).map_err(driver_error)?;
    let mut indices = alloc::<u32>(stream, total)?;
    if total > 0 {
        let func = backend.function(
            "filter_project",
            FILTER_PROJECT_SRC,
            "oxide_scatter_indices",
        )?;
        let mut launch = stream.launch_builder(&func);
        launch.arg(mask);
        launch.arg(&n);
        launch.arg(&offsets_dev);
        launch.arg(&mut indices);
        // SAFETY: arguments match `oxide_scatter_indices`; `indices` holds `total` slots,
        // the exact number of set mask bytes.
        unsafe { launch.launch(cfg) }.map_err(driver_error)?;
    }
    Ok((indices, total))
}

/// Gathers `total` rows of `col` selected by `indices`.
fn gather_column(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    col: &CudaColumn,
    indices: &CudaSlice<u32>,
    total: usize,
) -> Result<CudaColumn, EngineError> {
    let width = columns::row_width(&col.data_type)?;
    let width_u32 = u32_len(width, "row width")?;
    let total_u32 = u32_len(total, "gather output")?;
    let mut values = alloc::<u8>(stream, total * width)?;
    if total > 0 {
        let func = backend.function("filter_project", FILTER_PROJECT_SRC, "oxide_gather_fixed")?;
        let mut launch = stream.launch_builder(&func);
        launch.arg(&*col.values);
        launch.arg(&width_u32);
        launch.arg(indices);
        launch.arg(&total_u32);
        launch.arg(&mut values);
        // SAFETY: arguments match `oxide_gather_fixed`; `values` holds total * width bytes and
        // every index is < the source row count.
        unsafe { launch.launch(grid_for(total_u32)) }.map_err(driver_error)?;
    }
    let validity = match &col.validity {
        Some(src) => {
            let bytes = total.div_ceil(8);
            let mut dst = alloc::<u8>(stream, bytes)?;
            if total > 0 {
                let func = backend.function(
                    "filter_project",
                    FILTER_PROJECT_SRC,
                    "oxide_gather_validity",
                )?;
                let mut launch = stream.launch_builder(&func);
                launch.arg(&**src);
                launch.arg(indices);
                launch.arg(&total_u32);
                launch.arg(&mut dst);
                // SAFETY: arguments match `oxide_gather_validity`; one thread per output byte.
                unsafe { launch.launch(grid_for(u32_len(bytes, "validity bytes")?)) }
                    .map_err(driver_error)?;
            }
            Some(Arc::new(dst))
        }
        None => None,
    };
    let dim = vector_dimension(&col.data_type).unwrap_or(0);
    Ok(CudaColumn {
        data_type: col.data_type.clone(),
        len: total,
        child_len: total * dim,
        values_len: total * width,
        values: Arc::new(values),
        validity,
    })
}

/// Fused filter + projection.
pub(super) fn filter_project(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    args: FilterProjectArgs<'_>,
) -> Result<DeviceBatch, EngineError> {
    let input = cuda_batch(args.input)?;
    let schema = args.input.schema();
    let n = u32_len(input.num_rows, "filter input")?;
    let fields = args
        .projection
        .iter()
        .map(|&i| field(schema, i))
        .collect::<Result<Vec<_>, _>>()?;
    let out_schema = Arc::new(Schema::new(fields));
    // Validate the projection before launching anything.
    for &i in args.projection {
        column(input, i)?;
    }
    let mask = evaluate_mask(backend, stream, input, args.predicate, n)?;
    let (indices, total) = compact(backend, stream, &mask, n)?;
    let columns = args
        .projection
        .iter()
        .map(|&i| gather_column(backend, stream, column(input, i)?, &indices, total))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DeviceBatch::new(
        BackendKind::Cuda,
        out_schema,
        total,
        BatchInner::Cuda(CudaBatch {
            num_rows: total,
            columns,
        }),
    ))
}

// ---------------------------------------------------------------------------
// hash join
// ---------------------------------------------------------------------------

/// Inner hash join on one `Int64` key per side; output = left columns then right columns.
pub(super) fn hash_join(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    args: HashJoinArgs<'_>,
) -> Result<DeviceBatch, EngineError> {
    let left = cuda_batch(args.left)?;
    let right = cuda_batch(args.right)?;
    let lk = column(left, args.left_key)?;
    let rk = column(right, args.right_key)?;
    require_int64(lk, "left join key")?;
    require_int64(rk, "right join key")?;
    let n_left = u32_len(left.num_rows, "join probe side")?;
    let n_right = u32_len(right.num_rows, "join build side")?;
    let fields: Vec<FieldRef> = args
        .left
        .schema()
        .fields()
        .iter()
        .chain(args.right.schema().fields().iter())
        .cloned()
        .collect();
    let out_schema = Arc::new(Schema::new(fields));

    let capacity = (2 * right.num_rows).max(16).next_power_of_two();
    let cap_u32 = u32_len(capacity, "hash table capacity")?;
    let mut t_keys = alloc::<i64>(stream, capacity)?;
    let mut t_rows = alloc::<u32>(stream, capacity)?;
    let mut t_state = alloc::<u32>(stream, capacity)?;
    if n_right > 0 {
        let func = backend.function("hash_join", HASH_JOIN_SRC, "oxide_hj_build")?;
        let mut launch = stream.launch_builder(&func);
        launch.arg(&*rk.values);
        push_validity(&mut launch, &rk.validity);
        launch.arg(&n_right);
        launch.arg(&mut t_keys);
        launch.arg(&mut t_rows);
        launch.arg(&mut t_state);
        launch.arg(&cap_u32);
        // SAFETY: arguments match `oxide_hj_build`; capacity >= 2 * n_right so every insert finds a slot.
        unsafe { launch.launch(grid_for(n_right)) }.map_err(driver_error)?;
    }

    let mut counts = alloc::<u32>(stream, left.num_rows)?;
    if n_left > 0 && n_right > 0 {
        let func = backend.function("hash_join", HASH_JOIN_SRC, "oxide_hj_count")?;
        let mut launch = stream.launch_builder(&func);
        launch.arg(&*lk.values);
        push_validity(&mut launch, &lk.validity);
        launch.arg(&n_left);
        launch.arg(&t_keys);
        launch.arg(&t_state);
        launch.arg(&cap_u32);
        launch.arg(&mut counts);
        // SAFETY: arguments match `oxide_hj_count`; `counts` holds n_left slots.
        unsafe { launch.launch(grid_for(n_left)) }.map_err(driver_error)?;
    }
    let mut host_counts = stream.clone_dtoh(&counts).map_err(driver_error)?;
    host_counts.truncate(left.num_rows);
    let (offsets, total) = exclusive_scan(&host_counts)?;
    let offsets_dev = stream.clone_htod(&offsets).map_err(driver_error)?;
    let mut left_idx = alloc::<u32>(stream, total)?;
    let mut right_idx = alloc::<u32>(stream, total)?;
    if total > 0 {
        let func = backend.function("hash_join", HASH_JOIN_SRC, "oxide_hj_write")?;
        let mut launch = stream.launch_builder(&func);
        launch.arg(&*lk.values);
        push_validity(&mut launch, &lk.validity);
        launch.arg(&n_left);
        launch.arg(&t_keys);
        launch.arg(&t_rows);
        launch.arg(&t_state);
        launch.arg(&cap_u32);
        launch.arg(&offsets_dev);
        launch.arg(&mut left_idx);
        launch.arg(&mut right_idx);
        // SAFETY: arguments match `oxide_hj_write`; the index buffers hold exactly `total`
        // pairs, the sum of the per-row counts computed by `oxide_hj_count`.
        unsafe { launch.launch(grid_for(n_left)) }.map_err(driver_error)?;
    }

    let mut columns = Vec::with_capacity(left.columns.len() + right.columns.len());
    for col in &left.columns {
        columns.push(gather_column(backend, stream, col, &left_idx, total)?);
    }
    for col in &right.columns {
        columns.push(gather_column(backend, stream, col, &right_idx, total)?);
    }
    Ok(DeviceBatch::new(
        BackendKind::Cuda,
        out_schema,
        total,
        BatchInner::Cuda(CudaBatch {
            num_rows: total,
            columns,
        }),
    ))
}

// ---------------------------------------------------------------------------
// aggregate
// ---------------------------------------------------------------------------

enum HostAgg {
    SumI {
        sums: Vec<i64>,
        counts: Vec<u32>,
        null_sum: i64,
        null_count: u32,
    },
    SumF {
        sums: Vec<f64>,
        counts: Vec<u32>,
        null_sum: f64,
        null_count: u32,
    },
    MinMaxI {
        mins: Vec<i64>,
        maxs: Vec<i64>,
        counts: Vec<u32>,
        null_min: i64,
        null_max: i64,
        null_count: u32,
    },
    MinMaxF {
        mins: Vec<f64>,
        maxs: Vec<f64>,
        counts: Vec<u32>,
        null_min: f64,
        null_max: f64,
        null_count: u32,
    },
    Count {
        counts: Vec<u32>,
        null_count: u32,
    },
}

struct GroupTable<'a> {
    keys: &'a CudaColumn,
    n: u32,
    capacity: u32,
    g_keys: CudaSlice<i64>,
    g_state: CudaSlice<u32>,
}

fn first<T: Copy>(v: &[T], what: &str) -> Result<T, EngineError> {
    v.first()
        .copied()
        .ok_or_else(|| EngineError::execution(format!("device returned an empty {what} buffer")))
}

#[allow(clippy::too_many_lines)]
fn run_aggregate(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    table: &mut GroupTable<'_>,
    func: AggregateFunction,
    col: &CudaColumn,
) -> Result<HostAgg, EngineError> {
    let cap = table.capacity as usize;
    let n = table.n;
    let is_i64 = match col.data_type {
        DataType::Int64 => true,
        DataType::Float64 => false,
        ref other => {
            return Err(EngineError::unsupported(
                "cuda.aggregate",
                format!("aggregate input has type {other:?}; v1 supports Int64 and Float64"),
            ));
        }
    };
    let dtoh_u32 = |s: &CudaSlice<u32>| -> Result<Vec<u32>, EngineError> {
        let mut v = stream.clone_dtoh(s).map_err(driver_error)?;
        v.truncate(cap);
        Ok(v)
    };
    match (func, is_i64) {
        (AggregateFunction::Count, _) => {
            let mut counts = alloc::<u32>(stream, cap)?;
            let mut null_count = alloc::<u32>(stream, 1)?;
            if n > 0 {
                let f = backend.function("aggregation", AGGREGATION_SRC, "oxide_agg_count")?;
                let mut launch = stream.launch_builder(&f);
                launch.arg(&*table.keys.values);
                push_validity(&mut launch, &table.keys.validity);
                push_validity(&mut launch, &col.validity);
                launch.arg(&n);
                launch.arg(&mut table.g_keys);
                launch.arg(&mut table.g_state);
                launch.arg(&table.capacity);
                launch.arg(&mut counts);
                launch.arg(&mut null_count);
                // SAFETY: arguments match `oxide_agg_count`; per-slot buffers hold `capacity` entries.
                unsafe { launch.launch(grid_for(n)) }.map_err(driver_error)?;
            }
            Ok(HostAgg::Count {
                counts: dtoh_u32(&counts)?,
                null_count: first(
                    &stream.clone_dtoh(&null_count).map_err(driver_error)?,
                    "null count",
                )?,
            })
        }
        (AggregateFunction::Sum, true) => {
            let mut sums = alloc::<i64>(stream, cap)?;
            let mut counts = alloc::<u32>(stream, cap)?;
            let mut null_sum = alloc::<i64>(stream, 1)?;
            let mut null_count = alloc::<u32>(stream, 1)?;
            if n > 0 {
                let f = backend.function("aggregation", AGGREGATION_SRC, "oxide_agg_sum_i64")?;
                let mut launch = stream.launch_builder(&f);
                launch.arg(&*table.keys.values);
                push_validity(&mut launch, &table.keys.validity);
                launch.arg(&*col.values);
                push_validity(&mut launch, &col.validity);
                launch.arg(&n);
                launch.arg(&mut table.g_keys);
                launch.arg(&mut table.g_state);
                launch.arg(&table.capacity);
                launch.arg(&mut sums);
                launch.arg(&mut counts);
                launch.arg(&mut null_sum);
                launch.arg(&mut null_count);
                // SAFETY: arguments match `oxide_agg_sum_i64`; per-slot buffers hold `capacity` entries.
                unsafe { launch.launch(grid_for(n)) }.map_err(driver_error)?;
            }
            let mut s = stream.clone_dtoh(&sums).map_err(driver_error)?;
            s.truncate(cap);
            Ok(HostAgg::SumI {
                sums: s,
                counts: dtoh_u32(&counts)?,
                null_sum: first(
                    &stream.clone_dtoh(&null_sum).map_err(driver_error)?,
                    "null sum",
                )?,
                null_count: first(
                    &stream.clone_dtoh(&null_count).map_err(driver_error)?,
                    "null count",
                )?,
            })
        }
        (AggregateFunction::Sum, false) => {
            let mut sums = alloc::<f64>(stream, cap)?;
            let mut counts = alloc::<u32>(stream, cap)?;
            let mut null_sum = alloc::<f64>(stream, 1)?;
            let mut null_count = alloc::<u32>(stream, 1)?;
            if n > 0 {
                let f = backend.function("aggregation", AGGREGATION_SRC, "oxide_agg_sum_f64")?;
                let mut launch = stream.launch_builder(&f);
                launch.arg(&*table.keys.values);
                push_validity(&mut launch, &table.keys.validity);
                launch.arg(&*col.values);
                push_validity(&mut launch, &col.validity);
                launch.arg(&n);
                launch.arg(&mut table.g_keys);
                launch.arg(&mut table.g_state);
                launch.arg(&table.capacity);
                launch.arg(&mut sums);
                launch.arg(&mut counts);
                launch.arg(&mut null_sum);
                launch.arg(&mut null_count);
                // SAFETY: arguments match `oxide_agg_sum_f64`; per-slot buffers hold `capacity` entries.
                unsafe { launch.launch(grid_for(n)) }.map_err(driver_error)?;
            }
            let mut s = stream.clone_dtoh(&sums).map_err(driver_error)?;
            s.truncate(cap);
            Ok(HostAgg::SumF {
                sums: s,
                counts: dtoh_u32(&counts)?,
                null_sum: first(
                    &stream.clone_dtoh(&null_sum).map_err(driver_error)?,
                    "null sum",
                )?,
                null_count: first(
                    &stream.clone_dtoh(&null_count).map_err(driver_error)?,
                    "null count",
                )?,
            })
        }
        (AggregateFunction::Min | AggregateFunction::Max, true) => {
            let mut mins = stream
                .clone_htod(&vec![i64::MAX; cap])
                .map_err(driver_error)?;
            let mut maxs = stream
                .clone_htod(&vec![i64::MIN; cap])
                .map_err(driver_error)?;
            let mut counts = alloc::<u32>(stream, cap)?;
            let mut null_min = stream.clone_htod(&[i64::MAX]).map_err(driver_error)?;
            let mut null_max = stream.clone_htod(&[i64::MIN]).map_err(driver_error)?;
            let mut null_count = alloc::<u32>(stream, 1)?;
            if n > 0 {
                let f = backend.function("aggregation", AGGREGATION_SRC, "oxide_agg_minmax_i64")?;
                let mut launch = stream.launch_builder(&f);
                launch.arg(&*table.keys.values);
                push_validity(&mut launch, &table.keys.validity);
                launch.arg(&*col.values);
                push_validity(&mut launch, &col.validity);
                launch.arg(&n);
                launch.arg(&mut table.g_keys);
                launch.arg(&mut table.g_state);
                launch.arg(&table.capacity);
                launch.arg(&mut mins);
                launch.arg(&mut maxs);
                launch.arg(&mut counts);
                launch.arg(&mut null_min);
                launch.arg(&mut null_max);
                launch.arg(&mut null_count);
                // SAFETY: arguments match `oxide_agg_minmax_i64`; per-slot buffers hold `capacity` entries.
                unsafe { launch.launch(grid_for(n)) }.map_err(driver_error)?;
            }
            let mut lo = stream.clone_dtoh(&mins).map_err(driver_error)?;
            lo.truncate(cap);
            let mut hi = stream.clone_dtoh(&maxs).map_err(driver_error)?;
            hi.truncate(cap);
            Ok(HostAgg::MinMaxI {
                mins: lo,
                maxs: hi,
                counts: dtoh_u32(&counts)?,
                null_min: first(
                    &stream.clone_dtoh(&null_min).map_err(driver_error)?,
                    "null min",
                )?,
                null_max: first(
                    &stream.clone_dtoh(&null_max).map_err(driver_error)?,
                    "null max",
                )?,
                null_count: first(
                    &stream.clone_dtoh(&null_count).map_err(driver_error)?,
                    "null count",
                )?,
            })
        }
        (AggregateFunction::Min | AggregateFunction::Max, false) => {
            let mut mins = stream
                .clone_htod(&vec![f64::INFINITY; cap])
                .map_err(driver_error)?;
            let mut maxs = stream
                .clone_htod(&vec![f64::NEG_INFINITY; cap])
                .map_err(driver_error)?;
            let mut counts = alloc::<u32>(stream, cap)?;
            let mut null_min = stream.clone_htod(&[f64::INFINITY]).map_err(driver_error)?;
            let mut null_max = stream
                .clone_htod(&[f64::NEG_INFINITY])
                .map_err(driver_error)?;
            let mut null_count = alloc::<u32>(stream, 1)?;
            if n > 0 {
                let f = backend.function("aggregation", AGGREGATION_SRC, "oxide_agg_minmax_f64")?;
                let mut launch = stream.launch_builder(&f);
                launch.arg(&*table.keys.values);
                push_validity(&mut launch, &table.keys.validity);
                launch.arg(&*col.values);
                push_validity(&mut launch, &col.validity);
                launch.arg(&n);
                launch.arg(&mut table.g_keys);
                launch.arg(&mut table.g_state);
                launch.arg(&table.capacity);
                launch.arg(&mut mins);
                launch.arg(&mut maxs);
                launch.arg(&mut counts);
                launch.arg(&mut null_min);
                launch.arg(&mut null_max);
                launch.arg(&mut null_count);
                // SAFETY: arguments match `oxide_agg_minmax_f64`; per-slot buffers hold `capacity` entries.
                unsafe { launch.launch(grid_for(n)) }.map_err(driver_error)?;
            }
            let mut lo = stream.clone_dtoh(&mins).map_err(driver_error)?;
            lo.truncate(cap);
            let mut hi = stream.clone_dtoh(&maxs).map_err(driver_error)?;
            hi.truncate(cap);
            Ok(HostAgg::MinMaxF {
                mins: lo,
                maxs: hi,
                counts: dtoh_u32(&counts)?,
                null_min: first(
                    &stream.clone_dtoh(&null_min).map_err(driver_error)?,
                    "null min",
                )?,
                null_max: first(
                    &stream.clone_dtoh(&null_max).map_err(driver_error)?,
                    "null max",
                )?,
                null_count: first(
                    &stream.clone_dtoh(&null_count).map_err(driver_error)?,
                    "null count",
                )?,
            })
        }
    }
}

/// Grouped `SUM/COUNT/MIN/MAX` over one `Int64` key. Reductions run on the
/// device; the (small) group table is finalized on the host, so the result is
/// returned host-resident.
pub(super) fn aggregate(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    args: AggregateArgs<'_>,
) -> Result<DeviceBatch, EngineError> {
    let input = cuda_batch(args.input)?;
    let schema = args.input.schema();
    let spec = args.spec;
    let keys = column(input, spec.group_by)?;
    require_int64(keys, "group key")?;
    let n = u32_len(input.num_rows, "aggregate input")?;
    let capacity = (2 * input.num_rows).max(16).next_power_of_two();
    let cap_u32 = u32_len(capacity, "group table capacity")?;

    let mut table = GroupTable {
        keys,
        n,
        capacity: cap_u32,
        g_keys: alloc::<i64>(stream, capacity)?,
        g_state: alloc::<u32>(stream, capacity)?,
    };
    let mut null_flag = alloc::<u32>(stream, 1)?;
    if n > 0 {
        let f = backend.function("aggregation", AGGREGATION_SRC, "oxide_agg_insert_keys")?;
        let mut launch = stream.launch_builder(&f);
        launch.arg(&*keys.values);
        push_validity(&mut launch, &keys.validity);
        launch.arg(&n);
        launch.arg(&mut table.g_keys);
        launch.arg(&mut table.g_state);
        launch.arg(&cap_u32);
        launch.arg(&mut null_flag);
        // SAFETY: arguments match `oxide_agg_insert_keys`; capacity >= 2 * n so every key finds a slot.
        unsafe { launch.launch(grid_for(n)) }.map_err(driver_error)?;
    }

    let mut results = Vec::with_capacity(spec.aggregates.len());
    for (func, idx) in &spec.aggregates {
        let col = column(input, *idx)?;
        results.push(run_aggregate(backend, stream, &mut table, *func, col)?);
    }

    let mut host_keys = stream.clone_dtoh(&table.g_keys).map_err(driver_error)?;
    host_keys.truncate(capacity);
    let mut host_state = stream.clone_dtoh(&table.g_state).map_err(driver_error)?;
    host_state.truncate(capacity);
    let has_null_group = first(
        &stream.clone_dtoh(&null_flag).map_err(driver_error)?,
        "null flag",
    )? != 0;
    let mut slots: Vec<(i64, usize)> = host_state
        .iter()
        .enumerate()
        .filter(|(_, state)| **state == 2)
        .map(|(slot, _)| (host_keys[slot], slot))
        .collect();
    slots.sort_unstable();

    let key_field = field(schema, spec.group_by)?;
    let mut fields = vec![Field::new(key_field.name(), DataType::Int64, true)];
    let mut key_values: Vec<Option<i64>> = slots.iter().map(|(k, _)| Some(*k)).collect();
    if has_null_group {
        key_values.push(None);
    }
    let rows = key_values.len();
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(key_values))];

    for ((func, idx), agg) in spec.aggregates.iter().zip(&results) {
        let name = format!("{}({})", func.name(), field(schema, *idx)?.name());
        let (dt, array): (DataType, ArrayRef) = match agg {
            HostAgg::Count { counts, null_count } => {
                let mut v: Vec<Option<i64>> = slots
                    .iter()
                    .map(|(_, s)| Some(i64::from(counts[*s])))
                    .collect();
                if has_null_group {
                    v.push(Some(i64::from(*null_count)));
                }
                (DataType::Int64, Arc::new(Int64Array::from(v)))
            }
            HostAgg::SumI {
                sums,
                counts,
                null_sum,
                null_count,
            } => {
                let mut v: Vec<Option<i64>> = slots
                    .iter()
                    .map(|(_, s)| (counts[*s] > 0).then_some(sums[*s]))
                    .collect();
                if has_null_group {
                    v.push((*null_count > 0).then_some(*null_sum));
                }
                (DataType::Int64, Arc::new(Int64Array::from(v)))
            }
            HostAgg::SumF {
                sums,
                counts,
                null_sum,
                null_count,
            } => {
                let mut v: Vec<Option<f64>> = slots
                    .iter()
                    .map(|(_, s)| (counts[*s] > 0).then_some(sums[*s]))
                    .collect();
                if has_null_group {
                    v.push((*null_count > 0).then_some(*null_sum));
                }
                (DataType::Float64, Arc::new(Float64Array::from(v)))
            }
            HostAgg::MinMaxI {
                mins,
                maxs,
                counts,
                null_min,
                null_max,
                null_count,
            } => {
                let pick = |s: usize| {
                    if *func == AggregateFunction::Min {
                        mins[s]
                    } else {
                        maxs[s]
                    }
                };
                let mut v: Vec<Option<i64>> = slots
                    .iter()
                    .map(|(_, s)| (counts[*s] > 0).then(|| pick(*s)))
                    .collect();
                if has_null_group {
                    let nv = if *func == AggregateFunction::Min {
                        *null_min
                    } else {
                        *null_max
                    };
                    v.push((*null_count > 0).then_some(nv));
                }
                (DataType::Int64, Arc::new(Int64Array::from(v)))
            }
            HostAgg::MinMaxF {
                mins,
                maxs,
                counts,
                null_min,
                null_max,
                null_count,
            } => {
                let pick = |s: usize| {
                    if *func == AggregateFunction::Min {
                        mins[s]
                    } else {
                        maxs[s]
                    }
                };
                let mut v: Vec<Option<f64>> = slots
                    .iter()
                    .map(|(_, s)| (counts[*s] > 0).then(|| pick(*s)))
                    .collect();
                if has_null_group {
                    let nv = if *func == AggregateFunction::Min {
                        *null_min
                    } else {
                        *null_max
                    };
                    v.push((*null_count > 0).then_some(nv));
                }
                (DataType::Float64, Arc::new(Float64Array::from(v)))
            }
        };
        fields.push(Field::new(name, dt, true));
        arrays.push(array);
    }
    let out_schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&out_schema), arrays)?;
    Ok(DeviceBatch::new(
        BackendKind::Cuda,
        out_schema,
        rows,
        BatchInner::Host(batch),
    ))
}

// ---------------------------------------------------------------------------
// vector distance
// ---------------------------------------------------------------------------

/// Appends a `Float32` distance column; output validity equals the input's.
pub(super) fn vector_distance(
    backend: &CudaBackend,
    stream: &Arc<CudaStream>,
    args: VectorDistanceArgs<'_>,
) -> Result<DeviceBatch, EngineError> {
    let input = cuda_batch(args.input)?;
    let schema = args.input.schema();
    let col = column(input, args.column)?;
    let dim = vector_dimension(&col.data_type).ok_or_else(|| {
        EngineError::unsupported(
            "cuda.vector_distance",
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
    let shared_bytes = dim * 4;
    if shared_bytes > MAX_SHARED_QUERY_BYTES {
        return Err(EngineError::unsupported(
            "cuda.vector_distance",
            format!("dimension {dim} exceeds the shared-memory staging limit"),
        ));
    }
    let n = u32_len(input.num_rows, "vector input")?;
    let dim_u32 = u32_len(dim, "vector dimension")?;
    let metric: i32 = match args.metric {
        DistanceMetric::L2 => 0,
        DistanceMetric::Cosine => 1,
    };
    let query_dev = stream.clone_htod(args.query).map_err(driver_error)?;
    let mut out = alloc::<u8>(stream, input.num_rows * 4)?;
    if n > 0 {
        let f = backend.function("vector_distance", VECTOR_DISTANCE_SRC, "oxide_vec_distance")?;
        let mut launch = stream.launch_builder(&f);
        launch.arg(&*col.values);
        push_validity(&mut launch, &col.validity);
        launch.arg(&n);
        launch.arg(&dim_u32);
        launch.arg(&query_dev);
        launch.arg(&metric);
        launch.arg(&mut out);
        let cfg = LaunchConfig {
            grid_dim: (n.div_ceil(VEC_WARPS_PER_BLOCK).max(1), 1, 1),
            block_dim: (VEC_WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: u32_len(shared_bytes, "shared memory")?,
        };
        // SAFETY: arguments match `oxide_vec_distance`; one warp per row, `out` holds n floats,
        // dynamic shared memory holds the `dim` query floats.
        unsafe { launch.launch(cfg) }.map_err(driver_error)?;
    }
    let mut columns = input.columns.clone();
    columns.push(CudaColumn {
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
        BackendKind::Cuda,
        Arc::new(Schema::new(fields)),
        input.num_rows,
        BatchInner::Cuda(CudaBatch {
            num_rows: input.num_rows,
            columns,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusive_scan_totals() {
        let (offsets, total) = exclusive_scan(&[3, 0, 2, 5]).unwrap_or_default();
        assert_eq!(offsets, vec![0, 3, 3, 5]);
        assert_eq!(total, 10);
        assert!(exclusive_scan(&[u32::MAX, 1]).is_err());
    }

    #[test]
    fn kernel_sources_are_embedded() {
        assert!(FILTER_PROJECT_SRC.contains("oxide_scatter_indices"));
        assert!(HASH_JOIN_SRC.contains("oxide_hj_write"));
        assert!(AGGREGATION_SRC.contains("oxide_agg_minmax_f64"));
        assert!(VECTOR_DISTANCE_SRC.contains("oxide_vec_distance"));
    }
}
