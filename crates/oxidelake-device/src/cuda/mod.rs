//! The CUDA backend (feature `cuda`).
//!
//! Built on cudarc's driver API in **dynamic-loading** mode: `libcuda` and
//! `libnvrtc` are `dlopen`ed at runtime, so this module compiles on machines
//! without a CUDA installation. cudarc *panics* if a symbol is requested while
//! the library is absent, so nothing here touches the driver before
//! [`CudaBackend::driver_present`] has confirmed it can be loaded; without a
//! driver every constructor returns a typed error instead.
//!
//! Kernels are CUDA C sources (`kernels/cuda/*.cu`, embedded with
//! `include_str!`) compiled with NVRTC on first use and cached per backend
//! (see [`CudaBackend::function`]). The operator entry points live in [`ops`]
//! and orchestrate the launches on the caller's stream; anything outside the
//! bounded v1 coverage returns [`EngineError::Unsupported`] so `Gpu*Exec`
//! takes the CPU reference path.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow::array::RecordBatch;
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError};
use cudarc::nvrtc::compile_ptx;
use oxidelake_core::{BackendKind, DeviceId, EngineError};
use oxidelake_memory::HostAlloc;
use oxidelake_memory::pinned::CudaPinnedProvider;

use crate::args::{AggregateArgs, FilterProjectArgs, HashJoinArgs, VectorDistanceArgs};
use crate::backend::{GpuBackend, MemoryInfo};
use crate::columns::{self, ColumnBytes};
use crate::handles::{
    BatchInner, DeviceBatch, DeviceBuffer, DeviceBufferInner, DeviceStream, HostBuffer, StreamInner,
};

pub mod ops;

/// Converts a driver error into the engine error type.
pub(crate) fn driver_error(err: DriverError) -> EngineError {
    EngineError::device(BackendKind::Cuda, err.to_string())
}

pub(crate) fn foreign(what: &str, backend: BackendKind) -> EngineError {
    EngineError::device(
        BackendKind::Cuda,
        format!("{what} belongs to the {backend} backend, not the CUDA backend"),
    )
}

fn to_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// One column resident in device memory: packed values plus optional validity bitmap.
///
/// Buffers are shared (`Arc`) so operators that pass columns through unchanged
/// (vector distance appends a column) do not copy them.
#[derive(Clone)]
pub struct CudaColumn {
    pub(crate) data_type: DataType,
    pub(crate) len: usize,
    pub(crate) child_len: usize,
    pub(crate) values_len: usize,
    pub(crate) values: Arc<CudaSlice<u8>>,
    pub(crate) validity: Option<Arc<CudaSlice<u8>>>,
}

impl CudaColumn {
    /// Device bytes held by this column.
    pub fn device_bytes(&self) -> u64 {
        to_u64(self.values.len() + self.validity.as_ref().map_or(0, |v| v.len()))
    }
}

/// A record batch resident in device memory.
pub struct CudaBatch {
    pub(crate) num_rows: usize,
    pub(crate) columns: Vec<CudaColumn>,
}

impl CudaBatch {
    /// Device bytes held by all columns.
    pub fn device_bytes(&self) -> u64 {
        self.columns.iter().map(CudaColumn::device_bytes).sum()
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }
}

impl fmt::Debug for CudaBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBatch")
            .field("num_rows", &self.num_rows)
            .field("columns", &self.columns.len())
            .field("device_bytes", &self.device_bytes())
            .finish()
    }
}

/// The CUDA backend for one device.
pub struct CudaBackend {
    ctx: Arc<CudaContext>,
    device_id: DeviceId,
    pinned: CudaPinnedProvider,
    modules: Mutex<HashMap<String, Arc<CudaModule>>>,
    /// Resolved kernel entry points, keyed by `(module, entry)`, so a launch
    /// does not pay a `cuModuleGetFunction` lookup per batch.
    functions: Mutex<HashMap<(String, String), CudaFunction>>,
}

impl CudaBackend {
    /// `true` when `libcuda` can be loaded. Performs no driver call.
    pub fn driver_present() -> bool {
        // SAFETY: `is_culib_present` only attempts to `dlopen` the driver
        // library; it issues no CUDA calls and has no preconditions.
        unsafe { cudarc::driver::sys::is_culib_present() }
    }

    /// Number of CUDA devices, or zero when the driver is absent or fails.
    pub fn device_count() -> usize {
        if !Self::driver_present() {
            return 0;
        }
        CudaContext::device_count()
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0)
    }

    /// `true` when at least one device can be used.
    pub fn is_available() -> bool {
        Self::device_count() > 0
    }

    /// Opens device `ordinal`.
    pub fn new(ordinal: usize) -> Result<Self, EngineError> {
        if !Self::driver_present() {
            return Err(EngineError::device(
                BackendKind::Cuda,
                "libcuda could not be loaded: no NVIDIA driver on this machine",
            ));
        }
        let ctx = CudaContext::new(ordinal).map_err(driver_error)?;
        let ordinal = u32::try_from(ordinal)
            .map_err(|_| EngineError::device(BackendKind::Cuda, "device ordinal out of range"))?;
        Ok(Self {
            device_id: DeviceId::new(BackendKind::Cuda, ordinal),
            pinned: CudaPinnedProvider::new(Arc::clone(&ctx)),
            ctx,
            modules: Mutex::new(HashMap::new()),
            functions: Mutex::new(HashMap::new()),
        })
    }

    /// The driver context.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Compiles `source` with NVRTC (once per `name`) and loads it as a module.
    pub fn module(&self, name: &str, source: &str) -> Result<Arc<CudaModule>, EngineError> {
        let mut cache = self.modules.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(module) = cache.get(name) {
            return Ok(Arc::clone(module));
        }
        let ptx = compile_ptx(source).map_err(|e| {
            EngineError::device(
                BackendKind::Cuda,
                format!("NVRTC failed to compile kernel module '{name}': {e:?}"),
            )
        })?;
        let module = self.ctx.load_module(ptx).map_err(driver_error)?;
        cache.insert(name.to_owned(), Arc::clone(&module));
        tracing::debug!(module = name, "compiled CUDA kernel module with NVRTC");
        Ok(module)
    }

    /// Resolves kernel `entry` from module `module_name` (compiling the module
    /// on first use and caching the resolved function afterwards).
    pub fn function(
        &self,
        module_name: &str,
        source: &str,
        entry: &str,
    ) -> Result<CudaFunction, EngineError> {
        let key = (module_name.to_owned(), entry.to_owned());
        if let Some(func) = self
            .functions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
        {
            return Ok(func.clone());
        }
        let func = self
            .module(module_name, source)?
            .load_function(entry)
            .map_err(driver_error)?;
        self.functions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, func.clone());
        Ok(func)
    }

    pub(crate) fn stream<'a>(
        &self,
        stream: &'a DeviceStream,
    ) -> Result<&'a Arc<CudaStream>, EngineError> {
        match &stream.inner {
            StreamInner::Cuda(s) => Ok(s),
            StreamInner::Host => Err(foreign("stream", stream.backend())),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            StreamInner::Metal(_) => Err(foreign("stream", stream.backend())),
        }
    }

    pub(crate) fn upload_bytes(
        stream: &Arc<CudaStream>,
        bytes: &[u8],
    ) -> Result<CudaSlice<u8>, EngineError> {
        if bytes.is_empty() {
            stream.alloc_zeros::<u8>(1).map_err(driver_error)
        } else {
            stream.clone_htod(bytes).map_err(driver_error)
        }
    }

    fn upload_column(
        stream: &Arc<CudaStream>,
        col: &ColumnBytes,
    ) -> Result<CudaColumn, EngineError> {
        let values = Self::upload_bytes(stream, &col.values)?;
        let validity = match &col.validity {
            Some(bits) => Some(Self::upload_bytes(stream, bits)?),
            None => None,
        };
        Ok(CudaColumn {
            data_type: col.data_type.clone(),
            len: col.len,
            child_len: col.child_len,
            values_len: col.values.len(),
            values: Arc::new(values),
            validity: validity.map(Arc::new),
        })
    }

    fn download_column(
        stream: &Arc<CudaStream>,
        col: &CudaColumn,
    ) -> Result<ColumnBytes, EngineError> {
        let mut values = stream.clone_dtoh(&*col.values).map_err(driver_error)?;
        values.truncate(col.values_len);
        let validity = match &col.validity {
            Some(bits) => {
                let mut bytes = stream.clone_dtoh(&**bits).map_err(driver_error)?;
                bytes.truncate(col.len.div_ceil(8));
                Some(Buffer::from(bytes))
            }
            None => None,
        };
        Ok(ColumnBytes {
            data_type: col.data_type.clone(),
            len: col.len,
            // `Buffer::from(Vec)` takes the allocation over: the PCIe copy is
            // the only copy on the way back.
            values: Buffer::from(values),
            validity,
            child_len: col.child_len,
        })
    }
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device_id", &self.device_id)
            .field(
                "modules",
                &self
                    .modules
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .len(),
            )
            .finish()
    }
}

impl GpuBackend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn memory_info(&self) -> Result<MemoryInfo, EngineError> {
        let (free, total) = self.ctx.mem_get_info().map_err(driver_error)?;
        Ok(MemoryInfo {
            total_bytes: to_u64(total),
            free_bytes: to_u64(free),
            pinned_supported: true,
        })
    }

    fn alloc_device(&self, bytes: usize) -> Result<DeviceBuffer, EngineError> {
        let slice = self
            .ctx
            .default_stream()
            .alloc_zeros::<u8>(bytes.max(1))
            .map_err(driver_error)?;
        Ok(DeviceBuffer::new(
            BackendKind::Cuda,
            bytes,
            DeviceBufferInner::Cuda(slice),
        ))
    }

    fn alloc_pinned_host(&self, bytes: usize) -> Result<HostBuffer, EngineError> {
        HostAlloc::pinned_or_pageable(bytes, Some(&self.pinned))
    }

    fn create_stream(&self) -> Result<DeviceStream, EngineError> {
        let stream = self.ctx.new_stream().map_err(driver_error)?;
        Ok(DeviceStream::new(
            BackendKind::Cuda,
            StreamInner::Cuda(stream),
        ))
    }

    fn copy_h2d(
        &self,
        stream: &DeviceStream,
        src: &HostBuffer,
        dst: &mut DeviceBuffer,
    ) -> Result<(), EngineError> {
        let stream = self.stream(stream)?;
        if src.len() > dst.len() {
            return Err(EngineError::execution(format!(
                "copy_h2d: destination holds {} bytes, source has {}",
                dst.len(),
                src.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        let backend = dst.backend();
        let DeviceBufferInner::Cuda(device) = &mut dst.inner else {
            return Err(foreign("buffer", backend));
        };
        match src.as_pinned() {
            Some(pinned) => stream.memcpy_htod(pinned.raw(), device),
            None => stream.memcpy_htod(src.as_slice()?, device),
        }
        .map_err(driver_error)
    }

    fn copy_d2h(
        &self,
        stream: &DeviceStream,
        src: &DeviceBuffer,
        dst: &mut HostBuffer,
    ) -> Result<(), EngineError> {
        let stream = self.stream(stream)?;
        if src.len() > dst.len() {
            return Err(EngineError::execution(format!(
                "copy_d2h: destination holds {} bytes, source has {}",
                dst.len(),
                src.len()
            )));
        }
        if src.is_empty() {
            return Ok(());
        }
        let DeviceBufferInner::Cuda(device) = &src.inner else {
            return Err(foreign("buffer", src.backend()));
        };
        match dst.as_pinned_mut() {
            Some(pinned) => stream.memcpy_dtoh(device, pinned.raw_mut()),
            None => stream.memcpy_dtoh(device, dst.as_mut_slice()?),
        }
        .map_err(driver_error)
    }

    fn synchronize(&self, stream: &DeviceStream) -> Result<(), EngineError> {
        self.stream(stream)?.synchronize().map_err(driver_error)
    }

    fn upload(
        &self,
        stream: &DeviceStream,
        batch: &RecordBatch,
    ) -> Result<DeviceBatch, EngineError> {
        let stream = self.stream(stream)?;
        let extracted = columns::extract_batch(batch)?;
        let columns = extracted
            .iter()
            .map(|c| Self::upload_column(stream, c))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DeviceBatch::new(
            BackendKind::Cuda,
            batch.schema(),
            batch.num_rows(),
            BatchInner::Cuda(CudaBatch {
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
        let stream = self.stream(stream)?;
        match &batch.inner {
            BatchInner::Cuda(resident) => {
                let cols = resident
                    .columns
                    .iter()
                    .map(|c| Self::download_column(stream, c))
                    .collect::<Result<Vec<_>, _>>()?;
                columns::rebuild_batch(batch.schema(), cols)
            }
            BatchInner::Host(host) => Ok(host.clone()),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            BatchInner::Metal(_) => Err(foreign("batch", batch.backend())),
        }
    }

    fn filter_project(
        &self,
        stream: &DeviceStream,
        args: FilterProjectArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        ops::filter_project(self, self.stream(stream)?, args)
    }

    fn hash_join(
        &self,
        stream: &DeviceStream,
        args: HashJoinArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        ops::hash_join(self, self.stream(stream)?, args)
    }

    fn aggregate(
        &self,
        stream: &DeviceStream,
        args: AggregateArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        ops::aggregate(self, self.stream(stream)?, args)
    }

    fn vector_distance(
        &self,
        stream: &DeviceStream,
        args: VectorDistanceArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        ops::vector_distance(self, self.stream(stream)?, args)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use oxidelake_core::params::{AggregateFunction, AggregateSpec};
    use oxidelake_memory::MemoryClass;

    use super::*;

    fn sample() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])),
                Arc::new(Float64Array::from(vec![0.5, 1.5, 2.5])),
            ],
        )
        .unwrap()
    }

    /// Runs on every machine: probing must never panic, and without a driver the
    /// constructor must fail with a typed error.
    #[test]
    fn probing_without_a_driver_is_a_typed_error_not_a_panic() {
        let present = CudaBackend::driver_present();
        let count = CudaBackend::device_count();
        if !present {
            assert_eq!(count, 0);
            assert!(!CudaBackend::is_available());
            let err = CudaBackend::new(0).unwrap_err();
            assert!(
                matches!(
                    err,
                    EngineError::Device {
                        backend: BackendKind::Cuda,
                        ..
                    }
                ),
                "{err}"
            );
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and driver"]
    fn transfers_and_pinned_memory_round_trip_on_device() {
        let backend = CudaBackend::new(0).unwrap();
        let stream = backend.create_stream().unwrap();
        let info = backend.memory_info().unwrap();
        assert!(info.total_bytes > 0 && info.pinned_supported);

        let mut pinned = backend.alloc_pinned_host(64).unwrap();
        assert_eq!(pinned.class(), MemoryClass::Pinned);
        pinned.as_mut_slice().unwrap().copy_from_slice(&[7u8; 64]);
        let mut device = backend.alloc_device(64).unwrap();
        backend.copy_h2d(&stream, &pinned, &mut device).unwrap();
        let mut back = backend.alloc_pinned_host(64).unwrap();
        backend.copy_d2h(&stream, &device, &mut back).unwrap();
        backend.synchronize(&stream).unwrap();
        assert_eq!(back.as_slice().unwrap(), &[7u8; 64]);

        let batch = sample();
        let resident = backend.upload(&stream, &batch).unwrap();
        assert_eq!(resident.backend(), BackendKind::Cuda);
        assert_eq!(backend.download(&stream, &resident).unwrap(), batch);
    }

    /// The shape that failed on hardware first: the demo table's key
    /// distribution (100 keys, 1-in-50 NULL) and value distribution (1-in-40
    /// NULL), 4000 rows in one batch, as `oxidelake-runtime/tests/embedded.rs`
    /// hands it to the kernel after `concat_batches`. Before the fix the
    /// aggregate produced a duplicate group here on 8 of 8 runs on a Tesla T4.
    #[test]
    #[ignore = "requires an NVIDIA GPU and driver"]
    fn aggregate_matches_host_on_the_demo_distribution() {
        const ROWS: usize = 4000;
        const KEYS: i64 = 100;
        fn splitmix64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        let backend = CudaBackend::new(0).unwrap();
        let stream = backend.create_stream().unwrap();

        let mut state = 11u64;
        let mut keys: Vec<Option<i64>> = Vec::with_capacity(ROWS);
        let mut values: Vec<Option<f64>> = Vec::with_capacity(ROWS);
        for _ in 0..ROWS {
            let r = splitmix64(&mut state);
            keys.push((!r.is_multiple_of(50)).then_some((r >> 8) as i64 % KEYS));
            let r = splitmix64(&mut state);
            values.push((!r.is_multiple_of(40)).then(|| f64::from((r >> 8) as u32 % 100) / 4.0));
        }
        // Host reference: (key, sum, count) per group, NULL key as -1.
        let mut expected: std::collections::BTreeMap<i64, (f64, i64)> = Default::default();
        for (k, v) in keys.iter().zip(&values) {
            let e = expected.entry(k.unwrap_or(-1)).or_insert((0.0, 0));
            if let Some(v) = v {
                e.0 += v;
                e.1 += 1;
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(Float64Array::from(values)),
            ],
        )
        .unwrap();
        let resident = backend.upload(&stream, &batch).unwrap();
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![(AggregateFunction::Sum, 1), (AggregateFunction::Count, 1)],
        };
        for round in 0..8 {
            let out = backend
                .aggregate(
                    &stream,
                    AggregateArgs {
                        input: &resident,
                        spec: &spec,
                    },
                )
                .unwrap();
            let host = backend.download(&stream, &out).unwrap();
            let groups = host
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let sums = host
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let counts = host
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let mut seen: std::collections::BTreeMap<i64, (f64, i64)> = Default::default();
            for i in 0..host.num_rows() {
                let k = if groups.is_null(i) {
                    -1
                } else {
                    groups.value(i)
                };
                let dup = seen.insert(
                    k,
                    (
                        if sums.is_null(i) { 0.0 } else { sums.value(i) },
                        counts.value(i),
                    ),
                );
                assert!(dup.is_none(), "round {round}: group {k} emitted twice");
            }
            assert_eq!(seen, expected, "round {round}");
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU and driver"]
    fn nvrtc_pipeline_compiles_and_launches_a_kernel() {
        let backend = CudaBackend::new(0).unwrap();
        let ctx = backend.context();
        let stream = ctx.default_stream();
        let func = backend
            .function(
                "smoke",
                r#"extern "C" __global__ void oxide_smoke(int* out) { if (threadIdx.x == 0 && blockIdx.x == 0) { out[0] = 42; } }"#,
                "oxide_smoke",
            )
            .unwrap();
        let mut out = stream.alloc_zeros::<i32>(1).unwrap();
        let mut launch = stream.launch_builder(&func);
        launch.arg(&mut out);
        // SAFETY: the kernel writes one i32 into a one-element buffer.
        unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }.unwrap();
        let host = stream.clone_dtoh(&out).unwrap();
        assert_eq!(host, vec![42]);
    }
}
