//! The CUDA backend (feature `cuda`).
//!
//! Built on cudarc's driver API in **dynamic-loading** mode: `libcuda` and
//! `libnvrtc` are `dlopen`ed at runtime, so this module compiles on machines
//! without a CUDA installation. cudarc *panics* if a symbol is requested while
//! its library is absent, so nothing here touches the driver before
//! [`CudaBackend::driver_present`] has confirmed it can be loaded, nor NVRTC
//! before [`CudaBackend::nvrtc_present`]; without either, every constructor
//! returns a typed error instead and [`CudaBackend::is_available`] is `false`.
//! The two are separate on purpose: NVRTC ships with the CUDA toolkit, not
//! with the driver, so a driver-only host — or a container that has `libcuda`
//! injected and nothing else — has a working driver and no way to compile a
//! kernel (#106).
//!
//! Kernels are CUDA C sources (`kernels/cuda/*.cu`, embedded with
//! `include_str!`) compiled with NVRTC on first use and cached per backend
//! (see [`CudaBackend::function`]), for the device's own architecture
//! ([`CudaBackend::nvrtc_arch`]); a device below compute capability 7.0 is
//! refused. The operator entry points live in [`ops`] and orchestrate the
//! launches on the caller's stream; anything outside the bounded v1
//! coverage returns [`EngineError::Unsupported`] so `Gpu*Exec` takes the CPU
//! reference path.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow::array::RecordBatch;
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError};
use cudarc::nvrtc::result::NvrtcError;
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use oxidelake_core::{BackendKind, DeviceId, EngineError};
use oxidelake_memory::HostAlloc;
use oxidelake_memory::pinned::CudaPinnedProvider;

use crate::args::{
    AggregateArgs, FilterProjectArgs, HashJoinArgs, JoinBuildArgs, JoinProbeArgs,
    VectorDistanceArgs,
};
use crate::backend::{GpuBackend, MemoryInfo};
use crate::columns::{self, ColumnBytes};
use crate::handles::{
    BatchInner, DeviceBatch, DeviceBuffer, DeviceBufferInner, DeviceJoinMatches, DeviceJoinTable,
    DeviceStream, HostBuffer, StreamInner,
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

const NO_DRIVER: &str = "libcuda could not be loaded: no NVIDIA driver on this machine";

const NO_NVRTC: &str = "libnvrtc could not be loaded: the NVIDIA driver is installed, but the \
     CUDA kernels are compiled at run time with NVRTC, which ships with the CUDA toolkit rather \
     than the driver; install it (the cuda-nvrtc package) or put its directory on LD_LIBRARY_PATH";

/// Lowest architecture the kernels may be compiled for, numbered as NVRTC
/// numbers them (`70` is `compute_70`). `aggregation.cu`'s slot claim spins
/// on a word another lane of the same warp may hold, which only makes
/// progress under independent thread scheduling: Volta (7.0) and later, and
/// only in code compiled for `compute_70` or later — PTX for an older target
/// runs in lockstep even on a newer GPU. Below that, CUDA 12.x NVRTC's
/// default (`compute_52`) also lacks the kernel's `atomicAdd(double*)`, so
/// every aggregate failed to compile there (#107).
const MIN_ARCH: i32 = 70;

/// What [`CudaBackend::probe`] found for one device.
#[derive(Debug, Clone, Copy)]
struct Target {
    /// The device's compute capability, `(major, minor)`.
    capability: (i32, i32),
    /// The NVRTC version, `(major, minor)`.
    nvrtc: (i32, i32),
    /// The architecture the kernels are compiled for (see [`nvrtc_arch`]).
    arch: i32,
}

fn nvrtc_error(what: &str, err: NvrtcError) -> EngineError {
    EngineError::device(BackendKind::Cuda, format!("{what}: {err:?}"))
}

/// The compute capability of device `ordinal`, read without a context.
/// Requires `cuInit`, which `CudaContext::device_count` has made.
fn compute_capability(ordinal: usize) -> Result<(i32, i32), EngineError> {
    let ordinal = i32::try_from(ordinal)
        .map_err(|_| EngineError::device(BackendKind::Cuda, "device ordinal out of range"))?;
    let device = cudarc::driver::result::device::get(ordinal).map_err(driver_error)?;
    let attribute = |attribute| {
        // SAFETY: `device` was returned by `cuDeviceGet` just above, which is
        // all `cuDeviceGetAttribute` requires of it.
        unsafe { cudarc::driver::result::device::get_attribute(device, attribute) }
            .map_err(driver_error)
    };
    Ok((
        attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
        attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
    ))
}

/// The NVRTC version. Call only after [`CudaBackend::nvrtc_present`].
fn nvrtc_version() -> Result<(i32, i32), EngineError> {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: NVRTC is present (checked by the caller), so cudarc's loader
    // finds it; `nvrtcVersion` exists in every NVRTC and writes two ints.
    unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .map_err(|e| nvrtc_error("nvrtcVersion failed", e))?;
    Ok((major, minor))
}

/// The architectures this NVRTC compiles for, numbered as [`MIN_ARCH`] is.
fn nvrtc_supported_archs((major, minor): (i32, i32)) -> Result<Vec<i32>, EngineError> {
    // Both queries arrived in NVRTC 11.2; cudarc resolves symbols lazily and
    // panics on a missing one, so an older library is refused by version.
    if (major, minor) < (11, 2) {
        return Err(EngineError::device(
            BackendKind::Cuda,
            format!("NVRTC {major}.{minor} is too old: OxideLake needs NVRTC 11.2 or newer"),
        ));
    }
    let mut count = 0;
    // SAFETY: NVRTC 11.2+ (checked above) has the function; it writes one int.
    unsafe { cudarc::nvrtc::sys::nvrtcGetNumSupportedArchs(&mut count) }
        .result()
        .map_err(|e| nvrtc_error("nvrtcGetNumSupportedArchs failed", e))?;
    let mut archs = vec![0; usize::try_from(count).unwrap_or(0)];
    if !archs.is_empty() {
        // SAFETY: as above; `archs` has room for exactly the `count` ints
        // the call writes.
        unsafe { cudarc::nvrtc::sys::nvrtcGetSupportedArchs(archs.as_mut_ptr()) }
            .result()
            .map_err(|e| nvrtc_error("nvrtcGetSupportedArchs failed", e))?;
    }
    Ok(archs)
}

/// The architecture kernels for a device of compute capability `capability`
/// are compiled for, given what this NVRTC `supported`: the device's own, or
/// else the newest older one NVRTC knows. PTX for `compute_XY` runs on any
/// device of capability X.Y or newer (the driver compiles it on load), so a
/// GPU newer than its NVRTC still runs; one below 7.0 does not (#107).
fn nvrtc_arch(
    (major, minor): (i32, i32),
    (nvrtc_major, nvrtc_minor): (i32, i32),
    supported: &[i32],
) -> Result<i32, EngineError> {
    let device = major * 10 + minor;
    if device < MIN_ARCH {
        return Err(EngineError::device(
            BackendKind::Cuda,
            format!(
                "the GPU has compute capability {major}.{minor}; OxideLake's CUDA kernels need \
                 7.0 (Volta) or newer, because the aggregation kernel relies on independent \
                 thread scheduling"
            ),
        ));
    }
    supported
        .iter()
        .copied()
        .filter(|arch| (MIN_ARCH..=device).contains(arch))
        .max()
        .ok_or_else(|| {
            let range = match (supported.iter().min(), supported.iter().max()) {
                (Some(lo), Some(hi)) => format!("compute_{lo} to compute_{hi}"),
                _ => "no architecture at all".to_owned(),
            };
            EngineError::device(
                BackendKind::Cuda,
                format!(
                    "NVRTC {nvrtc_major}.{nvrtc_minor} cannot compile for this GPU (compute \
                     capability {major}.{minor}): it targets {range}, and the kernels need one \
                     between compute_{MIN_ARCH} and compute_{device}; install an NVRTC that \
                     still supports this GPU"
                ),
            )
        })
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
    /// `compute_XY`, the `--gpu-architecture` every module is compiled for.
    nvrtc_arch: String,
}

impl CudaBackend {
    /// `true` when `libcuda` can be loaded. Performs no driver call.
    pub fn driver_present() -> bool {
        // SAFETY: `is_culib_present` only attempts to `dlopen` the driver
        // library; it issues no CUDA calls and has no preconditions.
        unsafe { cudarc::driver::sys::is_culib_present() }
    }

    /// `true` when `libnvrtc` can be loaded. Performs no NVRTC call.
    ///
    /// The kernels are compiled with NVRTC on first use, so a backend without
    /// it would fail — inside cudarc, with a panic — on its first query.
    pub fn nvrtc_present() -> bool {
        // SAFETY: `is_culib_present` only attempts to `dlopen` the NVRTC
        // library; it issues no NVRTC calls and has no preconditions.
        unsafe { cudarc::nvrtc::sys::is_culib_present() }
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

    /// Checks, without creating a context, that device `ordinal` has what the
    /// backend needs: a driver, NVRTC, the device itself, a compute
    /// capability of at least 7.0 and an NVRTC that can compile for it. The
    /// error says which is missing; [`Self::new`] runs the same checks first,
    /// and the detector reports this error when it passes CUDA over.
    pub fn probe(ordinal: usize) -> Result<(), EngineError> {
        Self::target(ordinal).map(|_| ())
    }

    fn target(ordinal: usize) -> Result<Target, EngineError> {
        if !Self::driver_present() {
            return Err(EngineError::device(BackendKind::Cuda, NO_DRIVER));
        }
        // Before any NVRTC symbol is touched: cudarc resolves them lazily and
        // panics when the library is missing (#106).
        if !Self::nvrtc_present() {
            return Err(EngineError::device(BackendKind::Cuda, NO_NVRTC));
        }
        let count = CudaContext::device_count().map_err(|e| {
            EngineError::device(
                BackendKind::Cuda,
                format!("the NVIDIA driver could not be initialised: {e}"),
            )
        })?;
        match usize::try_from(count) {
            Ok(n) if ordinal < n => {}
            Ok(0) | Err(_) => {
                return Err(EngineError::device(
                    BackendKind::Cuda,
                    "the NVIDIA driver is installed but reports no CUDA device",
                ));
            }
            Ok(n) => {
                return Err(EngineError::device(
                    BackendKind::Cuda,
                    format!("CUDA device {ordinal} does not exist: the driver reports {n}"),
                ));
            }
        }
        let capability = compute_capability(ordinal)?;
        let nvrtc = nvrtc_version()?;
        let arch = nvrtc_arch(capability, nvrtc, &nvrtc_supported_archs(nvrtc)?)?;
        Ok(Target {
            capability,
            nvrtc,
            arch,
        })
    }

    /// `true` when device 0 can be used: [`Self::probe`] finds everything the
    /// backend needs, NVRTC included.
    pub fn is_available() -> bool {
        Self::probe(0).is_ok()
    }

    /// Opens device `ordinal`.
    pub fn new(ordinal: usize) -> Result<Self, EngineError> {
        let target = Self::target(ordinal)?;
        let ctx = CudaContext::new(ordinal).map_err(driver_error)?;
        let nvrtc_arch = format!("compute_{}", target.arch);
        tracing::info!(
            device = ordinal,
            compute_capability = %format_args!("{}.{}", target.capability.0, target.capability.1),
            nvrtc = %format_args!("{}.{}", target.nvrtc.0, target.nvrtc.1),
            arch = %nvrtc_arch,
            "CUDA backend ready"
        );
        let ordinal = u32::try_from(ordinal)
            .map_err(|_| EngineError::device(BackendKind::Cuda, "device ordinal out of range"))?;
        Ok(Self {
            device_id: DeviceId::new(BackendKind::Cuda, ordinal),
            pinned: CudaPinnedProvider::new(Arc::clone(&ctx)),
            ctx,
            modules: Mutex::new(HashMap::new()),
            functions: Mutex::new(HashMap::new()),
            nvrtc_arch,
        })
    }

    /// The driver context.
    pub const fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The architecture every kernel module is compiled for (`compute_75` on
    /// a T4): the device's own, or the newest older one this NVRTC supports.
    /// Without it NVRTC picked its own default — `compute_52` for CUDA 12.x —
    /// where the aggregation kernel does not compile (#107).
    pub fn nvrtc_arch(&self) -> &str {
        &self.nvrtc_arch
    }

    /// Compiles `source` with NVRTC (once per `name`) and loads it as a module.
    pub fn module(&self, name: &str, source: &str) -> Result<Arc<CudaModule>, EngineError> {
        let mut cache = self.modules.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(module) = cache.get(name) {
            return Ok(Arc::clone(module));
        }
        let options = CompileOptions {
            options: vec![format!("--gpu-architecture={}", self.nvrtc_arch)],
            ..CompileOptions::default()
        };
        let ptx = compile_ptx_with_opts(source, options).map_err(|e| {
            EngineError::device(
                BackendKind::Cuda,
                format!(
                    "NVRTC failed to compile kernel module '{name}' for {}: {e:?}",
                    self.nvrtc_arch
                ),
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
            .field("nvrtc_arch", &self.nvrtc_arch)
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

    fn prepare_join_build(
        &self,
        stream: &DeviceStream,
        args: JoinBuildArgs<'_>,
    ) -> Result<DeviceJoinTable, EngineError> {
        ops::prepare_join_build(self.stream(stream)?, args)
    }

    fn probe_join(
        &self,
        stream: &DeviceStream,
        args: JoinProbeArgs<'_>,
    ) -> Result<DeviceJoinMatches, EngineError> {
        ops::probe_join(self, self.stream(stream)?, args)
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

    /// Lists of the shape `nvrtcGetSupportedArchs` returns. The 13.3 one is
    /// measured (`nvcc --list-gpu-arch` of the CUDA 13.3 toolkit; 13.0
    /// dropped everything below Turing); the 12.x one is illustrative — a
    /// 12.x NVRTC spans `compute_50` to the newest it knows.
    const NVRTC_12_8: [i32; 17] = [
        50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 120,
    ];
    const NVRTC_13_3: [i32; 12] = [75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, 121];

    /// #107: NVRTC used to compile for its own default — `compute_52` under
    /// CUDA 12.x, where `aggregation.cu`'s `atomicAdd(double*)` does not
    /// exist. Now it compiles for the device.
    #[test]
    fn kernels_compile_for_the_devices_own_architecture() {
        assert_eq!(nvrtc_arch((7, 5), (12, 8), &NVRTC_12_8).unwrap(), 75); // T4
        assert_eq!(nvrtc_arch((7, 0), (12, 8), &NVRTC_12_8).unwrap(), 70); // V100
        assert_eq!(nvrtc_arch((8, 6), (13, 3), &NVRTC_13_3).unwrap(), 86);
        assert_eq!(nvrtc_arch((12, 0), (12, 8), &NVRTC_12_8).unwrap(), 120);
    }

    /// PTX for an older architecture runs on a newer device, so a GPU newer
    /// than its NVRTC compiles for the newest architecture NVRTC knows rather
    /// than being refused — which naming the device's own would have done.
    #[test]
    fn a_gpu_newer_than_its_nvrtc_gets_the_newest_architecture_nvrtc_knows() {
        let nvrtc_12_4 = &NVRTC_12_8[..14]; // up to compute_90
        assert_eq!(nvrtc_arch((12, 0), (12, 4), nvrtc_12_4).unwrap(), 90);
        assert_eq!(nvrtc_arch((8, 9), (12, 8), &[70, 75, 80, 86]).unwrap(), 86);
    }

    #[test]
    fn a_gpu_below_volta_is_refused() {
        for (major, minor) in [(6, 1), (6, 0), (5, 2), (0, 0)] {
            let err = nvrtc_arch((major, minor), (12, 8), &NVRTC_12_8).unwrap_err();
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
            let message = err.to_string();
            assert!(
                message.contains(&format!("compute capability {major}.{minor}")),
                "{message}"
            );
            assert!(message.contains("7.0"), "{message}");
        }
    }

    /// CUDA 13 cannot target Volta, so a V100 needs an older NVRTC; the
    /// error says which NVRTC it found and what that one targets.
    #[test]
    fn an_nvrtc_that_cannot_target_the_gpu_is_named() {
        let message = nvrtc_arch((7, 0), (13, 3), &NVRTC_13_3)
            .unwrap_err()
            .to_string();
        assert!(message.contains("NVRTC 13.3"), "{message}");
        assert!(message.contains("compute_75 to compute_121"), "{message}");
        let message = nvrtc_arch((7, 5), (12, 8), &[]).unwrap_err().to_string();
        assert!(message.contains("no architecture"), "{message}");
        // Compiling a Turing GPU's kernels for compute_61 would compile, and
        // run the slot-claim spin-wait in lockstep: a hang, not an error.
        assert!(nvrtc_arch((7, 5), (11, 0), &[52, 60, 61]).is_err());
    }

    /// #107 on hardware: every kernel module compiles for the device. Run it
    /// once with a CUDA 12.x NVRTC first on `LD_LIBRARY_PATH` (whose default
    /// target, `compute_52`, failed `aggregation.cu` before the fix) and
    /// once with the 13.x one.
    #[test]
    #[ignore = "requires an NVIDIA GPU and driver"]
    fn every_kernel_module_compiles_for_the_device_architecture() {
        let backend = CudaBackend::new(0).unwrap();
        let (major, minor) = backend.context().compute_capability().unwrap();
        let arch: i32 = backend
            .nvrtc_arch()
            .strip_prefix("compute_")
            .and_then(|n| n.parse().ok())
            .unwrap();
        assert!(
            (MIN_ARCH..=major * 10 + minor).contains(&arch),
            "{} for a {major}.{minor} device",
            backend.nvrtc_arch()
        );
        for (name, source) in [
            ("filter_project", ops::FILTER_PROJECT_SRC),
            ("hash_join", ops::HASH_JOIN_SRC),
            ("aggregation", ops::AGGREGATION_SRC),
            ("vector_distance", ops::VECTOR_DISTANCE_SRC),
        ] {
            if let Err(e) = backend.module(name, source) {
                panic!("{name} does not compile for {}: {e}", backend.nvrtc_arch());
            }
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

    /// `SUM(Int64)` wraps on overflow on the device exactly as the CPU
    /// reference — and stock DataFusion — does (#46, #100): #100's rows, a
    /// downward wrap, an overflow that ends back in range, and the `NULL`
    /// group, compared with `cpu::kernels::aggregate` value for value.
    #[test]
    #[ignore = "requires an NVIDIA GPU and driver"]
    fn aggregate_wraps_int64_sums_like_the_reference() {
        let backend = CudaBackend::new(0).unwrap();
        let stream = backend.create_stream().unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("x", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![
                    Some(1),
                    Some(1),
                    Some(2),
                    Some(3),
                    Some(3),
                    Some(4),
                    Some(4),
                    Some(4),
                    None,
                    None,
                ])),
                Arc::new(Int64Array::from(vec![
                    i64::MAX - 1,
                    i64::MAX - 1,
                    5,
                    i64::MIN,
                    -1,
                    i64::MAX,
                    1,
                    -2,
                    i64::MAX,
                    i64::MAX,
                ])),
            ],
        )
        .unwrap();
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![(AggregateFunction::Sum, 1), (AggregateFunction::Count, 1)],
        };
        let resident = backend.upload(&stream, &batch).unwrap();
        let out = backend
            .aggregate(
                &stream,
                AggregateArgs {
                    input: &resident,
                    spec: &spec,
                },
            )
            .unwrap();
        let device = backend.download(&stream, &out).unwrap();
        assert_eq!(
            device,
            crate::cpu::kernels::aggregate(&batch, &spec).unwrap()
        );
        let sums = device
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(
            sums.iter().collect::<Vec<_>>(),
            vec![
                Some(-4),
                Some(5),
                Some(i64::MAX),
                Some(i64::MAX - 1),
                Some(-2)
            ]
        );
    }

    /// MIN/MAX(Float64) on the device in IEEE 754 totalOrder, compared with
    /// the CPU reference bit for bit (#101): `NaN` greatest, a `NaN` with the
    /// sign bit set least, `-0.0` below `0.0`, and an all-`NaN` group `NaN`
    /// rather than the +inf/-inf the old kernel's accumulators started at.
    #[test]
    #[ignore = "requires an NVIDIA GPU and driver"]
    fn aggregate_orders_floats_like_the_reference() {
        let backend = CudaBackend::new(0).unwrap();
        let stream = backend.create_stream().unwrap();
        let nan = f64::NAN;
        let rows = [
            (Some(1), 1.0),
            (Some(1), nan),
            (Some(1), 3.0),
            (Some(2), nan),
            (Some(2), nan),
            (Some(3), 0.0),
            (Some(3), -0.0),
            (Some(4), -0.0),
            (Some(4), 0.0),
            (Some(5), f64::INFINITY),
            (Some(5), nan),
            (Some(6), f64::NEG_INFINITY),
            (Some(7), 1.0),
            (Some(7), -nan),
            (None, nan),
            (None, 2.0),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(
                    rows.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![(AggregateFunction::Min, 1), (AggregateFunction::Max, 1)],
        };
        let resident = backend.upload(&stream, &batch).unwrap();
        let out = backend
            .aggregate(
                &stream,
                AggregateArgs {
                    input: &resident,
                    spec: &spec,
                },
            )
            .unwrap();
        let device = backend.download(&stream, &out).unwrap();
        // Arrow compares primitive arrays by their bytes, so this tells
        // -0.0 from 0.0 and a NaN's sign.
        assert_eq!(
            device,
            crate::cpu::kernels::aggregate(&batch, &spec).unwrap()
        );
        let bits = |column: usize| {
            device
                .column(column)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        let want = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let inf = f64::INFINITY;
        assert_eq!(
            bits(1),
            want(&[1.0, nan, -0.0, -0.0, inf, f64::NEG_INFINITY, -nan, 2.0])
        );
        assert_eq!(
            bits(2),
            want(&[nan, nan, 0.0, 0.0, nan, f64::NEG_INFINITY, 1.0, nan])
        );
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
