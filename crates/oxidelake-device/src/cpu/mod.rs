//! The CPU backend: always available, fully functional, and the semantic
//! reference every GPU backend must match.
//!
//! "Device" buffers and batches are host memory; kernels delegate to Arrow
//! compute kernels and rayon (see [`kernels`]).

pub mod kernels;

use arrow::array::RecordBatch;
use oxidelake_core::{BackendKind, DeviceId, EngineError};
use oxidelake_memory::AlignedBuf;

use crate::args::{AggregateArgs, FilterProjectArgs, HashJoinArgs, VectorDistanceArgs};
use crate::backend::{GpuBackend, MemoryInfo};
use crate::handles::{
    DeviceBatch, DeviceBuffer, DeviceBufferInner, DeviceStream, HostBuffer, StreamInner,
};

/// The CPU backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

impl CpuBackend {
    /// Creates the CPU backend.
    pub const fn new() -> Self {
        Self
    }

    fn check_stream(stream: &DeviceStream) -> Result<(), EngineError> {
        match stream.inner {
            StreamInner::Host => Ok(()),
            #[cfg(feature = "cuda")]
            StreamInner::Cuda(_) => Err(foreign("stream", stream.backend())),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            StreamInner::Metal(_) => Err(foreign("stream", stream.backend())),
        }
    }
}

fn foreign(what: &str, backend: BackendKind) -> EngineError {
    EngineError::device(
        BackendKind::CpuSimd,
        format!("{what} belongs to the {backend} backend, not the CPU backend"),
    )
}

/// Host-resident input or a typed error.
pub(crate) fn host_batch(batch: &DeviceBatch) -> Result<&RecordBatch, EngineError> {
    batch
        .as_host()
        .ok_or_else(|| foreign("batch", batch.backend()))
}

/// Total and available host memory from `/proc/meminfo` (zeros elsewhere).
fn host_memory() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
            return (0, 0);
        };
        let field = |key: &str| -> u64 {
            text.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
                .map_or(0, |kb| kb * 1024)
        };
        (field("MemTotal:"), field("MemAvailable:"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        (0, 0)
    }
}

impl GpuBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::CpuSimd
    }

    fn device_id(&self) -> DeviceId {
        DeviceId::CPU
    }

    fn memory_info(&self) -> Result<MemoryInfo, EngineError> {
        let (total_bytes, free_bytes) = host_memory();
        Ok(MemoryInfo {
            total_bytes,
            free_bytes,
            pinned_supported: false,
        })
    }

    fn alloc_device(&self, bytes: usize) -> Result<DeviceBuffer, EngineError> {
        Ok(DeviceBuffer::new(
            BackendKind::CpuSimd,
            bytes,
            DeviceBufferInner::Host(AlignedBuf::device_staging(bytes)?),
        ))
    }

    fn alloc_pinned_host(&self, bytes: usize) -> Result<HostBuffer, EngineError> {
        HostBuffer::pageable(bytes)
    }

    fn create_stream(&self) -> Result<DeviceStream, EngineError> {
        Ok(DeviceStream::new(BackendKind::CpuSimd, StreamInner::Host))
    }

    fn copy_h2d(
        &self,
        stream: &DeviceStream,
        src: &HostBuffer,
        dst: &mut DeviceBuffer,
    ) -> Result<(), EngineError> {
        Self::check_stream(stream)?;
        let backend = dst.backend();
        let dst = dst
            .as_host_mut()
            .ok_or_else(|| foreign("buffer", backend))?;
        let src = src.as_slice()?;
        if dst.len() < src.len() {
            return Err(EngineError::execution(format!(
                "copy_h2d: destination holds {} bytes, source has {}",
                dst.len(),
                src.len()
            )));
        }
        dst.as_mut_slice()[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(
        &self,
        stream: &DeviceStream,
        src: &DeviceBuffer,
        dst: &mut HostBuffer,
    ) -> Result<(), EngineError> {
        Self::check_stream(stream)?;
        let src = src
            .as_host()
            .ok_or_else(|| foreign("buffer", src.backend()))?;
        let dst = dst.as_mut_slice()?;
        if dst.len() < src.len() {
            return Err(EngineError::execution(format!(
                "copy_d2h: destination holds {} bytes, source has {}",
                dst.len(),
                src.len()
            )));
        }
        dst[..src.len()].copy_from_slice(src.as_slice());
        Ok(())
    }

    fn synchronize(&self, stream: &DeviceStream) -> Result<(), EngineError> {
        Self::check_stream(stream)
    }

    fn upload(
        &self,
        stream: &DeviceStream,
        batch: &RecordBatch,
    ) -> Result<DeviceBatch, EngineError> {
        Self::check_stream(stream)?;
        Ok(DeviceBatch::from_host(batch.clone()))
    }

    fn download(
        &self,
        stream: &DeviceStream,
        batch: &DeviceBatch,
    ) -> Result<RecordBatch, EngineError> {
        Self::check_stream(stream)?;
        host_batch(batch).cloned()
    }

    fn filter_project(
        &self,
        stream: &DeviceStream,
        args: FilterProjectArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        Self::check_stream(stream)?;
        let out =
            kernels::filter_project(host_batch(args.input)?, args.predicate, args.projection)?;
        Ok(DeviceBatch::from_host(out))
    }

    fn hash_join(
        &self,
        stream: &DeviceStream,
        args: HashJoinArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        Self::check_stream(stream)?;
        let out = kernels::hash_join(
            host_batch(args.left)?,
            host_batch(args.right)?,
            args.left_key,
            args.right_key,
        )?;
        Ok(DeviceBatch::from_host(out))
    }

    fn aggregate(
        &self,
        stream: &DeviceStream,
        args: AggregateArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        Self::check_stream(stream)?;
        let out = kernels::aggregate(host_batch(args.input)?, args.spec)?;
        Ok(DeviceBatch::from_host(out))
    }

    fn vector_distance(
        &self,
        stream: &DeviceStream,
        args: VectorDistanceArgs<'_>,
    ) -> Result<DeviceBatch, EngineError> {
        Self::check_stream(stream)?;
        let out = kernels::vector_distance(
            host_batch(args.input)?,
            args.column,
            args.query,
            args.metric,
            args.output_name,
        )?;
        Ok(DeviceBatch::from_host(out))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use oxidelake_memory::MemoryClass;

    use super::*;

    #[test]
    fn buffers_round_trip_through_copies() {
        let cpu = CpuBackend::new();
        let stream = cpu.create_stream().unwrap();
        let mut src = cpu.alloc_pinned_host(16).unwrap();
        assert_eq!(src.class(), MemoryClass::Pageable);
        src.as_mut_slice().unwrap().copy_from_slice(&[3u8; 16]);
        let mut dev = cpu.alloc_device(16).unwrap();
        assert!((dev.as_host().unwrap().as_ptr() as usize).is_multiple_of(128));
        cpu.copy_h2d(&stream, &src, &mut dev).unwrap();
        let mut back = cpu.alloc_pinned_host(16).unwrap();
        cpu.copy_d2h(&stream, &dev, &mut back).unwrap();
        cpu.synchronize(&stream).unwrap();
        assert_eq!(back.as_slice().unwrap(), &[3u8; 16]);

        let mut small = cpu.alloc_device(4).unwrap();
        assert!(cpu.copy_h2d(&stream, &src, &mut small).is_err());
    }

    #[test]
    fn upload_download_and_memory_info() {
        let cpu = CpuBackend::new();
        let stream = cpu.create_stream().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 3]))]).unwrap();
        let dev = cpu.upload(&stream, &batch).unwrap();
        assert_eq!(dev.num_rows(), 3);
        assert_eq!(cpu.download(&stream, &dev).unwrap(), batch);
        let info = cpu.memory_info().unwrap();
        assert!(!info.pinned_supported);
        assert!(info.total_bytes >= info.free_bytes);
    }
}
