//! Runtime backend selection: probe CUDA → Metal → CPU, honouring `OXIDE_BACKEND`.

use std::sync::Arc;

use oxidelake_core::{BackendKind, EngineError};

use crate::backend::GpuBackend;
use crate::cpu::CpuBackend;

/// Environment variable that forces a backend (`cpu`, `cuda` or `metal`).
/// Requesting a backend that is unavailable is a startup error, never a silent
/// fallback.
pub const BACKEND_ENV: &str = "OXIDE_BACKEND";

/// Which GPU backends can actually run on this machine right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Availability {
    /// A CUDA driver and at least one device were found (requires feature `cuda`).
    pub cuda: bool,
    /// A Metal device was found (requires feature `metal`, macOS).
    pub metal: bool,
}

impl Availability {
    /// The backend the detector prefers: CUDA, then Metal, then CPU.
    pub const fn best(self) -> BackendKind {
        if self.cuda {
            BackendKind::Cuda
        } else if self.metal {
            BackendKind::Metal
        } else {
            BackendKind::CpuSimd
        }
    }
}

/// Probes the hardware and instantiates a backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct HardwareDetector;

impl HardwareDetector {
    /// Probes without instantiating anything. Never panics, even without drivers.
    pub fn availability() -> Availability {
        Availability {
            cuda: cuda_available(),
            metal: metal_available(),
        }
    }

    /// The backend [`Self::select`] would choose without an override.
    pub fn detected_kind() -> BackendKind {
        Self::availability().best()
    }

    /// Selects a backend, honouring [`BACKEND_ENV`] from the process environment.
    pub fn select() -> Result<Arc<dyn GpuBackend>, EngineError> {
        Self::select_with(std::env::var(BACKEND_ENV).ok().as_deref())
    }

    /// Selects a backend, honouring an explicit override value (the contents of
    /// [`BACKEND_ENV`]). Empty or missing means "best available".
    pub fn select_with(requested: Option<&str>) -> Result<Arc<dyn GpuBackend>, EngineError> {
        let availability = Self::availability();
        let kind = match requested.map(str::trim).filter(|s| !s.is_empty()) {
            Some(name) => name.parse::<BackendKind>()?,
            None => availability.best(),
        };
        Self::instantiate(kind, availability)
    }

    fn instantiate(
        kind: BackendKind,
        availability: Availability,
    ) -> Result<Arc<dyn GpuBackend>, EngineError> {
        match kind {
            BackendKind::CpuSimd => Ok(Arc::new(CpuBackend::new())),
            BackendKind::Cuda => {
                if !availability.cuda {
                    return Err(unavailable(kind, cfg!(feature = "cuda")));
                }
                instantiate_cuda()
            }
            BackendKind::Metal => {
                if !availability.metal {
                    return Err(unavailable(
                        kind,
                        cfg!(all(feature = "metal", target_os = "macos")),
                    ));
                }
                instantiate_metal()
            }
        }
    }
}

fn unavailable(kind: BackendKind, compiled_in: bool) -> EngineError {
    let detail = if compiled_in {
        format!(
            "requested via {BACKEND_ENV} but no {kind} device or driver is present on this machine"
        )
    } else {
        format!("requested via {BACKEND_ENV} but OxideLake was built without the `{kind}` feature")
    };
    EngineError::device(kind, detail)
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    crate::cuda::CudaBackend::is_available()
}

#[cfg(not(feature = "cuda"))]
fn cuda_available() -> bool {
    false
}

#[cfg(feature = "cuda")]
fn instantiate_cuda() -> Result<Arc<dyn GpuBackend>, EngineError> {
    Ok(Arc::new(crate::cuda::CudaBackend::new(0)?))
}

#[cfg(not(feature = "cuda"))]
fn instantiate_cuda() -> Result<Arc<dyn GpuBackend>, EngineError> {
    Err(unavailable(BackendKind::Cuda, false))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_available() -> bool {
    crate::metal::MetalBackend::is_available()
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_available() -> bool {
    false
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn instantiate_metal() -> Result<Arc<dyn GpuBackend>, EngineError> {
    Ok(Arc::new(crate::metal::MetalBackend::new()?))
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn instantiate_metal() -> Result<Arc<dyn GpuBackend>, EngineError> {
    Err(unavailable(BackendKind::Metal, false))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_selection_matches_detection_and_is_cpu_without_gpus() {
        let availability = HardwareDetector::availability();
        let backend = HardwareDetector::select_with(None).unwrap();
        assert_eq!(backend.kind(), HardwareDetector::detected_kind());
        if !availability.cuda && !availability.metal {
            assert_eq!(backend.kind(), BackendKind::CpuSimd);
        }
    }

    #[test]
    fn explicit_cpu_and_whitespace_are_accepted() {
        assert_eq!(
            HardwareDetector::select_with(Some(" CPU ")).unwrap().kind(),
            BackendKind::CpuSimd
        );
        assert_eq!(
            HardwareDetector::select_with(Some("")).unwrap().kind(),
            HardwareDetector::detected_kind()
        );
    }

    #[test]
    fn unavailable_backends_are_typed_errors() {
        let availability = HardwareDetector::availability();
        if !availability.cuda {
            let err = HardwareDetector::select_with(Some("cuda")).unwrap_err();
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
            assert!(err.to_string().contains(BACKEND_ENV));
        }
        if !availability.metal {
            let err = HardwareDetector::select_with(Some("metal")).unwrap_err();
            assert!(
                matches!(
                    err,
                    EngineError::Device {
                        backend: BackendKind::Metal,
                        ..
                    }
                ),
                "{err}"
            );
        }
        assert!(matches!(
            HardwareDetector::select_with(Some("fpga")).unwrap_err(),
            EngineError::Plan(_)
        ));
    }
}
