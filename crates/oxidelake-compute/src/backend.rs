//! Process-wide backend selection for operators.

use std::sync::{Arc, OnceLock};

use oxidelake_core::EngineError;
use oxidelake_device::{GpuBackend, HardwareDetector};

static LOCAL: OnceLock<Result<Arc<dyn GpuBackend>, String>> = OnceLock::new();

/// The backend `Gpu*Exec` operators run on in this process: detected once
/// (honouring `OXIDE_BACKEND`) and shared. A failed selection is remembered and
/// reported as an error on every call — an unavailable forced backend is a
/// startup error, never a silent fallback.
pub fn local_backend() -> Result<Arc<dyn GpuBackend>, EngineError> {
    match LOCAL.get_or_init(|| HardwareDetector::select().map_err(|e| e.to_string())) {
        Ok(backend) => Ok(Arc::clone(backend)),
        Err(message) => Err(EngineError::execution(format!(
            "backend selection failed: {message}"
        ))),
    }
}
