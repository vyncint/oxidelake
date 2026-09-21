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

/// Selects the process backend explicitly, ahead of anything that calls
/// [`local_backend`]. `requested` takes the same values as `OXIDE_BACKEND`
/// (`cpu`, `cuda`, `metal`); `None` means "best available" and ignores the
/// environment, so a caller that wants the environment honoured passes what
/// it read from there.
///
/// This exists so a command-line flag can win over `OXIDE_BACKEND` and still
/// reach the operators: they read [`local_backend`], which memoises its first
/// answer, so setting the choice after the first query would silently do
/// nothing. An unavailable backend is an error here exactly as it is through
/// the environment — never a silent fallback to the CPU.
///
/// Returns an error if a backend was already selected in this process: the
/// operators that have already run did so on the old one, and re-pointing
/// them mid-process would make a single query's backend a matter of timing.
pub fn init_local_backend(requested: Option<&str>) -> Result<Arc<dyn GpuBackend>, EngineError> {
    let selected = HardwareDetector::select_with(requested).map_err(|e| e.to_string());
    if LOCAL.set(selected).is_err() {
        return Err(EngineError::execution(
            "the process backend was already selected; init_local_backend must \
             run before the first local_backend() call",
        ));
    }
    local_backend()
}
