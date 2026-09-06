//! Resolution of the placement target backend.

use oxidelake_core::{BackendKind, EngineError};

/// Environment variable declaring the cluster's hardware capability for the
/// Ballista scheduler's placement decisions. Unset or empty means `cpu`
/// (no GPU rewrites).
pub const CLUSTER_TARGET_ENV: &str = "OXIDE_CLUSTER_BACKEND";

/// Resolves the declared cluster target from an environment value.
pub fn resolve_cluster_target(value: Option<&str>) -> Result<BackendKind, EngineError> {
    match value.map(str::trim) {
        None | Some("") => Ok(BackendKind::CpuSimd),
        Some(s) => s.parse(),
    }
}

/// Reads [`CLUSTER_TARGET_ENV`] from the process environment.
pub fn cluster_target_from_env() -> Result<BackendKind, EngineError> {
    resolve_cluster_target(std::env::var(CLUSTER_TARGET_ENV).ok().as_deref())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_cpu() {
        assert_eq!(resolve_cluster_target(None).unwrap(), BackendKind::CpuSimd);
        assert_eq!(
            resolve_cluster_target(Some("  ")).unwrap(),
            BackendKind::CpuSimd
        );
    }

    #[test]
    fn parses_gpu_targets_and_rejects_garbage() {
        assert_eq!(
            resolve_cluster_target(Some("cuda")).unwrap(),
            BackendKind::Cuda
        );
        assert_eq!(
            resolve_cluster_target(Some("Metal")).unwrap(),
            BackendKind::Metal
        );
        assert!(resolve_cluster_target(Some("fpga")).is_err());
    }
}
