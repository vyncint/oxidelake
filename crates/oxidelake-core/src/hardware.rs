//! Hardware backend and device identifiers.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::EngineError;

/// The execution backends OxideLake knows about.
///
/// `CpuSimd` is always available and is the correctness reference; the GPU
/// backends are opt-in cargo features and are selected at runtime by the
/// hardware detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    /// Vectorized CPU execution through Arrow/DataFusion kernels and rayon.
    CpuSimd,
    /// NVIDIA CUDA (feature `cuda`).
    Cuda,
    /// Apple Metal on unified memory (feature `metal`, macOS only).
    Metal,
}

impl BackendKind {
    /// Every backend kind, in detection priority order (GPU first).
    pub const ALL: [BackendKind; 3] = [BackendKind::Cuda, BackendKind::Metal, BackendKind::CpuSimd];

    /// The canonical lowercase name used in environment variables, `EXPLAIN`
    /// tags and logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            BackendKind::CpuSimd => "cpu",
            BackendKind::Cuda => "cuda",
            BackendKind::Metal => "metal",
        }
    }

    /// `true` for backends that execute on a device rather than the host CPU.
    pub const fn is_gpu(self) -> bool {
        !matches!(self, BackendKind::CpuSimd)
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BackendKind {
    type Err = EngineError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" | "cpusimd" | "cpu-simd" | "cpu_simd" => Ok(BackendKind::CpuSimd),
            "cuda" => Ok(BackendKind::Cuda),
            "metal" => Ok(BackendKind::Metal),
            other => Err(EngineError::plan(format!(
                "unknown backend '{other}'; expected one of cpu, cuda, metal"
            ))),
        }
    }
}

/// Identifies one device of a backend, e.g. `cuda:0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId {
    /// The backend the device belongs to.
    pub backend: BackendKind,
    /// The device ordinal within that backend (always `0` for the CPU).
    pub ordinal: u32,
}

impl DeviceId {
    /// The host CPU.
    pub const CPU: DeviceId = DeviceId {
        backend: BackendKind::CpuSimd,
        ordinal: 0,
    };

    /// Builds a device identifier.
    pub const fn new(backend: BackendKind, ordinal: u32) -> Self {
        Self { backend, ordinal }
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.backend, self.ordinal)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_names_case_insensitively() {
        assert_eq!("cpu".parse::<BackendKind>().unwrap(), BackendKind::CpuSimd);
        assert_eq!(" CUDA ".parse::<BackendKind>().unwrap(), BackendKind::Cuda);
        assert_eq!("Metal".parse::<BackendKind>().unwrap(), BackendKind::Metal);
        assert!("tpu".parse::<BackendKind>().is_err());
    }

    #[test]
    fn display_round_trips_through_parse() {
        for kind in BackendKind::ALL {
            assert_eq!(kind.as_str().parse::<BackendKind>().unwrap(), kind);
        }
    }

    #[test]
    fn device_id_display() {
        assert_eq!(DeviceId::new(BackendKind::Cuda, 1).to_string(), "cuda:1");
        assert_eq!(DeviceId::CPU.to_string(), "cpu:0");
        assert!(!BackendKind::CpuSimd.is_gpu());
        assert!(BackendKind::Metal.is_gpu());
    }
}
