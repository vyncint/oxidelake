//! The unified error type and result alias.

use datafusion::arrow::error::ArrowError;
use datafusion::error::DataFusionError;

use crate::BackendKind;

/// Result alias using [`EngineError`].
pub type Result<T, E = EngineError> = core::result::Result<T, E>;

/// Errors produced by OxideLake library crates.
///
/// Binaries convert this into `anyhow::Error`; DataFusion boundaries convert it
/// into [`DataFusionError::External`] via the provided `From` impl.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// An operating-system or object-store I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A host or device allocation failed.
    #[error("allocation of {bytes} bytes (align {align}) failed: {detail}")]
    Allocation {
        /// Requested size in bytes.
        bytes: usize,
        /// Requested alignment in bytes.
        align: usize,
        /// Backend- or allocator-specific detail.
        detail: String,
    },

    /// A device or driver call failed.
    #[error("{backend} device error: {detail}")]
    Device {
        /// The backend that reported the failure.
        backend: BackendKind,
        /// Driver- or runtime-specific detail.
        detail: String,
    },

    /// Operator execution failed.
    #[error("execution error: {0}")]
    Execution(String),

    /// Planning or plan (de)serialization failed.
    #[error("plan error: {0}")]
    Plan(String),

    /// A file or stream is malformed (corrupt or truncated data).
    #[error("format error: {0}")]
    Format(String),

    /// The requested capability is not implemented on this path.
    ///
    /// This is the only permitted "stub": callers must be able to fall back or
    /// report the gap; panicking placeholders are forbidden.
    #[error("unsupported: {feature} ({detail})")]
    Unsupported {
        /// Stable identifier of the missing capability, e.g. `"metal.hash_join"`.
        feature: &'static str,
        /// Human-readable detail.
        detail: String,
    },

    /// An Arrow error.
    #[error(transparent)]
    Arrow(#[from] ArrowError),

    /// A DataFusion error.
    #[error(transparent)]
    DataFusion(#[from] DataFusionError),
}

impl EngineError {
    /// Builds an [`EngineError::Unsupported`].
    pub fn unsupported(feature: &'static str, detail: impl Into<String>) -> Self {
        Self::Unsupported {
            feature,
            detail: detail.into(),
        }
    }

    /// Builds an [`EngineError::Execution`].
    pub fn execution(detail: impl Into<String>) -> Self {
        Self::Execution(detail.into())
    }

    /// Builds an [`EngineError::Plan`].
    pub fn plan(detail: impl Into<String>) -> Self {
        Self::Plan(detail.into())
    }

    /// Builds an [`EngineError::Format`].
    pub fn format(detail: impl Into<String>) -> Self {
        Self::Format(detail.into())
    }

    /// Builds an [`EngineError::Device`].
    pub fn device(backend: BackendKind, detail: impl Into<String>) -> Self {
        Self::Device {
            backend,
            detail: detail.into(),
        }
    }

    /// Builds an [`EngineError::Allocation`].
    pub fn allocation(bytes: usize, align: usize, detail: impl Into<String>) -> Self {
        Self::Allocation {
            bytes,
            align,
            detail: detail.into(),
        }
    }

    /// Returns `true` for [`EngineError::Unsupported`], the signal operators use
    /// to fall back to the CPU path.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported { .. })
    }
}

impl From<EngineError> for DataFusionError {
    fn from(err: EngineError) -> Self {
        match err {
            EngineError::DataFusion(inner) => inner,
            other => DataFusionError::External(Box::new(other)),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_is_detectable() {
        let err = EngineError::unsupported("metal.hash_join", "not in v1");
        assert!(err.is_unsupported());
        assert_eq!(err.to_string(), "unsupported: metal.hash_join (not in v1)");
    }

    #[test]
    fn datafusion_round_trip_unwraps_inner_error() {
        let inner = DataFusionError::Plan("boom".to_owned());
        let wrapped = EngineError::from(inner);
        let back: DataFusionError = wrapped.into();
        assert!(matches!(back, DataFusionError::Plan(ref m) if m == "boom"));
    }

    #[test]
    fn other_errors_become_external() {
        let back: DataFusionError = EngineError::format("truncated footer").into();
        assert!(matches!(back, DataFusionError::External(_)));
        assert!(back.to_string().contains("truncated footer"));
    }

    #[test]
    fn io_errors_convert() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err: EngineError = io.into();
        assert!(matches!(err, EngineError::Io(_)));
    }
}
