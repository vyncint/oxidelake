//! Core types shared by every OxideLake crate.
//!
//! * [`EngineError`] — the unified error type every library crate converts into.
//! * [`BackendKind`], [`DeviceId`] — hardware identifiers.
//! * [`BufferLayout`] — alignment/length descriptor for host and device buffers.
//! * [`params`] — operator parameter types shared by planner, codec and backends.
//! * [`TelemetryHub`] — the only coupling between the engine and the dashboard.
//! * [`BatchStream`] — the stream type every operator produces (DataFusion's).
//!
//! Dependency direction: `oxidelake-core` depends on no other OxideLake crate.

#![forbid(unsafe_code)]

pub mod error;
pub mod hardware;
pub mod layout;
pub mod params;
pub mod telemetry;

pub use error::{EngineError, Result};
pub use hardware::{BackendKind, DeviceId};
pub use layout::{BufferLayout, DEVICE_ALIGN, HOST_ALIGN};
pub use telemetry::{TelemetryHub, TelemetrySnapshot};

/// A stream of Arrow record batches, as produced by every OxideLake operator.
pub type BatchStream = datafusion::physical_plan::SendableRecordBatchStream;
