//! Hardware abstraction layer (docs/SPEC.md §2.2): the object-safe [`GpuBackend`]
//! trait, opaque backend-tagged handles, the always-available [`CpuBackend`]
//! (the correctness reference), the opt-in CUDA and Metal backends, and the
//! [`HardwareDetector`] that picks one at runtime.
//!
//! GPU backends are cargo features (`cuda`, `metal`) and never part of the
//! default build; `metal` is additionally macOS-only.
//!
//! Dependency direction: depends on `oxidelake-core` and `oxidelake-memory`.

pub mod args;
pub mod backend;
pub mod columns;
pub mod cpu;
pub mod detector;
pub mod handles;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

pub use args::{AggregateArgs, FilterProjectArgs, HashJoinArgs, VectorDistanceArgs};
pub use backend::{GpuBackend, MemoryInfo};
pub use cpu::CpuBackend;
pub use detector::{Availability, BACKEND_ENV, HardwareDetector};
pub use handles::{DeviceBatch, DeviceBuffer, DeviceStream, HostBuffer};
pub use oxidelake_core::{BackendKind, DeviceId};
