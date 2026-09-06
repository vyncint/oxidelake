//! Memory management for OxideLake (docs/SPEC.md §2.1): aligned host buffers,
//! pinned (CUDA) and unified (Metal) allocations, Arrow IPC encoding for
//! spill files, and the three-tier spill manager.
//!
//! * [`AlignedBuf`] — zeroed, explicitly aligned host allocation.
//! * [`HostAlloc`] — host buffer that is pinned when a [`PinnedProvider`] can
//!   deliver it and pageable otherwise; the class is observable.
//! * [`SpillManager`] — device → host → disk demotion with LRU eviction; the
//!   disk tier is Arrow IPC written through an `object_store::ObjectStore`.
//!
//! Dependency direction: depends on `oxidelake-core` only.

pub mod aligned;
pub mod class;
pub mod host;
pub mod ipc;
pub mod spill;

#[cfg(feature = "cuda")]
pub mod pinned;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

pub use aligned::AlignedBuf;
pub use class::MemoryClass;
pub use host::{HostAlloc, PinnedProvider};
pub use spill::{BatchId, DeviceResident, SpillBudget, SpillManager, Tier};
