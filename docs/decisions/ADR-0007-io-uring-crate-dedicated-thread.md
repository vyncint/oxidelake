# ADR-0007: `io-uring` 0.7 on a dedicated storage thread instead of `tokio-uring`

- Status: accepted (amended 2026-08-25 by ADR-0014)
- Date: 2026-08-25

## Context

The draft asked for "async Linux io_uring" without naming a crate. `tokio-uring` is at 0.5.0, pins `io-uring ^0.6`, and requires its own current-thread runtime, which fights a multi-threaded tokio application. The low-level `io-uring` crate (tokio-rs) is at 0.7 and actively maintained.

## Decision

Behind the `io-uring` feature (Linux target-gated), a dedicated storage thread owns the ring, submits reads and writes, and completes `oneshot` futures. Tests skip with a logged reason if `io_uring_setup` returns EPERM/ENOSYS (sandboxed CI).

**Amendment (ADR-0014):** the fast path is exposed as `UringLocalFileSystem`, an implementation of `object_store::ObjectStore` — the trait DataFusion reads through — rather than a private `StorageIo` trait. The default is `object_store::local::LocalFileSystem`; both implementations pass one shared conformance test, and the spill manager's Arrow IPC files go through the same store. The macOS mmap path is dropped.

## Consequences

No runtime coupling. Linux-only fast path that also accelerates DataFusion's Parquet scans, not just our own files. `async-trait` appears in the workspace solely because `ObjectStore` is a foreign `#[async_trait]` trait.
