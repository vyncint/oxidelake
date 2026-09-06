# ADR-0005: Object-safe, synchronous `GpuBackend`

- Status: accepted
- Date: 2026-08-25

## Context

The draft trait had associated types (`type DeviceBuffer; type Stream;`) and `#[async_trait]` on methods that were not async. Associated types prevent `dyn GpuBackend`, yet `HardwareDetector` must choose CUDA, Metal or CPU at runtime, which needs a trait object (or enum dispatch). `async-trait` was also absent from the dependency list.

## Decision

No associated types. Opaque RAII handle structs `DeviceBuffer`, `HostBuffer` and `DeviceStream` carry backend-tagged internals and free themselves on `Drop`. The trait is synchronous; asynchrony lives on device streams and at the operator layer (oneshot bridges, `spawn_blocking`). DataFusion's `ExecutionPlan::execute` is synchronous as well, so no async trait is needed anywhere.

## Consequences

`Arc<dyn GpuBackend>` everywhere. Backends downcast handle internals privately and return `EngineError::Device` on a handle from another backend. The trait is small enough to mock in planner tests. (`async-trait` later entered the workspace only to implement `object_store`'s foreign `ObjectStore` trait — see ADR-0007 — never for our own traits.)
