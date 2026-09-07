# Architecture decision records

Each record: Status · Date · Context · Decision · Consequences. New decisions take the next number and are linked here. These records exist so that the corrections made to the original plan are not silently undone later.

| ADR | Decision |
|---|---|
| [ADR-0001](ADR-0001-registry-verified-dependency-set.md) | Registry-verified dependency chain with a single root (amended by ADR-0013) |
| [ADR-0002](ADR-0002-stable-toolchain-no-std-simd.md) | Stable toolchain; no `std::simd` |
| [ADR-0003](ADR-0003-cudarc-dynamic-loading-nvrtc.md) | `cudarc` with runtime dynamic loading and NVRTC JIT; no `nvcc` |
| [ADR-0004](ADR-0004-objc2-metal-target-gated.md) | `objc2-metal`, macOS target-gated; MSL compiled at runtime |
| [ADR-0005](ADR-0005-object-safe-sync-gpubackend.md) | Object-safe, synchronous `GpuBackend` |
| [ADR-0006](ADR-0006-pure-cpu-default-features-forwarded.md) | Pure-CPU default build; GPU and io_uring as opt-in features forwarded by `oxidelake-runtime` |
| [ADR-0007](ADR-0007-io-uring-crate-dedicated-thread.md) | `io-uring` 0.7 on a dedicated storage thread instead of `tokio-uring` (amended by ADR-0014) |
| [ADR-0008](ADR-0008-no-panic-abort.md) | No `panic = "abort"` in the release profile |
| [ADR-0009](ADR-0009-datafusion-reexports.md) | Depend on `datafusion` only; use its re-exports (amended by ADR-0013) |
| [ADR-0010](ADR-0010-placement-as-physical-optimizer-rule.md) | Hardware placement is a DataFusion `PhysicalOptimizerRule` |
| [ADR-0011](ADR-0011-tui-testing-layers.md) | TUI testing: `TestBackend` in-process plus `termlens` against a deterministic demo binary |
| [ADR-0012](ADR-0012-gates-and-honesty.md) | Per-phase acceptance gates and the `STATUS.md` honesty rule |
| [ADR-0013](ADR-0013-cluster-mode-on-ballista.md) | Cluster mode runs on Apache DataFusion Ballista; Ballista roots the dependency chain |
| [ADR-0014](ADR-0014-parquet-and-arrow-ipc-not-a-custom-format.md) | Parquet and Arrow IPC instead of a custom `.oxide` file format |
| [ADR-0015](ADR-0015-in-database-inference-at-the-udf-layer.md) | In-database inference (`predict`) at the UDF layer, not the operator layer; `oxmera` with `default-features = false` |
| [ADR-0016](ADR-0016-public-repository-and-crate-rename.md) | Public repository from one commit; crates renamed `oxidelake-*`; kernels inside the device crate |
| [ADR-0017](ADR-0017-ci-dependency-reuse.md) | Dependency caching, shared Metal builds, independent Linux feature gates and tested documentation skips |
