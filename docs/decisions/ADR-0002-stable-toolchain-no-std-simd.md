# ADR-0002: Stable toolchain; no `std::simd`

- Status: accepted
- Date: 2026-08-25

## Context

The draft required `std::simd` for the CPU fallback. `std::simd` (`portable_simd`) is nightly-only. A nightly pin makes the MSRV meaningless, weakens `rust-toolchain.toml` reproducibility, and gains little: Arrow compute kernels are already vectorized and DataFusion's operators are the CPU path.

## Decision

Stable Rust 1.98.0, pinned in `rust-toolchain.toml`. CPU vectorization comes from Arrow/DataFusion kernels, `rayon` data parallelism and auto-vectorization. The `wide` crate is the stable escape hatch, added only after profiling shows a hot loop that needs explicit SIMD.

## Consequences

No nightly features anywhere in the workspace. Explicit SIMD becomes a measured, later decision rather than a day-one constraint.
