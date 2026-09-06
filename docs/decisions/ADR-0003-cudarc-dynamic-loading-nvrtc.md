# ADR-0003: `cudarc` with runtime dynamic loading and NVRTC JIT; no `nvcc`

- Status: accepted
- Date: 2026-08-25

## Context

The draft named "cuda-oxide / cudarc" and implied PTX compiled from `.cu` files, i.e. an `nvcc` build step. The development machine has no GPU and no CUDA toolkit; a build-time CUDA dependency would make the `cuda` feature impossible to build or lint here. `cuda-oxide` is at 0.4 and dormant. `cudarc` 0.19.9 is maintained and offers `dynamic-loading` (dlopen through `libloading` at runtime) and `nvrtc`.

## Decision

`cudarc = { version = "0.19", default-features = false, features = ["std", "driver", "nvrtc", "dynamic-loading", "cuda-12080"] }`. Kernels are `.cu` source files embedded with `include_str!`, compiled on first use through NVRTC, with PTX cached per device. Exactly one `cuda-*` API-version feature is enabled (they are alternatives); `cuda-version-from-build-system` is unusable without a toolkit.

## Consequences

`cargo check` and `cargo clippy -p oxidelake-runtime --features cuda` pass on the GPU-less box and are part of the gate. Kernel *execution* is verified only on GPU machines (`cargo test … -- --ignored`) and recorded as such in `STATUS.md`. First-use JIT latency is bounded by the PTX cache. Switching CUDA API version means changing that single feature.
