# CUDA kernels

Device code for the `cuda` backend. These are **source files**, not build inputs:
`oxidelake-device` embeds them with `include_str!` and compiles them at runtime with
NVRTC on first use (PTX cached per device). There is no `nvcc` step and no CUDA
toolkit is needed to build OxideLake; a driver is needed only to *run* them.

Conventions (docs/SPEC.md §2.3):

- One `extern "C" __global__` entry point per operator, C ABI, no templates in signatures.
- Explicit grid-size math on the host; every kernel bounds-checks each global access.
- Inputs and outputs are raw Arrow buffers (values + validity bitmaps), 128-byte aligned.

| File | Operator | Notes |
|---|---|---|
| `filter_project.cu` | fused filter + projection | predicate mask + block-scan stream compaction |
| `hash_join.cu` | inner hash join on one `Int64` key | open addressing, atomic CAS build, parallel probe |
| `aggregation.cu` | `SUM/COUNT/MIN/MAX` grouped by one `Int64` key | atomic / radix reduction |
| `vector_distance.cu` | L2 and cosine distance | shared memory + warp shuffles |

Execution of these kernels is verified only on machines with an NVIDIA GPU
(`cargo test -p oxidelake-compute --features cuda -- --ignored`); see `STATUS.md`.
