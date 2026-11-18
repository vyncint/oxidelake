# CUDA kernels

Device code for the `cuda` backend. These are **source files**, not build inputs:
`oxidelake-device` embeds them with `include_str!` and compiles them at runtime with
NVRTC on first use (PTX cached per device). There is no `nvcc` step and no CUDA
toolkit is needed to build OxideLake. To *run* them a machine needs the NVIDIA driver
and NVRTC (`libnvrtc`, part of the CUDA toolkit rather than the driver); with the
driver alone, detection picks the CPU and `OXIDE_BACKEND=cuda` fails naming `libnvrtc`.

Every module is compiled for the GPU's own architecture (`--gpu-architecture=compute_XY`
from its compute capability, or the newest older target the NVRTC supports), never for
NVRTC's default: that is `compute_52` under CUDA 12.x, where `aggregation.cu`'s
`atomicAdd(double*)` does not exist. GPUs below compute capability 7.0 are refused —
`aggregation.cu`'s slot claim spins on a word another lane of its warp may hold, which
needs independent thread scheduling (Volta or newer, in code compiled for `compute_70`
or later).

Conventions (docs/SPEC.md §2.3):

- One `extern "C" __global__` entry point per operator, C ABI, no templates in signatures.
- Explicit grid-size math on the host; every kernel bounds-checks each global access.
- Inputs and outputs are raw Arrow buffers (values + validity bitmaps), 128-byte aligned.

| File | Operator | Notes |
|---|---|---|
| `filter_project.cu` | fused filter + projection | predicate mask + block-scan stream compaction |
| `hash_join.cu` | inner hash join on one `Int64` key | one slot per distinct key (indexed on the host, read-only on the device), parallel probe, output expanded in ranges |
| `aggregation.cu` | `SUM/COUNT/MIN/MAX` grouped by one `Int64` key | open-addressing index per batch (atomic CAS), groups numbered densely on the device, atomic reduction into one entry per group; only the groups are read back |
| `vector_distance.cu` | L2 and cosine distance | shared memory + warp shuffles |

Execution of these kernels is verified only on machines with an NVIDIA GPU
(`cargo test -p oxidelake-compute --features cuda -- --ignored`); see `STATUS.md`.
