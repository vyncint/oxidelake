# Metal kernels

Metal Shading Language source for the `metal` backend (macOS, Apple silicon).
`oxidelake-device` embeds these files with `include_str!` and compiles them at
runtime through `MTLDevice::newLibraryWithSource`; there is no Xcode build step.
Buffers use `MTLResourceStorageModeShared`, so CPU and GPU share one allocation.

| File | Operator |
|---|---|
| `filter_project.metal` | fused filter + projection |
| `vector_distance.metal` | L2 and cosine distance |

Joins and aggregates stay on the CPU under Metal in v1 (docs/SPEC.md §2.3). Metal
code is compiled and executed only on macOS; on Linux the `metal` feature is a
no-op. See `STATUS.md` for what has actually been verified.
