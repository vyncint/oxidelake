# ADR-0004: `objc2-metal`, macOS target-gated; MSL compiled at runtime

- Status: accepted
- Date: 2026-08-25

## Context

The draft named "metal-rs / objc2-metal". `objc2-metal` 0.3 (with `objc2` 0.6 and `objc2-foundation` 0.3) is the maintained binding. The development machine is Linux, so Metal must not touch the build there.

## Decision

Metal dependencies are declared only under `[target.'cfg(target_os = "macos")'.dependencies]`; Metal code is `#[cfg(all(feature = "metal", target_os = "macos"))]`. `.metal` sources are embedded and compiled with `MTLDevice::newLibraryWithSource` — no Xcode build step. Buffers use `MTLResourceStorageModeShared` (unified memory, zero-copy).

## Consequences

Enabling `metal` on Linux compiles as a no-op. Metal correctness is verified only on macOS. In v1 Metal covers filter + project and vector distance; joins and aggregates stay on CPU there, and the docs say so.
