# ADR-0006: Pure-CPU default build; GPU and io_uring as opt-in features forwarded by `oxidelake-runtime`

- Status: accepted
- Date: 2026-08-25

## Context

The draft had no feature or platform gating: CUDA, Metal and io_uring code would compile unconditionally, so the workspace could never build on the Linux/no-GPU development box, nor on macOS.

## Decision

Default features are pure CPU with zero GPU or OS-specific dependencies. `cuda`, `metal` and `io-uring` are cargo features on the implementing crates and are forwarded by `oxidelake-runtime` and `oxidelake-api`. Because a virtual workspace root rejects `--features`, `cargo check -p oxidelake-runtime --features cuda` is the single GPU compile gate.

## Consequences

Every environment gets a green default build. GPU code stays compile-checked in CI without hardware. Feature combinations are enumerable and tested per ../SPEC.md §5.
