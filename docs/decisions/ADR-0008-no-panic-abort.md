# ADR-0008: No `panic = "abort"` in the release profile

- Status: accepted
- Date: 2026-08-25

## Context

The draft set `panic = "abort"` for release. `oxidelake-api` may expose PyO3 bindings, where a panic must unwind into a Python exception rather than kill the interpreter; long-running coordinator and worker binaries need backtraces; `catch_unwind`-based isolation in harnesses stops working.

## Decision

Keep the default `panic = "unwind"`. The release profile is `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `strip = "symbols"`.

## Consequences

Marginally larger binaries. A separate size-optimized profile can be added later if a use case appears.
