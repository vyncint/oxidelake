# ADR-0009: Depend on `datafusion` only; use its re-exports

- Status: accepted (amended 2026-08-25 by ADR-0013)
- Date: 2026-08-25

## Context

The draft pinned `datafusion-expr` and `datafusion-physical-plan` separately. They must match `datafusion` exactly; independent pins can skew after any update and yield duplicate crates with incompatible types.

## Decision

Code uses `datafusion::logical_expr`, `datafusion::physical_plan`, `datafusion::physical_optimizer`, and the other re-exports; no `datafusion-*` sub-crate is declared.

**Amendment (ADR-0013):** the single exception is `datafusion-proto`, which provides the `PhysicalExtensionCodec` trait Ballista needs to ship `Gpu*Exec` nodes to executors and is not re-exported by `datafusion`. It is pinned to the identical version as `datafusion`, and the duplicate gate covers `datafusion*`.

## Consequences

One version to bump (plus its proto twin). The coherence gate stays simple.
