# ADR-0001: Registry-verified dependency chain with a single root

- Status: accepted (amended 2026-08-25 by ADR-0013)
- Date: 2026-08-25

## Context

The first draft pinned `arrow 53`, `datafusion 43`, `tonic 0.12`, `prost 0.13`, `ratatui 0.28`. On 2026-08-25 the registry has DataFusion 55 (requires `arrow ^59.2`), arrow-flight 59.2 (requires `tonic` / `prost ^0.14`) and ratatui 0.30 (pairs with crossterm 0.29). Mixing majors in the Arrow ecosystem produces two copies of every arrow crate and type mismatches at every crate boundary — the most common way a Rust/Arrow workspace fails to compile.

## Decision

Resolve the set against the live registry and record every requirement in `docs/dependencies.md`. The chain has one root that fixes every other major; `cargo tree --workspace -d -e normal` showing no duplicate `arrow-*`, `parquet`, `datafusion*`, `object_store`, `tonic` or `prost` majors is part of the gate. Any bump re-derives the whole chain.

**Amendment (ADR-0013):** the root is Ballista, not DataFusion. Ballista's documentation requires `datafusion` to match Ballista's version, so the current chain is ballista 54.1 → datafusion / datafusion-proto 54.1 → arrow / parquet 58 → arrow-flight 58 (transitive) → tonic / prost 0.14, and `rust-version = "1.88"`. DataFusion 55 stays unused until Ballista moves.

## Consequences

One Arrow, one Parquet, one DataFusion in the graph. Upgrades are all-or-nothing and follow the procedure in `docs/dependencies.md`, starting from `cargo info ballista`.
