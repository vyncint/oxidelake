# ADR-0012: Per-phase acceptance gates and the `STATUS.md` honesty rule

- Status: accepted
- Date: 2026-08-25

## Context

The draft said "implement and verify" with no definition of done, and its verification section could not run as written (no feature gating; GPU tests on a machine without a GPU). Autonomous execution needs objective stop conditions and an auditable record.

## Decision

Every phase has acceptance criteria (`docs/roadmap.md`) and ends with the gate (`docs/verification.md`) green and one commit. `STATUS.md` keeps a matrix — executed here / compiled only / needs GPU or macOS. No fabricated benchmarks; anything phrased as achieved must have run. Stubs are typed `EngineError::Unsupported`, never `todo!()`.

## Consequences

Progress is measurable per commit. Claims are auditable. GPU-dependent work is explicitly deferred instead of silently assumed.
