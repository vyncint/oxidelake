# Docs

- [roadmap.md](roadmap.md) — phases 0–7 as checklists with acceptance criteria
- [architecture.md](architecture.md) — crate map, execution flow, memory tiers, hardware abstraction, storage (Parquet, Arrow IPC, object store), TUI
- [dependencies.md](dependencies.md) — pinned set, coherence evidence, re-derivation procedure, MSRV
- [RELEASING.md](RELEASING.md) — cutting a release: nine crates, Trusted Publishing, the first-publish exception
- [verification.md](verification.md) — the gate (`make gate`), CI mapping, GPU-machine commands, honesty rule
- [audit-2026-09-02.md](audit-2026-09-02.md) — security and performance audit: findings, fixes, deferred items with plans
- [decisions/](decisions/README.md) — architecture decision records

Project-level documents live at the repository root: [README](../README.md),
[CONTRIBUTING](../CONTRIBUTING.md), [SECURITY](../SECURITY.md),
[CHANGELOG](../CHANGELOG.md), [STATUS](../STATUS.md).

The specification is [SPEC.md](SPEC.md). These documents elaborate and track it; when they disagree, SPEC.md wins and the document is fixed.
