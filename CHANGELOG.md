# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0, minor
versions (0.x) may contain breaking changes; they are always listed under a
**Changed** or **Removed** heading. The nine crates version together.

## [Unreleased]

### Changed

- **Release profile: thin LTO instead of fat.** Fat LTO with one codegen unit
  made each binary's link a 13.6 GB whole-program pass; three in parallel
  killed the 16 GB Linux release runners and left the 7 GB macOS runners
  swapping for four hours per target. Thin LTO with one codegen unit peaks
  at 7.4 GB and links in half the time; binaries grow about 9% (`oxide`
  68 → 74 MB) and a 10M-row aggregate ran in the same 40 ms under both.
  `binaries.yml` also gains a 90-minute timeout so a swapping link fails
  loudly instead of holding a macOS slot.

### Fixed

- **Linux release binaries build again.** `binaries.yml` links the three
  binaries one at a time: the release profile's fat LTO made Cargo run three
  whole-program links in parallel, which exhausted the 16 GB Linux runners
  and ended every musl job with exit 143 (v0.1.0 and v0.1.2 both shipped
  without Linux archives). A re-run against an existing tag now skips targets
  whose archive is already attached instead of rebuilding them.

## [0.1.2] - 2026-09-07

CI and build-gate changes only; no crate code changed since 0.1.1.

### Changed

- **CI dependency reuse and shorter queues.** Linux default/io-uring/predict
  tests run independently; the Metal and PTY tests share a dependency graph
  with the conformance pass. External dependency caches accelerate ordinary
  test jobs while release CI keeps a clean full gate. Superseded ordinary
  main CI is cancelled; known documentation-only changes can skip Rust jobs
  under a tested required-check policy. Stress builds once per OS and retains
  all three thread counts and their iteration weights. Cargo timing artifacts
  make compilation costs inspectable; no benchmarked speedup is claimed yet.
- `make gate` includes MSRV, the complete macOS Metal lane, and all-feature
  documentation and dependency-policy checks to match the CI requirements.

### Fixed

- The crate-metadata gate runs under macOS's bundled Bash 3.2, preserving
  empty-field validation without the unavailable `mapfile` builtin.
- The MSRV target invokes Rustup explicitly, so a standalone Cargo earlier
  on `PATH` cannot bypass the declared compiler selection.

## [0.1.1] - 2026-09-07

Metadata-only for the published crates; no code changed since 0.1.0 beyond
what this section lists.

### Fixed

- **crates.io showed the nine crates with no README, repository, homepage,
  keywords or categories.** `[workspace.package]` declared all of them, but a
  workspace field reaches a crate only when its manifest opts in with
  `field.workspace = true`, and the manifests inherited only
  version/edition/rust-version/license/authors. Every crate now inherits the
  rest and carries its own `documentation = "https://docs.rs/<crate>"`;
  `make crate-metadata` (also in CI) fails if any published crate is missing
  what crates.io shows, or if `README.md` is not in its packaged file list.

## [0.1.0] - 2026-09-06

The first public release. OxideLake was developed privately from 2026-08-25 as
`vyncint/oxidelake` (now `vyncint/oxidelake-old`); this repository starts from
one commit carrying the whole tree, not from that history. What follows is what
that tree contains and what changed on the way over.

### Changed

- **Crates renamed `oxide-*` → `oxidelake-*`.** `oxide-core` and `oxide-api`
  were already taken on crates.io by unrelated projects, so nothing could have
  published under the old names. Binary names are unchanged: `oxide`,
  `oxide-scheduler`, `oxide-worker`.
- **GPU kernel sources moved into `oxidelake-device/kernels/`.** They were at
  the repository root and reached by `include_str!("../../../../kernels/…")`,
  outside the crate directory — a published `oxidelake-device` would not have
  compiled, and no CI could see it because CI has the whole repository. The
  install workflow now exists to catch exactly this class.
- **The specification is `docs/SPEC.md`**, reframed from the implementation
  prompt it started as: the same normative content and section numbers that
  source comments cite, without the prompt framing or the local path.
- **Every crate is publishable.** `publish = false` is gone; the workspace
  carries the crates.io metadata; the release pipeline publishes all nine in
  dependency order through Trusted Publishing.

- Operators upload only the columns a kernel reads; projected columns of types
  the device cannot hold (`Utf8`, …) stay on the host and are gathered at the
  compacted row ids. A string column in a projection no longer forces the
  whole batch onto the CPU path.
- Backends declare supported operations up front (`supports_predicate`,
  `supports_hash_join`, `supports_aggregate`) so unsupported work is never
  uploaded.
- Hash join build sides are hashed once (`JoinBuild`) and uploaded once per
  operator; probes gather only projected columns.
- Column transfers are zero-copy where the transport allows (`Buffer`-based
  `ColumnBytes`; Metal downloads alias the shared `MTLBuffer`).
- Metal: cached compute pipelines, one command buffer per phase, pooled command
  queues, command-buffer errors surfaced.
- GPU-targeted sessions use 65 536-row batches.
- `oxide gen-data` streams chunks to the writer (bounded memory for any `--rows`).
- `rust-version` is the lockfile's real floor (1.94.1) and is verified in CI.
- Repository hygiene: `Makefile` gate, `deny.toml`, `rustfmt.toml`,
  `clippy.toml`, `.editorconfig`, contribution/security/conduct documents,
  Dependabot, issue and PR templates.

### Added

- **The vyncint contributor pattern**: DCO sign-off and no-AI-attribution
  enforced by `commit-policy` on every commit, generated `AGENTS.md` and
  `CONTRIBUTING.md`, `CODEOWNERS`, form-based issue templates.
- **A full release pipeline**: tag-guarded, changelog-gated, `cargo-semver-checks`,
  ordered multi-crate publish, GitHub Release, static binaries for four
  targets, and a fresh-machine install check.
- **CI shape**: one job per concern, every action pinned to a commit SHA,
  `zizmor` at pedantic, a `required-green` aggregate for branch protection, no
  build cache. The `predict` and `cuda` features are linted on every run; the
  PTY suite runs on macOS as well as Linux.
- **`predict` SQL UDF** — in-database inference over safetensors models via
  `oxmera`, behind the `predict` feature (ADR-0015).

### Fixed

- `predict`'s module documentation linked a private item, which `cargo doc
  --all-features` rejects. Never seen before because no job built the feature
  with docs — the gap the new CI closes.

- **(merged before the tag was cut, so it is in 0.1.0 — the entry was filed under
  Unreleased at the time.)** **CUDA grouped aggregation could emit a duplicate group.** A thread that
  lost the race for a group-table slot compared the slot's key through a
  plain load, which the SM's non-coherent L1 could serve from a stale line;
  the mismatch made it probe on and insert the same key a second time, and
  the extra slot surfaced as a duplicate group with `COUNT` 0 and `NULL`
  aggregates. The key is now read through a `volatile` load behind a fence
  that pairs with the publisher's. Found by the first run of the CUDA suites
  on real hardware (Tesla T4): `oxidelake-runtime/tests/embedded.rs` failed
  8 of 8 runs before the fix. A device-layer regression test with the same
  key distribution (`aggregate_matches_host_on_the_demo_distribution`,
  `#[ignore]`, needs a GPU) is red on the old kernel and green on the fix.

- STATUS.md over-claimed that the Metal conformance suite had executed on the
  device; the fixtures had taken the CPU fallback. The suite now asserts device
  execution through the operator transfer counters.

### Security

- **thrift < 0.23.0 (CVE-2026-43868) accepted with a written reason**, reached
  through `parquet 58`. The fix is `parquet 59`, which drops thrift entirely
  but needs DataFusion 55 — blocked on Ballista, still on ^54. Exposure is
  Parquet footer decode; `SECURITY.md` says what that means for what you feed
  the engine. Recorded in `deny.toml` before RustSec carries the advisory, so
  the gate stays green with the reason on file. Tracked in #4.

- CUDA aggregation kernel: the slot-claim spin-wait now reads through a
  `volatile` pointer; the previous plain load could be hoisted into an infinite
  loop (GPU hang). Compile-verified only — see `STATUS.md`.
- `GpuFilterExec` predicates are capped at 64 comparison leaves so a plan
  payload cannot drive the recursive evaluators arbitrarily deep.
- The dashboard quotes table identifiers before using them in generated SQL.
- `TelemetryHub` keeps at most 4096 operator registrations (oldest evicted), so
  long-lived sessions no longer grow without bound.
- `cargo deny` (advisories, licenses, bans, sources) runs in CI; `SECURITY.md`
  documents the trust model.

## [0.1.0] — 2026-09-01

First complete version (spec phases 0–7): the nine-crate workspace on
DataFusion 54 / Ballista 54 / arrow 58; CPU reference kernels with CUDA (NVRTC)
and Metal (MSL) backends behind one object-safe `GpuBackend`; the four
`Gpu*Exec` operators with per-batch CPU fallback; Parquet storage with proven
pruning and Arrow IPC spill; the `HardwarePlacementRule` and `OxidePhysicalCodec`
for Ballista cluster mode; the terminal dashboard with PTY tests; the
`oxide` / `oxide-scheduler` / `oxide-worker` binaries; the `OxideFrame`
DataFrame API and `l2_distance` / `cosine_distance` SQL UDFs with GPU lowering.
Metal executes on Apple silicon (verified on an M4 Pro and on GitHub's macOS
runners); CUDA compiles and lints without a CUDA installation but has not yet
run on a CUDA machine.

[Unreleased]: https://github.com/vyncint/oxidelake/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/vyncint/oxidelake/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/vyncint/oxidelake/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/vyncint/oxidelake/releases/tag/v0.1.0
