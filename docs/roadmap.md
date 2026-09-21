# Roadmap — phases and acceptance criteria

Each phase ends with the [gate](verification.md#the-gate) green and one commit `phase N: <summary>`. Phases run strictly in order; a phase never starts on a red gate. Tick boxes here as work lands and summarize in [STATUS.md](../STATUS.md). Detailed specifications are in [SPEC.md](SPEC.md) §2–§5.

## Phase 0 — bootstrap

Deliverables
- [x] `rust-toolchain.toml` — `channel = "1.98.0"`, `components = ["rustfmt", "clippy"]`
- [x] `origin` reachable, working tree clean (repo, LICENSE, `.gitignore`, README, STATUS, CI and docs already exist — do not recreate)
- [x] `STATUS.md` Phase 0 row updated

Gate: none yet (no Rust). Commit `phase 0: bootstrap`.

## Phase 1 — workspace scaffold

Deliverables
- [x] Root `Cargo.toml` exactly per PROMPT §4 (members glob, workspace deps rooted at Ballista 54, lints, profiles)
- [x] `cargo update`; coherence check `cargo tree --workspace -d -e normal` shows no duplicate `arrow-*` / `parquet` / `datafusion*` / `object_store` / `tonic` / `prost`
- [x] Nine crates with real minimal exports — crate docs, error types, feature wiring (`cuda`, `metal`, `io-uring` forwarded by `oxidelake-runtime` and `oxidelake-api`); no placeholder types that exist only to compile
- [x] `crates/oxidelake-device/kernels/{cuda,metal}/README.md`
- [x] Resolved dependency majors recorded in `STATUS.md`

Acceptance
- [x] Gate green (fmt, clippy default + cuda feature, test, `check -p oxidelake-runtime --features cuda`, doc)

## Phase 2 — core, memory, device

Deliverables
- [x] `oxidelake-core`: `EngineError` (Io, Allocation, Device, Execution, Plan, Format, Unsupported…), `DeviceId`, `BackendKind`, `BufferLayout`, `BatchStream`, `TelemetryHub`
- [x] `oxidelake-memory`: `AlignedBuf` (64/128 B), pinned host allocation via cudarc with observable `MemoryClass::{Pinned, Pageable}` fallback, Metal shared buffers (macOS), `SpillManager` (3 tiers, watermarks, LRU/clock, async demote/promote to Arrow IPC spill files, metrics)
- [x] `oxidelake-device`: object-safe sync `GpuBackend`; `CpuBackend` fully functional; `CudaBackend` with the NVRTC pipeline wired; `MetalBackend` with runtime MSL compile; `HardwareDetector` with `OXIDE_BACKEND` override

Acceptance
- [x] Alignment tests: host `% 64 == 0`, device `% 128 == 0`
- [x] Pinned-vs-pageable path observable and tested
- [x] `SpillManager` with kilobyte budgets demotes host → disk and promotes back byte-identical (`tempfile`); metrics move
- [x] Detector yields `CpuSimd` here; unavailable `OXIDE_BACKEND` → typed error
- [x] Gate green, including `clippy -p oxidelake-runtime --features cuda`

## Phase 3 — kernels + operators

Deliverables
- [x] `crates/oxidelake-device/kernels/cuda/{filter_project,hash_join,aggregation,vector_distance}.cu` — `extern "C"` entry points, explicit grid math, bounds checks on every global access
- [x] `crates/oxidelake-device/kernels/metal/{filter_project,vector_distance}.metal`
- [x] `GpuFilterExec`, `GpuHashJoinExec`, `GpuAggregateExec`, `GpuVectorDistanceExec` with execute-time backend selection and per-batch CPU fallback — transfers are **serial per batch** (`round_trip` is upload → kernel → download → `synchronize`); overlapping them with double-buffered streams is deferred to P6 in Phase 8
- [x] Conformance suite vs stock DataFusion operators (seeded random batches; nulls, empty, 0/1/odd rows); GPU variants `#[ignore]`

Acceptance
- [x] Conformance green on `CpuBackend`
- [x] Operators appear in `EXPLAIN`
- [x] `STATUS.md` states "kernel execution verified only where a device exists"
- [x] Gate green

## Phase 4 — storage (Parquet, Arrow IPC, object store)

Deliverables
- [x] Parquet writer properties for generated datasets (row-group size, dictionaries, Bloom filters on keys, page statistics, `--compression none|lz4|zstd`)
- [x] Session settings with statistics pruning, Bloom-filter-on-read and page index enabled
- [x] Arrow IPC spill/cache file writer + reader (64-byte alignment, no compression)
- [x] `object_store::ObjectStore` wiring: `LocalFileSystem` default; `UringLocalFileSystem` behind `io-uring` (dedicated ring thread, `oneshot` completions)
- [ ] `UringLocalFileSystem` registered into `RuntimeEnv` — implemented and conformance-tested, but no session constructs it, so the feature flag changes nothing at run time (#24). Blocked on a measurement: the 2026-09-02 audit found the ring slower than `LocalFileSystem` here.
- [x] Error mapping to `EngineError::Format` / `EngineError::Io`

Acceptance
- [x] Selective query over a generated Parquet dataset shows row groups pruned by statistics *and* by Bloom filter in scan metrics, with results equal to the pruning-disabled run
- [x] Arrow IPC spill round-trip byte-identical, buffers 64-byte aligned
- [x] `UringLocalFileSystem` passes the same `ObjectStore` conformance test as `LocalFileSystem` (or skips with a logged sandbox reason)
- [x] Truncated/corrupt Parquet → `EngineError::Format`, no panic
- [x] Gate green (incl. `cargo test -p oxidelake-storage --features io-uring`)

## Phase 5 — planner + Ballista runtime

Deliverables
- [x] `HardwarePlacementRule` (`PhysicalOptimizerRule`), target backend = local detector (embedded) or `OXIDE_CLUSTER_BACKEND` (scheduler); `EXPLAIN` shows `GpuFilterExec[cuda]`-style tags
- [x] `OxidePhysicalCodec` (`datafusion_proto::physical_plan::PhysicalExtensionCodec`) for the four `Gpu*Exec` nodes, `postcard` payloads
- [x] `OxideSession::local()` / `OxideSession::connect("df://host:port")` with identical query API
- [x] `oxide-scheduler` (wraps `ballista_scheduler`: `override_physical_codec`, `override_session_builder`) and `oxide-worker` (wraps `ballista_executor`: `override_physical_codec`, `override_function_registry`)

Acceptance
- [x] Placement unit tests with a mocked GPU-present detector (eligible rewritten, ineligible untouched)
- [x] Codec round-trip tests: encode → decode → identical `EXPLAIN` text and schema
- [x] In-process Ballista scheduler + 2 executors: distributed results equal embedded results on aggregation and join with `Gpu*Exec` in the plan (`OXIDE_CLUSTER_BACKEND=cuda`, executors fall back to CPU); cluster `EXPLAIN` shows tags; clean shutdown
- [x] `oxide-scheduler --help` and `oxide-worker --help` work
- [x] Gate green

## Phase 6 — TUI

Deliverables
- [x] `AppState` with pure `render` and `on_event`; repaints bracketed in DEC 2026 synchronized updates
- [x] Panels: Plan DAG (hardware tags), Inspector, Telemetry gauges, Describe (P25/P50/P99)
- [x] `oxidelake-tui-demo` binary rendering a fixed synthetic telemetry snapshot (no clock, no animation)
- [x] `tests/tui_render_test.rs` (TestBackend + insta at 80×24 and 120×40) and `tests/tui_pty_test.rs` (termlens)

Acceptance
- [x] Snapshots committed; PTY tests cover initial screen, ↓/↑, Tab, resize, `q` exits 0; no TTY needed
- [x] Gate green

## Phase 7 — API, end-to-end, docs

Deliverables
- [x] DataFrame API (`read_parquet`, `register_parquet`, `filter`, `select`, `join`, `aggregate`, `vector_distance`, `explain`, `collect`), `session.sql`, `l2_distance` / `cosine_distance` UDFs (also registered on executors)
- [x] CLI: `oxide gen-data`, `oxide sql [--cluster df://…]`, `oxide explain`, `oxide tui`
- [x] README quickstart containing only commands that run; final `STATUS.md` matrix
- [ ] Optional: PyO3 `python` feature — **deferred** with a `STATUS.md` note (crate-type and packaging rationale recorded there)

Acceptance
- [x] `oxide gen-data --rows 1000000 --out data/` writes Parquet
- [x] `oxide sql -q "SELECT k, SUM(v) FROM t GROUP BY k ORDER BY k"` is correct, asserted via `assert_cmd` (against an independent `parquet`-crate read of the file)
- [x] The same test spawns `oxide-scheduler` + one `oxide-worker` on ephemeral ports and `oxide sql --cluster df://127.0.0.1:<port> …` returns identical results
- [x] `oxide explain` shows placement tags (`--target cpu|cuda|metal` picks the planning target on any machine)
- [x] `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` clean
- [x] Gate green

## Phase 8 — production readiness (milestone `v0.2.0`)

Phases 0–7 were the build. This is the list that stands between the
published 0.1.x and something to run in production, and it is the one
place that list lives: the audit's deferred plans
(`docs/audit-2026-09-02.md` P1-rest, P4, P6, P12, P13, P14) are folded in
by issue number rather than kept in a second document that drifts.

Every line names an open issue and what would close it. Ticked lines link
the release that shipped them.

**0.2.0 did not finish this milestone.** It shipped the operability half —
the CPU-fallback counter and placement notes (#32), spans, a per-query log
line and a metrics endpoint (#33), the plan-codec fingerprint (#42), the CLI
knobs (#49), `predict`'s activation header (#47), the `non_exhaustive` /
semver policy (#41) — and settled the spill manager's scope honestly (#25).
Ten lines remain, and the three that matter most cannot be closed here: #28,
#45 and #46 need a CUDA device to verify on, and #29's streaming aggregate is
what #25's real wiring waits for. Do not read 0.2.0 as "production ready";
read it as "a running deployment can now tell whether its GPU is being used".

### Correctness

- [x] io_uring worker reaps its CQE before returning (#27) — *Acceptance:* a
      signal delivered while the ring waits cannot free the caller's buffer
      or misattribute a completion; the regression test fails without the
      fix. **0.1.4**
- [x] `GetOptions` preconditions honoured by both stores (#44) —
      *Acceptance:* `if_match`/`if_none_match` cases in the shared
      conformance body pass for `LocalFileSystem` and the io_uring store.
      **0.1.4**
- [x] `oxide sql` survives a closed pipe (#34) — *Acceptance:* `… | head -1`
      exits 0 with no panic, on an output larger than a pipe buffer.
      **0.1.4**
- [ ] CUDA filter matches Arrow totalOrder float comparison (#28) —
      *Acceptance:* NaN and `-0.0` rows in the conformance suite agree with
      stock DataFusion on a CUDA device.
- [ ] `GpuAggregateExec` streams and partitions instead of collecting the
      whole input (#29, audit P4) — *Acceptance:* a input larger than device
      memory completes.
- [ ] CUDA `SUM` detects Int64 overflow instead of wrapping (#46) —
      *Acceptance:* an overflowing sum is an error, not a wrong number.
- [x] `SpillManager` and `MemoryInfo` reach the operators, or the docs and
      the TUI say they are library-only (#25) — *closed the second way*:
      tier capacities come from the backend's `MemoryInfo`, and the panel,
      README, STATUS and architecture say the spill manager is a library
      that no query path calls. Wiring it in needs the streaming aggregate
      (#29) first. **0.2.0**

### Performance

- [x] Pinned staging for operator transfers (#26, audit P6) — *closed as
      documentation*: the operator path is pageable and says so; the
      measurement and the change are Phase 9. **0.1.4**
- [ ] Double-buffered stream pipelines (audit P6-rest) — *Acceptance:*
      transfer and compute overlap for a multi-batch scan, measured.
- [ ] CUDA join table persists across probe batches (#45) — *Acceptance:*
      the build side is uploaded once per join, not once per batch.
- [x] io_uring store wired into sessions, or the claim stays demoted (#24,
      audit P12) — *the claim stays demoted*: no session constructs it, and
      every document says so. **0.1.4**

### Operability

- [x] Per-batch CPU fallbacks and planner skip reasons are countable (#32)
      — `fallback_batches` on `OperatorStats`, a `cpu fallback` line in the
      TUI Inspector, one `warn!` per operator on first fallback, and
      `oxide explain` printing `placement notes` naming every node the rule
      left on the CPU and why. **0.2.0**
- [x] Tracing spans on the query path and an exportable metrics surface
      (#33) — an `oxide.operator` span per operator and partition, one
      `INFO` line per query from `OxideSession::collect`, and
      `--metrics-port` on the worker and scheduler behind the `metrics`
      feature. **0.2.0**
- [ ] Worker loss, Ballista retry settings and `SIGTERM` handling (#31,
      audit P13).
- [ ] TLS/auth for cluster mode, or an enforced private-network posture
      (#30) — *Acceptance:* the production cluster instructions do not
      describe an unauthenticated listener.
- [x] Plan codec version fingerprint, failing fast on mixed builds (#42).
      **0.2.0**
- [x] `--batch-size`, `--output table|json|csv`, worker `--backend` (#49) —
      every knob in one README table, each with an `assert_cmd` test.
      **0.2.0**
- [x] `predict` reads its activation from the model header (#47) — a
      header-less model is refused by name and a GELU model works. It also
      found that the loader applied *no* activation at all while three
      documents said ReLU. **0.2.0**

### Supply chain and release

- [ ] Ballista 55 / DataFusion 55 / arrow 59 chain bump (#40) — blocks #4
      and #54.
- [ ] thrift advisory retired once parquet ≥ 59 lands (#4, #54).
- [x] Public enums `non_exhaustive`, DataFusion-coupling semver policy
      written (#41) — landed in a minor, with the plan vocabulary left
      exhaustive on purpose. **0.2.0**
- [ ] On-demand CUDA execution job, made a release prerequisite (#38).
- [x] CI build-cache policy decided and documented (#39). **0.1.4**
- [x] termlens-cli installed prebuilt in CI (#48). **0.1.4**
- [x] README, STATUS and CHANGELOG agree on CUDA execution and the
      published version (#36). **0.1.4**
- [x] Page-index pruning proved beside statistics and Bloom filters (#43).
      **0.1.4**

## Out of scope for v1

GPUDirect Storage / cuFile and GPU-side Parquet decode · tuning Ballista's retry or AQE behaviour · TLS/auth for the Ballista cluster · catalog/metastore (Iceberg, Delta) · transactions · cost-based optimization beyond DataFusion defaults · Lance integration (v2) · cluster-wide telemetry in the TUI · Windows.
