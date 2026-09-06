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
- [x] `GpuFilterExec`, `GpuHashJoinExec`, `GpuAggregateExec`, `GpuVectorDistanceExec` with double-buffered stream pipelines, execute-time backend selection and per-batch CPU fallback
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
- [x] `object_store::ObjectStore` wiring: `LocalFileSystem` default; `UringLocalFileSystem` behind `io-uring` (dedicated ring thread, `oneshot` completions), registered into `RuntimeEnv`
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

## Out of scope for v1

GPUDirect Storage / cuFile and GPU-side Parquet decode · tuning Ballista's retry or AQE behaviour · TLS/auth for the Ballista cluster · catalog/metastore (Iceberg, Delta) · transactions · cost-based optimization beyond DataFusion defaults · Lance integration (v2) · cluster-wide telemetry in the TUI · Windows.
