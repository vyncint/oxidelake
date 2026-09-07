# OxideLake — Specification

**This is the specification the engine was built to.** It is normative: `docs/` elaborates it, and when the two disagree this document wins and the other is fixed. Section numbers are cited from source comments (`docs/SPEC.md §2.8`), so keep them stable.

## Mission

**OxideLake** is a GPU-accelerated, Arrow-native analytical query engine for lakehouse data (Parquet), written in Rust, with embedded and distributed execution. Host code is Rust; device kernels are CUDA C / Metal Shading Language source embedded in the binary and compiled at runtime.

OxideLake runs the same plans in two modes over the Apache Arrow columnar memory model:

- **Embedded mode** (DuckDB-like) — in-process DataFusion; operators exchange `Arc<RecordBatch>` over bounded `tokio::sync::mpsc` channels; no serialization, no network.
- **Cluster mode** (Spark-like) — **Apache DataFusion Ballista** provides the scheduler, the executors and the Arrow Flight shuffle. OxideLake contributes the GPU operators, the placement rule and the plan codec that lets Ballista ship those operators to executors. We do not write a scheduler (ADR-0013).

Acceleration backends: **CPU (always available; the correctness reference)**, **NVIDIA CUDA** (opt-in), **Apple Metal on unified memory** (opt-in, macOS only).

The differentiator is the operator layer — Rust host code with its own CUDA and Metal kernels behind one hardware abstraction — not a file format and not a scheduler. Storage is Parquet pruned by DataFusion plus Arrow IPC for spill and hot caches (ADR-0014); distribution is Ballista. Spend effort accordingly.

## 0. Reference environment

Verified facts about the development machine (2026-08-25): Linux 7.0 x86_64, 20 cores, 61 GiB RAM; Rust **1.98.0 stable**, cargo 1.98, git 2.53; **no NVIDIA GPU, no `nvcc`, no `nvidia-smi`, no `/dev/nvidia*`; not macOS**; crates.io reachable. Everything you claim as done must be provable on this machine.

1. **Default features = pure CPU.** `cargo check --workspace` and `cargo test --workspace` with default features must be green with zero GPU or OS-specific dependencies.
2. **`cuda` feature** (in `oxidelake-memory`, `oxidelake-device`, `oxidelake-compute`; forwarded by `oxidelake-runtime` and `oxidelake-api`) uses `cudarc` in **`dynamic-loading`** mode: the driver and NVRTC libraries are `dlopen`ed at runtime, so the feature **builds on this GPU-less box** and degrades to CPU at startup when no driver is found. Because `cuda-version-from-build-system` needs a toolkit, pin one CUDA API version feature explicitly (`cuda-12080`); cudarc's `cuda-*` features are alternatives, so never enable two. CUDA kernels are `.cu` **source files** in `oxidelake-device/kernels/cuda/`, embedded with `include_str!` and JIT-compiled with **NVRTC** on first use, PTX cached per device. There is **no `nvcc` build step**.
3. **`metal` feature**: all Metal code is additionally `#[cfg(target_os = "macos")]`, and its dependencies (`objc2`, `objc2-metal`, `objc2-foundation`) are declared only under `[target.'cfg(target_os = "macos")'.dependencies]`. Enabling `metal` on Linux must be a compiling no-op. MSL kernels in `oxidelake-device/kernels/metal/` are embedded via `include_str!` and compiled at runtime with `MTLDevice::newLibraryWithSource`. No Xcode build step.
4. **`io-uring` feature** (Linux-only, target-gated): an `object_store::ObjectStore` implementation for local files built on the low-level `io-uring` crate, driven from a dedicated storage thread. The default path is `object_store::local::LocalFileSystem` and both must behave identically (shared conformance test). If `io_uring_setup` is denied by a sandbox (EPERM/ENOSYS), tests skip with a logged reason instead of failing.
5. **Honesty rule.** GPU code cannot *execute* here. Verify what is verifiable — compilation and clippy under `--features cuda`, and a shared conformance suite that runs every operator against the CPU backend, and against CUDA/Metal only when a device is detected (`#[ignore]` by default, with a documented `-- --ignored` invocation for GPU machines). Record in `STATUS.md` exactly what ran and what only compiled. Never fabricate benchmark numbers or "works on GPU" claims. `README.md` may describe design intent; anything phrased as achieved must have run.
6. **Stub policy.** `todo!()`, `unimplemented!()`, and panicking placeholders are forbidden everywhere, features included. A genuinely unimplemented path returns `Err(EngineError::Unsupported { feature, detail })`.

## 1. Engineering standards (non-negotiable)

- Public APIs are 100% safe Rust. `unsafe` is confined to allocator/FFI internals; every block carries a `// SAFETY:` comment stating the invariant it relies on, and every raw resource is owned by an RAII type whose `Drop` releases it exactly once.
- Errors: `thiserror` enums in library crates, all convertible into `oxidelake_core::EngineError`; `anyhow` only in binaries. No `.unwrap()`/`.expect()` outside tests — enforced by the lints below.
- Every public item has a doc comment; every crate has `//!` docs stating its role and which internal crates it may depend on.
- Observability via `tracing` spans/events (operator execution, allocations, spills, plan submission). No `println!` outside the CLI's user-facing output.
- Tests: colocated unit tests plus `tests/` integration tests; snapshot tests via `insta` (commit the `.snap` files). Deterministic: seed every RNG; no wall-clock dependence in rendered output.
- Toolchain: **stable Rust**, pinned in `rust-toolchain.toml` (`channel = "1.98.0"`, `components = ["rustfmt", "clippy"]`); `workspace.package.rust-version = "1.88"` (the MSRV of Ballista 54, DataFusion 54 and ratatui 0.30 — the highest in the dependency graph). Do **not** use nightly `std::simd`: CPU vectorization comes from Arrow/DataFusion compute kernels plus `rayon` data parallelism and auto-vectorization. (If profiling later justifies explicit SIMD, the `wide` crate is the stable escape hatch — do not add it now.)
- Workspace lints (every crate sets `[lints] workspace = true`; test modules may locally `#[allow(clippy::unwrap_used, clippy::expect_used)]`):

```toml
[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "deny"
missing_docs = "warn"

[workspace.lints.clippy]
unwrap_used = "deny"
expect_used = "deny"
todo = "deny"
unimplemented = "deny"
dbg_macro = "deny"
```

## 2. Architecture

### 2.1 Memory model — `oxidelake-core`, `oxidelake-memory`

- All in-memory data is Arrow `RecordBatch` (arrow-rs). Alignment: host buffers ≥ **64 B** (cache line / AVX-512); buffers bound for devices **128 B** (coalesced global-memory transactions); Arrow IPC spill and cache files are written with 64-byte buffer alignment.
- `AlignedBuf`: RAII 64/128-byte-aligned host allocation (`std::alloc` with an explicit `Layout`).
- Pinned host memory: with `cuda` active and a driver present, allocate page-locked memory through cudarc (the `cuMemHostAlloc` path) for true DMA without staging copies; otherwise fall back transparently to `AlignedBuf`. The path taken is observable (`MemoryClass::{Pinned, Pageable}`) so tests and the TUI can report it.
- Metal (macOS + `metal`): `MTLResourceStorageModeShared` buffers so CPU and GPU share one physical allocation (UMA, zero-copy).
- **3-tier spill manager** (`SpillManager`): budgets for device VRAM, pinned/host RAM, and a local spill directory. Tracks registered batches (tier, bytes, last touch); at a high-watermark it asynchronously demotes cold batches device → host → disk and promotes on access (LRU/clock). Disk tier = Arrow IPC spill files written through the storage layer's object store (§2.4). Exposes atomic metrics (bytes per tier, demotions, promotions, spill throughput) for the TUI. On a GPU-less box the device tier is simply empty — the host ↔ disk machinery is fully testable with kilobyte budgets and `tempfile`.

### 2.2 Hardware abstraction — `oxidelake-device`

Runtime-selected, **object-safe** backend behind `Arc<dyn GpuBackend>`. Two deliberate design rules:

- **No associated types on the trait** — runtime selection requires trait objects. Opaque handle structs (`DeviceBuffer`, `HostBuffer`, `DeviceStream`) own backend-tagged internals and release themselves on `Drop`.
- **The trait is sync.** Asynchrony lives on device streams and at the operator layer (bridge completions into futures with `oneshot` channels or `spawn_blocking`). No `async_trait`.

```rust
pub enum BackendKind { CpuSimd, Cuda, Metal }

pub trait GpuBackend: Send + Sync + std::fmt::Debug {
    fn kind(&self) -> BackendKind;
    fn memory_info(&self) -> Result<MemoryInfo, EngineError>;
    fn alloc_device(&self, bytes: usize) -> Result<DeviceBuffer, EngineError>;
    fn alloc_pinned_host(&self, bytes: usize) -> Result<HostBuffer, EngineError>;
    fn create_stream(&self) -> Result<DeviceStream, EngineError>;
    fn copy_h2d(&self, s: &DeviceStream, src: &HostBuffer, dst: &DeviceBuffer) -> Result<(), EngineError>;
    fn copy_d2h(&self, s: &DeviceStream, src: &DeviceBuffer, dst: &mut HostBuffer) -> Result<(), EngineError>;
    fn synchronize(&self, s: &DeviceStream) -> Result<(), EngineError>;

    // Operator kernels over device-resident Arrow buffers. Supported types are
    // listed in §2.3; anything else returns EngineError::Unsupported and the
    // planner must never route it here.
    fn filter_project(&self, s: &DeviceStream, a: FilterProjectArgs<'_>) -> Result<DeviceBatch, EngineError>;
    fn hash_join(&self, s: &DeviceStream, a: HashJoinArgs<'_>) -> Result<DeviceBatch, EngineError>;
    fn aggregate(&self, s: &DeviceStream, a: AggregateArgs<'_>) -> Result<DeviceBatch, EngineError>;
    fn vector_distance(&self, s: &DeviceStream, a: VectorDistanceArgs<'_>) -> Result<DeviceBatch, EngineError>;
}
```

- `CpuBackend` — always present and **fully functional**: "device" buffers are host buffers; kernels delegate to Arrow compute kernels + `rayon`. It is the semantic reference every GPU backend must match.
- `CudaBackend` (`cuda`) — cudarc driver API; NVRTC-compiles kernel sources on first use and caches PTX per device.
- `MetalBackend` (`metal`, macOS) — objc2-metal; runtime-compiles MSL.
- `HardwareDetector::select()` — probe CUDA (device count > 0) → Metal → CPU; overridable with `OXIDE_BACKEND=cpu|cuda|metal` (an unavailable choice is a clean startup error, not a silent fallback).

### 2.3 Operators & kernels — `oxidelake-device/kernels/`, `oxidelake-compute`

DataFusion `ExecutionPlan` implementations: `GpuFilterExec` (fused filter + projection), `GpuHashJoinExec`, `GpuAggregateExec`, `GpuVectorDistanceExec`. Each one:

- pipelines batches through its stream with double-buffered H2D copy / kernel / D2H so PCIe transfer overlaps compute (CUDA); on Metal UMA there is no transfer stage;
- **chooses its backend at `execute()` time from the local `HardwareDetector`** — on a machine without the device it runs the CPU reference path. This is what keeps a heterogeneous Ballista cluster correct (§2.6);
- falls back per batch to the CPU path on `Unsupported`;
- reports rows in/out, batch latency and bytes moved into the telemetry hub (§2.7) and `tracing`.

**Deliberately bounded GPU type coverage for v1** (everything outside it stays on CPU by planner rule):

- filter + project: predicates `=, <, <=, >, >=` combined with `AND` over `Int64`/`Float64` columns against literals; projection = column selection; output is stream-compacted (block-scan prefix sums).
- hash join: inner join on a single `Int64` key; open-addressing linear-probing table in device memory built with atomic CAS, probed in parallel.
- aggregate: `SUM/COUNT/MIN/MAX` over `Int64`/`Float64` grouped by one `Int64` key of low cardinality (atomic/radix reduction).
- vector distance: L2 and cosine between a query vector and a `FixedSizeList<Float32>` column (shared memory + warp shuffles on CUDA; threadgroup memory + simdgroup reductions on Metal).

CUDA sources: `filter_project.cu`, `hash_join.cu`, `aggregation.cu`, `vector_distance.cu`. Metal sources: `filter_project.metal`, `vector_distance.metal` (joins and aggregates stay on CPU under Metal in v1 — say so in the docs). Kernels are plain `extern "C"` entry points with explicit grid-size math and bounds checks on every global access.

**Conformance suite** (in `oxidelake-compute`): for seeded random batches — including nulls, empty batches, and 0/1/odd row counts — every `Gpu*Exec` running on `CpuBackend` must equal the corresponding stock DataFusion operator. The same suite runs against CUDA/Metal when a device is present (`#[ignore]` by default).

### 2.4 Storage — `oxidelake-storage`: Parquet, Arrow IPC, io_uring object store

There is no private file format (ADR-0014). `oxidelake-storage` owns configuration, the zero-decode paths, the IO layer, and the proofs.

- **Lake data is Parquet**, read through DataFusion's listing tables (`read_parquet`, `CREATE EXTERNAL TABLE … STORED AS PARQUET`). Pruning is DataFusion's own: row-group statistics, page index and split-block Bloom filters — exactly the zonemap/Bloom mechanics a custom format would have re-implemented. `oxidelake-storage` provides: writer properties for our datasets (row-group size matched to the engine's batch size, dictionary encoding, Bloom filters on join/filter keys, page-level statistics, `--compression none|lz4|zstd`); session settings with statistics pruning, Bloom-filter-on-read and the page index enabled; and tests that assert pruning happened via the scan's metrics (row groups pruned by statistics and by Bloom filter) *and* that results equal the same query run with pruning disabled.
- **Arrow IPC for zero-decode data**: spill files and an optional hot cache are Arrow IPC files (`arrow_ipc::writer::FileWriter`, 64-byte buffer alignment, no compression) — raw Arrow buffers that come back as `RecordBatch`es without decode.
- **IO abstraction is `object_store::ObjectStore`**, the trait DataFusion reads through. Default `object_store::local::LocalFileSystem`. `io-uring` feature (Linux target-gated): `UringLocalFileSystem` implementing `ObjectStore` for `file://` — `get`, ranged `get_opts`/`get_ranges`, `head`, `put`, `list`, `delete` — on a dedicated ring thread with completions bridged to `oneshot` futures, registered into the session's `RuntimeEnv` so Parquet scans and spill IO both use it. A shared conformance test runs against both implementations.
- Parquet/DataFusion errors at our API boundary map to `EngineError::Format` (corrupt or truncated files) or `EngineError::Io`; never a panic.
- Out of scope, documented: GPUDirect Storage / cuFile and GPU-side Parquet decoding. Lance is a documented v2 option for vector indices (currently coherent: lance 10 → DataFusion 54 / arrow 58) and is not adopted in v1.

### 2.5 Planner — `oxidelake-planner`: placement and plan codec

- `HardwarePlacementRule` is a DataFusion **`PhysicalOptimizerRule`** (not an AST pass). It rewrites eligible `FilterExec`/`ProjectionExec`/`HashJoinExec`/`AggregateExec` nodes into `Gpu*Exec` when (a) the target backend is not CPU and (b) the node fits the §2.3 coverage; regex, JSON, UDF and string-heavy nodes stay on CPU. In embedded mode the target backend is the local `HardwareDetector` result; on the Ballista scheduler (where the rule is installed through `SchedulerConfig::override_session_builder`) it is the declared cluster capability `OXIDE_CLUSTER_BACKEND=cpu|cuda|metal` (default `cpu` = no rewrites), and executors pick the real local backend at execute time (§2.3). `EXPLAIN` output must show placement, e.g. `GpuFilterExec[cuda]`. Unit tests assert rewrite decisions on representative plans using a mocked "GPU present" detector.
- `OxidePhysicalCodec` implements `datafusion_proto::physical_plan::PhysicalExtensionCodec` for the four `Gpu*Exec` nodes (payload: `postcard`-serialized operator parameters; children are encoded by DataFusion). Round-trip tests: encode → decode → identical `EXPLAIN` text and schema.

### 2.6 Distributed — `oxidelake-runtime` on Apache DataFusion Ballista

- `OxideSession::local()` → a DataFusion `SessionContext` with the placement rule, the UDFs and the object store. `OxideSession::connect("df://host:port")` → Ballista's `SessionContext::remote` with `OxidePhysicalCodec` installed through Ballista's session-config extension. Both expose the same query API.
- Binaries: `oxide` (CLI: `gen-data`, `sql [--cluster df://…]`, `explain`, `tui`); `oxide-scheduler` — a thin wrapper over `ballista_scheduler` configured with `SchedulerConfig { override_physical_codec, override_session_builder, .. }`; `oxide-worker` — a thin wrapper over `ballista_executor` configured with `ExecutorProcessConfig { override_physical_codec, override_function_registry, .. }`. **Ballista 54 is newer than your training data: read the `ballista`, `ballista-core`, `ballista-scheduler` and `ballista-executor` docs on docs.rs for the exact entry points and codec configuration; do not guess.**
- Ballista owns stage splitting, task scheduling, the Arrow Flight shuffle, retries (`task_max_failures`, `stage_max_failures`) and `submit_physical_plan` for pre-built plans. We configure; we do not wrap or tune those in v1.
- Equivalence gate: an in-process test starts a Ballista scheduler and **two** executors on ephemeral ports through the library entry points (`SessionContext::standalone()` is fine for a one-executor smoke test but not for the gate), runs a partitioned aggregation and a join with `OXIDE_CLUSTER_BACKEND=cuda` so that `Gpu*Exec` nodes travel through the codec (the executors have no GPU and take the CPU path), and asserts result equality with embedded mode, that cluster-mode `EXPLAIN` shows placement tags, and clean shutdown.

### 2.7 Telemetry & TUI — `oxidelake-tui`

- `TelemetryHub` (in `oxidelake-core`): atomic counters/gauges written by the engine — per-operator rows and latency, bytes per memory tier, transfer and spill rates, plan summary — with cheap consistent snapshot reads. It is the *only* coupling between engine and TUI, which makes the TUI testable with synthetic snapshots. In v1 the TUI observes the local process only; cluster-wide telemetry is out of scope.
- The app is a state machine decoupled from rendering: `fn render(state: &AppState, frame: &mut Frame)` is pure; `fn on_event(state: &mut AppState, ev: Event) -> Transition` handles input. Bracket every repaint in DEC 2026 synchronized updates (crossterm `BeginSynchronizedUpdate`/`EndSynchronizedUpdate`) so PTY tests observe complete frames only.
- Panels: (1) **Plan DAG** — tree of physical operators with a hardware tag per node `[CUDA]`/`[Metal]`/`[CPU]`; (2) **Inspector** — selected operator's rows in/out, batch latency (ms), memory allocated; (3) **Telemetry** — gauges for VRAM / pinned RAM / disk spill and PCIe/NVMe transfer rates; (4) **Describe** — per-column min, max, null count, quantiles P25/P50/P99. Keys: `↑`/`↓` select DAG node, `Tab` cycles panels, `q` quits.
- A deterministic demo binary `oxidelake-tui-demo` (`[[bin]]` inside `oxidelake-tui`) renders the dashboard from a fixed synthetic telemetry snapshot — no clock, no animation — so PTY tests can spawn it via `env!("CARGO_BIN_EXE_oxide-tui-demo")`.
- **Headless tests, two layers:**
  1. `crates/oxidelake-tui/tests/tui_render_test.rs` — in-process: `ratatui::backend::TestBackend` + `insta::assert_snapshot!` of the rendered buffer at 80×24 and 120×40; state transitions (`↓`/`↑` selection, `Tab`, `q` → quit) asserted on `AppState`.
  2. `crates/oxidelake-tui/tests/tui_pty_test.rs` — end-to-end with **`termlens`** (dev-dependency; its default feature enables `insta` integration): `Terminal::builder().size(80, 24).env_clear().timeout(..).spawn(env!("CARGO_BIN_EXE_oxide-tui-demo"))`, then `wait_until(|s| s.contains("OxideLake"))` or `wait_frame(..)`, `termlens::assert_screen_snapshot!(t.screen())`, `send(Key::Down)` and re-snapshot, `resize(120, 40)` and re-snapshot, `send(Key::Char('q'))` then `wait_exit()?.success()`. Read https://docs.rs/termlens/0.6.1 before writing these — the API is newer than your training data. Never `sleep`; always use the `wait_*` methods. termlens is a real-PTY harness, so no physical TTY is needed.

### 2.8 API — `oxidelake-api`

- Fluent DataFrame API over DataFusion — `read_parquet`, `register_parquet`, `filter`, `select`, `join`, `aggregate`, `vector_distance`, `explain`, `collect` — plus `session.sql("…")`; `l2_distance` / `cosine_distance` registered as scalar UDFs for SQL (and on executors through `override_function_registry`).
- PyO3 bindings behind a `python` feature, **off by default and excluded from every gate** (needs Python headers). Implement last; if time runs short, defer with a `STATUS.md` note.

## 3. Workspace layout

```
oxidelake/
├── Cargo.toml                 # virtual workspace: members, shared deps, lints, profiles
├── rust-toolchain.toml        # channel = "1.98.0", components = ["rustfmt", "clippy"]
├── .gitignore                 # target/, *.snap.new, data/, .DS_Store
├── LICENSE                    # Apache-2.0
├── README.md                  # what it is, quickstart (only commands that actually run)
├── STATUS.md                  # phase table: done / verified how / deferred
├── .github/workflows/         # ci.yml (the §5 gate, one job per concern), release.yml, binaries.yml, install.yml, stress.yml, commit-policy.yml
├── docs/                      # this spec, roadmap checklists, architecture, dependency evidence, verification, releasing, ADRs
└── crates/
    ├── oxidelake-core/            # EngineError, DeviceId, BackendKind, BufferLayout, BatchStream, TelemetryHub
    ├── oxidelake-memory/          # AlignedBuf, pinned/UMA allocators, SpillManager
    ├── oxidelake-device/          # GpuBackend trait, Cpu/Cuda/Metal backends, HardwareDetector
    │   └── kernels/               # INSIDE the crate: `cargo publish` packages only the crate directory (ADR-0016)
    │       ├── cuda/              # NVRTC-compiled at runtime; no nvcc anywhere
    │       │   ├── filter_project.cu
    │       │   ├── hash_join.cu
    │       │   ├── aggregation.cu
    │       │   └── vector_distance.cu
    │       └── metal/             # newLibraryWithSource at runtime; no Xcode step
    │           ├── filter_project.metal
    │           └── vector_distance.metal
    ├── oxidelake-compute/         # Gpu*Exec ExecutionPlans, CPU reference paths, conformance suite
    ├── oxidelake-storage/         # Parquet writer/pruning config, Arrow IPC spill files, io_uring ObjectStore
    ├── oxidelake-planner/         # HardwarePlacementRule, OxidePhysicalCodec
    ├── oxidelake-runtime/         # OxideSession, Ballista scheduler/executor wrappers, `oxide` CLI (+ binaries)
    ├── oxidelake-tui/             # dashboard lib, oxidelake-tui-demo bin, TestBackend + termlens tests
    └── oxidelake-api/             # DataFrame/SQL API, optional PyO3 bindings
```

Internal dependency direction (must stay acyclic):

| crate | may depend on |
|---|---|
| oxidelake-core | — |
| oxidelake-memory | core |
| oxidelake-device | core, memory |
| oxidelake-compute | core, memory, device |
| oxidelake-storage | core, memory |
| oxidelake-planner | core, compute (+ `datafusion-proto`) |
| oxidelake-tui | core |
| oxidelake-runtime | all of the above (+ the `ballista*` crates; its `oxide` binary pulls oxidelake-tui) |
| oxidelake-api | runtime |

## 4. Dependencies — verified coherent set (crates.io, 2026-08-25)

Rules:

1. Every external dependency is declared once in `[workspace.dependencies]`; members use `{ workspace = true }`.
2. **Version coherence has one root: Ballista.** Its docs say "Make sure the version of `datafusion` is the same as `ballista`'s!" — so `ballista*` fixes the `datafusion` / `datafusion-proto` major, DataFusion fixes the `arrow` / `parquet` / `object_store` majors, and `arrow-flight` / `tonic` / `prost` arrive transitively through Ballista (not declared directly). Never depend on other `datafusion-*` sub-crates; `datafusion-proto` is the single exception (needed for the plan codec) and is pinned to the identical version (ADR-0009). Gate: `cargo tree --workspace -d -e normal` shows **no duplicate `arrow-*`, `parquet`, `datafusion*`, `object_store`, `tonic` or `prost` majors**.
3. Evidence for the set below (live registry): ballista / ballista-core / ballista-scheduler / ballista-executor 54.1.0 (2026-08-09) require `datafusion ^54`, `datafusion-proto ^54`, `arrow-flight ^58.3`, `object_store ^0.13.2`, `tonic ^0.14`, `prost ^0.14`; datafusion 54.1.0 requires `arrow ^58.3`, `parquet ^58.3`, `object_store ^0.13.2`, `tokio ^1.52`; arrow-flight 58.4 requires `tonic` / `prost ^0.14.1`; ratatui 0.30.2 pairs with crossterm 0.29 through `crossterm_0_29`; cudarc 0.19.9 ships `dynamic-loading`, `nvrtc` and `cuda-*`; objc2-metal 0.3 requires objc2 0.6 and objc2-foundation 0.3; termlens 0.6.1 defaults to `insta`. MSRVs: Ballista, DataFusion 54 and ratatui 1.88 → `rust-version = "1.88"`. DataFusion 55 exists but Ballista is on 54 — Ballista wins. When Ballista moves to a new DataFusion major, re-derive the whole chain; never bump one crate alone.

```toml
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.88"
license = "Apache-2.0"
authors = ["OxideLake Contributors"]

[workspace.dependencies]
# data plane — Ballista 54 is the root of this chain
ballista = "54"                      # client: SessionContext::remote / standalone
ballista-core = "54"
ballista-scheduler = "54"
ballista-executor = "54"
datafusion = "54"
datafusion-proto = "54"              # PhysicalExtensionCodec for Gpu*Exec; always identical to datafusion
arrow = { version = "58", features = ["prettyprint"] }
parquet = "58"                       # lake format: statistics, page index, Bloom filters
object_store = "0.13"                # ObjectStore trait; io_uring local store implements it
# async
tokio = { version = "1.52", features = ["full"] }
futures = "0.3"
bytes = "1"
async-trait = "0.1"                  # only to implement object_store's ObjectStore (foreign trait)
# parallelism
rayon = "1"
# errors + observability
thiserror = "2"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
# serialization (codec payloads, config)
serde = { version = "1", features = ["derive"] }
postcard = { version = "1", features = ["use-std"] }
# storage fast path
io-uring = "0.7"                     # Linux target-gated, behind the io-uring feature
# GPU (all optional)
cudarc = { version = "0.19", default-features = false, features = ["std", "driver", "nvrtc", "dynamic-loading", "cuda-12080"] }
objc2 = "0.6"                        # macOS target-gated, behind the metal feature
objc2-metal = "0.3"
objc2-foundation = "0.3"
# UI + CLI
ratatui = { version = "0.30", features = ["crossterm_0_29"] }
crossterm = "0.29"
clap = { version = "4", features = ["derive"] }
# test-only
insta = "1"
termlens = "0.6"
assert_cmd = "2"
tempfile = "3"
rand = "0.10"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = "symbols"
# Deliberately no `panic = "abort"`: unwinding must stay sound across PyO3/FFI
# boundaries and workers must keep backtraces.

[profile.dev.package."*"]
opt-level = 2      # arrow/datafusion are unusably slow at -O0 in tests
```

Feature wiring:

- `oxidelake-memory`, `oxidelake-device`, `oxidelake-compute`: `cuda = ["dep:cudarc"]`; `metal = [...]` activating the macOS target-gated optional deps.
- `oxidelake-storage`: `io-uring = ["dep:io-uring", "dep:async-trait"]` (Linux target-gated).
- `oxidelake-runtime` and `oxidelake-api` forward: `cuda = ["oxidelake-device/cuda", "oxidelake-memory/cuda", "oxidelake-compute/cuda"]`, `metal = [...]`, `io-uring = ["oxidelake-storage/io-uring"]`. This is what makes `cargo check -p oxidelake-runtime --features cuda` the single GPU compile gate (a virtual workspace root rejects `--features`).
- `oxidelake-api`: `python = ["dep:pyo3", "arrow/ffi"]` — add `pyo3` (latest, `abi3-py310`) only when Phase 7 reaches it.

## 5. Phase plan with acceptance gates

Run the **gate** after every phase; every command must pass before that phase's commit. `.github/workflows/ci.yml` uses the Make targets documented in [verification.md](verification.md), with independent feature jobs, dependency reuse, and conservative documentation-only skips per [ADR-0017](decisions/ADR-0017-ci-dependency-reuse.md). Release calls always run the full gate without build caches.

```bash
make gate
```

On a GPU or macOS machine (not available here; document, don't claim): `cargo test -p oxidelake-compute --features cuda -- --ignored` and `cargo test -p oxidelake-compute --features metal -- --ignored`.

**Phase 0 — bootstrap.** The repository already exists with `origin` on GitHub and `LICENSE`, `.gitignore`, `README.md`, `STATUS.md`, `docs/` and `.github/workflows/ci.yml` committed — do not recreate or overwrite them. Add `rust-toolchain.toml` (`channel = "1.98.0"`, `components = ["rustfmt", "clippy"]`), confirm `git status` is clean and `origin` is reachable, and update the Phase 0 row in `STATUS.md`. Commit `phase 0: bootstrap`.

**Phase 1 — workspace scaffold.** Root manifest exactly per §4; `cargo update` and the coherence check; all nine crates with real minimal exports (crate docs, error types, feature wiring — no placeholder types that exist only to compile); `oxidelake-device/kernels/cuda/` and `oxidelake-device/kernels/metal/` with a README each. Record the resolved set (`cargo tree` majors for arrow, parquet, datafusion, ballista, tonic, prost, object_store) in `STATUS.md`. Gate → commit.

**Phase 2 — core, memory, device** (§2.1–2.2).
Acceptance: alignment tests (`ptr % 64 == 0`; device buffers `% 128 == 0`); pinned-vs-pageable path observable and tested; `SpillManager` with kilobyte budgets demotes cold batches host → disk (Arrow IPC through the default object store) and promotes them back byte-identical (`tempfile`); metrics counters move; `HardwareDetector` yields `CpuSimd` here and honors `OXIDE_BACKEND` (unavailable choice → typed error); `CudaBackend`/`MetalBackend` compile under their features with the NVRTC/MSL compile pipelines wired end to end. Gate → commit.

**Phase 3 — kernels + operators** (§2.3). Four CUDA sources, two Metal sources, four `ExecutionPlan`s with CPU reference paths and execute-time backend selection, conformance suite (CPU-executed; GPU variants `#[ignore]`).
Acceptance: conformance green incl. nulls/empty/odd sizes; operators display in `EXPLAIN`; every kernel bounds-checks global accesses; `STATUS.md` states "kernel execution verified only where a device exists". Gate → commit.

**Phase 4 — storage** (§2.4).
Acceptance: a generated Parquet dataset (Bloom filters on the key column, page statistics) queried with a selective predicate shows row groups pruned by statistics *and* by Bloom filter in the scan metrics, and its results equal the run with pruning disabled; Arrow IPC spill file round-trip is byte-identical with 64-byte-aligned buffers; `UringLocalFileSystem` passes the same `ObjectStore` conformance test as `LocalFileSystem` (or skips with a logged sandbox reason); truncated/corrupt Parquet → `EngineError::Format`, no panic. Gate → commit.

**Phase 5 — planner + Ballista runtime** (§2.5–2.6).
Acceptance: placement-rule unit tests (mocked GPU-present detector: eligible nodes rewritten, ineligible untouched, `EXPLAIN` shows tags); `OxidePhysicalCodec` round-trip tests for all four nodes; in-process Ballista scheduler + two executors: distributed results equal embedded results on aggregation and join with `Gpu*Exec` nodes in the plan, cluster `EXPLAIN` shows placement tags, clean shutdown; `oxide-scheduler` and `oxide-worker` binaries build and print `--help`. Gate → commit.

**Phase 6 — TUI** (§2.7).
Acceptance: `TestBackend` snapshots at 80×24 and 120×40; termlens PTY tests pass against `oxidelake-tui-demo` (initial screen, `↓`/`↑`, `Tab`, resize, `q` exits 0); no TTY needed; `.snap` files committed. Gate → commit.

**Phase 7 — API, end-to-end, docs** (§2.8).
Acceptance: `oxide gen-data --rows 1000000 --out data/` writes Parquet; `oxide sql -q "SELECT k, SUM(v) FROM t GROUP BY k ORDER BY k"` prints correct results, asserted by an `assert_cmd` test in `oxidelake-runtime`; the same test spawns `oxide-scheduler` and one `oxide-worker` on ephemeral ports and asserts `oxide sql --cluster df://127.0.0.1:<port> …` returns the same results; `oxide explain` shows placement tags; README quickstart commands all run as written; final `STATUS.md` matrix (component / done / verified how / deferred). Gate → commit.

## 6. Process directives

- Execute phases strictly in order; never start a phase on a red gate. Keep the tree compiling between commits; prefer small vertical slices over broad stubs. Commit at each green gate as `phase N: <summary>`.
- Do not stop to ask questions mid-run: make the decision this document implies, record it in `STATUS.md`, and continue.
- On an API mismatch (a resolved crate differs from what you expect — Ballista and termlens especially), read that version's docs (`cargo doc -p <crate>` or docs.rs) and adapt the code; record the deviation in `STATUS.md`. Never downgrade a dependency to dodge a compile error unless §4's coherence rule demands it.
- At every gate also tick the matching boxes in `docs/roadmap.md`; `STATUS.md` stays the summary, the roadmap the checklist.
- If you must stop early (context or time), stop at a green gate and leave precise next steps in `STATUS.md`.
- Out of scope for v1 — document, don't build: GPUDirect Storage / cuFile and GPU-side Parquet decode; tuning Ballista's retry or AQE behaviour; TLS/auth for the Ballista cluster; catalog/metastore (Iceberg, Delta); transactions; cost-based optimization beyond DataFusion defaults; Lance integration (v2); cluster-wide telemetry in the TUI; Windows.

## Appendix — corrections and scope decisions baked in (do not reintroduce)

1. Stale, incoherent pins (`arrow 53`/`datafusion 43`/`tonic 0.12`/`prost 0.13`/`ratatui 0.28`) → a registry-verified chain (§4).
2. Nightly-only `std::simd` → stable toolchain; Arrow kernels + rayon.
3. "`cuda-oxide` / `cudarc`" ambiguity → `cudarc` 0.19 only, `dynamic-loading` + `nvrtc` + explicit `cuda-12080`, so the GPU feature builds and lints on a machine with no CUDA.
4. "`metal-rs` / `objc2-metal`" ambiguity → `objc2-metal` 0.3 + `objc2` 0.6, macOS target-gated.
5. `GpuBackend` with associated types and `#[async_trait]` on synchronous methods → object-safe sync trait with opaque RAII handles.
6. No feature or platform gating → default build is pure CPU; `cuda`/`metal`/`io-uring` are opt-in and forwarded by `oxidelake-runtime`.
7. `tokio-uring` (0.5, stale, pins `io-uring 0.6`) → the maintained `io-uring` 0.7 crate on a dedicated storage thread.
8. `panic = "abort"` in release → removed (PyO3/FFI unwind soundness, worker backtraces).
9. Separate pins of `datafusion-*` sub-crates → `datafusion::` re-exports only, with `datafusion-proto` as the single pinned exception for the codec.
10. "Analyze the query AST" for placement → a DataFusion `PhysicalOptimizerRule`.
11. termlens was named but unexplained → a dedicated deterministic `oxidelake-tui-demo` binary and a concrete PTY test plan; `TestBackend` covers pure rendering.
12. No acceptance criteria and an aspirational verification section → per-phase gates, a single GPU compile gate, and the `STATUS.md` honesty rule.
13. Own coordinator, control plane and Arrow Flight shuffle (`oxide-flight`) → Apache DataFusion Ballista 54 with our `PhysicalExtensionCodec` and placement rule installed through its override hooks; Ballista's DataFusion major (54 → arrow 58) is the root of the dependency chain (ADR-0013).
14. Custom `.oxide` file format with hand-rolled zonemaps and Bloom filters → Parquet pruned by DataFusion's statistics, page index and Bloom filters, Arrow IPC for spill and hot cache, and an io_uring `ObjectStore` implementation instead of a private `StorageIo` trait (ADR-0014).
