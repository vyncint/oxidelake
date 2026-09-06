# Verification

## The gate

Run after every phase; every command must pass before that phase's commit. `make gate` runs the list below (plus `cargo deny`, and the io_uring / Metal lanes on the platforms that have them); [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs the same list on every pull request as one job per concern — `fmt`, `clippy` (default / `cuda` / `predict`), `test` (Linux: default, io_uring, predict, coherence), `metal` (macOS: the Metal lane and the PTY suite), `msrv`, `docs` (all features, warnings denied), `deny`, `release-scripts`, `zizmor` — aggregated by `required-green`, the one check branch protection requires. `commit-policy` runs beside it. Keep the Makefile, this page and the workflow in sync.

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p oxidelake-runtime --all-targets --features cuda -- -D warnings   # from Phase 2
cargo test --workspace
cargo test -p oxidelake-storage --features io-uring                              # from Phase 4 (Linux)
cargo check -p oxidelake-runtime --features cuda                                 # GPU code builds with no CUDA installed
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo tree --workspace -d -e normal        # must list no duplicate arrow-* / parquet / datafusion* / object_store / tonic / prost majors
```

The Metal lane (macOS, since 2026-09-01 — CI runs it as the `metal` job on a
macOS arm64 runner):

```bash
cargo clippy -p oxidelake-runtime --all-targets --features metal -- -D warnings
cargo test -p oxidelake-memory -p oxidelake-device -p oxidelake-compute -p oxidelake-runtime --features metal
cargo test -p oxidelake-compute --features metal -- --ignored   # conformance ON the Metal device; needs one
```

Supply chain and toolchain claims are part of the gate too:

```bash
cargo deny check                      # RustSec advisories, license allow-list, banned crates, registry sources (deny.toml)
make msrv                             # the declared rust-version really builds the workspace (CI job `msrv`)
```

Every `deny.toml` ignore carries the reason the affected code path is unreachable in OxideLake and is revisited when the dependency chain moves ([dependencies.md](dependencies.md)).

Why the GPU gate is `-p oxidelake-runtime --features cuda`: the workspace root is virtual and rejects `--features`; `oxidelake-runtime` forwards `cuda` / `metal` / `io-uring` into the crates that implement them, so one command covers the whole GPU code path. It must pass on a machine with **no CUDA installed** — cudarc runs in `dynamic-loading` mode and kernels are JIT-compiled at runtime ([ADR-0003](decisions/ADR-0003-cudarc-dynamic-loading-nvrtc.md), [ADR-0006](decisions/ADR-0006-pure-cpu-default-features-forwarded.md)).

## What can and cannot be verified on the dev boxes

Two dev boxes so far: Linux 7.0 x86_64 (2026-08-25, phases 0–6; no GPU, no
CUDA toolkit, not macOS) and macOS on an Apple M4 Pro (since 2026-09-01,
Metal 4, 24 GiB unified memory), where the whole default gate re-ran green
and the Metal backend executes for real.

| Claim | Verifiable | How / where |
|---|---|---|
| Pure-CPU build, tests, clippy, docs | yes | the gate (both boxes) |
| CUDA code compiles and lints | yes | `check` / `clippy -p oxidelake-runtime --features cuda` (no CUDA installed anywhere) |
| `metal` feature is a no-op on Linux | yes | `cargo check -p oxidelake-runtime --features metal` on Linux |
| Metal compile, lint, **execution** | yes (macOS) | the Metal lane above; the conformance suite runs on the actual device |
| Cluster mode (Ballista scheduler + executors, codec, placement) | yes | in-process scheduler + 2 executors; spawned binaries in the Phase 7 E2E — executors take the CPU path |
| Embedded GPU-target ≡ CPU results | yes | `oxidelake-runtime/tests/embedded.rs` over cpu/cuda/metal targets |
| Parquet pruning | yes | scan metrics + result equality against a pruning-disabled run |
| CPU ↔ CUDA operator equivalence | **no** | conformance suite is `#[ignore]` for CUDA; run on a CUDA box |
| CUDA kernel *execution* and performance | **no** | never claim; record as "compiled only" |
| io_uring object store | Linux only | skips with a logged reason when `io_uring_setup` returns EPERM/ENOSYS |

On a CUDA machine (still unverified — no such box has run this yet):

```bash
cargo test -p oxidelake-compute --features cuda -- --ignored
```

## Honesty rule

- A component is "done" only when its acceptance criteria in [roadmap.md](roadmap.md) ran green **here**, or it is explicitly listed as compiled-only / needs-GPU in the `STATUS.md` verification matrix.
- No fabricated benchmarks. Performance numbers appear only with the machine, dataset, command and date that produced them.
- `README.md` may describe design intent; anything phrased as achieved must have run.
- Stubs are typed errors (`EngineError::Unsupported`), never `todo!()` / `unimplemented!()` — enforced by workspace lints.

## Test layers

| Layer | Where | Tooling |
|---|---|---|
| Unit | each crate, colocated `#[cfg(test)]` | std test; `insta` where output is structural |
| Conformance | `oxidelake-compute` | `Gpu*Exec` on `CpuBackend` vs stock DataFusion operators over seeded random batches |
| Storage | `oxidelake-storage/tests` | Parquet pruning metrics + equality; Arrow IPC spill round-trip; shared `ObjectStore` conformance for `LocalFileSystem` and `UringLocalFileSystem` (feature matrix) |
| Placement + codec | `oxidelake-planner` | mocked GPU-present detector; `EXPLAIN` text; codec encode → decode round-trip |
| Distributed equivalence | `oxidelake-runtime/tests` | in-process Ballista scheduler + 2 executors on ephemeral ports; `Gpu*Exec` nodes through the codec |
| Embedded GPU-target equivalence | `oxidelake-runtime/tests/embedded.rs` | the same queries under cpu/cuda/metal placement targets return identical rows (multi-partition plans included) |
| SQL UDFs | `oxidelake-compute/src/udf.rs` | `l2_distance` / `cosine_distance` against the reference kernels, null and coercion cases |
| DataFrame API | `oxidelake-api/tests/dataframe.rs` | every verb against its SQL equivalent; GPU-target plans show `Gpu*Exec` for fluent pipelines |
| TUI in-process | `oxidelake-tui/tests/tui_render_test.rs` | `ratatui::backend::TestBackend` + `insta` |
| TUI end-to-end | `oxidelake-tui/tests/tui_pty_test.rs` | `termlens` driving `oxidelake-tui-demo` in a real PTY |
| CLI end-to-end | `oxidelake-runtime/tests/cli.rs` | `assert_cmd` on `oxide` — `gen-data` output verified by an independent `parquet`-crate read; spawned `oxide-scheduler` + `oxide-worker` must print byte-identical `--cluster` results |

## Updating STATUS.md

At every gate: set the phase row's status, name the command(s) that verified it, list what was deferred, and add each component to the verification matrix under exactly one column.
