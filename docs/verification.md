# Verification

## The gate

Run after every phase; every command must pass before that phase's commit. `make gate` runs the checks below, including MSRV, plus the io_uring lane on Linux and the full Metal lane on macOS. [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) uses the same Make targets. `commit-policy` runs beside it. Keep the Makefile, this page and the workflow in sync.

CI separates default, io_uring and predict tests into three Linux jobs. Clippy
also has default / CUDA / predict jobs. macOS runs Metal lint, Metal package
tests together with the PTY suite, then on-device conformance when a device is
available. The remaining checks are MSRV, all-feature rustdoc, `cargo deny`,
release scripts, crate metadata, CI policy tests, the vendored termlens
skill's version and `zizmor`. The Linux and macOS test lanes run with
`TERMLENS_ARTIFACT_DIR` set and, on failure, render every screen the PTY
suite left behind into the job summary. `required-green`
aggregates their results under the stable name required by branch protection.

```bash
make fmt-check lint lint-cuda lint-predict
make test test-predict
make test-io-uring                    # Linux only
make check-cuda check-predict-no-second-cuda
make doc coherence deny
make release-scripts crate-metadata ci-scripts skill-version zizmor
```

`make skill-version` compares the `Written against **termlens X.Y.Z**` line in
`.claude/skills/termlens/SKILL.md` with the `termlens` requirement in
`Cargo.toml` (major.minor). CI runs it as the `skill-version` job.

One CI step is deliberately **not** in `make gate`:

```bash
make test-termlens-cli                # installs termlens-cli from crates.io
```

It runs the `#[ignore]`d `oxidelake-tui/tests/termlens_cli.rs` suite, which
drives the committed screens through the `termlens` command at the version
`Cargo.lock` names. The gate has to run on a machine with no network, and a
`cargo test` on a published crate must not install anything behind a
contributor's back — so CI asks for it by name, in the default Linux test
lane. The macOS on-device pass skips it (`--skip termlens_cli`) for the same
reason.

The Metal lane (macOS, since 2026-09-01 — CI runs it as the `metal` job on a
macOS arm64 runner):

```bash
make lint-metal test-metal test-metal-device
```

`test-metal` and `test-metal-device` use the same package/feature arguments,
including the TUI test targets, so Cargo can reuse their dependencies and test
binaries. The latter probes `MTLCreateSystemDefaultDevice()` and logs a skip
when no device exists. Otherwise it sets `OXIDE_BACKEND=metal` and runs the
ignored conformance test, which asserts GPU transfers as well as correct
results. CPU fallback alone cannot satisfy it.

Supply chain and toolchain claims are part of the gate too:

```bash
make deny                             # all-feature RustSec/license/ban/source checks
make msrv                             # the declared rust-version really builds the workspace (CI job `msrv`)
```

Every `deny.toml` ignore carries the reason the affected code path is unreachable in OxideLake and is revisited when the dependency chain moves ([dependencies.md](dependencies.md)).

## CI reuse, selective checks and measurement

Ordinary PR/main CI caches external dependencies for the Linux test lanes and
Metal job. Each lane has its own OS/architecture key; the pinned cache action
also keys by compiler, manifests, lockfile and compiler environment. Workspace
crates and installed tools are excluded. All Cargo commands still run after a
cache hit. Clippy/MSRV/rustdoc keep their independent checks; initially their
shorter builds are uncached to limit cache storage and eviction.

The release workflow always runs a full gate without restoring or saving build
caches. Manual CI dispatch also runs the full gate and accepts `clean=true`
to bypass caches. Ordinary main/PR CI cancels a superseded run; reusable release
gates and manual runs are not cancelled by that policy. See [ADR-0017](decisions/ADR-0017-ci-dependency-reuse.md).

The `changes` job tests the CI scripts and classifies the actual checkout diff.
Only Markdown under `docs/` and the explicit root documentation allowlist in
`.github/scripts/ci-policy.py` can skip Rust jobs. `README.md`, crate files,
workflows, scripts, lockfiles and unknown paths still run the full gate. Missing
base commits and empty diffs also run it. Release scripts, package metadata,
CI policy tests and workflow security checks always run. `required-green`
allows only these documented skips; any failure, cancellation, missing job or
invalid classification fails the gate.

`make ci-scripts` covers source deletion/rename handling, malformed results,
each required job's failures, Metal build-argument reuse, and stress coverage.
It also verifies the metadata checker under the platform's Bash and the MSRV
target with a standalone Cargo ahead of the Rustup proxy on `PATH`.
`make stress-tui` builds release test targets once per OS, then runs thread
counts 1, 4 and 16 with 20/40/40 percent of `ITERS` (default 100), at least once
per count. The first failing build or iteration fails the workflow.

Test builds emit Cargo `--timings` reports. CI and stress upload these as
`timings-*` artifacts retained for seven days. Compare a clean run, a warm run,
and a small source change on the same toolchain; record queue time separately
from job execution and include cache transfer time and aggregate runner minutes.
Do not describe expected cache or parallelism gains as measured speedups.

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
| TUI end-to-end | `oxidelake-tui/tests/tui_pty_test.rs` | `termlens` driving `oxidelake-tui-demo` in a real PTY: three text snapshots, one styled, and the focus/backend colours as cell assertions |
| TUI emulator invariants | `oxidelake-tui/tests/emulation.rs` | `Screen::unsupported()` pinned exactly, no terminal mode left set, snapshot-text and JSON round trips |
| TUI saved screens | `oxidelake-tui/tests/termlens_cli.rs` | `termlens-cli` on the committed `.snap` files — `render`, and `diff`'s 0/1/2 exit codes (`#[ignore]`d; `make test-termlens-cli`) |
| Shipped TUI binary | `oxidelake-runtime/tests/oxide_tui_pty.rs` | `oxide tui` in a PTY: the same frame `oxidelake-tui` snapshots, and no log line on the grid |
| CLI end-to-end | `oxidelake-runtime/tests/cli.rs` | `assert_cmd` on `oxide` — `gen-data` output verified by an independent `parquet`-crate read; spawned `oxide-scheduler` + `oxide-worker` must print byte-identical `--cluster` results |

## Updating STATUS.md

At every gate: set the phase row's status, name the command(s) that verified it, list what was deferred, and add each component to the verification matrix under exactly one column.
