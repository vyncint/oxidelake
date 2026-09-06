# Dependencies — pinned set and evidence

The pins live in [SPEC.md §4](SPEC.md). This page records *why* those numbers, so the chain can be re-derived instead of guessed.

## Coherence rules

1. **The chain has one root: Ballista.** Its documentation states: "Make sure the version of `datafusion` is the same as `ballista`'s!" Ballista fixes the `datafusion` / `datafusion-proto` major; DataFusion fixes `arrow` / `parquet` / `object_store`; `arrow-flight`, `tonic` and `prost` arrive transitively through Ballista and are **not declared directly** ([ADR-0013](decisions/ADR-0013-cluster-mode-on-ballista.md)).
2. Never depend on `datafusion-*` sub-crates — use `datafusion::` re-exports. Single exception: `datafusion-proto`, required for the plan codec, pinned to the identical version ([ADR-0009](decisions/ADR-0009-datafusion-reexports.md)).
3. Gate: `cargo tree --workspace -d -e normal` shows no duplicate `arrow-*`, `parquet`, `datafusion*`, `object_store`, `tonic` or `prost` majors.
4. Bump the whole chain or nothing. DataFusion 55 exists (2026-08) but Ballista is on 54 — Ballista wins until it moves.
5. `rust-version` is the **lockfile's** floor, not the direct dependencies' (2026-09-02: 1.94.1). `cargo deny check` gates advisories, licenses and sources; an advisory that cannot be fixed within the pinned chain is ignored only with a written reason in `deny.toml` (today: `quick-xml` behind `object_store`'s unused cloud backends, `paste` as a compile-time proc-macro) and re-examined at every chain bump. Dependabot proposes minor/patch updates weekly and is told to leave the chain's majors alone.

## Evidence (crates.io, 2026-08-25)

| Crate | Pin | Requires / pairs with | Notes |
|---|---|---|---|
| `ballista`, `ballista-core`, `ballista-scheduler`, `ballista-executor` | 54 (54.1.0, 2026-08-09) | `datafusion ^54`, `datafusion-proto ^54`, `arrow-flight ^58.3`, `object_store ^0.13.2`, `tonic ^0.14`, `prost ^0.14` | Declares MSRV 1.88, but `ballista-core` bundles the AWS SDK (`aws-config` → `aws-types` …) whose locked versions need **rustc 1.94.1** — the lockfile's true floor and therefore the workspace `rust-version` (verified by `make msrv` / the `msrv` CI job) |
| `datafusion` | 54 (54.1.0) | `arrow ^58.3.0`, `parquet ^58.3.0`, `object_store ^0.13.2`, `tokio ^1.52`, `sqlparser ^0.62` | MSRV 1.88 |
| `datafusion-proto` | 54 (54.1.0) | `arrow ^58.3.0`, `prost ^0.14.1` | `PhysicalExtensionCodec` for `Gpu*Exec` |
| `arrow` | 58 (58.4.0) | — | default features + `prettyprint`; `ffi` only under `oxidelake-api/python`; MSRV 1.85 |
| `parquet` | 58 (58.4.0) | — | default features cover arrow, snap, lz4, zstd; Bloom filters and page index are built in; MSRV 1.85 |
| `object_store` | 0.13 | — | `ObjectStore` trait implemented by `UringLocalFileSystem` |
| `arrow-flight` / `tonic` / `prost` | *(transitive)* | arrow-flight 58.4 → `tonic ^0.14.1`, `prost ^0.14.1` | brought by Ballista; checked only by the duplicate gate |
| `tokio` | 1.52+ (`full`) | — | DataFusion's floor |
| `async-trait` | 0.1 | — | only to implement `object_store`'s `ObjectStore` (a foreign `#[async_trait]` trait); never for our own traits |
| `ratatui` | 0.30 | `ratatui-crossterm ^0.1.2`; dev-dep `crossterm ^0.29` | enable feature `crossterm_0_29`; MSRV 1.88 |
| `crossterm` | 0.29 | — | must match ratatui's `crossterm_0_29` |
| `cudarc` | 0.19 | `libloading ^0.9` | `default-features = false`, features `std, driver, nvrtc, dynamic-loading, cuda-12080`; the `cuda-*` features are alternatives — enable exactly one; `cuda-version-from-build-system` needs a toolkit and is unusable here |
| `objc2-metal` | 0.3 | `objc2 >=0.6.2, <0.8`, `objc2-foundation ^0.3.2` | macOS target-gated, behind `metal` |
| `objc2` / `objc2-foundation` | 0.6 / 0.3 | — | |
| `io-uring` | 0.7 | — | Linux target-gated, behind `io-uring`; chosen over `tokio-uring` 0.5, which pins `io-uring ^0.6` and needs its own runtime ([ADR-0007](decisions/ADR-0007-io-uring-crate-dedicated-thread.md)) |
| `termlens` | 0.6 (dev) | default feature `insta` | real-PTY harness driving `oxidelake-tui-demo`; MSRV 1.85; read <https://docs.rs/termlens/0.6.1> before use |
| `insta` | 1 (dev) | — | snapshot files are committed |
| `assert_cmd` | 2 (dev) | — | CLI end-to-end tests, including the spawned scheduler + worker |
| `postcard` | 1 (`use-std`) | `serde` | codec payloads for `Gpu*Exec` parameters |
| `rand` | 0.10 (dev) | — | seeded generators only |
| `thiserror` / `anyhow` | 2 / 1 | — | libraries / binaries |
| `tracing` / `tracing-subscriber` | 0.1 / 0.3 (`env-filter`) | — | |
| `rayon`, `futures`, `bytes`, `clap` (`derive`), `serde` (`derive`), `tempfile` | 1 / 0.3 / 1 / 4 / 1 / 3 | — | |

Considered and not adopted in v1: **Lance** 10.0.0 (2026-08-07) requires `datafusion ^54.0.0` and `arrow ^58.0.0` — coherent with this chain today — but adds a third coupling to DataFusion's major and its ANN indices are CPU-side; documented as a v2 option ([ADR-0014](decisions/ADR-0014-parquet-and-arrow-ipc-not-a-custom-format.md)). `async-trait` for our own traits ([ADR-0005](decisions/ADR-0005-object-safe-sync-gpubackend.md)), `std::simd` ([ADR-0002](decisions/ADR-0002-stable-toolchain-no-std-simd.md)), `cuda-oxide` ([ADR-0003](decisions/ADR-0003-cudarc-dynamic-loading-nvrtc.md)), `tokio-uring` ([ADR-0007](decisions/ADR-0007-io-uring-crate-dedicated-thread.md)), `metal` / `metal-rs` ([ADR-0004](decisions/ADR-0004-objc2-metal-target-gated.md)), `memmap2` and `xxhash-rust` (no longer needed without a private format).

## Re-deriving the chain

```bash
cargo info ballista                        # latest version and rust-version — the root
cargo info datafusion                      # confirm the major Ballista requires exists / is latest of that major
cargo tree -p datafusion -i arrow          # after a first resolve: the arrow major DataFusion pulls
cargo tree --workspace -d -e normal        # must show no arrow / parquet / datafusion / object_store / tonic / prost duplicates
```

`cargo info <crate>` also prints the feature list — after any bump re-check `cudarc`'s `cuda-*` and loading-mode features and `ratatui`'s `crossterm_0_XX` features.
