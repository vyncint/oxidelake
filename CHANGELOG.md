# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0, minor
versions (0.x) may contain breaking changes; they are always listed under a
**Changed** or **Removed** heading. The nine crates version together.

## [Unreleased]

## [0.2.0] - 2026-09-21

The observability release. A GPU deployment that silently ran everything on
the CPU used to produce identical rows, identical `EXPLAIN` tags and nothing
above `debug` — this release makes that visible, from a counter in the
dashboard to a Prometheus endpoint on every worker.

**It does not complete the production-readiness milestone.** Ten lines of
[Phase 8](docs/roadmap.md) remain open; three of them (#28, #45, #46) need a
CUDA device to verify on and one (#29) is what the spill manager's real
wiring waits for.

### Added

- **The CPU-fallback counter** (#32). `fallback_batches` on `OperatorStats`
  and `OperatorSnapshot` counts every batch that took the CPU reference path
  although its operator was placed on a device. The TUI Inspector shows
  `cpu fallback N / M` — green at zero, yellow when some batches fell back,
  red when all of them did — and each operator logs one `warn!` on its first
  fallback naming the reason. A silent fallback was previously detectable
  only by re-running with `RUST_LOG=oxidelake_compute=debug`.

- **Placement notes in `oxide explain`** (#32). A node the rule leaves on the
  CPU is an ordinary DataFusion operator in the plan, indistinguishable from
  one that was never eligible. `oxide explain` now prints, under the plan,
  every node that was skipped and why — `AggregateExec: \`avg\` is not one of
  the aggregates the device kernels implement (sum, count, min, max)`, `the
  grouping key \`s\` is Utf8View; the device kernel groups on Int64`. A plan
  that lowered completely prints nothing.

- **Spans, a per-query log line and a metrics endpoint** (#33). An
  `oxide.operator` span per operator and partition carrying `operator`,
  `partition`, `target` (what the plan says) and `backend` (what is
  executing). `OxideSession::collect` logs one `INFO` line per query with the
  mode, rows, elapsed time and fallback count; `oxide sql` uses it, so
  `RUST_LOG=oxidelake_runtime=info oxide sql …` prints one line per query.
  Behind the new `metrics` feature, `oxide-worker --metrics-port` and
  `oxide-scheduler --metrics-port` serve `oxide_*` counters as Prometheus
  text. Cluster executors never run the placement rule, so the plan codec
  attaches a process-wide `TelemetryHub` to every `Gpu*Exec` it decodes —
  without that a worker's `/metrics` would describe no work at all. The
  endpoint is unauthenticated, like the Ballista ports themselves: bind it on
  a private interface. Without the feature `--metrics-port` is refused, never
  ignored.

- **`--batch-size`, `--output table|json|csv`, worker `--backend`** (#49).
  `--batch-size` overrides the CPU default (8192) and the GPU-target default
  (65536) on `sql`, `explain` and `tui`; zero is refused rather than silently
  returning nothing. `--output json` writes an array of objects and
  `--output csv` RFC 4180 with a header — an empty result still prints the
  header and `[]`, so a script can tell "no rows" from "the query failed".
  `oxide-worker --backend` mirrors `OXIDE_BACKEND` and is applied before the
  first task, so it reaches the operators rather than being accepted and
  ignored. Every flag and environment variable is now in one README table.

- **`SessionOptions`** (#49), with `OxideSession::local_with_options` and
  `connect_with_options`. `#[non_exhaustive]` with builder methods, so a knob
  added later is a minor release.

### Changed

- **`predict` reads its activation from the model file** (#47). **Breaking
  for existing model files.** A `safetensors` header declares
  `{"__metadata__": {"oxidelake.activation": "relu"}}` — `relu`, `gelu`,
  `sigmoid`, `tanh` or `none`, applied between the `Linear` layers and never
  after the last. A file that declares nothing is refused with a message
  naming the key and the alternative: `predict(path, features, 'relu')`, a
  new third argument that must agree with the header when the file has one.

  Fixing this found a worse bug. The loader built a stack of bare `Linear`
  layers and applied **no activation at all**, while the module docs, the
  README and ADR-0015 all said "ReLU between them" — so every non-linear
  model returned a linear model's numbers, and the only test of the numbers
  compared them against the same bare stack. The `none` case is now checked
  against oxmera's own `Sequential::forward`, which does not share the
  loader's loop, and `relu`, `gelu` and `none` must disagree with each other.

- **Public enums are `#[non_exhaustive]`** (#41). `BackendKind`,
  `Compression`, `SessionMode`, `MemoryClass`, `MemoryTier`, `Tier`,
  `ModelSpec`, `Activation` and the TUI's `Panel` / `KeyInput` / `Transition`
  now need a `_` arm in a downstream `match`. `ModelSpec::SequentialMlpRelu`
  is gone, replaced by `ModelSpec::SequentialMlp { activation }` (see
  `predict`, below). The **plan vocabulary stays
  exhaustive on purpose** — `Predicate`, `Comparison`, `Literal`,
  `AggregateFunction`, `DistanceMetric` and the codec's `GpuNode`: every
  kernel must handle every variant, so a `_` arm in the CPU correctness
  reference is a wrong answer that compiles. `docs/RELEASING.md` records the
  distinction, and the DataFusion coupling that `cargo-semver-checks` cannot
  see.

- **The plan codec carries a version fingerprint** (#42). `postcard` is not
  self-describing, so a `GpuNode` field added or reordered between builds
  decodes without complaint into the wrong parameters and a mixed-build fleet
  computes a confident wrong answer. Every encoded plan now carries a magic,
  a format version and a fingerprint over the crate version and the compute
  layer's feature mask; a plan from a different build is refused at decode
  with both fingerprints named. A committed snapshot pins the bytes of all
  four variants, so a wire-format change cannot land unnoticed.

- **Memory-tier gauges show what the backend reports** (#25). Capacities come
  from the local backend's `MemoryInfo` — the device's total for a GPU
  backend, host RAM for the CPU one — instead of the fixed 8/4/2 GiB
  constants the gauges used to be drawn against, and a tier the engine has no
  number for is drawn without a ratio rather than against a plausible one.

- **The dashboard says the spill manager is library-only** (#25).
  `SpillManager` is not on the query path: no operator registers a batch with
  it, so the tier gauges and spill counters stay at zero however much memory
  a query used. The Telemetry panel says so (`spill library only: no query
  registers batches`), as do the README, `STATUS.md` and
  `docs/architecture.md`. Wiring it in is not a matter of calling it from the
  join — the build side is probed by every batch — and waits on the streaming
  aggregate (#29).

- **`OperatorSnapshot` and `TelemetrySnapshot` gained public fields**
  (`fallback_batches`, `capacity`). Both are exhaustively constructible, so a
  downstream struct literal over them no longer compiles. They are snapshot
  types read field by field in practice, which is why they stay plain structs
  rather than becoming `#[non_exhaustive]` — a reader should be able to
  destructure one.

- **An unavailable backend names the backend, not the way it was asked for.**
  `oxide-worker --backend cuda` reaches the same code as `OXIDE_BACKEND`, so
  the message no longer sends the reader to a variable they did not set.

- **Logs are uncoloured when stderr is not a terminal.** ANSI escapes in a
  redirected log file are noise, and they break a grep for `field=value`.

### Fixed

- **`oxide sql` prints the columns of an empty result.** The CSV header and
  the JSON `[]` are written from the plan's schema, which an empty result
  still has.


## [0.1.4] - 2026-09-20

### Fixed

- **io_uring: the worker reaps its completion before returning** (#27).
  `submit_one` pushed an SQE and returned early when `submit_and_wait`
  failed — but after a successful push the SQE belongs to the kernel, so
  that dropped the caller's read buffer while the kernel could still be
  writing into it, and left the completion to be reaped by the *next* job,
  which read the stale result as its own. `user_data` was a constant per
  operation kind, so nothing could tell the two apart. EINTR is a signal
  rather than a failure and is retried; other errors are reported only
  after the completion is in hand; every SQE carries the worker's next
  submission number and a foreign completion is a hard error. Exposure was
  limited only because no session constructs this store (#24).

- **`GetOptions` preconditions are honoured by both stores** (#44).
  `UringLocalFileSystem::get_opts` read `range` and `head` and ignored the
  rest, while `LocalFileSystem` checks `if_match` / `if_none_match` /
  `if_modified_since` before reading a byte — so a conditional read was
  honoured or silently dropped depending on which store a caller held, of
  two the documentation calls identical. The cases live in the shared
  conformance body now. The worker's job channel is bounded too: one SQE
  is in flight at a time, so an unbounded queue only moved the backlog out
  of the callers and into memory.

- **`oxide sql` survives a closed pipe** (#34). `… | head -1` exited 101
  with `failed printing to stdout: Broken pipe`, from `println!` inside
  DataFusion's `DataFrame::show()`. Output is formatted and written
  directly now, and `BrokenPipe` means success — the reader got what it
  asked for. `explain` and `gen-data` take the same path.

### Added

- **A proof for page-index pruning**, beside the ones for statistics and
  Bloom filters (#43). The README claimed three and two had proofs. It
  needs a dataset the other two cannot touch or it restates them: a sorted
  column in a single row group, where min/max cannot exclude the row group
  and there is no Bloom filter. A 1,000-row predicate prunes 198,192 of
  200,000 rows with no row group pruned at all. The test writes that file
  with the `parquet` crate directly, because the 1,000-row data page limit
  it needs is a test's requirement rather than a lake's — no public API
  changed.

### Changed

- **CI installs `termlens-cli` prebuilt** instead of compiling it from
  source on every run (#48), at the version `Cargo.lock` names. The test
  refuses a binary whose `--version` disagrees with the lockfile, so the
  shortcut cannot silently test a different tool.

### Documentation

- **README and STATUS agree about CUDA again** (#36). README said the CUDA
  backend "has not yet run on a CUDA machine"; STATUS's matrix has
  recorded since 2026-09-06 that a T4 NVRTC-compiled and launched every
  kernel — and that the aggregation kernel's correctness bug was found
  there. Under-claiming breaks the ADR-0012 ledger as thoroughly as
  over-claiming. The header names the published version, says what "v1
  complete" means (the internal phase plan, not a 1.0), and the README
  leads with "query engine" and says a catalog, a metastore and
  transactions are out of scope.
- **The io_uring store is documented as not wired into sessions** (#24).
  SPEC §2.4 said it was "registered into the session's `RuntimeEnv`";
  nothing constructs it, so `--features io-uring` changes nothing at run
  time. Demoted rather than wired, because the 2026-09-02 audit measured
  the ring slower than `LocalFileSystem` here — the roadmap line is
  unticked with that measurement named as the blocker.
- **Pinned memory is available, not what the operators use** (#26).
  architecture.md and SPEC §2.1 described DMA without staging copies; the
  operator path uploads from pageable Arrow buffers and downloads into
  pageable `Vec`s, so any transfer number measured today is a pageable
  number.
- **Phase 8 — production readiness** in the roadmap (#23), listing every
  open issue in the `v0.2.0` milestone with the acceptance criterion that
  would close it, and folding in the audit's deferred plans so there is
  one list. The Phase 6 claim of "double-buffered stream pipelines" is
  unticked: `round_trip` is serial per batch, as the audit itself says.
- **The CI build-cache policy is measured and settled** (#39). SECURITY.md
  and ADR-0016 said "no build cache" while three jobs cached; the
  exception had never been justified with a number. Measured: 7 min warm,
  19–29 min cold, against a 59 min pre-cache median. The caches stay and
  the documents say what they actually are — dependency-only, never
  restored for a release — with the run URLs in
  `docs/ci-cache-measurement.md`.

### Changed

- **termlens 0.10.1 → 0.11**, with the vendored skill and the report
  action's pin in `ci.yml` and `stress.yml`. 0.11 is termlens's stability
  candidate: from it no promised item changes incompatibly before its 1.0,
  so this requirement should hold for a while.

  Its one breaking change lands here as a simplification.
  `Screen::unsupported()` returns a view instead of a slice of `Arc<str>`,
  and `unsupported_overflow()` folds into it — so the pinned list and "the
  record is not truncated" are one assertion, in both PTY suites: the
  dashboard's (`oxidelake-tui/tests/emulation.rs`) and the shipped
  binary's (`oxidelake-runtime/tests/oxide_tui_pty.rs`). The view compares
  equal to a slice only when the retained shapes match *and* nothing
  overflowed the bound, so a truncated record can no longer pass as a
  shorter list.

  The pin's known-defect caveat goes with it: termlens#320, which named
  blink and strikethrough as unsupported although the attribute shadow
  implements them, was fixed upstream in 0.10.2.

- **The PTY test harness moved to termlens 0.10.1** (from 0.9). The three
  committed screen snapshots are unchanged: `assert_screen_snapshot!` records
  styles by default in 0.10, and the text snapshots opt out with
  `styles = false` so a colour change lands in one new styled snapshot rather
  than rewriting three large files. The vendored agent skill
  (`.claude/skills/termlens/SKILL.md`) was refreshed to match and is now
  checked against the dependency by `make skill-version`, a CI job — it had
  drifted two releases behind without anything noticing.

### Added

- **The dashboard's colours are tested.** Nothing asserted them before:
  `TestBackend::to_string()` is text-only and the PTY snapshots were plain,
  so the focused panel's yellow border and the green/blue backend tags were
  invisible to the whole suite. `[CUDA]` is now green and `[CPU]` blue at
  *every* occurrence (`Screen::find_all`), `Tab` is asserted to move the
  highlight without changing one character of text, and `↓` is asserted to
  leave the telemetry gauges and the Describe table untouched
  (`Screen::diff`).

- **`crates/oxidelake-tui/tests/emulation.rs`**: the invariant the rest of the
  PTY suite rests on. `Screen::unsupported()` is pinned to exactly
  `["^[[59m"]` — ratatui's underline-colour reset, which changes no cell — so
  a sequence the emulator silently drops can no longer make every screen
  assertion true against a wrong grid. Insert mode, bells, wrapped rows and
  mouse modes are pinned beside it, and a dashboard screen is round-tripped
  through the snapshot text format and through JSON.

- **`crates/oxidelake-runtime/tests/oxide_tui_pty.rs`**: `oxide tui` — the
  dashboard as users install it — in a real PTY. It renders the same frame
  `oxidelake-tui` snapshots, it gives the terminal back, and the `tracing`
  subscriber writing to stderr (the dashboard's own stream in a terminal)
  puts nothing on the grid, `RUST_LOG=info` included. `assert_cmd` captures
  pipes and could see none of that.

- **`crates/oxidelake-tui/tests/termlens_cli.rs`**: the committed `.snap`
  files read back with `termlens-cli` — `render --text/--svg/--html` keeping
  the palette, and `diff`'s 0/1/2 exit codes on this repository's own
  screens. `#[ignore]`d, because a published crate's `cargo test` must not
  install a tool behind a contributor's back; CI runs it as
  `make test-termlens-cli`.

- **CI renders a failing PTY screen instead of logging it.** The Linux and
  macOS test lanes run with `TERMLENS_ARTIFACT_DIR` set, and on failure the
  pinned `vyncint/termlens` report action writes every screen the suite left
  behind — and every insta `.snap.new` — into the job summary, with SVG and
  HTML uploaded.

### Fixed

- **`oxide tui` now rejects non-interactive stdin/stdout before terminal setup.**
  It prints a clear hint to use `oxide sql` or `oxide explain` for scripted output
  and exits with usage code 2 instead of leaking a raw ENXIO-style terminal error.

- **The dashboard's Describe panel no longer prints a truncated percentile as
  if it were the value.** `approx_percentile_cont` renders full precision, and
  the panel's six-character columns were clipped from the right, so the 2M-row
  demo table's P99 of `id` (1979969.24) displayed as `197999` — ten times too
  small and below the median in the same row. Numeric cells that do not fit
  are now rounded to the most decimals that fit, falling back to an exponent
  form (`1.98e6`); a value that already fits is left exactly as the query
  rendered it. Over-long text cells are marked with an ellipsis rather than
  cut silently, and the column widths now live in one constant the cell
  formatters and the layout share.

## [0.1.3] - 2026-09-07

Release-build changes only; no crate code changed since 0.1.1. This is the
first release whose binaries were built under thin LTO.

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

[Unreleased]: https://github.com/vyncint/oxidelake/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/vyncint/oxidelake/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/vyncint/oxidelake/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/vyncint/oxidelake/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/vyncint/oxidelake/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/vyncint/oxidelake/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/vyncint/oxidelake/releases/tag/v0.1.0
