# Working on oxidelake

Instructions for coding agents — and useful to humans. **oxidelake** is a GPU-accelerated, Arrow-native distributed analytical query engine and columnar lakehouse: the same plans run embedded (DuckDB-like) or across a Ballista cluster (Spark-like), with GPU operators that always fall back to a CPU reference.

This file is the canonical brief; `CLAUDE.md` points here. `CONTRIBUTING.md`
is the full contributor document and wins wherever the two disagree.

## Layout

- `crates/oxidelake-core|memory|device` — types, buffers, and the object-safe
  `GpuBackend` behind which CUDA and Metal live. `oxidelake-device/cpu` is the
  **correctness reference**; every GPU kernel is asserted equal to it.
- `crates/oxidelake-compute` — the `Gpu*Exec` DataFusion `ExecutionPlan`s and the
  SQL UDFs. Each operator falls back per batch to the CPU path, which is what
  keeps a heterogeneous cluster correct.
- `crates/oxidelake-planner|runtime|storage|api|tui` — placement rule and plan
  codec, sessions and the three binaries, Parquet/Arrow IPC, the DataFrame
  surface, and the terminal dashboard.
- `docs/decisions/` — the ADRs. **Read the one that covers what you are
  touching before you touch it**; most non-obvious choices here are recorded
  rather than inferable.

## Build and test

```sh
make gate            # the whole quality gate — run this before pushing
make test            # the default-feature suite
make lint-cuda       # the GPU compile gate: cuda builds with no CUDA installed
```

The workspace root is virtual and rejects `--features`, so a feature build is
always `-p <crate>`. Stable toolchain, MSRV 1.94 checked against the committed
lockfile.

## Things that will bite you here

- **The default build is pure CPU, and every optional capability is a
  feature** (ADR-0006): `cuda`, `metal`, `io-uring`, `predict`. A feature that
  no job builds is a feature that is already broken — add it to `make gate`
  and to `ci.yml` in the same change that adds it.
- **The CPU backend is the reference, not a fallback of convenience.** If a
  GPU kernel and the CPU path disagree, the CPU path is right until proven
  otherwise, and the conformance suite is where that is settled.
- **`predict` must never pull a second CUDA stack into the process.** oxmera
  is depended on with `default-features = false` because it registers CUDA
  from a `#[ctor]` that runs before `main`; `make check-predict-no-second-cuda`
  enforces it (ADR-0015).
- **The TUI emits DEC 2026 synchronized updates**, which most ratatui apps do
  not — so its PTY tests use termlens's `wait_frame`, not `snapshot_after`.
  A snapshot taken any other way can catch a half-painted screen, and did.
- **`.claude/skills/termlens/SKILL.md` is a vendored copy, and `make gate`
  checks its version against the dependency.** Bumping `termlens` in
  `Cargo.toml` means copying the skill over in the same change
  (`cp ../termlens/skills/termlens/SKILL.md .claude/skills/termlens/SKILL.md`);
  the `skill-version` job fails otherwise. Guidance for a version that is no
  longer here is worse than none.
- **The three text screen snapshots are text on purpose.** termlens 0.10
  records styles by default; `tui_pty_test.rs` passes `styles = false` there
  and keeps the colours in one styled snapshot plus direct cell assertions,
  so a colour change does not rewrite three large files. `emulation.rs` pins
  what the emulator could not render — read it before trusting a screen
  assertion that suddenly went green.

## The rules that will fail CI

Three, and they are the same in every one of these repositories.

1. **Conventional Commits.** `feat:`, `fix:`, `docs:`, `test:`, `ci:`,
   `chore:`, `refactor:`, `perf:` — imperative mood, subject line under 72
   characters, scope optional (`fix(screen): …`).
2. **DCO sign-off.** `git commit -s`, and the `Signed-off-by:` email must
   match the commit author's. Forgot? `git commit --amend -s --no-edit`, or
   `git rebase --signoff main` for a branch.
3. **No AI attribution.** See below — this one is about you, and it is the
   rule most likely to catch an agent out.

Run them yourself before pushing; both scripts take a commit range:

```sh
.github/scripts/check-dco.sh main..HEAD
.github/scripts/check-no-ai-attribution.sh main..HEAD
```

## Using AI here

**You are welcome.** Every one of these projects was built with AI assistance
and says so in its CONTRIBUTING. Use whatever helps.

**You are not a contributor.** Do not add yourself to the history:

- no `Co-Authored-By:` trailer naming an assistant, a model, or a vendor,
- no "Generated with …" footer, no robot emoji,
- no bot account as author or committer — save a named dependency bot, which
  is exempt from the identity rule only and still checked on its message.

The human who opens the pull request is the author of record and takes
responsibility for the change under the DCO. That is what the sign-off
certifies, and it cannot be certified by a tool. `.claude/settings.json`
turns co-author trailers off for agents that read it; the check in CI is the
boundary, and it reads every commit in the range.

If CI catches one, the fix is to rewrite the message, not to argue with it:

```sh
git commit --amend            # the last commit
git rebase -i main            # several, marking each `reword`
git push --force-with-lease
```

## What good work looks like here

These repositories share a house style, and it is stricter than most:

- **Evidence over assertion.** A bug report says what was measured against
  which released version. "Reproduced against 0.4.0" is the standard; "the
  code looks wrong" is not. Issues in these repos read *Today / Why it is
  worth fixing / Fix / Done when*, with a concrete reproduction.
- **Every change lands with a test**, and the test must be able to fail. If
  you add a guard, prove it catches the thing — break it once and watch it go
  red before you commit.
- **Comments say *why*, never *what*.** The diff shows what. A comment earns
  its place by recording the reason, the alternative rejected, or the failure
  that motivated the line.
- **Say what you did not do.** A pull request that lists what it left out and
  why is worth more than one that implies completeness. If something is
  unverified, say so — an honest gap is cheap and a false claim is expensive.
- **Documentation is checked, not maintained.** Where a README states a fact
  the code owns, there is usually a test asserting the two agree. Do not
  break that pattern by hand-editing the doc.

## Pull requests

Branch from `main` (`feat/…`, `fix/…`, `docs/…`, `ci/…`). PRs are
**squash-merged**, so the PR title becomes the commit subject on `main` —
write it as a Conventional Commit. Update `CHANGELOG.md` under
`[Unreleased]` for anything user-facing.

Direct pushes to `main` are blocked by a ruleset; everything goes through a
pull request, including releases.

## Releasing

Tag `vX.Y.Z` on `main`; `release.yml` verifies the tag against the workspace version and the changelog, re-runs the gate, publishes nine crates in dependency order via Trusted Publishing, and cuts the GitHub Release with binaries. See `docs/RELEASING.md`.
