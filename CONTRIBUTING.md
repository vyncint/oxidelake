# Contributing to OxideLake

Thanks for your interest. This page is the short version of how the project
works; the long version is the executable spec in [docs/SPEC.md](docs/SPEC.md) and
the architecture decision records in [docs/decisions](docs/decisions/README.md).

> **These four projects share one contributor pattern** — the same commit
> rules, the same DCO, the same AI policy, the same CI and release shape:
> [termlens](https://github.com/vyncint/termlens),
> [mossaic](https://github.com/vyncint/mossaic),
> [launchbound](https://github.com/vyncint/launchbound),
> [reconverge](https://github.com/vyncint/reconverge). Learn it once.

## 1. Dev setup

`rust-toolchain.toml` pins the development toolchain (stable, with `rustfmt`
and `clippy`). `Cargo.toml`'s `rust-version` is the minimum the workspace
builds with — it is the *lockfile's* floor and is verified by `make msrv` and
the `msrv` CI job. If you raise it, say why in the commit message and update
`clippy.toml`.

```bash
make help          # every target
make gate          # what CI runs
make metal         # macOS: Metal lint, tests and the on-device conformance suite
make quickstart    # release build + 1M-row demo + a query
```

The workspace root is virtual and rejects `--features`, so a feature build is
always `-p <crate>` — `cargo check -p oxidelake-runtime --features cuda`.

## 2. Project layout

See [AGENTS.md](AGENTS.md) for the crate-by-crate map, and
[docs/architecture.md](docs/architecture.md) for how the layers fit together.
The ADRs under [docs/decisions/](docs/decisions/README.md) record the
non-obvious choices; read the one covering what you are touching first.

## 3. Testing policy

| Layer | Where |
|---|---|
| Unit | colocated `#[cfg(test)]` modules |
| Operator conformance vs stock DataFusion | `crates/oxidelake-compute/tests/conformance.rs` |
| Placement + codec | `crates/oxidelake-planner/tests/placement.rs` |
| Storage (pruning proofs, IPC, object stores) | `crates/oxidelake-storage/tests/` |
| Embedded GPU-target equivalence, in-process cluster, CLI end to end | `crates/oxidelake-runtime/tests/` |
| DataFrame API vs SQL | `crates/oxidelake-api/tests/dataframe.rs` |
| TUI snapshots and real-PTY driving | `crates/oxidelake-tui/tests/` |

New behaviour needs a test at the lowest layer that can observe it; a bug fix
needs a regression test that is red on the old code (say so in the PR).

## 4. Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, `refactor:`, `perf:` —
scope optional (`feat(planner): …`). Subject line: imperative mood,
≤ 72 characters.

## 5. Developer Certificate of Origin (DCO)

Every commit must be signed off:

```sh
git commit -s
```

This appends `Signed-off-by: Your Name <you@example.com>` and certifies you
wrote the change or otherwise have the right to submit it under the project
license — the [Developer Certificate of Origin](https://developercertificate.org),
the same lightweight model the Linux kernel uses. The sign-off email must
match the commit author email; CI enforces this on every commit in a PR.

**There is no CLA. DCO only.** You keep your copyright.

Forgot to sign off? `git commit --amend -s` for the last commit, or
`git rebase --signoff main` for a whole branch, then force-push.

One exception, and it is GitHub's rather than ours: a pull request
**squash-merged through the web UI** has its author email rewritten by GitHub
*after* the sign-off was written, so an exact match is impossible by
construction. Such a commit must carry a sign-off, but is not matched against
an author it did not choose. The commits that went into the PR were already
checked, address and all, on the branch.

GitHub also *writes* that message, and it drops the trailers of the commits it
squashed whenever the branch contained a merge commit — pressing **Update
branch** is enough to cause it. The merge then lands on main carrying no
sign-off, and main is linear and non-fast-forward, so it cannot be repaired.
The check therefore exempts exactly one commit — the tip of a push to main,
which can only get there through a pull request that was already checked
strictly. **Keep your branch up to date by rebasing, not merging:**

```sh
git fetch origin && git rebase origin/main
git push --force-with-lease
```

That also matches what main requires: linear history, so a merge commit on
your branch is only ever going to be squashed away.

## 6. AI tooling policy

**AI assistance is welcome here — use whatever helps.** Every one of these
projects was built with it. There is an [AGENTS.md](AGENTS.md) briefing coding
agents on the layout, the commands, and the house style.

**AI attribution is not welcome.** No `Co-Authored-By` trailer naming an
assistant, model or vendor; no "Generated with …" footer; no robot emoji; no
bot identity as author or committer, save the one carve-out below. Whoever
opens the pull request is the author of record, takes responsibility under the
DCO, and the history should say so — a tool cannot certify the DCO, which is
the whole point of it.

This is enforced, not requested: `commit-policy.yml` runs
[`check-no-ai-attribution.sh`](.github/scripts/check-no-ai-attribution.sh) and
[`check-dco.sh`](.github/scripts/check-dco.sh) over every commit in a pull
request. Run them yourself first — both take a range:

```sh
.github/scripts/check-dco.sh main..HEAD
.github/scripts/check-no-ai-attribution.sh main..HEAD
```

If a check fails, rewrite the message rather than arguing with it:

```sh
git commit --amend            # the last commit
git rebase -i main            # several, marking each `reword`
git push --force-with-lease
```

`.claude/settings.json` turns co-author trailers off for agents that read
repository settings. That is a courtesy; the check in CI is the boundary.
Contributions authored *by* an autonomous account are not accepted.

**One carve-out: a named dependency bot.** Dependabot is exempt from the
*identity* half of the check and from nothing else. The rule exists so that a
human is not displaced as the author of record, and a version bump has no
human to displace — it is not somebody's work with the credit misassigned. The
message rules still apply in full, so a bot cannot carry an AI co-author
trailer, a "Generated with" footer or a robot emoji past the check either.
Adding another bot means naming it in `check-no-ai-attribution.sh`: the
allowlist is a list on purpose, so that widening it is a visible decision.

## 7. PR flow

- Branch from `main`; name branches `feat/…`, `fix/…`, `docs/…`, `ci/…`.
- PRs are **squash-merged** — keep the PR title in Conventional Commit form,
  since it becomes the commit subject on `main`. Branches are deleted on merge.
- Required checks: `required-green` (CI policy, fmt, clippy on
  default/cuda/predict, Linux tests on default/io-uring/predict, Metal and PTY
  tests, MSRV, docs, deny, release scripts, zizmor), plus `commit-policy`
  (DCO + attribution). Both must pass before merge; direct pushes to `main`
  are blocked by a ruleset. Only known documentation-only changes may skip
  Rust jobs; failures and unexpected skips still fail `required-green`.
  [Verification](docs/verification.md) explains the cache and skip policy.
- **Every change lands with a test, and the test must be able to fail.** If
  you add a guard, break it once and watch it go red before you commit.
- **Say what you did not do.** A PR that lists what it left out and why is
  worth more than one implying completeness. An honest gap is cheap; a false
  claim is expensive.
- **Contributing from a fork?** Two things are normal. On your first PR the
  workflows wait for a maintainer to approve them — GitHub's standard
  first-time-contributor safeguard, nothing you did wrong. And when
  `commit-policy` fails on a fork PR it cannot post its explanatory comment
  (fork PRs get a read-only token); the job log carries the full explanation,
  including the offending commit and the command that fixes it.
- Review: expect actionable review within a few days. Small, focused PRs get
  reviewed faster. Update `CHANGELOG.md` under `[Unreleased]` for any
  user-facing change.

## 8. Release process

Releases are cut by maintainers only; the checklist lives in
[docs/RELEASING.md](docs/RELEASING.md). Nine crates version together and
publish in dependency order through crates.io Trusted Publishing.

## 9. Ground rules

1. **The gate is the definition of done.** `make gate` must be green before a
   change is pushed; CI runs the same list (see
   [docs/verification.md](docs/verification.md)). Formatting, clippy with
   `-D warnings` on the default, `cuda`, `predict` and (on macOS) `metal`
   features, the full test suite, rustdoc with warnings denied, the
   dependency-coherence check and `cargo deny` are all part of it.
2. **Honesty rule.** Nothing is reported as working unless it ran. GPU code that
   only compiled is recorded as exactly that in [STATUS.md](STATUS.md)'s
   verification matrix. Never fabricate benchmark numbers.
3. **No panics in library code.** `unwrap`, `expect`, `todo!` and
   `unimplemented!` are denied by the workspace lints (test modules may allow
   them locally). Unimplemented paths return `EngineError::Unsupported`.
4. **`unsafe` is confined and justified.** Allocator/FFI internals only; every
   block carries a `// SAFETY:` comment stating the invariant it relies on, and
   every raw resource is owned by an RAII type.
5. **Dependency chain has one root.** Ballista fixes the DataFusion major,
   DataFusion fixes arrow/parquet/object_store. Bump the whole chain or
   nothing; never add a `datafusion-*` sub-crate ([docs/dependencies.md](docs/dependencies.md)).
6. **An optional capability is a feature, and a feature needs a job.** `cuda`,
   `metal`, `io-uring` and `predict` are all off by default (ADR-0006). A
   feature that no CI job builds is a feature that is already broken — add it
   to `make gate` and to `ci.yml` in the same change that adds it.

## 10. Working on a change

- Keep changes small and vertical; keep the tree compiling between commits.
- A change that alters an architectural decision gets a new ADR under
  `docs/decisions/` rather than a silent edit of an old one.
- Update `STATUS.md` when a component's verification status changes, and
  `CHANGELOG.md` under *Unreleased* for anything user-visible.
- For spec phases, `phase N: …` is an accepted commit subject alongside the
  conventional prefixes in section 4.

## 11. Reporting security issues

See [SECURITY.md](SECURITY.md).
