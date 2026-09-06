# Releasing OxideLake

The pipeline does the work; this is the human side of it. Nine crates publish
together from one workspace version, in dependency order, through crates.io
Trusted Publishing.

## Prerequisites

- **crates.io Trusted Publishing** linked to this repository for every crate
  (Settings → Trusted Publishing on each crate, pointing at `release.yml` and
  the `release` environment). Steady state stores no registry credential.

  **The first publish of each crate is the exception, and cannot be
  otherwise:** Trusted Publishing is configured *on a crate*, so it cannot be
  set up for one that has never been published. For `v0.1.0`, set a
  `CARGO_REGISTRY_TOKEN` repository secret holding a maintainer token scoped
  to `publish-new`; `release.yml` uses it only for crates that do not yet
  exist and OIDC for the rest. Delete the secret and revoke the token as soon
  as all nine crates exist, then configure their trusted publishers. On that
  first run the OIDC exchange itself fails with `400 No Trusted Publishing
  config found for repository` — expected, and non-fatal: the step is
  `continue-on-error` and the loop uses the registry token instead.
- A **`release` GitHub environment** whose deployment branches are restricted
  to `v*` tags, so an OIDC publish token can never be minted from a branch.
- **`v*` tags protected by a ruleset**, so only a repository admin can create
  one.

## Cutting vX.Y.Z

```sh
# 0. Green main, and no flakes: run the stress workflow and wait for it.
gh workflow run stress.yml && gh run watch

# 1. Bump `version` in the workspace [workspace.package] AND every internal
#    `version = "…"` in [workspace.dependencies], then refresh the lockfile.
cargo check --workspace         # rewrites the nine Cargo.lock entries

# 2. Move the CHANGELOG section: [Unreleased] -> [X.Y.Z] - YYYY-MM-DD, leaving an
#    empty [Unreleased] above it — and repoint the link definitions at the foot,
#    or the heading renders as literal `[X.Y.Z]`:
#      [Unreleased]: …/compare/vX.Y.Z...HEAD
#      [X.Y.Z]:      …/compare/vPREV...vX.Y.Z
.github/scripts/extract-changelog.sh X.Y.Z   # must print the section, non-empty

# 3. Land it through a pull request (main takes nothing else).
git switch -c release/vX.Y.Z && git commit -sam "release: vX.Y.Z" && gh pr create

# 4. Tag the squash-merged commit on main, tag only.
git switch main && git pull
git tag vX.Y.Z && git push origin refs/tags/vX.Y.Z
```

Never name a branch after the tag it will produce: `git push origin vX.Y.Z`
becomes ambiguous and is refused.

The tag triggers `release.yml`, which:

1. fails unless the tag matches the workspace version **and** `CHANGELOG.md`
   has a non-empty section for it — both checked before anything irreversible;
2. re-runs the full CI gate (`workflow_call` into `ci.yml`);
3. runs `cargo-semver-checks` against the last published release — skipped on
   the first release, when there is no baseline;
4. publishes the nine crates in dependency order (`core → memory → device →
   compute → storage → planner → tui → runtime → api`), skipping any already
   on the registry so a rate-limited run is resumable via `workflow_dispatch`;
5. creates the GitHub Release with notes extracted from `CHANGELOG.md`;
6. builds static binaries for four targets and attaches them (`binaries.yml`),
   and installs the published crate on a machine that has never seen this
   repository (`install.yml`).

## What a version number means here

Nine crates version together; a bump anywhere is a bump everywhere.

- **Breaking** (minor, pre-1.0): a removed or renamed public item in any crate;
  a changed CLI flag or output format; a change to what a plan executes that a
  user would have to relearn.
- **Not breaking**: new operators, new features (they are all off by default),
  a backend newly supported, a kernel that computes the same thing faster.
- **MSRV bumps are minor**, never patch, and the lockfile is the floor.

## If something fails mid-release

- **Before publish**: fix, delete the tag (`git push --delete origin vX.Y.Z`),
  re-tag.
- **After publish**: crates.io releases are permanent. Yank the affected crates
  (`cargo yank --version X.Y.Z -p <crate>`) and ship a patch. Do not delete the
  tag — the published crates point at it.
- **Partway through the nine**: nothing to fix. Re-run with
  `gh workflow run release.yml -f tag=vX.Y.Z`; published crates are skipped.
