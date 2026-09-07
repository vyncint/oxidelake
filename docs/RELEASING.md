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
- **`HOMEBREW_TAP_TOKEN`** repository secret (optional): a fine-grained PAT with
  `contents: write` on `vyncint/homebrew-tap` and nothing else. `binaries.yml`
  regenerates `Formula/oxidelake.rb` in the tap from the uploaded archives'
  checksums; without the secret the job says so and the tap keeps the previous
  version. To refresh the formula for a tag whose release already ran (or ran
  from an older workflow): `gh workflow run binaries.yml --ref main -f tag=vX.Y.Z`
  — targets whose archive is already on the release are skipped without
  building, only the missing archives and the formula move.
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
2. re-runs the full CI gate without build caches or documentation skips
   (`workflow_call` into `ci.yml`); ordinary main CI cancellation does not
   cancel this release gate;
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
- **Partway through the nine**: nothing to fix. Re-run the failed jobs of the
  same run (`gh run rerun <run-id> --failed`) or dispatch the workflow **at the
  tag**: `gh workflow run release.yml --ref vX.Y.Z -f tag=vX.Y.Z`. Published
  crates are skipped. The `--ref` matters: the `release` environment deploys
  from `v*` tags alone, so a dispatch from `main` is rejected by environment
  protection before the publish job starts (`Branch "main" is not allowed to
  deploy to release`) — and that is the point, not a bug.
- **The workflow itself is what failed**: a dispatch runs the workflow file
  *at the ref it is dispatched from*, so a fix merged to `main` is not picked
  up by re-running the tag. Before anything is published, fix on `main`, then
  move the tag onto the fixed commit (`git tag -f vX.Y.Z origin/main`, push
  the deletion, push the tag). After anything is published the tag stays;
  finish the remaining crates from a fixed *new* tag only if their versions
  match, otherwise ship a patch release.
