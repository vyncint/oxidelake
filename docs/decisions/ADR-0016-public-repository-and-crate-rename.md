# ADR-0016: Public repository from one commit; crates renamed `oxidelake-*`; kernels inside the device crate

- Status: accepted
- Date: 2026-09-06

## Context

OxideLake was developed privately as `vyncint/oxidelake` from 2026-08-25. Going
public with the intent to publish to crates.io surfaced three facts about the
tree as it stood:

1. **Two crate names were already taken.** `oxide-core` (0.5.0) and `oxide-api`
   (0.1.0-rc.41) belong to unrelated projects on crates.io. Nothing could have
   published under the `oxide-*` family.
2. **The published device crate would not have compiled.** GPU kernel sources
   lived at the repository root and were reached with
   `include_str!("../../../../kernels/cuda/…")`. `cargo publish` packages only
   the crate directory, so the `.cu`/`.metal` files would have been absent from
   the archive. No CI could see this, because CI has the whole repository — the
   failure exists only for a user who has just the crate.
3. **The history was written for a private workbench.** Fifteen commits with no
   DCO sign-off and four with AI co-author trailers, an implementation prompt
   as the authoritative spec containing a local filesystem path, and no branch
   protection. Each was fine privately and none is what a public project should
   ship.

## Decision

- **A new public repository, starting from one commit.** The private
  repository is renamed `vyncint/oxidelake-old` and kept; its history is not
  migrated. One commit carries the whole tree, DCO-signed, with no AI
  attribution, and the commit policy applies from it forward.
- **Crates are `oxidelake-*`.** All eleven candidate names were free. Binary
  names are unchanged (`oxide`, `oxide-scheduler`, `oxide-worker`): binaries are
  not namespaced on crates.io and every document and test names them.
- **Kernels live in `crates/oxidelake-device/kernels/`.** `cargo package --list`
  is the proof, and `install.yml` — installing the published crate on a
  machine that has never seen the repository — is the standing guard.
- **The specification is `docs/SPEC.md`.** Same normative content and section
  numbers (thirty files cite them), without the prompt framing or the path.
- **The repository takes the vyncint contributor pattern in full**: the
  generated policy scripts and `commit-policy` workflow, a one-job-per-concern
  CI aggregated by `required-green`, every action pinned to a commit SHA,
  `zizmor` at pedantic, no build cache, a tag-guarded and changelog-gated
  release pipeline publishing nine crates in dependency order through Trusted
  Publishing, static binaries, and an install-from-registry check.

## Consequences

- `cargo install oxidelake-runtime` and `cargo add oxidelake-api` become
  possible; the `oxide-*` names never were.
- Anyone reading the public history sees the project as it is, not as it was
  built; the private repository remains the record of how it was built.
- The first publish of each crate cannot use Trusted Publishing (it is
  configured per crate, after the crate exists) and uses a temporary token
  that is then revoked — `docs/RELEASING.md`.
- Windows binaries are not built (out of scope, `docs/roadmap.md`), and there
  is no Homebrew tap.
