# ADR-0017: Reuse CI dependencies while preserving executable gates

- Status: accepted
- Date: 2026-09-07

## Context

The no-cache policy assumed short builds. In GitHub Actions run
[34074441576](https://github.com/vyncint/oxidelake/actions/runs/34074441576),
macOS took 54m12s, including 53m13s in Cargo build/check phases. Arrow and
DataFusion compiled in three separate test steps. The conformance step built
for 14m28s and its device test ran for approximately one second. Linux spent
47m05s on sequential feature builds. Main workflow serialization and macOS
runner queues added further waiting. These are baseline measurements, not
claims about the new implementation's speed.

## Decision

- Cache only external dependencies in ordinary CI test jobs, separated by
  OS, architecture, toolchain, manifests/lockfile, compiler settings and lane.
  Use a full-SHA-pinned cache action and GitHub's branch-scoped caches. Always
  invoke Cargo and run the tests after restoration. Begin with the expensive
  test jobs to limit cache size; keep workspace crates and installed tools out.
- Release calls always run the full gate without build-cache restore/save.
  Manual CI can request the same clean build. This retains an independent
  check of dependency rebuilds without making every PR pay that cost.
- Run Linux default, io-uring and predict tests independently. Keep both
  pure-CPU defaults and standalone feature coverage. Keep CUDA checks and
  the no-second-CUDA/dependency-coherence gates.
- Select identical packages and features for the normal and ignored Metal
  test passes, including TUI tests. Probe the Metal API before the ignored
  pass, then require actual device execution through the existing conformance
  assertions. Shared build arguments live in the Makefile.
- Cancel superseded main/PR CI, never a reusable release gate. Build release
  stress targets once per OS, then run the existing thread counts and weights.
- Classify only an explicit documentation allowlist as safe to skip Rust
  jobs. Unknown paths, missing bases, empty diffs, release calls and manual
  runs take the full gate. Always run script/security/metadata checks.
  `required-green` rejects failures, cancellations, missing results and skips
  outside that policy. Regression tests exercise these failure paths.
- Publish Cargo timing artifacts so queueing, compilation, test execution
  and cache-transfer costs can be measured separately.

## Consequences

Dependency reuse changes the earlier no-cache workflow policy while keeping
the acceptance and honesty rules of ADR-0012. Linux parallelism can increase
cold runner minutes even as it shortens elapsed time. Cache storage and transfer
costs must be observed before expanding caching to every job. Production
optimization profiles, registry publication order/retries and required GPU
coverage are unchanged. The TUI's dependency on DataFusion remains an
architectural follow-up rather than part of this CI change.
