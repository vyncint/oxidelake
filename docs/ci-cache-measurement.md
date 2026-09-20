# CI build cache: the measurement, and what it decided

ADR-0017 added dependency-only `Swatinem/rust-cache` steps to three jobs as
a deliberate exception to "no build cache" (SECURITY.md, ADR-0016). The
2026-09-07 CI audit that motivated it recorded the pre-cache baseline and
asked that cold and warm runs be compared before any saving was claimed.
Issue #39 is that the comparison had not been made, so two documents
disagreed with the workflows and nothing said which was right.

## What was measured

Wall-clock duration of successful `ci.yml` runs on `main`, from the GitHub
API (`createdAt` → `updatedAt`, so queue time is included — that is what a
contributor waits).

| Run | Commit | Duration | Cache state |
|---|---|---|---|
| [35512130222](https://github.com/vyncint/oxidelake/actions/runs/35512130222) | `c3e292b` | **7 min** | warm |
| [35447806485](https://github.com/vyncint/oxidelake/actions/runs/35447806485) | `2b8dd2f` | **19 min** | cold — `Cargo.lock` moved (termlens 0.11.2) |
| [34789572325](https://github.com/vyncint/oxidelake/actions/runs/34789572325) | `6f06490` | 7 min | warm |
| [34767674318](https://github.com/vyncint/oxidelake/actions/runs/34767674318) | `28c42fc` | 7 min | warm |
| [34551775543](https://github.com/vyncint/oxidelake/actions/runs/34551775543) | `2b12da6` | 7 min | warm |
| [34279749543](https://github.com/vyncint/oxidelake/actions/runs/34279749543) | `5179c21` | **29 min** | cold |
| [34119478858](https://github.com/vyncint/oxidelake/actions/runs/34119478858) | `9e88562` | 7 min | warm |

**Pre-cache baseline: median 59 min** (2026-09-07 CI audit).

- Warm (no dependency change): **7 min**, five of five runs.
- Cold (dependency graph moved, cache miss): **19–29 min**.
- A cold run is still under half the pre-cache median, because the cache is
  only one of the things ADR-0017 changed.

## The decision

**Option (a): keep the caches, and say so where the posture is written
down.** The exception has paid for itself by roughly 8× on the common path
and about 2–3× on the worst one, which is the number ADR-0017 was asked
for and did not have.

What the caches are, precisely, and why that is the part the security
posture cares about:

- **Dependencies only.** `cache-bin: false` and
  `cache-workspace-crates: false`, so no workspace crate and no tool binary
  is ever restored. What comes back is third-party dependency builds keyed
  by lockfile, which is what `Cargo.lock` already pins by hash.
- **Never for a release.** Every step is guarded by
  `github.workflow == 'CI' && !inputs.clean`, so a `workflow_call` from
  `release.yml` — and any run dispatched with `clean` — restores nothing and
  builds from scratch. A published artifact is never made from a restored
  cache.

"No build cache that could serve a stale artifact" was true of the artifact
and wrong about the mechanism. SECURITY.md and ADR-0016 now say
dependency-only, never restored for a release, and point here.

## When to revisit

If a cold run's margin over the baseline closes, or if the cache is ever
extended beyond dependencies, this measurement is void and ADR-0017's
exception has to argue for itself again. Re-run the three-way comparison —
cold, warm, one-line source change — and replace the table above.
