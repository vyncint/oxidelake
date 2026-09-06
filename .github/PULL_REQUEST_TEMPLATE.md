## What and why

<!-- One paragraph: the problem, the change, and why this approach. -->

## How it was verified

<!-- Name the tests that prove it. For bug fixes, confirm the regression test is red on the old code. -->

- [ ] `make gate` is green locally
- [ ] New behaviour has a test at the lowest layer that can observe it
- [ ] GPU paths: state what *executed* vs what only *compiled* (honesty rule)

## Checklist

- [ ] `STATUS.md` updated if a component's verification status changed
- [ ] `CHANGELOG.md` *Unreleased* updated for user-visible changes
- [ ] New `unsafe` blocks carry a `// SAFETY:` comment
- [ ] No new `datafusion-*` sub-crate; dependency chain still coherent (`make coherence`)
- [ ] Architectural decisions recorded as an ADR in `docs/decisions/`
- [ ] No AI attribution in any commit — no `Co-Authored-By` naming an assistant,
      no "Generated with" watermark. You are the author of record; CI checks this.
