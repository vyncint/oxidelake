# ADR-0011: TUI testing — `TestBackend` in-process plus `termlens` against a deterministic demo binary

- Status: accepted
- Date: 2026-08-25

## Context

The draft asked for both `TestBackend` and `termlens` tests without saying how they relate. `termlens` (0.6.1) spawns a real binary in a real PTY, renders through a VT emulator and snapshots the screen with `insta`; `TestBackend` renders in-process without a PTY. A PTY test needs a binary in the same crate (`env!("CARGO_BIN_EXE_…")`) and deterministic output.

## Decision

The TUI is a state machine with a pure `render(state, frame)` and an `on_event` handler. `tests/tui_render_test.rs` snapshots `TestBackend` buffers at 80×24 and 120×40 and asserts state transitions. `tests/tui_pty_test.rs` uses termlens to drive `oxidelake-tui-demo`, a `[[bin]]` in `oxidelake-tui` that renders a fixed synthetic telemetry snapshot (no clock, no animation). Repaints are bracketed in DEC 2026 synchronized updates so `wait_frame` observes complete frames; waits use `wait_until` / `wait_frame`, never `sleep`. The termlens API is read from docs.rs before use.

## Consequences

Both layers run headless under `cargo test`. The demo binary doubles as a visual smoke test. Snapshots are committed and reviewed like code.
