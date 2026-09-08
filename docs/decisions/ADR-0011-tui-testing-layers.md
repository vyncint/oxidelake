# ADR-0011: TUI testing — `TestBackend` in-process plus `termlens` against a deterministic demo binary

- Status: accepted
- Date: 2026-08-25

## Context

The draft asked for both `TestBackend` and `termlens` tests without saying how they relate. `termlens` (0.6.1) spawns a real binary in a real PTY, renders through a VT emulator and snapshots the screen with `insta`; `TestBackend` renders in-process without a PTY. A PTY test needs a binary in the same crate (`env!("CARGO_BIN_EXE_…")`) and deterministic output.

## Decision

The TUI is a state machine with a pure `render(state, frame)` and an `on_event` handler. `tests/tui_render_test.rs` snapshots `TestBackend` buffers at 80×24 and 120×40 and asserts state transitions. `tests/tui_pty_test.rs` uses termlens to drive `oxidelake-tui-demo`, a `[[bin]]` in `oxidelake-tui` that renders a fixed synthetic telemetry snapshot (no clock, no animation). Repaints are bracketed in DEC 2026 synchronized updates so `wait_frame` observes complete frames; waits use `wait_until` / `wait_frame`, never `sleep`. The termlens API is read from docs.rs before use.

## Consequences

Both layers run headless under `cargo test`. The demo binary doubles as a visual smoke test. Snapshots are committed and reviewed like code.

## Update — 2026-09-08 (termlens 0.10)

The two layers stand; the dependency moved 0.9 → 0.10.1 and the PTY layer grew a third and fourth file (this ADR's own text above still said 0.6.1, which is what it was when the decision was taken — the manifest had since moved to 0.9 without the record following), because 0.10 made claims checkable that were previously only assumed:

- `tests/emulation.rs` pins `Screen::unsupported()` — the sequences the emulator did *not* implement. Every other assertion in this crate reads a grid the emulator built, so a dropped sequence makes all of them plausible and wrong. Measured here: exactly `["^[[59m"]`, ratatui's underline-colour reset, which changes no cell.
- `tests/tui_pty_test.rs` keeps its three text snapshots byte-identical (`styles = false`; 0.10 records styles by default) and adds one styled snapshot plus direct cell assertions for the two things this dashboard says in colour alone: the focused panel's border and each operator's backend tag. `TestBackend::to_string()` cannot see either.
- `crates/oxidelake-runtime/tests/oxide_tui_pty.rs` extends the layer to the binary users install: `oxide tui` renders the same frame `oxidelake-tui` snapshots, and the `tracing` subscriber writing to stderr — the same stream as the dashboard in a PTY — puts nothing on the grid.
- `tests/termlens_cli.rs` reads the committed `.snap` files back with `termlens-cli`. It is `#[ignore]`d: these crates are published, and `cargo test` must not install a tool behind a contributor's back. CI runs it (`make test-termlens-cli`).

The vendored agent skill is checked against the dependency by `make skill-version`, because it had already drifted two releases behind.
