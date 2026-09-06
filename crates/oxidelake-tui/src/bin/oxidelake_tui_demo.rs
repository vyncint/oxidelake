//! `oxidelake-tui-demo` — renders the dashboard from a fixed synthetic telemetry
//! snapshot: no clock, no animation, repaints only on input or resize. Used by
//! the headless PTY tests (termlens) and as a visual smoke test.

fn main() -> std::io::Result<()> {
    oxidelake_tui::run_terminal(oxidelake_tui::demo_model(), None, |_| {})
}
