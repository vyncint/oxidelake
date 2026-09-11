//! `oxide tui` in a real pseudo-terminal — the dashboard as users get it.
//!
//! `crates/oxidelake-tui` tests the dashboard through `oxidelake-tui-demo`, a
//! binary that exists for the tests. What people install is `oxide`
//! (`cargo install oxidelake-runtime`, the release archives, Homebrew), and
//! `oxide tui` with no `--query` renders the same `demo_model()` through the
//! same `run_terminal` — but behind `clap`, `#[tokio::main]` and a
//! `tracing_subscriber` writing to stderr. `tests/cli.rs` drives the other
//! subcommands with `assert_cmd`, which captures pipes and cannot see any of
//! that: in a terminal stdout and stderr are one stream, so a log line lands
//! *on the dashboard*, and a program that never restores the terminal leaves
//! the user's shell in the alternate screen. Only a PTY test can tell.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use termlens::{Key, Screen, Terminal};

/// The first frame `oxidelake-tui` commits to for the demo model, at 80x24 —
/// the file `oxidelake-tui`'s own PTY suite snapshots, read back with the
/// parser termlens ships for it.
const DEMO_SNAPSHOT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../oxidelake-tui/tests/snapshots/tui_pty_test__the_first_frame_carries_its_colours_too.snap"
);

/// `oxide tui`, painted and settled.
fn tui(log: Option<&str>) -> termlens::Result<Terminal> {
    let builder = Terminal::builder()
        .size(80, 24)
        .env_clear()
        .timeout(Duration::from_secs(60))
        .args(["tui"]);
    let builder = match log {
        // The subscriber is installed before `clap` parses, and its writer is
        // stderr — which in a PTY is the dashboard's own stream.
        Some(level) => builder.env("RUST_LOG", level),
        None => builder,
    };
    let mut t = builder.spawn(env!("CARGO_BIN_EXE_oxide"))?;
    t.wait_frame(|s| s.contains("OxideLake") && s.contains("Plan DAG"))?;
    Ok(t)
}

/// The screen `oxidelake-tui` committed, as a `Screen`.
fn committed_frame() -> Screen {
    let raw = std::fs::read_to_string(DEMO_SNAPSHOT).unwrap_or_else(|e| {
        panic!(
            "{DEMO_SNAPSHOT}: {e} — oxidelake-tui's PTY snapshot is what this test compares against"
        )
    });
    let mut lines = raw.lines();
    assert_eq!(lines.next(), Some("---"), "insta writes a header first");
    let body: Vec<&str> = lines.skip_while(|line| *line != "---").skip(1).collect();
    Screen::parse(&format!("{}\n", body.join("\n"))).expect("the committed snapshot parses")
}

/// The shipped binary draws the dashboard the library's own suite snapshots
/// — cell for cell, colours included — and hands the terminal back.
///
/// Comparing against `oxidelake-tui`'s committed screen rather than a second
/// snapshot of its own is the whole point: the failure this guards against
/// is `oxide tui` drifting away from the frame the dashboard crate tests,
/// and two independent snapshots would drift together silently.
#[test]
fn oxide_tui_paints_the_dashboard_the_tui_crate_snapshots() -> termlens::Result<()> {
    let mut t = tui(None)?;
    let live = t.screen();
    assert!(
        live.alternate_screen(),
        "a full-screen TUI holds the alternate screen while it runs"
    );

    let committed = committed_frame();
    let diff = committed.diff(&live);
    assert!(
        diff.is_empty(),
        "`oxide tui` no longer renders the frame oxidelake-tui snapshots:\n{diff}"
    );

    // The same emulator invariant `oxidelake-tui/tests/emulation.rs` pins,
    // asserted again here because this is a different binary: `oxide` links
    // the whole engine, and anything it printed would come through the same
    // stream. `^[[59m` is ratatui's underline-colour reset; it changes no
    // cell.
    // The view compares equal to a slice only when the retained shapes match
    // *and* nothing overflowed the bound, so this is the whole record.
    assert_eq!(
        live.unsupported(),
        ["^[[59m"],
        "the shipped binary emitted a sequence termlens does not model, or \
         the record was truncated:\n{live}"
    );

    t.send(Key::Char('q'))?;
    let status = t.wait_exit()?;
    assert!(status.success(), "`oxide tui` exited with {status:?}");
    let after = t.screen();
    assert!(
        !after.alternate_screen(),
        "left the user's shell inside the alternate screen"
    );
    assert!(after.cursor().2, "left the cursor hidden");
    assert!(
        after.text().trim().is_empty() && after.scrollback_rows() == 0,
        "`oxide tui` left output behind on the restored terminal:\n{after}"
    );
    Ok(())
}

/// The log stream shares the terminal with the dashboard, and must stay off
/// the grid.
///
/// `main` installs a `tracing_subscriber` writing to stderr before anything
/// else happens, and in a PTY stderr *is* the screen: one `info!` on the
/// startup path would paint over a panel, scroll the frame, or leave a line
/// behind after the restore. The demo path opens no session, so at `info`
/// there is nothing to say — and this is how that stays true. If it fails,
/// the fix is to route the subscriber somewhere other than the terminal the
/// dashboard owns, not to lower the level.
#[test]
fn logging_at_info_puts_nothing_on_the_dashboard() -> termlens::Result<()> {
    let quiet = tui(None)?.screen();
    let mut t = tui(Some("info"))?;
    let noisy = t.screen();

    let diff = quiet.diff(&noisy);
    assert!(
        diff.is_empty(),
        "RUST_LOG=info changed what `oxide tui` draws — a log line is on the \
         dashboard:\n{diff}"
    );
    assert_eq!(
        noisy.scrollback_rows(),
        0,
        "a log line scrolled the frame:\n{noisy}"
    );

    t.send(Key::Char('q'))?;
    assert!(t.wait_exit()?.success());
    let after = t.screen();
    assert!(
        after.text().trim().is_empty(),
        "a log line survived the restore:\n{after}"
    );
    Ok(())
}
