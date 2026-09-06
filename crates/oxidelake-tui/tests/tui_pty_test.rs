//! End-to-end PTY tests with `termlens`: the real `oxidelake-tui-demo` binary is
//! spawned in a pseudo-terminal, its output is rendered by a VT emulator, and
//! the screen is snapshotted and driven with key presses and a resize. No
//! physical TTY is needed.
//!
//! **This app brackets its repaints in DEC 2026 synchronized updates**, which
//! most ratatui applications do not — measured here as `screen.repaints() == 1`
//! after the first paint, where stock ratatui with crossterm stays at 0. That
//! is what makes `wait_frame` usable: it returns only on a *completed* frame,
//! so a snapshot can never catch a half-drawn screen. Everywhere else in this
//! ecosystem the answer is `snapshot_after`; here the stronger tool is
//! available and worth using.
//!
//! Never a sleep. Every test also asserts the terminal was given back — an
//! application that leaves the user in the alternate screen with a hidden
//! cursor is a bug only a real PTY can see.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use termlens::{Key, Screen, Terminal};

/// The demo, painted and settled.
///
/// `bin!` supplies size/`env_clear`/timeout/the binary path and makes a
/// misspelled binary name a compile error rather than a spawn failure at run
/// time. The 20-second timeout is kept: this binary builds a plan first.
fn spawn(cols: u16, rows: u16) -> termlens::Result<Terminal> {
    termlens::bin!(
        "oxidelake-tui-demo",
        size(cols, rows),
        timeout(std::time::Duration::from_secs(20))
    )
}

/// Quit, and check the borrow was returned.
///
/// Every test goes through this, so every test is also a teardown test. The
/// values were measured before this was written — `alternate_screen()` false
/// and the cursor visible after `q` — so this pins behaviour that is already
/// correct rather than describing a fix.
fn quit(t: &mut Terminal) -> termlens::Result<()> {
    t.send(Key::Char('q'))?;
    let status = t.wait_exit()?;
    assert!(status.success(), "exited with {status:?}");
    let after = t.screen();
    assert!(
        !after.alternate_screen(),
        "left the shell inside the alternate screen"
    );
    assert!(after.cursor().2, "left the cursor hidden");
    Ok(())
}

#[test]
fn demo_renders_navigates_resizes_and_quits() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    // One predicate per instant: everything asserted about this moment is in
    // the closure, and `wait_frame` hands back the completed frame that made
    // it true — so there is no second read to race with a repaint.
    let first: Screen = t.wait_frame(|s| {
        s.contains("OxideLake") && s.contains("Plan DAG") && s.contains("Describe")
    })?;
    assert!(
        first.alternate_screen(),
        "a full-screen TUI should hold the alternate screen while it runs"
    );
    termlens::assert_screen_snapshot!(first);

    // ↓ selects GpuFilterExec: the inspector shows its 1,000,000 input rows.
    t.send(Key::Down)?;
    let selected = t.wait_frame(|s| s.contains("1000000") && s.contains("32.0 MiB"))?;
    termlens::assert_screen_snapshot!(selected);

    // Tab moves the focus highlight; ↓ selects DataSourceExec (8.0 MiB).
    t.send(Key::Tab)?;
    t.send(Key::Down)?;
    t.wait_frame(|s| s.contains("8.0 MiB"))?;

    // Resize is delivered through SIGWINCH; the app repaints at the new size.
    t.resize(120, 40)?;
    let resized =
        t.wait_frame(|s| s.size() == (120, 40) && s.contains("Describe") && s.contains("8.0 MiB"))?;
    termlens::assert_screen_snapshot!(resized);

    quit(&mut t)
}

#[test]
fn quits_immediately_from_the_first_screen() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    t.wait_frame(|s| s.contains("OxideLake"))?;
    quit(&mut t)
}

/// The claim the module docs rest on, asserted rather than assumed.
///
/// If this app ever stops emitting synchronized updates, `wait_frame` above
/// starts timing out with a message saying exactly that — but only for
/// whoever runs the suite next. This fails immediately and says why.
#[test]
fn the_app_really_emits_synchronized_frames() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    let s = t.wait_frame(|s| s.contains("OxideLake"))?;
    assert!(
        s.repaints() >= 1,
        "no DEC 2026 synchronized update was observed, so `wait_frame` is the \
         wrong tool for this app — switch these tests to `snapshot_after`"
    );
    quit(&mut t)
}
