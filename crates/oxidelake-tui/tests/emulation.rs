//! What the emulator can and cannot see of the dashboard — the assertion the
//! rest of the PTY suite rests on.
//!
//! Every screen assertion in `tui_pty_test.rs` reads a grid a VT emulator
//! built out of ratatui's bytes. If the application emits a sequence the
//! emulator does not implement, that grid is quietly wrong and the three
//! committed snapshots, the `contains("1000000")` waits and the colour rules
//! are all being made against a plausible-looking fiction. termlens 0.10 made
//! that checkable: `Screen::unsupported` lists what was dropped.
//!
//! These are whole-suite invariants rather than feature tests. They are
//! cheap, and when one breaks the right response is to distrust the other
//! files until it is understood.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use termlens::{Key, Screen, Terminal};

/// The only sequence this dashboard emits that termlens does not model.
///
/// `SGR 59` is "underline colour: default". `ratatui-crossterm` queues
/// `SetUnderlineColor(Reset)` at the tail of every `draw` when its
/// `underline-color` feature is on, whether or not anything is underlined;
/// termlens carries no underline colour, so it records the sequence and
/// moves on. It changes no cell, which is the whole reason this list can be
/// pinned exactly: anything joining it is a sequence that *might* change a
/// cell and has to be read before the suite is trusted again.
///
/// The pin needs no exception. It used to carry one: termlens reported
/// blink and strikethrough as unsupported although its attribute shadow
/// implements them (termlens#320), so those four sequences were false
/// positives wherever they appeared. Fixed in termlens 0.10.2 — an entry
/// here is a real gap now, whatever a dashboard's modifiers are.
const EXPECTED_UNSUPPORTED: [&str; 1] = ["^[[59m"];

fn spawn(cols: u16, rows: u16) -> termlens::Result<Terminal> {
    termlens::bin!(
        "oxidelake-tui-demo",
        size(cols, rows),
        timeout(std::time::Duration::from_secs(20))
    )
}

fn check(label: &str, screen: &Screen) {
    // One comparison for both halves of the record: termlens 0.11's
    // `Unsupported` view is equal to a slice only when the retained shapes
    // match *and* nothing overflowed the bound, so a truncated record fails
    // here rather than passing as a shorter list.
    assert_eq!(
        screen.unsupported(),
        EXPECTED_UNSUPPORTED,
        "{label}: the dashboard emitted a sequence termlens does not model, \
         or the record was truncated. Until it is understood, every screen \
         assertion in this crate is being made against a grid that may be \
         wrong.\n{screen}"
    );
}

/// The invariant, at every instant the suite actually asserts against: the
/// first paint, a navigation repaint, a SIGWINCH repaint at another size,
/// and the teardown that hands the terminal back.
///
/// The resize is the one worth checking hardest — it is the only path that
/// re-enters the alternate screen machinery, and a dropped sequence there
/// would land in the 120x40 snapshot.
#[test]
fn the_emulator_drops_nothing_that_could_change_a_cell() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    check(
        "first paint",
        &t.wait_frame(|s| s.contains("OxideLake") && s.contains("Describe"))?,
    );

    t.send(Key::Down)?;
    check(
        "after ↓",
        &t.wait_frame(|s| s.contains("1000000") && s.contains("32.0 MiB"))?,
    );

    t.resize(120, 40)?;
    check(
        "after SIGWINCH",
        &t.wait_frame(|s| s.size() == (120, 40) && s.contains("Describe"))?,
    );

    t.send(Key::Char('q'))?;
    assert!(t.wait_exit()?.success());
    check("after quit", &t.screen());
    Ok(())
}

/// Four smaller invariants that would each make the grid a lie, and that
/// nothing else in the suite would notice.
#[test]
fn the_dashboard_leaves_the_terminal_modes_alone() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    let screen = t.wait_frame(|s| s.contains("OxideLake") && s.contains("Describe"))?;

    // Insert mode pushes the rest of a row right. An application that left
    // it on would draw a correct-looking dashboard with every row shifted,
    // and the box drawing would still line up.
    assert!(!screen.insert_mode(), "the dashboard never sets IRM");
    // A visual bell is a flash the grid cannot show.
    assert_eq!(screen.visual_bells(), 0, "no ESC g");
    assert_eq!(screen.bells(), 0, "and no audible bell either");
    // Nothing wraps: ratatui lays out to the width it was given, so a
    // wrapped row means a panel overflowed and its text is silently on two
    // rows — which reads as a correct dashboard one row short.
    assert!(
        !(0..screen.rows()).any(|row| screen.row_wrapped(row)),
        "a wrapped row means the layout overflowed:\n{screen}"
    );
    assert_eq!(
        screen.logical_text(),
        screen.text(),
        "with no wrapped row the logical text is the grid text"
    );
    // The dashboard enables no mouse tracking, so `click`/`drag`/`scroll`
    // have nothing to reach here — asserted rather than left as folklore,
    // because "the mouse does not work" is otherwise indistinguishable from
    // "nobody wrote the test".
    assert!(
        screen.mouse_modes().is_empty(),
        "no mouse mode is enabled: {:?}",
        screen.mouse_modes()
    );

    // And the shell gets its terminal back with nothing left in it: the app
    // holds the alternate screen for its whole life, so a line printed
    // outside it (a panic message, a stray println) would survive the
    // restore and show up here.
    t.send(Key::Char('q'))?;
    assert!(t.wait_exit()?.success());
    let after = t.screen();
    assert!(!after.alternate_screen());
    assert_eq!(after.scrollback_rows(), 0, "nothing scrolled into history");
    assert!(
        after.text().trim().is_empty(),
        "the restored screen is not blank:\n{after}"
    );
    Ok(())
}

/// A dashboard screen has to survive being saved and read back, because that
/// is what a bug report and a CI artefact are: the box drawing, the block
/// glyphs and every colour come back, or the format is not carrying what
/// this application draws.
#[test]
fn a_dashboard_screen_survives_the_snapshot_format_and_json() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    let screen = t.wait_frame(|s| s.contains("OxideLake") && s.contains("[CUDA]"))?;

    // The text format: a committed `.snap`, the block a wait error prints,
    // and what `termlens diff` reads back (see `termlens_cli.rs`).
    let saved = screen.with_styles().to_string();
    let parsed = Screen::parse(&saved)?;
    assert!(screen.diff(&parsed).is_empty(), "{}", screen.diff(&parsed));
    assert_eq!(parsed.with_styles().to_string(), saved, "byte for byte");

    // And JSON, which is what `TERMLENS_ARTIFACT_DIR` writes in CI — the
    // file the report action renders when a wait fails on a runner.
    let json = serde_json::to_string(&screen).expect("a Screen serializes");
    let back: Screen = serde_json::from_str(&json).expect("and comes back");
    assert!(screen.diff(&back).is_empty(), "{}", screen.diff(&back));

    // The colours specifically: the panels encode the backend and the focus
    // in colour alone, so a round trip that dropped styles would still pass
    // a text comparison.
    let (row, col) = screen.find("[CUDA]").expect("the plan tags its operators");
    let cell = screen.cell(row, col).unwrap();
    assert_ne!(cell.style().fg, termlens::Color::Default, "measured green");
    for (label, other) in [("the text format", &parsed), ("JSON", &back)] {
        assert_eq!(
            other.cell(row, col).unwrap().style(),
            cell.style(),
            "{label} lost the colour of the [CUDA] tag at ({row},{col})"
        );
    }
    t.send(Key::Char('q'))?;
    assert!(t.wait_exit()?.success());
    Ok(())
}
