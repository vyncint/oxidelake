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
//!
//! **Text and colour are snapshotted separately, on purpose.** termlens 0.10
//! records styles by default; the three flow snapshots below stay text-only
//! (`styles = false`) so a colour change shows up in *one* place —
//! `the_first_frame_carries_its_colours_too` — instead of rewriting three
//! large files at once. The claims colour actually encodes (the focused
//! panel, the backend of each operator) are asserted directly against cells
//! further down, where a failure names the rule that broke rather than
//! handing over a grid to read.

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
    termlens::assert_screen_snapshot!(&first, styles = false);

    // ↓ selects GpuFilterExec: the inspector shows its 1,000,000 input rows.
    t.send(Key::Down)?;
    let selected = t.wait_frame(|s| s.contains("1000000") && s.contains("32.0 MiB"))?;
    termlens::assert_screen_snapshot!(&selected, styles = false);

    // Tab moves the focus highlight; ↓ selects DataSourceExec (8.0 MiB).
    t.send(Key::Tab)?;
    t.send(Key::Down)?;
    t.wait_frame(|s| s.contains("8.0 MiB"))?;

    // Resize is delivered through SIGWINCH; the app repaints at the new size.
    t.resize(120, 40)?;
    let resized =
        t.wait_frame(|s| s.size() == (120, 40) && s.contains("Describe") && s.contains("8.0 MiB"))?;
    termlens::assert_screen_snapshot!(&resized, styles = false);

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

/// The same first frame, with every style it is drawn in.
///
/// The three snapshots above are text; this is the only place in the
/// repository where the dashboard's colours are recorded wholesale —
/// `TestBackend::to_string()` in `tui_render_test.rs` cannot see them at
/// all. Its own name, so accepting a colour change never touches the text
/// snapshots and vice versa.
#[test]
fn the_first_frame_carries_its_colours_too() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    let first = t.wait_frame(|s| {
        s.contains("OxideLake") && s.contains("Plan DAG") && s.contains("Describe")
    })?;
    termlens::assert_screen_snapshot!(&first);
    quit(&mut t)
}

/// Yellow index in the 256-colour palette: `Color::Yellow` through crossterm.
const YELLOW: termlens::Color = termlens::Color::Indexed(3);
/// Green: what `backend_tag` paints a CUDA operator (`render.rs`).
const GREEN: termlens::Color = termlens::Color::Indexed(2);
/// Blue: the same function's CPU tag.
const BLUE: termlens::Color = termlens::Color::Indexed(4);

/// The title of `panel` as it is drawn on this screen.
fn border_title(screen: &Screen, panel: &str) -> termlens::Style {
    let (row, col) = screen
        .find(panel)
        .unwrap_or_else(|| panic!("no `{panel}` border on:\n{screen}"));
    *screen
        .cell(row, col)
        .expect("the cell find just reported")
        .style()
}

/// Every occurrence of `tag`, and the style of each of its characters.
fn tag_styles(screen: &Screen, tag: &str) -> Vec<(u16, u16, termlens::Style)> {
    screen
        .find_all(tag)
        .into_iter()
        .flat_map(|(row, col)| {
            (0..tag.chars().count() as u16).map(move |offset| (row, col + offset))
        })
        .map(|(row, col)| {
            (
                row,
                col,
                *screen
                    .cell(row, col)
                    .unwrap_or_else(|| panic!("({row},{col}) is off the grid"))
                    .style(),
            )
        })
        .collect()
}

/// The placement rule, as a rule rather than as a picture.
///
/// `render.rs` paints `[CUDA]` green and `[CPU]` blue wherever an operator
/// is shown — twice in the Plan DAG, once again in the Inspector — and that
/// is the one thing on this dashboard a user reads at a glance. A snapshot
/// would pin those three positions; this pins the *rule*, so moving the
/// panels around keeps it true and mixing the two colours up does not.
///
/// `find_all` is the reason it can be written at all: 0.9 could only find
/// the first occurrence, and the Inspector's copy is the third.
#[test]
fn every_backend_tag_is_drawn_in_the_colour_of_its_backend() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    let first = t.wait_frame(|s| s.contains("[CUDA]") && s.contains("[CPU]"))?;

    // Two GPU operators in the plan and the selected one in the Inspector.
    let cuda = tag_styles(&first, "[CUDA]");
    assert_eq!(
        first.find_all("[CUDA]").len(),
        3,
        "the demo plan has two CUDA operators, and the Inspector repeats the \
         selected one:\n{first}"
    );
    for (row, col, style) in &cuda {
        assert_eq!(style.fg, GREEN, "[CUDA] at ({row},{col}) is not green");
        assert!(style.bold, "[CUDA] at ({row},{col}) is not bold");
    }
    for (row, col, style) in tag_styles(&first, "[CPU]") {
        assert_eq!(style.fg, BLUE, "[CPU] at ({row},{col}) is not blue");
        assert!(style.bold, "[CPU] at ({row},{col}) is not bold");
    }
    assert!(
        first.find_all("[Metal]").is_empty(),
        "the demo model has no Metal operator, so nothing should be magenta"
    );

    // And the rule survives the layout moving: ↓ re-renders the Inspector
    // around a different operator, so the third tag lands elsewhere.
    t.send(Key::Down)?;
    let selected = t.wait_frame(|s| s.contains("1000000") && s.contains("32.0 MiB"))?;
    assert_ne!(
        selected.find_all("[CUDA]"),
        first.find_all("[CUDA]"),
        "↓ should have moved the Inspector's copy of the tag"
    );
    for (row, col, style) in tag_styles(&selected, "[CUDA]") {
        assert_eq!(style.fg, GREEN, "[CUDA] at ({row},{col}) lost its colour");
    }
    for (row, col, style) in tag_styles(&selected, "[CPU]") {
        assert_eq!(style.fg, BLUE, "[CPU] at ({row},{col}) lost its colour");
    }
    quit(&mut t)
}

/// Tab moves the focus, and the focus is *only* a colour.
///
/// Nothing else in this repository can state this: the border characters are
/// identical either way, so the text snapshots and every `TestBackend`
/// snapshot are byte-for-byte the same before and after. `diff` sees it
/// because it compares cells, styles included — so the assertion is that the
/// text did not move and the picture still changed.
#[test]
fn tab_moves_the_focus_highlight_and_changes_no_text() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    let before = t.wait_frame(|s| s.contains("Plan DAG") && s.contains("Inspector"))?;
    assert_eq!(
        border_title(&before, "Plan DAG").fg,
        YELLOW,
        "the Plan DAG panel starts focused"
    );
    assert_eq!(
        border_title(&before, "Inspector").fg,
        termlens::Color::Default
    );

    t.send(Key::Tab)?;
    let after = t.wait_frame(|s| border_title(s, "Inspector").fg == YELLOW)?;
    assert_eq!(
        border_title(&after, "Plan DAG").fg,
        termlens::Color::Default,
        "the highlight moved off Plan DAG rather than being drawn twice"
    );

    assert_eq!(
        before.text(),
        after.text(),
        "Tab changed the text; it should only change which border is yellow"
    );
    let diff = before.diff(&after);
    assert!(
        !diff.is_empty(),
        "Tab changed nothing at all — the focus highlight is gone:\n{after}"
    );
    // Plan DAG and Inspector are the two panels stacked in the left half, so
    // a repaint of the right half means the focus move dragged the telemetry
    // gauges and the Describe table along with it.
    let stray: Vec<(u16, u16)> = diff
        .cells()
        .map(|(row, col, ..)| (row, col))
        .filter(|(_, col)| *col >= 40)
        .collect();
    assert!(
        stray.is_empty(),
        "Tab repainted the right half at {stray:?}:\n{diff}"
    );
    quit(&mut t)
}

/// ↓ selects the next operator, and touches nothing it has no business
/// touching.
///
/// The selection lives in the Plan DAG and the Inspector — both in the left
/// half — so the gauges and the Describe table must come through a keystroke
/// untouched. Nothing forbade a full repaint before this; a plan panel that
/// redrew the whole screen on every arrow key would pass every other test
/// here and flicker on a real terminal.
#[test]
fn moving_the_selection_leaves_the_right_half_alone() -> termlens::Result<()> {
    let mut t = spawn(80, 24)?;
    let before = t.wait_frame(|s| s.contains("16.0 MiB"))?;
    t.send(Key::Down)?;
    let after = t.wait_frame(|s| s.contains("1000000") && s.contains("32.0 MiB"))?;

    let diff = before.diff(&after);
    assert!(!diff.is_empty(), "↓ selected nothing:\n{after}");
    let stray: Vec<(u16, u16)> = diff
        .cells()
        .map(|(row, col, ..)| (row, col))
        .filter(|(_, col)| *col >= 40)
        .collect();
    assert!(
        stray.is_empty(),
        "↓ repainted the Telemetry / Describe half at {stray:?}:\n{diff}"
    );
    // Said the other way round, on the text rather than the cells.
    assert_eq!(
        before.rect_text(40.., ..),
        after.rect_text(40.., ..),
        "the right half of the dashboard changed when the selection moved"
    );
    // The rows it did touch are the two panels that show a selection.
    let mut rows: Vec<u16> = diff.cells().map(|(row, ..)| row).collect();
    rows.dedup();
    assert!(
        rows.iter()
            .all(|row| (2..=3).contains(row) || (14..=21).contains(row)),
        "↓ should repaint the plan rows and the Inspector body, got {rows:?}:\n{diff}"
    );
    quit(&mut t)
}
