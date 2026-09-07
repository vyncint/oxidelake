//! In-process rendering tests: `ratatui::backend::TestBackend` frames are
//! snapshotted with `insta` at 80×24 and 120×40, and keyboard transitions are
//! asserted on `AppState`. No terminal or PTY is involved.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use oxidelake_tui::DashboardModel;
use oxidelake_tui::{AppState, ColumnProfile, KeyInput, Panel, Transition, demo_model, render};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

fn frame(width: u16, height: u16, state: &AppState) -> String {
    frame_of(width, height, state, &demo_model())
}

fn frame_of(width: u16, height: u16, state: &AppState, model: &DashboardModel) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|f| render(state, model, f)).unwrap();
    terminal.backend().to_string()
}

/// The row of a rendered frame that profiles `column`.
fn describe_row<'a>(frame: &'a str, column: &str) -> &'a str {
    frame
        .lines()
        .find(|line| line.contains(&format!("{column} ")) && line.contains("Int64"))
        .unwrap_or_else(|| panic!("no Describe row for {column}:\n{frame}"))
}

#[test]
fn initial_frame_80x24() {
    let text = frame(80, 24, &AppState::new(3));
    for needle in [
        "OxideLake",
        "Plan DAG",
        "Inspector",
        "Telemetry",
        "Describe",
        "GpuAggregateExec",
        "[CUDA]",
        "[CPU]",
    ] {
        assert!(text.contains(needle), "missing {needle}:\n{text}");
    }
    insta::assert_snapshot!(text);
}

#[test]
fn initial_frame_120x40() {
    let text = frame(120, 40, &AppState::new(3));
    assert!(
        text.contains("group_by=[k], aggr=[SUM(v), COUNT(v)]"),
        "{text}"
    );
    insta::assert_snapshot!(text);
}

#[test]
fn selection_moves_and_panels_cycle() {
    let mut state = AppState::new(3);
    assert_eq!(state.on_key(KeyInput::Down), Transition::Continue);
    assert_eq!(state.on_key(KeyInput::Down), Transition::Continue);
    assert_eq!(state.on_key(KeyInput::Down), Transition::Continue); // saturates
    assert_eq!(state.selected(), 2);
    assert_eq!(state.on_key(KeyInput::Tab), Transition::Continue);
    assert_eq!(state.panel(), Panel::Inspector);
    let text = frame(80, 24, &state);
    assert!(text.contains("DataSourceExec"), "{text}");
    insta::assert_snapshot!(text);

    assert_eq!(state.on_key(KeyInput::Up), Transition::Continue);
    assert_eq!(state.selected(), 1);
    assert_eq!(state.on_key(KeyInput::Other), Transition::Continue);
    assert_eq!(state.on_key(KeyInput::Quit), Transition::Quit);
    assert!(state.should_quit());
}

#[test]
fn rendering_is_deterministic_and_stable_across_sizes() {
    let state = AppState::new(3);
    assert_eq!(frame(80, 24, &state), frame(80, 24, &state));
    for (w, h) in [(80, 24), (100, 30), (120, 40), (200, 60)] {
        let text = frame(w, h, &state);
        for needle in ["Plan DAG", "Inspector", "Telemetry", "Describe", "quit"] {
            assert!(text.contains(needle), "{w}x{h} missing {needle}:\n{text}");
        }
    }
}

/// Describe renders what `approx_percentile_cont` actually returns: full
/// precision. The 2M-row demo table's P99 of `id` is `1979969.2416513609`,
/// and clipping it to the column's six characters printed `197999` — below
/// the median in the same row, with nothing to mark the cut.
#[test]
fn wide_percentiles_render_in_order_in_the_panel() {
    let mut model = demo_model();
    model.profiles = vec![ColumnProfile {
        name: "id".into(),
        data_type: "Int64".into(),
        min: "0".into(),
        max: "1999999".into(),
        null_count: 0,
        p25: "505700.81345880596".into(),
        p50: "996257.6842335836".into(),
        p99: "1979969.2416513609".into(),
    }];
    let text = frame_of(160, 40, &AppState::new(3), &model);
    let row = describe_row(&text, "id");
    assert!(
        !row.contains("197999 ") && !row.contains("197999|"),
        "P99 was clipped mid-integer: {row}"
    );
    let numbers: Vec<f64> = row
        .split_whitespace()
        .filter_map(|cell| cell.parse::<f64>().ok())
        .collect();
    let (p25, p50, p99) = (505_700.81, 996_257.68, 1_979_969.24);
    for want in [p25, p50, p99] {
        assert!(
            numbers.iter().any(|got| (got - want).abs() / want < 0.01),
            "no cell within 1% of {want} in: {row}\nparsed {numbers:?}"
        );
    }
    let p99_cell = numbers
        .iter()
        .copied()
        .find(|got| (got - p99).abs() / p99 < 0.01)
        .unwrap_or(f64::NAN);
    let p50_cell = numbers
        .iter()
        .copied()
        .find(|got| (got - p50).abs() / p50 < 0.01)
        .unwrap_or(f64::NAN);
    assert!(
        p99_cell > p50_cell,
        "P99 {p99_cell} not above P50 {p50_cell}"
    );
}
