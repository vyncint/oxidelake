//! In-process rendering tests: `ratatui::backend::TestBackend` frames are
//! snapshotted with `insta` at 80×24 and 120×40, and keyboard transitions are
//! asserted on `AppState`. No terminal or PTY is involved.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use oxidelake_tui::{AppState, KeyInput, Panel, Transition, demo_model, render};
use ratatui::Terminal;
use ratatui::backend::TestBackend;

fn frame(width: u16, height: u16, state: &AppState) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    let model = demo_model();
    terminal.draw(|f| render(state, &model, f)).unwrap();
    terminal.backend().to_string()
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
