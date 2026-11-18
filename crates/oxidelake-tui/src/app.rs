//! The real-terminal event loop (crossterm) with DEC 2026 synchronized updates.

use std::io::{self, IsTerminal, Write};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{BeginSynchronizedUpdate, EndSynchronizedUpdate};
use ratatui::DefaultTerminal;

use crate::model::DashboardModel;
use crate::render::render;
use crate::state::{AppState, KeyInput, Transition};

/// Message shown when the dashboard is started without an interactive terminal.
pub const NON_INTERACTIVE_TERMINAL_MESSAGE: &str = "oxide tui needs an interactive terminal; use oxide sql or oxide explain for non-interactive output";

/// Whether both input and output are attached to an interactive terminal.
pub fn is_interactive_terminal() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

/// Maps a crossterm key event onto the dashboard's input alphabet. Key
/// releases (reported by some terminals) are ignored.
pub fn key_input_from_crossterm(key: &KeyEvent) -> Option<KeyInput> {
    if key.kind == KeyEventKind::Release {
        return None;
    }
    Some(match key.code {
        KeyCode::Up | KeyCode::Char('k') => KeyInput::Up,
        KeyCode::Down | KeyCode::Char('j') => KeyInput::Down,
        KeyCode::Tab => KeyInput::Tab,
        KeyCode::Char('q') | KeyCode::Esc => KeyInput::Quit,
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => KeyInput::Quit,
        _ => KeyInput::Other,
    })
}

/// Draws one frame, bracketed in a synchronized update so the terminal (or a
/// PTY harness) never observes a torn repaint.
pub fn draw_frame(
    terminal: &mut DefaultTerminal,
    state: &AppState,
    model: &DashboardModel,
) -> io::Result<()> {
    let mut out = io::stdout();
    execute!(out, BeginSynchronizedUpdate)?;
    terminal.draw(|frame| render(state, model, frame))?;
    execute!(out, EndSynchronizedUpdate)?;
    out.flush()
}

/// Runs the dashboard on the real terminal until the user quits.
///
/// With `tick = None` the loop blocks on input and only repaints on key or
/// resize events (fully deterministic — what the demo binary uses). With a
/// tick, `refresh` is called before every repaint to pull live telemetry.
/// It may bring a plan or change its length: the selection follows, and stays
/// on its row for as long as that row exists.
pub fn run_terminal(
    mut model: DashboardModel,
    tick: Option<Duration>,
    mut refresh: impl FnMut(&mut DashboardModel),
) -> io::Result<()> {
    if !is_interactive_terminal() {
        return Err(io::Error::new(
            io::ErrorKind::NotConnected,
            NON_INTERACTIVE_TERMINAL_MESSAGE,
        ));
    }
    let mut terminal = ratatui::try_init()?;
    let result = event_loop(
        &mut model,
        &mut refresh,
        |state, model| draw_frame(&mut terminal, state, model),
        || next_event(tick),
    );
    ratatui::restore();
    result
}

/// The next input event from the terminal, or `None` when `tick` passed
/// without one. With no tick it waits for as long as it takes.
fn next_event(tick: Option<Duration>) -> io::Result<Option<Event>> {
    match tick {
        Some(timeout) if !event::poll(timeout)? => Ok(None),
        _ => event::read().map(Some),
    }
}

/// [`run_terminal`]'s loop with the terminal at arm's length, so that a test
/// can drive it too: `draw` paints one frame and `next_event` waits for one
/// input — `None` when the tick passed first. A resize, like any event that
/// is not a key, only leads to the next repaint.
fn event_loop(
    model: &mut DashboardModel,
    refresh: &mut impl FnMut(&mut DashboardModel),
    mut draw: impl FnMut(&AppState, &DashboardModel) -> io::Result<()>,
    mut next_event: impl FnMut() -> io::Result<Option<Event>>,
) -> io::Result<()> {
    let mut state = AppState::new(model.plan().len());
    loop {
        refresh(model);
        // The refresh may have brought a plan, or changed its length (#126).
        state.set_node_count(model.plan().len());
        draw(&state, model)?;
        let Some(event) = next_event()? else {
            continue;
        };
        if let Event::Key(key) = event
            && let Some(input) = key_input_from_crossterm(&key)
            && state.on_key(input) == Transition::Quit
        {
            return Ok(());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn maps_keys() {
        assert_eq!(
            key_input_from_crossterm(&press(KeyCode::Up)),
            Some(KeyInput::Up)
        );
        assert_eq!(
            key_input_from_crossterm(&press(KeyCode::Char('j'))),
            Some(KeyInput::Down)
        );
        assert_eq!(
            key_input_from_crossterm(&press(KeyCode::Tab)),
            Some(KeyInput::Tab)
        );
        assert_eq!(
            key_input_from_crossterm(&press(KeyCode::Char('q'))),
            Some(KeyInput::Quit)
        );
        assert_eq!(
            key_input_from_crossterm(&press(KeyCode::Esc)),
            Some(KeyInput::Quit)
        );
        assert_eq!(
            key_input_from_crossterm(&KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Some(KeyInput::Quit)
        );
        assert_eq!(
            key_input_from_crossterm(&press(KeyCode::Char('x'))),
            Some(KeyInput::Other)
        );
        let mut release = press(KeyCode::Char('q'));
        release.kind = KeyEventKind::Release;
        assert_eq!(key_input_from_crossterm(&release), None);
    }

    /// Drives `event_loop` with `inputs` — a key, or `None` for a tick that
    /// passed with none — and a `q` after them, and returns the selection
    /// each frame was drawn with.
    fn selections(
        mut model: DashboardModel,
        mut refresh: impl FnMut(&mut DashboardModel),
        inputs: &[Option<KeyCode>],
    ) -> Vec<usize> {
        let mut inputs = inputs
            .iter()
            .copied()
            .chain([Some(KeyCode::Char('q'))])
            .map(|input| input.map(|code| Event::Key(press(code))));
        let mut drawn = Vec::new();
        event_loop(
            &mut model,
            &mut refresh,
            |state, model| {
                assert!(
                    model.plan().is_empty() || state.selected() < model.plan().len(),
                    "drew a selection outside the plan: {} of {}",
                    state.selected(),
                    model.plan().len()
                );
                drawn.push(state.selected());
                Ok(())
            },
            // Asked for an input past the `q`, the loop ignored it: end it
            // with an error so the test fails rather than spins.
            || {
                inputs
                    .next()
                    .ok_or_else(|| io::Error::other("the loop did not quit on q"))
            },
        )
        .unwrap();
        drawn
    }

    /// `run_terminal`'s rustdoc invites a refresh that supplies the plan, and
    /// the selection was sized once, before the first refresh: a plan the
    /// refresh brought could not be walked past its old length, which for an
    /// empty start means not at all (#126).
    #[test]
    fn a_plan_the_refresh_brings_can_be_walked_to_its_last_operator() {
        let refresh = |model: &mut DashboardModel| {
            if model.plan().is_empty() {
                *model = crate::demo_model();
            }
        };
        let down = Some(KeyCode::Down);
        assert_eq!(
            selections(DashboardModel::default(), refresh, &[down, down, down]),
            [0, 1, 2, 2],
            "↓ walks the three operators the refresh brought, and stops at the last"
        );
    }

    /// The other direction: a plan that shrinks under the selection takes it
    /// to the last operator left, not to a row that no longer exists.
    #[test]
    fn a_plan_that_shrinks_keeps_the_selection_on_it() {
        let mut refreshes = 0;
        let refresh = |model: &mut DashboardModel| {
            refreshes += 1;
            if refreshes == 4 {
                model.telemetry.plan.truncate(1);
                model.telemetry.operators.truncate(1);
            }
        };
        let down = Some(KeyCode::Down);
        assert_eq!(
            selections(crate::demo_model(), refresh, &[down, down, None]),
            [0, 1, 2, 0]
        );
    }
}
