//! The real-terminal event loop (crossterm) with DEC 2026 synchronized updates.

use std::io::{self, Write};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{BeginSynchronizedUpdate, EndSynchronizedUpdate};
use ratatui::DefaultTerminal;

use crate::model::DashboardModel;
use crate::render::render;
use crate::state::{AppState, KeyInput, Transition};

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
pub fn run_terminal(
    mut model: DashboardModel,
    tick: Option<Duration>,
    mut refresh: impl FnMut(&mut DashboardModel),
) -> io::Result<()> {
    let mut terminal = ratatui::try_init()?;
    let mut state = AppState::new(model.plan().len());
    let result = (|| -> io::Result<()> {
        loop {
            refresh(&mut model);
            draw_frame(&mut terminal, &state, &model)?;
            let ready = match tick {
                Some(timeout) => event::poll(timeout)?,
                None => true,
            };
            if !ready {
                continue;
            }
            match event::read()? {
                Event::Key(key) => {
                    if let Some(input) = key_input_from_crossterm(&key)
                        && state.on_key(input) == Transition::Quit
                    {
                        return Ok(());
                    }
                }
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    })();
    ratatui::restore();
    result
}

#[cfg(test)]
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
}
