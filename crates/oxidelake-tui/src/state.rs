//! Pure application state for the dashboard.

/// The four dashboard panels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Panel {
    /// Physical plan tree with a hardware tag per operator.
    PlanDag,
    /// Live statistics for the selected operator.
    Inspector,
    /// Memory tier gauges and transfer/spill rates.
    Telemetry,
    /// Per-column profiling summary.
    Describe,
}

impl Panel {
    /// All panels in `Tab` order.
    pub const ALL: [Panel; 4] = [
        Panel::PlanDag,
        Panel::Inspector,
        Panel::Telemetry,
        Panel::Describe,
    ];

    /// The panel after `self` in `Tab` order (wrapping).
    pub const fn next(self) -> Panel {
        match self {
            Panel::PlanDag => Panel::Inspector,
            Panel::Inspector => Panel::Telemetry,
            Panel::Telemetry => Panel::Describe,
            Panel::Describe => Panel::PlanDag,
        }
    }

    /// Human-readable title.
    pub const fn title(self) -> &'static str {
        match self {
            Panel::PlanDag => "Plan DAG",
            Panel::Inspector => "Inspector",
            Panel::Telemetry => "Telemetry",
            Panel::Describe => "Describe",
        }
    }
}

/// Keyboard input, abstracted from the terminal backend so the state machine
/// stays pure and testable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyInput {
    /// Move the selection up.
    Up,
    /// Move the selection down.
    Down,
    /// Cycle the focused panel.
    Tab,
    /// Quit the application.
    Quit,
    /// Any other key (ignored).
    Other,
}

/// The outcome of handling one input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    /// Keep running.
    Continue,
    /// Leave the event loop.
    Quit,
}

/// Dashboard state: which operator is selected, which panel is focused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppState {
    node_count: usize,
    selected: usize,
    panel: Panel,
    quit: bool,
}

impl AppState {
    /// Creates a state for a plan with `node_count` operators, selecting the first.
    pub const fn new(node_count: usize) -> Self {
        Self {
            node_count,
            selected: 0,
            panel: Panel::PlanDag,
            quit: false,
        }
    }

    /// Number of operators in the displayed plan.
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Index of the selected operator (always `< node_count` unless the plan is empty).
    pub const fn selected(&self) -> usize {
        self.selected
    }

    /// The focused panel.
    pub const fn panel(&self) -> Panel {
        self.panel
    }

    /// `true` once a quit was requested.
    pub const fn should_quit(&self) -> bool {
        self.quit
    }

    /// Selects the next operator, saturating at the last one.
    pub fn select_next(&mut self) {
        if self.node_count > 0 && self.selected + 1 < self.node_count {
            self.selected += 1;
        }
    }

    /// Selects the previous operator, saturating at the first one.
    pub fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    /// Focuses the next panel.
    pub fn cycle_panel(&mut self) {
        self.panel = self.panel.next();
    }

    /// Applies one input and reports whether the loop should continue.
    pub fn on_key(&mut self, key: KeyInput) -> Transition {
        match key {
            KeyInput::Up => self.select_prev(),
            KeyInput::Down => self.select_next(),
            KeyInput::Tab => self.cycle_panel(),
            KeyInput::Quit => self.quit = true,
            KeyInput::Other => {}
        }
        if self.quit {
            Transition::Quit
        } else {
            Transition::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_saturates_at_both_ends() {
        let mut s = AppState::new(3);
        assert_eq!(s.on_key(KeyInput::Up), Transition::Continue);
        assert_eq!(s.selected(), 0);
        s.on_key(KeyInput::Down);
        s.on_key(KeyInput::Down);
        s.on_key(KeyInput::Down);
        assert_eq!(s.selected(), 2);
    }

    #[test]
    fn empty_plan_never_selects() {
        let mut s = AppState::new(0);
        s.on_key(KeyInput::Down);
        assert_eq!(s.selected(), 0);
    }

    #[test]
    fn tab_cycles_all_panels() {
        let mut s = AppState::new(1);
        for expected in [
            Panel::Inspector,
            Panel::Telemetry,
            Panel::Describe,
            Panel::PlanDag,
        ] {
            s.on_key(KeyInput::Tab);
            assert_eq!(s.panel(), expected);
        }
    }

    #[test]
    fn quit_transitions() {
        let mut s = AppState::new(1);
        assert_eq!(s.on_key(KeyInput::Other), Transition::Continue);
        assert_eq!(s.on_key(KeyInput::Quit), Transition::Quit);
        assert!(s.should_quit());
    }
}
