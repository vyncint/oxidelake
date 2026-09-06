//! Terminal dashboard built on ratatui + crossterm (docs/SPEC.md §2.7).
//!
//! The application is a state machine decoupled from rendering:
//! [`AppState`] + [`AppState::on_key`] are pure and unit-tested; [`render()`] is
//! a pure function of `(state, model, frame)`; [`run_terminal`] is the only
//! place that touches a real terminal, and it brackets every repaint in a
//! DEC 2026 synchronized update so PTY tests observe complete frames.
//!
//! Panels: Plan DAG (hardware tag per operator), Inspector (selected operator's
//! counters), Telemetry (memory tiers, PCIe/NVMe rates) and Describe (column
//! profiles). Keys: `↑`/`↓` select, `Tab` cycles panels, `q` quits.
//!
//! In v1 the dashboard observes the local process only (a
//! [`DashboardModel`] built from the session's `TelemetryHub`).
//!
//! Dependency direction: depends on `oxidelake-core` only.

pub mod app;
pub mod model;
pub mod render;
pub mod state;

pub use app::{draw_frame, key_input_from_crossterm, run_terminal};
pub use model::{ColumnProfile, DashboardModel, demo_model};
pub use render::render;
pub use state::{AppState, KeyInput, Panel, Transition};
