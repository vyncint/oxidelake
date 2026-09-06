//! Pure rendering: `render(state, model, frame)` draws the four panels and
//! never reads a clock or any global, so the same inputs always produce the
//! same frame.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Gauge, List, ListItem, ListState, Paragraph, Row, Table,
};

use oxidelake_core::BackendKind;

use crate::model::DashboardModel;
use crate::state::{AppState, Panel};

/// Human-readable byte count with a binary unit.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn backend_tag(backend: BackendKind) -> Span<'static> {
    let (text, color) = match backend {
        BackendKind::Cuda => ("[CUDA]", Color::Green),
        BackendKind::Metal => ("[Metal]", Color::Magenta),
        BackendKind::CpuSimd => ("[CPU]", Color::Blue),
    };
    Span::styled(
        text,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )
}

fn panel_block(title: &str, panel: Panel, focused: Panel) -> Block<'static> {
    let style = if panel == focused {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    Block::default()
        .title(Span::styled(format!(" {title} "), style))
        .borders(Borders::ALL)
        .border_style(style)
}

/// Draws the whole dashboard into `frame`.
pub fn render(state: &AppState, model: &DashboardModel, frame: &mut Frame<'_>) {
    let area = frame.area();
    let [header, body, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(8),
        Constraint::Length(1),
    ])
    .areas(area);
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(body);
    let [plan_area, inspector_area] =
        Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(left);
    let [telemetry_area, describe_area] =
        Layout::vertical([Constraint::Percentage(45), Constraint::Percentage(55)]).areas(right);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "OxideLake",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::raw(model.title.clone()),
        ])),
        header,
    );
    render_plan(state, model, frame, plan_area);
    render_inspector(state, model, frame, inspector_area);
    render_telemetry(state, model, frame, telemetry_area);
    render_describe(state, model, frame, describe_area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ↑/↓ ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("select operator  "),
            Span::styled("Tab", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" next panel  "),
            Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" quit"),
        ])),
        footer,
    );
}

fn render_plan(state: &AppState, model: &DashboardModel, frame: &mut Frame<'_>, area: Rect) {
    let items: Vec<ListItem<'_>> = model
        .plan()
        .iter()
        .map(|node| {
            let indent = "  ".repeat(node.depth);
            let branch = if node.depth == 0 { "" } else { "└─ " };
            ListItem::new(Line::from(vec![
                Span::raw(format!("{indent}{branch}")),
                Span::styled(
                    node.name.clone(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                backend_tag(node.backend),
            ]))
        })
        .collect();
    let list = List::new(items)
        .block(panel_block("Plan DAG", Panel::PlanDag, state.panel()))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▶ ");
    let mut list_state = ListState::default();
    if !model.plan().is_empty() {
        list_state.select(Some(state.selected()));
    }
    frame.render_stateful_widget(list, area, &mut list_state);
}

fn render_inspector(state: &AppState, model: &DashboardModel, frame: &mut Frame<'_>, area: Rect) {
    let block = panel_block("Inspector", Panel::Inspector, state.panel());
    let lines: Vec<Line<'_>> = match (
        model.plan().get(state.selected()),
        model.operator(state.selected()),
    ) {
        (Some(node), Some(op)) => vec![
            Line::from(vec![
                Span::styled(
                    node.name.clone(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                backend_tag(node.backend),
            ]),
            Line::from(node.detail.clone()),
            Line::from(""),
            Line::from(format!("rows in      {:>14}", op.rows_in)),
            Line::from(format!("rows out     {:>14}", op.rows_out)),
            Line::from(format!("batches      {:>14}", op.batches)),
            Line::from(format!(
                "latency/batch{:>11.3} ms",
                op.mean_batch_latency_ms()
            )),
            Line::from(format!("memory       {:>14}", human_bytes(op.memory_bytes))),
            Line::from(format!(
                "H2D / D2H    {:>6} / {:>6}",
                human_bytes(op.bytes_h2d),
                human_bytes(op.bytes_d2h)
            )),
        ],
        (Some(node), None) => vec![Line::from(node.name.clone()), Line::from("no counters yet")],
        _ => vec![Line::from("no operator selected")],
    };
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn ratio(used: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (used as f64 / total as f64).clamp(0.0, 1.0)
    }
}

fn render_telemetry(state: &AppState, model: &DashboardModel, frame: &mut Frame<'_>, area: Rect) {
    let block = panel_block("Telemetry", Panel::Telemetry, state.panel());
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [vram, host, disk, rates] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Length(2),
        Constraint::Min(1),
    ])
    .areas(inner);
    let tiers = &model.telemetry.tiers;
    // Fixed reference capacities keep the gauges deterministic; real capacity
    // arrives with backend MemoryInfo in a later release.
    const VRAM_CAP: u64 = 8 * 1024 * 1024 * 1024;
    const HOST_CAP: u64 = 4 * 1024 * 1024 * 1024;
    const DISK_CAP: u64 = 2 * 1024 * 1024 * 1024;
    for (area, label, used, cap, color) in [
        (vram, "VRAM", tiers.device_bytes, VRAM_CAP, Color::Green),
        (host, "pinned RAM", tiers.host_bytes, HOST_CAP, Color::Cyan),
        (
            disk,
            "disk spill",
            tiers.disk_bytes,
            DISK_CAP,
            Color::Yellow,
        ),
    ] {
        frame.render_widget(
            Gauge::default()
                .gauge_style(Style::default().fg(color))
                .ratio(ratio(used, cap))
                .label(format!(
                    "{label}: {} / {}",
                    human_bytes(used),
                    human_bytes(cap)
                )),
            area,
        );
    }
    let spill = &model.telemetry.spill;
    let h2d: u64 = model.telemetry.operators.iter().map(|o| o.bytes_h2d).sum();
    let d2h: u64 = model.telemetry.operators.iter().map(|o| o.bytes_d2h).sum();
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(format!(
                "PCIe  H2D {}  D2H {}",
                human_bytes(h2d),
                human_bytes(d2h)
            )),
            Line::from(format!(
                "NVMe  spilled {} ({} demotions)  reloaded {} ({} promotions)",
                human_bytes(spill.spilled_bytes),
                spill.demotions,
                human_bytes(spill.reloaded_bytes),
                spill.promotions
            )),
        ]),
        rates,
    );
}

fn render_describe(state: &AppState, model: &DashboardModel, frame: &mut Frame<'_>, area: Rect) {
    let header =
        Row::new(["column", "type", "min", "max", "nulls", "p25", "p50", "p99"].map(Cell::from))
            .style(Style::default().add_modifier(Modifier::BOLD));
    let rows: Vec<Row<'_>> = model
        .profiles
        .iter()
        .map(|p| {
            Row::new(vec![
                Cell::from(p.name.clone()),
                Cell::from(p.data_type.clone()),
                Cell::from(p.min.clone()),
                Cell::from(p.max.clone()),
                Cell::from(p.null_count.to_string()),
                Cell::from(p.p25.clone()),
                Cell::from(p.p50.clone()),
                Cell::from(p.p99.clone()),
            ])
        })
        .collect();
    let widths = [
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(6),
    ];
    frame.render_widget(
        Table::new(rows, widths).header(header).block(panel_block(
            "Describe",
            Panel::Describe,
            state.panel(),
        )),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_are_human() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(6 * 1024 * 1024 * 1024), "6.0 GiB");
    }

    #[test]
    fn ratio_is_clamped() {
        assert_eq!(ratio(1, 0), 0.0);
        assert_eq!(ratio(5, 4), 1.0);
        assert!((ratio(1, 4) - 0.25).abs() < f64::EPSILON);
    }
}
