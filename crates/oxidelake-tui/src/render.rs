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

/// Describe's column widths, in header order. Named because the cell
/// formatters below have to agree with the layout: ratatui clips a cell that
/// overflows its column, and for a number that silently changes its value.
const DESCRIBE_WIDTHS: [u16; 8] = [8, 8, 7, 7, 7, 6, 6, 6];

/// Drops a decimal fraction's trailing zeros, and the point left behind.
fn trim_zeros(rendered: &str) -> String {
    if !rendered.contains('.') {
        return rendered.to_owned();
    }
    rendered
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

/// Text that cannot fit, marked as shortened rather than quietly cut.
fn fit_text(raw: &str, width: usize) -> String {
    if raw.chars().count() <= width {
        return raw.to_owned();
    }
    match width {
        0 => String::new(),
        _ => raw.chars().take(width - 1).chain(['…']).collect(),
    }
}

/// Fits a rendered number into `width` columns without ever cutting a digit.
///
/// `approx_percentile_cont` renders full precision — the 2M-row demo table's
/// P99 of `id` arrives as `1979969.2416513609`. Clipped to the six columns
/// Describe has, that printed `197999`: an order of magnitude low, and below
/// the median displayed above it, with nothing to show it had been cut. So a
/// value that already fits is kept exactly as the query rendered it, and
/// anything longer is *rounded* to the most decimals that fit, falling back to
/// an exponent form when even the integer part is too wide.
fn fit_number(raw: &str, width: usize) -> String {
    if raw.chars().count() <= width {
        return raw.to_owned();
    }
    let Ok(value) = raw.parse::<f64>() else {
        return fit_text(raw, width);
    };
    if !value.is_finite() {
        return fit_text(raw, width);
    }
    for precision in (0..=3).rev() {
        let candidate = trim_zeros(&format!("{value:.precision$}"));
        if candidate.chars().count() <= width {
            return candidate;
        }
    }
    for precision in (0..=2).rev() {
        let candidate = format!("{value:.precision$e}");
        if candidate.chars().count() <= width {
            return candidate;
        }
    }
    fit_text(raw, width)
}

fn render_describe(state: &AppState, model: &DashboardModel, frame: &mut Frame<'_>, area: Rect) {
    let header =
        Row::new(["column", "type", "min", "max", "nulls", "p25", "p50", "p99"].map(Cell::from))
            .style(Style::default().add_modifier(Modifier::BOLD));
    let rows: Vec<Row<'_>> = model
        .profiles
        .iter()
        .map(|p| {
            let w = DESCRIBE_WIDTHS.map(usize::from);
            Row::new(vec![
                Cell::from(fit_text(&p.name, w[0])),
                Cell::from(fit_text(&p.data_type, w[1])),
                Cell::from(fit_number(&p.min, w[2])),
                Cell::from(fit_number(&p.max, w[3])),
                Cell::from(fit_number(&p.null_count.to_string(), w[4])),
                Cell::from(fit_number(&p.p25, w[5])),
                Cell::from(fit_number(&p.p50, w[6])),
                Cell::from(fit_number(&p.p99, w[7])),
            ])
        })
        .collect();
    let widths = DESCRIBE_WIDTHS.map(Constraint::Length);
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

    /// A percentile parses back to the number it stands for.
    fn num(rendered: &str) -> f64 {
        rendered.parse().unwrap_or(f64::NAN)
    }

    #[test]
    fn a_value_that_fits_is_left_exactly_as_the_query_rendered_it() {
        for raw in ["0", "0.0", "99", "24.75", "1999999", "", "s0"] {
            assert_eq!(fit_number(raw, 7), raw);
        }
        assert_eq!(fit_text("Int64", 8), "Int64");
    }

    /// The bug: `id`'s P99 on the 2M-row demo table arrives as
    /// `1979969.2416513609`, and six columns of hard clipping printed
    /// `197999` — ten times too small, and below the median above it.
    #[test]
    fn wide_percentiles_round_instead_of_losing_a_digit() {
        let (p25, p50, p99) = (
            fit_number("505700.81345880596", 6),
            fit_number("996257.6842335836", 6),
            fit_number("1979969.2416513609", 6),
        );
        assert_ne!(p99, "197999", "P99 was clipped mid-integer");
        assert!(
            num(&p25) < num(&p50) && num(&p50) < num(&p99),
            "percentiles out of order: {p25} {p50} {p99}"
        );
        for (rendered, want) in [(&p25, 505_700.81), (&p50, 996_257.68), (&p99, 1_979_969.24)] {
            let error = (num(rendered) - want).abs() / want;
            assert!(error < 0.01, "{rendered} is not within 1% of {want}");
        }
    }

    #[test]
    fn no_cell_can_overflow_its_column() {
        let samples = [
            "1979969.2416513609",
            "-1979969.2416513609",
            "0.000012345678",
            "123456789012345",
            "1e300",
            "-1e-300",
            "6.122512376708984",
            "not a number at all",
            "",
        ];
        for raw in samples {
            for width in 1..=8 {
                let cell = fit_number(raw, width);
                assert!(
                    cell.chars().count() <= width,
                    "{raw:?} rendered {cell:?}, wider than {width}"
                );
            }
        }
    }

    #[test]
    fn text_too_long_is_marked_as_shortened() {
        assert_eq!(fit_text("FixedSizeList(8 x Float32)", 8), "FixedSi…");
        assert_eq!(fit_text("Float64", 1), "…");
        assert_eq!(fit_text("abc", 0), "");
    }

    #[test]
    fn trailing_zeros_go_but_the_value_stays() {
        assert_eq!(trim_zeros("24.750"), "24.75");
        assert_eq!(trim_zeros("0.000"), "0");
        assert_eq!(trim_zeros("996258"), "996258");
        assert_eq!(trim_zeros("100"), "100");
    }

    #[test]
    fn every_describe_column_has_a_width() {
        assert_eq!(DESCRIBE_WIDTHS.len(), 8, "one width per Describe column");
    }

    #[test]
    fn ratio_is_clamped() {
        assert_eq!(ratio(1, 0), 0.0);
        assert_eq!(ratio(5, 4), 1.0);
        assert!((ratio(1, 4) - 0.25).abs() < f64::EPSILON);
    }
}
