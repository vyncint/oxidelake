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
use oxidelake_core::telemetry::OperatorSnapshot;

use crate::model::{ColumnProfile, DashboardModel};
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
        // `BackendKind` is `#[non_exhaustive]`, so a backend added to
        // oxidelake-core compiles here instead of breaking the dashboard.
        // It renders as a question rather than as nothing: a tag the
        // dashboard does not recognise is worth seeing, and silently
        // drawing it as `[CPU]` would misreport where the work ran.
        _ => ("[?]", Color::Yellow),
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

/// The Inspector's height: the ten lines [`render_inspector`] draws for an
/// operator, and its border. As 45% of the body it had eight lines at 24
/// rows, which cut the last two — H2D / D2H and the CPU-fallback count, the
/// line that tells a GPU run from a silent CPU one (#32) — below 28 rows
/// (#122).
const INSPECTOR_HEIGHT: u16 = 12;

/// The Telemetry panel's: three two-row gauges, three lines under them and
/// the border. As 45% of the body it cut the last line, which says whether
/// the gauges can move at all (#25), below 26 rows (#122).
const TELEMETRY_HEIGHT: u16 = 11;

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
    let [plan_area, inspector_area] = Layout::vertical([
        Constraint::Percentage(55),
        Constraint::Min(INSPECTOR_HEIGHT),
    ])
    .areas(left);
    let [telemetry_area, describe_area] = Layout::vertical([
        Constraint::Min(TELEMETRY_HEIGHT),
        Constraint::Percentage(55),
    ])
    .areas(right);

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
            fallback_line(op),
        ],
        (Some(node), None) => vec![Line::from(node.name.clone()), Line::from("no counters yet")],
        _ => vec![Line::from("no operator selected")],
    };
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// Says whether the three gauges above can move at all (#25).
///
/// `SpillManager` is a library in 0.x: no query registers a batch with it, so
/// the tiers read zero however much memory a query used. A panel that showed
/// a bounded memory model the engine does not have would be a lie told in
/// colour, so it says which of the two it is showing.
fn spill_scope_line(on_query_path: bool) -> Line<'static> {
    if on_query_path {
        Line::from(vec![
            Span::raw("spill "),
            Span::styled("on the query path", Style::default().fg(Color::Green)),
        ])
    } else {
        Line::from(vec![
            Span::raw("spill "),
            Span::styled(
                "library only: no query registers batches",
                Style::default().fg(Color::DarkGray),
            ),
        ])
    }
}

/// The CPU-fallback line (#32).
///
/// A GPU deployment that silently runs everything on the CPU produces the
/// same rows and the same `EXPLAIN` tags as one that does not, so this is the
/// line that tells them apart. It is coloured rather than merely printed:
/// zero is the claim being made by the backend tag above it, and any other
/// number contradicts it.
fn fallback_line(op: &OperatorSnapshot) -> Line<'static> {
    let style = if op.fell_back_entirely() {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else if op.fallback_batches > 0 {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Green)
    };
    Line::from(vec![
        Span::raw("cpu fallback "),
        Span::styled(
            format!(
                "{:>14}",
                format!("{} / {}", op.fallback_batches, op.batches)
            ),
            style,
        ),
    ])
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
    let capacity = model.telemetry.capacity;
    // Capacities come from the backend's `MemoryInfo`, and a tier the engine
    // has no number for says so rather than being drawn against a plausible
    // constant — which is what these gauges used to do (#25). The disk tier
    // has no capacity at all: the spill directory's free space is the
    // filesystem's business, not the engine's.
    for (area, label, used, cap, color) in [
        (
            vram,
            "VRAM",
            tiers.device_bytes,
            capacity.device_bytes,
            Color::Green,
        ),
        (
            host,
            "host RAM",
            tiers.host_bytes,
            capacity.host_bytes,
            Color::Cyan,
        ),
        (disk, "disk spill", tiers.disk_bytes, None, Color::Yellow),
    ] {
        let label = match cap {
            Some(cap) => format!("{label}: {} / {}", human_bytes(used), human_bytes(cap)),
            None => format!("{label}: {}", human_bytes(used)),
        };
        frame.render_widget(
            Gauge::default()
                .gauge_style(Style::default().fg(color))
                .ratio(cap.map_or(0.0, |cap| ratio(used, cap)))
                .label(label),
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
            spill_scope_line(model.telemetry.capacity.spill_on_query_path),
        ]),
        rates,
    );
}

/// Describe's columns in header order, each with the most cells it may take.
/// A cell longer than that is fitted to it before anything is measured — a
/// number rounded, never cut (#18), text marked with `…` — because ratatui
/// clips a cell that overflows its column, and for a number that silently
/// changes its value.
const DESCRIBE_COLUMNS: [(&str, usize); 8] = [
    ("column", 8),
    ("type", 8),
    ("min", 7),
    ("max", 7),
    ("nulls", 7),
    ("p25", 6),
    ("p50", 6),
    ("p99", 6),
];

/// Which columns Describe keeps when the panel cannot hold them all, most
/// important first, as indices into [`DESCRIBE_COLUMNS`]: the null count, the
/// range, the median, the type, then the outer percentiles. The column name
/// is not listed because it is always shown: a row of numbers is meaningless
/// without it.
const DESCRIBE_PRIORITY: [usize; 7] = [4, 2, 3, 6, 1, 7, 5];

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
/// anything longer is *rounded*: of the fixed-point and exponent forms that
/// fit, the one closest to the value, fixed-point with the most decimals on a
/// tie. Taking the first fixed-point form that fit, as this did, rounded a
/// small value to the decimals there was room for — `-0.000012345678` in
/// seven columns printed `-0`, a different number rather than a shorter one
/// (#122); it is `-1.2e-5` now. A form as far from the value as zero is never
/// taken: with no room for any other, the number is cut visibly, with `…`.
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
    let error = |candidate: &String| {
        candidate.parse::<f64>().map_or(f64::INFINITY, |shown| {
            if value == 0.0 {
                shown.abs()
            } else {
                ((shown - value) / value).abs()
            }
        })
    };
    let fixed = (0..=3)
        .rev()
        .map(|precision| trim_zeros(&format!("{value:.precision$}")));
    let exponent = (0..=2)
        .rev()
        .map(|precision| format!("{value:.precision$e}"));
    fixed
        .chain(exponent)
        .filter(|candidate| candidate.chars().count() <= width)
        // Zero is 100% off any value but zero.
        .filter(|candidate| value == 0.0 || error(candidate) < 1.0)
        // `min_by` keeps the first of equal candidates.
        .min_by(|a, b| error(a).total_cmp(&error(b)))
        .unwrap_or_else(|| fit_text(raw, width))
}

fn render_describe(state: &AppState, model: &DashboardModel, frame: &mut Frame<'_>, area: Rect) {
    let cells: Vec<[String; 8]> = model.profiles.iter().map(describe_cells).collect();
    let inside = Block::default().borders(Borders::ALL).inner(area).width;
    let shown = describe_columns(&cells, usize::from(inside));
    let hidden: Vec<&str> = DESCRIBE_COLUMNS
        .iter()
        .enumerate()
        .filter(|(index, _)| !shown.iter().any(|(column, _)| column == index))
        .map(|(_, (header, _))| *header)
        .collect();
    let title = if hidden.is_empty() {
        "Describe".to_owned()
    } else {
        format!("Describe (hidden: {})", hidden.join(" "))
    };
    let header = Row::new(
        shown
            .iter()
            .map(|&(column, _)| Cell::from(DESCRIBE_COLUMNS[column].0)),
    )
    .style(Style::default().add_modifier(Modifier::BOLD));
    let rows = cells.iter().map(|row| {
        Row::new(
            shown
                .iter()
                .map(|&(column, _)| Cell::from(row[column].as_str())),
        )
    });
    let widths = shown.iter().map(|&(_, width)| Constraint::Length(width));
    frame.render_widget(
        Table::new(rows, widths)
            .header(header)
            .column_spacing(1)
            .block(panel_block(&title, Panel::Describe, state.panel())),
        area,
    );
}

/// One profile's cells in [`DESCRIBE_COLUMNS`] order, each fitted to its
/// column's cap.
fn describe_cells(profile: &ColumnProfile) -> [String; 8] {
    let cap = DESCRIBE_COLUMNS.map(|(_, cap)| cap);
    [
        fit_text(&profile.name, cap[0]),
        fit_text(&profile.data_type, cap[1]),
        fit_number(&profile.min, cap[2]),
        fit_number(&profile.max, cap[3]),
        fit_number(&profile.null_count.to_string(), cap[4]),
        fit_number(&profile.p25, cap[5]),
        fit_number(&profile.p50, cap[6]),
        fit_number(&profile.p99, cap[7]),
    ]
}

/// Which Describe columns fit in `width` cells, and how wide each is, as
/// `(index into DESCRIBE_COLUMNS, width)` in header order.
///
/// The table used to ask for 62 cells whatever it held, which only a
/// terminal 128 columns wide could give it. Narrower, ratatui squeezed every
/// column below the width its cells had been fitted to and cut them with no
/// mark: at 80x24 the demo's 76923 nulls printed as `769`, and the README's
/// 1M-row table's largest `id`, 999999, as `9999` (#122). A column is now as
/// wide as its header or its widest cell, and when they do not all fit, the
/// panel keeps whole columns in [`DESCRIBE_PRIORITY`] order and the title
/// names the rest: a value is never cut to make room.
///
/// Each column is kept if it still fits once every more important one has
/// been placed. Dropping the least important columns until the rest fit
/// would hide one that has room: at 80x24 the demo's p25 (four cells) would
/// go before p99 (six), and then p99 too, though without p99 it fits.
fn describe_columns(cells: &[[String; 8]], width: usize) -> Vec<(usize, u16)> {
    let natural: [usize; 8] = std::array::from_fn(|column| {
        cells
            .iter()
            .map(|row| row[column].chars().count())
            .fold(DESCRIBE_COLUMNS[column].0.len(), usize::max)
    });
    let mut shown = vec![0];
    let mut used = natural[0];
    for column in DESCRIBE_PRIORITY {
        // One cell between columns, as `Table::column_spacing(1)` draws it.
        let needs = natural[column] + 1;
        if used + needs <= width {
            shown.push(column);
            used += needs;
        }
    }
    shown.sort_unstable();
    shown
        .into_iter()
        .map(|column| (column, u16::try_from(natural[column]).unwrap_or(u16::MAX)))
        .collect()
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

    /// The first fixed-point form that fit used to win, and for a small value
    /// that is zero: `-0.000012345678` in seven columns printed `-0`.
    #[test]
    fn a_small_value_is_never_rounded_to_zero() {
        assert_eq!(fit_number("-0.000012345678", 7), "-1.2e-5");
        assert_eq!(fit_number("0.000012345678", 7), "1.23e-5");
        assert_eq!(fit_number("0.0005000000001", 5), "5e-4");
        // No exponent form fits in four: the cut is marked instead.
        assert_eq!(fit_number("-0.000012345678", 4), "-0.…");
        for raw in ["-0.000012345678", "0.000012345678", "0.00049", "-0.0005"] {
            for width in 4..=8 {
                let cell = fit_number(raw, width);
                assert_ne!(num(&cell), 0.0, "{raw:?} in {width} columns is {cell:?}");
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
    fn every_describe_column_but_the_name_has_a_priority_once() {
        let mut order = DESCRIBE_PRIORITY.to_vec();
        order.sort_unstable();
        assert_eq!(order, (1..DESCRIBE_COLUMNS.len()).collect::<Vec<_>>());
    }

    fn row(cells: [&str; 8]) -> [String; 8] {
        cells.map(str::to_owned)
    }

    #[test]
    fn describe_columns_are_as_wide_as_their_widest_cell() {
        let cells = [row([
            "v", "Float64", "0.0", "24.0", "76923", "6.0", "12.0", "23.75",
        ])];
        let all: Vec<(usize, u16)> = [6, 7, 3, 4, 5, 3, 4, 5].into_iter().enumerate().collect();
        // 37 cells of content and seven gaps.
        assert_eq!(describe_columns(&cells, 44), all);
        // One cell short, and the least important column goes.
        let without_p25: Vec<(usize, u16)> = all.iter().copied().filter(|&(c, _)| c != 5).collect();
        assert_eq!(describe_columns(&cells, 43), without_p25);
    }

    #[test]
    fn describe_keeps_columns_by_priority_and_always_the_name() {
        let cells = [row([
            "v", "Float64", "0.0", "24.0", "76923", "6.0", "12.0", "23.75",
        ])];
        let names = |width| {
            describe_columns(&cells, width)
                .into_iter()
                .map(|(column, _)| DESCRIBE_COLUMNS[column].0)
                .collect::<Vec<_>>()
        };
        // The 80x24 panel: 38 cells inside the border. p99 (six cells with
        // its gap) does not fit; p25 (four) then does, and is shown.
        assert_eq!(
            names(38),
            ["column", "type", "min", "max", "nulls", "p25", "p50"]
        );
        assert_eq!(names(29), ["column", "min", "max", "nulls", "p50"]);
        assert_eq!(names(0), ["column"]);
    }

    #[test]
    fn ratio_is_clamped() {
        assert_eq!(ratio(1, 0), 0.0);
        assert_eq!(ratio(5, 4), 1.0);
        assert!((ratio(1, 4) - 0.25).abs() < f64::EPSILON);
    }
}
