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

/// Describe's header, in the order the panel draws its columns.
const DESCRIBE_HEADER: [&str; 8] = ["column", "type", "min", "max", "nulls", "p25", "p50", "p99"];

/// The Describe panel as drawn: its title, and each profile row as
/// `(header, cell)` pairs.
struct Describe {
    title: String,
    headers: Vec<String>,
    rows: Vec<Vec<(String, String)>>,
}

/// Reads the Describe panel back off a rendered buffer. A cell runs from
/// where its header starts to where the next one does, so an empty cell stays
/// empty instead of shifting the rest of its row left.
fn describe_panel(width: u16, height: u16, model: &DashboardModel) -> (Describe, String) {
    let state = AppState::new(model.plan().len());
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|f| render(&state, model, f)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    // Describe sits in the right half; one char per cell (no wide glyphs in
    // these fixtures).
    let right = |y: u16| -> Vec<char> {
        (width / 2..width)
            .map(|x| {
                let symbol = buffer.cell((x, y)).unwrap().symbol();
                symbol.chars().next().unwrap_or(' ')
            })
            .collect()
    };
    let top = (0..height)
        .find(|&y| {
            right(y)
                .iter()
                .collect::<String>()
                .starts_with("┌ Describe")
        })
        .unwrap_or_else(|| panic!("no Describe panel at {width}x{height}"));
    let title_row = right(top);
    let title: String = title_row[1..]
        .iter()
        .take_while(|&&c| c != '─')
        .collect::<String>()
        .trim()
        .to_owned();
    let inner = |y: u16| -> Vec<char> {
        let line = right(y);
        line[1..line.len() - 1].to_vec()
    };
    let header = inner(top + 1);
    let starts: Vec<usize> = (0..header.len())
        .filter(|&i| header[i] != ' ' && (i == 0 || header[i - 1] == ' '))
        .collect();
    let cell = |line: &[char], i: usize| -> String {
        let end = starts.get(i + 1).copied().unwrap_or(line.len());
        line[starts[i]..end]
            .iter()
            .collect::<String>()
            .trim()
            .to_owned()
    };
    let headers: Vec<String> = (0..starts.len()).map(|i| cell(&header, i)).collect();
    let mut rows = Vec::new();
    for y in top + 2..height {
        let line = right(y);
        if line[0] == '└' {
            break;
        }
        let line = inner(y);
        if line.iter().all(|&c| c == ' ') {
            break;
        }
        rows.push(
            (0..starts.len())
                .map(|i| (headers[i].clone(), cell(&line, i)))
                .collect(),
        );
    }
    let text = terminal.backend().to_string();
    (
        Describe {
            title,
            headers,
            rows,
        },
        text,
    )
}

/// What a Describe cell may show for `raw`: the value itself, a number
/// rounded to two significant digits or more (within 5%: `-1.2e-5` for
/// `-0.000012345678` is the most seven columns hold), or text visibly cut
/// with `…`. Never a number with digits cut off — `769` for 76923 — which is
/// what a column squeezed below the width its cells were fitted to printed
/// (#122), nor one rounded to zero.
fn faithful(raw: &str, cell: &str) -> bool {
    if cell == raw {
        return true;
    }
    if let (Ok(want), Ok(got)) = (raw.parse::<f64>(), cell.parse::<f64>()) {
        return if want == 0.0 {
            got == 0.0
        } else {
            ((got - want) / want).abs() <= 0.05
        };
    }
    cell.strip_suffix('…')
        .is_some_and(|kept| !kept.is_empty() && raw.starts_with(kept))
}

/// The raw value a profile holds for the Describe column `header`.
fn raw_value(profile: &ColumnProfile, header: &str) -> String {
    match header {
        "column" => profile.name.clone(),
        "type" => profile.data_type.clone(),
        "min" => profile.min.clone(),
        "max" => profile.max.clone(),
        "nulls" => profile.null_count.to_string(),
        "p25" => profile.p25.clone(),
        "p50" => profile.p50.clone(),
        "p99" => profile.p99.clone(),
        other => panic!("Describe drew a column named {other:?}"),
    }
}

fn profile(name: &str, data_type: &str, values: [&str; 5], null_count: u64) -> ColumnProfile {
    let [min, max, p25, p50, p99] = values.map(str::to_owned);
    ColumnProfile {
        name: name.into(),
        data_type: data_type.into(),
        min,
        max,
        null_count,
        p25,
        p50,
        p99,
    }
}

/// Values wider than any column: a long name, a nested type, a 2M-row
/// table's percentiles, more nulls than a column has room for, and a
/// vector's lexicographic minimum.
fn wide_values() -> DashboardModel {
    let mut model = demo_model();
    model.profiles = vec![
        profile(
            "passenger_count",
            "Int64",
            [
                "0",
                "1999999",
                "505700.81345880596",
                "996257.6842335836",
                "1979969.2416513609",
            ],
            123_456_789,
        ),
        profile(
            "emb",
            "FixedSizeList(8 x Float32)",
            [
                "[-0.99, 0.12, 0.5, 0.25, 0.75, 0.3, 0.1, 0.4]",
                "[0.99, -0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]",
                "",
                "",
                "",
            ],
            0,
        ),
        profile(
            "ratio",
            "Float64",
            ["-0.000012345678", "6.122512376708984", "0.5", "1", "6.1"],
            76_923,
        ),
    ];
    model
}

/// The profile `oxide tui` computes for the README's quickstart table
/// (`oxide gen-data --rows 1000000`, seed 42), as the profile query returned
/// it on 2026-11-17. At 80x24 the 0.2.0 panel drew `id` max as `9999`, `k`
/// nulls as `198` and `s`'s `Utf8View` as `Utf8`.
fn readme_table() -> DashboardModel {
    let mut model = demo_model();
    model.profiles = vec![
        profile(
            "id",
            "Int64",
            [
                "0",
                "999999",
                "249846.32028453206",
                "501364.6977001327",
                "989986.4407488042",
            ],
            0,
        ),
        profile(
            "k",
            "Int64",
            [
                "0",
                "99",
                "24.378284823745137",
                "49.32863782006587",
                "98.34708257311004",
            ],
            19_880,
        ),
        profile(
            "v",
            "Float64",
            [
                "0.0",
                "24.75",
                "6.112882594432291",
                "12.341245304777704",
                "24.593571868496074",
            ],
            24_830,
        ),
        profile("s", "Utf8View", ["s0", "s9", "", "", ""], 0),
        profile(
            "emb",
            "FixedSizeList(8 x Float32)",
            [
                "[-10.0, -9.97, -2.13, -8.93, -2.46, 2.3599997, 7.5599995, 9.1]",
                "[9.99, 9.99, -0.59000015, -6.79, 1.6300001, -4.07, 9.889999, -3.15]",
                "",
                "",
                "",
            ],
            0,
        ),
    ];
    model
}

/// Every number Describe draws is the true value, rounded at worst, at every
/// size a terminal is commonly opened at — and every column it cannot fit is
/// left out whole and named in its title, never squeezed until a value
/// changes. The old fixed 62-cell table fitted only from 128 columns up.
#[test]
fn describe_never_draws_a_number_it_cut() {
    for (fixture, model) in [
        ("demo", demo_model()),
        ("README table", readme_table()),
        ("wide values", wide_values()),
    ] {
        for (width, height) in [(80, 24), (100, 30), (120, 40), (160, 40), (200, 60)] {
            let (panel, text) = describe_panel(width, height, &model);
            let at = format!("{fixture} at {width}x{height}");
            let expected: Vec<&str> = DESCRIBE_HEADER
                .into_iter()
                .filter(|h| panel.headers.iter().any(|got| got == h))
                .collect();
            assert_eq!(
                panel.headers, expected,
                "{at}: a header was cut, renamed or reordered:\n{text}"
            );
            assert_eq!(panel.headers.first().map(String::as_str), Some("column"));
            let hidden: Vec<&str> = DESCRIBE_HEADER
                .into_iter()
                .filter(|h| !panel.headers.iter().any(|got| got == h))
                .collect();
            if hidden.is_empty() {
                assert_eq!(panel.title, "Describe", "{at}");
            } else {
                assert_eq!(
                    panel.title,
                    format!("Describe (hidden: {})", hidden.join(" ")),
                    "{at}: a column left out must be named:\n{text}"
                );
            }
            assert_eq!(panel.rows.len(), model.profiles.len(), "{at}:\n{text}");
            for (row, profile) in panel.rows.iter().zip(&model.profiles) {
                for (header, cell) in row {
                    let raw = raw_value(profile, header);
                    assert!(
                        faithful(&raw, cell),
                        "{at}: {} {header} is {raw:?} but Describe drew {cell:?}:\n{text}",
                        profile.name
                    );
                }
            }
        }
    }
}

/// At the size a terminal opens at, the demo's numbers are exact — `76923`
/// nulls, not the `769` the committed snapshots used to pin — and the two
/// lines that say what the other panels' numbers mean are on screen: the
/// CPU fallback count (#32), which needed 28 rows, and whether the spill
/// gauges can move at all (#25), which needed 26.
#[test]
fn at_80x24_the_numbers_are_exact_and_no_line_is_lost() {
    let model = demo_model();
    let (panel, text) = describe_panel(80, 24, &model);
    let v = &panel.rows[1];
    assert!(
        v.contains(&("nulls".into(), "76923".into())),
        "v's nulls: {v:?}\n{text}"
    );
    for (row, profile) in panel.rows.iter().zip(&model.profiles) {
        for (header, cell) in row {
            assert_eq!(
                cell,
                &raw_value(profile, header),
                "{} {header} is not exact at 80x24:\n{text}",
                profile.name
            );
        }
    }
    for line in ["H2D / D2H", "cpu fallback", "spill on the query path"] {
        assert!(
            text.contains(line),
            "`{line}` is not on screen at 80x24:\n{text}"
        );
    }
}
