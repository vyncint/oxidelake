//! `termlens-cli` — the harness this suite already uses, at a shell prompt.
//!
//! The rest of the suite asks whether the dashboard draws the right thing.
//! This asks what a maintainer does *after* it draws the wrong thing: point
//! the tool at the real binary, read a committed `.snap` back, and diff two
//! of them to see what a keystroke changed. Those `.snap` files are the
//! repository's own record of what OxideLake looks like, and nothing
//! verified that anything except `cargo insta` could still read them.
//!
//! Every test here is `#[ignore]`d. They need `termlens-cli` on the machine,
//! and these crates are published — a `cargo test` that quietly
//! `cargo install`s something is a surprise a contributor should not get.
//! CI runs them by name (`make test-termlens-cli`), and the version it
//! installs is the one `Cargo.lock` names, so the tool and the library can
//! never be two different releases.
//!
//! ```sh
//! cargo test -p oxidelake-tui --test termlens_cli -- --ignored
//! TERMLENS_CLI=$(command -v termlens) cargo test -p oxidelake-tui --test termlens_cli -- --ignored
//! ```
//!
//! Every test name starts with `termlens_cli_` on purpose: the macOS Metal
//! lane runs the whole `-p oxidelake-tui` graph a second time with
//! `-- --ignored` for the on-device conformance suite, and skips these by
//! that prefix rather than installing a tool from the network mid-gate.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

use termlens::Screen;

/// The three screens this repository has committed, as `termlens diff` and
/// `termlens render` see them: an insta `.snap` is the snapshot text format
/// with a header, which the tool reads.
const SNAPSHOTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/");
/// The first frame, text only.
const FIRST: &str = "tui_pty_test__demo_renders_navigates_resizes_and_quits.snap";
/// The same frame after `↓` — so a diff of the two is what `↓` did.
const AFTER_DOWN: &str = "tui_pty_test__demo_renders_navigates_resizes_and_quits-2.snap";
/// The first frame again, with its `styles:` block.
const STYLED: &str = "tui_pty_test__the_first_frame_carries_its_colours_too.snap";

fn snapshot(name: &str) -> PathBuf {
    PathBuf::from(SNAPSHOTS).join(name)
}

/// The termlens version this suite is measured against, read from the
/// lockfile so the tool and the library can never be two different releases.
fn version_under_test() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        let lock =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"))
                .expect("Cargo.lock is committed at the workspace root");
        let mut lines = lock.lines();
        while let Some(line) = lines.next() {
            if line.trim() == "name = \"termlens\"" {
                for next in lines.by_ref() {
                    if let Some(rest) = next.trim().strip_prefix("version = \"") {
                        return rest.trim_end_matches('"').to_owned();
                    }
                }
            }
        }
        panic!("no termlens version in Cargo.lock");
    })
}

/// The workspace `target/`, which is where an installed tool belongs: not
/// `crates/oxidelake-tui/target/`, which nothing else in this workspace uses
/// and `make clean` would miss.
fn target_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CARGO_TARGET_DIR") {
        return PathBuf::from(dir);
    }
    // The test binary is <target>/<profile>/deps/<name>.
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().nth(3).map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target")))
}

/// The `termlens` binary: `$TERMLENS_CLI` if the environment provides one,
/// otherwise installed once into `target/` at the version under test.
fn cli() -> &'static PathBuf {
    static BIN: OnceLock<PathBuf> = OnceLock::new();
    BIN.get_or_init(|| {
        if let Some(given) = std::env::var_os("TERMLENS_CLI") {
            return PathBuf::from(given);
        }
        let root = target_dir().join("termlens-cli");
        let bin = root
            .join("bin")
            .join(format!("termlens{}", std::env::consts::EXE_SUFFIX));
        let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
        let status = Command::new(cargo)
            .args(["install", "termlens-cli", "--version", version_under_test()])
            .args(["--locked", "--root"])
            .arg(&root)
            .status()
            .expect("cargo install termlens-cli");
        assert!(
            status.success(),
            "cargo install termlens-cli --version {} failed. It is published \
             alongside the library; if this version of termlens exists on \
             crates.io and termlens-cli does not, the two releases went out \
             of lockstep.",
            version_under_test()
        );
        bin
    })
}

fn run(args: &[&str]) -> Output {
    Command::new(cli())
        .args(args)
        .output()
        .expect("run termlens")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A `.snap` without insta's `---` header: the screen the file records.
fn snapshot_body(name: &str) -> String {
    let raw = std::fs::read_to_string(snapshot(name)).expect("a committed snapshot");
    let mut lines = raw.lines();
    assert_eq!(lines.next(), Some("---"), "insta writes a header first");
    let body: Vec<&str> = lines.skip_while(|line| *line != "---").skip(1).collect();
    format!("{}\n", body.join("\n"))
}

#[test]
#[ignore = "needs termlens-cli; run with --ignored (CI does)"]
fn termlens_cli_is_the_release_the_library_comes_from() {
    let out = run(&["--version"]);
    assert!(out.status.success());
    assert_eq!(
        stdout(&out).trim(),
        format!("termlens {}", version_under_test()),
        "the installed CLI is not the version this suite tests against"
    );
}

/// `inspect` points the harness at a binary without writing a test, which is
/// the first thing a contributor reaches for — and the first thing the
/// README of a TUI should be able to promise. Pointed at the dashboard it
/// has to survive raw mode, the alternate screen and DEC 2026 frames.
#[test]
#[ignore = "needs termlens-cli; run with --ignored (CI does)"]
fn termlens_cli_inspect_drives_the_real_dashboard() {
    let out = Command::new(cli())
        .args(["inspect", "--size", "100x30", "--idle", "600"])
        .arg(env!("CARGO_BIN_EXE_oxidelake-tui-demo"))
        .output()
        .expect("run inspect");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let screen = stdout(&out);
    assert!(screen.contains("size: 100x30"), "{screen}");
    for panel in ["Plan DAG", "Inspector", "Telemetry", "Describe"] {
        assert!(screen.contains(panel), "no {panel} panel:\n{screen}");
    }
    assert!(screen.contains("[CUDA]"), "the plan tags its operators");
    // The trailer goes to **stderr** since termlens 0.11 (termlens#340), so
    // what stdout carries is a saved screen the tool reads back unedited.
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("still running at the deadline"),
        "the dashboard is a TUI, so inspect reports the deadline rather than \
         an exit — a binary that fell out of the event loop would say so here: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !screen.contains("--- "),
        "and stdout is the screen alone:\n{screen}"
    );
}

/// The committed screens are still readable, and still carry their colours.
///
/// `render --text` is the round trip: what the tool prints must be the file,
/// exactly. The image renderings are what a bug report or a step summary
/// shows, and for this dashboard the colour *is* the information — a green
/// `[CUDA]` beside a blue `[CPU]` is how a reader sees where an operator
/// ran.
#[test]
#[ignore = "needs termlens-cli; run with --ignored (CI does)"]
fn termlens_cli_renders_the_committed_screens_with_their_colours() {
    let styled = snapshot(STYLED);
    let path = styled.to_str().unwrap();

    let text = run(&["render", "--text", path]);
    assert!(text.status.success());
    assert_eq!(
        stdout(&text),
        snapshot_body(STYLED),
        "`termlens render --text` no longer reproduces the committed screen"
    );

    // What the screen itself says the tags are drawn in, read with the same
    // parser the CLI uses, so the expectation below is measured rather than
    // remembered.
    let screen = Screen::parse(&snapshot_body(STYLED)).expect("the snapshot parses");
    let (row, col) = screen.find("[CUDA]").expect("a tagged operator");
    assert_eq!(
        screen.cell(row, col).unwrap().style().fg,
        termlens::Color::Indexed(2),
        "the CUDA tag is palette green (render.rs: Color::Green)"
    );

    // `#0dbc79` and `#2472c8` are how termlens renders palette entries 2 and
    // 4 into an image. Tying the colour to the text in one assertion is the
    // point: an SVG with the right words in the wrong colours is exactly the
    // regression a text snapshot cannot see.
    let svg = run(&["render", "--svg", path]);
    assert!(svg.status.success());
    let body = stdout(&svg);
    assert_eq!(
        body.matches(r##"fill="#0dbc79" xml:space="preserve" font-weight="bold">[CUDA]</text>"##)
            .count(),
        screen.find_all("[CUDA]").len(),
        "every [CUDA] tag should be green in the image:\n{body}"
    );
    assert_eq!(
        body.matches(r##"fill="#2472c8" xml:space="preserve" font-weight="bold">[CPU]</text>"##)
            .count(),
        screen.find_all("[CPU]").len(),
        "and every [CPU] tag blue:\n{body}"
    );

    let html = run(&["render", "--html", path]);
    assert!(html.status.success());
    let body = stdout(&html);
    assert!(body.contains("color:#0dbc79"), "the HTML keeps the palette");
    assert!(body.contains("[CUDA]"), "and the text:\n{body}");
}

/// The workflow a maintainer runs when a frame changed and they want to know
/// exactly how — on this repository's own files. The three exit codes are
/// the part a script reads: 0 same, 1 different, 2 could not be read.
#[test]
#[ignore = "needs termlens-cli; run with --ignored (CI does)"]
fn termlens_cli_diffs_this_repositorys_own_snapshots() {
    let first = snapshot(FIRST);
    let first = first.to_str().unwrap();
    let after_down = snapshot(AFTER_DOWN);
    let after_down = after_down.to_str().unwrap();

    let same = run(&["diff", "--color", "never", first, first]);
    assert_eq!(same.status.code(), Some(0), "a screen equals itself");
    assert!(stdout(&same).contains("no difference"));

    // These two files are the frames either side of `↓` in
    // `demo_renders_navigates_resizes_and_quits`, so the diff is that
    // keystroke, spelled out.
    let moved = run(&["diff", "--color", "never", first, after_down]);
    assert_eq!(
        moved.status.code(),
        Some(1),
        "↓ selects another operator, so the two frames differ"
    );
    let rendered = stdout(&moved);
    assert!(rendered.contains("size: 80x24"), "the header:\n{rendered}");
    assert!(
        rendered.contains("rows unchanged"),
        "and a count of what did not move:\n{rendered}"
    );
    assert!(
        rendered.contains("1000000"),
        "the Inspector's new row count is what ↓ produced:\n{rendered}"
    );

    // Exit 2 is the tool failing, not a difference: a script that treats
    // "could not read the file" as "the screens differ" reports a phantom
    // regression on every typo.
    let not_a_screen = run(&[
        "diff",
        "--color",
        "never",
        first,
        concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"),
    ]);
    assert_eq!(
        not_a_screen.status.code(),
        Some(2),
        "a manifest is not a screen"
    );
    assert!(
        String::from_utf8_lossy(&not_a_screen.stderr).contains("could not parse a saved screen"),
        "and it says so"
    );
}
