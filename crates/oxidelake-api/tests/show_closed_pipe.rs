//! `OxideFrame::show()` into a reader that goes away (#125).
//!
//! `… | head -1` is the most ordinary thing a shell does to a program that
//! prints a table, and `show()` used to panic on it: it was DataFusion's
//! `DataFrame::show`, whose `println!` panics when stdout is a closed pipe —
//! the crash #34 fixed in the `oxide` binary and left in the library.
//!
//! This is its own test binary, with nothing beside it that forks. A process
//! that another test forks while this one holds the pipe's read end keeps a
//! copy until it execs, and no write fails with `EPIPE` while a reader is
//! left. With a table this size that only delays the result (a sibling that
//! held the read end for 2 s delayed the panic, and the fix's clean stop, by
//! 2 s), but a closed-pipe test should not depend on its neighbours' timing.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use oxidelake_api::prelude::*;

/// Set in the child the test spawns: the process whose stdout is closed.
const CHILD: &str = "OXIDELAKE_TEST_SHOW_CHILD";
const NAME: &str = "show_into_a_closed_pipe_returns_instead_of_panicking";

/// Prints a table far larger than a pipe buffer, says on stderr what `show`
/// returned, and exits with that answer before the test harness can print
/// anything more into the closed pipe.
async fn child() {
    let session = OxideSession::local().unwrap();
    // About 1.3 MB of table: a pipe buffer holds 64 KiB, so `show` is still
    // writing when the reader goes.
    let frame = session
        .sql_frame("SELECT value AS id, value * 3 AS k FROM generate_series(1, 50000)")
        .await
        .unwrap();
    match frame.show().await {
        Ok(()) => {
            eprintln!("show returned Ok");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("show returned Err: {e}");
            std::process::exit(3);
        }
    }
}

/// The reader takes the table's first line and goes, as `head -1` does; the
/// writer must stop cleanly, with `Ok`, as `oxide sql` does (#34).
#[tokio::test]
async fn show_into_a_closed_pipe_returns_instead_of_panicking() {
    if std::env::var_os(CHILD).is_some() {
        child().await;
        return;
    }
    // `--nocapture`, or the harness would capture the old `println!` into a
    // buffer and the closed pipe would never be written to; `--quiet`, so the
    // harness prints nothing between its first line and the table.
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([NAME, "--exact", "--nocapture", "--quiet"])
        .env(CHILD, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    let mut saw_table = false;
    while stdout.read_line(&mut line).unwrap() > 0 {
        if line.starts_with("+-") {
            saw_table = true;
            break;
        }
        line.clear();
    }
    drop(stdout);

    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(saw_table, "the child printed no table:\n{stderr}");
    assert!(
        !stderr.contains("panicked"),
        "a closed pipe is not a crash:\n{stderr}"
    );
    assert!(
        stderr.contains("show returned Ok"),
        "a reader that goes away is a clean stop:\n{stderr}"
    );
    assert_eq!(out.status.code(), Some(0), "{stderr}");
}

/// `show` stays spawnable. The stdout lock it writes through is not `Send`,
/// so it is taken after the query's last await; held across the query, it
/// would stop every `tokio::spawn(frame.show())` from compiling. No process
/// is forked here, so this shares the binary safely.
#[test]
fn show_is_send() {
    fn spawnable<F: std::future::Future + Send>(_: F) {}
    let session = OxideSession::local().unwrap();
    let frame = OxideFrame::new(session.ctx().read_empty().unwrap());
    spawnable(frame.show());
}
