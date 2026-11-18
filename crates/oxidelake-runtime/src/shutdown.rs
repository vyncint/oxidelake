//! Stopping a scheduler or a worker without dropping work on the floor (#31).
//!
//! Ballista's executor already handles `SIGTERM` and `SIGINT`: it stops taking
//! tasks, tells the scheduler (which re-runs the tasks it was running
//! elsewhere), waits for those tasks, and returns. It does not bound that
//! wait, so a worker whose task does not finish ignores `SIGTERM` until it is
//! killed. Ballista's scheduler handles neither signal: `SIGTERM` killed it
//! with every query it was running, and their clients saw a dropped
//! connection. This module supplies the signal, the bound, and the
//! scheduler's half: refuse new queries, wait for the running ones, exit.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ballista_core::serde::protobuf::job_status;
use ballista_scheduler::cluster::JobState;
use ballista_scheduler::scheduler_server::SessionBuilder;
use datafusion::error::DataFusionError;
use tokio::time::Instant;

/// What a client is told when it sends a query to a scheduler that is
/// shutting down.
pub(crate) const SHUTTING_DOWN: &str = "this scheduler is shutting down: it finishes the \
     queries it is running and takes no new ones; send this query to another scheduler, or \
     again once this one has restarted";

/// How long a drained scheduler stays up after its last query finished, so
/// that query's client learns it did: the final status reaches a client over
/// its status stream, or at its next poll, 50 ms apart.
const LINGER: Duration = Duration::from_secs(1);

/// How often a draining scheduler looks at its queries.
const POLL: Duration = Duration::from_millis(100);

/// How a scheduler or a worker stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stopped {
    /// It finished what it was running, or had nothing running.
    Drained,
    /// It stopped with work still running, because the drain timeout passed
    /// or a second signal came. A worker's scheduler re-runs that work
    /// elsewhere; a scheduler's running queries fail.
    Abandoned,
}

/// The signals that ask a process to stop: `SIGTERM` (an orchestrator,
/// `kill`) and `SIGINT` (Ctrl-C).
///
/// Listening starts when this is made, not when it is first awaited, so a
/// signal that arrives while the scheduler or worker is still starting is
/// held rather than killing the process.
#[derive(Debug)]
pub struct ShutdownSignal {
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
}

impl ShutdownSignal {
    /// Starts listening. Must be called inside a Tokio runtime.
    pub fn listen() -> io::Result<Self> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            Ok(Self {
                terminate: signal(SignalKind::terminate())?,
                interrupt: signal(SignalKind::interrupt())?,
            })
        }
        #[cfg(not(unix))]
        {
            Ok(Self {})
        }
    }

    /// Resolves on the next signal and names it.
    pub async fn recv(&mut self) -> &'static str {
        #[cfg(unix)]
        {
            tokio::select! {
                _ = self.terminate.recv() => "SIGTERM",
                _ = self.interrupt.recv() => "SIGINT",
            }
        }
        #[cfg(not(unix))]
        {
            // Ctrl-C is the one stop request Windows delivers this way; the
            // project does not support Windows, but must still compile there.
            match tokio::signal::ctrl_c().await {
                Ok(()) => "Ctrl-C",
                Err(_) => std::future::pending().await,
            }
        }
    }
}

/// Wraps `builder` so that, once `draining` is set, a new query fails at once
/// saying the scheduler is shutting down. The scheduler builds a session for
/// every query it is sent and for nothing a running query needs, so this
/// turns away new work without touching the queries being drained.
pub(crate) fn refusing_while_draining(
    builder: SessionBuilder,
    draining: Arc<AtomicBool>,
) -> SessionBuilder {
    Arc::new(move |config| {
        if draining.load(Ordering::SeqCst) {
            Err(DataFusionError::Execution(SHUTTING_DOWN.to_owned()))
        } else {
            builder(config)
        }
    })
}

/// How many queries `jobs` holds that have not finished: queued, being
/// planned, or running.
pub(crate) async fn unfinished_jobs(jobs: &dyn JobState) -> usize {
    // Ballista's in-memory `get_jobs` returns the *completed* jobs, so the
    // unfinished ones are counted from every job's status instead.
    let Ok(ids) = jobs.get_all_jobs().await else {
        return 0;
    };
    let mut unfinished = 0;
    for id in ids {
        let finished = matches!(
            jobs.get_job_status(&id).await,
            Ok(Some(ballista_core::serde::protobuf::JobStatus {
                status: Some(job_status::Status::Successful(_) | job_status::Status::Failed(_)),
                ..
            }))
        );
        if !finished {
            unfinished += 1;
        }
    }
    unfinished
}

/// Waits until `jobs` has had no unfinished query for [`LINGER`], or until
/// `timeout` has passed, and returns how many were still unfinished.
pub(crate) async fn drain(jobs: &dyn JobState, timeout: Duration) -> usize {
    let deadline = Instant::now() + timeout;
    let mut idle_since: Option<Instant> = None;
    loop {
        let unfinished = unfinished_jobs(jobs).await;
        let now = Instant::now();
        if unfinished == 0 {
            let since = *idle_since.get_or_insert(now);
            if now.duration_since(since) >= LINGER {
                return 0;
            }
        } else {
            idle_since = None;
        }
        if now >= deadline {
            return unfinished;
        }
        tokio::time::sleep(POLL.min(deadline - now)).await;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ballista_scheduler::cluster::memory::InMemoryJobState;
    use datafusion::prelude::SessionConfig;

    fn job_state() -> InMemoryJobState {
        InMemoryJobState::new(
            "scheduler",
            Arc::new(ballista_core::utils::default_session_builder),
            Arc::new(SessionConfig::new),
        )
    }

    #[test]
    fn a_draining_scheduler_refuses_a_new_query_by_saying_so() {
        let draining = Arc::new(AtomicBool::new(false));
        let builder: SessionBuilder = Arc::new(ballista_core::utils::default_session_builder);
        let builder = refusing_while_draining(builder, Arc::clone(&draining));
        builder(SessionConfig::new()).expect("not draining yet");
        draining.store(true, Ordering::SeqCst);
        let err = builder(SessionConfig::new()).unwrap_err().to_string();
        assert!(err.contains("shutting down"), "{err}");
    }

    // Real time, with the lower bounds exact and the upper ones loose: Tokio's
    // paused clock needs its `test-util` feature, which this workspace does
    // not build.

    /// A queued query holds the drain until the timeout, and the count says
    /// so; once it is finished the drain ends after the linger.
    #[tokio::test]
    async fn the_drain_waits_for_an_unfinished_query_and_no_longer() {
        let jobs = job_state();
        jobs.accept_job(&"job-1".into(), "q", 0).unwrap();
        assert_eq!(unfinished_jobs(&jobs).await, 1);
        let started = Instant::now();
        assert_eq!(drain(&jobs, Duration::from_millis(400)).await, 1);
        let waited = started.elapsed();
        assert!(waited >= Duration::from_millis(400), "{waited:?}");
        assert!(waited < Duration::from_secs(5), "{waited:?}");

        jobs.fail_unscheduled_job(&"job-1".into(), "planning failed".to_owned())
            .await
            .unwrap();
        assert_eq!(
            unfinished_jobs(&jobs).await,
            0,
            "a failed query is finished"
        );
        let started = Instant::now();
        assert_eq!(drain(&jobs, Duration::from_secs(30)).await, 0);
        let waited = started.elapsed();
        assert!(waited >= LINGER, "{waited:?}");
        assert!(waited < Duration::from_secs(10), "{waited:?}");
    }

    #[tokio::test]
    async fn a_zero_drain_timeout_stops_at_once() {
        let jobs = job_state();
        jobs.accept_job(&"job-1".into(), "q", 0).unwrap();
        let started = Instant::now();
        assert_eq!(drain(&jobs, Duration::ZERO).await, 1);
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
