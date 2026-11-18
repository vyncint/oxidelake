//! A query that loses a worker fails, naming the worker, instead of being
//! rolled back and run again (#148).
//!
//! When a worker goes (killed, stopped with `SIGTERM`, or silent past the
//! scheduler's `--worker-timeout`), Ballista 55 rolls back every stage that
//! read the worker's shuffle output and runs it again. That recovery is not
//! safe in 55.0.0: the rolled-back stage keeps its partial outputs, so some
//! rows can come back twice with exit 0 (fixed upstream by
//! apache/datafusion-ballista#2677, merged after 55.0.0 and not released),
//! and the tasks it cancels never return their slots, so the query can wait
//! for capacity that never comes back (see apache/datafusion-ballista#2539
//! and #2698). Correctness comes first: a query that held output on a lost
//! worker fails instead, and says which worker it lost.
//!
//! Every way Ballista removes a worker passes through the cluster state's
//! `remove_executor`, and Ballista posts the event that starts the rollback
//! only after that returns. The cluster state calls [`WorkerLoss::lost`]
//! there, which puts a cancellation of each affected query ahead of the
//! rollback in the scheduler's event queue, and the job state then reports
//! this module's reason where Ballista would say "Cancelled". A query whose
//! tasks on that worker had not finished any (just launched, or a launch
//! that failed) is left to Ballista: nothing of theirs was kept, so no stage
//! is rolled back, and those tasks simply run again elsewhere.
//!
//! The other way Ballista meets a lost worker, a shuffle fetch from it
//! failing, happens before the worker is removed; `--stage-max-failures`
//! (default 1) decides that one.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use async_trait::async_trait;
use ballista_core::error::Result;
use ballista_core::serde::protobuf::{JobStatus, SuccessfulTask, job_status, task_status};
use ballista_core::{JobId, JobStatusSubscriber};
use ballista_scheduler::cluster::memory::InMemoryJobState;
use ballista_scheduler::cluster::{JobState, JobStateEventStream};
use ballista_scheduler::scheduler_server::SchedulerServer;
use ballista_scheduler::state::execution_graph::{ExecutionGraph, ExecutionGraphBox};
use ballista_scheduler::state::execution_stage::ExecutionStage;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_proto::protobuf::{LogicalPlanNode, PhysicalPlanNode};

/// The scheduler `oxide-scheduler` runs.
pub(crate) type Scheduler = SchedulerServer<LogicalPlanNode, PhysicalPlanNode>;

/// How many failed queries' reasons are kept until their final status is
/// saved: bounded, so a reason whose query raced to an end of its own cannot
/// accumulate.
const REMEMBERED_REASONS: usize = 64;

/// The tracking issue every message points to.
pub(crate) const ISSUE: &str = "#148";

/// Fails the queries a lost worker held output for (see the module docs).
#[derive(Default)]
pub(crate) struct WorkerLoss {
    scheduler: OnceLock<Scheduler>,
    reasons: Mutex<VecDeque<(JobId, String)>>,
}

impl WorkerLoss {
    /// Hands over the scheduler once it exists; until then a lost worker
    /// cannot have held anything, and nothing is done.
    pub(crate) fn attach(&self, scheduler: Scheduler) {
        let _ = self.scheduler.set(scheduler);
    }

    /// `worker` (reachable at `address`) is being removed: cancel every
    /// running query that holds finished output on it, before Ballista's
    /// rollback can run.
    pub(crate) async fn lost(&self, worker: &str, address: &str) {
        let Some(scheduler) = self.scheduler.get() else {
            return;
        };
        let jobs = scheduler.state.task_manager.get_running_job_cache();
        for (job_id, info) in jobs.iter() {
            if !held_output(info.execution_graph.read().await.as_ref(), worker) {
                continue;
            }
            let reason = format!(
                "worker {worker} ({address}) was lost while this query ran, and it held shuffle \
                 output the query still needed. OxideLake fails the query rather than let \
                 Ballista 55 run those stages again: that recovery can return duplicated rows \
                 or never finish ({ISSUE}). Run the query again"
            );
            tracing::warn!(job = %job_id, "{reason}");
            self.remember(job_id.clone(), reason);
            if let Err(err) = scheduler.cancel_job(job_id.clone()).await {
                tracing::error!(job = %job_id, "could not fail the query: {err}");
            }
        }
    }

    fn remember(&self, job_id: JobId, reason: String) {
        let mut reasons = self.reasons.lock().unwrap_or_else(PoisonError::into_inner);
        if reasons.len() >= REMEMBERED_REASONS {
            reasons.pop_front();
        }
        reasons.push_back((job_id, reason));
    }

    /// Why this module failed `job_id`, once: the job state reports it in
    /// place of Ballista's "Cancelled".
    fn take_reason(&self, job_id: &JobId) -> Option<String> {
        let mut reasons = self.reasons.lock().unwrap_or_else(PoisonError::into_inner);
        let index = reasons.iter().position(|(id, _)| id == job_id)?;
        reasons.remove(index).map(|(_, reason)| reason)
    }
}

/// Whether a task of `graph` finished on `worker`, so that losing the worker
/// loses output a stage may read.
fn held_output(graph: &dyn ExecutionGraph, worker: &str) -> bool {
    graph.stages().values().any(|stage| {
        let tasks = match stage {
            ExecutionStage::Running(stage) => &stage.task_infos,
            ExecutionStage::Successful(stage) => &stage.task_infos,
            _ => return false,
        };
        tasks.iter().any(|task| {
            matches!(
                &task.task_status,
                task_status::Status::Successful(SuccessfulTask { executor_id, .. })
                    if executor_id == worker
            )
        })
    })
}

fn is_failed(status: &JobStatus) -> bool {
    matches!(status.status, Some(job_status::Status::Failed(_)))
}

/// Ballista's in-memory job state, saving a query that [`WorkerLoss`] failed
/// with the reason it failed rather than "Cancelled". Everything else is
/// Ballista's own.
pub(crate) struct ReportingJobState {
    inner: InMemoryJobState,
    loss: Arc<WorkerLoss>,
}

impl ReportingJobState {
    pub(crate) fn new(inner: InMemoryJobState, loss: Arc<WorkerLoss>) -> Self {
        Self { inner, loss }
    }
}

#[async_trait]
impl JobState for ReportingJobState {
    fn accept_job(&self, job_id: &JobId, job_name: &str, queued_at: u64) -> Result<()> {
        self.inner.accept_job(job_id, job_name, queued_at)
    }

    fn pending_job_number(&self) -> usize {
        self.inner.pending_job_number()
    }

    async fn submit_job(
        &self,
        job_id: JobId,
        graph: &ExecutionGraphBox,
        subscriber: Option<JobStatusSubscriber>,
    ) -> Result<()> {
        self.inner.submit_job(job_id, graph, subscriber).await
    }

    async fn get_jobs(&self) -> Result<HashSet<JobId>> {
        self.inner.get_jobs().await
    }

    async fn get_all_jobs(&self) -> Result<HashSet<JobId>> {
        self.inner.get_all_jobs().await
    }

    async fn get_job_status(&self, job_id: &JobId) -> Result<Option<JobStatus>> {
        self.inner.get_job_status(job_id).await
    }

    async fn get_execution_graph(&self, job_id: &JobId) -> Result<Option<ExecutionGraphBox>> {
        self.inner.get_execution_graph(job_id).await
    }

    async fn save_job(&self, job_id: &JobId, graph: &ExecutionGraphBox) -> Result<()> {
        if is_failed(graph.status())
            && let Some(reason) = self.loss.take_reason(job_id)
        {
            let mut graph = graph.cloned();
            graph.fail_job(reason);
            return self.inner.save_job(job_id, &graph).await;
        }
        self.inner.save_job(job_id, graph).await
    }

    async fn fail_unscheduled_job(&self, job_id: &JobId, reason: String) -> Result<()> {
        self.inner.fail_unscheduled_job(job_id, reason).await
    }

    async fn remove_job(&self, job_id: &JobId) -> Result<()> {
        self.inner.remove_job(job_id).await
    }

    async fn try_acquire_job(&self, job_id: &JobId) -> Result<Option<ExecutionGraphBox>> {
        self.inner.try_acquire_job(job_id).await
    }

    async fn job_state_events(&self) -> Result<JobStateEventStream> {
        self.inner.job_state_events().await
    }

    async fn create_or_update_session(
        &self,
        session_id: &str,
        config: &SessionConfig,
    ) -> Result<Arc<SessionContext>> {
        self.inner
            .create_or_update_session(session_id, config)
            .await
    }

    async fn remove_session(&self, session_id: &str) -> Result<()> {
        self.inner.remove_session(session_id).await
    }

    fn produce_config(&self) -> SessionConfig {
        self.inner.produce_config()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn a_reason_is_given_once_and_the_kept_ones_are_bounded() {
        let loss = WorkerLoss::default();
        loss.remember("job-1".into(), "lost w2".to_owned());
        assert_eq!(
            loss.take_reason(&"job-1".into()).as_deref(),
            Some("lost w2")
        );
        assert_eq!(loss.take_reason(&"job-1".into()), None);
        for i in 0..2 * REMEMBERED_REASONS {
            loss.remember(format!("job-{i}").as_str().into(), format!("r{i}"));
        }
        assert_eq!(loss.reasons.lock().unwrap().len(), REMEMBERED_REASONS);
        assert_eq!(loss.take_reason(&"job-0".into()), None, "the oldest went");
        let last = 2 * REMEMBERED_REASONS - 1;
        assert!(
            loss.take_reason(&format!("job-{last}").as_str().into())
                .is_some()
        );
    }
}
