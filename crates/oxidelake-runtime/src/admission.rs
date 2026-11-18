//! Which workers a scheduler admits (#92).
//!
//! A plan is only safe on a worker of the scheduler's own build: the plan
//! codec's payloads are not self-describing, and `predict` changes what a plan
//! may reference ([`PLAN_FINGERPRINT`]). The codec refuses a payload from
//! another build when it decodes one, but on a cluster that happens inside a
//! task launch, where the client learns only that a task "could not be
//! prepared or was rejected by executor" and the reason stays in the worker's
//! log (under Ballista 54 the launch was retried for ever instead). Checking
//! once, when a worker registers, makes the mismatch what it is: a deployment
//! error, named at both ends, before any query is planned for that worker.
//!
//! Ballista's registration carries nothing free-form but the executor id, so
//! an `oxide-worker` puts its fingerprint there: `oxl<8 hex digits>-<uuid>`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use ballista_core::JobId;
use ballista_core::error::{BallistaError, Result};
use ballista_core::serde::protobuf::{ExecutorHeartbeat, executor_status};
use ballista_core::serde::scheduler::{ExecutorData, ExecutorMetadata};
use ballista_scheduler::cluster::memory::InMemoryClusterState;
use ballista_scheduler::cluster::{BoundTask, ClusterState, ClusterStateEventStream, ExecutorSlot};
use ballista_scheduler::config::TaskDistributionPolicy;
use ballista_scheduler::scheduler_server::SessionBuilder;
use ballista_scheduler::state::task_manager::JobInfoCache;
use datafusion::error::DataFusionError;
use oxidelake_planner::codec::{PLAN_FINGERPRINT, build_description};

/// Starts the build tag in an `oxide-worker`'s executor id.
const BUILD_TAG: &str = "oxl";

/// How many refusals a scheduler keeps for the message a query gets: enough
/// to name the most recent, bounded so a worker restarting in a loop cannot
/// grow the scheduler's memory.
const REMEMBERED_REFUSALS: usize = 8;

/// How every refusal starts, so a worker can tell its own refusal from the
/// other ways registration fails and add its side of the story.
pub(crate) const REFUSED: &str = "refused worker";

/// An executor id that carries this build's plan fingerprint.
pub(crate) fn tagged_executor_id(instance: &str) -> String {
    tagged(PLAN_FINGERPRINT, instance)
}

fn tagged(fingerprint: u32, instance: &str) -> String {
    format!("{BUILD_TAG}{fingerprint:08x}-{instance}")
}

/// The plan fingerprint an executor id carries, if it carries one.
pub(crate) fn fingerprint_of(executor_id: &str) -> Option<u32> {
    let (hex, _instance) = executor_id.strip_prefix(BUILD_TAG)?.split_once('-')?;
    // `from_str_radix` alone would also take a leading `+`.
    if hex.len() != 8 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(hex, 16).ok()
}

/// Why the worker `metadata` describes cannot join a scheduler of this
/// build, or `None` when it can.
fn refusal(metadata: &ExecutorMetadata) -> Option<String> {
    let worker = format!(
        "{REFUSED} {} ({}:{})",
        metadata.id, metadata.host, metadata.grpc_port
    );
    match fingerprint_of(&metadata.id) {
        Some(theirs) if theirs == PLAN_FINGERPRINT => None,
        Some(theirs) => Some(format!(
            "{worker}: it was built with plan fingerprint {theirs:#010x}, and this scheduler \
             is {}. A worker must be the same OxideLake release as its scheduler, built with \
             `predict` if and only if the scheduler is; rebuild the one that differs",
            build_description()
        )),
        None => Some(format!(
            "{worker}: its id carries no OxideLake build fingerprint, so it is a plain \
             Ballista executor or an `oxide-worker` older than 0.3.0, and this scheduler is \
             {}. Run the `oxide-worker` of the scheduler's release",
            build_description()
        )),
    }
}

fn is_active(heartbeat: &ExecutorHeartbeat) -> bool {
    matches!(
        heartbeat.status.as_ref().and_then(|s| s.status.as_ref()),
        Some(executor_status::Status::Active(_))
    )
}

/// Ballista's in-memory cluster state, admitting only workers of this build.
///
/// Every other method is Ballista's own: this is the in-memory state that
/// `BallistaCluster::new_from_config` would have built, with one check in
/// front of the two ways a worker's metadata enters it.
#[derive(Default)]
pub(crate) struct SameBuildClusterState {
    inner: InMemoryClusterState,
    refusals: Mutex<Vec<String>>,
}

impl SameBuildClusterState {
    fn admit(&self, metadata: &ExecutorMetadata) -> Result<()> {
        let Some(why) = refusal(metadata) else {
            return Ok(());
        };
        tracing::warn!("{why}");
        let mut refusals = self.refusals.lock().unwrap_or_else(PoisonError::into_inner);
        if refusals.len() >= REMEMBERED_REFUSALS {
            refusals.remove(0);
        }
        refusals.push(why.clone());
        Err(BallistaError::General(why))
    }

    /// Why no query can run here, when the reason is a refusal: no worker is
    /// active and at least one was turned away. A cluster still waiting for
    /// its first worker refused nothing, and keeps Ballista's behaviour of
    /// holding a query until a worker arrives.
    pub(crate) fn no_usable_worker(&self) -> Option<String> {
        if self.inner.executor_heartbeats().values().any(is_active) {
            return None;
        }
        let refusals = self.refusals.lock().unwrap_or_else(PoisonError::into_inner);
        refusals.last().map(|last| {
            format!(
                "this cluster has no worker it can use: the workers that tried to join were \
                 refused, most recently: {last}"
            )
        })
    }
}

#[async_trait]
impl ClusterState for SameBuildClusterState {
    async fn init(&self) -> Result<()> {
        self.inner.init().await
    }

    async fn bind_schedulable_tasks(
        &self,
        distribution: TaskDistributionPolicy,
        active_jobs: Arc<HashMap<JobId, JobInfoCache>>,
        executors: Option<HashSet<String>>,
    ) -> Result<Vec<BoundTask>> {
        self.inner
            .bind_schedulable_tasks(distribution, active_jobs, executors)
            .await
    }

    async fn unbind_tasks(&self, executor_slots: Vec<ExecutorSlot>) -> Result<()> {
        self.inner.unbind_tasks(executor_slots).await
    }

    async fn register_executor(
        &self,
        metadata: ExecutorMetadata,
        spec: ExecutorData,
    ) -> Result<()> {
        self.admit(&metadata)?;
        self.inner.register_executor(metadata, spec).await
    }

    // A pull-mode executor never registers: `poll_work` saves its metadata
    // instead, so the same check stands here.
    async fn save_executor_metadata(&self, metadata: ExecutorMetadata) -> Result<()> {
        self.admit(&metadata)?;
        self.inner.save_executor_metadata(metadata).await
    }

    async fn get_executor_metadata(&self, executor_id: &str) -> Result<ExecutorMetadata> {
        self.inner.get_executor_metadata(executor_id).await
    }

    async fn registered_executor_metadata(&self) -> Vec<ExecutorMetadata> {
        self.inner.registered_executor_metadata().await
    }

    async fn save_executor_heartbeat(&self, heartbeat: ExecutorHeartbeat) -> Result<()> {
        self.inner.save_executor_heartbeat(heartbeat).await
    }

    async fn remove_executor(&self, executor_id: &str) -> Result<()> {
        self.inner.remove_executor(executor_id).await
    }

    fn executor_heartbeats(&self) -> HashMap<String, ExecutorHeartbeat> {
        self.inner.executor_heartbeats()
    }

    fn get_executor_heartbeat(&self, executor_id: &str) -> Option<ExecutorHeartbeat> {
        self.inner.get_executor_heartbeat(executor_id)
    }

    async fn cluster_state_events(&self) -> Result<ClusterStateEventStream> {
        self.inner.cluster_state_events().await
    }
}

/// Wraps `builder` so a query fails at once, saying why, on a cluster whose
/// only workers were refused: Ballista would hold it until a worker arrived,
/// and none of the refused ones will. The scheduler builds a session for
/// every query it is sent, which makes this the first point a query can be
/// answered with a reason.
pub(crate) fn failing_without_a_usable_worker(
    builder: SessionBuilder,
    state: Arc<SameBuildClusterState>,
) -> SessionBuilder {
    Arc::new(move |config| match state.no_usable_worker() {
        Some(why) => Err(DataFusionError::Execution(why)),
        None => builder(config),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ballista_core::serde::scheduler::{
        ExecutorOperatingSystemSpecification, ExecutorSpecification,
    };
    use datafusion::prelude::SessionConfig;

    fn worker(id: &str) -> (ExecutorMetadata, ExecutorData) {
        let metadata = ExecutorMetadata {
            id: id.to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 50051,
            grpc_port: 50052,
            specification: ExecutorSpecification::default().with_vcores(2),
            os_info: ExecutorOperatingSystemSpecification::default(),
        };
        let data = ExecutorData {
            executor_id: id.to_owned(),
            total_vcores: 2,
            available_vcores: 2,
        };
        (metadata, data)
    }

    #[test]
    fn an_executor_id_carries_the_fingerprint_and_gives_it_back() {
        let id = tagged_executor_id("1f0e-instance");
        assert_eq!(fingerprint_of(&id), Some(PLAN_FINGERPRINT));
        assert_eq!(fingerprint_of(&tagged(0, "x")), Some(0));
        // A plain Ballista id, and tags that are not quite ours.
        for id in [
            "6c2e0f4e-7a1b-4c39-9a51-0d1e7c3b9f20",
            "oxl1234567-x",
            "oxl123456789-x",
            "oxl+2345678-x",
            "oxl1234567g-x",
            "oxl12345678",
        ] {
            assert_eq!(fingerprint_of(id), None, "{id}");
        }
    }

    #[tokio::test]
    async fn a_worker_of_another_build_is_refused_by_name_and_this_one_admitted() {
        let state = SameBuildClusterState::default();
        let theirs = !PLAN_FINGERPRINT;
        for id in [tagged(theirs, "a"), "6c2e0f4e-plain".to_owned()] {
            let (metadata, data) = worker(&id);
            let err = state
                .register_executor(metadata, data)
                .await
                .expect_err("refused")
                .to_string();
            assert!(err.contains(REFUSED) && err.contains(&id), "{err}");
            assert!(err.contains(&build_description()), "{err}");
        }
        let err = state.no_usable_worker().expect("only refused workers");
        assert!(err.contains("no worker it can use"), "{err}");
        assert!(err.contains("6c2e0f4e-plain"), "the most recent: {err}");
        assert!(state.registered_executor_metadata().await.is_empty());

        // A foreign one cannot get in through pull-mode metadata either.
        let (metadata, _) = worker(&tagged(theirs, "b"));
        assert!(state.save_executor_metadata(metadata).await.is_err());

        let (metadata, data) = worker(&tagged_executor_id("c"));
        state.register_executor(metadata, data).await.unwrap();
        assert_eq!(state.no_usable_worker(), None, "a usable worker joined");
    }

    #[test]
    fn a_cluster_waiting_for_its_first_worker_keeps_waiting() {
        assert_eq!(SameBuildClusterState::default().no_usable_worker(), None);
    }

    #[tokio::test]
    async fn the_refusals_kept_are_bounded() {
        let state = SameBuildClusterState::default();
        for i in 0..3 * REMEMBERED_REFUSALS {
            let (metadata, data) = worker(&format!("plain-{i}"));
            let _ = state.register_executor(metadata, data).await;
        }
        let kept = state.refusals.lock().unwrap();
        assert_eq!(kept.len(), REMEMBERED_REFUSALS);
        assert!(
            kept.last()
                .unwrap()
                .contains(&format!("plain-{}", 3 * REMEMBERED_REFUSALS - 1))
        );
    }

    #[tokio::test]
    async fn a_query_on_a_cluster_of_refused_workers_fails_with_the_reason() {
        let state = Arc::new(SameBuildClusterState::default());
        let builder: SessionBuilder = Arc::new(ballista_core::utils::default_session_builder);
        let builder = failing_without_a_usable_worker(builder, Arc::clone(&state));
        builder(SessionConfig::new()).expect("nothing refused yet: Ballista waits");

        let (metadata, data) = worker(&tagged(!PLAN_FINGERPRINT, "a"));
        let _ = state.register_executor(metadata, data).await;
        let err = builder(SessionConfig::new()).unwrap_err().to_string();
        assert!(err.contains("no worker it can use"), "{err}");
        assert!(
            err.contains(&format!("{:#010x}", !PLAN_FINGERPRINT)),
            "{err}"
        );

        let (metadata, data) = worker(&tagged_executor_id("b"));
        state.register_executor(metadata, data).await.unwrap();
        builder(SessionConfig::new()).expect("a usable worker joined");
    }
}
