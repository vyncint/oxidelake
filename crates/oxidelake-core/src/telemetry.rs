//! Telemetry hub: atomic counters and gauges written by the engine and read as
//! consistent snapshots by the dashboard. It is the *only* coupling between
//! `oxidelake-tui` and the rest of the engine.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, PoisonError, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::BackendKind;

/// The three memory tiers tracked by the spill manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MemoryTier {
    /// Device (VRAM) memory.
    Device,
    /// Pinned or pageable host RAM.
    Host,
    /// Local disk spill files.
    Disk,
}

impl MemoryTier {
    const fn index(self) -> usize {
        match self {
            MemoryTier::Device => 0,
            MemoryTier::Host => 1,
            MemoryTier::Disk => 2,
        }
    }
}

/// Live counters for one physical operator instance.
#[derive(Debug)]
pub struct OperatorStats {
    id: usize,
    name: String,
    backend: BackendKind,
    rows_in: AtomicU64,
    rows_out: AtomicU64,
    batches: AtomicU64,
    elapsed_ns: AtomicU64,
    bytes_h2d: AtomicU64,
    bytes_d2h: AtomicU64,
    memory_bytes: AtomicU64,
    fallback_batches: AtomicU64,
}

impl OperatorStats {
    fn new(id: usize, name: String, backend: BackendKind) -> Self {
        Self {
            id,
            name,
            backend,
            rows_in: AtomicU64::new(0),
            rows_out: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            elapsed_ns: AtomicU64::new(0),
            bytes_h2d: AtomicU64::new(0),
            bytes_d2h: AtomicU64::new(0),
            memory_bytes: AtomicU64::new(0),
            fallback_batches: AtomicU64::new(0),
        }
    }

    /// Registration index (stable for the life of the hub).
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Operator display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The backend the operator was planned for.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Records one processed batch.
    pub fn record_batch(&self, rows_in: u64, rows_out: u64, elapsed: Duration) {
        self.rows_in.fetch_add(rows_in, Ordering::Relaxed);
        self.rows_out.fetch_add(rows_out, Ordering::Relaxed);
        self.batches.fetch_add(1, Ordering::Relaxed);
        let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.elapsed_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Records bytes moved host→device and device→host.
    pub fn record_transfer(&self, h2d: u64, d2h: u64) {
        self.bytes_h2d.fetch_add(h2d, Ordering::Relaxed);
        self.bytes_d2h.fetch_add(d2h, Ordering::Relaxed);
    }

    /// Sets the operator's currently allocated bytes.
    pub fn set_memory_bytes(&self, bytes: u64) {
        self.memory_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Records one batch that took the CPU reference path although the
    /// operator was placed on a device (#32).
    ///
    /// Identical results and identical `EXPLAIN` tags are exactly what a
    /// silently-falling-back cluster produces, so this is the number that
    /// distinguishes "the GPU ran it" from "something ran it". A batch is
    /// counted once, where the decision is made.
    pub fn record_fallback(&self) {
        self.fallback_batches.fetch_add(1, Ordering::Relaxed);
    }

    /// Batches that took the CPU reference path.
    pub fn fallback_batches(&self) -> u64 {
        self.fallback_batches.load(Ordering::Relaxed)
    }

    /// A point-in-time copy of the counters.
    pub fn snapshot(&self) -> OperatorSnapshot {
        OperatorSnapshot {
            id: self.id,
            name: self.name.clone(),
            backend: self.backend,
            rows_in: self.rows_in.load(Ordering::Relaxed),
            rows_out: self.rows_out.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            elapsed_ns: self.elapsed_ns.load(Ordering::Relaxed),
            bytes_h2d: self.bytes_h2d.load(Ordering::Relaxed),
            bytes_d2h: self.bytes_d2h.load(Ordering::Relaxed),
            memory_bytes: self.memory_bytes.load(Ordering::Relaxed),
            fallback_batches: self.fallback_batches.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of one operator's counters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorSnapshot {
    /// Registration index.
    pub id: usize,
    /// Operator display name.
    pub name: String,
    /// Planned backend.
    pub backend: BackendKind,
    /// Input rows.
    pub rows_in: u64,
    /// Output rows.
    pub rows_out: u64,
    /// Batches processed.
    pub batches: u64,
    /// Total processing time in nanoseconds.
    pub elapsed_ns: u64,
    /// Bytes copied host→device.
    pub bytes_h2d: u64,
    /// Bytes copied device→host.
    pub bytes_d2h: u64,
    /// Currently allocated bytes.
    pub memory_bytes: u64,
    /// Batches that ran on the CPU reference although the operator was
    /// placed on a device. Zero is the claim "this operator is accelerated";
    /// anything else is the size of the gap between the plan and what ran.
    pub fallback_batches: u64,
}

impl OperatorSnapshot {
    /// `true` when every batch this operator processed took the CPU
    /// reference path — the shape of a GPU deployment that is not using its
    /// GPU at all.
    pub fn fell_back_entirely(&self) -> bool {
        self.batches > 0 && self.fallback_batches >= self.batches
    }

    /// Mean per-batch latency in milliseconds (zero when nothing ran).
    pub fn mean_batch_latency_ms(&self) -> f64 {
        if self.batches == 0 {
            0.0
        } else {
            self.elapsed_ns as f64 / self.batches as f64 / 1_000_000.0
        }
    }
}

/// Bytes resident per memory tier.
#[derive(Debug, Default)]
pub struct TierGauges {
    bytes: [AtomicU64; 3],
}

impl TierGauges {
    /// Overwrites a tier's resident bytes.
    pub fn set(&self, tier: MemoryTier, bytes: u64) {
        self.bytes[tier.index()].store(bytes, Ordering::Relaxed);
    }

    /// Adds to a tier's resident bytes.
    pub fn add(&self, tier: MemoryTier, bytes: u64) {
        self.bytes[tier.index()].fetch_add(bytes, Ordering::Relaxed);
    }

    /// Subtracts from a tier's resident bytes, saturating at zero.
    pub fn sub(&self, tier: MemoryTier, bytes: u64) {
        let _ = self.bytes[tier.index()].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
            Some(v.saturating_sub(bytes))
        });
    }

    /// Current resident bytes for a tier.
    pub fn get(&self, tier: MemoryTier) -> u64 {
        self.bytes[tier.index()].load(Ordering::Relaxed)
    }

    /// Point-in-time copy.
    pub fn snapshot(&self) -> TierSnapshot {
        TierSnapshot {
            device_bytes: self.get(MemoryTier::Device),
            host_bytes: self.get(MemoryTier::Host),
            disk_bytes: self.get(MemoryTier::Disk),
        }
    }
}

/// Snapshot of the tier gauges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TierSnapshot {
    /// Bytes resident on devices.
    pub device_bytes: u64,
    /// Bytes resident in host RAM.
    pub host_bytes: u64,
    /// Bytes spilled to disk.
    pub disk_bytes: u64,
}

/// Spill activity counters.
#[derive(Debug, Default)]
pub struct SpillCounters {
    demotions: AtomicU64,
    promotions: AtomicU64,
    spilled_bytes: AtomicU64,
    reloaded_bytes: AtomicU64,
}

impl SpillCounters {
    /// Records a batch moving to a colder tier.
    pub fn record_demotion(&self, bytes: u64) {
        self.demotions.fetch_add(1, Ordering::Relaxed);
        self.spilled_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Records a batch moving to a hotter tier.
    pub fn record_promotion(&self, bytes: u64) {
        self.promotions.fetch_add(1, Ordering::Relaxed);
        self.reloaded_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Point-in-time copy.
    pub fn snapshot(&self) -> SpillSnapshot {
        SpillSnapshot {
            demotions: self.demotions.load(Ordering::Relaxed),
            promotions: self.promotions.load(Ordering::Relaxed),
            spilled_bytes: self.spilled_bytes.load(Ordering::Relaxed),
            reloaded_bytes: self.reloaded_bytes.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of the spill counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SpillSnapshot {
    /// Batches demoted to a colder tier.
    pub demotions: u64,
    /// Batches promoted to a hotter tier.
    pub promotions: u64,
    /// Total bytes demoted.
    pub spilled_bytes: u64,
    /// Total bytes promoted.
    pub reloaded_bytes: u64,
}

/// One node of the displayed physical plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanNodeSummary {
    /// Depth in the plan tree (root = 0).
    pub depth: usize,
    /// Operator name, e.g. `GpuFilterExec`.
    pub name: String,
    /// Backend the node is placed on.
    pub backend: BackendKind,
    /// One-line detail (predicate, keys, …).
    pub detail: String,
}

/// Why the placement rule left a node on the CPU (#32).
///
/// A skipped node is invisible in `EXPLAIN`: it is an ordinary DataFusion
/// operator, indistinguishable from one that was never eligible. The reason
/// was written to `debug!` and nowhere else, so the only way to find out was
/// to re-run the query with `RUST_LOG` turned up — which is not available to
/// someone reading a plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementSkip {
    /// The DataFusion operator that stayed on the CPU, e.g. `HashJoinExec`.
    pub node: String,
    /// Why, in one line.
    pub reason: String,
}

/// Most placement notes a hub keeps for one plan. A pathological plan should
/// not grow the hub without bound, and a reader stops reading long before
/// this.
pub const MAX_PLACEMENT_SKIPS: usize = 256;

/// Most operator registrations a hub keeps. A hub lives as long as its session
/// and every `Gpu*Exec` instance registers once, so without a bound a
/// long-lived session would accumulate one entry per operator of every query
/// it ever ran. Older entries are evicted first; live `OperatorStats` handles
/// stay valid, they just stop appearing in snapshots.
pub const MAX_OPERATORS: usize = 4096;

/// The engine-wide telemetry hub.
#[derive(Debug, Default)]
pub struct TelemetryHub {
    operators: RwLock<VecDeque<Arc<OperatorStats>>>,
    next_operator_id: AtomicUsize,
    tiers: TierGauges,
    spill: SpillCounters,
    plan: RwLock<Vec<PlanNodeSummary>>,
    skips: RwLock<Vec<PlacementSkip>>,
}

/// The process-wide hub.
///
/// Cluster executors do not build their operators — the plan arrives over the
/// wire and the codec rebuilds it — so there is no session object to hand a
/// hub to. This is that hub: the codec attaches it to every `Gpu*Exec` it
/// decodes, which is what makes a worker's `/metrics` describe the work the
/// worker actually did rather than an empty struct. Embedded sessions own
/// their own hub and never touch this one.
static GLOBAL: OnceLock<Arc<TelemetryHub>> = OnceLock::new();

impl TelemetryHub {
    /// Creates a shared hub.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The process-wide hub (see [`GLOBAL`]).
    pub fn global() -> &'static Arc<TelemetryHub> {
        GLOBAL.get_or_init(TelemetryHub::new)
    }

    /// Registers an operator and returns its live counters. Ids increase
    /// monotonically for the life of the hub; the oldest registrations are
    /// evicted once [`MAX_OPERATORS`] are held.
    pub fn register_operator(
        &self,
        name: impl Into<String>,
        backend: BackendKind,
    ) -> Arc<OperatorStats> {
        let id = self.next_operator_id.fetch_add(1, Ordering::Relaxed);
        let stats = Arc::new(OperatorStats::new(id, name.into(), backend));
        let mut ops = self
            .operators
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        if ops.len() >= MAX_OPERATORS {
            ops.pop_front();
        }
        ops.push_back(Arc::clone(&stats));
        stats
    }

    /// Forgets every registered operator (e.g. before running a new query
    /// whose dashboard should not show the previous one's counters).
    pub fn clear_operators(&self) {
        self.operators
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// Records why the placement rule left a node on the CPU.
    pub fn record_skip(&self, node: impl Into<String>, reason: impl Into<String>) {
        let mut skips = self.skips.write().unwrap_or_else(PoisonError::into_inner);
        if skips.len() >= MAX_PLACEMENT_SKIPS {
            return;
        }
        skips.push(PlacementSkip {
            node: node.into(),
            reason: reason.into(),
        });
    }

    /// The placement notes recorded since the last [`Self::clear_skips`].
    pub fn skips(&self) -> Vec<PlacementSkip> {
        self.skips
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Forgets the placement notes (called before planning a fresh query, so
    /// the notes belong to the plan being looked at).
    pub fn clear_skips(&self) {
        self.skips
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// Replaces the displayed plan.
    pub fn set_plan(&self, nodes: Vec<PlanNodeSummary>) {
        *self.plan.write().unwrap_or_else(PoisonError::into_inner) = nodes;
    }

    /// Memory tier gauges.
    pub const fn tiers(&self) -> &TierGauges {
        &self.tiers
    }

    /// Spill counters.
    pub const fn spill(&self) -> &SpillCounters {
        &self.spill
    }

    /// A consistent point-in-time copy of everything the dashboard renders.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let operators = self
            .operators
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|o| o.snapshot())
            .collect();
        let plan = self
            .plan
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        TelemetrySnapshot {
            operators,
            tiers: self.tiers.snapshot(),
            spill: self.spill.snapshot(),
            plan,
        }
    }
}

/// Everything the dashboard renders, captured at one instant.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    /// Per-operator counters, in registration order.
    pub operators: Vec<OperatorSnapshot>,
    /// Memory tier gauges.
    pub tiers: TierSnapshot,
    /// Spill counters.
    pub spill: SpillSnapshot,
    /// The displayed plan.
    pub plan: Vec<PlanNodeSummary>,
}

impl TelemetrySnapshot {
    /// Renders the snapshot as Prometheus text (`text/plain; version=0.0.4`).
    ///
    /// Written by hand rather than through a metrics crate: the counters are
    /// already the right shape, the output is a dozen lines, and a scrape
    /// endpoint is not worth a dependency tree in a query engine. Operator
    /// counters carry `operator` and `backend` labels; the tier gauges and
    /// spill counters carry none.
    ///
    /// Label values are escaped per the exposition format, because an
    /// operator name is `GpuFilterExec` today and a user-supplied string the
    /// moment someone adds one.
    pub fn to_prometheus(&self) -> String {
        let mut out = String::new();
        let counters: [OperatorCounter; 7] = [
            ("oxide_operator_rows_in_total", "Input rows.", |o| o.rows_in),
            ("oxide_operator_rows_out_total", "Output rows.", |o| {
                o.rows_out
            }),
            ("oxide_operator_batches_total", "Batches processed.", |o| {
                o.batches
            }),
            (
                "oxide_operator_fallback_batches_total",
                "Batches that ran on the CPU reference although the operator was placed on a device.",
                |o| o.fallback_batches,
            ),
            (
                "oxide_operator_elapsed_nanoseconds_total",
                "Processing time.",
                |o| o.elapsed_ns,
            ),
            (
                "oxide_operator_bytes_h2d_total",
                "Bytes copied host to device.",
                |o| o.bytes_h2d,
            ),
            (
                "oxide_operator_bytes_d2h_total",
                "Bytes copied device to host.",
                |o| o.bytes_d2h,
            ),
        ];
        for (name, help, value) in counters {
            out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n"));
            for op in &self.operators {
                out.push_str(&format!(
                    "{name}{{operator=\"{}\",backend=\"{}\"}} {}\n",
                    escape_label(&op.name),
                    op.backend,
                    value(op)
                ));
            }
        }
        out.push_str("# HELP oxide_operator_memory_bytes Currently allocated bytes.\n");
        out.push_str("# TYPE oxide_operator_memory_bytes gauge\n");
        for op in &self.operators {
            out.push_str(&format!(
                "oxide_operator_memory_bytes{{operator=\"{}\",backend=\"{}\"}} {}\n",
                escape_label(&op.name),
                op.backend,
                op.memory_bytes
            ));
        }
        let gauges = [
            ("oxide_tier_device_bytes", self.tiers.device_bytes),
            ("oxide_tier_host_bytes", self.tiers.host_bytes),
            ("oxide_tier_disk_bytes", self.tiers.disk_bytes),
        ];
        for (name, value) in gauges {
            out.push_str(&format!(
                "# HELP {name} Bytes resident in this memory tier.\n# TYPE {name} gauge\n{name} {value}\n"
            ));
        }
        let spill = [
            ("oxide_spill_demotions_total", self.spill.demotions),
            ("oxide_spill_promotions_total", self.spill.promotions),
            ("oxide_spill_spilled_bytes_total", self.spill.spilled_bytes),
            (
                "oxide_spill_reloaded_bytes_total",
                self.spill.reloaded_bytes,
            ),
        ];
        for (name, value) in spill {
            out.push_str(&format!(
                "# HELP {name} Spill manager activity.\n# TYPE {name} counter\n{name} {value}\n"
            ));
        }
        out
    }
}

/// One exported operator counter: metric name, HELP text, and how to read it
/// off a snapshot.
type OperatorCounter = (&'static str, &'static str, fn(&OperatorSnapshot) -> u64);

/// Escapes a Prometheus label value: backslash, double quote, newline.
fn escape_label(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_snapshot_renders_as_prometheus_text() {
        let hub = TelemetryHub::new();
        let op = hub.register_operator("GpuFilterExec", BackendKind::Cuda);
        op.record_batch(1_000, 400, Duration::from_millis(3));
        op.record_transfer(2_048, 512);
        op.record_fallback();
        hub.tiers().set(MemoryTier::Disk, 4_096);
        let text = hub.snapshot().to_prometheus();

        assert!(
            text.contains(
                "oxide_operator_rows_in_total{operator=\"GpuFilterExec\",backend=\"cuda\"} 1000"
            ),
            "{text}"
        );
        assert!(
            text.contains("oxide_operator_fallback_batches_total{operator=\"GpuFilterExec\",backend=\"cuda\"} 1"),
            "{text}"
        );
        assert!(text.contains("oxide_tier_disk_bytes 4096"), "{text}");
        // Every metric is declared before it is used, which is what a scraper
        // needs and what a hand-written exporter is most likely to forget.
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let name = line.split(['{', ' ']).next().unwrap_or_default();
            assert!(
                text.contains(&format!("# TYPE {name} ")),
                "{name} has no TYPE"
            );
        }
    }

    /// An operator name is a Rust type name today and could be anything
    /// tomorrow; an unescaped quote would produce a file a scraper rejects.
    #[test]
    fn label_values_are_escaped() {
        let hub = TelemetryHub::new();
        hub.register_operator("we\"ird\\name", BackendKind::CpuSimd);
        let text = hub.snapshot().to_prometheus();
        assert!(text.contains("operator=\"we\\\"ird\\\\name\""), "{text}");
    }

    #[test]
    fn operators_register_in_order_and_accumulate() {
        let hub = TelemetryHub::new();
        let a = hub.register_operator("GpuFilterExec", BackendKind::Cuda);
        let b = hub.register_operator("GpuAggregateExec", BackendKind::CpuSimd);
        assert_eq!((a.id(), b.id()), (0, 1));
        a.record_batch(100, 40, Duration::from_millis(2));
        a.record_batch(100, 60, Duration::from_millis(4));
        a.record_transfer(800, 480);
        a.set_memory_bytes(4096);
        let snap = hub.snapshot();
        assert_eq!(snap.operators.len(), 2);
        let op = &snap.operators[0];
        assert_eq!((op.rows_in, op.rows_out, op.batches), (200, 100, 2));
        assert_eq!(
            (op.bytes_h2d, op.bytes_d2h, op.memory_bytes),
            (800, 480, 4096)
        );
        assert!((op.mean_batch_latency_ms() - 3.0).abs() < 1e-9);
        assert_eq!(snap.operators[1].mean_batch_latency_ms(), 0.0);
    }

    #[test]
    fn tiers_and_spill_counters() {
        let hub = TelemetryHub::new();
        hub.tiers().add(MemoryTier::Host, 1000);
        hub.tiers().sub(MemoryTier::Host, 300);
        hub.tiers().sub(MemoryTier::Disk, 5); // saturates at zero
        hub.tiers().set(MemoryTier::Device, 42);
        hub.spill().record_demotion(700);
        hub.spill().record_promotion(700);
        let snap = hub.snapshot();
        assert_eq!(
            snap.tiers,
            TierSnapshot {
                device_bytes: 42,
                host_bytes: 700,
                disk_bytes: 0
            }
        );
        assert_eq!(snap.spill.demotions, 1);
        assert_eq!(snap.spill.promotions, 1);
        assert_eq!(snap.spill.spilled_bytes, 700);
    }

    #[test]
    fn plan_summary_is_replaced() {
        let hub = TelemetryHub::new();
        hub.set_plan(vec![PlanNodeSummary {
            depth: 0,
            name: "GpuFilterExec".into(),
            backend: BackendKind::Metal,
            detail: "a > 1".into(),
        }]);
        assert_eq!(hub.snapshot().plan.len(), 1);
        hub.set_plan(Vec::new());
        assert!(hub.snapshot().plan.is_empty());
    }
}
