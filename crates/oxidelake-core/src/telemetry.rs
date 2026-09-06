//! Telemetry hub: atomic counters and gauges written by the engine and read as
//! consistent snapshots by the dashboard. It is the *only* coupling between
//! `oxidelake-tui` and the rest of the engine.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, PoisonError, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::BackendKind;

/// The three memory tiers tracked by the spill manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
}

impl OperatorSnapshot {
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
}

impl TelemetryHub {
    /// Creates a shared hub.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
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

#[cfg(test)]
mod tests {
    use super::*;

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
