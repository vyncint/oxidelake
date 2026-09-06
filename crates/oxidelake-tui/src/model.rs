//! The data the dashboard renders: a [`TelemetrySnapshot`] plus column
//! profiles, and the synthetic fixture the demo binary and snapshot tests use.

use oxidelake_core::BackendKind;
use oxidelake_core::telemetry::{
    OperatorSnapshot, PlanNodeSummary, SpillSnapshot, TelemetrySnapshot, TierSnapshot,
};

/// Profiling summary of one column for the Describe panel.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnProfile {
    /// Column name.
    pub name: String,
    /// Arrow type name.
    pub data_type: String,
    /// Minimum (rendered).
    pub min: String,
    /// Maximum (rendered).
    pub max: String,
    /// Null count.
    pub null_count: u64,
    /// 25th percentile (rendered; empty for non-numeric columns).
    pub p25: String,
    /// Median (rendered).
    pub p50: String,
    /// 99th percentile (rendered).
    pub p99: String,
}

/// Everything one frame renders.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DashboardModel {
    /// Engine telemetry: operators, memory tiers, spill counters, plan.
    pub telemetry: TelemetrySnapshot,
    /// Column profiles of the current table.
    pub profiles: Vec<ColumnProfile>,
    /// Title shown in the header (dataset or query name).
    pub title: String,
}

impl DashboardModel {
    /// Operators in plan order.
    pub fn plan(&self) -> &[PlanNodeSummary] {
        &self.telemetry.plan
    }

    /// The operator counters for the `index`-th plan node, if any.
    pub fn operator(&self, index: usize) -> Option<&OperatorSnapshot> {
        self.telemetry.operators.get(index)
    }
}

/// A fixed, clock-free fixture: a three-operator plan with CUDA and CPU
/// placement, populated counters, memory tiers and spill activity. Used by
/// `oxidelake-tui-demo` and the snapshot tests, so both layers see the same frame.
pub fn demo_model() -> DashboardModel {
    let plan = vec![
        PlanNodeSummary {
            depth: 0,
            name: "GpuAggregateExec".into(),
            backend: BackendKind::Cuda,
            detail: "group_by=[k], aggr=[SUM(v), COUNT(v)]".into(),
        },
        PlanNodeSummary {
            depth: 1,
            name: "GpuFilterExec".into(),
            backend: BackendKind::Cuda,
            detail: "k >= 2 AND v < 4.0, projection=[k, v]".into(),
        },
        PlanNodeSummary {
            depth: 2,
            name: "DataSourceExec".into(),
            backend: BackendKind::CpuSimd,
            detail: "parquet: data/t.parquet, 100 row groups".into(),
        },
    ];
    let operators = vec![
        OperatorSnapshot {
            id: 0,
            name: "GpuAggregateExec".into(),
            backend: BackendKind::Cuda,
            rows_in: 812_400,
            rows_out: 9,
            batches: 100,
            elapsed_ns: 184_000_000,
            bytes_h2d: 13_000_000,
            bytes_d2h: 512,
            memory_bytes: 16_777_216,
        },
        OperatorSnapshot {
            id: 1,
            name: "GpuFilterExec".into(),
            backend: BackendKind::Cuda,
            rows_in: 1_000_000,
            rows_out: 812_400,
            batches: 100,
            elapsed_ns: 96_000_000,
            bytes_h2d: 24_000_000,
            bytes_d2h: 19_500_000,
            memory_bytes: 33_554_432,
        },
        OperatorSnapshot {
            id: 2,
            name: "DataSourceExec".into(),
            backend: BackendKind::CpuSimd,
            rows_in: 1_000_000,
            rows_out: 1_000_000,
            batches: 100,
            elapsed_ns: 410_000_000,
            bytes_h2d: 0,
            bytes_d2h: 0,
            memory_bytes: 8_388_608,
        },
    ];
    DashboardModel {
        telemetry: TelemetrySnapshot {
            operators,
            tiers: TierSnapshot {
                device_bytes: 6 * 1024 * 1024 * 1024,
                host_bytes: 3 * 1024 * 1024 * 1024 / 2,
                disk_bytes: 512 * 1024 * 1024,
            },
            spill: SpillSnapshot {
                demotions: 42,
                promotions: 7,
                spilled_bytes: 3 * 1024 * 1024 * 1024,
                reloaded_bytes: 256 * 1024 * 1024,
            },
            plan,
        },
        profiles: vec![
            ColumnProfile {
                name: "k".into(),
                data_type: "Int64".into(),
                min: "0".into(),
                max: "99".into(),
                null_count: 0,
                p25: "24".into(),
                p50: "49".into(),
                p99: "98".into(),
            },
            ColumnProfile {
                name: "v".into(),
                data_type: "Float64".into(),
                min: "0.0".into(),
                max: "24.0".into(),
                null_count: 76_923,
                p25: "6.0".into(),
                p50: "12.0".into(),
                p99: "23.75".into(),
            },
            ColumnProfile {
                name: "s".into(),
                data_type: "Utf8".into(),
                min: "s0".into(),
                max: "s6".into(),
                null_count: 0,
                p25: String::new(),
                p50: String::new(),
                p99: String::new(),
            },
        ],
        title:
            "data/t.parquet — SELECT k, sum(v), count(v) FROM t WHERE k >= 2 AND v < 4.0 GROUP BY k"
                .into(),
    }
}
