//! Session configuration for Parquet pruning, table registration, and the
//! proof that pruning actually happened (read back from the scan's metrics).

use std::convert::Infallible;

use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanVisitor, accept};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use oxidelake_core::EngineError;

use crate::errors::classify_error;

/// Enables every Parquet pruning mechanism DataFusion offers — row-group
/// statistics, the page index and Bloom filters — and keeps expression
/// projections as `ProjectionExec` nodes. This is every OxideLake session's
/// base configuration.
///
/// Leaf-expression pushdown is disabled deliberately: it folds expression
/// projections (e.g. a `l2_distance(emb, …)` call) into `DataSourceExec`
/// itself, where the placement rule cannot lower them to `Gpu*Exec` — small
/// scans would silently lose GPU placement while large (repartitioned) ones
/// kept it. Column-only pushdown into scans is unaffected.
pub fn with_pruning(mut config: SessionConfig) -> SessionConfig {
    config
        .options_mut()
        .optimizer
        .enable_leaf_expression_pushdown = false;
    config
        .with_parquet_pruning(true)
        .with_parquet_bloom_filter_pruning(true)
        .with_parquet_page_index_pruning(true)
}

/// Rows per batch for sessions whose placement target is a GPU. Every device
/// dispatch carries a fixed host↔device round-trip cost, so GPU-targeted
/// plans amortise it over larger batches than DataFusion's CPU-tuned default
/// (8192). Batch boundaries never change query results.
pub const GPU_BATCH_SIZE: usize = 65_536;

/// Applies [`GPU_BATCH_SIZE`] to `config`.
pub fn with_gpu_batch_size(config: SessionConfig) -> SessionConfig {
    config.with_batch_size(GPU_BATCH_SIZE)
}

/// Disables all Parquet pruning (the reference configuration for equality tests).
pub fn with_pruning_disabled(config: SessionConfig) -> SessionConfig {
    config
        .with_parquet_pruning(false)
        .with_parquet_bloom_filter_pruning(false)
        .with_parquet_page_index_pruning(false)
}

/// A session with pruning enabled.
pub fn pruning_session() -> SessionContext {
    SessionContext::new_with_config(with_pruning(SessionConfig::new()))
}

/// Registers a Parquet file or directory as table `name`; file errors are
/// classified into [`EngineError::Format`] / [`EngineError::Io`].
pub async fn register_parquet_table(
    ctx: &SessionContext,
    name: &str,
    path: &str,
) -> Result<(), EngineError> {
    ctx.register_parquet(name, path, ParquetReadOptions::default())
        .await
        .map_err(classify_error)
}

/// Pruning counters summed over every scan node of an executed plan.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PruningSummary {
    /// Row groups skipped thanks to min/max statistics.
    pub row_groups_pruned_statistics: usize,
    /// Row groups that statistics could not exclude.
    pub row_groups_matched_statistics: usize,
    /// Row groups skipped thanks to Bloom filters.
    pub row_groups_pruned_bloom_filter: usize,
    /// Row groups that Bloom filters could not exclude.
    pub row_groups_matched_bloom_filter: usize,
    /// Rows skipped thanks to the page index.
    pub page_index_rows_pruned: usize,
    /// Rows the page index could not exclude.
    pub page_index_rows_matched: usize,
}

impl PruningSummary {
    /// Total row groups pruned by any mechanism.
    pub const fn row_groups_pruned(&self) -> usize {
        self.row_groups_pruned_statistics + self.row_groups_pruned_bloom_filter
    }
}

#[derive(Default)]
struct Collector(PruningSummary);

impl ExecutionPlanVisitor for Collector {
    type Error = Infallible;

    fn pre_visit(&mut self, plan: &dyn ExecutionPlan) -> Result<bool, Infallible> {
        if let Some(set) = plan.metrics() {
            for metric in set.iter() {
                if let MetricValue::PruningMetrics {
                    name,
                    pruning_metrics,
                } = metric.value()
                {
                    let (pruned, matched) = (pruning_metrics.pruned(), pruning_metrics.matched());
                    match name.as_ref() {
                        "row_groups_pruned_statistics" => {
                            self.0.row_groups_pruned_statistics += pruned;
                            self.0.row_groups_matched_statistics += matched;
                        }
                        "row_groups_pruned_bloom_filter" => {
                            self.0.row_groups_pruned_bloom_filter += pruned;
                            self.0.row_groups_matched_bloom_filter += matched;
                        }
                        "page_index_rows_pruned" => {
                            self.0.page_index_rows_pruned += pruned;
                            self.0.page_index_rows_matched += matched;
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(true)
    }
}

/// Reads the pruning counters from an *executed* plan tree.
pub fn scan_pruning_metrics(plan: &dyn ExecutionPlan) -> PruningSummary {
    let mut collector = Collector::default();
    match accept(plan, &mut collector) {
        Ok(()) => collector.0,
        Err(never) => match never {},
    }
}
