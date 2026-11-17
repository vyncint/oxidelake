//! Session configuration for Parquet pruning, table registration, and the
//! proof that pruning actually happened (read back from the scan's metrics).

use std::convert::Infallible;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use datafusion::datasource::listing::ListingTableUrl;
// `metadata::`'s, the type DataFusion's metadata cache reads back: an entry
// of the crate-root `CachedParquetMetaData` (another type) is a miss.
use datafusion::datasource::physical_plan::parquet::metadata::CachedParquetMetaData;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::cache::cache_manager::CachedFileMetadataEntry;
use datafusion::physical_plan::metrics::MetricValue;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanVisitor, accept};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use futures::future::BoxFuture;
use futures::{FutureExt, TryStreamExt};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use oxidelake_core::EngineError;

use crate::errors::classify_error;
use crate::parquet_check::{Fetch, read_checked_metadata};
use crate::scan_guard::ParquetScanGuard;
use crate::store::{default_object_store, register_local_store};

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

/// A session with pruning enabled, the local store registered through
/// [`register_local_store`] (ranged reads bounded by the object) and
/// [`ParquetScanGuard`] installed (files checked when a scan opens them).
pub fn pruning_session() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(with_pruning(SessionConfig::new()))
        .with_physical_optimizer_rule(Arc::new(ParquetScanGuard::new()))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_local_store(&ctx, default_object_store());
    ctx
}

/// Registers a Parquet file or directory as table `name`; file errors are
/// classified into [`EngineError::Format`] / [`EngineError::Io`].
///
/// Every file the table has is checked before DataFusion reads any of it
/// (#109): the footer's encoding, every column chunk, page index and Bloom
/// filter range against the file's size, and every page against its chunk.
/// A file that fails is refused with [`EngineError::Format`] naming it, and
/// the row group and column where there is one; the table is not registered.
/// The metadata that passed is handed to the session's metadata cache, so
/// the scans that follow read what was checked rather than decoding the
/// footer again.
///
/// This checks the files present now. A file added or replaced later is
/// checked when a scan opens it only in sessions that install
/// [`ParquetScanGuard`].
pub async fn register_parquet_table(
    ctx: &SessionContext,
    name: &str,
    path: &str,
) -> Result<(), EngineError> {
    let options = ParquetReadOptions::default();
    check_parquet_files(ctx, path, options.file_extension).await?;
    ctx.register_parquet(name, path, options)
        .await
        .map_err(classify_error)
}

/// Lists the files `path` names the way DataFusion's table registration will,
/// and checks each one.
async fn check_parquet_files(
    ctx: &SessionContext,
    path: &str,
    extension: &str,
) -> Result<(), EngineError> {
    let url = ListingTableUrl::parse(path).map_err(classify_error)?;
    let state = ctx.state();
    let store = ctx
        .runtime_env()
        .object_store(url.object_store())
        .map_err(classify_error)?;
    let files: Vec<_> = url
        .list_all_files(&state, store.as_ref(), extension)
        .await
        .map_err(classify_error)?
        .try_collect()
        .await
        .map_err(classify_error)?;
    let cache = ctx.runtime_env().cache_manager.get_file_metadata_cache();
    for meta in files {
        let mut fetch = StoreFetch {
            store: store.as_ref(),
            location: &meta.location,
        };
        let metadata = read_checked_metadata(&mut fetch, meta.location.as_ref(), meta.size).await?;
        cache.put(
            &meta.location,
            CachedFileMetadataEntry::new(
                meta.clone(),
                Arc::new(CachedParquetMetaData::new(Arc::new(metadata))),
            ),
        );
    }
    Ok(())
}

/// [`Fetch`] through the session's object store.
struct StoreFetch<'a> {
    store: &'a dyn ObjectStore,
    location: &'a Path,
}

impl Fetch for StoreFetch<'_> {
    fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes, EngineError>> {
        async move {
            self.store
                .get_range(self.location, range)
                .await
                .map_err(|e| EngineError::Io(std::io::Error::other(e.to_string())))
        }
        .boxed()
    }
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
