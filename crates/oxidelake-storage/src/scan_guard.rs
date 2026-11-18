//! Checks a Parquet file's metadata when a scan opens it, not only when its
//! table was registered (#109).
//!
//! [`register_parquet_table`](crate::register_parquet_table) checks every
//! file a table has at that moment. A file added to a table's directory
//! later, or replaced under its name, reaches the scan unchecked — and the
//! parquet reader trusts whatever its footer says. [`ParquetScanGuard`] closes
//! that gap for the sessions that install it: it wraps the reader factory of
//! every Parquet scan in the plan with one whose readers check the metadata
//! (page index included) before the reader uses it, refuse reads outside the
//! file, and turn a panic inside the reader into an error.
//!
//! The rule changes no plan shape and no `EXPLAIN` line: the scan is the same
//! `DataSourceExec`, carrying a different reader factory. That factory is not
//! part of a serialized plan, so a Ballista executor scans with DataFusion's
//! own reader; what an executor reads was checked at registration and, when
//! the scheduler's rule carries its metadata cache, again when the scheduler
//! planned the query ([`ParquetScanGuard::with_metadata_cache`]).

use std::fmt;
use std::ops::Range;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use bytes::Bytes;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::datasource::listing::PartitionedFile;
// DataFusion 55 has two types of this name; the metadata cache holds
// `metadata::`'s, and a downcast to `reader::`'s (the crate-root re-export)
// finds nothing in it.
use datafusion::datasource::physical_plan::parquet::metadata::CachedParquetMetaData;
use datafusion::datasource::physical_plan::{
    FileScanConfig, FileScanConfigBuilder, ParquetFileReaderFactory, ParquetSource,
};
use datafusion::datasource::source::DataSourceExec;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::cache::cache_manager::FileMetadataCache;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::FutureExt;
use futures::future::BoxFuture;
use oxidelake_core::EngineError;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::errors::ParquetError;
use parquet::file::metadata::ParquetMetaData;

use crate::parquet_check::{Fetch, check_chunks, check_pages, load_page_index, panic_message};

/// The physical optimizer rule that makes every Parquet scan in a plan check
/// its files' metadata before reading by it. Install it after every other
/// rule (`SessionStateBuilder::with_physical_optimizer_rule` appends), so the
/// scans it wraps are the final ones.
#[derive(Default, Clone)]
pub struct ParquetScanGuard {
    metadata_cache: Option<Arc<FileMetadataCache>>,
}

impl ParquetScanGuard {
    /// The rule for a session that executes its own scans (embedded).
    pub fn new() -> Self {
        Self::default()
    }

    /// Also checks, at planning time, the metadata DataFusion already read
    /// into `cache` while planning (row-group statistics, on by default).
    ///
    /// This is the check a Ballista scheduler can make on behalf of its
    /// executors, whose scans do not carry the wrapped reader factory: a file
    /// corrupted after registration fails the query on the scheduler, before
    /// any executor opens it. Pass the scheduler session's own cache
    /// (`RuntimeEnv::cache_manager.get_file_metadata_cache()`); a file
    /// whose metadata is not in it is not read here, only at scan time.
    pub fn with_metadata_cache(mut self, cache: Arc<FileMetadataCache>) -> Self {
        self.metadata_cache = Some(cache);
        self
    }

    fn guard(&self, node: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
        let Some(exec) = node.downcast_ref::<DataSourceExec>() else {
            return Ok(Transformed::no(node));
        };
        let Some(config) = exec.data_source().downcast_ref::<FileScanConfig>() else {
            return Ok(Transformed::no(node));
        };
        let Some(source) = config.file_source().downcast_ref::<ParquetSource>() else {
            return Ok(Transformed::no(node));
        };
        if let Some(cache) = &self.metadata_cache {
            check_cached(cache.as_ref(), config)?;
        }
        // DataFusion's ParquetFormat sets a factory on every listing-table
        // scan; a source without one would build a default reader from a
        // store this rule cannot reach, so it is left as it is.
        let Some(inner) = source.parquet_file_reader_factory() else {
            return Ok(Transformed::no(node));
        };
        let source =
            source
                .clone()
                .with_parquet_file_reader_factory(Arc::new(GuardedReaderFactory {
                    inner: Arc::clone(inner),
                }));
        let config = FileScanConfigBuilder::from(config.clone())
            .with_source(Arc::new(source))
            .build();
        Ok(Transformed::yes(Arc::new(
            exec.clone().with_data_source(Arc::new(config)),
        )))
    }
}

impl fmt::Debug for ParquetScanGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParquetScanGuard")
            .field("checks_metadata_cache", &self.metadata_cache.is_some())
            .finish()
    }
}

impl PhysicalOptimizerRule for ParquetScanGuard {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| self.guard(node)).map(|t| t.data)
    }

    fn name(&self) -> &str {
        "ParquetScanGuard"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// The plan-time check: every file of the scan whose current metadata the
/// session already holds.
fn check_cached(cache: &FileMetadataCache, config: &FileScanConfig) -> Result<()> {
    for file in config.file_groups.iter().flat_map(|group| group.files()) {
        let meta = &file.object_meta;
        let Some(entry) = cache.get(&meta.location) else {
            continue;
        };
        if !entry.is_valid_for(meta) {
            continue;
        }
        let Some(cached) = entry
            .file_metadata
            .as_any()
            .downcast_ref::<CachedParquetMetaData>()
        else {
            continue;
        };
        let location = meta.location.to_string();
        let metadata = cached.parquet_metadata();
        check_chunks(&location, meta.size, metadata)
            .and_then(|()| check_pages(&location, metadata))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
    }
    Ok(())
}

/// Wraps the reader factory DataFusion chose, so every reader it makes is a
/// [`GuardedReader`].
#[derive(Debug)]
struct GuardedReaderFactory {
    inner: Arc<dyn ParquetFileReaderFactory>,
}

impl ParquetFileReaderFactory for GuardedReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Box<dyn AsyncFileReader + Send>> {
        let location = partitioned_file.object_meta.location.to_string();
        let size = partitioned_file.object_meta.size;
        let inner = self.inner.create_reader(
            partition_index,
            partitioned_file,
            metadata_size_hint,
            metrics,
        )?;
        Ok(Box::new(GuardedReader {
            inner,
            location,
            size,
        }))
    }
}

/// A reader that checks what the reader it wraps would otherwise trust.
struct GuardedReader {
    inner: Box<dyn AsyncFileReader + Send>,
    location: String,
    /// The size the scan listed the file with: the bound every read is
    /// checked against, before the store allocates for it.
    size: u64,
}

fn parquet_error(err: EngineError) -> ParquetError {
    ParquetError::External(Box::new(err))
}

impl GuardedReader {
    fn check(&self, range: &Range<u64>) -> parquet::errors::Result<()> {
        if range.start > range.end || range.end > self.size {
            return Err(parquet_error(EngineError::format(format!(
                "parquet file {}: a read of bytes {}..{} lies outside the {}-byte file",
                self.location, range.start, range.end, self.size
            ))));
        }
        Ok(())
    }

    fn panicked(&self, panic: &(dyn std::any::Any + Send)) -> ParquetError {
        parquet_error(EngineError::format(format!(
            "parquet file {}: the reader panicked: {}",
            self.location,
            panic_message(panic)
        )))
    }

    async fn metadata(
        &mut self,
        options: Option<&ArrowReaderOptions>,
    ) -> parquet::errors::Result<Arc<ParquetMetaData>> {
        let metadata = match AssertUnwindSafe(self.inner.get_metadata(options))
            .catch_unwind()
            .await
        {
            Ok(result) => result?,
            Err(panic) => return Err(self.panicked(panic.as_ref())),
        };
        check_chunks(&self.location, self.size, &metadata).map_err(parquet_error)?;
        // A scan asks for the footer alone and loads the page index later,
        // through a path that decodes it unchecked. Loaded here instead (when
        // the file has one), so it is checked before anything uses it.
        let metadata = if metadata.offset_index().is_none() && has_offset_index(&metadata) {
            let owned = Arc::try_unwrap(metadata).unwrap_or_else(|shared| (*shared).clone());
            let (location, size) = (self.location.clone(), self.size);
            let mut fetch = ReaderFetch { reader: self };
            Arc::new(
                load_page_index(&mut fetch, &location, size, owned)
                    .await
                    .map_err(parquet_error)?,
            )
        } else {
            metadata
        };
        check_pages(&self.location, &metadata).map_err(parquet_error)?;
        Ok(metadata)
    }
}

fn has_offset_index(metadata: &ParquetMetaData) -> bool {
    metadata
        .row_groups()
        .iter()
        .flat_map(|rg| rg.columns())
        .any(|c| c.offset_index_offset().is_some() && c.offset_index_length().is_some())
}

impl AsyncFileReader for GuardedReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        async move {
            self.check(&range)?;
            match AssertUnwindSafe(self.inner.get_bytes(range))
                .catch_unwind()
                .await
            {
                Ok(result) => result,
                Err(panic) => Err(self.panicked(panic.as_ref())),
            }
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        async move {
            for range in &ranges {
                self.check(range)?;
            }
            match AssertUnwindSafe(self.inner.get_byte_ranges(ranges))
                .catch_unwind()
                .await
            {
                Ok(result) => result,
                Err(panic) => Err(self.panicked(panic.as_ref())),
            }
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        self.metadata(options).boxed()
    }
}

/// [`Fetch`] through a guarded reader: bounded by its checks.
struct ReaderFetch<'a> {
    reader: &'a mut GuardedReader,
}

impl Fetch for ReaderFetch<'_> {
    fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes, EngineError>> {
        async move {
            self.reader
                .get_bytes(range)
                .await
                .map_err(|e| EngineError::format(format!("parquet: {e}")))
        }
        .boxed()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// An inner reader that panics on every call, counting the calls that
    /// reached it.
    struct Panicking(Arc<AtomicUsize>);

    impl AsyncFileReader for Panicking {
        fn get_bytes(&mut self, _: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            async { panic!("the inner reader fell over") }.boxed()
        }

        fn get_metadata<'a>(
            &'a mut self,
            _: Option<&'a ArrowReaderOptions>,
        ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            async { panic!("the footer decoder fell over") }.boxed()
        }
    }

    fn reader(calls: &Arc<AtomicUsize>) -> GuardedReader {
        GuardedReader {
            inner: Box::new(Panicking(Arc::clone(calls))),
            location: "t.parquet".into(),
            size: 100,
        }
    }

    fn format_message(err: ParquetError) -> String {
        match err {
            ParquetError::External(e) => match e.downcast::<EngineError>() {
                Ok(engine) => match *engine {
                    EngineError::Format(m) => m,
                    other => panic!("expected Format, got {other:?}"),
                },
                Err(e) => panic!("expected an EngineError, got {e}"),
            },
            other => panic!("expected External, got {other:?}"),
        }
    }

    /// A panic inside the reader comes back as the file's `Format` error.
    #[tokio::test]
    async fn a_panic_inside_the_reader_is_a_format_error() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut guarded = reader(&calls);
        let m = format_message(guarded.get_metadata(None).await.unwrap_err());
        assert!(
            m.contains("t.parquet") && m.contains("footer decoder fell over"),
            "{m}"
        );
        let m = format_message(guarded.get_bytes(0..10).await.unwrap_err());
        assert!(m.contains("inner reader fell over"), "{m}");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// A read outside the file never reaches the inner reader, which is
    /// what would reserve memory for it.
    #[tokio::test]
    async fn a_read_outside_the_file_never_reaches_the_inner_reader() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut guarded = reader(&calls);
        #[allow(clippy::reversed_empty_ranges)]
        let inverted = 50..40;
        for bad in [0..(1u64 << 40), 90..101, inverted] {
            let m = format_message(guarded.get_bytes(bad.clone()).await.unwrap_err());
            assert!(m.contains("lies outside the 100-byte file"), "{bad:?}: {m}");
            let m = format_message(
                guarded
                    .get_byte_ranges(vec![0..1, bad.clone()])
                    .await
                    .unwrap_err(),
            );
            assert!(m.contains("lies outside the 100-byte file"), "{bad:?}: {m}");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}
