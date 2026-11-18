//! Object-store construction and registration.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::prelude::SessionContext;
use futures::stream::BoxStream;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult, RenameOptions, Result,
};
use oxidelake_core::EngineError;

/// The default local object store used for Parquet scans and spill files.
///
/// With the `io-uring` feature callers can substitute `UringLocalFileSystem`
/// (`oxidelake_storage::uring`); both behave identically.
pub fn default_object_store() -> Arc<dyn ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

/// Registers `store` as the session's `file://` object store, so Parquet scans
/// and spill IO for local paths go through it.
///
/// The store is registered behind [`BoundedReadStore`], so no ranged read in
/// the session can reserve more memory than the object holds, whatever a
/// file's footer declares (#109).
pub fn register_local_store(ctx: &SessionContext, store: Arc<dyn ObjectStore>) {
    ctx.register_object_store(
        ObjectStoreUrl::local_filesystem().as_ref(),
        Arc::new(BoundedReadStore::new(store)),
    );
}

const BOUNDED: &str = "BoundedReadStore";

/// An object store that refuses a multi-range read reaching past the end of
/// the object before the store under it allocates the ranges.
///
/// `LocalFileSystem::get_ranges` (object_store 0.13) reserves and zero-fills
/// each requested range before reading it, and fails only afterwards. Every
/// range DataFusion's Parquet reader asks for comes from the file's own
/// footer, so a corrupt length there used to cost its full declared size: 2
/// GB of RSS for a 98 KB file, or an abort at 2^40 (#109). The check costs one
/// `head` per `get_ranges` call. Single-range reads (`get_opts`) need no
/// guard: the store clamps them to the size it has just looked up.
#[derive(Debug)]
pub struct BoundedReadStore {
    inner: Arc<dyn ObjectStore>,
}

impl BoundedReadStore {
    /// Wraps `inner`.
    pub fn new(inner: Arc<dyn ObjectStore>) -> Self {
        Self { inner }
    }
}

impl fmt::Display for BoundedReadStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{BOUNDED}({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for BoundedReadStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        let size = self.inner.head(location).await?.size;
        if let Some(range) = ranges
            .iter()
            .find(|range| range.start > range.end || range.end > size)
        {
            // A Format error inside: every range here came from a file's
            // own metadata, so one past the end means the file is corrupt or
            // truncated, and classify_error reports it that way.
            return Err(object_store::Error::Generic {
                store: BOUNDED,
                source: Box::new(EngineError::format(format!(
                    "{location}: bytes {}..{} lie outside the {size}-byte object; refusing \
                     the read rather than reserving memory for it",
                    range.start, range.end
                ))),
            });
        }
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_store_is_local() {
        let store = default_object_store();
        assert!(store.to_string().contains("LocalFileSystem"));
    }

    #[test]
    fn registration_replaces_the_local_store() {
        let ctx = SessionContext::new();
        register_local_store(&ctx, default_object_store());
        let url = ObjectStoreUrl::local_filesystem();
        let store = ctx.runtime_env().object_store(&url).unwrap();
        assert!(
            store.to_string().starts_with("BoundedReadStore("),
            "{store}"
        );
    }

    /// The read the review measured at 2 GB of RSS: one range declared far
    /// past the object. Refused before the inner store reserves anything;
    /// ranges inside the object read as before.
    ///
    /// Far, not huge. This runs in the library's test binary with no memory
    /// cap, and `LocalFileSystem` zero-fills a range before it reads into
    /// it: were the bound to regress, a range of 2^40 bytes would grow the
    /// binary until something outside the test killed it (where overcommit
    /// is on, the reservation succeeds and every page is then touched; 54 GB
    /// before the OOM killer, measured). At 64 MiB a regression costs 64 MiB
    /// and still fails the assertion below.
    #[tokio::test]
    async fn a_range_past_the_end_is_refused_before_it_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let store = BoundedReadStore::new(Arc::new(
            LocalFileSystem::new_with_prefix(dir.path()).unwrap(),
        ));
        let p = Path::from("t.parquet");
        store
            .put(&p, PutPayload::from_static(b"0123456789"))
            .await
            .unwrap();
        assert_eq!(
            store.get_ranges(&p, &[2..4, 8..10]).await.unwrap(),
            vec![Bytes::from_static(b"23"), Bytes::from_static(b"89")]
        );
        #[allow(clippy::reversed_empty_ranges)]
        let inverted = 5..3;
        for bad in [0..(64u64 << 20), 9..11, inverted] {
            let err = store
                .get_ranges(&p, &[0..1, bad.clone()])
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains("lie outside the 10-byte object"),
                "{bad:?}: {err}"
            );
        }
        assert!(matches!(
            store
                .get_ranges(&Path::from("missing"), std::slice::from_ref(&(0..1)))
                .await,
            Err(object_store::Error::NotFound { .. })
        ));
    }
}
