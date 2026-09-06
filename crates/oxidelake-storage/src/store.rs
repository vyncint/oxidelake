//! Object-store construction and registration.

use std::sync::Arc;

use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::prelude::SessionContext;
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;

/// The default local object store used for Parquet scans and spill files.
///
/// With the `io-uring` feature callers can substitute `UringLocalFileSystem`
/// (`oxidelake_storage::uring`); both behave identically.
pub fn default_object_store() -> Arc<dyn ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

/// Registers `store` as the session's `file://` object store, so Parquet scans
/// and spill IO for local paths go through it.
pub fn register_local_store(ctx: &SessionContext, store: Arc<dyn ObjectStore>) {
    ctx.register_object_store(ObjectStoreUrl::local_filesystem().as_ref(), store);
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
        assert!(ctx.runtime_env().object_store(&url).is_ok());
    }
}
