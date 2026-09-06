//! One behavioural suite for every local object store OxideLake can use. The
//! default `LocalFileSystem` always runs; the io_uring store runs under the
//! `io-uring` feature and skips with a logged reason where io_uring is
//! unavailable (sandboxed CI, old kernels).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use bytes::Bytes;
use futures::TryStreamExt;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

fn pattern(len: usize, seed: u8) -> Bytes {
    Bytes::from(
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect::<Vec<u8>>(),
    )
}

async fn exercise(store: Arc<dyn ObjectStore>) {
    let a = Path::from("data/a.bin");
    let b = Path::from("data/sub/b.bin");
    let data_a = pattern(100_003, 1);
    let data_b = pattern(4_096, 2);
    store
        .put(&a, PutPayload::from_bytes(data_a.clone()))
        .await
        .unwrap();
    store
        .put(&b, PutPayload::from_bytes(data_b.clone()))
        .await
        .unwrap();

    // whole object
    let got = store.get(&a).await.unwrap();
    assert_eq!(got.meta.size, data_a.len() as u64);
    assert_eq!(got.range, 0..data_a.len() as u64);
    assert_eq!(got.bytes().await.unwrap(), data_a);

    // ranged reads, including an unaligned tail range and a suffix past the end
    assert_eq!(
        store.get_range(&a, 1_000..5_000).await.unwrap(),
        data_a.slice(1_000..5_000)
    );
    assert_eq!(
        store.get_range(&a, 99_990..100_003).await.unwrap(),
        data_a.slice(99_990..)
    );
    let ranges = store
        .get_ranges(&a, &[0..10, 50_000..50_100, 7..8])
        .await
        .unwrap();
    assert_eq!(
        ranges,
        vec![
            data_a.slice(0..10),
            data_a.slice(50_000..50_100),
            data_a.slice(7..8)
        ]
    );

    // head + list
    let head = store.head(&b).await.unwrap();
    assert_eq!(head.size, data_b.len() as u64);
    assert_eq!(head.location, b);
    let listed: Vec<_> = store
        .list(Some(&Path::from("data")))
        .try_collect()
        .await
        .unwrap();
    let mut names: Vec<String> = listed.iter().map(|m| m.location.to_string()).collect();
    names.sort();
    assert_eq!(names, vec!["data/a.bin", "data/sub/b.bin"]);
    let with_delim = store
        .list_with_delimiter(Some(&Path::from("data")))
        .await
        .unwrap();
    assert_eq!(with_delim.objects.len(), 1);
    assert_eq!(with_delim.common_prefixes, vec![Path::from("data/sub")]);

    // overwrite + delete + missing
    store
        .put(&b, PutPayload::from_bytes(pattern(10, 9)))
        .await
        .unwrap();
    assert_eq!(
        store.get(&b).await.unwrap().bytes().await.unwrap(),
        pattern(10, 9)
    );
    store.delete(&b).await.unwrap();
    assert!(matches!(
        store.get(&b).await,
        Err(object_store::Error::NotFound { .. })
    ));
    assert!(matches!(
        store.head(&b).await,
        Err(object_store::Error::NotFound { .. })
    ));
    store.delete(&a).await.unwrap();
}

#[tokio::test]
async fn local_file_system_behaviour() {
    let dir = tempfile::tempdir().unwrap();
    exercise(Arc::new(
        LocalFileSystem::new_with_prefix(dir.path()).unwrap(),
    ))
    .await;
}

#[cfg(all(feature = "io-uring", target_os = "linux"))]
#[tokio::test]
async fn io_uring_store_behaves_like_local_file_system() {
    use oxidelake_storage::UringLocalFileSystem;
    if let Err(err) = UringLocalFileSystem::probe() {
        eprintln!(
            "SKIPPED: io_uring unavailable in this environment ({err}); the default LocalFileSystem path is used"
        );
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let uring =
        UringLocalFileSystem::try_new(LocalFileSystem::new_with_prefix(dir.path()).unwrap(), 32)
            .unwrap();
    assert!(uring.to_string().starts_with("UringLocalFileSystem("));
    exercise(Arc::new(uring)).await;

    // Cross-check: bytes written through io_uring read back identically through the plain store, and vice versa.
    let uring = Arc::new(
        UringLocalFileSystem::try_new(LocalFileSystem::new_with_prefix(dir.path()).unwrap(), 32)
            .unwrap(),
    );
    let plain: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    let p = Path::from("x/cross.bin");
    let data = pattern(65_537, 5);
    uring
        .put(&p, PutPayload::from_bytes(data.clone()))
        .await
        .unwrap();
    assert_eq!(plain.get(&p).await.unwrap().bytes().await.unwrap(), data);
    let q = Path::from("x/cross2.bin");
    plain
        .put(&q, PutPayload::from_bytes(data.clone()))
        .await
        .unwrap();
    assert_eq!(
        uring.get_range(&q, 65_000..65_537).await.unwrap(),
        data.slice(65_000..)
    );
}
