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
use object_store::{
    Attribute, Attributes, GetOptions, ObjectStore, ObjectStoreExt, PutOptions, PutPayload,
};

/// Prefix of every line a child reports, so the parent can find them in the
/// test harness's own output.
const RECORD: &str = "@@";

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

    // Conditional requests: `LocalFileSystem` checks these before reading and
    // the io_uring store used to ignore them, so a caller's `if_match` was
    // honoured or dropped depending on which store it happened to hold (#44).
    // Both stores answer these the same way now, and this is where that is
    // held.
    let tag = store.head(&a).await.unwrap().e_tag;
    if let Some(tag) = tag {
        let matching = GetOptions {
            if_match: Some(tag.clone()),
            ..Default::default()
        };
        let got = store.get_opts(&a, matching).await.unwrap();
        assert_eq!(
            got.bytes().await.unwrap(),
            data_a,
            "if_match on the current etag reads the object"
        );

        let stale = GetOptions {
            if_match: Some("\"not-the-etag\"".to_string()),
            ..Default::default()
        };
        assert!(
            matches!(
                store.get_opts(&a, stale).await,
                Err(object_store::Error::Precondition { .. })
            ),
            "if_match on a stale etag is a precondition failure"
        );

        let unchanged = GetOptions {
            if_none_match: Some(tag),
            ..Default::default()
        };
        assert!(
            matches!(
                store.get_opts(&a, unchanged).await,
                Err(object_store::Error::NotModified { .. })
            ),
            "if_none_match on the current etag is not-modified"
        );
    }

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

/// The stores every case below runs against: `LocalFileSystem` always, the
/// io_uring store where it is built and io_uring is allowed.
fn store_names() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut stores = vec!["local"];
    #[cfg(all(feature = "io-uring", target_os = "linux"))]
    match oxidelake_storage::UringLocalFileSystem::probe() {
        Ok(()) => stores.push("uring"),
        Err(err) => eprintln!("SKIPPED the io_uring half: io_uring unavailable ({err})"),
    }
    stores
}

/// A new store of kind `which` rooted at `dir`.
fn store_named(which: &str, dir: &std::path::Path) -> Arc<dyn ObjectStore> {
    let local = LocalFileSystem::new_with_prefix(dir).unwrap();
    match which {
        "local" => Arc::new(local),
        #[cfg(all(feature = "io-uring", target_os = "linux"))]
        "uring" => Arc::new(oxidelake_storage::UringLocalFileSystem::try_new(local, 32).unwrap()),
        other => panic!("unknown store {other}"),
    }
}

/// A put returns the e_tag `head` then reports (#111); the io_uring store
/// returned none.
#[tokio::test]
async fn a_put_returns_the_e_tag_head_reports() {
    for which in store_names() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_named(which, dir.path());
        let p = Path::from("put/tagged.bin");
        let put = store
            .put(&p, PutPayload::from_bytes(pattern(1_000, 3)))
            .await
            .unwrap();
        assert!(put.e_tag.is_some(), "{which}: a put returns its e_tag");
        assert_eq!(
            put.e_tag,
            store.head(&p).await.unwrap().e_tag,
            "{which}: the e_tag head reports"
        );

        // Several parts, as a payload built up chunk by chunk arrives, over
        // an existing object.
        let parts: PutPayload = [pattern(5, 4), pattern(70_000, 5), pattern(3, 6)]
            .into_iter()
            .collect();
        let put = store.put(&p, parts).await.unwrap();
        let mut whole = pattern(5, 4).to_vec();
        whole.extend_from_slice(&pattern(70_000, 5));
        whole.extend_from_slice(&pattern(3, 6));
        assert_eq!(
            store.get(&p).await.unwrap().bytes().await.unwrap(),
            whole,
            "{which}: a payload of several parts is written in order"
        );
        assert_eq!(
            put.e_tag,
            store.head(&p).await.unwrap().e_tag,
            "{which}: the overwrite's e_tag"
        );
    }
}

/// A put with attributes the store cannot keep is refused, not accepted
/// with the attributes dropped (#111).
#[tokio::test]
async fn a_put_refuses_attributes_it_cannot_store() {
    for which in store_names() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_named(which, dir.path());
        let p = Path::from("put/attributes.bin");
        store
            .put(&p, PutPayload::from_static(b"before"))
            .await
            .unwrap();
        let mut attributes = Attributes::new();
        attributes.insert(Attribute::ContentType, "application/octet-stream".into());
        let opts = PutOptions {
            attributes,
            ..Default::default()
        };
        let refused = store
            .put_opts(&p, PutPayload::from_static(b"after"), opts)
            .await;
        assert!(
            matches!(refused, Err(object_store::Error::NotImplemented { .. })),
            "{which}: attributes are refused, not dropped: {refused:?}"
        );
        assert_eq!(
            store.get(&p).await.unwrap().bytes().await.unwrap(),
            Bytes::from_static(b"before"),
            "{which}: a refused put leaves the object alone"
        );
    }
}

/// A put to a path that is a symlink replaces the link (#111). Writing
/// through it, as the io_uring store did, writes outside the store's root.
#[cfg(unix)]
#[tokio::test]
async fn a_put_replaces_a_symlink_rather_than_writing_through_it() {
    for which in store_names() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let store = store_named(which, root.path());
        let target = outside.path().join("outside.bin");
        std::fs::write(&target, b"outside").unwrap();
        let link = root.path().join("link.bin");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        store
            .put(&Path::from("link.bin"), PutPayload::from_static(b"inside"))
            .await
            .unwrap();
        assert_eq!(
            std::fs::read(&target).unwrap(),
            b"outside",
            "{which}: the file the link pointed at was written"
        );
        assert!(
            std::fs::symlink_metadata(&link).unwrap().is_file(),
            "{which}: the link was not replaced by the object"
        );
        assert_eq!(std::fs::read(&link).unwrap(), b"inside", "{which}");
    }
}

/// Waits for a file to be opened, by anyone: an inotify `IN_OPEN` watch.
/// The test below uses it to put its own steps between the store's,
/// rather than guessing how long the store takes to reach them.
#[cfg(target_os = "linux")]
struct OpenWatch(std::os::fd::OwnedFd);

#[cfg(target_os = "linux")]
impl OpenWatch {
    fn new(path: &std::path::Path) -> Self {
        use std::os::fd::{AsRawFd, FromRawFd};
        use std::os::unix::ffi::OsStrExt;
        // SAFETY: inotify_init1 takes flags only and returns a new descriptor
        // or -1.
        let fd = unsafe { libc::inotify_init1(libc::IN_CLOEXEC) };
        assert!(
            fd >= 0,
            "inotify_init1: {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: `fd` was just returned by inotify_init1 and nothing else
        // owns it.
        let fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) };
        let path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: a live inotify descriptor and a NUL-terminated path that
        // outlives the call.
        let watch =
            unsafe { libc::inotify_add_watch(fd.as_raw_fd(), path.as_ptr(), libc::IN_OPEN) };
        assert!(
            watch >= 0,
            "inotify_add_watch: {}",
            std::io::Error::last_os_error()
        );
        Self(fd)
    }

    /// Returns once the watched file has been opened; fails the test after
    /// a minute. Waits off the runtime, so the store's tasks keep running.
    async fn opened(self) {
        tokio::task::spawn_blocking(move || {
            use std::os::fd::AsRawFd;
            let mut poll = libc::pollfd {
                fd: self.0.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one live pollfd, for the length of the call.
            let ready = unsafe { libc::poll(&mut poll, 1, 60_000) };
            assert_eq!(ready, 1, "nothing opened the file within a minute");
            let mut event = [0u8; 4096];
            // SAFETY: reading into a buffer of the length given.
            let n =
                unsafe { libc::read(self.0.as_raw_fd(), event.as_mut_ptr().cast(), event.len()) };
            assert!(n >= 16, "inotify read: {}", std::io::Error::last_os_error());
            // struct inotify_event { int wd; uint32_t mask; ... }
            let mask = u32::from_ne_bytes(event[4..8].try_into().unwrap());
            assert!(mask & libc::IN_OPEN != 0, "inotify event {mask:#x}");
        })
        .await
        .unwrap();
    }
}

/// A file replaced between a get's precondition check and its read (#113).
///
/// The io_uring store checked `if_match` against a stat of the path, then
/// had its worker open the path again to read it, so a file replaced in
/// between came back as the new bytes under the old file's size and e_tag.
/// The window is held open here and the replacement put inside it, with no
/// sleeps: the store's single worker is parked on a read from an empty
/// FIFO (every later read queues behind it), and inotify says when the
/// store has opened the FIFO and then the object, which is when the object
/// is replaced. Only then is the worker let go.
///
/// Two outcomes are correct: the old object's bytes under its e_tag (the
/// store read the file it checked), or a precondition failure (it checked
/// the replacement). The old e_tag over the new bytes never is. A store
/// with no single worker (`LocalFileSystem`) is not parked at all.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_get_reads_the_file_its_preconditions_were_checked_against() {
    use std::io::Write;
    use std::os::unix::ffi::OsStrExt;
    use std::time::Duration;

    for which in store_names() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let store = store_named(which, root);
        let p = Path::from("swap/target.bin");
        let v1 = Bytes::from(vec![b'1'; 1_000]);
        store
            .put(&p, PutPayload::from_bytes(v1.clone()))
            .await
            .unwrap();
        let v1_tag = store.head(&p).await.unwrap().e_tag;
        assert!(v1_tag.is_some(), "{which}");

        // Opened read-write here, so the store's own open of it returns at
        // once and its read waits for data.
        let fifo = root.join("swap/fifo");
        let c_fifo = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
        // SAFETY: a NUL-terminated path that outlives the call.
        assert_eq!(unsafe { libc::mkfifo(c_fifo.as_ptr(), 0o600) }, 0);
        let mut fifo_end = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&fifo)
            .unwrap();
        let fifo_opened = OpenWatch::new(&fifo);
        let target_opened = OpenWatch::new(&root.join("swap/target.bin"));

        let parked = {
            let store = Arc::clone(&store);
            tokio::spawn(async move {
                store
                    .get_ranges(&Path::from("swap/fifo"), std::slice::from_ref(&(0..4)))
                    .await
            })
        };
        fifo_opened.opened().await;
        let get = {
            let store = Arc::clone(&store);
            let options = GetOptions {
                if_match: v1_tag.clone(),
                ..Default::default()
            };
            tokio::spawn(async move {
                let got = store
                    .get_opts(&Path::from("swap/target.bin"), options)
                    .await?;
                let meta = got.meta.clone();
                Ok::<_, object_store::Error>((meta, got.bytes().await?))
            })
        };
        target_opened.opened().await;

        // Replaced the way writers replace: a new file renamed over the old.
        let staged = root.join("swap/staged");
        std::fs::write(&staged, vec![b'2'; 2_000]).unwrap();
        std::fs::rename(&staged, root.join("swap/target.bin")).unwrap();
        fifo_end.write_all(b"open").unwrap();

        let minute = Duration::from_secs(60);
        // `LocalFileSystem` reads a FIFO with pread, which fails (ESPIPE);
        // only the parking matters here, not what the read returns.
        let _ = tokio::time::timeout(minute, parked).await.unwrap();
        match tokio::time::timeout(minute, get).await.unwrap().unwrap() {
            Ok((meta, bytes)) => {
                assert_eq!(meta.e_tag, v1_tag, "{which}");
                assert_eq!(meta.size, 1_000, "{which}");
                assert!(
                    bytes == v1,
                    "{which}: the e_tag and size are the old object's, the bytes are not \
                     ({} bytes, the first {:?})",
                    bytes.len(),
                    bytes.first().map(|b| char::from(*b))
                );
            }
            Err(object_store::Error::Precondition { .. }) => {}
            Err(other) => panic!("{which}: {other}"),
        }

        // With the replacement in place, the old e_tag fails the same way in
        // both stores.
        let stale = GetOptions {
            if_match: v1_tag,
            ..Default::default()
        };
        let stale = store.get_opts(&p, stale).await;
        assert!(
            matches!(stale, Err(object_store::Error::Precondition { .. })),
            "{which}: {stale:?}"
        );
    }
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

// ---- Cases that need process limits run in a child ----
//
// A limit set on this process would hold for every test in the binary, so
// these cases re-execute the binary as [`child`] with the limit applied to
// the child alone. Every child also runs under an address-space cap: a
// store that sizes an allocation by what it reads fails there, cleanly,
// instead of growing until something outside the test kills it.

/// Turns [`child`] from a no-op into one case run against one store:
/// `<case>:<store>`.
const CHILD_VAR: &str = "OXIDELAKE_STORE_CHILD";
const CHILD_DIR: &str = "OXIDELAKE_STORE_CHILD_DIR";

/// The address space every child gets.
#[cfg(target_os = "linux")]
const ADDRESS_SPACE_CAP: u64 = 2_000_000_000;

/// The child half: a no-op in a normal test run, one case when
/// [`run_child`] re-executes this binary with [`CHILD_VAR`] set.
#[tokio::test]
async fn child() {
    let Ok(spec) = std::env::var(CHILD_VAR) else {
        return;
    };
    let dir = std::path::PathBuf::from(std::env::var(CHILD_DIR).unwrap());
    let (case, which) = spec.split_once(':').unwrap();
    let store = store_named(which, &dir);
    match case {
        // Run under a 1 MiB file size limit (#111).
        "failed-put" => {
            let p = Path::from("old.bin");
            store
                .put(&p, PutPayload::from_static(b"old object"))
                .await
                .unwrap();
            let put = store
                .put(&p, PutPayload::from_bytes(pattern(4 << 20, 7)))
                .await;
            match put {
                Ok(_) => println!("{RECORD}PUT Ok"),
                Err(e) => println!("{RECORD}PUT Err {e}"),
            }
            let after = store.get(&p).await.unwrap().bytes().await.unwrap();
            println!(
                "{RECORD}AFTER {} bytes, the old object: {}",
                after.len(),
                after == Bytes::from_static(b"old object")
            );
            let mut names: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            println!("{RECORD}FILES {}", names.join(" "));
        }
        // Streams an object larger than this child's whole address space
        // (#112). The file is sparse: it costs no disk, and reading it costs
        // only what the store holds at once. Markers at a few offsets, one
        // across the 8 MiB mark, check that every chunk lands in place.
        "stream" => {
            use futures::StreamExt;
            let file = std::fs::File::create(dir.join("big.bin")).unwrap();
            file.set_len(STREAMED).unwrap();
            for (i, at) in MARKERS.iter().enumerate() {
                std::os::unix::fs::FileExt::write_all_at(&file, &marker(i), *at).unwrap();
            }
            drop(file);
            let mut stream = store
                .get(&Path::from("big.bin"))
                .await
                .unwrap()
                .into_stream();
            let (mut chunks, mut largest, mut total) = (0u64, 0u64, 0u64);
            let mut seen = [[0u8; 8]; MARKERS.len()];
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.unwrap();
                let len = chunk.len() as u64;
                for (i, at) in MARKERS.iter().enumerate() {
                    for k in 0..8 {
                        if let Some(j) = (at + k).checked_sub(total).filter(|j| *j < len) {
                            seen[i][k as usize] = chunk[j as usize];
                        }
                    }
                }
                chunks += 1;
                largest = largest.max(len);
                total += len;
            }
            println!("{RECORD}TOTAL {total}");
            println!("{RECORD}CHUNKS {chunks}");
            println!("{RECORD}LARGEST {largest}");
            let markers = (0..MARKERS.len()).all(|i| seen[i] == marker(i));
            println!("{RECORD}MARKERS {markers}");
            println!("{RECORD}MAXRSS_KIB {}", max_rss_kib());
        }
        other => panic!("unknown case {other}"),
    }
}

/// The streamed object's size: more than [`ADDRESS_SPACE_CAP`], so a store
/// that buffers the whole of it cannot even reserve the buffer.
const STREAMED: u64 = 3 << 30;

/// Where the streamed object carries non-zero bytes: its first and last
/// eight, eight across the 8 MiB mark, and eight in the middle.
const MARKERS: [u64; 4] = [0, (8 << 20) - 4, (3 << 29) + 12_345, STREAMED - 8];

fn marker(i: usize) -> [u8; 8] {
    (0x0101_0101_0101_0101u64 * (i as u64 + 1)).to_le_bytes()
}

/// Peak resident set of this process, in KiB (Linux reports `ru_maxrss` in
/// KiB).
fn max_rss_kib() -> i64 {
    // SAFETY: getrusage writes into the zeroed struct it is given.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage.ru_maxrss
    }
}

#[cfg(target_os = "linux")]
struct ChildRun {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

#[cfg(target_os = "linux")]
impl ChildRun {
    /// The value the child reported under `key`. The harness prints "test
    /// child ... " before the child's first line, so a record is found by
    /// its marker rather than at the start of a line.
    fn value(&self, key: &str) -> &str {
        let marker = format!("{RECORD}{key} ");
        self.stdout
            .lines()
            .find_map(|l| l.split_once(marker.as_str()).map(|(_, rest)| rest))
            .unwrap_or_else(|| panic!("the child reported no {key}\n{}", self.explain()))
    }

    fn explain(&self) -> String {
        let stderr: Vec<&str> = self.stderr.lines().collect();
        let tail = stderr[stderr.len().saturating_sub(15)..].join("\n");
        format!(
            "status {:?}\n--- stdout\n{}\n--- stderr (tail)\n{tail}",
            self.status, self.stdout
        )
    }
}

#[cfg(target_os = "linux")]
fn cap_address_space() -> std::io::Result<()> {
    let cap = libc::rlimit {
        rlim_cur: ADDRESS_SPACE_CAP,
        rlim_max: ADDRESS_SPACE_CAP,
    };
    // SAFETY: setrlimit on the calling (child) process with a valid struct.
    if unsafe { libc::setrlimit(libc::RLIMIT_AS, &cap) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Re-executes this test binary as [`child`] for one case and store, under
/// [`ADDRESS_SPACE_CAP`] and `limit`, and waits (bounded) for it to exit.
#[cfg(target_os = "linux")]
fn run_child(case: &str, which: &str, limit: fn() -> std::io::Result<()>) -> ChildRun {
    use std::io::Read;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    let dir = tempfile::tempdir().unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["child", "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_VAR, format!("{case}:{which}"))
        .env(CHILD_DIR, dir.path())
        // One malloc arena per thread reserves 64 MiB of address space each;
        // under the cap that is space the case under test cannot use.
        .env("MALLOC_ARENA_MAX", "2")
        .env("RUST_BACKTRACE", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // SAFETY: both make only async-signal-safe calls (setrlimit, signal)
    // and affect only the child.
    unsafe {
        command.pre_exec(cap_address_space);
        command.pre_exec(limit);
    }
    let mut child = command.spawn().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let out = std::thread::spawn(move || {
        let mut s = String::new();
        stdout.read_to_string(&mut s).unwrap();
        s
    });
    let err = std::thread::spawn(move || {
        let mut s = String::new();
        stderr.read_to_string(&mut s).unwrap();
        s
    });
    // Generous: a loaded machine is slow, and a hung child is a failure,
    // not something to wait on forever.
    let deadline = Instant::now() + Duration::from_secs(300);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            break child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let run = ChildRun {
        status,
        stdout: out.join().unwrap(),
        stderr: err.join().unwrap(),
    };
    assert!(
        run.status.success(),
        "{case}:{which}: the child did not exit cleanly\n{}",
        run.explain()
    );
    run
}

/// A put that fails part way leaves the object it was replacing (#111). The
/// io_uring store truncated the object and wrote over it in place, so a put
/// stopped by the file size limit left 1 MiB of the new bytes where the old
/// object had been; `LocalFileSystem` writes a staged file and renames it.
#[cfg(target_os = "linux")]
#[test]
fn a_failed_put_leaves_the_old_object() {
    fn one_mebibyte_files() -> std::io::Result<()> {
        // SAFETY: plain syscalls on the calling (child) process. SIGXFSZ is
        // ignored so the write that crosses the limit fails with EFBIG
        // instead of killing the child.
        unsafe {
            if libc::signal(libc::SIGXFSZ, libc::SIG_IGN) == libc::SIG_ERR {
                return Err(std::io::Error::last_os_error());
            }
            let cap = libc::rlimit {
                rlim_cur: 1 << 20,
                rlim_max: 1 << 20,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &cap) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(())
    }
    for which in store_names() {
        let run = run_child("failed-put", which, one_mebibyte_files);
        assert!(
            run.value("PUT").starts_with("Err "),
            "{which}: a 4 MiB put under a 1 MiB file size limit must fail\n{}",
            run.explain()
        );
        assert!(
            run.value("AFTER").ends_with("the old object: true"),
            "{which}: the failed put destroyed the object it was replacing\n{}",
            run.explain()
        );
        assert_eq!(
            run.value("FILES"),
            "old.bin",
            "{which}: a staged file was left behind\n{}",
            run.explain()
        );
    }
}

/// Streaming an object holds a bounded part of it at a time, whatever its
/// size (#112). The io_uring store read the whole range into one buffer
/// before returning the stream: one chunk the size of the object, a peak
/// RSS to match (1,027 MiB for a 1 GiB object in the review), and an
/// allocation failure for an object larger than the address space left.
/// The object here is larger than the child's whole address space.
#[cfg(target_os = "linux")]
#[test]
fn streaming_an_object_larger_than_the_address_space_holds_a_bounded_part_of_it() {
    /// Measured on Linux 7.0: 12,704 KiB for the io_uring store (8 MiB
    /// chunks), 4,452 KiB for `LocalFileSystem` (8 KiB chunks). Far below
    /// any buffer of the whole object; above the few chunks a streaming
    /// store holds at once.
    const RSS_LIMIT_KIB: i64 = 64 * 1024;
    fn no_further_limit() -> std::io::Result<()> {
        Ok(())
    }
    for which in store_names() {
        let run = run_child("stream", which, no_further_limit);
        let value = |key: &str| -> u64 { run.value(key).parse().unwrap() };
        assert_eq!(
            value("TOTAL"),
            STREAMED,
            "{which}: bytes streamed\n{}",
            run.explain()
        );
        assert!(
            value("LARGEST") <= 8 << 20,
            "{which}: a chunk of more than 8 MiB\n{}",
            run.explain()
        );
        assert_eq!(
            run.value("MARKERS"),
            "true",
            "{which}: a chunk out of place\n{}",
            run.explain()
        );
        let rss: i64 = run.value("MAXRSS_KIB").parse().unwrap();
        assert!(
            rss < RSS_LIMIT_KIB,
            "{which}: peak RSS {rss} KiB streaming {STREAMED} bytes\n{}",
            run.explain()
        );
        println!(
            "{which}: {} chunks, the largest {} bytes, peak RSS {rss} KiB",
            run.value("CHUNKS"),
            run.value("LARGEST")
        );
    }
}
