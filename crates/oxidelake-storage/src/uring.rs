//! io_uring-backed local object store (feature `io-uring`, Linux only).
//!
//! A dedicated storage thread owns the ring: ranged reads and whole-object
//! writes are submitted as `IORING_OP_READ` / `IORING_OP_WRITE` and their
//! completions are bridged back to async callers through `oneshot` channels.
//! Metadata operations (`list`, `head`, `delete`, `copy`, multipart uploads)
//! delegate to the wrapped [`LocalFileSystem`], so both stores behave
//! identically — the shared conformance test in `tests/` runs against both.
//!
//! Construction fails with a typed error where io_uring is unavailable
//! (EPERM/ENOSYS in sandboxes, old kernels); callers fall back to
//! [`LocalFileSystem`].

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt};
use io_uring::{IoUring, opcode, types};
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    Attributes, CopyOptions, Error, GetOptions, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, ObjectStoreExt, PutMode, PutMultipartOptions,
    PutOptions, PutPayload, PutResult, RenameOptions, Result,
};
use oxidelake_core::EngineError;
use tokio::sync::{mpsc, oneshot};

const STORE: &str = "UringLocalFileSystem";

enum Job {
    Read {
        path: PathBuf,
        range: Range<u64>,
        reply: oneshot::Sender<io::Result<Bytes>>,
    },
    Write {
        path: PathBuf,
        data: Bytes,
        reply: oneshot::Sender<io::Result<()>>,
    },
}

/// How many times to re-enter the ring after a submit error that is not
/// `EINTR` before giving up on ever seeing the completion. Only reachable if
/// the ring itself is broken: this worker keeps at most one SQE in flight and
/// the completion queue is twice the submission queue, so the conditions that
/// produce `EBUSY`/`EAGAIN` cannot arise here.
const REAP_ATTEMPTS: u32 = 1024;

/// Submits one SQE, waits for *its* completion, and returns the raw result.
///
/// `seq` is the worker's submission counter; this call takes the next value
/// and tags the SQE with it, so the completion can be matched to the
/// submission that produced it rather than assumed.
///
/// The contract that matters is that this function does not return while the
/// kernel may still be writing into the caller's buffer (#27). A signal
/// delivered to this thread makes `submit_and_wait` return `EINTR` with the
/// SQE still in flight; returning there dropped the caller's `Vec` under the
/// kernel and left the completion to be read by the *next* job as its own.
/// So every exit below reaps first, and the error, if any, is reported after
/// the completion is in hand rather than instead of it.
fn submit_one(
    ring: &mut IoUring,
    entry: &io_uring::squeue::Entry,
    seq: &mut u64,
) -> io::Result<i32> {
    *seq = seq.wrapping_add(1);
    let tag = *seq;
    let entry = entry.clone().user_data(tag);

    // SAFETY: the buffer `entry` points at is owned by the caller and lives
    // until this function returns, and this function returns only after the
    // kernel's completion for this exact SQE has been reaped below.
    unsafe { ring.submission().push(&entry) }
        .map_err(|e| io::Error::other(format!("io_uring submission queue is full: {e:?}")))?;

    // Past this point the SQE belongs to the kernel.
    let mut failure: Option<io::Error> = None;
    let mut attempts = 0u32;
    let result = loop {
        if let Err(e) = ring.submit_and_wait(1) {
            if e.kind() == io::ErrorKind::Interrupted {
                // A signal, not a failure: the SQE is still in flight and the
                // completion is still coming. This is the path the test in
                // `tests/uring_signals.rs` drives.
                continue;
            }
            if failure.is_none() {
                failure = Some(e);
            }
        }
        match ring.completion().next() {
            Some(cqe) if cqe.user_data() == tag => break cqe.result(),
            Some(cqe) => {
                // A completion for something else means the ring and this
                // worker disagree about what is outstanding, so no result
                // from it can be trusted -- including this one.
                return Err(io::Error::other(format!(
                    "io_uring completion carries submission {} where {tag} was expected; \
                     the ring is out of step with the worker",
                    cqe.user_data()
                )));
            }
            None => {
                attempts += 1;
                assert!(
                    failure.is_none() || attempts < REAP_ATTEMPTS,
                    "io_uring: submission {tag} was accepted by the kernel but its completion \
                     never arrived after {REAP_ATTEMPTS} attempts ({}); returning would free a \
                     buffer the kernel may still be writing into (#27)",
                    failure
                        .as_ref()
                        .map_or_else(|| "no error".to_owned(), |e| e.to_string()),
                );
            }
        }
    };
    if let Some(e) = failure {
        return Err(e);
    }
    if result < 0 {
        return Err(io::Error::from_raw_os_error(-result));
    }
    Ok(result)
}

fn read_range(
    ring: &mut IoUring,
    seq: &mut u64,
    path: &PathBuf,
    range: Range<u64>,
) -> io::Result<Bytes> {
    let file = File::open(path)?;
    let len = usize::try_from(range.end.saturating_sub(range.start))
        .map_err(|_| io::Error::other("range too large"))?;
    let mut buf = vec![0u8; len];
    let mut done = 0usize;
    while done < len {
        let chunk = u32::try_from(len - done).unwrap_or(u32::MAX);
        let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), buf[done..].as_mut_ptr(), chunk)
            .offset(range.start + done as u64)
            .build();
        let n = submit_one(ring, &entry, seq)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "short read: {done} of {len} bytes at offset {}",
                    range.start
                ),
            ));
        }
        done += usize::try_from(n).unwrap_or(0);
    }
    Ok(Bytes::from(buf))
}

fn write_all(ring: &mut IoUring, seq: &mut u64, path: &PathBuf, data: &Bytes) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    let mut done = 0usize;
    while done < data.len() {
        let chunk = u32::try_from(data.len() - done).unwrap_or(u32::MAX);
        let entry = opcode::Write::new(types::Fd(file.as_raw_fd()), data[done..].as_ptr(), chunk)
            .offset(done as u64)
            .build();
        let n = submit_one(ring, &entry, seq)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "io_uring wrote zero bytes",
            ));
        }
        done += usize::try_from(n).unwrap_or(0);
    }
    Ok(())
}

fn worker(mut rx: mpsc::UnboundedReceiver<Job>, mut ring: IoUring) {
    // Submission counter for the ring this thread owns. Every SQE is tagged
    // with the next value so its completion can be recognised rather than
    // assumed to be the only one outstanding (#27).
    let mut seq = 0u64;
    while let Some(job) = rx.blocking_recv() {
        match job {
            Job::Read { path, range, reply } => {
                let _ = reply.send(read_range(&mut ring, &mut seq, &path, range));
            }
            Job::Write { path, data, reply } => {
                let _ = reply.send(write_all(&mut ring, &mut seq, &path, &data));
            }
        }
    }
}

/// A local object store whose data-path reads and writes go through io_uring.
pub struct UringLocalFileSystem {
    inner: Arc<LocalFileSystem>,
    tx: mpsc::UnboundedSender<Job>,
}

impl UringLocalFileSystem {
    /// Creates the store around `inner`, spawning the ring thread with
    /// `entries` submission slots. Fails when io_uring is unavailable.
    pub fn try_new(inner: LocalFileSystem, entries: u32) -> std::result::Result<Self, EngineError> {
        let ring = IoUring::new(entries.max(2)).map_err(|e| {
            EngineError::Io(io::Error::new(
                e.kind(),
                format!("io_uring is unavailable on this machine ({e}); use LocalFileSystem"),
            ))
        })?;
        let (tx, rx) = mpsc::unbounded_channel();
        std::thread::Builder::new()
            .name("oxide-io-uring".into())
            .spawn(move || worker(rx, ring))?;
        Ok(Self {
            inner: Arc::new(inner),
            tx,
        })
    }

    /// `Err` with the reason when io_uring cannot be set up here.
    pub fn probe() -> io::Result<()> {
        IoUring::new(2).map(|_| ())
    }

    fn filesystem_path(&self, location: &Path) -> Result<PathBuf> {
        self.inner.path_to_filesystem(location)
    }

    fn io_error(location: &Path, err: io::Error) -> Error {
        if err.kind() == io::ErrorKind::NotFound {
            Error::NotFound {
                path: location.to_string(),
                source: Box::new(err),
            }
        } else {
            Error::Generic {
                store: STORE,
                source: Box::new(err),
            }
        }
    }

    async fn read(&self, location: &Path, range: Range<u64>) -> Result<Bytes> {
        let path = self.filesystem_path(location)?;
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job::Read { path, range, reply })
            .map_err(|_| Error::Generic {
                store: STORE,
                source: "io_uring worker thread has exited".into(),
            })?;
        let result = rx.await.map_err(|_| Error::Generic {
            store: STORE,
            source: "io_uring worker dropped the reply".into(),
        })?;
        result.map_err(|e| Self::io_error(location, e))
    }

    async fn write(&self, location: &Path, data: Bytes) -> Result<()> {
        let path = self.filesystem_path(location)?;
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(Job::Write { path, data, reply })
            .map_err(|_| Error::Generic {
                store: STORE,
                source: "io_uring worker thread has exited".into(),
            })?;
        let result = rx.await.map_err(|_| Error::Generic {
            store: STORE,
            source: "io_uring worker dropped the reply".into(),
        })?;
        result.map_err(|e| Self::io_error(location, e))
    }
}

impl fmt::Debug for UringLocalFileSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(STORE).field("inner", &self.inner).finish()
    }
}

impl fmt::Display for UringLocalFileSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{STORE}({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for UringLocalFileSystem {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        match opts.mode {
            PutMode::Overwrite => {
                self.write(location, Bytes::from(payload)).await?;
                Ok(PutResult {
                    e_tag: None,
                    version: None,
                })
            }
            _ => self.inner.put_opts(location, payload, opts).await,
        }
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult> {
        let meta: ObjectMeta = self.inner.head(location).await?;
        let range = match &options.range {
            Some(r) => r.as_range(meta.size).map_err(|e| Error::Generic {
                store: STORE,
                source: Box::new(e),
            })?,
            None => 0..meta.size,
        };
        if options.head {
            return Ok(GetResult {
                payload: GetResultPayload::Stream(stream::empty().boxed()),
                meta,
                range,
                attributes: Attributes::default(),
            });
        }
        let bytes = self.read(location, range.clone()).await?;
        Ok(GetResult {
            payload: GetResultPayload::Stream(stream::once(async move { Ok(bytes) }).boxed()),
            meta,
            range,
            attributes: Attributes::default(),
        })
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>> {
        let mut out = Vec::with_capacity(ranges.len());
        for range in ranges {
            out.push(self.read(location, range.clone()).await?);
        }
        Ok(out)
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

    /// A handler that does nothing, installed *without* `SA_RESTART` so the
    /// interrupted syscall really returns `EINTR` rather than being restarted
    /// by the kernel. Both halves matter: with no handler at all `SIGUSR1`
    /// terminates the process, and with `SA_RESTART` the test would pass
    /// against the very bug it exists to catch.
    extern "C" fn noop(_: libc::c_int) {}

    fn install_interrupting_handler() {
        // SAFETY: a plain extern "C" handler with an empty mask; `noop`
        // touches nothing, so it is async-signal-safe.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = noop as *const () as usize;
            action.sa_flags = 0;
            libc::sigemptyset(&mut action.sa_mask);
            assert_eq!(
                libc::sigaction(libc::SIGUSR1, &action, std::ptr::null_mut()),
                0,
                "could not install the SIGUSR1 handler: {}",
                io::Error::last_os_error()
            );
        }
    }

    /// The regression test for #27, aimed rather than sprayed.
    ///
    /// An end-to-end read is the wrong instrument: a local read completes in
    /// microseconds, so a signaller racing it almost never lands inside
    /// `submit_and_wait`, and such a test passes against the bug. This drives
    /// the path directly instead — a read submitted against an *empty pipe*,
    /// which cannot complete until somebody writes, so the worker is
    /// guaranteed to be blocked in `submit_and_wait` when the signal arrives.
    /// `tgkill` targets this exact thread rather than the process.
    ///
    /// Before the fix `submit_one` returned `Err(EINTR)` here, leaving the
    /// SQE in flight and `buf` to be dropped under the kernel. Now the signal
    /// is invisible: the read completes and its bytes arrive.
    #[test]
    fn a_signal_during_the_wait_does_not_abandon_the_completion() {
        let Ok(mut ring) = IoUring::new(4) else {
            eprintln!("SKIPPED: io_uring unavailable in this environment");
            return;
        };
        install_interrupting_handler();

        // SAFETY: a pipe pair; both descriptors are closed at the end.
        let mut fds = [0 as libc::c_int; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let (read_fd, write_fd) = (fds[0], fds[1]);

        // SAFETY: `gettid` and `getpid` take no arguments and only read.
        let tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        let pid = unsafe { libc::getpid() };

        let signaller = std::thread::spawn(move || {
            // Long enough that the read is certainly parked in the kernel.
            std::thread::sleep(std::time::Duration::from_millis(50));
            for _ in 0..5 {
                // SAFETY: signalling a thread of our own process.
                unsafe { libc::syscall(libc::SYS_tgkill, pid, tid, libc::SIGUSR1) };
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            // Only now can the read complete, so every signal above landed
            // while the ring was waiting.
            let payload = b"uring-ok";
            // SAFETY: writing 8 bytes from a live buffer to the pipe.
            unsafe { libc::write(write_fd, payload.as_ptr().cast(), payload.len()) };
        });

        let mut buf = [0u8; 8];
        let entry = opcode::Read::new(types::Fd(read_fd), buf.as_mut_ptr(), 8).build();
        let mut seq = 0u64;
        let n = submit_one(&mut ring, &entry, &mut seq)
            .expect("a signal during the wait is not a failure: the SQE is still in flight");

        signaller.join().unwrap();
        assert_eq!(n, 8, "the whole payload was read");
        assert_eq!(&buf, b"uring-ok", "the bytes are the ones written");
        assert_eq!(seq, 1, "the submission was tagged");

        // SAFETY: closing descriptors this test opened and no longer uses.
        unsafe {
            libc::close(read_fd);
            libc::close(write_fd);
        }
    }

    /// The tag is what makes a completion recognisable as *this* submission's
    /// (#27). Two submissions in a row must carry different ones, or a stale
    /// completion is indistinguishable from the expected one.
    #[test]
    fn each_submission_is_tagged_with_a_fresh_value() {
        let Ok(mut ring) = IoUring::new(4) else {
            eprintln!("SKIPPED: io_uring unavailable in this environment");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tagged.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let file = File::open(&path).unwrap();

        let mut seq = 0u64;
        for expected in 1..=3u64 {
            let mut buf = [0u8; 10];
            let entry =
                opcode::Read::new(types::Fd(file.as_raw_fd()), buf.as_mut_ptr(), 10).build();
            let n = submit_one(&mut ring, &entry, &mut seq).unwrap();
            assert_eq!(n, 10);
            assert_eq!(&buf, b"0123456789");
            assert_eq!(seq, expected, "each submission takes the next tag");
        }
    }
}
