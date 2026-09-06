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

/// Submits one SQE and waits for its completion, returning the raw result.
fn submit_one(ring: &mut IoUring, entry: &io_uring::squeue::Entry) -> io::Result<i32> {
    // SAFETY: the caller guarantees the buffers referenced by `entry` stay alive
    // and untouched until this function returns, which only happens after the
    // completion has been reaped.
    unsafe { ring.submission().push(entry) }
        .map_err(|e| io::Error::other(format!("io_uring submission queue is full: {e:?}")))?;
    ring.submit_and_wait(1)?;
    let cqe = ring
        .completion()
        .next()
        .ok_or_else(|| io::Error::other("io_uring returned no completion"))?;
    let result = cqe.result();
    if result < 0 {
        return Err(io::Error::from_raw_os_error(-result));
    }
    Ok(result)
}

fn read_range(ring: &mut IoUring, path: &PathBuf, range: Range<u64>) -> io::Result<Bytes> {
    let file = File::open(path)?;
    let len = usize::try_from(range.end.saturating_sub(range.start))
        .map_err(|_| io::Error::other("range too large"))?;
    let mut buf = vec![0u8; len];
    let mut done = 0usize;
    while done < len {
        let chunk = u32::try_from(len - done).unwrap_or(u32::MAX);
        let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), buf[done..].as_mut_ptr(), chunk)
            .offset(range.start + done as u64)
            .build()
            .user_data(1);
        let n = submit_one(ring, &entry)?;
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

fn write_all(ring: &mut IoUring, path: &PathBuf, data: &Bytes) -> io::Result<()> {
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
            .build()
            .user_data(2);
        let n = submit_one(ring, &entry)?;
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
    while let Some(job) = rx.blocking_recv() {
        match job {
            Job::Read { path, range, reply } => {
                let _ = reply.send(read_range(&mut ring, &path, range));
            }
            Job::Write { path, data, reply } => {
                let _ = reply.send(write_all(&mut ring, &path, &data));
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
