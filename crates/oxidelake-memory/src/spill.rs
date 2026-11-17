//! Three-tier spill manager: device VRAM → host RAM → local disk.
//!
//! Batches are registered in a tier and tracked with a logical clock. When a
//! tier exceeds its budget the coldest batch is demoted one tier down; reading
//! a batch that lives on disk promotes it back into host RAM. The disk tier is
//! Arrow IPC (see [`crate::ipc`]) written through an [`ObjectStore`], so the
//! same code runs over `LocalFileSystem` or the io_uring store.
//!
//! Device residency is abstracted behind [`DeviceResident`] so this crate does
//! not depend on any backend: a backend hands over something that knows its
//! size and how to copy itself back to a host `RecordBatch`.

use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use arrow::array::{Array, ArrayData, ArrayRef, MAX_INLINE_VIEW_LEN, RecordBatch};
use arrow::datatypes::{DataType, SchemaRef};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use oxidelake_core::EngineError;
use oxidelake_core::telemetry::{MemoryTier, TelemetryHub};
use tokio::sync::Mutex;

use crate::ipc;

/// Identifier of a batch registered with the spill manager.
pub type BatchId = u64;

/// Where a batch currently lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Tier {
    /// Device memory.
    Device,
    /// Host RAM.
    Host,
    /// Disk (Arrow IPC file).
    Disk,
}

impl Tier {
    const fn index(self) -> usize {
        match self {
            Tier::Device => 0,
            Tier::Host => 1,
            Tier::Disk => 2,
        }
    }

    const fn gauge(self) -> MemoryTier {
        match self {
            Tier::Device => MemoryTier::Device,
            Tier::Host => MemoryTier::Host,
            Tier::Disk => MemoryTier::Disk,
        }
    }
}

/// Byte budgets per tier. The disk tier is unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillBudget {
    /// Maximum bytes resident on the device.
    pub device_bytes: u64,
    /// Maximum bytes resident in host RAM.
    pub host_bytes: u64,
}

/// A batch resident in device memory, as seen by the spill manager.
pub trait DeviceResident: Send + Sync + fmt::Debug {
    /// Bytes occupied on the device.
    fn bytes(&self) -> u64;
    /// Copies the batch back to host memory.
    fn to_host(&self) -> Result<RecordBatch, EngineError>;
}

enum Payload {
    Device(Box<dyn DeviceResident>),
    Host(RecordBatch),
    Disk(Path),
}

struct Entry {
    tier: Tier,
    /// Bytes in the current tier (disk size while spilled).
    bytes: u64,
    /// Host size ([`SpillManager::batch_bytes`]), fixed at first host
    /// residency and reused on promotion, so a batch back from disk is
    /// charged what it was charged before: the batch decoded from its spill
    /// file need not measure the same as the one that was written.
    host_bytes: u64,
    last_touch: u64,
    schema: SchemaRef,
    payload: Payload,
}

#[derive(Default)]
struct State {
    entries: HashMap<BatchId, Entry>,
    used: [u64; 3],
}

impl State {
    fn coldest(&self, tier: Tier) -> Option<BatchId> {
        self.entries
            .iter()
            .filter(|(_, e)| e.tier == tier)
            .min_by_key(|(_, e)| e.last_touch)
            .map(|(id, _)| *id)
    }
}

/// The spill manager. Cheap to share via `Arc`.
///
/// Its spill files are named `batch-<tag>-<id>.arrow` under its prefix, with
/// a tag drawn at random for this manager, so managers can share a store and
/// a prefix (#114). Dropping the manager deletes the spill files it still
/// holds, waiting up to 30 seconds for the deletes (#117).
pub struct SpillManager {
    budget: SpillBudget,
    store: Arc<dyn ObjectStore>,
    prefix: Path,
    telemetry: Arc<TelemetryHub>,
    state: Mutex<State>,
    next_id: AtomicU64,
    clock: AtomicU64,
    /// Sets this manager's file names apart from any other's (#114).
    tag: String,
}

fn store_error(err: object_store::Error) -> EngineError {
    match err {
        object_store::Error::NotFound { .. } => EngineError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            err.to_string(),
        )),
        other => EngineError::Io(std::io::Error::other(other.to_string())),
    }
}

impl SpillManager {
    /// Creates a manager writing spill files under `prefix` in `store`.
    pub fn new(
        budget: SpillBudget,
        store: Arc<dyn ObjectStore>,
        prefix: Path,
        telemetry: Arc<TelemetryHub>,
    ) -> Arc<Self> {
        Arc::new(Self {
            budget,
            store,
            prefix,
            telemetry,
            state: Mutex::new(State::default()),
            next_id: AtomicU64::new(1),
            clock: AtomicU64::new(0),
            tag: instance_tag(),
        })
    }

    /// The configured budgets.
    pub const fn budget(&self) -> SpillBudget {
        self.budget
    }

    /// The telemetry hub this manager reports to.
    pub const fn telemetry(&self) -> &Arc<TelemetryHub> {
        &self.telemetry
    }

    /// Host bytes a batch is charged against the budgets (#116).
    ///
    /// For each column, the bytes of its own slice of the buffers it
    /// references: its rows' values, offsets and validity, and for a
    /// `Utf8View` or `BinaryView` column the out-of-line strings its views
    /// point to. A slice of a larger batch is charged for its own rows, so
    /// the slices of one batch together are charged what the batch is. The
    /// parent's buffers stay allocated while any slice of them lives, which
    /// no per-batch measure can see: the budgets bound what the registered
    /// batches are charged, not the allocator's resident set. A column arrow
    /// cannot size by its slice is charged its buffers whole.
    pub fn batch_bytes(batch: &RecordBatch) -> u64 {
        let bytes = batch.columns().iter().fold(0_usize, |sum, column| {
            sum.saturating_add(column_bytes(column))
        });
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn spill_path(&self, id: BatchId) -> Path {
        Path::from(format!("{}/batch-{}-{id}.arrow", self.prefix, self.tag))
    }

    fn publish(&self, state: &State) {
        for tier in [Tier::Device, Tier::Host, Tier::Disk] {
            self.telemetry
                .tiers()
                .set(tier.gauge(), state.used[tier.index()]);
        }
    }

    /// Registers a host-resident batch and enforces the host budget.
    ///
    /// If room cannot be made, the error is returned and the batch is not
    /// kept: the manager holds and counts what it did before the call, but
    /// for batches the eviction had already moved (#115).
    pub async fn insert_host(&self, batch: RecordBatch) -> Result<BatchId, EngineError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let bytes = Self::batch_bytes(&batch);
        let mut state = self.state.lock().await;
        state.entries.insert(
            id,
            Entry {
                tier: Tier::Host,
                bytes,
                host_bytes: bytes,
                last_touch: self.tick(),
                schema: batch.schema(),
                payload: Payload::Host(batch),
            },
        );
        state.used[Tier::Host.index()] += bytes;
        let enforced = self.enforce_host(&mut state).await;
        self.finish_insert(&mut state, id, enforced)
    }

    /// Registers a device-resident batch and enforces the device and host budgets.
    ///
    /// If room cannot be made, the error is returned and the batch is not
    /// kept: the manager holds and counts what it did before the call, but
    /// for batches the eviction had already moved (#115).
    pub async fn insert_device(
        &self,
        resident: Box<dyn DeviceResident>,
        schema: SchemaRef,
    ) -> Result<BatchId, EngineError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let bytes = resident.bytes();
        let mut state = self.state.lock().await;
        state.entries.insert(
            id,
            Entry {
                tier: Tier::Device,
                bytes,
                host_bytes: 0,
                last_touch: self.tick(),
                schema,
                payload: Payload::Device(resident),
            },
        );
        state.used[Tier::Device.index()] += bytes;
        let enforced = match self.enforce_device(&mut state) {
            Ok(()) => self.enforce_host(&mut state).await,
            Err(err) => Err(err),
        };
        self.finish_insert(&mut state, id, enforced)
    }

    /// Ends an insert. One whose budgets could not be enforced forgets the
    /// batch it registered, with its bytes, before returning the error
    /// (#115): the caller never receives the id, so a batch kept registered
    /// could be neither read nor removed, and its bytes would count against
    /// every later insert. Batches the eviction had already moved stay
    /// where it moved them.
    fn finish_insert(
        &self,
        state: &mut State,
        id: BatchId,
        enforced: Result<(), EngineError>,
    ) -> Result<BatchId, EngineError> {
        if enforced.is_err() {
            // The batch is still in memory, never on disk: eviction takes
            // the coldest batch first and this one is the newest, so it is
            // written only once it is the last batch on the host, and
            // writing it empties the host.
            if let Some(entry) = state.entries.remove(&id) {
                let tier = entry.tier.index();
                state.used[tier] = state.used[tier].saturating_sub(entry.bytes);
            }
        }
        self.publish(state);
        enforced.map(|()| id)
    }

    /// Returns a batch, promoting it from disk into host RAM if necessary.
    pub async fn get(&self, id: BatchId) -> Result<RecordBatch, EngineError> {
        let mut state = self.state.lock().await;
        let touch = self.tick();
        let (tier, disk_path, bytes, stored_host_bytes) = {
            let entry = state
                .entries
                .get_mut(&id)
                .ok_or_else(|| EngineError::execution(format!("spill: unknown batch id {id}")))?;
            entry.last_touch = touch;
            match &entry.payload {
                Payload::Host(batch) => return Ok(batch.clone()),
                Payload::Device(resident) => return resident.to_host(),
                Payload::Disk(path) => (entry.tier, path.clone(), entry.bytes, entry.host_bytes),
            }
        };
        debug_assert_eq!(tier, Tier::Disk);
        let batch = self.read_spill(&disk_path).await?;
        let host_bytes = if stored_host_bytes > 0 {
            stored_host_bytes
        } else {
            Self::batch_bytes(&batch)
        };
        if let Some(entry) = state.entries.get_mut(&id) {
            entry.tier = Tier::Host;
            entry.bytes = host_bytes;
            entry.payload = Payload::Host(batch.clone());
        }
        state.used[Tier::Disk.index()] = state.used[Tier::Disk.index()].saturating_sub(bytes);
        state.used[Tier::Host.index()] += host_bytes;
        self.telemetry.spill().record_promotion(host_bytes);
        if let Err(err) = self.store.delete(&disk_path).await {
            tracing::warn!(error = %err, path = %disk_path, "spill file could not be deleted after promotion");
        }
        self.enforce_host(&mut state).await?;
        self.publish(&state);
        Ok(batch)
    }

    /// The tier a batch currently lives in.
    pub async fn tier_of(&self, id: BatchId) -> Option<Tier> {
        self.state.lock().await.entries.get(&id).map(|e| e.tier)
    }

    /// The schema of a registered batch.
    pub async fn schema_of(&self, id: BatchId) -> Option<SchemaRef> {
        self.state
            .lock()
            .await
            .entries
            .get(&id)
            .map(|e| Arc::clone(&e.schema))
    }

    /// Bytes resident per tier: `[device, host, disk]`.
    pub async fn usage(&self) -> [u64; 3] {
        self.state.lock().await.used
    }

    /// Forgets a batch, deleting its spill file if it has one. Returns whether it existed.
    pub async fn remove(&self, id: BatchId) -> Result<bool, EngineError> {
        let mut state = self.state.lock().await;
        let Some(entry) = state.entries.remove(&id) else {
            return Ok(false);
        };
        state.used[entry.tier.index()] = state.used[entry.tier.index()].saturating_sub(entry.bytes);
        if let Payload::Disk(path) = &entry.payload {
            match self.store.delete(path).await {
                Ok(()) | Err(object_store::Error::NotFound { .. }) => {}
                Err(err) => return Err(store_error(err)),
            }
        }
        self.publish(&state);
        Ok(true)
    }

    fn enforce_device(&self, state: &mut State) -> Result<(), EngineError> {
        while state.used[Tier::Device.index()] > self.budget.device_bytes {
            let Some(victim) = state.coldest(Tier::Device) else {
                break;
            };
            let Some(entry) = state.entries.get_mut(&victim) else {
                break;
            };
            let Payload::Device(resident) = &entry.payload else {
                break;
            };
            let batch = resident.to_host()?;
            let device_bytes = entry.bytes;
            let host_bytes = Self::batch_bytes(&batch);
            entry.tier = Tier::Host;
            entry.bytes = host_bytes;
            entry.host_bytes = host_bytes;
            entry.payload = Payload::Host(batch);
            state.used[Tier::Device.index()] =
                state.used[Tier::Device.index()].saturating_sub(device_bytes);
            state.used[Tier::Host.index()] += host_bytes;
            self.telemetry.spill().record_demotion(device_bytes);
            tracing::debug!(
                batch = victim,
                device_bytes,
                host_bytes,
                "demoted device → host"
            );
        }
        Ok(())
    }

    async fn enforce_host(&self, state: &mut State) -> Result<(), EngineError> {
        while state.used[Tier::Host.index()] > self.budget.host_bytes {
            let Some(victim) = state.coldest(Tier::Host) else {
                break;
            };
            let batch = match state.entries.get(&victim).map(|e| &e.payload) {
                Some(Payload::Host(batch)) => batch.clone(),
                _ => break,
            };
            let path = self.spill_path(victim);
            let disk_bytes = self.write_spill(&path, &batch).await?;
            let Some(entry) = state.entries.get_mut(&victim) else {
                break;
            };
            let host_bytes = entry.bytes;
            entry.tier = Tier::Disk;
            entry.bytes = disk_bytes;
            entry.payload = Payload::Disk(path);
            state.used[Tier::Host.index()] =
                state.used[Tier::Host.index()].saturating_sub(host_bytes);
            state.used[Tier::Disk.index()] += disk_bytes;
            self.telemetry.spill().record_demotion(host_bytes);
            tracing::debug!(
                batch = victim,
                host_bytes,
                disk_bytes,
                "demoted host → disk"
            );
        }
        Ok(())
    }

    async fn write_spill(&self, path: &Path, batch: &RecordBatch) -> Result<u64, EngineError> {
        let bytes = ipc::encode_batch(batch)?;
        let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        self.store
            .put(path, PutPayload::from_bytes(bytes))
            .await
            .map_err(store_error)?;
        Ok(len)
    }

    async fn read_spill(&self, path: &Path) -> Result<RecordBatch, EngineError> {
        let result = self.store.get(path).await.map_err(store_error)?;
        let bytes = result.bytes().await.map_err(store_error)?;
        ipc::decode_batch(&bytes)
    }
}

/// A tag for one manager's file names that no other manager will draw, in
/// this process or another one writing to the same store (#114). Every
/// manager numbers its batches from 1, so two on one prefix would otherwise
/// write, read and delete each other's files.
fn instance_tag() -> String {
    static MANAGERS: AtomicU64 = AtomicU64::new(0);
    // `RandomState` keys its hasher from the operating system's randomness,
    // and no two instances share keys: the entropy std offers. The process
    // id and a count of managers make the input distinct as well.
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u32(std::process::id());
    hasher.write_u64(MANAGERS.fetch_add(1, Ordering::Relaxed));
    format!("{:016x}", hasher.finish())
}

/// Bytes in one Arrow view: a string's length, then its first bytes, or the
/// buffer and offset where it lives when it is too long to be inlined.
const VIEW_LEN: usize = 16;

/// One column's share of [`SpillManager::batch_bytes`].
fn column_bytes(column: &ArrayRef) -> usize {
    let data = column.to_data();
    data.get_slice_memory_size()
        .map(|bytes| bytes.saturating_add(view_bytes(&data)))
        .unwrap_or_else(|_| column.get_array_memory_size())
}

/// The strings `data`'s view arrays point to, which `get_slice_memory_size`
/// leaves out: it counts each view's 16 bytes, not a string too long to be
/// inlined in one.
fn view_bytes(data: &ArrayData) -> usize {
    let own = match data.data_type() {
        DataType::Utf8View | DataType::BinaryView => data.buffers().first().map_or(0, |views| {
            let start = data.offset().saturating_mul(VIEW_LEN);
            let end = start.saturating_add(data.len().saturating_mul(VIEW_LEN));
            views.as_slice().get(start..end).map_or(0, |views| {
                views
                    .as_chunks::<VIEW_LEN>()
                    .0
                    .iter()
                    // A view starts with its string's length, little-endian.
                    .map(|&[a, b, c, d, ..]| u32::from_le_bytes([a, b, c, d]))
                    .filter(|&len| len > MAX_INLINE_VIEW_LEN)
                    .fold(0_usize, |sum, len| {
                        sum.saturating_add(usize::try_from(len).unwrap_or(usize::MAX))
                    })
            })
        }),
        _ => 0,
    };
    data.child_data()
        .iter()
        .fold(own, |sum, child| sum.saturating_add(view_bytes(child)))
}

impl Drop for SpillManager {
    fn drop(&mut self) {
        let paths: Vec<Path> = self
            .state
            .get_mut()
            .entries
            .values()
            .filter_map(|entry| match &entry.payload {
                Payload::Disk(path) => Some(path.clone()),
                Payload::Device(_) | Payload::Host(_) => None,
            })
            .collect();
        if !paths.is_empty() {
            delete_on_drop(Arc::clone(&self.store), paths);
        }
    }
}

/// How long dropping a manager waits for its spill files to be deleted
/// before it lets the deletes finish in the background.
const DROP_DELETE_WAIT: Duration = Duration::from_secs(30);

/// Deletes the spill files of a dropped manager (#117).
///
/// `drop` cannot await, so the deletes run on a thread of their own with a
/// runtime of its own, and the drop waits for them: blocking in place panics
/// on a current-thread runtime, and a task spawned on the caller's runtime
/// would not have run by the time the drop returns, or at all if that
/// runtime is shutting down. The wait is bounded because a store whose
/// deletes need the very runtime the dropping thread is blocking (a network
/// store on a current-thread runtime) would hang the drop; past the bound
/// the deletes carry on in the background.
fn delete_on_drop(store: Arc<dyn ObjectStore>, paths: Vec<Path>) {
    let files = paths.len();
    let (done, finished) = mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("oxide-spill-drop".to_owned())
        .spawn(move || {
            match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime.block_on(async {
                    for path in &paths {
                        match store.delete(path).await {
                            Ok(()) | Err(object_store::Error::NotFound { .. }) => {}
                            Err(err) => tracing::warn!(
                                error = %err,
                                path = %path,
                                "spill file left behind: its manager was dropped and it could not be deleted"
                            ),
                        }
                    }
                }),
                Err(err) => tracing::warn!(
                    error = %err,
                    files,
                    "spill files left behind: no runtime to delete them with"
                ),
            }
            // Fails only once the drop has stopped waiting.
            let _ = done.send(());
        });
    match spawned {
        Ok(thread) => match finished.recv_timeout(DROP_DELETE_WAIT) {
            Ok(()) => {
                // The runtime is gone by now; this only reaps the thread.
                let _ = thread.join();
            }
            Err(mpsc::RecvTimeoutError::Timeout) => tracing::warn!(
                files,
                "dropped spill manager: its spill files are still being deleted after {} s; \
                 carrying on in the background",
                DROP_DELETE_WAIT.as_secs()
            ),
            Err(mpsc::RecvTimeoutError::Disconnected) => tracing::warn!(
                files,
                "spill files may be left behind: the thread deleting them panicked"
            ),
        },
        Err(err) => tracing::warn!(
            error = %err,
            files,
            "spill files left behind: no thread to delete them on"
        ),
    }
}

impl fmt::Debug for SpillManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillManager")
            .field("budget", &self.budget)
            .field("prefix", &self.prefix)
            .field("tag", &self.tag)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use object_store::local::LocalFileSystem;

    use super::*;

    fn batch(start: i64, rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let values: Vec<Option<i64>> = (0..rows)
            .map(|i| {
                if i % 7 == 3 {
                    None
                } else {
                    Some(start + i as i64)
                }
            })
            .collect();
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap()
    }

    fn manager(dir: &std::path::Path, device_bytes: u64, host_bytes: u64) -> Arc<SpillManager> {
        let store: Arc<dyn ObjectStore> = Arc::new(LocalFileSystem::new_with_prefix(dir).unwrap());
        SpillManager::new(
            SpillBudget {
                device_bytes,
                host_bytes,
            },
            store,
            Path::from("spill"),
            TelemetryHub::new(),
        )
    }

    #[tokio::test]
    async fn cold_batches_spill_to_disk_and_come_back_identical() {
        let dir = tempfile::tempdir().unwrap();
        let b0 = batch(0, 256);
        let per_batch = SpillManager::batch_bytes(&b0);
        // Room for two batches in host RAM, not three.
        let mgr = manager(dir.path(), 0, per_batch * 2 + per_batch / 2);

        let id0 = mgr.insert_host(b0.clone()).await.unwrap();
        let id1 = mgr.insert_host(batch(1000, 256)).await.unwrap();
        let id2 = mgr.insert_host(batch(2000, 256)).await.unwrap();

        assert_eq!(
            mgr.tier_of(id0).await,
            Some(Tier::Disk),
            "coldest batch spills"
        );
        assert_eq!(mgr.tier_of(id1).await, Some(Tier::Host));
        assert_eq!(mgr.tier_of(id2).await, Some(Tier::Host));
        let files: Vec<_> = std::fs::read_dir(dir.path().join("spill"))
            .unwrap()
            .collect();
        assert_eq!(files.len(), 1);

        let usage = mgr.usage().await;
        assert_eq!(usage[1], per_batch * 2);
        assert!(usage[2] > 0);
        let snap = mgr.telemetry().snapshot();
        assert_eq!(snap.tiers.host_bytes, per_batch * 2);
        assert_eq!(snap.spill.demotions, 1);

        // Reading the spilled batch promotes it (and evicts the now-coldest one).
        let back = mgr.get(id0).await.unwrap();
        assert_eq!(back, b0);
        assert_eq!(mgr.tier_of(id0).await, Some(Tier::Host));
        assert_eq!(mgr.tier_of(id1).await, Some(Tier::Disk));
        let snap = mgr.telemetry().snapshot();
        assert_eq!(snap.spill.promotions, 1);
        assert_eq!(snap.spill.demotions, 2);

        assert!(mgr.remove(id1).await.unwrap());
        assert!(!mgr.remove(id1).await.unwrap());
        let files: Vec<_> = std::fs::read_dir(dir.path().join("spill"))
            .unwrap()
            .collect();
        assert!(files.is_empty(), "spill file deleted on remove");
        assert!(mgr.get(id1).await.is_err());
    }

    #[derive(Debug)]
    struct FakeResident {
        batch: RecordBatch,
        bytes: u64,
    }

    impl DeviceResident for FakeResident {
        fn bytes(&self) -> u64 {
            self.bytes
        }
        fn to_host(&self) -> Result<RecordBatch, EngineError> {
            Ok(self.batch.clone())
        }
    }

    #[tokio::test]
    async fn device_tier_demotes_to_host_then_disk() {
        let dir = tempfile::tempdir().unwrap();
        let b = batch(0, 64);
        let host_bytes = SpillManager::batch_bytes(&b);
        // Device holds one resident batch; host holds one batch.
        let mgr = manager(dir.path(), 4096, host_bytes + host_bytes / 2);

        let d0 = mgr
            .insert_device(
                Box::new(FakeResident {
                    batch: b.clone(),
                    bytes: 4096,
                }),
                b.schema(),
            )
            .await
            .unwrap();
        assert_eq!(mgr.tier_of(d0).await, Some(Tier::Device));
        assert_eq!(mgr.get(d0).await.unwrap(), b);

        let d1 = mgr
            .insert_device(
                Box::new(FakeResident {
                    batch: batch(100, 64),
                    bytes: 4096,
                }),
                b.schema(),
            )
            .await
            .unwrap();
        // d0 was colder → demoted to host; device now holds d1 only.
        assert_eq!(mgr.tier_of(d0).await, Some(Tier::Host));
        assert_eq!(mgr.tier_of(d1).await, Some(Tier::Device));
        assert_eq!(mgr.usage().await[0], 4096);

        let h = mgr.insert_host(batch(200, 64)).await.unwrap();
        // Host budget fits one batch: d0 (colder) goes to disk.
        assert_eq!(mgr.tier_of(d0).await, Some(Tier::Disk));
        assert_eq!(mgr.tier_of(h).await, Some(Tier::Host));
        assert_eq!(mgr.get(d0).await.unwrap(), b);
        assert_eq!(mgr.telemetry().snapshot().tiers.device_bytes, 4096);
    }

    #[tokio::test]
    async fn schema_and_unknown_ids() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = manager(dir.path(), 0, u64::MAX);
        let id = mgr.insert_host(batch(0, 4)).await.unwrap();
        assert_eq!(mgr.schema_of(id).await.unwrap().fields().len(), 1);
        assert_eq!(mgr.tier_of(999).await, None);
        assert!(mgr.get(999).await.is_err());
    }

    // #114: ids start at 1 in every manager, so the file names must carry
    // something only this manager has.
    #[tokio::test]
    async fn two_managers_on_one_store_and_prefix_keep_their_own_batches() {
        let dir = tempfile::tempdir().unwrap();
        // A host budget of zero spills every batch as it is inserted.
        let a = manager(dir.path(), 0, 0);
        let b = manager(dir.path(), 0, 0);
        let mine = batch(0, 64);
        let theirs = batch(1_000_000, 64);
        let ia = a.insert_host(mine.clone()).await.unwrap();
        let ib = b.insert_host(theirs.clone()).await.unwrap();
        assert_eq!(a.tier_of(ia).await, Some(Tier::Disk));
        assert_eq!(b.tier_of(ib).await, Some(Tier::Disk));
        assert_eq!(a.get(ia).await.unwrap(), mine);
        assert_eq!(b.get(ib).await.unwrap(), theirs);
    }

    // #115
    #[tokio::test]
    async fn a_failed_insert_leaves_nothing_registered_or_counted() {
        let dir = tempfile::tempdir().unwrap();
        // A file where the spill directory belongs: every spill write fails,
        // whoever runs the test (permissions would not stop root).
        std::fs::write(dir.path().join("spill"), b"not a directory").unwrap();
        let first = batch(0, 256);
        let per_batch = SpillManager::batch_bytes(&first);
        // Room for one batch: a second one must evict the first to disk.
        let mgr = manager(dir.path(), 0, per_batch + per_batch / 2);
        let kept = mgr.insert_host(first).await.unwrap();

        let unchanged = |what: &str, usage: [u64; 3], entries: usize, gauge: u64| {
            assert_eq!(usage, [0, per_batch, 0], "{what}: usage");
            assert_eq!(
                entries, 1,
                "{what}: only the batch whose id the caller holds"
            );
            assert_eq!(gauge, per_batch, "{what}: the host gauge follows usage");
        };
        for start in [1000, 2000, 3000] {
            let err = mgr.insert_host(batch(start, 256)).await.unwrap_err();
            assert!(matches!(err, EngineError::Io(_)), "{err}");
            unchanged(
                "insert_host",
                mgr.usage().await,
                mgr.state.lock().await.entries.len(),
                mgr.telemetry().snapshot().tiers.host_bytes,
            );
        }
        // A device batch over a zero device budget is demoted to the host,
        // which then has to evict — and cannot.
        let err = mgr
            .insert_device(
                Box::new(FakeResident {
                    batch: batch(4000, 256),
                    bytes: 4096,
                }),
                batch(0, 1).schema(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::Io(_)), "{err}");
        unchanged(
            "insert_device",
            mgr.usage().await,
            mgr.state.lock().await.entries.len(),
            mgr.telemetry().snapshot().tiers.host_bytes,
        );
        assert_eq!(mgr.tier_of(kept).await, Some(Tier::Host));
    }

    #[tokio::test]
    async fn a_failed_insert_publishes_what_its_eviction_did_move() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("spill"), b"not a directory").unwrap();
        let first = batch(0, 256);
        let per_batch = SpillManager::batch_bytes(&first);
        // Room for one device batch and one host batch.
        let mgr = manager(dir.path(), 4096, per_batch + per_batch / 2);
        let device = |start| {
            Box::new(FakeResident {
                batch: batch(start, 256),
                bytes: 4096,
            })
        };
        let kept = mgr.insert_host(first).await.unwrap();
        let moved = mgr
            .insert_device(device(1000), batch(0, 1).schema())
            .await
            .unwrap();
        // A second device batch demotes `moved` to the host, which must then
        // evict `kept` to disk, and cannot. The demotion stands.
        let err = mgr
            .insert_device(device(2000), batch(0, 1).schema())
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::Io(_)), "{err}");
        assert_eq!(mgr.tier_of(kept).await, Some(Tier::Host));
        assert_eq!(mgr.tier_of(moved).await, Some(Tier::Host));
        assert_eq!(mgr.usage().await, [0, 2 * per_batch, 0]);
        let tiers = mgr.telemetry().snapshot().tiers;
        assert_eq!(
            [tiers.device_bytes, tiers.host_bytes, tiers.disk_bytes],
            [0, 2 * per_batch, 0],
            "the gauges follow usage"
        );
    }

    // #116
    #[tokio::test]
    async fn slices_of_one_batch_are_charged_their_own_rows() {
        let dir = tempfile::tempdir().unwrap();
        let whole = batch(0, 100_000);
        let whole_bytes = SpillManager::batch_bytes(&whole);
        // Room for the batch once and a half: its four quarters fit.
        let mgr = manager(dir.path(), 0, whole_bytes + whole_bytes / 2);
        let quarter = 25_000;
        let mut ids = Vec::new();
        for i in 0..4 {
            let slice = whole.slice(i * quarter, quarter);
            ids.push(mgr.insert_host(slice).await.unwrap());
        }
        for id in ids {
            assert_eq!(mgr.tier_of(id).await, Some(Tier::Host), "batch {id}");
        }
        assert_eq!(mgr.usage().await, [0, whole_bytes, 0]);
        assert!(!dir.path().join("spill").exists(), "nothing spilled");
    }

    #[test]
    fn a_view_column_is_charged_the_strings_it_points_to() {
        use arrow::array::StringViewArray;

        // 100-byte strings are too long to inline in a 16-byte view.
        let strings: Vec<String> = (0..1000).map(|i| format!("{i:0>100}")).collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::Utf8View,
            false,
        )]));
        let whole = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringViewArray::from_iter_values(&strings))],
        )
        .unwrap();
        let whole_bytes = SpillManager::batch_bytes(&whole);
        assert!(
            whole_bytes >= 100 * 1000,
            "{whole_bytes}: the strings count"
        );
        let quarters: u64 = (0..4)
            .map(|i| SpillManager::batch_bytes(&whole.slice(i * 250, 250)))
            .sum();
        assert_eq!(quarters, whole_bytes);
    }

    // #117
    /// The names of the files in the spill directory, sorted.
    fn spill_files(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir.join("spill"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    /// A spill directory that already holds a file of someone else's.
    fn occupied_spill_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("spill")).unwrap();
        std::fs::write(dir.path().join("spill/keep.arrow"), b"not the manager's").unwrap();
        dir
    }

    /// A manager with three batches spilled to `dir`.
    async fn spill_three(dir: &std::path::Path) -> Arc<SpillManager> {
        let mgr = manager(dir, 0, 0);
        for start in [0, 1000, 2000] {
            let id = mgr.insert_host(batch(start, 64)).await.unwrap();
            assert_eq!(mgr.tier_of(id).await, Some(Tier::Disk));
        }
        assert_eq!(spill_files(dir).len(), 4);
        mgr
    }

    #[test]
    fn a_manager_dropped_outside_a_runtime_deletes_its_spill_files() {
        let dir = occupied_spill_dir();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mgr = rt.block_on(spill_three(dir.path()));
        drop(mgr);
        assert_eq!(spill_files(dir.path()), ["keep.arrow"]);
    }

    #[tokio::test]
    async fn a_manager_dropped_on_a_current_thread_runtime_deletes_its_spill_files() {
        let dir = occupied_spill_dir();
        drop(spill_three(dir.path()).await);
        assert_eq!(spill_files(dir.path()), ["keep.arrow"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_manager_dropped_on_a_multi_thread_runtime_deletes_its_spill_files() {
        let dir = occupied_spill_dir();
        drop(spill_three(dir.path()).await);
        assert_eq!(spill_files(dir.path()), ["keep.arrow"]);
    }
}
