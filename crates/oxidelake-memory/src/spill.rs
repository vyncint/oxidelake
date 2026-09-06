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
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
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
    /// Logical host size, fixed at first host residency. Reused on promotion
    /// because IPC-decoded batches over-report `get_array_memory_size`
    /// (every column counts the shared message buffer).
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
pub struct SpillManager {
    budget: SpillBudget,
    store: Arc<dyn ObjectStore>,
    prefix: Path,
    telemetry: Arc<TelemetryHub>,
    state: Mutex<State>,
    next_id: AtomicU64,
    clock: AtomicU64,
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

    /// Host bytes a batch occupies.
    pub fn batch_bytes(batch: &RecordBatch) -> u64 {
        u64::try_from(batch.get_array_memory_size()).unwrap_or(u64::MAX)
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn spill_path(&self, id: BatchId) -> Path {
        Path::from(format!("{}/batch-{id}.arrow", self.prefix))
    }

    fn publish(&self, state: &State) {
        for tier in [Tier::Device, Tier::Host, Tier::Disk] {
            self.telemetry
                .tiers()
                .set(tier.gauge(), state.used[tier.index()]);
        }
    }

    /// Registers a host-resident batch and enforces the host budget.
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
        self.enforce_host(&mut state).await?;
        self.publish(&state);
        Ok(id)
    }

    /// Registers a device-resident batch and enforces the device and host budgets.
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
        self.enforce_device(&mut state)?;
        self.enforce_host(&mut state).await?;
        self.publish(&state);
        Ok(id)
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

impl fmt::Debug for SpillManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillManager")
            .field("budget", &self.budget)
            .field("prefix", &self.prefix)
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
}
