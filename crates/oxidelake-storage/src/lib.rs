//! Storage layer (docs/SPEC.md §2.4): Parquet writer and pruning configuration,
//! Arrow IPC spill/cache files, and the object-store IO layer.
//!
//! There is no private file format (ADR-0014). Lake data is Parquet read
//! through DataFusion, whose statistics, page-index and Bloom-filter pruning
//! this crate configures and *proves* (see [`scan_pruning_metrics`]); spill and
//! cache files are Arrow IPC with 64-byte-aligned buffers; all IO goes through
//! [`object_store::ObjectStore`] — `LocalFileSystem` by default, the io_uring
//! implementation behind the `io-uring` feature on Linux.
//!
//! Dependency direction: depends on `oxidelake-core` and `oxidelake-memory`.

pub mod datagen;
pub mod errors;
pub mod ipc;
pub mod parquet_io;
pub mod parquet_opts;
pub mod session;
pub mod store;

#[cfg(all(feature = "io-uring", target_os = "linux"))]
pub mod uring;

pub use datagen::{
    CHUNK_ROWS, DemoTable, demo_batches, demo_chunks, demo_schema, demo_write_options,
    write_demo_table,
};
pub use errors::classify_error;
pub use ipc::{read_ipc_file, write_ipc_file};
pub use parquet_io::{ParquetFileWriter, write_parquet};
pub use parquet_opts::{Compression, ParquetWriteOptions};
pub use session::{
    GPU_BATCH_SIZE, PruningSummary, pruning_session, register_parquet_table, scan_pruning_metrics,
    with_gpu_batch_size, with_pruning, with_pruning_disabled,
};
pub use store::{default_object_store, register_local_store};

#[cfg(all(feature = "io-uring", target_os = "linux"))]
pub use uring::UringLocalFileSystem;
