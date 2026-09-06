# ADR-0014: Parquet and Arrow IPC instead of a custom `.oxide` file format

- Status: accepted
- Date: 2026-08-25

## Context

The plan owned a columnar format: an `OXIDELAKE\x01` header, 128-byte-aligned chunks of raw Arrow buffers, a postcard footer with per-chunk zonemaps, dictionaries and split-block Bloom filters, and a custom reader with predicate pushdown. Everything it promised already exists in Parquet — row-group statistics, page index, split-block Bloom filters, dictionaries — and DataFusion already prunes on all of them. "Lakehouse" implies reading data that exists (Parquet today, Iceberg/Delta later), which a private format cannot do. Arrow IPC already provides raw-Arrow-buffer, zero-decode files for hot data. Alternatives considered: **Lance** (Rust, vector indices; coherent today at lance 10 → DataFusion 54 / arrow 58, but a third coupling to DataFusion's major and CPU-side ANN — deferred to v2), **Vortex** (young, moving quickly).

## Decision

Lake data is Parquet through DataFusion's listing tables, with pruning proven by scan metrics (row groups pruned by statistics and by Bloom filter) and result equality against a pruning-disabled run. Spill files and the optional hot cache are Arrow IPC files with 64-byte buffer alignment and no compression. The IO abstraction is `object_store::ObjectStore` — the trait DataFusion reads through — with `LocalFileSystem` by default and a `UringLocalFileSystem` implementation behind the `io-uring` feature, replacing the private `StorageIo` trait and the macOS mmap path. `oxidelake-storage` owns writer properties, session pruning settings, the IPC spill files, the object store and the pruning proofs.

## Consequences

Phase 4 loses the format writer, reader, footer, zonemap and Bloom-filter implementation (roughly 2–3k lines) and gains interoperability with existing lakes. `postcard` stays only for codec payloads; `xxhash-rust` and `memmap2` are dropped. The zero-decode property is kept where it pays (spill, cache), not for cold lake data, which DataFusion's Parquet reader decodes on the CPU; GPU-side Parquet decoding and GPUDirect Storage are explicitly out of scope.
