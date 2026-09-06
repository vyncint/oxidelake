//! Deterministic demo-dataset generator behind `oxide gen-data`.
//!
//! The table exercises every v1 operator: `id` is a unique `Int64` Bloom-filter
//! probe target, `k` a low-cardinality `Int64` group/join key, `v` a `Float64`
//! measure in exact multiples of 0.25 (so sums compare exactly), `s` a string
//! column that stays on the CPU, and `emb` a `FixedSizeList<Float32, 8>`
//! embedding for the vector-distance operator. Generation is seeded and
//! chunked, so any row count re-generates identically on every platform.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use oxidelake_core::EngineError;

use crate::parquet_io::ParquetFileWriter;
use crate::parquet_opts::ParquetWriteOptions;

/// Dimension of the `emb` vector column.
pub const EMBEDDING_DIM: usize = 8;

/// Distinct values in the `k` group/join key.
pub const KEY_CARDINALITY: i64 = 100;

/// Rows per generated [`RecordBatch`]. [`write_demo_table`] streams one chunk
/// at a time into the Parquet writer, so peak memory is one chunk plus one row
/// group regardless of `--rows`.
pub const CHUNK_ROWS: u64 = 65_536;

/// The 13 values the `s` column cycles through.
const S_VALUES: [&str; 13] = [
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12",
];

/// SplitMix64: a tiny, well-mixed, platform-independent PRNG. Using it keeps
/// `rand` a test-only dependency and pins the dataset bytes to the seed alone.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The demo table's schema.
pub fn demo_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
        Field::new("s", DataType::Utf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                EMBEDDING_DIM as i32,
            ),
            false,
        ),
    ]))
}

fn gen_chunk(
    schema: &SchemaRef,
    first_id: u64,
    rows: u64,
    seed: u64,
) -> Result<RecordBatch, EngineError> {
    // Reseed per chunk from (seed, first_id) so a chunk's bytes do not depend
    // on how earlier chunks were sized.
    let mut state = seed ^ first_id.wrapping_mul(0xA076_1D64_78BD_642F);
    let n = usize::try_from(rows).unwrap_or(usize::MAX);
    let mut id = Vec::with_capacity(n);
    let mut k = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    let mut emb = Vec::with_capacity(n * EMBEDDING_DIM);
    for row in 0..rows {
        id.push(i64::try_from(first_id + row).unwrap_or(i64::MAX));
        let r = splitmix64(&mut state);
        k.push((!r.is_multiple_of(50)).then_some((r >> 8) as i64 % KEY_CARDINALITY));
        let r = splitmix64(&mut state);
        v.push((!r.is_multiple_of(40)).then(|| f64::from((r >> 8) as u32 % 100) / 4.0));
        for _ in 0..EMBEDDING_DIM {
            let r = splitmix64(&mut state);
            emb.push(f32::from((r % 2000) as u16) / 100.0 - 10.0);
        }
    }
    // `s` cycles through 13 fixed values: build it straight from static strings
    // instead of one `String` per row.
    let s = StringArray::from_iter_values(
        (0..rows).map(|row| S_VALUES[usize::try_from((first_id + row) % 13).unwrap_or(0)]),
    );
    let emb = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        EMBEDDING_DIM as i32,
        Arc::new(Float32Array::from(emb)) as ArrayRef,
        None,
    );
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(Int64Array::from(id)),
            Arc::new(Int64Array::from(k)),
            Arc::new(Float64Array::from(v)),
            Arc::new(s),
            Arc::new(emb),
        ],
    )
    .map_err(|e| EngineError::execution(format!("datagen batch: {e}")))
}

/// The table as a lazy sequence of chunks (deterministic; each chunk is
/// reseeded from its first row id, so the sequence never depends on how
/// earlier chunks were sized).
pub fn demo_chunks(rows: u64, seed: u64) -> impl Iterator<Item = Result<RecordBatch, EngineError>> {
    let schema = demo_schema();
    (0..rows)
        .step_by(usize::try_from(CHUNK_ROWS).unwrap_or(usize::MAX))
        .map(move |first| {
            let n = CHUNK_ROWS.min(rows - first);
            gen_chunk(&schema, first, n, seed)
        })
}

/// Generates the whole table in memory (tests and small datasets; the CLI
/// streams [`demo_chunks`] instead).
pub fn demo_batches(rows: u64, seed: u64) -> Result<Vec<RecordBatch>, EngineError> {
    demo_chunks(rows, seed).collect()
}

/// What [`write_demo_table`] produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemoTable {
    /// The Parquet file.
    pub path: PathBuf,
    /// Rows written.
    pub rows: u64,
}

/// Writes the demo table to `<dir>/t.parquet` with Bloom filters on `id` and
/// `k` and page statistics on, creating `dir` if needed.
pub fn write_demo_table(
    dir: &Path,
    rows: u64,
    seed: u64,
    options: &ParquetWriteOptions,
) -> Result<DemoTable, EngineError> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("t.parquet");
    let mut writer = ParquetFileWriter::create(&path, demo_schema(), options)?;
    for chunk in demo_chunks(rows, seed) {
        writer.write(&chunk?)?;
    }
    let written = writer.close()?;
    Ok(DemoTable {
        path,
        rows: written,
    })
}

/// The writer options `oxide gen-data` uses: Bloom filters on the `id` probe
/// column and the `k` join/filter key, `row_group_rows` from the caller.
pub fn demo_write_options(
    row_group_rows: usize,
    compression: crate::Compression,
) -> ParquetWriteOptions {
    ParquetWriteOptions {
        row_group_rows,
        compression,
        ..ParquetWriteOptions::default()
    }
    .with_bloom_filter("id")
    .with_bloom_filter("k")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::*;
    use crate::Compression;

    #[test]
    fn generation_is_deterministic_and_chunk_independent() {
        let a = demo_batches(1000, 42).unwrap();
        let b = demo_batches(1000, 42).unwrap();
        assert_eq!(a, b);
        let c = demo_batches(1000, 43).unwrap();
        assert_ne!(a, c);
        // Chunk reseeding: the first rows of a longer run equal a shorter run.
        let long = demo_batches(100, 7).unwrap();
        let short = demo_batches(40, 7).unwrap();
        assert_eq!(long[0].slice(0, 40), short[0].slice(0, 40));
    }

    #[test]
    fn writes_bloom_filtered_row_groups() {
        let dir = tempfile::tempdir().unwrap();
        let opts = demo_write_options(500, Compression::None);
        let table = write_demo_table(dir.path(), 2000, 1, &opts).unwrap();
        assert_eq!(table.rows, 2000);
        let reader = SerializedFileReader::new(std::fs::File::open(&table.path).unwrap()).unwrap();
        let meta = reader.metadata();
        assert_eq!(meta.num_row_groups(), 4);
        let schema = meta.file_metadata().schema_descr();
        let id_idx = (0..schema.num_columns())
            .find(|&i| schema.column(i).name() == "id")
            .unwrap();
        assert!(
            meta.row_group(0)
                .column(id_idx)
                .bloom_filter_offset()
                .is_some()
        );
    }
}
