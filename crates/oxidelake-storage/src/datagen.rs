//! Deterministic demo-dataset generator behind `oxide gen-data`.
//!
//! The table exercises every v1 operator: `id` is a unique `Int64` Bloom-filter
//! probe target, `k` a low-cardinality `Int64` group/join key, `v` a `Float64`
//! measure in exact multiples of 0.25 (so sums compare exactly), `s` a string
//! column that stays on the CPU, and `emb` a `FixedSizeList<Float32, 8>`
//! embedding for the vector-distance operator. Generation is seeded and
//! chunked, so any row count re-generates identically on every platform.
//!
//! [`DemoOptions::with_special_floats`] (`oxide gen-data --special-floats`)
//! puts IEEE special values into `v` on every [`SPECIAL_FLOAT_EVERY`]th row,
//! the values where backends' float comparisons and aggregates can disagree
//! (#28). It is an overlay on the same rows: every other value, and the
//! default table itself, stay exactly as they were.

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

/// With [`DemoOptions::with_special_floats`], every row whose `id` is a
/// multiple of this holds one of [`SPECIAL_FLOATS`] in `v`.
pub const SPECIAL_FLOAT_EVERY: u64 = 16;

/// The values the special-float rows cycle through, in order: NaN, -0.0,
/// +0.0, +inf, -inf, and NaN with the sign bit set — the NaN an x86 `0.0 /
/// 0.0` produces, and the least value in IEEE totalOrder where NaN is the
/// greatest.
pub const SPECIAL_FLOATS: [f64; 6] = [
    f64::NAN,
    -0.0,
    0.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    -f64::NAN,
];

/// What goes into the demo table besides its size and seed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DemoOptions {
    /// See [`Self::with_special_floats`].
    pub special_floats: bool,
}

impl DemoOptions {
    /// The default table.
    pub const fn new() -> Self {
        Self {
            special_floats: false,
        }
    }

    /// Puts one of [`SPECIAL_FLOATS`] into `v` on every
    /// [`SPECIAL_FLOAT_EVERY`]th row (`id` 0, 16, 32, … in turn), replacing
    /// whatever value — or NULL — the row had. Nothing else changes.
    #[must_use]
    pub const fn with_special_floats(mut self, on: bool) -> Self {
        self.special_floats = on;
        self
    }
}

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

/// The special value row `row_id` holds under
/// [`DemoOptions::with_special_floats`], if it holds one.
fn special_float(row_id: u64) -> Option<f64> {
    if !row_id.is_multiple_of(SPECIAL_FLOAT_EVERY) {
        return None;
    }
    let turn = (row_id / SPECIAL_FLOAT_EVERY) % SPECIAL_FLOATS.len() as u64;
    SPECIAL_FLOATS.get(usize::try_from(turn).ok()?).copied()
}

fn gen_chunk(
    schema: &SchemaRef,
    first_id: u64,
    rows: u64,
    seed: u64,
    demo: DemoOptions,
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
        let row_id = first_id + row;
        id.push(i64::try_from(row_id).unwrap_or(i64::MAX));
        let r = splitmix64(&mut state);
        k.push((!r.is_multiple_of(50)).then_some((r >> 8) as i64 % KEY_CARDINALITY));
        let r = splitmix64(&mut state);
        let drawn = (!r.is_multiple_of(40)).then(|| f64::from((r >> 8) as u32 % 100) / 4.0);
        // Replaced after the draw, never instead of it: the PRNG stream, and
        // so every other value in the table, is the same with or without.
        let special = if demo.special_floats {
            special_float(row_id)
        } else {
            None
        };
        v.push(special.or(drawn));
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
    demo_chunks_with(rows, seed, DemoOptions::new())
}

/// [`demo_chunks`] with [`DemoOptions`].
pub fn demo_chunks_with(
    rows: u64,
    seed: u64,
    demo: DemoOptions,
) -> impl Iterator<Item = Result<RecordBatch, EngineError>> {
    let schema = demo_schema();
    (0..rows)
        .step_by(usize::try_from(CHUNK_ROWS).unwrap_or(usize::MAX))
        .map(move |first| {
            let n = CHUNK_ROWS.min(rows - first);
            gen_chunk(&schema, first, n, seed, demo)
        })
}

/// Generates the whole table in memory (tests and small datasets; the CLI
/// streams [`demo_chunks`] instead).
pub fn demo_batches(rows: u64, seed: u64) -> Result<Vec<RecordBatch>, EngineError> {
    demo_chunks(rows, seed).collect()
}

/// [`demo_batches`] with [`DemoOptions`].
pub fn demo_batches_with(
    rows: u64,
    seed: u64,
    demo: DemoOptions,
) -> Result<Vec<RecordBatch>, EngineError> {
    demo_chunks_with(rows, seed, demo).collect()
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
/// `k` and page statistics on, creating `dir` if needed. An existing
/// `t.parquet` is replaced only once the new one is complete, and invalid
/// `options` are refused before anything is created (#119).
pub fn write_demo_table(
    dir: &Path,
    rows: u64,
    seed: u64,
    options: &ParquetWriteOptions,
) -> Result<DemoTable, EngineError> {
    write_demo_table_with(dir, rows, seed, DemoOptions::new(), options)
}

/// [`write_demo_table`] with [`DemoOptions`].
pub fn write_demo_table_with(
    dir: &Path,
    rows: u64,
    seed: u64,
    demo: DemoOptions,
    options: &ParquetWriteOptions,
) -> Result<DemoTable, EngineError> {
    // `ParquetFileWriter::create` checks them too, but only after the
    // directory below exists; a refused run should leave nothing behind.
    options.writer_properties()?;
    std::fs::create_dir_all(dir)?;
    let path = dir.join("t.parquet");
    let mut writer = ParquetFileWriter::create(&path, demo_schema(), options)?;
    for chunk in demo_chunks_with(rows, seed, demo) {
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

    /// FNV-1a over every generated value (`v` and `emb` by bit pattern, NULL
    /// as its own marker): the table's content, independent of how Arrow or
    /// Parquet happen to lay it out.
    fn content_digest(batches: &[RecordBatch]) -> u64 {
        use arrow::array::{Array, AsArray};
        use arrow::datatypes::{Float32Type, Float64Type, Int64Type};
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        for batch in batches {
            let id = batch.column(0).as_primitive::<Int64Type>();
            let k = batch.column(1).as_primitive::<Int64Type>();
            let v = batch.column(2).as_primitive::<Float64Type>();
            let s = batch.column(3).as_string::<i32>();
            let emb = batch.column(4).as_fixed_size_list();
            let emb = emb.values().as_primitive::<Float32Type>();
            for row in 0..batch.num_rows() {
                eat(&id.value(row).to_le_bytes());
                if k.is_null(row) {
                    eat(b"null");
                } else {
                    eat(&k.value(row).to_le_bytes());
                }
                if v.is_null(row) {
                    eat(b"null");
                } else {
                    eat(&v.value(row).to_bits().to_le_bytes());
                }
                eat(s.value(row).as_bytes());
                for e in 0..EMBEDDING_DIM {
                    eat(&emb.value(row * EMBEDDING_DIM + e).to_bits().to_le_bytes());
                }
            }
        }
        hash
    }

    /// The default table is pinned: what `oxide gen-data` writes must not
    /// move unless someone means it to (a test elsewhere, or a user's own
    /// comparison, may depend on the exact rows). Recorded from 0.2.0's
    /// generator; two chunks, so the per-chunk reseeding is covered.
    #[test]
    fn the_default_table_is_unchanged() {
        let rows = CHUNK_ROWS + 1000;
        assert_eq!(
            content_digest(&demo_batches(rows, 42).unwrap()),
            13_955_388_935_552_808_329,
            "the default demo table changed"
        );
    }

    /// The special-float table is the default table with `v` replaced on
    /// every sixteenth row, cycling through all six values — so `oxide
    /// gen-data --special-floats` pins how a backend handles each of them.
    #[test]
    fn special_floats_overlay_the_default_table() {
        use arrow::array::{Array, AsArray};
        use arrow::datatypes::Float64Type;
        let rows = CHUNK_ROWS + 1000;
        let plain = demo_batches(rows, 42).unwrap();
        let special =
            demo_batches_with(rows, 42, DemoOptions::new().with_special_floats(true)).unwrap();
        assert_eq!(plain.len(), special.len());
        let mut seen = [0usize; SPECIAL_FLOATS.len()];
        for (p, s) in plain.iter().zip(&special) {
            // Every column but `v` is untouched.
            for column in [0, 1, 3, 4] {
                assert_eq!(p.column(column), s.column(column), "column {column}");
            }
            let ids = s.column(0).as_primitive::<arrow::datatypes::Int64Type>();
            let (pv, sv) = (
                p.column(2).as_primitive::<Float64Type>(),
                s.column(2).as_primitive::<Float64Type>(),
            );
            for row in 0..s.num_rows() {
                let id = ids.value(row).unsigned_abs();
                if id.is_multiple_of(SPECIAL_FLOAT_EVERY) {
                    let turn = (id / SPECIAL_FLOAT_EVERY) as usize % SPECIAL_FLOATS.len();
                    assert!(sv.is_valid(row), "id {id}");
                    assert_eq!(
                        sv.value(row).to_bits(),
                        SPECIAL_FLOATS[turn].to_bits(),
                        "id {id}"
                    );
                    seen[turn] += 1;
                } else {
                    assert_eq!(sv.is_valid(row), pv.is_valid(row), "id {id}");
                    if sv.is_valid(row) {
                        assert_eq!(sv.value(row).to_bits(), pv.value(row).to_bits(), "id {id}");
                    }
                }
            }
        }
        assert!(seen.iter().all(|&n| n > 0), "{seen:?}");
        // The default table is not touched by the option existing.
        assert_eq!(
            demo_batches_with(rows, 42, DemoOptions::default()).unwrap(),
            plain
        );
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

    /// A refused run changes nothing (#119): the existing table keeps its
    /// bytes and a missing output directory is not created.
    #[test]
    fn a_refused_run_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let table = write_demo_table(
            dir.path(),
            1000,
            1,
            &demo_write_options(500, Compression::None),
        )
        .unwrap();
        let before = std::fs::read(&table.path).unwrap();
        let refused = demo_write_options(0, Compression::None);
        assert!(write_demo_table(dir.path(), 2000, 2, &refused).is_err());
        assert!(std::fs::read(&table.path).unwrap() == before);
        let missing = dir.path().join("new");
        assert!(write_demo_table(&missing, 2000, 2, &refused).is_err());
        assert!(!missing.exists(), "a refused run creates no directory");
    }
}
