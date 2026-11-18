//! Writing Parquet files with OxideLake's writer settings.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use oxidelake_core::EngineError;
use parquet::arrow::ArrowWriter;
use parquet::errors::ParquetError;

use crate::parquet_opts::ParquetWriteOptions;

fn writer_error(err: ParquetError) -> EngineError {
    EngineError::execution(format!("parquet writer: {err}"))
}

/// A file written under a temporary name next to its destination, and
/// removed again unless [`Staged::publish`] renames it into place.
struct Staged {
    path: PathBuf,
    destination: PathBuf,
    published: bool,
}

impl Staged {
    /// Creates `.<name>.<pid>.<n>.tmp` beside `destination`: the same
    /// directory, so the rename that publishes it cannot cross filesystems.
    /// The leading dot and the extension keep a half-written file out of a
    /// table directory a query is reading, which lists `*.parquet`.
    fn create(destination: &Path) -> Result<(File, Self), EngineError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = destination.file_name().ok_or_else(|| {
            EngineError::plan(format!("{} does not name a file", destination.display()))
        })?;
        loop {
            let mut staged = OsString::from(".");
            staged.push(name);
            staged.push(format!(
                ".{}.{}.tmp",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let path = destination.with_file_name(staged);
            // `create_new`: never a file someone else is writing, and never
            // through a symlink planted under the temporary name.
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok((
                        file,
                        Self {
                            path,
                            destination: destination.to_path_buf(),
                            published: false,
                        },
                    ));
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn publish(mut self) -> Result<(), EngineError> {
        std::fs::rename(&self.path, &self.destination)?;
        self.published = true;
        Ok(())
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        if !self.published {
            // Best effort: the destination is untouched either way.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// An open Parquet file being written batch by batch with OxideLake's writer
/// settings. Row groups are flushed as they fill, so memory stays bounded by
/// one row group regardless of how many batches are streamed in.
///
/// Nothing at the destination changes until [`close`](Self::close)
/// succeeds: the file is written under a temporary name beside it and
/// renamed into place, so rejected options, a failed write or a writer
/// dropped part way leave whatever was there before as it was (#119).
pub struct ParquetFileWriter {
    writer: ArrowWriter<File>,
    staged: Staged,
}

impl ParquetFileWriter {
    /// Starts writing a table with `schema` that [`close`](Self::close) will
    /// put at `path`, replacing any file there. Fails before creating
    /// anything if `options` are invalid.
    pub fn create(
        path: &Path,
        schema: SchemaRef,
        options: &ParquetWriteOptions,
    ) -> Result<Self, EngineError> {
        // Before anything touches the disk: an option rejected here used to
        // truncate the file it was going to replace first (#119).
        let props = options.writer_properties()?;
        let (file, staged) = Staged::create(path)?;
        let writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(writer_error)?;
        Ok(Self { writer, staged })
    }

    /// Appends one batch.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), EngineError> {
        self.writer.write(batch).map_err(writer_error)
    }

    /// Finishes the file, renames it into place and returns the row count it
    /// holds.
    pub fn close(self) -> Result<u64, EngineError> {
        let Self { writer, staged } = self;
        // `close` flushes the writer's buffer and reports a failure, so a
        // short write is an error here rather than a truncated file renamed
        // over the old one.
        let metadata = writer.close().map_err(writer_error)?;
        staged.publish()?;
        Ok(u64::try_from(metadata.file_metadata().num_rows()).unwrap_or(0))
    }
}

/// Writes `batches` to a Parquet file at `path` and returns the row count.
/// A file already at `path` is replaced only once the new one is complete
/// (see [`ParquetFileWriter`]).
pub fn write_parquet(
    path: &Path,
    schema: SchemaRef,
    batches: &[RecordBatch],
    options: &ParquetWriteOptions,
) -> Result<u64, EngineError> {
    let mut writer = ParquetFileWriter::create(path, schema, options)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.close()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use parquet::file::reader::{FileReader, SerializedFileReader};

    use super::*;
    use crate::parquet_opts::Compression;

    #[test]
    fn writes_row_groups_with_bloom_filters() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.parquet");
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..1000))],
        )
        .unwrap();
        let opts = ParquetWriteOptions {
            row_group_rows: 100,
            compression: Compression::None,
            ..Default::default()
        }
        .with_bloom_filter("k");
        assert_eq!(write_parquet(&path, schema, &[batch], &opts).unwrap(), 1000);
        let reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
        let meta = reader.metadata();
        assert_eq!(meta.num_row_groups(), 10);
        assert!(meta.row_group(0).column(0).bloom_filter_offset().is_some());
    }

    fn table(first: i64) -> (SchemaRef, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(first..first + 1000))],
        )
        .unwrap();
        (schema, batch)
    }

    /// `oxide gen-data --row-group-rows 0` emptied the table it was about to
    /// replace and only then refused the flag (#119): the file was created,
    /// truncating it, before the options were checked. Nothing reaches the
    /// destination now until a complete file does.
    #[test]
    fn a_rejected_or_abandoned_write_leaves_the_old_file_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.parquet");
        let (schema, old) = table(0);
        let options = ParquetWriteOptions::default();
        write_parquet(&path, Arc::clone(&schema), &[old], &options).unwrap();
        let before = std::fs::read(&path).unwrap();
        let files = || std::fs::read_dir(dir.path()).unwrap().count();
        let intact = |what: &str| {
            let now = std::fs::read(&path).unwrap();
            assert!(
                now == before,
                "{what}: t.parquet is now {} bytes, was {} bytes of the old table",
                now.len(),
                before.len()
            );
        };

        let rejected = ParquetWriteOptions {
            row_group_rows: 0,
            ..ParquetWriteOptions::default()
        };
        let (_, new) = table(5_000);
        let err = write_parquet(
            &path,
            Arc::clone(&schema),
            std::slice::from_ref(&new),
            &rejected,
        )
        .unwrap_err();
        assert!(err.to_string().contains("row_group_rows"), "{err}");
        intact("a rejected write");
        assert_eq!(files(), 1, "a rejected write creates nothing");

        // Dropped part way, as when the batches feeding it fail.
        let mut writer = ParquetFileWriter::create(&path, Arc::clone(&schema), &options).unwrap();
        writer.write(&new).unwrap();
        drop(writer);
        intact("an abandoned write");
        assert_eq!(files(), 1, "the staged file is removed");

        // A write that fails part way: a batch of another schema.
        let mut writer = ParquetFileWriter::create(&path, Arc::clone(&schema), &options).unwrap();
        writer.write(&new).unwrap();
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, false)])),
            vec![Arc::new(arrow::array::StringArray::from(vec!["x"]))],
        )
        .unwrap();
        assert!(writer.write(&other).is_err());
        drop(writer);
        intact("a failed write");
        assert_eq!(files(), 1, "the staged file is removed after a failure");

        // A finished write replaces the file, and leaves nothing beside it.
        assert_eq!(
            write_parquet(&path, schema, &[new], &options).unwrap(),
            1000
        );
        assert_ne!(std::fs::read(&path).unwrap(), before);
        assert_eq!(files(), 1);
        let reader = SerializedFileReader::new(File::open(&path).unwrap()).unwrap();
        assert_eq!(reader.metadata().file_metadata().num_rows(), 1000);
    }
}
