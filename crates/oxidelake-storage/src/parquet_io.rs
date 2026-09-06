//! Writing Parquet files with OxideLake's writer settings.

use std::fs::File;
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use oxidelake_core::EngineError;
use parquet::arrow::ArrowWriter;
use parquet::errors::ParquetError;

use crate::parquet_opts::ParquetWriteOptions;

fn writer_error(err: ParquetError) -> EngineError {
    EngineError::execution(format!("parquet writer: {err}"))
}

/// An open Parquet file being written batch by batch with OxideLake's writer
/// settings. Row groups are flushed as they fill, so memory stays bounded by
/// one row group regardless of how many batches are streamed in.
pub struct ParquetFileWriter {
    writer: ArrowWriter<File>,
}

impl ParquetFileWriter {
    /// Creates (truncating) `path` for a table with `schema`.
    pub fn create(
        path: &Path,
        schema: SchemaRef,
        options: &ParquetWriteOptions,
    ) -> Result<Self, EngineError> {
        let file = File::create(path)?;
        let props = options.writer_properties()?;
        let writer = ArrowWriter::try_new(file, schema, Some(props)).map_err(writer_error)?;
        Ok(Self { writer })
    }

    /// Appends one batch.
    pub fn write(&mut self, batch: &RecordBatch) -> Result<(), EngineError> {
        self.writer.write(batch).map_err(writer_error)
    }

    /// Finishes the file and returns the row count it holds.
    pub fn close(self) -> Result<u64, EngineError> {
        let metadata = self.writer.close().map_err(writer_error)?;
        Ok(u64::try_from(metadata.file_metadata().num_rows()).unwrap_or(0))
    }
}

/// Writes `batches` to a new Parquet file at `path` and returns the row count.
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
    use arrow::datatypes::{DataType, Field, Schema};
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
}
