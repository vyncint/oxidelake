//! Arrow IPC files for spill and hot caches: raw Arrow buffers, 64-byte
//! aligned, no compression, so batches come back without decode.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use arrow::ipc::MetadataVersion;
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use oxidelake_core::EngineError;
pub use oxidelake_memory::ipc::{IPC_ALIGNMENT, decode_batch, encode_batch};

/// Writes `batches` as one Arrow IPC file; returns the file size in bytes.
pub fn write_ipc_file(
    path: &Path,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<u64, EngineError> {
    let options = IpcWriteOptions::try_new(IPC_ALIGNMENT, false, MetadataVersion::V5)?;
    let file = File::create(path)?;
    let mut writer = FileWriter::try_new_with_options(BufWriter::new(file), schema, options)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    let mut inner = writer.into_inner()?;
    inner.flush()?;
    Ok(std::fs::metadata(path)?.len())
}

/// Reads every batch of an Arrow IPC file. Corrupt or truncated files yield
/// [`EngineError::Format`].
pub fn read_ipc_file(path: &Path) -> Result<Vec<RecordBatch>, EngineError> {
    let file = File::open(path)?;
    let reader = FileReader::try_new(BufReader::new(file), None).map_err(|e| {
        EngineError::format(format!("invalid Arrow IPC file {}: {e}", path.display()))
    })?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| EngineError::format(format!("corrupt Arrow IPC file {}: {e}", path.display())))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};

    use super::*;

    fn batch(start: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, false),
            Field::new("s", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(
                    (start..start + 100)
                        .map(|i| (i % 7 != 0).then_some(i))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    (0..100).map(f64::from).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    (0..100)
                        .map(|i| (i % 5 != 0).then(|| format!("s{i}")))
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    #[test]
    fn round_trip_is_identical_and_buffers_are_64_byte_aligned() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.arrow");
        let batches = vec![batch(0), batch(1000), batch(2000)];
        let bytes = write_ipc_file(&path, &batches[0].schema(), &batches).unwrap();
        assert!(bytes > 0);
        let back = read_ipc_file(&path).unwrap();
        assert_eq!(back, batches);
        for b in &back {
            for col in b.columns() {
                for buf in col.to_data().buffers() {
                    assert!(
                        (buf.as_ptr() as usize).is_multiple_of(IPC_ALIGNMENT),
                        "buffer not 64-byte aligned"
                    );
                }
            }
        }
    }

    #[test]
    fn truncated_files_are_format_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.arrow");
        let batches = vec![batch(0)];
        let len = write_ipc_file(&path, &batches[0].schema(), &batches).unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len / 2).unwrap();
        assert!(matches!(
            read_ipc_file(&path).unwrap_err(),
            EngineError::Format(_)
        ));
        std::fs::write(&path, b"not arrow at all").unwrap();
        assert!(matches!(
            read_ipc_file(&path).unwrap_err(),
            EngineError::Format(_)
        ));
        assert!(matches!(
            read_ipc_file(&dir.path().join("missing")).unwrap_err(),
            EngineError::Io(_)
        ));
    }
}
