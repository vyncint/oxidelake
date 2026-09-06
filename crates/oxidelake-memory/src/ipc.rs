//! Arrow IPC encoding for spill and cache files: raw Arrow buffers with
//! 64-byte alignment and no compression, so batches come back without decode.

use std::io::Cursor;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use arrow::ipc::MetadataVersion;
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use bytes::Bytes;
use oxidelake_core::EngineError;

/// Buffer alignment used for spill files (the maximum Arrow IPC permits).
pub const IPC_ALIGNMENT: usize = 64;

/// Encodes one batch as an Arrow IPC file.
pub fn encode_batch(batch: &RecordBatch) -> Result<Bytes, EngineError> {
    let options = IpcWriteOptions::try_new(IPC_ALIGNMENT, false, MetadataVersion::V5)?;
    let mut out = Vec::with_capacity(batch.get_array_memory_size() + 1024);
    {
        let mut writer = FileWriter::try_new_with_options(&mut out, &batch.schema(), options)?;
        writer.write(batch)?;
        writer.finish()?;
    }
    Ok(Bytes::from(out))
}

/// Decodes an Arrow IPC file into one batch (concatenating if it holds several).
///
/// Truncated or corrupt input yields [`EngineError::Format`].
pub fn decode_batch(bytes: &Bytes) -> Result<RecordBatch, EngineError> {
    let reader = FileReader::try_new(Cursor::new(bytes.clone()), None)
        .map_err(|e| EngineError::format(format!("invalid Arrow IPC spill file: {e}")))?;
    let schema: SchemaRef = reader.schema();
    let batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| EngineError::format(format!("corrupt Arrow IPC spill file: {e}")))?;
    match batches.len() {
        0 => Ok(RecordBatch::new_empty(schema)),
        1 => Ok(batches
            .into_iter()
            .next()
            .unwrap_or_else(|| RecordBatch::new_empty(schema))),
        _ => Ok(concat_batches(&schema, &batches)?),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn sample() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])),
                Arc::new(Float64Array::from(vec![Some(0.5), Some(1.5), None])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("ccc")])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn round_trip_is_identical_including_nulls() {
        let batch = sample();
        let bytes = encode_batch(&batch).unwrap();
        let back = decode_batch(&bytes).unwrap();
        assert_eq!(back, batch);
        // Re-encoding the decoded batch yields identical bytes.
        assert_eq!(encode_batch(&back).unwrap(), bytes);
    }

    #[test]
    fn truncated_file_is_a_format_error() {
        let bytes = encode_batch(&sample()).unwrap();
        let truncated = bytes.slice(..bytes.len() / 2);
        let err = decode_batch(&truncated).unwrap_err();
        assert!(matches!(err, EngineError::Format(_)), "{err}");
        let err = decode_batch(&Bytes::from_static(b"not arrow")).unwrap_err();
        assert!(matches!(err, EngineError::Format(_)), "{err}");
    }
}
