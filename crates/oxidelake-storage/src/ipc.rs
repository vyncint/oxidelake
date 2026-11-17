//! Arrow IPC files for spill and hot caches: raw Arrow buffers, 64-byte
//! aligned, no compression, so batches come back without decode.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use arrow::ipc::MetadataVersion;
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use oxidelake_core::EngineError;
use oxidelake_memory::ipc::read_batches;
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

/// Reads every batch of an Arrow IPC file.
///
/// The file is checked before arrow-ipc decodes it (#110, see
/// [`oxidelake_memory::ipc`]): a corrupt or truncated file, or one whose
/// batches are compressed, yields [`EngineError::Format`] naming the file and
/// the defect, never a panic or an allocation the file cannot back. A file
/// that cannot be opened or read yields [`EngineError::Io`].
pub fn read_ipc_file(path: &Path) -> Result<Vec<RecordBatch>, EngineError> {
    let file = File::open(path)?;
    match read_batches(file) {
        Ok((_, batches)) => Ok(batches),
        Err(EngineError::Format(reason)) => {
            Err(EngineError::format(format!("{}: {reason}", path.display())))
        }
        Err(other) => Err(other),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};
    use arrow::ipc::{Block, root_as_footer};

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

    /// Writes one batch to a file, rewrites its record batch's `Block` in the
    /// footer with `patch`, and reads the file back.
    fn read_with_block(
        patch: impl FnOnce(Block) -> Block,
    ) -> Result<Vec<RecordBatch>, EngineError> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.arrow");
        write_ipc_file(&path, &batch(0).schema(), &[batch(0)]).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let (at, new) = {
            // The file ends with the footer, its length and the magic `ARROW1`.
            let n = bytes.len();
            let footer_len = u32::from_le_bytes(bytes[n - 10..n - 6].try_into().unwrap());
            let footer_start = n - 10 - usize::try_from(footer_len).unwrap();
            let footer = root_as_footer(&bytes[footer_start..n - 10]).unwrap();
            let block = footer.recordBatches().unwrap().get(0);
            let at = std::ptr::from_ref(block) as usize - bytes.as_ptr() as usize;
            (at, patch(*block))
        };
        bytes[at..at + new.0.len()].copy_from_slice(&new.0);
        std::fs::write(&path, bytes).unwrap();
        read_ipc_file(&path)
    }

    fn assert_format_error_naming_the_file(result: Result<Vec<RecordBatch>, EngineError>) {
        match result {
            Err(EngineError::Format(message)) => {
                assert!(message.contains("cache.arrow"), "{message}");
            }
            other => panic!("expected a format error, got {other:?}"),
        }
    }

    // #110: arrow-ipc unwraps a block's lengths and allocates their sum
    // before reading a byte; a corrupt length must not reach either.
    #[test]
    fn a_negative_block_length_is_a_format_error() {
        assert_format_error_naming_the_file(read_with_block(|b| {
            Block::new(b.offset(), b.metaDataLength(), -1)
        }));
    }

    #[test]
    fn a_huge_block_length_is_a_format_error() {
        assert_format_error_naming_the_file(read_with_block(|b| {
            Block::new(b.offset(), b.metaDataLength(), 1 << 62)
        }));
    }

    #[test]
    fn a_zero_length_block_is_a_format_error() {
        assert_format_error_naming_the_file(read_with_block(|b| Block::new(b.offset(), 0, 0)));
    }
}
