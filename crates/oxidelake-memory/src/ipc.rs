//! Arrow IPC encoding for spill and cache files: raw Arrow buffers with
//! 64-byte alignment and no compression, so batches come back without decode.
//!
//! Reading does not trust the file (#110). arrow-ipc's reader takes the
//! lengths and offsets a file declares at their word: a block's lengths are
//! unwrapped and allocated before a byte is read, a buffer's range is sliced
//! with an assertion, a missing schema is unwrapped, and a compressed
//! buffer's declared size is allocated before it is decompressed. One
//! corrupt field could panic the reader or abort the process on a failed
//! allocation. So the footer, every block and every message's buffers are
//! checked against the bytes that are actually there before arrow decodes
//! them, and nothing is allocated at a size the file has not been shown to
//! hold. Compressed bodies are refused: [`encode_batch`] never writes one,
//! and their declared size could only be checked by allocating it. The
//! decode runs under `catch_unwind`, so a defect arrow notices only by
//! panicking is an [`EngineError::Format`] too. An allocation failure
//! aborts rather than panics and cannot be caught, which is why the checks
//! come first.

use std::any::Any;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use arrow::ipc::convert::fb_to_schema;
use arrow::ipc::reader::{FileDecoder, read_footer_length};
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use arrow::ipc::{Block, MessageHeader, MetadataVersion, root_as_footer, root_as_message};
use bytes::Bytes;
use oxidelake_core::EngineError;

/// Buffer alignment used for spill files (the maximum Arrow IPC permits).
pub const IPC_ALIGNMENT: usize = 64;

/// A file ends with its footer's length (`i32`) and the magic `ARROW1`.
const TRAILER_LEN: u64 = 10;

/// What a message's metadata starts with since Arrow 0.15; it is followed by
/// the metadata's length. Older writers wrote the length alone.
const CONTINUATION_MARKER: [u8; 4] = [0xff; 4];

/// The fewest metadata bytes a block can declare: the marker and the length.
/// arrow-ipc indexes both before it looks at the length.
const MIN_METADATA_LEN: usize = 8;

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
/// Truncated or corrupt input yields [`EngineError::Format`], as does a file
/// whose batches are compressed (see the [module docs](self)).
pub fn decode_batch(bytes: &Bytes) -> Result<RecordBatch, EngineError> {
    guarded(|| {
        let (schema, batches) = read_file(Cursor::new(bytes.as_ref()))?;
        match batches.len() {
            0 => Ok(RecordBatch::new_empty(schema)),
            1 => Ok(batches
                .into_iter()
                .next()
                .unwrap_or_else(|| RecordBatch::new_empty(schema))),
            _ => concat_batches(&schema, &batches)
                .map_err(|e| Fault::Format(format!("its batches do not concatenate: {e}"))),
        }
    })
    .map_err(|fault| fault.into_engine("invalid Arrow IPC spill file"))
}

/// Reads every batch of the Arrow IPC file `reader` holds, and its schema.
///
/// The file is checked before arrow decodes it (see the [module docs](self)):
/// a truncated or corrupt file, or one whose batches are compressed, yields
/// [`EngineError::Format`]; a failing read yields [`EngineError::Io`].
/// Nothing is allocated beyond what the file holds, and each block is read
/// into a buffer of its own, aligned for Arrow, so aligned columns come back
/// without a copy.
pub fn read_batches<R: Read + Seek>(
    reader: R,
) -> Result<(SchemaRef, Vec<RecordBatch>), EngineError> {
    guarded(|| read_file(reader)).map_err(|fault| fault.into_engine("invalid Arrow IPC file"))
}

/// Why a file could not be read.
enum Fault {
    Io(std::io::Error),
    Format(String),
}

impl From<std::io::Error> for Fault {
    fn from(err: std::io::Error) -> Self {
        Fault::Io(err)
    }
}

impl Fault {
    fn into_engine(self, what: &str) -> EngineError {
        match self {
            Fault::Io(err) => EngineError::Io(err),
            Fault::Format(reason) => EngineError::format(format!("{what}: {reason}")),
        }
    }
}

/// Runs `read`, turning a panic into a [`Fault::Format`]. The last resort
/// for defects the checks below do not name: arrow validates each array as
/// it builds it, and a few of those checks are assertions.
fn guarded<T>(read: impl FnOnce() -> Result<T, Fault>) -> Result<T, Fault> {
    catch_unwind(AssertUnwindSafe(read)).unwrap_or_else(|payload| {
        Err(Fault::Format(format!(
            "arrow-ipc panicked decoding it: {}",
            panic_message(payload.as_ref())
        )))
    })
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a panic without a message")
}

fn format_fault(reason: impl Into<String>) -> Fault {
    Fault::Format(reason.into())
}

/// The footer's lists of blocks, for messages.
#[derive(Clone, Copy)]
enum BlockKind {
    Dictionary,
    RecordBatch,
}

impl BlockKind {
    const fn name(self) -> &'static str {
        match self {
            BlockKind::Dictionary => "dictionary",
            BlockKind::RecordBatch => "record batch",
        }
    }
}

/// arrow-ipc's `FileReaderBuilder::build` and `FileReader`, with every length
/// checked against the file before it is used.
fn read_file<R: Read + Seek>(mut reader: R) -> Result<(SchemaRef, Vec<RecordBatch>), Fault> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    let trailer_start = file_len.checked_sub(TRAILER_LEN).ok_or_else(|| {
        format_fault(format!(
            "{file_len} bytes cannot hold the {TRAILER_LEN}-byte trailer"
        ))
    })?;
    let mut trailer = [0_u8; TRAILER_LEN as usize];
    reader.seek(SeekFrom::Start(trailer_start))?;
    reader.read_exact(&mut trailer)?;
    let footer_len = read_footer_length(trailer).map_err(|e| format_fault(e.to_string()))?;
    let footer_start = u64::try_from(footer_len)
        .ok()
        .and_then(|len| trailer_start.checked_sub(len))
        .ok_or_else(|| {
            format_fault(format!(
                "its footer is said to be {footer_len} bytes, in a {file_len}-byte file"
            ))
        })?;

    let mut footer_bytes = vec![0_u8; footer_len];
    reader.seek(SeekFrom::Start(footer_start))?;
    reader.read_exact(&mut footer_bytes)?;
    let footer = root_as_footer(&footer_bytes)
        .map_err(|e| format_fault(format!("its footer does not parse: {e}")))?;
    let ipc_schema = footer
        .schema()
        .ok_or_else(|| format_fault("its footer has no schema"))?;
    if !ipc_schema.endianness().equals_to_target_endianness() {
        return Err(format_fault(
            "it was written on a machine of the other endianness",
        ));
    }
    let schema = Arc::new(fb_to_schema(ipc_schema));
    let mut decoder = FileDecoder::new(Arc::clone(&schema), footer.version());

    for (index, block) in footer.dictionaries().into_iter().flatten().enumerate() {
        let buf = read_block(
            &mut reader,
            block,
            footer_start,
            BlockKind::Dictionary,
            index,
        )?;
        decoder
            .read_dictionary(block, &buf)
            .map_err(|e| format_fault(format!("dictionary block {index}: {e}")))?;
    }

    let blocks = footer
        .recordBatches()
        .ok_or_else(|| format_fault("its footer lists no record batches"))?;
    let mut batches = Vec::with_capacity(blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        let buf = read_block(
            &mut reader,
            block,
            footer_start,
            BlockKind::RecordBatch,
            index,
        )?;
        let batch = decoder
            .read_record_batch(block, &buf)
            .map_err(|e| format_fault(format!("record batch block {index}: {e}")))?
            .ok_or_else(|| {
                format_fault(format!("record batch block {index} holds no record batch"))
            })?;
        batches.push(batch);
    }
    Ok((schema, batches))
}

/// Reads one block, after checking that it lies in the file before the
/// footer (`limit`), and checks its message before arrow decodes it.
fn read_block<R: Read + Seek>(
    reader: &mut R,
    block: &Block,
    limit: u64,
    kind: BlockKind,
    index: usize,
) -> Result<Buffer, Fault> {
    let invalid = |reason: String| format_fault(format!("{} block {index}: {reason}", kind.name()));
    let metadata_len = usize::try_from(block.metaDataLength())
        .ok()
        .filter(|&len| len >= MIN_METADATA_LEN)
        .ok_or_else(|| {
            invalid(format!(
                "its metadata length {} is under the {MIN_METADATA_LEN} bytes any message needs",
                block.metaDataLength()
            ))
        })?;
    let body_len = u64::try_from(block.bodyLength()).map_err(|_| {
        invalid(format!(
            "its body length {} is negative",
            block.bodyLength()
        ))
    })?;
    let start = u64::try_from(block.offset())
        .map_err(|_| invalid(format!("its offset {} is negative", block.offset())))?;
    let end = u64::try_from(metadata_len)
        .ok()
        .and_then(|metadata| metadata.checked_add(body_len))
        .and_then(|len| start.checked_add(len))
        .filter(|&end| end <= limit)
        .ok_or_else(|| {
            invalid(format!(
                "{metadata_len} metadata and {body_len} body bytes at offset {start} run past \
                 the {limit} bytes before the footer"
            ))
        })?;
    // `end <= limit`, which came from the file's own length.
    let len = usize::try_from(end - start)
        .map_err(|_| invalid(format!("{} bytes do not fit in memory", end - start)))?;
    reader.seek(SeekFrom::Start(start))?;
    // A buffer of its own, aligned like arrow-ipc's: the writer aligned
    // every buffer within the block, so this keeps them aligned in memory.
    let mut buf = MutableBuffer::from_len_zeroed(len);
    reader.read_exact(buf.as_slice_mut())?;
    let buf = Buffer::from(buf);
    check_message(&buf, body_len, kind).map_err(invalid)?;
    Ok(buf)
}

/// Checks what arrow-ipc would take on trust from a block's message: that it
/// holds the batch the footer says it does, that its body is not compressed,
/// that its buffers lie in the body, and that the field lengths and
/// variadic buffer counts arrow casts or indexes by are in range.
fn check_message(block: &[u8], body_len: u64, kind: BlockKind) -> Result<(), String> {
    // The same view of the message arrow-ipc parses: from after the prefix
    // to the end of the block.
    let prefix = if block.starts_with(&CONTINUATION_MARKER) {
        8
    } else {
        4
    };
    let message = block
        .get(prefix..)
        .ok_or_else(|| "it is shorter than its message prefix".to_owned())
        .and_then(|rest| {
            root_as_message(rest).map_err(|e| format!("its message does not parse: {e}"))
        })?;
    let batch = match (kind, message.header_type()) {
        (BlockKind::RecordBatch, MessageHeader::RecordBatch) => message.header_as_record_batch(),
        (BlockKind::Dictionary, MessageHeader::DictionaryBatch) => message
            .header_as_dictionary_batch()
            .and_then(|dictionary| dictionary.data()),
        (_, header) => return Err(format!("it holds a {header:?} message")),
    }
    .ok_or_else(|| "its message has no batch".to_owned())?;

    if let Some(compression) = batch.compression() {
        return Err(format!(
            "its body is {:?}-compressed; spill and cache files are written uncompressed, and \
             a compressed buffer's declared size is allocated before it can be checked",
            compression.codec()
        ));
    }
    for (i, node) in batch.nodes().into_iter().flatten().enumerate() {
        // arrow-ipc casts the length to `usize` and asserts the validity
        // bitmap covers it.
        if node.length() < 0 {
            return Err(format!(
                "field node {i} declares {} values with {} nulls",
                node.length(),
                node.null_count()
            ));
        }
    }
    let buffers = batch.buffers().map_or(0, |buffers| buffers.len());
    for (i, buffer) in batch.buffers().into_iter().flatten().enumerate() {
        let inside = u64::try_from(buffer.offset())
            .ok()
            .zip(u64::try_from(buffer.length()).ok())
            .and_then(|(offset, length)| offset.checked_add(length))
            .is_some_and(|end| end <= body_len);
        if !inside {
            return Err(format!(
                "buffer {i} ({} bytes at offset {}) is not inside the {body_len}-byte body",
                buffer.length(),
                buffer.offset()
            ));
        }
    }
    // arrow-ipc reads `count + 2` buffers for each view column, and indexes
    // the ones it got: a negative count leaves it none.
    for count in batch.variadicBufferCounts().into_iter().flatten() {
        if !usize::try_from(count).is_ok_and(|count| count <= buffers) {
            return Err(format!(
                "it declares {count} variadic buffers, of {buffers} buffers in all"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{DictionaryArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};

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

    /// The reader replaced arrow-ipc's `FileReader` (#110); this is the part
    /// of the format the sample does not reach: dictionary blocks, several
    /// record batches, and none at all.
    #[test]
    fn dictionaries_and_several_batches_read_back_whole() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ]));
        let batch = |start: i64| {
            let keys: DictionaryArray<Int32Type> =
                vec![Some("red"), None, Some("blue"), Some("red")]
                    .into_iter()
                    .collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(start..start + 4)),
                    Arc::new(keys),
                ],
            )
            .unwrap()
        };
        let write = |batches: &[RecordBatch]| {
            let options =
                IpcWriteOptions::try_new(IPC_ALIGNMENT, false, MetadataVersion::V5).unwrap();
            let mut out = Vec::new();
            let mut writer = FileWriter::try_new_with_options(&mut out, &schema, options).unwrap();
            for b in batches {
                writer.write(b).unwrap();
            }
            writer.finish().unwrap();
            drop(writer);
            Bytes::from(out)
        };

        let both = [batch(0), batch(100)];
        let file = write(&both);
        let (read_schema, read) = read_batches(Cursor::new(file.as_ref())).unwrap();
        assert_eq!(read_schema, schema);
        assert_eq!(read, both);
        assert_eq!(
            decode_batch(&file).unwrap(),
            concat_batches(&schema, &both).unwrap()
        );

        let empty = write(&[]);
        assert_eq!(
            decode_batch(&empty).unwrap(),
            RecordBatch::new_empty(schema)
        );
    }
}
