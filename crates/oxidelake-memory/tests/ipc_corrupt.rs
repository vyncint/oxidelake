//! A corrupt Arrow IPC file is refused with [`EngineError::Format`], never
//! decoded into a panic or an abort (#110).
//!
//! Each case changes one field of a valid file (two for the zero-length
//! block) at a position found by parsing that file, so the cases follow the
//! format rather than hard-coded offsets.
//!
//! This binary runs under an allocator that refuses any single request above
//! [`CAP`]. A length read off a corrupt file that reaches an allocation fails
//! here at once, aborting the binary and turning the run red, instead of
//! depending on whether the machine has that much memory to give: arrow-ipc
//! sizes the footer, every block and every decompressed buffer by what the
//! file declares. Every file here is a few kilobytes.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use arrow::ipc::{
    Block, Buffer as IpcBuffer, CompressionType, FieldNode, Footer, Message, MetadataVersion,
    root_as_footer, root_as_message,
};
use bytes::Bytes;
use oxidelake_core::EngineError;
use oxidelake_memory::ipc::{IPC_ALIGNMENT, decode_batch, encode_batch, read_batches};

/// The largest single allocation this binary grants.
const CAP: usize = 64 << 20;

struct Capped;

// SAFETY: every method either forwards to `System` with the caller's
// arguments unchanged or returns null, which `GlobalAlloc` defines as an
// allocation failure.
unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > CAP {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller upholds `alloc`'s contract; it is passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() > CAP {
            return std::ptr::null_mut();
        }
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System` through this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > CAP {
            return std::ptr::null_mut();
        }
        // SAFETY: `ptr` came from `System` through this allocator, and the
        // caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Capped = Capped;

/// Bytes after the footer: its length (`i32`) and the magic `ARROW1`.
const TRAILER: usize = 10;

fn sample() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Float64, false),
        Field::new("s", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(
                (0..64)
                    .map(|i| (i % 7 != 3).then_some(i))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (0..64).map(f64::from).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..64)
                    .map(|i| (i % 5 != 0).then(|| format!("s{i}")))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// Where the fields a corruption rewrites live in a valid file.
struct Anatomy {
    bytes: Vec<u8>,
    /// The footer's length field, just before the closing magic.
    footer_len_at: usize,
    /// The footer table's vtable slot that says where its schema is.
    schema_slot_at: usize,
    /// The record batch's `Block` in the footer, and its value.
    block_at: usize,
    block: Block,
    /// The record batch's buffers and field nodes, in its metadata.
    buffers: Vec<(usize, IpcBuffer)>,
    nodes: Vec<(usize, FieldNode)>,
    /// The message's header type, one byte.
    header_type_at: usize,
    /// The first of the batch's variadic buffer counts, if it has any.
    variadic_count_at: Option<usize>,
}

/// The position of `field` inside `bytes`, which it must point into.
fn position<T>(bytes: &[u8], field: &T) -> usize {
    let at = std::ptr::from_ref(field) as usize - bytes.as_ptr() as usize;
    assert!(at < bytes.len(), "field does not point into the file");
    at
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}

/// The root table of the flatbuffer at `start`, and the position of its
/// vtable slot for the field at vtable offset `field`.
///
/// A flatbuffer starts with the offset of its root table; the table starts
/// with the signed distance back to its vtable, whose slots hold each
/// field's offset in the table (0: absent).
fn root_table(bytes: &[u8], start: usize, field: u16) -> (usize, usize) {
    let table = start + usize::try_from(read_u32(bytes, start)).unwrap();
    let back = i32::from_le_bytes(bytes[table..table + 4].try_into().unwrap());
    let vtable = usize::try_from(i64::try_from(table).unwrap() - i64::from(back)).unwrap();
    let slot = vtable + usize::from(field);
    assert_ne!(
        read_u16(bytes, slot),
        0,
        "the field must be present to be found"
    );
    (table, slot)
}

fn anatomy(bytes: Vec<u8>) -> Anatomy {
    let n = bytes.len();
    let footer_len_at = n - TRAILER;
    let footer_len = usize::try_from(read_u32(&bytes, footer_len_at)).unwrap();
    let footer_start = footer_len_at - footer_len;
    let footer = root_as_footer(&bytes[footer_start..footer_len_at]).unwrap();
    assert!(footer.schema().is_some());

    let (_, schema_slot_at) = root_table(&bytes, footer_start, Footer::VT_SCHEMA);

    let blocks = footer.recordBatches().unwrap();
    assert_eq!(blocks.len(), 1);
    let block = blocks.get(0);
    let block_at = position(&bytes, block);
    let block = *block;

    let start = usize::try_from(block.offset()).unwrap();
    let metadata = usize::try_from(block.metaDataLength()).unwrap();
    // The metadata starts with the continuation marker and its own length.
    let message = root_as_message(&bytes[start + 8..start + metadata]).unwrap();
    let batch = message.header_as_record_batch().unwrap();
    let buffers = batch
        .buffers()
        .unwrap()
        .iter()
        .map(|b| (position(&bytes, b), *b))
        .collect();
    let nodes = batch
        .nodes()
        .unwrap()
        .iter()
        .map(|n| (position(&bytes, n), *n))
        .collect();
    let variadic_count_at = batch
        .variadicBufferCounts()
        .filter(|counts| !counts.is_empty())
        .map(|counts| position(&bytes, &counts.bytes()[0]));
    let (table, slot) = root_table(&bytes, start + 8, Message::VT_HEADER_TYPE);
    let header_type_at = table + usize::from(read_u16(&bytes, slot));
    Anatomy {
        footer_len_at,
        schema_slot_at,
        block_at,
        block,
        buffers,
        nodes,
        header_type_at,
        variadic_count_at,
        bytes,
    }
}

impl Anatomy {
    fn of_sample() -> Self {
        anatomy(encode_batch(&sample()).unwrap().to_vec())
    }

    /// The file with `new` written over the bytes at `at`.
    fn with(&self, at: usize, new: &[u8]) -> Vec<u8> {
        let mut bytes = self.bytes.clone();
        bytes[at..at + new.len()].copy_from_slice(new);
        bytes
    }

    fn with_block(&self, block: Block) -> Vec<u8> {
        self.with(self.block_at, &block.0)
    }

    fn body_start(&self) -> usize {
        usize::try_from(self.block.offset() + i64::from(self.block.metaDataLength())).unwrap()
    }
}

/// Every reader refuses `bytes` with a format error that names the defect:
/// `reason` is part of the message. A check that names the defect is what
/// keeps the reader from relying on the panic guard, so a case that only
/// that guard catches would fail here.
fn assert_refused(bytes: Vec<u8>, reason: &str) {
    let check = |reader: &str, result: Result<(), EngineError>| match result {
        Err(EngineError::Format(message)) => {
            assert!(
                message.contains(reason),
                "{reader}: expected {reason:?} in {message:?}"
            );
        }
        other => panic!("{reader}: expected a format error naming {reason:?}, got {other:?}"),
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt.arrow");
    std::fs::write(&path, &bytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    check("read_batches", read_batches(file).map(drop));
    check("decode_batch", decode_batch(&Bytes::from(bytes)).map(drop));
}

#[test]
fn the_unchanged_file_decodes() {
    let file = Anatomy::of_sample();
    let (_, batches) = read_batches(std::io::Cursor::new(&file.bytes)).unwrap();
    assert_eq!(batches, [sample()]);
    assert_eq!(decode_batch(&Bytes::from(file.bytes)).unwrap(), sample());
}

#[test]
fn a_buffer_past_the_end_of_the_body_is_refused() {
    let file = Anatomy::of_sample();
    // The buffer furthest into the body, stretched to the body's length.
    let (at, buffer) = file
        .buffers
        .iter()
        .max_by_key(|(_, b)| b.offset())
        .copied()
        .unwrap();
    assert!(buffer.offset() > 0);
    let past = IpcBuffer::new(buffer.offset(), file.block.bodyLength());
    assert_refused(file.with(at, &past.0), "is not inside the");
}

#[test]
fn a_huge_body_length_is_refused() {
    let file = Anatomy::of_sample();
    let b = file.block;
    let huge = Block::new(b.offset(), b.metaDataLength(), 1 << 62);
    assert_refused(file.with_block(huge), "run past the");
}

#[test]
fn a_negative_body_length_is_refused() {
    let file = Anatomy::of_sample();
    let b = file.block;
    let negative = Block::new(b.offset(), b.metaDataLength(), -1);
    assert_refused(file.with_block(negative), "its body length -1 is negative");
}

#[test]
fn a_negative_metadata_length_is_refused() {
    let file = Anatomy::of_sample();
    let b = file.block;
    let negative = Block::new(b.offset(), -1, b.bodyLength());
    assert_refused(file.with_block(negative), "its metadata length -1");
}

#[test]
fn a_zero_length_block_is_refused() {
    let file = Anatomy::of_sample();
    let empty = Block::new(file.block.offset(), 0, 0);
    assert_refused(file.with_block(empty), "its metadata length 0");
}

#[test]
fn a_block_past_the_end_of_the_file_is_refused() {
    let file = Anatomy::of_sample();
    let b = file.block;
    let end = i64::try_from(file.bytes.len()).unwrap();
    let past = Block::new(end, b.metaDataLength(), b.bodyLength());
    assert_refused(file.with_block(past), "run past the");
}

#[test]
fn a_footer_without_a_schema_is_refused() {
    let file = Anatomy::of_sample();
    assert_refused(
        file.with(file.schema_slot_at, &[0, 0]),
        "its footer has no schema",
    );
}

#[test]
fn a_huge_footer_length_is_refused() {
    let file = Anatomy::of_sample();
    let huge = i32::MAX.to_le_bytes();
    assert_refused(
        file.with(file.footer_len_at, &huge),
        "its footer is said to be",
    );
}

#[test]
fn a_negative_field_length_is_refused() {
    let file = Anatomy::of_sample();
    // The first column has nulls, so its validity bitmap is read at the
    // length the node declares.
    let (at, node) = file.nodes[0];
    assert!(node.null_count() > 0);
    let negative = FieldNode::new(-1, node.null_count());
    assert_refused(file.with(at, &negative.0), "declares -1 values");
}

#[test]
fn a_negative_block_offset_is_refused() {
    let file = Anatomy::of_sample();
    let b = file.block;
    let negative = Block::new(-8, b.metaDataLength(), b.bodyLength());
    assert_refused(file.with_block(negative), "its offset -8 is negative");
}

#[test]
fn a_message_without_a_header_is_refused() {
    // arrow-ipc reads a record batch block whose message has no header as
    // the end of the file, and returns the batches before it, with no error.
    let file = Anatomy::of_sample();
    assert_refused(file.with(file.header_type_at, &[0]), "holds a NONE message");
}

#[test]
fn a_negative_variadic_buffer_count_is_refused() {
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)]));
    let strings = (0..64).map(|i| (i % 5 != 0).then(|| format!("a string too long to inline {i}")));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(StringViewArray::from_iter(strings))]).unwrap();
    let file = anatomy(encode_batch(&batch).unwrap().to_vec());
    assert_eq!(
        decode_batch(&Bytes::from(file.bytes.clone())).unwrap(),
        batch
    );
    let at = file.variadic_count_at.unwrap();
    assert_refused(
        file.with(at, &(-3_i64).to_le_bytes()),
        "declares -3 variadic buffers",
    );
}

#[test]
fn a_field_longer_than_its_validity_bitmap_is_refused() {
    // Nothing checks this before arrow does, and arrow checks it with an
    // assertion while building the array: the panic guard's case.
    let file = Anatomy::of_sample();
    let (at, node) = file.nodes[0];
    assert!(node.null_count() > 0);
    let long = FieldNode::new(1_000_000, node.null_count());
    assert_refused(file.with(at, &long.0), "arrow-ipc panicked");
}

#[test]
fn a_compressed_buffer_declaring_a_huge_size_is_refused() {
    // A compressed buffer starts with its decompressed size, which arrow-ipc
    // allocates before decompressing a byte.
    let options = IpcWriteOptions::try_new(IPC_ALIGNMENT, false, MetadataVersion::V5)
        .unwrap()
        .try_with_compression(Some(CompressionType::LZ4_FRAME))
        .unwrap();
    let mut bytes = Vec::new();
    {
        let batch = sample();
        let mut writer =
            FileWriter::try_new_with_options(&mut bytes, &batch.schema(), options).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    let file = anatomy(bytes);
    let (_, buffer) = file
        .buffers
        .iter()
        .find(|(_, b)| b.length() >= 8)
        .copied()
        .unwrap();
    let at = file.body_start() + usize::try_from(buffer.offset()).unwrap();
    let huge = (1_i64 << 62).to_le_bytes();
    assert_refused(file.with(at, &huge), "LZ4_FRAME-compressed");
}
