//! What a Parquet file's metadata must satisfy before anything reads by it
//! (#109).
//!
//! The parquet reader takes a footer at its word. A column chunk's start and
//! length become a byte range to fetch (it asserts both are non-negative), a
//! page location becomes a slice of its chunk, and a few declared counts
//! become `Vec::with_capacity` before one element is read. A file whose
//! footer decodes cleanly but lies in any of them panics the reader, aborts
//! the process on an allocation the size of the lie, or allocates and
//! zero-fills it. Measured on parquet 59.3.2: a column chunk length of 2^40
//! in a 98 KB file aborts on a 1 TiB allocation, -1 panics the scan, and
//! 2,000,000,000 costs 2 GB of RSS before failing.
//!
//! So nothing here trusts a number it has not checked against bytes that
//! exist: every range lies inside the file, every page inside its chunk, and
//! every count in the footer's Thrift encoding is walked element by element
//! before the decoder is allowed to reserve room for it. A decoder panic that
//! gets past all of that is caught and reported as the file's fault.

use std::fmt::Display;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};

use bytes::Bytes;
use futures::future::BoxFuture;
use oxidelake_core::EngineError;
use parquet::DecodeResult;
use parquet::file::metadata::{
    ColumnChunkMetaData, PageIndexPolicy, ParquetMetaData, ParquetMetaDataPushDecoder,
    ParquetMetaDataReader,
};
use parquet::file::page_index::column_index::ColumnIndexMetaData;

/// "PAR1" at the start of every Parquet file.
const HEAD_MAGIC: u64 = 4;
/// The footer's last eight bytes: its length, then "PAR1".
const FOOTER_TAIL: u64 = 8;
const MAGIC: &[u8; 4] = b"PAR1";
const ENCRYPTED_MAGIC: &[u8; 4] = b"PARE";

/// Deepest Thrift nesting the walk follows (structs in lists in structs);
/// parquet's own skip gives up at the same depth.
const MAX_THRIFT_DEPTH: u32 = 64;

/// Deepest schema accepted. Each layer that turns a Parquet schema into an
/// Arrow one recurses once per level, so a footer nesting thousands of groups
/// overflows the stack, which is an abort rather than an error. Nothing real
/// comes close: 64 levels of nested groups is already unusual.
const MAX_SCHEMA_DEPTH: usize = 64;

/// A refusal naming the file and, where there is one, the column.
fn refuse(location: &str, what: impl Display) -> EngineError {
    EngineError::format(format!("parquet file {location}: {what}"))
}

/// Runs a parquet decoding step that must not take the process down: a panic
/// in it is the file's fault, reported as such rather than unwound through
/// the caller (the last line of defence behind the checks below).
pub(crate) fn guarded<T>(
    location: &str,
    what: &str,
    step: impl FnOnce() -> parquet::errors::Result<T>,
) -> Result<T, EngineError> {
    match catch_unwind(AssertUnwindSafe(step)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(refuse(location, format!("{what}: {e}"))),
        Err(panic) => Err(refuse(
            location,
            format!(
                "the parquet decoder panicked reading the {what}: {}",
                panic_message(panic.as_ref())
            ),
        )),
    }
}

pub(crate) fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "a non-string panic payload".to_owned()
    }
}

/// Reads byte ranges of one file. Callers only ask for ranges already checked
/// to lie inside it.
pub(crate) trait Fetch: Send {
    fn fetch(&mut self, range: Range<u64>) -> BoxFuture<'_, Result<Bytes, EngineError>>;
}

/// Reads, checks and decodes the metadata of a `size`-byte Parquet file,
/// page index included, without asking `fetch` for a byte outside it.
pub(crate) async fn read_checked_metadata<F: Fetch>(
    fetch: &mut F,
    location: &str,
    size: u64,
) -> Result<ParquetMetaData, EngineError> {
    if size < HEAD_MAGIC + FOOTER_TAIL {
        return Err(refuse(
            location,
            format!("{size} bytes is too short to be a Parquet file"),
        ));
    }
    let tail = fetch_exact(fetch, location, size - FOOTER_TAIL..size).await?;
    let magic = &tail[4..8];
    if magic == ENCRYPTED_MAGIC {
        return Err(refuse(
            location,
            "its footer is encrypted, which is not supported",
        ));
    }
    if magic != MAGIC {
        return Err(refuse(location, "it does not end with the Parquet magic"));
    }
    let footer_len = u64::from(u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]));
    // parquet 59's push decoder computes `size - 8 - footer_len` unchecked:
    // a length larger than the file underflows (a panic in debug builds, a
    // read at a wrapped offset in release ones).
    if footer_len > size - HEAD_MAGIC - FOOTER_TAIL {
        return Err(refuse(
            location,
            format!("the footer claims {footer_len} bytes of a {size}-byte file"),
        ));
    }
    let footer_start = size - FOOTER_TAIL - footer_len;
    let footer = fetch_exact(fetch, location, footer_start..size - FOOTER_TAIL).await?;
    walk_footer(&footer).map_err(|e| refuse(location, format!("footer: {e}")))?;
    let metadata = guarded(location, "footer", || {
        ParquetMetaDataReader::decode_metadata(&footer)
    })?;
    check_chunks(location, size, &metadata)?;
    let metadata = load_page_index(fetch, location, size, metadata).await?;
    check_pages(location, &metadata)?;
    Ok(metadata)
}

async fn fetch_exact<F: Fetch>(
    fetch: &mut F,
    location: &str,
    range: Range<u64>,
) -> Result<Bytes, EngineError> {
    let want = range.end - range.start;
    let bytes = fetch.fetch(range.clone()).await?;
    if bytes.len() as u64 != want {
        return Err(refuse(
            location,
            format!(
                "bytes {}..{} came back {} long; the file changed while it was read",
                range.start,
                range.end,
                bytes.len()
            ),
        ));
    }
    Ok(bytes)
}

/// Loads the page index of `metadata` (if the file has one) through `fetch`,
/// walking each column's index before the decoder sees it. A file without a
/// page index comes back unchanged.
pub(crate) async fn load_page_index<F: Fetch>(
    fetch: &mut F,
    location: &str,
    size: u64,
    metadata: ParquetMetaData,
) -> Result<ParquetMetaData, EngineError> {
    // check_chunks has put every index range inside the file; collected
    // before the metadata moves into the decoder.
    let mut indexes = Vec::new();
    for (r, row_group) in metadata.row_groups().iter().enumerate() {
        for column in row_group.columns() {
            let name = column.column_path().string();
            for (kind, offset, length) in [
                (
                    "column index",
                    column.column_index_offset(),
                    column.column_index_length(),
                ),
                (
                    "offset index",
                    column.offset_index_offset(),
                    column.offset_index_length(),
                ),
            ] {
                if let (Some(offset), Some(length)) = (offset, length)
                    && let (Ok(offset), Ok(length)) = (u64::try_from(offset), u64::try_from(length))
                {
                    indexes.push((r, name.clone(), kind, offset..offset + length));
                }
            }
        }
    }
    if indexes.is_empty() {
        return Ok(metadata);
    }
    let mut decoder = guarded(location, "page index", || {
        ParquetMetaDataPushDecoder::try_new_with_metadata(size, metadata)
    })?
    .with_page_index_policy(PageIndexPolicy::Optional);
    loop {
        match guarded(location, "page index", || decoder.try_decode())? {
            DecodeResult::Data(metadata) => return Ok(metadata),
            DecodeResult::NeedsData(ranges) => {
                for range in ranges {
                    if range.start > range.end || range.end > size {
                        return Err(refuse(
                            location,
                            format!(
                                "the page index at {}..{} is outside the {size}-byte file",
                                range.start, range.end
                            ),
                        ));
                    }
                    let bytes = fetch_exact(fetch, location, range.clone()).await?;
                    for (r, name, kind, index) in &indexes {
                        if index.start >= range.start && index.end <= range.end {
                            let at = usize::try_from(index.start - range.start).unwrap_or(0);
                            let end = usize::try_from(index.end - range.start).unwrap_or(0);
                            walk_struct(&bytes[at..end]).map_err(|e| {
                                refuse(
                                    location,
                                    format!("row group {r} column {name}: {kind}: {e}"),
                                )
                            })?;
                        }
                    }
                    decoder
                        .push_range(range, bytes)
                        .map_err(|e| refuse(location, format!("page index: {e}")))?;
                }
            }
            DecodeResult::Finished => {
                return Err(refuse(
                    location,
                    "the page index decoder returned no metadata",
                ));
            }
        }
    }
}

/// The byte range a column chunk occupies, as the reader computes it
/// (dictionary page first when there is one), checked to lie in the file.
fn chunk_range(
    location: &str,
    row_group: usize,
    column: &ColumnChunkMetaData,
    size: u64,
) -> Result<Range<u64>, EngineError> {
    let at = |what: String| {
        refuse(
            location,
            format!(
                "row group {row_group} column {}: {what}",
                column.column_path().string()
            ),
        )
    };
    let data = column.data_page_offset();
    let start = column.dictionary_page_offset().unwrap_or(data);
    let length = column.compressed_size();
    if start < 0 || data < 0 {
        return Err(at(format!(
            "column chunk starts at a negative offset ({})",
            start.min(data)
        )));
    }
    if length < 0 {
        return Err(at(format!("column chunk has a negative length ({length})")));
    }
    let (start, length) = (start.unsigned_abs(), length.unsigned_abs());
    match start.checked_add(length) {
        Some(end) if end <= size && data.unsigned_abs() <= size => Ok(start..end),
        _ => Err(at(format!(
            "column chunk at {start} with length {length} ends past the end of the \
             {size}-byte file"
        ))),
    }
}

/// An optional `offset`/`length` pair the reader may fetch, checked to lie
/// inside the file.
fn optional_range(
    location: &str,
    row_group: usize,
    column: &ColumnChunkMetaData,
    what: &str,
    offset: Option<i64>,
    length: Option<i32>,
    size: u64,
) -> Result<(), EngineError> {
    let offset = offset.map(i128::from);
    let length = length.map(i128::from);
    let start = offset.unwrap_or(0);
    let end = start + length.unwrap_or(0);
    let bad = start < 0
        || length.is_some_and(|l| l < 0)
        || end > i128::from(size)
        || offset.is_some_and(|o| o > i128::from(size));
    if bad {
        return Err(refuse(
            location,
            format!(
                "row group {row_group} column {}: the {what} at {} with length {} lies \
                 outside the {size}-byte file",
                column.column_path().string(),
                offset.map_or_else(|| "-".to_owned(), |o| o.to_string()),
                length.map_or_else(|| "-".to_owned(), |l| l.to_string()),
            ),
        ));
    }
    Ok(())
}

/// Every range the column chunk metadata of a `size`-byte file declares:
/// the chunks, their page indexes and their Bloom filters lie inside the
/// file, and no row group has a negative row count.
pub(crate) fn check_chunks(
    location: &str,
    size: u64,
    metadata: &ParquetMetaData,
) -> Result<(), EngineError> {
    for (r, row_group) in metadata.row_groups().iter().enumerate() {
        if row_group.num_rows() < 0 {
            return Err(refuse(
                location,
                format!(
                    "row group {r} has a negative row count ({})",
                    row_group.num_rows()
                ),
            ));
        }
        for column in row_group.columns() {
            chunk_range(location, r, column, size)?;
            optional_range(
                location,
                r,
                column,
                "column index",
                column.column_index_offset(),
                column.column_index_length(),
                size,
            )?;
            optional_range(
                location,
                r,
                column,
                "offset index",
                column.offset_index_offset(),
                column.offset_index_length(),
                size,
            )?;
            optional_range(
                location,
                r,
                column,
                "Bloom filter",
                column.bloom_filter_offset(),
                column.bloom_filter_length(),
                size,
            )?;
        }
    }
    Ok(())
}

/// The page index against the chunks it describes: one offset index per
/// column of every row group, each page inside its chunk and after the page
/// before it, row numbers that only go forward and stay inside the row
/// group, and a column index (where there is one) describing as many pages.
/// A file without an offset index passes: there is nothing to check.
pub(crate) fn check_pages(location: &str, metadata: &ParquetMetaData) -> Result<(), EngineError> {
    let Some(offset_index) = metadata.offset_index() else {
        return Ok(());
    };
    let row_groups = metadata.row_groups();
    if offset_index.len() != row_groups.len() {
        return Err(refuse(
            location,
            format!(
                "the offset index describes {} row groups of {}",
                offset_index.len(),
                row_groups.len()
            ),
        ));
    }
    for (r, (row_group, columns)) in row_groups.iter().zip(offset_index).enumerate() {
        if columns.len() != row_group.num_columns() {
            return Err(refuse(
                location,
                format!(
                    "row group {r}: the offset index describes {} columns of {}",
                    columns.len(),
                    row_group.num_columns()
                ),
            ));
        }
        let rows = row_group.num_rows();
        for (c, (column, index)) in row_group.columns().iter().zip(columns).enumerate() {
            let name = column.column_path().string();
            let chunk = chunk_range(location, r, column, u64::MAX)?;
            let (mut previous_end, mut previous_row) = (chunk.start, 0i64);
            for (p, page) in index.page_locations().iter().enumerate() {
                let at = |what: String| {
                    refuse(
                        location,
                        format!("row group {r} column {name}: page {p} {what}"),
                    )
                };
                if page.offset < 0 || page.compressed_page_size < 0 {
                    return Err(at(format!(
                        "has a negative offset or size ({}, {})",
                        page.offset, page.compressed_page_size
                    )));
                }
                let start = page.offset.unsigned_abs();
                let end = start + u64::from(page.compressed_page_size.unsigned_abs());
                if start < previous_end || end > chunk.end {
                    return Err(at(format!(
                        "at {start}..{end} lies outside its column chunk {}..{} or overlaps \
                         the page before it",
                        chunk.start, chunk.end
                    )));
                }
                if page.first_row_index < previous_row || page.first_row_index > rows {
                    return Err(at(format!(
                        "starts at row {}, outside rows {previous_row}..={rows}",
                        page.first_row_index
                    )));
                }
                previous_end = end;
                previous_row = page.first_row_index;
            }
            if let Some(column_index) = metadata.column_index() {
                let pages = column_index
                    .get(r)
                    .and_then(|columns| columns.get(c))
                    .filter(|index| !matches!(index, ColumnIndexMetaData::NONE))
                    .map(ColumnIndexMetaData::num_pages);
                let located = index.page_locations().len() as u64;
                if let Some(pages) = pages
                    && pages != located
                {
                    return Err(refuse(
                        location,
                        format!(
                            "row group {r} column {name}: the column index describes {pages} \
                             pages and the offset index {located}"
                        ),
                    ));
                }
            }
        }
    }
    if let Some(column_index) = metadata.column_index()
        && (column_index.len() != row_groups.len()
            || column_index
                .iter()
                .zip(row_groups)
                .any(|(columns, row_group)| columns.len() != row_group.num_columns()))
    {
        return Err(refuse(
            location,
            "the column index does not describe every column of every row group",
        ));
    }
    Ok(())
}

/// Checks the footer's Thrift encoding (a `FileMetaData` struct) without
/// decoding it: every list, map and binary length it declares is backed by
/// the bytes that follow, structs nest no deeper than [`MAX_THRIFT_DEPTH`],
/// and the schema's group sizes and depth are plausible.
///
/// parquet 59 reserves `Vec::with_capacity(n)` for the row-group list and for
/// each schema group's children straight from the declared `n`, so a footer
/// of a few hundred bytes can ask for hundreds of GiB; walking first means
/// every count the decoder reserves for was already matched by real
/// elements.
pub(crate) fn walk_footer(footer: &[u8]) -> Result<(), String> {
    let mut walk = Walk {
        buf: footer,
        pos: 0,
    };
    let mut last = 0i16;
    while let Some((id, ty)) = walk.field_header(&mut last)? {
        if id == 2 && ty == LIST {
            walk.schema()?;
        } else {
            walk.skip(ty, 1)?;
        }
    }
    Ok(())
}

/// Checks one Thrift struct (a column or offset index) the same way. The
/// offset index decoder's fast path reserves its page list from the declared
/// count too.
pub(crate) fn walk_struct(bytes: &[u8]) -> Result<(), String> {
    let mut walk = Walk { buf: bytes, pos: 0 };
    walk.skip(STRUCT, 1)
}

// Thrift compact protocol type ids.
const BOOL_TRUE: u8 = 1;
const BOOL_FALSE: u8 = 2;
const BYTE: u8 = 3;
const I16: u8 = 4;
const I32: u8 = 5;
const I64: u8 = 6;
const DOUBLE: u8 = 7;
const BINARY: u8 = 8;
const LIST: u8 = 9;
const SET: u8 = 10;
const MAP: u8 = 11;
const STRUCT: u8 = 12;
const UUID: u8 = 13;

struct Walk<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Walk<'_> {
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn byte(&mut self) -> Result<u8, String> {
        let byte = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| format!("ends at byte {} in the middle of a value", self.pos))?;
        self.pos += 1;
        Ok(byte)
    }

    fn skip_bytes(&mut self, n: u64) -> Result<(), String> {
        if n > self.remaining() as u64 {
            return Err(format!(
                "declares {n} bytes at byte {} where {} remain",
                self.pos,
                self.remaining()
            ));
        }
        self.pos += n as usize;
        Ok(())
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(format!(
            "a varint longer than 10 bytes at byte {}",
            self.pos
        ))
    }

    fn zigzag(&mut self) -> Result<i64, String> {
        let v = self.varint()?;
        Ok((v >> 1) as i64 ^ -((v & 1) as i64))
    }

    /// A field header, or `None` at the struct's stop byte.
    fn field_header(&mut self, last: &mut i16) -> Result<Option<(i16, u8)>, String> {
        let byte = self.byte()?;
        if byte == 0 {
            return Ok(None);
        }
        let (delta, ty) = (byte >> 4, byte & 0x0f);
        let id = if delta == 0 {
            i16::try_from(self.zigzag()?).map_err(|_| "a field id out of range".to_owned())?
        } else {
            last.checked_add(i16::from(delta))
                .ok_or_else(|| "a field id out of range".to_owned())?
        };
        *last = id;
        Ok(Some((id, ty)))
    }

    /// A list or set header: element type and a count, which must fit in
    /// the bytes left (every element takes at least one).
    fn list_header(&mut self) -> Result<(u8, u64), String> {
        let header = self.byte()?;
        if header == 0 {
            // Some writers encode an empty list as a bare zero byte.
            return Ok((BYTE, 0));
        }
        let ty = header & 0x0f;
        let short = header >> 4;
        let size = if short == 15 {
            self.varint()?
        } else {
            u64::from(short)
        };
        if size > i32::MAX as u64 || size > self.remaining() as u64 {
            return Err(format!(
                "a list declares {size} elements at byte {} where {} bytes remain",
                self.pos,
                self.remaining()
            ));
        }
        Ok((ty, size))
    }

    fn element(&mut self, ty: u8, depth: u32) -> Result<(), String> {
        // Inside a collection a bool is a whole byte, not part of a header.
        if ty == BOOL_TRUE || ty == BOOL_FALSE {
            self.skip_bytes(1)
        } else {
            self.skip(ty, depth)
        }
    }

    fn skip(&mut self, ty: u8, depth: u32) -> Result<(), String> {
        if depth > MAX_THRIFT_DEPTH {
            return Err(format!("nests deeper than {MAX_THRIFT_DEPTH} levels"));
        }
        match ty {
            BOOL_TRUE | BOOL_FALSE => Ok(()),
            BYTE => self.skip_bytes(1),
            I16 | I32 | I64 => self.varint().map(drop),
            DOUBLE => self.skip_bytes(8),
            UUID => self.skip_bytes(16),
            BINARY => {
                let n = self.varint()?;
                self.skip_bytes(n)
            }
            LIST | SET => {
                let (element, size) = self.list_header()?;
                for _ in 0..size {
                    self.element(element, depth + 1)?;
                }
                Ok(())
            }
            MAP => {
                let size = self.varint()?;
                if size > self.remaining() as u64 {
                    return Err(format!(
                        "a map declares {size} entries at byte {} where {} bytes remain",
                        self.pos,
                        self.remaining()
                    ));
                }
                if size > 0 {
                    let types = self.byte()?;
                    for _ in 0..size {
                        self.element(types >> 4, depth + 1)?;
                        self.element(types & 0x0f, depth + 1)?;
                    }
                }
                Ok(())
            }
            STRUCT => {
                let mut last = 0i16;
                while let Some((_, ty)) = self.field_header(&mut last)? {
                    self.skip(ty, depth + 1)?;
                }
                Ok(())
            }
            other => Err(format!(
                "an unknown Thrift type {other} at byte {}",
                self.pos
            )),
        }
    }

    /// `FileMetaData.schema`: a flattened tree whose groups say how many of
    /// the following elements are their children. A group claiming more
    /// children than there are elements would have the decoder reserve that
    /// many slots; a chain of nested groups would have it recurse that deep.
    fn schema(&mut self) -> Result<(), String> {
        let (ty, size) = self.list_header()?;
        if ty != STRUCT {
            for _ in 0..size {
                self.element(ty, 2)?;
            }
            return Ok(());
        }
        // Children still expected by each group the walk is inside.
        let mut open: Vec<u64> = Vec::new();
        for i in 0..size {
            while open.last() == Some(&0) {
                open.pop();
            }
            if let Some(expected) = open.last_mut() {
                *expected -= 1;
            }
            let mut last = 0i16;
            let mut children = None;
            while let Some((id, ty)) = self.field_header(&mut last)? {
                if id == 5 && ty == I32 {
                    children = Some(self.zigzag()?);
                } else {
                    self.skip(ty, 3)?;
                }
            }
            if let Some(children) = children.filter(|&n| n != 0) {
                if children < 0 || children.unsigned_abs() > size {
                    return Err(format!(
                        "schema element {i} declares {children} children of a \
                         {size}-element schema"
                    ));
                }
                open.push(children.unsigned_abs());
                if open.len() > MAX_SCHEMA_DEPTH {
                    return Err(format!(
                        "the schema nests deeper than {MAX_SCHEMA_DEPTH} levels"
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The last line of defence: a decoder that panics is a refusal naming
    /// the file, not an unwind through the caller.
    #[test]
    fn a_panicking_decoder_is_a_format_error() {
        let err =
            guarded::<()>("t.parquet", "footer", || panic!("index out of bounds")).unwrap_err();
        match err {
            EngineError::Format(m) => {
                assert!(m.contains("t.parquet"), "{m}");
                assert!(m.contains("panicked reading the footer"), "{m}");
                assert!(m.contains("index out of bounds"), "{m}");
            }
            other => panic!("expected Format, got {other:?}"),
        }
        let err = guarded::<()>("t.parquet", "page index", || {
            Err(parquet::errors::ParquetError::General("bad".into()))
        })
        .unwrap_err();
        assert!(
            matches!(err, EngineError::Format(m) if m.contains("page index: Parquet error: bad"))
        );
    }

    #[test]
    fn the_walk_accepts_what_parquet_writes_and_refuses_counts_without_bytes() {
        // struct { 1: list<i32> [1, 2] }, then stop.
        assert!(walk_struct(&[0x19, 0x25, 0x02, 0x04, 0x00]).is_ok());
        // The same list declaring 2^31 - 1 elements in the long form.
        let err = walk_struct(&[0x19, 0xF5, 0xFF, 0xFF, 0xFF, 0xFF, 0x07, 0x00]).unwrap_err();
        assert!(err.contains("declares 2147483647 elements"), "{err}");
        // A binary longer than what is left.
        let err = walk_struct(&[0x18, 0x7F, 0x00]).unwrap_err();
        assert!(err.contains("declares 127 bytes"), "{err}");
        // A map declaring more entries than bytes.
        let err = walk_struct(&[0x1B, 0x7F, 0x55]).unwrap_err();
        assert!(err.contains("map declares 127 entries"), "{err}");
        // Structs nested past the limit.
        let mut deep = vec![0x1C; 80];
        deep.extend(std::iter::repeat_n(0x00, 81));
        let err = walk_struct(&deep).unwrap_err();
        assert!(err.contains("nests deeper"), "{err}");
        // Truncated in the middle of a value.
        assert!(walk_struct(&[0x15]).unwrap_err().contains("ends at byte"));
        // A bool field has no payload; a bool list stores a byte each.
        assert!(walk_struct(&[0x11, 0x19, 0x21, 0x01, 0x00, 0x00]).is_ok());
    }
}
