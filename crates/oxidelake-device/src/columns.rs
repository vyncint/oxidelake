//! Byte-level (de)serialization of Arrow columns for device transfers.
//!
//! GPU backends move raw Arrow buffers: fixed-width values plus a packed
//! validity bitmap. This module extracts them from any (possibly sliced)
//! array of a supported type and rebuilds arrays from downloaded bytes. It
//! is backend-independent, so it is unit-tested on the CPU. The bitmap has
//! one bit per row: a vector column's elements travel without validity of
//! their own, so [`extract`] refuses a NULL element inside a valid row rather
//! than return it as a number (#105).
//!
//! Extraction is zero-copy: values are [`Buffer`] slices of the array's own
//! allocation and validity is the array's bitmap (re-packed only when a slice
//! does not start on a byte boundary). Rebuilding takes ownership of the
//! downloaded buffers, so a round trip copies bytes exactly as often as the
//! transport requires — once each way over PCIe, not at all on unified memory
//! when the backend wraps its own allocation in a `Buffer`.

use std::sync::Arc;

use arrow::array::{Array, ArrayData, ArrayRef, make_array};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow::datatypes::DataType;
use oxidelake_core::EngineError;

/// Raw bytes of one column: values, optional packed validity bitmap, and for
/// fixed-size lists the number of child values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnBytes {
    /// Arrow type of the column.
    pub data_type: DataType,
    /// Number of rows.
    pub len: usize,
    /// Fixed-width values, tightly packed from row 0.
    pub values: Buffer,
    /// Validity bitmap (LSB-first, one bit per row, starting at bit 0), `None`
    /// when all rows are valid.
    pub validity: Option<Buffer>,
    /// For `FixedSizeList` columns: total child values (`len * list_size`).
    pub child_len: usize,
}

impl ColumnBytes {
    /// Bytes occupied by values and validity together.
    pub fn total_bytes(&self) -> u64 {
        let v = self.validity.as_ref().map_or(0, Buffer::len);
        u64::try_from(self.values.len() + v).unwrap_or(u64::MAX)
    }
}

/// Byte width of a supported fixed-width primitive type.
pub const fn primitive_width(dt: &DataType) -> Option<usize> {
    match dt {
        DataType::Int64 | DataType::Float64 => Some(8),
        DataType::Float32 => Some(4),
        _ => None,
    }
}

/// `true` when columns of this type can live on a device in v1
/// (`Int64`, `Float64`, `Float32`, `FixedSizeList<Float32>`).
pub fn is_device_eligible(dt: &DataType) -> bool {
    primitive_width(dt).is_some() || oxidelake_core::params::vector_dimension(dt).is_some()
}

/// Bytes per row of a supported column type (`4 * n` for `FixedSizeList<Float32, n>`).
pub fn row_width(dt: &DataType) -> Result<usize, EngineError> {
    if let Some(w) = primitive_width(dt) {
        return Ok(w);
    }
    oxidelake_core::params::vector_dimension(dt)
        .map(|dim| dim * 4)
        .ok_or_else(|| {
            EngineError::unsupported(
                "device.column_type",
                format!("{dt:?} columns are not device-resident in v1"),
            )
        })
}

/// The validity bitmap starting at bit 0: the array's own buffer when the
/// slice is byte-aligned, otherwise a re-packed copy.
fn validity_bytes(nulls: &NullBuffer) -> Buffer {
    nulls.inner().sliced()
}

fn slice_checked(
    buffer: &Buffer,
    start: usize,
    len: usize,
    what: &str,
) -> Result<Buffer, EngineError> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| EngineError::execution(format!("{what}: buffer range overflows")))?;
    if end > buffer.len() {
        return Err(EngineError::format(format!(
            "{what}: array needs {end} bytes but its buffer holds {}",
            buffer.len()
        )));
    }
    Ok(buffer.slice_with_length(start, len))
}

/// The first row of a vector column that is valid but holds a NULL element,
/// or `None`, which it always is for a column of any other type.
///
/// It is the one content of a device-eligible column that the device format
/// cannot carry: the bitmap has one bit per *row*, so the element would come
/// back from a device as a valid number (#105). [`extract`] refuses such a
/// column, and the operator layer keeps it on the host instead of uploading
/// it. NULL elements under a NULL row do not count: the row is NULL whatever
/// they hold, and a nullable vector column built with `append_null` has
/// exactly those, so counting them would keep every nullable vector column
/// off the device.
pub fn row_with_null_element(array: &dyn Array) -> Result<Option<usize>, EngineError> {
    let DataType::FixedSizeList(field, size) = array.data_type() else {
        return Ok(None);
    };
    if *field.data_type() != DataType::Float32 {
        return Ok(None);
    }
    let size =
        usize::try_from(*size).map_err(|_| EngineError::plan("negative FixedSizeList size"))?;
    let data = array.to_data();
    let child = data
        .child_data()
        .first()
        .ok_or_else(|| EngineError::format("FixedSizeList array without child data"))?;
    null_element_row(data.nulls(), child.nulls(), data.offset(), data.len(), size)
}

/// [`row_with_null_element`] over a list's parts: `len` rows of `size`
/// elements from row `offset`, with `rows` and `elements` the list's and its
/// child's validity.
fn null_element_row(
    rows: Option<&NullBuffer>,
    elements: Option<&NullBuffer>,
    offset: usize,
    len: usize,
    size: usize,
) -> Result<Option<usize>, EngineError> {
    let Some(elements) = elements.filter(|e| e.null_count() > 0) else {
        return Ok(None);
    };
    let end = offset
        .checked_add(len)
        .and_then(|rows| rows.checked_mul(size));
    if end.is_none_or(|end| end > elements.len()) {
        return Err(EngineError::format(format!(
            "vector column: rows {offset}..{} of {size} elements reach past its {} child values",
            offset.saturating_add(len),
            elements.len()
        )));
    }
    let elements = elements.slice(offset * size, len * size);
    let nulls = elements.null_count();
    if nulls == 0 {
        return Ok(None);
    }
    // A nullable column whose NULL rows hold NULL elements is the common
    // shape, and every upload of one asks this. Counting the NULL elements
    // under NULL rows (a popcount per NULL row) answers it without visiting
    // each element of each valid row: if they are all of them, no valid row
    // holds one.
    if let Some(rows) = rows {
        let bits = elements.inner();
        let valid_in = |row: usize| {
            bits.inner()
                .count_set_bits_offset(bits.offset() + row * size, size)
        };
        let under_null_rows: usize = (0..len)
            .filter(|&row| rows.is_null(row))
            .map(|row| size - valid_in(row))
            .sum();
        if under_null_rows == nulls {
            return Ok(None);
        }
    }
    Ok(elements
        .iter()
        .enumerate()
        .filter(|&(_, valid)| !valid)
        .map(|(element, _)| element / size)
        .find(|&row| rows.is_none_or(|r| r.is_valid(row))))
}

/// Extracts the raw bytes of a supported column without copying them.
///
/// A vector column with a NULL element inside a valid row
/// ([`row_with_null_element`]) is refused as [`EngineError::Unsupported`]:
/// this format cannot carry the element's NULL.
pub fn extract(array: &ArrayRef) -> Result<ColumnBytes, EngineError> {
    let data = array.to_data();
    let len = data.len();
    let offset = data.offset();
    let validity = data.nulls().map(validity_bytes);
    match array.data_type() {
        dt @ (DataType::Int64 | DataType::Float64 | DataType::Float32) => {
            let width = primitive_width(dt).unwrap_or(8);
            let buffer = data
                .buffers()
                .first()
                .ok_or_else(|| EngineError::format("primitive array without a value buffer"))?;
            let values = slice_checked(buffer, offset * width, len * width, "primitive column")?;
            Ok(ColumnBytes {
                data_type: dt.clone(),
                len,
                values,
                validity,
                child_len: 0,
            })
        }
        dt @ DataType::FixedSizeList(field, size) if *field.data_type() == DataType::Float32 => {
            let size = usize::try_from(*size)
                .map_err(|_| EngineError::plan("negative FixedSizeList size"))?;
            let child = data
                .child_data()
                .first()
                .ok_or_else(|| EngineError::format("FixedSizeList array without child data"))?;
            let buffer = child
                .buffers()
                .first()
                .ok_or_else(|| EngineError::format("FixedSizeList child without a value buffer"))?;
            let start = (child.offset() + offset * size) * 4;
            let values = slice_checked(buffer, start, len * size * 4, "vector column")?;
            if let Some(row) = null_element_row(data.nulls(), child.nulls(), offset, len, size)? {
                return Err(EngineError::unsupported(
                    "device.vector_null_element",
                    format!(
                        "row {row} of the vector column is valid but holds a NULL element, \
                         which the device format (one validity bit per row) would return \
                         as a number"
                    ),
                ));
            }
            Ok(ColumnBytes {
                data_type: dt.clone(),
                len,
                values,
                validity,
                child_len: len * size,
            })
        }
        other => Err(EngineError::unsupported(
            "device.column_type",
            format!("{other:?} columns are not device-resident in v1"),
        )),
    }
}

/// Rebuilds an Arrow array from extracted or downloaded bytes, taking
/// ownership of the buffers (no copy).
pub fn rebuild(col: ColumnBytes) -> Result<ArrayRef, EngineError> {
    let ColumnBytes {
        data_type,
        len,
        values,
        validity,
        child_len,
    } = col;
    let nulls = validity.map(|bytes| NullBuffer::new(BooleanBuffer::new(bytes, 0, len)));
    let data: ArrayData = match &data_type {
        DataType::Int64 | DataType::Float64 | DataType::Float32 => {
            ArrayData::builder(data_type.clone())
                .len(len)
                .nulls(nulls)
                .add_buffer(values)
                .build()?
        }
        DataType::FixedSizeList(field, _) => {
            let child = ArrayData::builder(field.data_type().clone())
                .len(child_len)
                .add_buffer(values)
                .build()?;
            ArrayData::builder(data_type.clone())
                .len(len)
                .nulls(nulls)
                .add_child_data(child)
                .build()?
        }
        other => {
            return Err(EngineError::unsupported(
                "device.column_type",
                format!("cannot rebuild {other:?} columns"),
            ));
        }
    };
    Ok(make_array(data))
}

/// Extracts every column of a batch.
pub fn extract_batch(batch: &arrow::array::RecordBatch) -> Result<Vec<ColumnBytes>, EngineError> {
    batch.columns().iter().map(extract).collect()
}

/// Rebuilds a batch from per-column bytes.
pub fn rebuild_batch(
    schema: &arrow::datatypes::SchemaRef,
    columns: Vec<ColumnBytes>,
) -> Result<arrow::array::RecordBatch, EngineError> {
    let arrays = columns
        .into_iter()
        .map(rebuild)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(arrow::array::RecordBatch::try_new(
        Arc::clone(schema),
        arrays,
    )?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use arrow::array::{
        FixedSizeListArray, Float32Array, Float64Array, Int64Array, RecordBatch, StringArray,
    };
    use arrow::datatypes::{Field, Schema};

    use super::*;

    fn sample() -> RecordBatch {
        let vectors = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            2,
            Arc::new(Float32Array::from(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ])),
            Some(NullBuffer::from(vec![true, false, true, true])),
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
            Field::new("f", DataType::Float32, false),
            Field::new("vec", vectors.data_type().clone(), true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3), Some(4)])),
                Arc::new(Float64Array::from(vec![
                    Some(0.5),
                    Some(1.5),
                    None,
                    Some(3.5),
                ])),
                Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(vectors),
            ],
        )
        .unwrap()
    }

    #[test]
    fn round_trip_full_batch() {
        let batch = sample();
        let cols = extract_batch(&batch).unwrap();
        assert_eq!(
            cols[0].validity.as_ref().map(Buffer::as_slice),
            Some(&[0b1101u8][..])
        );
        assert!(cols[2].validity.is_none());
        assert_eq!(cols[3].child_len, 8);
        let back = rebuild_batch(&batch.schema(), cols).unwrap();
        assert_eq!(back, batch);
    }

    #[test]
    fn extraction_does_not_copy_values() {
        let batch = sample();
        let cols = extract_batch(&batch).unwrap();
        let original = batch.column(0).to_data();
        assert_eq!(
            cols[0].values.as_ptr(),
            original.buffers()[0].as_ptr(),
            "Int64 values should alias the array's buffer"
        );
    }

    #[test]
    fn round_trip_sliced_batch_uses_offsets() {
        let batch = sample().slice(1, 2);
        let cols = extract_batch(&batch).unwrap();
        assert_eq!(cols[0].len, 2);
        assert_eq!(cols[0].values.len(), 16);
        assert_eq!(cols[3].values.len(), 2 * 2 * 4);
        // A slice starting at bit 1 re-packs the bitmap so bit 0 is row 0.
        assert_eq!(
            cols[0].validity.as_ref().unwrap().as_slice()[0] & 0b11,
            0b10
        );
        let back = rebuild_batch(&batch.schema(), cols).unwrap();
        assert_eq!(back, batch);
    }

    #[test]
    fn row_widths_and_eligibility() {
        assert_eq!(row_width(&DataType::Int64).unwrap(), 8);
        assert_eq!(row_width(&DataType::Float32).unwrap(), 4);
        let vec3 =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 3);
        assert_eq!(row_width(&vec3).unwrap(), 12);
        assert!(row_width(&DataType::Utf8).unwrap_err().is_unsupported());
        assert!(is_device_eligible(&vec3));
        assert!(!is_device_eligible(&DataType::Utf8));
    }

    #[test]
    fn unsupported_types_are_reported_not_panicked() {
        let arr: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        let err = extract(&arr).unwrap_err();
        assert!(err.is_unsupported());
    }

    /// Two-element vectors over `elements`, with optional row validity.
    fn vectors(elements: Vec<Option<f32>>, rows: Option<Vec<bool>>) -> ArrayRef {
        Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                2,
                Arc::new(Float32Array::from(elements)),
                rows.map(NullBuffer::from),
            )
            .unwrap(),
        )
    }

    /// #105: the rebuilt column used to hold `[1.0, 0.0]` where the original
    /// held `[1.0, NULL]` — the element bitmap was dropped on extraction.
    #[test]
    fn a_null_element_in_a_valid_row_is_refused() {
        let arr = vectors(
            vec![Some(1.0), None, Some(3.0), Some(4.0)],
            Some(vec![true, true]),
        );
        let err = extract(&arr).unwrap_err();
        assert!(err.is_unsupported(), "{err}");
        assert!(err.to_string().contains("row 0"), "{err}");
    }

    /// What a builder's `append_null` leaves: NULL elements under a NULL row.
    /// The row is NULL whatever they hold, so the column stays eligible and
    /// round-trips equal.
    #[test]
    fn null_elements_under_null_rows_round_trip() {
        let arr = vectors(
            vec![Some(1.0), Some(2.0), None, None, Some(5.0), Some(6.0)],
            Some(vec![true, false, true]),
        );
        let back = rebuild(extract(&arr).unwrap()).unwrap();
        assert_eq!(back.as_ref(), arr.as_ref());
    }

    /// Only the extracted rows are checked: a slice that leaves the NULL
    /// element out extracts, and round-trips equal.
    #[test]
    fn only_the_extracted_rows_are_checked() {
        let arr = vectors(
            vec![Some(1.0), None, Some(3.0), Some(4.0), Some(5.0), Some(6.0)],
            None,
        );
        assert!(extract(&arr).unwrap_err().is_unsupported());
        assert!(extract(&arr.slice(0, 1)).unwrap_err().is_unsupported());
        let tail = arr.slice(1, 2);
        let back = rebuild(extract(&tail).unwrap()).unwrap();
        assert_eq!(back.as_ref(), tail.as_ref());
    }

    /// The shapes that decide the check: a NULL element in a valid row is
    /// found, NULL elements under NULL rows never are, and the counting
    /// shortcut is not fooled when the counts match but the positions do not.
    #[test]
    fn null_elements_count_only_in_valid_rows() {
        let found = |elements: Vec<Option<f32>>, rows: Option<Vec<bool>>| {
            row_with_null_element(vectors(elements, rows).as_ref()).unwrap()
        };
        assert_eq!(
            found(vec![Some(1.0), Some(2.0), Some(3.0), None], None),
            Some(1)
        );
        assert_eq!(
            found(
                vec![None, None, Some(3.0), Some(4.0), Some(5.0), None],
                Some(vec![false, true, false]),
            ),
            None
        );
        // Two NULL elements, and two elements under the NULL row — but one of
        // the NULLs is in valid row 0.
        assert_eq!(
            found(
                vec![Some(1.0), None, None, Some(4.0)],
                Some(vec![true, false]),
            ),
            Some(0)
        );
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![None, Some(1)]));
        assert_eq!(row_with_null_element(ints.as_ref()).unwrap(), None);
    }

    /// arrow-rs lists always present offset 0 with a sliced child, so a
    /// list's own offset only reaches the check through its parts.
    #[test]
    fn the_check_counts_rows_from_the_list_offset() {
        let elements = NullBuffer::from(vec![true, false, true, true, false, false]);
        // Rows 1 and 2 of a list of pairs: [valid, valid] and [NULL, NULL]
        // under a NULL row; the NULL element in row 0 is outside.
        let rows = NullBuffer::from(vec![true, false]);
        assert_eq!(
            null_element_row(Some(&rows), Some(&elements), 1, 2, 2).unwrap(),
            None
        );
        assert_eq!(
            null_element_row(None, Some(&elements), 0, 1, 2).unwrap(),
            Some(0)
        );
        // Rows past the child are an error, not a panic.
        assert!(null_element_row(None, Some(&elements), 2, 2, 2).is_err());
    }
}
