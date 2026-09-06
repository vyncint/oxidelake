//! Byte-level (de)serialization of Arrow columns for device transfers.
//!
//! GPU backends move raw Arrow buffers: fixed-width values plus a packed
//! validity bitmap. This module extracts them from any (possibly sliced)
//! array of a supported type and rebuilds arrays from downloaded bytes. It
//! is backend-independent, so it is unit-tested on the CPU.
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

/// Extracts the raw bytes of a supported column without copying them.
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
}
