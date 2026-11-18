//! How `oxide sql` renders a result set (#49).
//!
//! The table form is for a person reading a terminal; `json` and `csv` exist
//! so the same query can feed a script without a second tool to parse the box
//! drawing. All three render the *same* batches — the format is a rendering
//! choice made after execution, never a planning one, so a result cannot
//! differ between them.

use std::io::Write;
use std::str::FromStr;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::{Array, ArrowPrimitiveType, AsArray, PrimitiveArray, RecordBatch};
use datafusion::arrow::datatypes::{
    DataType, FieldRef, Float16Type, Float32Type, Float64Type, SchemaRef,
};
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::json::writer::{
    Encoder, EncoderFactory, EncoderOptions, JsonArray, NullableEncoder, WriterBuilder,
    make_encoder,
};
use datafusion::arrow::util::pretty::pretty_format_batches;
use oxidelake_core::EngineError;

/// Output formats accepted by `oxide sql --output`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OutputFormat {
    /// An aligned ASCII table (the default).
    #[default]
    Table,
    /// A JSON array of objects, one per row, each carrying every column: a
    /// NULL is `null`, and NaN and the infinities, which JSON has no number
    /// for, are the strings `"NaN"`, `"Infinity"` and `"-Infinity"`.
    Json,
    /// RFC 4180 CSV with a header row.
    Csv,
}

impl OutputFormat {
    /// Every format, in the order `--help` lists them.
    pub const ALL: [OutputFormat; 3] = [OutputFormat::Table, OutputFormat::Json, OutputFormat::Csv];

    /// The canonical lowercase name used on the command line.
    pub const fn as_str(self) -> &'static str {
        match self {
            OutputFormat::Table => "table",
            OutputFormat::Json => "json",
            OutputFormat::Csv => "csv",
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OutputFormat {
    type Err = EngineError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "table" => Ok(OutputFormat::Table),
            "json" => Ok(OutputFormat::Json),
            "csv" => Ok(OutputFormat::Csv),
            other => Err(EngineError::plan(format!(
                "unknown output format '{other}'; expected one of table, json, csv"
            ))),
        }
    }
}

/// Renders `batches` in `format`, terminated by a newline.
///
/// `schema` is carried separately because an empty result has no batch to take
/// it from, and a CSV reader handed nothing at all cannot tell an empty result
/// from a failed one — so the header row is written even when no row is.
pub fn render(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    format: OutputFormat,
) -> Result<String, EngineError> {
    match format {
        OutputFormat::Table => Ok(format!("{}\n", pretty_format_batches(batches)?)),
        OutputFormat::Json => {
            let mut buffer = Vec::new();
            // arrow-json leaves a NULL's key out of its object by default, so
            // a script could not tell a NULL from a column that does not
            // exist; and it writes NaN and both infinities as `null`, so it
            // could not tell them from a NULL either (#124).
            let mut writer = WriterBuilder::new()
                .with_explicit_nulls(true)
                .with_encoder_factory(Arc::new(NonFiniteFloats))
                .build::<_, JsonArray>(&mut buffer);
            writer.write_batches(&batches.iter().collect::<Vec<_>>())?;
            writer.finish()?;
            finish(buffer)
        }
        OutputFormat::Csv => {
            let mut buffer = Vec::new();
            let mut writer = datafusion::arrow::csv::Writer::new(&mut buffer);
            let empty;
            let batches = if batches.is_empty() {
                empty = [RecordBatch::new_empty(SchemaRef::clone(schema))];
                &empty[..]
            } else {
                batches
            };
            for batch in batches {
                writer.write(batch)?;
            }
            drop(writer);
            finish(buffer)
        }
    }
}

/// arrow-json's options with no encoder factory, which [`NonFiniteFloats`]
/// builds its fallback encoders with: the same options would hand the array
/// straight back to the factory.
static PLAIN: LazyLock<EncoderOptions> = LazyLock::new(EncoderOptions::default);

/// Writes NaN, infinity and negative infinity as the JSON strings `"NaN"`,
/// `"Infinity"` and `"-Infinity"` (#124).
///
/// JSON has no number for any of the three, and arrow-json writes all of
/// them as `null`: a NaN read back as a NULL, which a query engine must not
/// confuse. These strings are what Python's `float()` and JavaScript's
/// `Number()` read back as the same values. Every finite value is handed to
/// arrow-json's own encoder, so its formatting does not change. arrow-json
/// asks the factory for every array it encodes, the values of lists,
/// structs and dictionaries included, so a float nested anywhere is covered.
#[derive(Debug)]
struct NonFiniteFloats;

impl EncoderFactory for NonFiniteFloats {
    fn make_default_encoder<'a>(
        &self,
        field: &'a FieldRef,
        array: &'a dyn Array,
        _options: &'a EncoderOptions,
    ) -> Result<Option<NullableEncoder<'a>>, ArrowError> {
        let encoder: Box<dyn Encoder + 'a> = match array.data_type() {
            DataType::Float16 => Box::new(FloatEncoder::new(
                array.as_primitive::<Float16Type>(),
                field,
                |v| v.to_f64(),
            )?),
            DataType::Float32 => Box::new(FloatEncoder::new(
                array.as_primitive::<Float32Type>(),
                field,
                f64::from,
            )?),
            DataType::Float64 => Box::new(FloatEncoder::new(
                array.as_primitive::<Float64Type>(),
                field,
                |v| v,
            )?),
            _ => return Ok(None),
        };
        Ok(Some(NullableEncoder::new(encoder, array.nulls().cloned())))
    }
}

/// One float array: the three non-finite values as strings, everything else
/// through arrow-json's encoder for the same array.
struct FloatEncoder<'a, T: ArrowPrimitiveType> {
    array: &'a PrimitiveArray<T>,
    to_f64: fn(T::Native) -> f64,
    finite: NullableEncoder<'a>,
}

impl<'a, T: ArrowPrimitiveType> FloatEncoder<'a, T> {
    fn new(
        array: &'a PrimitiveArray<T>,
        field: &'a FieldRef,
        to_f64: fn(T::Native) -> f64,
    ) -> Result<Self, ArrowError> {
        Ok(Self {
            array,
            to_f64,
            finite: make_encoder(field, array, &PLAIN)?,
        })
    }
}

impl<T: ArrowPrimitiveType> Encoder for FloatEncoder<'_, T> {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        let value = (self.to_f64)(self.array.value(idx));
        if value.is_nan() {
            out.extend_from_slice(b"\"NaN\"");
        } else if value == f64::INFINITY {
            out.extend_from_slice(b"\"Infinity\"");
        } else if value == f64::NEG_INFINITY {
            out.extend_from_slice(b"\"-Infinity\"");
        } else {
            self.finite.encode(idx, out);
        }
    }
}

/// Turns a writer's buffer into a newline-terminated `String`.
fn finish(mut buffer: Vec<u8>) -> Result<String, EngineError> {
    if !buffer.ends_with(b"\n") {
        buffer.write_all(b"\n").map_err(EngineError::from)?;
    }
    String::from_utf8(buffer)
        .map_err(|e| EngineError::execution(format!("rendered output was not UTF-8: {e}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{
        ArrayRef, DictionaryArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array,
        Int64Array, StringArray, StructArray,
    };
    use datafusion::arrow::compute::cast;
    use datafusion::arrow::datatypes::{DataType, Field, Int8Type, Schema};

    use super::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn batch() -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(vec![Some(1), None])),
                Arc::new(StringArray::from(vec![Some("a,b"), None])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn every_name_round_trips() {
        for format in OutputFormat::ALL {
            assert_eq!(format.as_str().parse::<OutputFormat>().unwrap(), format);
        }
        assert_eq!("  CSV ".parse::<OutputFormat>().unwrap(), OutputFormat::Csv);
    }

    #[test]
    fn an_unknown_name_names_the_alternatives() {
        let message = "yaml".parse::<OutputFormat>().unwrap_err().to_string();
        assert!(message.contains("yaml"), "{message}");
        assert!(message.contains("table, json, csv"), "{message}");
    }

    /// Every object carries every column, a NULL as `null` (#124). arrow-json
    /// leaves a NULL's key out by default, and a script could not tell a
    /// NULL from a column that does not exist: this row used to be `{}`.
    #[test]
    fn json_is_an_array_of_objects_with_every_column() {
        let text = render(&schema(), &[batch()], OutputFormat::Json).unwrap();
        assert_eq!(
            text,
            "[{\"k\":1,\"name\":\"a,b\"},{\"k\":null,\"name\":null}]\n"
        );
    }

    /// NaN and the infinities have no JSON number, and arrow-json wrote all
    /// three as `null`, which is how a NULL is written (#124). They are
    /// strings now, in every float type and inside a list, while finite
    /// values and NULLs keep their encoding.
    #[test]
    fn json_writes_nan_and_the_infinities_as_strings() {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Float64, true),
            Field::new("f", DataType::Float32, true),
            Field::new("h", DataType::Float16, true),
            Field::new("v", DataType::FixedSizeList(Arc::clone(&item), 2), true),
        ]));
        let d = Float64Array::from(vec![
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(1.5),
            None,
        ]);
        let f = Float32Array::from(vec![
            Some(f32::NEG_INFINITY),
            Some(f32::NAN),
            Some(0.25),
            Some(f32::INFINITY),
            None,
        ]);
        // Arrow's cast keeps NaN and the infinities, and spares this crate a
        // dependency on `half` for one test column.
        let h = cast(
            &Float32Array::from(vec![
                Some(0.5),
                Some(f32::NEG_INFINITY),
                Some(f32::NAN),
                Some(f32::INFINITY),
                None,
            ]),
            &DataType::Float16,
        )
        .unwrap();
        let v = FixedSizeListArray::new(
            item,
            2,
            Arc::new(Float32Array::from(vec![
                Some(f32::NAN),
                Some(0.5),
                None,
                Some(f32::INFINITY),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
            ])),
            Some(vec![true, true, true, true, false].into()),
        );
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(d), Arc::new(f), h, Arc::new(v)],
        )
        .unwrap();
        let text = render(&schema, &[batch], OutputFormat::Json).unwrap();
        assert_eq!(
            text,
            concat!(
                "[{\"d\":\"NaN\",\"f\":\"-Infinity\",\"h\":0.5,\"v\":[\"NaN\",0.5]},",
                "{\"d\":\"Infinity\",\"f\":\"NaN\",\"h\":\"-Infinity\",\"v\":[null,\"Infinity\"]},",
                "{\"d\":\"-Infinity\",\"f\":0.25,\"h\":\"NaN\",\"v\":[0.0,0.0]},",
                "{\"d\":1.5,\"f\":\"Infinity\",\"h\":\"Infinity\",\"v\":[0.0,0.0]},",
                "{\"d\":null,\"f\":null,\"h\":null,\"v\":null}]\n",
            )
        );
    }

    /// arrow-json builds the encoder of a struct's fields and of a
    /// dictionary's values through the same factory, so a float there is
    /// written the same way, and a NULL field of a struct keeps its key.
    #[test]
    fn json_reaches_floats_inside_structs_and_dictionaries() {
        let x = Arc::new(Float64Array::from(vec![Some(f64::NAN), None])) as ArrayRef;
        let s = StructArray::from(vec![(
            Arc::new(Field::new("x", DataType::Float64, true)),
            x,
        )]);
        let e = DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(vec![1, 0]),
            Arc::new(Float64Array::from(vec![f64::NEG_INFINITY, 2.5])),
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", s.data_type().clone(), true),
            Field::new("e", e.data_type().clone(), true),
        ]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(s), Arc::new(e)]).unwrap();
        let text = render(&schema, &[batch], OutputFormat::Json).unwrap();
        assert_eq!(
            text,
            "[{\"s\":{\"x\":\"NaN\"},\"e\":2.5},{\"s\":{\"x\":null},\"e\":\"-Infinity\"}]\n"
        );
    }

    #[test]
    fn csv_quotes_an_embedded_comma_and_keeps_the_header() {
        let text = render(&schema(), &[batch()], OutputFormat::Csv).unwrap();
        assert_eq!(text, "k,name\n1,\"a,b\"\n,\n");
    }

    /// An empty result still says what the columns were: a script reading the
    /// CSV can tell "no rows" from "the query failed".
    #[test]
    fn an_empty_result_still_has_a_csv_header() {
        assert_eq!(
            render(&schema(), &[], OutputFormat::Csv).unwrap(),
            "k,name\n"
        );
        assert_eq!(render(&schema(), &[], OutputFormat::Json).unwrap(), "[]\n");
    }

    #[test]
    fn the_table_form_is_the_default() {
        assert_eq!(OutputFormat::default(), OutputFormat::Table);
        let text = render(&schema(), &[batch()], OutputFormat::Table).unwrap();
        assert!(text.contains("| k "), "{text}");
        assert!(text.ends_with("\n"), "{text}");
    }
}
