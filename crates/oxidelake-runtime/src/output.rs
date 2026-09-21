//! How `oxide sql` renders a result set (#49).
//!
//! The table form is for a person reading a terminal; `json` and `csv` exist
//! so the same query can feed a script without a second tool to parse the box
//! drawing. All three render the *same* batches — the format is a rendering
//! choice made after execution, never a planning one, so a result cannot
//! differ between them.

use std::io::Write;
use std::str::FromStr;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::util::pretty::pretty_format_batches;
use oxidelake_core::EngineError;

/// Output formats accepted by `oxide sql --output`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OutputFormat {
    /// An aligned ASCII table (the default).
    #[default]
    Table,
    /// A JSON array of objects, one per row.
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
            let mut writer = datafusion::arrow::json::ArrayWriter::new(&mut buffer);
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

    use datafusion::arrow::array::{Int64Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

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

    #[test]
    fn json_is_an_array_of_objects_and_drops_nulls() {
        let text = render(&schema(), &[batch()], OutputFormat::Json).unwrap();
        assert_eq!(text, "[{\"k\":1,\"name\":\"a,b\"},{}]\n");
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
