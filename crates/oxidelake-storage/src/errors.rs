//! Maps DataFusion/Parquet/object-store failures onto [`EngineError`] at the
//! storage boundary: corrupt or truncated files are `Format`, IO failures are
//! `Io`, our own errors come back unchanged.

use datafusion::arrow::error::ArrowError;
use datafusion::error::DataFusionError;
use oxidelake_core::EngineError;

/// Classifies a DataFusion error raised while reading or planning over files.
pub fn classify_error(err: DataFusionError) -> EngineError {
    match err {
        DataFusionError::ParquetError(e) => EngineError::format(format!("parquet: {e}")),
        DataFusionError::ArrowError(e, context) => match *e {
            ArrowError::IoError(_, io) => EngineError::Io(io),
            other => {
                let ctx = context.map(|c| format!(" ({c})")).unwrap_or_default();
                EngineError::format(format!("arrow: {other}{ctx}"))
            }
        },
        DataFusionError::IoError(io) => EngineError::Io(io),
        DataFusionError::ObjectStore(e) => match *e {
            object_store::Error::NotFound { path, source } => EngineError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{path}: {source}"),
            )),
            other => EngineError::Io(std::io::Error::other(other.to_string())),
        },
        DataFusionError::Context(_, inner) => classify_error(*inner),
        DataFusionError::External(e) => match e.downcast::<EngineError>() {
            Ok(engine) => *engine,
            Err(e) => EngineError::DataFusion(DataFusionError::External(e)),
        },
        other => EngineError::DataFusion(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parquet_and_arrow_parse_errors_are_format_errors() {
        let e = classify_error(DataFusionError::ParquetError(Box::new(
            parquet::errors::ParquetError::EOF("footer".into()),
        )));
        assert!(matches!(e, EngineError::Format(_)), "{e}");
        let e = classify_error(DataFusionError::ArrowError(
            Box::new(ArrowError::ParseError("bad".into())),
            Some("ctx".into()),
        ));
        assert!(matches!(e, EngineError::Format(m) if m.contains("ctx")));
    }

    #[test]
    fn io_and_own_errors_round_trip() {
        let e = classify_error(DataFusionError::IoError(std::io::Error::other("disk")));
        assert!(matches!(e, EngineError::Io(_)));
        let own = EngineError::unsupported("x", "y");
        let e = classify_error(DataFusionError::External(Box::new(own)));
        assert!(e.is_unsupported());
        let e = classify_error(DataFusionError::Context(
            "while scanning".into(),
            Box::new(DataFusionError::ParquetError(Box::new(
                parquet::errors::ParquetError::General("g".into()),
            ))),
        ));
        assert!(matches!(e, EngineError::Format(_)));
    }
}
