//! Maps DataFusion/Parquet/object-store failures onto [`EngineError`] at the
//! storage boundary: corrupt or truncated files are `Format`, IO failures are
//! `Io`, our own errors come back unchanged.

use std::sync::Arc;

use datafusion::arrow::error::ArrowError;
use datafusion::error::DataFusionError;
use oxidelake_core::EngineError;
use parquet::errors::ParquetError;

/// Classifies a DataFusion error raised while reading or planning over files.
///
/// DataFusion rarely hands a scan's failure back bare. A repartitioned scan
/// fans one error out to every consumer as `Shared`, planning wraps errors in
/// `Diagnostic`, and some paths collect several into a `Collection` whose
/// first entry is the one DataFusion itself reports. All three are looked
/// through here, so a corrupt page found mid-scan is a `Format` error like a
/// corrupt footer found at registration (#118).
pub fn classify_error(err: DataFusionError) -> EngineError {
    match err {
        // Our own refusals (the Parquet scan guard's) travel through the
        // parquet reader as `External`; they come back as themselves.
        DataFusionError::ParquetError(e) => match *e {
            ParquetError::External(inner) => match inner.downcast::<EngineError>() {
                Ok(engine) => *engine,
                Err(inner) => {
                    EngineError::format(format!("parquet: {}", ParquetError::External(inner)))
                }
            },
            other => EngineError::format(format!("parquet: {other}")),
        },
        DataFusionError::ArrowError(e, context) => match *e {
            ArrowError::IoError(_, io) => EngineError::Io(io),
            other => {
                let ctx = context
                    .filter(|c| !c.is_empty())
                    .map(|c| format!(" ({c})"))
                    .unwrap_or_default();
                EngineError::format(format!("arrow: {other}{ctx}"))
            }
        },
        DataFusionError::IoError(io) => EngineError::Io(io),
        DataFusionError::ObjectStore(e) => match *e {
            object_store::Error::NotFound { path, source } => EngineError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{path}: {source}"),
            )),
            // A store of ours refusing a read says why (BoundedReadStore).
            object_store::Error::Generic { source, .. } if source.is::<EngineError>() => {
                match source.downcast::<EngineError>() {
                    Ok(engine) => *engine,
                    Err(source) => EngineError::Io(std::io::Error::other(source.to_string())),
                }
            }
            other => EngineError::Io(std::io::Error::other(other.to_string())),
        },
        DataFusionError::Context(_, inner) | DataFusionError::Diagnostic(_, inner) => {
            classify_error(*inner)
        }
        DataFusionError::Collection(errors) => {
            let mut errors = errors.into_iter();
            match errors.next() {
                Some(first) => classify_error(first),
                None => EngineError::DataFusion(DataFusionError::Collection(Vec::new())),
            }
        }
        DataFusionError::Shared(shared) => match Arc::try_unwrap(shared) {
            Ok(inner) => classify_error(inner),
            // Still held by another consumer of the same scan: the error
            // cannot be moved out, so its class is rebuilt from a borrow,
            // and anything without one stays as DataFusion delivered it.
            Err(shared) => classify_shared(&shared)
                .unwrap_or(EngineError::DataFusion(DataFusionError::Shared(shared))),
        },
        DataFusionError::External(e) => match e.downcast::<EngineError>() {
            Ok(engine) => *engine,
            Err(e) => EngineError::DataFusion(DataFusionError::External(e)),
        },
        other => EngineError::DataFusion(other),
    }
}

/// The `Format`/`Io` class of an error that can only be borrowed, rebuilt
/// with the same message (and the same `io::ErrorKind`). `None` for errors
/// that have no storage class; those are reported unchanged.
fn classify_shared(err: &DataFusionError) -> Option<EngineError> {
    match err {
        DataFusionError::ParquetError(e) => Some(match e.as_ref() {
            ParquetError::External(inner) => match inner.downcast_ref::<EngineError>() {
                Some(engine) => copy_class(engine)?,
                None => EngineError::format(format!("parquet: {e}")),
            },
            other => EngineError::format(format!("parquet: {other}")),
        }),
        DataFusionError::ArrowError(e, context) => Some(match e.as_ref() {
            ArrowError::IoError(_, io) => {
                EngineError::Io(std::io::Error::new(io.kind(), io.to_string()))
            }
            other => {
                let ctx = context
                    .as_ref()
                    .filter(|c| !c.is_empty())
                    .map(|c| format!(" ({c})"))
                    .unwrap_or_default();
                EngineError::format(format!("arrow: {other}{ctx}"))
            }
        }),
        DataFusionError::IoError(io) => Some(EngineError::Io(std::io::Error::new(
            io.kind(),
            io.to_string(),
        ))),
        DataFusionError::ObjectStore(e) => match e.as_ref() {
            object_store::Error::NotFound { path, source } => Some(EngineError::Io(
                std::io::Error::new(std::io::ErrorKind::NotFound, format!("{path}: {source}")),
            )),
            object_store::Error::Generic { source, .. } if source.is::<EngineError>() => {
                source.downcast_ref::<EngineError>().and_then(copy_class)
            }
            other => Some(EngineError::Io(std::io::Error::other(other.to_string()))),
        },
        DataFusionError::Context(_, inner) | DataFusionError::Diagnostic(_, inner) => {
            classify_shared(inner)
        }
        DataFusionError::Collection(errors) => errors.first().and_then(classify_shared),
        DataFusionError::Shared(inner) => classify_shared(inner),
        DataFusionError::External(e) => e.downcast_ref::<EngineError>().and_then(copy_class),
        _ => None,
    }
}

/// A copy of a borrowed `Format` or `Io` error; `None` for other classes.
fn copy_class(err: &EngineError) -> Option<EngineError> {
    match err {
        EngineError::Format(m) => Some(EngineError::Format(m.clone())),
        EngineError::Io(io) => Some(EngineError::Io(std::io::Error::new(
            io.kind(),
            io.to_string(),
        ))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use datafusion::common::Diagnostic;

    use super::*;

    fn parquet_error() -> DataFusionError {
        DataFusionError::ParquetError(Box::new(parquet::errors::ParquetError::General(
            "Unexpected struct field type 15".into(),
        )))
    }

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

    /// The three wrappers a scan's error arrives in (#118). Before, each fell
    /// through to `EngineError::DataFusion` however corrupt the file was.
    #[test]
    fn wrapped_scan_errors_keep_their_class() {
        let shared = DataFusionError::Shared(Arc::new(parquet_error()));
        let e = classify_error(shared);
        assert!(
            matches!(&e, EngineError::Format(m) if m.contains("struct field type 15")),
            "{e:?}"
        );

        let diagnostic = DataFusionError::Diagnostic(
            Box::new(Diagnostic::new_error("while scanning t", None)),
            Box::new(parquet_error()),
        );
        assert!(matches!(classify_error(diagnostic), EngineError::Format(_)));

        let collection = DataFusionError::Collection(vec![
            DataFusionError::IoError(std::io::Error::other("disk")),
            parquet_error(),
        ]);
        assert!(
            matches!(classify_error(collection), EngineError::Io(_)),
            "the first error of a collection is the one DataFusion reports"
        );

        // Nested: a shared error inside a context inside a diagnostic.
        let nested = DataFusionError::Diagnostic(
            Box::new(Diagnostic::new_error("d", None)),
            Box::new(DataFusionError::Context(
                "c".into(),
                Box::new(DataFusionError::Shared(Arc::new(parquet_error()))),
            )),
        );
        assert!(matches!(classify_error(nested), EngineError::Format(_)));
    }

    /// The scan guard's and the bounded store's refusals carry an
    /// `EngineError` through parquet and object_store; it comes back as
    /// itself, not wrapped in "parquet: External: ...".
    #[test]
    fn our_own_refusals_come_back_as_themselves() {
        let refusal = || EngineError::format("parquet file t.parquet: row group 0 column k: bad");
        let via_parquet =
            DataFusionError::ParquetError(Box::new(ParquetError::External(Box::new(refusal()))));
        assert!(matches!(
            classify_error(via_parquet),
            EngineError::Format(m) if m.starts_with("parquet file t.parquet")
        ));
        let via_store = DataFusionError::ObjectStore(Box::new(object_store::Error::Generic {
            store: "BoundedReadStore",
            source: Box::new(refusal()),
        }));
        assert!(matches!(classify_error(via_store), EngineError::Format(_)));
        let shared = Arc::new(DataFusionError::ObjectStore(Box::new(
            object_store::Error::Generic {
                store: "BoundedReadStore",
                source: Box::new(refusal()),
            },
        )));
        let _other_consumer = Arc::clone(&shared);
        assert!(matches!(
            classify_error(DataFusionError::Shared(shared)),
            EngineError::Format(_)
        ));
        // Any other store error is still IO.
        let other = DataFusionError::ObjectStore(Box::new(object_store::Error::Generic {
            store: "LocalFileSystem",
            source: "disk".into(),
        }));
        assert!(matches!(classify_error(other), EngineError::Io(_)));
    }

    /// A repartitioned scan hands the *same* `Arc` to every consumer, so the
    /// one being classified usually cannot be unwrapped. Its class is still
    /// found, with the original message and IO error kind.
    #[test]
    fn a_shared_error_still_held_elsewhere_keeps_its_class() {
        let inner = Arc::new(parquet_error());
        let _other_consumer = Arc::clone(&inner);
        let e = classify_error(DataFusionError::Shared(inner));
        assert!(
            matches!(&e, EngineError::Format(m) if m.contains("struct field type 15")),
            "{e:?}"
        );

        let io = Arc::new(DataFusionError::IoError(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "spill dir",
        )));
        let _other_consumer = Arc::clone(&io);
        match classify_error(DataFusionError::Shared(io)) {
            EngineError::Io(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied);
                assert!(e.to_string().contains("spill dir"));
            }
            other => panic!("expected Io, got {other:?}"),
        }

        let own = Arc::new(DataFusionError::External(Box::new(EngineError::format(
            "bad block",
        ))));
        let _other_consumer = Arc::clone(&own);
        assert!(matches!(
            classify_error(DataFusionError::Shared(own)),
            EngineError::Format(m) if m == "bad block"
        ));

        // No storage class: delivered unchanged rather than guessed at.
        let plan = Arc::new(DataFusionError::Plan("no such column".into()));
        let _other_consumer = Arc::clone(&plan);
        assert!(matches!(
            classify_error(DataFusionError::Shared(plan)),
            EngineError::DataFusion(DataFusionError::Shared(_))
        ));
    }
}
