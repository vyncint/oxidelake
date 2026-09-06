//! Parquet writer configuration tuned for OxideLake datasets.

use std::str::FromStr;

use oxidelake_core::EngineError;
use parquet::basic::{Compression as PqCompression, ZstdLevel};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;

/// Compression codecs exposed on the command line (`--compression`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// ZSTD level 1: good ratio, fast decode (the default).
    #[default]
    Zstd,
    /// LZ4 raw: fastest decode.
    Lz4,
    /// No compression: cheapest CPU path into device memory.
    None,
}

impl Compression {
    /// Canonical lowercase name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Compression::Zstd => "zstd",
            Compression::Lz4 => "lz4",
            Compression::None => "none",
        }
    }

    fn to_parquet(self) -> Result<PqCompression, EngineError> {
        Ok(match self {
            Compression::Zstd => PqCompression::ZSTD(
                ZstdLevel::try_new(1).map_err(|e| EngineError::plan(e.to_string()))?,
            ),
            Compression::Lz4 => PqCompression::LZ4_RAW,
            Compression::None => PqCompression::UNCOMPRESSED,
        })
    }
}

impl FromStr for Compression {
    type Err = EngineError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "zstd" => Ok(Compression::Zstd),
            "lz4" => Ok(Compression::Lz4),
            "none" | "uncompressed" => Ok(Compression::None),
            other => Err(EngineError::plan(format!(
                "unknown compression '{other}'; expected none, lz4 or zstd"
            ))),
        }
    }
}

/// Writer settings for datasets OxideLake produces.
///
/// Row groups are sized to the engine's batch size so one row group maps to a
/// handful of device batches; page statistics and Bloom filters on key columns
/// are what DataFusion prunes on at read time.
#[derive(Debug, Clone, PartialEq)]
pub struct ParquetWriteOptions {
    /// Maximum rows per row group.
    pub row_group_rows: usize,
    /// Compression codec.
    pub compression: Compression,
    /// Dictionary-encode columns when beneficial.
    pub dictionary: bool,
    /// Columns that get a split-block Bloom filter (join/filter keys).
    pub bloom_filter_columns: Vec<String>,
    /// Bloom filter false-positive probability.
    pub bloom_filter_fpp: f64,
}

impl Default for ParquetWriteOptions {
    fn default() -> Self {
        Self {
            row_group_rows: 262_144,
            compression: Compression::default(),
            dictionary: true,
            bloom_filter_columns: Vec::new(),
            bloom_filter_fpp: 0.01,
        }
    }
}

impl ParquetWriteOptions {
    /// Adds a Bloom-filtered column.
    pub fn with_bloom_filter(mut self, column: impl Into<String>) -> Self {
        self.bloom_filter_columns.push(column.into());
        self
    }

    /// Materializes the `parquet` crate's [`WriterProperties`].
    pub fn writer_properties(&self) -> Result<WriterProperties, EngineError> {
        if self.row_group_rows == 0 {
            return Err(EngineError::plan("row_group_rows must be positive"));
        }
        if !(0.0 < self.bloom_filter_fpp && self.bloom_filter_fpp < 1.0) {
            return Err(EngineError::plan("bloom_filter_fpp must be in (0, 1)"));
        }
        let mut builder = WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_max_row_group_row_count(Some(self.row_group_rows))
            .set_compression(self.compression.to_parquet()?)
            .set_dictionary_enabled(self.dictionary)
            .set_statistics_enabled(EnabledStatistics::Page)
            .set_bloom_filter_enabled(false);
        for column in &self.bloom_filter_columns {
            let path = ColumnPath::from(column.as_str());
            builder = builder
                .set_column_bloom_filter_enabled(path.clone(), true)
                .set_column_bloom_filter_fpp(path, self.bloom_filter_fpp);
        }
        Ok(builder.build())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_compression_names() {
        assert_eq!("ZSTD".parse::<Compression>().unwrap(), Compression::Zstd);
        assert_eq!("lz4".parse::<Compression>().unwrap(), Compression::Lz4);
        assert_eq!("none".parse::<Compression>().unwrap(), Compression::None);
        assert!("snappy".parse::<Compression>().is_err());
    }

    #[test]
    fn builds_properties_with_bloom_filters() {
        let opts = ParquetWriteOptions::default().with_bloom_filter("k");
        let props = opts.writer_properties().unwrap();
        assert_eq!(props.max_row_group_row_count(), Some(262_144));
        let k = ColumnPath::from("k");
        assert!(props.bloom_filter_properties(&k).is_some());
        let v = ColumnPath::from("v");
        assert!(props.bloom_filter_properties(&v).is_none());
        assert_eq!(props.statistics_enabled(&v), EnabledStatistics::Page);
    }

    #[test]
    fn rejects_invalid_settings() {
        let opts = ParquetWriteOptions {
            row_group_rows: 0,
            ..Default::default()
        };
        assert!(opts.writer_properties().is_err());
        let opts = ParquetWriteOptions {
            bloom_filter_fpp: 1.5,
            ..Default::default()
        };
        assert!(opts.writer_properties().is_err());
    }
}
