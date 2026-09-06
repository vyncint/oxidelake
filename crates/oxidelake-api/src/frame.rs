//! [`OxideFrame`]: the fluent DataFrame API over a session's DataFusion
//! `DataFrame`, plus [`OxideSessionExt`] — the frame-producing entry points on
//! [`OxideSession`].
//!
//! Every verb builds standard DataFusion logical plans, so the session's
//! placement rule decides hardware exactly as it does for SQL:
//! [`OxideFrame::vector_distance`] plans the same projection as
//! `SELECT *, l2_distance(emb, …) AS d` and lowers to `GpuVectorDistanceExec`
//! on a GPU target.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::dataframe::DataFrame;
use datafusion::logical_expr::JoinType;
use datafusion::physical_plan::displayable;
use datafusion::prelude::{Expr, ParquetReadOptions, lit};
use oxidelake_core::EngineError;
use oxidelake_core::params::DistanceMetric;
use oxidelake_runtime::OxideSession;
use oxidelake_runtime::udf::{cosine_distance_udf, l2_distance_udf, query_literal};

/// A lazily built query over one session, collected with [`Self::collect`].
#[derive(Debug, Clone)]
pub struct OxideFrame {
    inner: DataFrame,
}

impl OxideFrame {
    /// Wraps a DataFusion [`DataFrame`].
    pub fn new(inner: DataFrame) -> Self {
        Self { inner }
    }

    /// The underlying DataFusion [`DataFrame`], for verbs this API does not wrap.
    pub fn into_inner(self) -> DataFrame {
        self.inner
    }

    /// The frame's Arrow schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::new(self.inner.schema().as_arrow().clone())
    }

    /// Keeps rows matching `predicate` (e.g. `col("k").gt_eq(lit(2))`).
    pub fn filter(self, predicate: Expr) -> Result<Self, EngineError> {
        Ok(Self::new(self.inner.filter(predicate)?))
    }

    /// Keeps the named columns, in order.
    pub fn select(self, columns: &[&str]) -> Result<Self, EngineError> {
        Ok(Self::new(self.inner.select_columns(columns)?))
    }

    /// Renames the frame's table qualifier (required to self-join a table).
    pub fn alias(self, name: &str) -> Result<Self, EngineError> {
        Ok(Self::new(self.inner.alias(name)?))
    }

    /// Inner-joins `right` on `left_key = right_key`.
    pub fn join(
        self,
        right: OxideFrame,
        left_key: &str,
        right_key: &str,
    ) -> Result<Self, EngineError> {
        Ok(Self::new(self.inner.join(
            right.inner,
            JoinType::Inner,
            &[left_key],
            &[right_key],
            None,
        )?))
    }

    /// Groups by `group_by` and computes `aggregates`
    /// (e.g. `aggregate(vec![col("k")], vec![sum(col("v"))])`).
    pub fn aggregate(
        self,
        group_by: Vec<Expr>,
        aggregates: Vec<Expr>,
    ) -> Result<Self, EngineError> {
        Ok(Self::new(self.inner.aggregate(group_by, aggregates)?))
    }

    /// Sorts by `expr` (e.g. `col("d").sort(true, false)` for ascending).
    pub fn sort(self, exprs: Vec<datafusion::logical_expr::SortExpr>) -> Result<Self, EngineError> {
        Ok(Self::new(self.inner.sort(exprs)?))
    }

    /// Keeps at most `n` rows.
    pub fn limit(self, n: usize) -> Result<Self, EngineError> {
        Ok(Self::new(self.inner.limit(0, Some(n))?))
    }

    /// Appends `output` — the `metric` distance between the vector column
    /// `column` (`FixedSizeList<Float32>`) and `query` — as a nullable
    /// `Float32` column. Plans the `l2_distance` / `cosine_distance` UDF, so
    /// a GPU-target session lowers it to `GpuVectorDistanceExec`.
    pub fn vector_distance(
        self,
        column: &str,
        query: &[f32],
        metric: DistanceMetric,
        output: &str,
    ) -> Result<Self, EngineError> {
        let udf = match metric {
            DistanceMetric::L2 => l2_distance_udf(),
            DistanceMetric::Cosine => cosine_distance_udf(),
        };
        let call = udf.call(vec![
            datafusion::prelude::col(column),
            lit(query_literal(query)),
        ]);
        Ok(Self::new(self.inner.with_column(output, call)?))
    }

    /// The indented physical plan, with placement tags on `Gpu*Exec` nodes.
    pub async fn explain(&self) -> Result<String, EngineError> {
        let plan = self.inner.clone().create_physical_plan().await?;
        Ok(displayable(plan.as_ref()).indent(true).to_string())
    }

    /// Executes the frame and returns every batch.
    pub async fn collect(self) -> Result<Vec<RecordBatch>, EngineError> {
        Ok(self.inner.collect().await?)
    }

    /// Executes the frame and prints it as a table.
    pub async fn show(self) -> Result<(), EngineError> {
        Ok(self.inner.show().await?)
    }
}

impl From<DataFrame> for OxideFrame {
    fn from(inner: DataFrame) -> Self {
        Self::new(inner)
    }
}

/// Frame-producing entry points on [`OxideSession`].
pub trait OxideSessionExt {
    /// Reads a Parquet file or directory as a frame.
    fn read_parquet(
        &self,
        path: &str,
    ) -> impl Future<Output = Result<OxideFrame, EngineError>> + Send;

    /// A frame over a registered table.
    fn table(&self, name: &str) -> impl Future<Output = Result<OxideFrame, EngineError>> + Send;

    /// Plans a SQL statement as a frame.
    fn sql_frame(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<OxideFrame, EngineError>> + Send;
}

impl OxideSessionExt for OxideSession {
    async fn read_parquet(&self, path: &str) -> Result<OxideFrame, EngineError> {
        Ok(OxideFrame::new(
            self.ctx()
                .read_parquet(path, ParquetReadOptions::default())
                .await?,
        ))
    }

    async fn table(&self, name: &str) -> Result<OxideFrame, EngineError> {
        Ok(OxideFrame::new(self.ctx().table(name).await?))
    }

    async fn sql_frame(&self, query: &str) -> Result<OxideFrame, EngineError> {
        Ok(OxideFrame::new(self.sql(query).await?))
    }
}
