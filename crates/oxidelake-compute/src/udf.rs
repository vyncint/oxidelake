//! `l2_distance` / `cosine_distance` scalar UDFs (docs/SPEC.md §2.8).
//!
//! The SQL and DataFrame surface of the vector-distance operator. Both take a
//! `FixedSizeList<Float32>` vector and a query vector (any list of numbers —
//! coercion casts it to `FixedSizeList<Float32>` with the column's dimension)
//! and return a nullable `Float32` distance. The scalar implementation calls
//! the same reference arithmetic as the CPU kernel, so a projection the
//! placement rule lowers to [`crate::GpuVectorDistanceExec`] computes exactly
//! what the un-lowered projection would.

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use oxidelake_core::params::DistanceMetric;
use oxidelake_device::cpu::kernels;

/// Name of the L2 (Euclidean) distance UDF.
pub const L2_DISTANCE: &str = "l2_distance";
/// Name of the cosine distance (`1 - cos(θ)`) UDF.
pub const COSINE_DISTANCE: &str = "cosine_distance";

/// The UDF name for `metric`.
pub const fn distance_udf_name(metric: DistanceMetric) -> &'static str {
    match metric {
        DistanceMetric::L2 => L2_DISTANCE,
        DistanceMetric::Cosine => COSINE_DISTANCE,
    }
}

/// The metric a UDF name stands for, if it is one of ours.
pub fn distance_metric_for(name: &str) -> Option<DistanceMetric> {
    match name {
        L2_DISTANCE => Some(DistanceMetric::L2),
        COSINE_DISTANCE => Some(DistanceMetric::Cosine),
        _ => None,
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct DistanceUdf {
    metric: DistanceMetric,
    signature: Signature,
}

impl DistanceUdf {
    fn new(metric: DistanceMetric) -> Self {
        Self {
            metric,
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

fn list_element(dt: &DataType) -> Option<&FieldRef> {
    match dt {
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => Some(f),
        _ => None,
    }
}

fn fixed_dimension(dt: &DataType) -> Option<i32> {
    match dt {
        DataType::FixedSizeList(_, n) => Some(*n),
        _ => None,
    }
}

impl ScalarUDFImpl for DistanceUdf {
    fn name(&self) -> &str {
        distance_udf_name(self.metric)
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    /// Both arguments must be lists of numbers and at least one must be a
    /// `FixedSizeList` so the dimension is known. An argument that is already
    /// `FixedSizeList<Float32>` keeps its exact type (no cast on the column);
    /// anything else is cast to `FixedSizeList<Float32>` of that dimension.
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let name = self.name();
        if arg_types.len() != 2 {
            return Err(DataFusionError::Plan(format!(
                "{name} takes 2 arguments, got {}",
                arg_types.len()
            )));
        }
        for dt in arg_types {
            let element = list_element(dt).ok_or_else(|| {
                DataFusionError::Plan(format!("{name}: expected a list of numbers, got {dt:?}"))
            })?;
            if !element.data_type().is_numeric() {
                return Err(DataFusionError::Plan(format!(
                    "{name}: vector elements must be numeric, got {:?}",
                    element.data_type()
                )));
            }
        }
        let dims: Vec<i32> = arg_types.iter().filter_map(fixed_dimension).collect();
        let Some(&dim) = dims.first() else {
            return Err(DataFusionError::Plan(format!(
                "{name}: at least one argument must be a FixedSizeList so the \
                 dimension is known"
            )));
        };
        if dims.iter().any(|&d| d != dim) {
            return Err(DataFusionError::Plan(format!(
                "{name}: vectors have different dimensions: {dims:?}"
            )));
        }
        let target =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim);
        Ok(arg_types
            .iter()
            .map(|dt| match dt {
                DataType::FixedSizeList(f, _) if *f.data_type() == DataType::Float32 => dt.clone(),
                _ => target.clone(),
            })
            .collect())
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let name = self.name();
        let [a, b] = args.args.as_slice() else {
            return Err(DataFusionError::Execution(format!(
                "{name} takes 2 arguments, got {}",
                args.args.len()
            )));
        };
        let (a, a_scalar) = fixed_size_list(name, a)?;
        let (b, b_scalar) = fixed_size_list(name, b)?;
        let (a_vals, a_dim) = float_values(name, &a)?;
        let (b_vals, b_dim) = float_values(name, &b)?;
        if a_dim != b_dim {
            return Err(DataFusionError::Execution(format!(
                "{name}: vectors have {a_dim} and {b_dim} dimensions"
            )));
        }
        let rows = args.number_rows;
        let distance = |row: usize| -> Option<f32> {
            let (ra, rb) = (
                if a_scalar { 0 } else { row },
                if b_scalar { 0 } else { row },
            );
            if a.is_null(ra) || b.is_null(rb) {
                return None;
            }
            let sa = slice(a_vals, &a, ra, a_dim);
            let sb = slice(b_vals, &b, rb, b_dim);
            Some(match self.metric {
                DistanceMetric::L2 => kernels::l2(sa, sb),
                DistanceMetric::Cosine => kernels::cosine(sa, sb),
            })
        };
        if a_scalar && b_scalar {
            return Ok(ColumnarValue::Scalar(ScalarValue::Float32(distance(0))));
        }
        let out: Float32Array = (0..rows).map(distance).collect();
        Ok(ColumnarValue::Array(Arc::new(out)))
    }
}

fn fixed_size_list(name: &str, value: &ColumnarValue) -> Result<(FixedSizeListArray, bool)> {
    match value {
        ColumnarValue::Array(array) => Ok((
            array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "{name}: expected FixedSizeList, got {:?}",
                        array.data_type()
                    ))
                })?
                .clone(),
            false,
        )),
        ColumnarValue::Scalar(ScalarValue::FixedSizeList(array)) => {
            Ok((array.as_ref().clone(), true))
        }
        ColumnarValue::Scalar(other) => Err(DataFusionError::Execution(format!(
            "{name}: expected FixedSizeList, got {:?}",
            other.data_type()
        ))),
    }
}

fn float_values<'a>(name: &str, list: &'a FixedSizeListArray) -> Result<(&'a [f32], usize)> {
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            DataFusionError::Execution(format!(
                "{name}: vector elements are {:?}, expected Float32",
                list.value_type()
            ))
        })?;
    let dim = usize::try_from(list.value_length())
        .map_err(|_| DataFusionError::Execution(format!("{name}: negative list size")))?;
    Ok((values.values(), dim))
}

fn slice<'a>(values: &'a [f32], list: &FixedSizeListArray, row: usize, dim: usize) -> &'a [f32] {
    let start = usize::try_from(list.value_offset(row)).unwrap_or(0);
    &values[start..start + dim]
}

/// The `l2_distance` scalar UDF.
pub fn l2_distance_udf() -> ScalarUDF {
    ScalarUDF::from(DistanceUdf::new(DistanceMetric::L2))
}

/// The `cosine_distance` scalar UDF.
pub fn cosine_distance_udf() -> ScalarUDF {
    ScalarUDF::from(DistanceUdf::new(DistanceMetric::Cosine))
}

/// Every OxideLake SQL UDF — registered on embedded sessions, cluster clients,
/// the scheduler's session builder and every executor's function registry.
///
/// `predict` joins the list only when the `predict` feature is on. That makes
/// it a *cluster-wide* decision rather than a per-node one: an executor built
/// without the feature would plan a query the client planned with it and fail
/// at execution. Building the whole cluster the same way is the same rule the
/// GPU features already follow.
pub fn oxide_udfs() -> Vec<Arc<ScalarUDF>> {
    // Chained rather than `let mut` + conditional push: the `mut` is unused
    // when the feature is off, and this crate lints with `-D warnings` in
    // both configurations.
    let always = [Arc::new(l2_distance_udf()), Arc::new(cosine_distance_udf())];
    #[cfg(feature = "predict")]
    let optional = vec![Arc::new(crate::predict::predict_udf())];
    #[cfg(not(feature = "predict"))]
    let optional: Vec<Arc<ScalarUDF>> = Vec::new();
    always.into_iter().chain(optional).collect()
}

/// The query vector as the `FixedSizeList<Float32>` literal our planner rule
/// recognizes ([`ScalarValue::FixedSizeList`], no nulls).
pub fn query_literal(query: &[f32]) -> ScalarValue {
    let values = Float32Array::from(query.to_vec());
    ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        i32::try_from(query.len()).unwrap_or(i32::MAX),
        Arc::new(values),
        None,
    )))
}

/// Extracts the query vector from a [`ScalarValue::FixedSizeList`] literal:
/// one non-null row of non-null `Float32`s, as the placement rule requires.
pub fn literal_query(value: &ScalarValue) -> Option<Vec<f32>> {
    let ScalarValue::FixedSizeList(list) = value else {
        return None;
    };
    if list.len() != 1 || list.is_null(0) {
        return None;
    }
    let values = list.values().as_any().downcast_ref::<Float32Array>()?;
    if values.null_count() != 0 {
        return None;
    }
    let dim = usize::try_from(list.value_length()).ok()?;
    let start = usize::try_from(list.value_offset(0)).ok()?;
    Some(values.values()[start..start + dim].to_vec())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use arrow::array::RecordBatch;
    use arrow::datatypes::Schema;
    use datafusion::prelude::SessionContext;

    use super::*;

    fn vectors(rows: &[Option<&[f32]>], dim: i32) -> FixedSizeListArray {
        let mut values = Vec::new();
        let mut valid = Vec::new();
        for row in rows {
            match row {
                Some(v) => {
                    values.extend_from_slice(v);
                    valid.push(true);
                }
                None => {
                    values.extend(std::iter::repeat_n(0.0, dim as usize));
                    valid.push(false);
                }
            }
        }
        FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            dim,
            Arc::new(Float32Array::from(values)),
            Some(valid.into()),
        )
    }

    async fn distances(ctx: &SessionContext, sql: &str) -> Vec<Option<f32>> {
        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let mut out = Vec::new();
        for batch in &batches {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            out.extend((0..col.len()).map(|i| (!col.is_null(i)).then(|| col.value(i))));
        }
        out
    }

    fn session_with_vectors() -> SessionContext {
        let ctx = SessionContext::new();
        for udf in oxide_udfs() {
            ctx.register_udf(udf.as_ref().clone());
        }
        let emb = vectors(
            &[
                Some(&[1.0, 0.0]),
                Some(&[0.0, 3.0]),
                None,
                Some(&[2.0, 0.0]),
            ],
            2,
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "emb",
            emb.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(emb)]).unwrap();
        ctx.register_batch("t", batch).unwrap();
        ctx
    }

    #[tokio::test]
    async fn sql_l2_matches_reference_and_keeps_nulls() {
        let ctx = session_with_vectors();
        let got = distances(&ctx, "SELECT l2_distance(emb, [1.0, 0.0]) FROM t").await;
        assert_eq!(
            got,
            vec![
                Some(0.0),
                Some(kernels::l2(&[0.0, 3.0], &[1.0, 0.0])),
                None,
                Some(1.0)
            ]
        );
    }

    #[tokio::test]
    async fn sql_cosine_matches_reference_and_argument_order_is_symmetric() {
        let ctx = session_with_vectors();
        let got = distances(&ctx, "SELECT cosine_distance([0.0, 1.0], emb) FROM t").await;
        assert_eq!(
            got,
            vec![
                Some(kernels::cosine(&[0.0, 1.0], &[1.0, 0.0])),
                Some(0.0),
                None,
                Some(kernels::cosine(&[0.0, 1.0], &[2.0, 0.0])),
            ]
        );
    }

    /// A wrong-sized query vector and a call with no `FixedSizeList` argument
    /// are errors (at plan time or, for the cast of a wrong-length list, at
    /// execution time) — never a silent wrong answer.
    #[tokio::test]
    async fn dimension_mismatch_and_unknown_dimension_are_errors() {
        let ctx = session_with_vectors();
        for sql in [
            "SELECT l2_distance(emb, [1.0, 2.0, 3.0]) FROM t",
            "SELECT l2_distance([1.0], [1.0])",
        ] {
            let result = match ctx.sql(sql).await {
                Ok(df) => df.collect().await.map(|_| ()),
                Err(e) => Err(e),
            };
            assert!(result.is_err(), "{sql} should fail");
        }
    }

    #[test]
    fn query_literal_round_trips() {
        let q = [1.5f32, -2.0, 0.25];
        assert_eq!(literal_query(&query_literal(&q)).unwrap(), q);
        assert_eq!(literal_query(&ScalarValue::Float32(Some(1.0))), None);
    }
}
