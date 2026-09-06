//! Operator parameter types and the bounded v1 GPU type coverage.
//!
//! These types are deliberately independent of DataFusion's `PhysicalExpr`
//! tree: the planner lowers eligible DataFusion expressions into them, the
//! codec serializes them with `postcard`, and every backend executes them.

use datafusion::arrow::datatypes::DataType;
use serde::{Deserialize, Serialize};

/// Comparison operators supported by the fused filter kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Comparison {
    /// `=`
    Eq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
}

impl Comparison {
    /// SQL spelling, used in `EXPLAIN` output.
    pub const fn symbol(self) -> &'static str {
        match self {
            Comparison::Eq => "=",
            Comparison::Lt => "<",
            Comparison::LtEq => "<=",
            Comparison::Gt => ">",
            Comparison::GtEq => ">=",
        }
    }
}

/// A literal a column is compared against.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    /// 64-bit signed integer.
    Int64(i64),
    /// 64-bit float.
    Float64(f64),
}

impl Literal {
    /// The Arrow type this literal compares against.
    pub const fn data_type(&self) -> DataType {
        match self {
            Literal::Int64(_) => DataType::Int64,
            Literal::Float64(_) => DataType::Float64,
        }
    }
}

/// A filter predicate in the bounded v1 grammar: column-vs-literal comparisons
/// combined with `AND`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Predicate {
    /// `column <op> literal`
    Compare {
        /// Index of the column in the input schema.
        column: usize,
        /// The comparison operator.
        op: Comparison,
        /// The literal operand.
        literal: Literal,
    },
    /// Logical conjunction.
    And(Box<Predicate>, Box<Predicate>),
}

impl Predicate {
    /// Builds a comparison predicate.
    pub const fn compare(column: usize, op: Comparison, literal: Literal) -> Self {
        Self::Compare {
            column,
            op,
            literal,
        }
    }

    /// Conjoins two predicates.
    pub fn and(left: Predicate, right: Predicate) -> Self {
        Self::And(Box::new(left), Box::new(right))
    }

    /// Every column index referenced, in evaluation order (duplicates kept).
    pub fn columns(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.collect_columns(&mut out);
        out
    }

    fn collect_columns(&self, out: &mut Vec<usize>) {
        match self {
            Predicate::Compare { column, .. } => out.push(*column),
            Predicate::And(l, r) => {
                l.collect_columns(out);
                r.collect_columns(out);
            }
        }
    }

    /// Number of comparison leaves.
    pub fn leaf_count(&self) -> usize {
        match self {
            Predicate::Compare { .. } => 1,
            Predicate::And(l, r) => l.leaf_count() + r.leaf_count(),
        }
    }
}

/// Aggregate functions supported by the grouped-aggregation kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregateFunction {
    /// `SUM`
    Sum,
    /// `COUNT`
    Count,
    /// `MIN`
    Min,
    /// `MAX`
    Max,
}

impl AggregateFunction {
    /// SQL spelling, used in `EXPLAIN` output and output column names.
    pub const fn name(self) -> &'static str {
        match self {
            AggregateFunction::Sum => "SUM",
            AggregateFunction::Count => "COUNT",
            AggregateFunction::Min => "MIN",
            AggregateFunction::Max => "MAX",
        }
    }
}

/// A grouped aggregation over one `Int64` key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateSpec {
    /// Index of the `Int64` group-by column.
    pub group_by: usize,
    /// `(function, input column index)` pairs, in output order.
    pub aggregates: Vec<(AggregateFunction, usize)>,
}

/// Distance metric for the vector kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance.
    L2,
    /// Cosine distance (`1 - cosine similarity`).
    Cosine,
}

impl DistanceMetric {
    /// Lowercase name used in `EXPLAIN` output and SQL function names.
    pub const fn name(self) -> &'static str {
        match self {
            DistanceMetric::L2 => "l2",
            DistanceMetric::Cosine => "cosine",
        }
    }
}

/// `true` for the scalar types the v1 GPU kernels accept as filter, join-key
/// and aggregate inputs.
pub const fn gpu_eligible_scalar(dt: &DataType) -> bool {
    matches!(dt, DataType::Int64 | DataType::Float64)
}

/// For a `FixedSizeList<Float32, n>` column returns `Some(n)`; otherwise `None`.
pub fn vector_dimension(dt: &DataType) -> Option<usize> {
    match dt {
        DataType::FixedSizeList(field, n) if *field.data_type() == DataType::Float32 && *n > 0 => {
            usize::try_from(*n).ok()
        }
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::datatypes::Field;

    use super::*;

    #[test]
    fn predicate_collects_columns_in_order() {
        let p = Predicate::and(
            Predicate::compare(2, Comparison::Gt, Literal::Int64(1)),
            Predicate::and(
                Predicate::compare(0, Comparison::LtEq, Literal::Float64(0.5)),
                Predicate::compare(2, Comparison::Eq, Literal::Int64(9)),
            ),
        );
        assert_eq!(p.columns(), vec![2, 0, 2]);
        assert_eq!(p.leaf_count(), 3);
    }

    #[test]
    fn coverage_rules() {
        assert!(gpu_eligible_scalar(&DataType::Int64));
        assert!(gpu_eligible_scalar(&DataType::Float64));
        assert!(!gpu_eligible_scalar(&DataType::Int32));
        assert!(!gpu_eligible_scalar(&DataType::Utf8));

        let vec3 =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 3);
        assert_eq!(vector_dimension(&vec3), Some(3));
        let f64s =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, false)), 3);
        assert_eq!(vector_dimension(&f64s), None);
        assert_eq!(vector_dimension(&DataType::Int64), None);
    }

    #[test]
    fn names_are_stable_for_explain_output() {
        assert_eq!(AggregateFunction::Sum.name(), "SUM");
        assert_eq!(AggregateFunction::Count.name(), "COUNT");
        assert_eq!(DistanceMetric::Cosine.name(), "cosine");
        assert_eq!(Comparison::GtEq.symbol(), ">=");
        assert_eq!(Literal::Float64(1.5).data_type(), DataType::Float64);
    }
}
