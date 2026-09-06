//! `EXPLAIN` rendering helpers shared by the execs.

use arrow::datatypes::Schema;
use oxidelake_core::params::{AggregateSpec, Literal, Predicate};

fn column_name(schema: &Schema, index: usize) -> String {
    schema
        .fields()
        .get(index)
        .map_or_else(|| format!("col{index}"), |f| f.name().clone())
}

fn literal(l: &Literal) -> String {
    match l {
        Literal::Int64(v) => v.to_string(),
        Literal::Float64(v) => format!("{v:?}"),
    }
}

/// Renders a predicate against `schema`, e.g. `k >= 2 AND v < 4.0`.
pub fn predicate(p: &Predicate, schema: &Schema) -> String {
    match p {
        Predicate::Compare {
            column,
            op,
            literal: l,
        } => {
            format!(
                "{} {} {}",
                column_name(schema, *column),
                op.symbol(),
                literal(l)
            )
        }
        Predicate::And(l, r) => format!("{} AND {}", predicate(l, schema), predicate(r, schema)),
    }
}

/// Renders an aggregate spec, e.g. `group_by=[k], aggr=[SUM(v), COUNT(v)]`.
pub fn aggregate(spec: &AggregateSpec, schema: &Schema) -> String {
    let aggr: Vec<String> = spec
        .aggregates
        .iter()
        .map(|(f, c)| format!("{}({})", f.name(), column_name(schema, *c)))
        .collect();
    format!(
        "group_by=[{}], aggr=[{}]",
        column_name(schema, spec.group_by),
        aggr.join(", ")
    )
}

/// Renders projected column names.
pub fn projection(projection: &[usize], schema: &Schema) -> String {
    projection
        .iter()
        .map(|&i| column_name(schema, i))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field};
    use oxidelake_core::params::{AggregateFunction, Comparison};

    use super::*;

    #[test]
    fn renders_predicates_and_specs() {
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]);
        let p = Predicate::and(
            Predicate::compare(0, Comparison::GtEq, Literal::Int64(2)),
            Predicate::compare(1, Comparison::Lt, Literal::Float64(4.0)),
        );
        assert_eq!(predicate(&p, &schema), "k >= 2 AND v < 4.0");
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![(AggregateFunction::Sum, 1), (AggregateFunction::Count, 7)],
        };
        assert_eq!(
            aggregate(&spec, &schema),
            "group_by=[k], aggr=[SUM(v), COUNT(col7)]"
        );
        assert_eq!(projection(&[1, 0], &schema), "v, k");
    }
}
