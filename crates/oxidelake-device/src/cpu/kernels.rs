//! CPU reference implementations of the four v1 operators, over host
//! `RecordBatch`es. These define the semantics the GPU kernels must match:
//!
//! * filter: a comparison against `NULL` is `NULL`, and `NULL` predicates drop
//!   the row (SQL `WHERE` semantics);
//! * join: `NULL` keys never match;
//! * aggregate: `NULL` keys form their own group; `SUM/MIN/MAX` skip `NULL`s
//!   and are `NULL` for groups with no values; `COUNT` counts non-`NULL`s;
//!   output is sorted by key with the `NULL` group last;
//! * vector distance: a `NULL` vector yields a `NULL` distance; cosine distance
//!   is `1 - cos(θ)` and is `NaN` when either norm is zero.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
    RecordBatch, Scalar, UInt32Array,
};
use arrow::compute::kernels::cmp;
use arrow::compute::{and_kleene, filter_record_batch, take};
use arrow::datatypes::{DataType, Field, Schema};
use oxidelake_core::EngineError;
use oxidelake_core::params::{
    AggregateFunction, AggregateSpec, Comparison, DistanceMetric, Literal, Predicate,
};
use rayon::prelude::*;

fn column(batch: &RecordBatch, index: usize) -> Result<&ArrayRef, EngineError> {
    batch.columns().get(index).ok_or_else(|| {
        EngineError::plan(format!(
            "column index {index} out of range for a batch with {} columns",
            batch.num_columns()
        ))
    })
}

fn int64_column<'a>(
    batch: &'a RecordBatch,
    index: usize,
    role: &str,
) -> Result<&'a Int64Array, EngineError> {
    let col = column(batch, index)?;
    col.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        EngineError::unsupported(
            "cpu.int64_column",
            format!(
                "{role} column {index} has type {:?}; v1 supports Int64",
                col.data_type()
            ),
        )
    })
}

/// Evaluates a predicate to a boolean mask (nulls where any operand is null).
pub fn evaluate_predicate(
    batch: &RecordBatch,
    predicate: &Predicate,
) -> Result<BooleanArray, EngineError> {
    match predicate {
        Predicate::Compare {
            column: idx,
            op,
            literal,
        } => {
            let col = column(batch, *idx)?;
            if *col.data_type() != literal.data_type() {
                return Err(EngineError::unsupported(
                    "cpu.filter",
                    format!(
                        "column {idx} has type {:?} but the literal is {:?}",
                        col.data_type(),
                        literal.data_type()
                    ),
                ));
            }
            let scalar: ArrayRef = match literal {
                Literal::Int64(v) => Arc::new(Int64Array::from(vec![*v])),
                Literal::Float64(v) => Arc::new(Float64Array::from(vec![*v])),
            };
            let scalar = Scalar::new(scalar);
            let mask = match op {
                Comparison::Eq => cmp::eq(col, &scalar),
                Comparison::Lt => cmp::lt(col, &scalar),
                Comparison::LtEq => cmp::lt_eq(col, &scalar),
                Comparison::Gt => cmp::gt(col, &scalar),
                Comparison::GtEq => cmp::gt_eq(col, &scalar),
            }?;
            Ok(mask)
        }
        Predicate::And(left, right) => {
            let l = evaluate_predicate(batch, left)?;
            let r = evaluate_predicate(batch, right)?;
            Ok(and_kleene(&l, &r)?)
        }
    }
}

/// Fused filter + projection.
pub fn filter_project(
    batch: &RecordBatch,
    predicate: &Predicate,
    projection: &[usize],
) -> Result<RecordBatch, EngineError> {
    let mask = evaluate_predicate(batch, predicate)?;
    let projected = batch.project(projection)?;
    Ok(filter_record_batch(&projected, &mask)?)
}

/// Hasher for `Int64` join keys: one multiply-xorshift round (SplitMix64's
/// finalizer) instead of SipHash. Keys are data, not attacker-chosen hash-flood
/// input at this layer — the build side is bounded by the query's own join.
#[derive(Default, Clone, Copy)]
struct KeyHasher(u64);

impl std::hash::Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 = self.0.rotate_left(5) ^ u64::from_le_bytes(word);
        }
    }

    fn write_i64(&mut self, i: i64) {
        self.0 = self.0.rotate_left(5) ^ (i as u64);
    }
}

/// The build side of a hash join, hashed once: `key -> range` into a flat
/// row-index array grouped by key (no per-key allocation). Built once per
/// join and probed by every probe batch — see [`hash_join_probe`].
#[derive(Debug, Clone)]
pub struct JoinTable {
    ranges: HashMap<i64, (u32, u32), std::hash::BuildHasherDefault<KeyHasher>>,
    rows: Vec<u32>,
}

impl std::fmt::Debug for KeyHasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("KeyHasher")
    }
}

impl JoinTable {
    /// Hashes the `Int64` key column `key` of `build`; `NULL` keys are skipped
    /// (they never match).
    pub fn build(build: &RecordBatch, key: usize) -> Result<Self, EngineError> {
        let keys = int64_column(build, key, "build join key")?;
        // Pass 1: rows per key. Pass 2: prefix offsets. Pass 3: scatter rows.
        let mut counts: HashMap<i64, u32, std::hash::BuildHasherDefault<KeyHasher>> =
            HashMap::with_capacity_and_hasher(keys.len(), Default::default());
        for key in keys.iter().flatten() {
            *counts.entry(key).or_insert(0) += 1;
        }
        let mut ranges = HashMap::with_capacity_and_hasher(counts.len(), Default::default());
        let mut next: u32 = 0;
        for (key, count) in counts {
            ranges.insert(key, (next, count));
            next = next
                .checked_add(count)
                .ok_or_else(|| EngineError::execution("build side exceeds u32::MAX rows"))?;
        }
        let mut cursor: HashMap<i64, u32, std::hash::BuildHasherDefault<KeyHasher>> =
            ranges.iter().map(|(k, (start, _))| (*k, *start)).collect();
        let mut rows = vec![0u32; next as usize];
        for (j, key) in keys.iter().enumerate() {
            let Some(key) = key else { continue };
            if let Some(slot) = cursor.get_mut(&key) {
                rows[*slot as usize] = index_u32(j)?;
                *slot += 1;
            }
        }
        Ok(Self { ranges, rows })
    }

    /// Build rows whose key equals `key`, in build order.
    pub fn matches(&self, key: i64) -> &[u32] {
        match self.ranges.get(&key) {
            Some(&(start, len)) => &self.rows[start as usize..(start + len) as usize],
            None => &[],
        }
    }

    /// Number of distinct non-`NULL` keys.
    pub fn distinct_keys(&self) -> usize {
        self.ranges.len()
    }

    /// Number of build rows with a non-`NULL` key.
    pub fn rows(&self) -> usize {
        self.rows.len()
    }
}

/// Inner hash join on one `Int64` key per side; output = left columns then right columns.
/// Builds the table for `right` and probes once — see [`hash_join_probe`] for
/// the reusable form.
pub fn hash_join(
    left: &RecordBatch,
    right: &RecordBatch,
    left_key: usize,
    right_key: usize,
) -> Result<RecordBatch, EngineError> {
    let table = JoinTable::build(right, right_key)?;
    hash_join_probe(left, left_key, right, &table, None)
}

/// Probes `left` against a prebuilt [`JoinTable`] over `right`. The output is
/// `left ++ right` columns, or the `projection` subset (indices into that
/// concatenation) in projection order — columns outside it are never gathered.
pub fn hash_join_probe(
    left: &RecordBatch,
    left_key: usize,
    right: &RecordBatch,
    table: &JoinTable,
    projection: Option<&[usize]>,
) -> Result<RecordBatch, EngineError> {
    let lk = int64_column(left, left_key, "left join key")?;
    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();
    for (i, key) in lk.iter().enumerate() {
        let Some(key) = key else { continue };
        let matches = table.matches(key);
        if matches.is_empty() {
            continue;
        }
        let i = index_u32(i)?;
        left_idx.extend(std::iter::repeat_n(i, matches.len()));
        right_idx.extend_from_slice(matches);
    }
    let left_idx = UInt32Array::from(left_idx);
    let right_idx = UInt32Array::from(right_idx);

    let all_fields: Vec<_> = left
        .schema()
        .fields()
        .iter()
        .chain(right.schema().fields().iter())
        .cloned()
        .collect();
    let width = left.num_columns();
    let selected: Vec<usize> = match projection {
        Some(indices) => indices.to_vec(),
        None => (0..all_fields.len()).collect(),
    };
    let mut fields = Vec::with_capacity(selected.len());
    let mut columns = Vec::with_capacity(selected.len());
    for &index in &selected {
        let field = all_fields.get(index).ok_or_else(|| {
            EngineError::plan(format!(
                "join projection index {index} out of range for {} columns",
                all_fields.len()
            ))
        })?;
        let array = if index < width {
            take(left.column(index).as_ref(), &left_idx, None)?
        } else {
            take(right.column(index - width).as_ref(), &right_idx, None)?
        };
        fields.push(Arc::clone(field));
        columns.push(array);
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

fn index_u32(i: usize) -> Result<u32, EngineError> {
    u32::try_from(i).map_err(|_| EngineError::execution("batch exceeds u32::MAX rows"))
}

enum Acc {
    I64 {
        count: i64,
        sum: i128,
        min: i64,
        max: i64,
    },
    F64 {
        count: i64,
        sum: f64,
        min: f64,
        max: f64,
    },
}

impl Acc {
    fn for_type(dt: &DataType, index: usize) -> Result<Acc, EngineError> {
        match dt {
            DataType::Int64 => Ok(Acc::I64 {
                count: 0,
                sum: 0,
                min: i64::MAX,
                max: i64::MIN,
            }),
            DataType::Float64 => Ok(Acc::F64 {
                count: 0,
                sum: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
            }),
            other => Err(EngineError::unsupported(
                "cpu.aggregate",
                format!(
                    "aggregate input column {index} has type {other:?}; v1 supports Int64 and Float64"
                ),
            )),
        }
    }
}

enum Input<'a> {
    I(&'a Int64Array),
    F(&'a Float64Array),
}

/// Grouped `SUM/COUNT/MIN/MAX` over one `Int64` key.
pub fn aggregate(batch: &RecordBatch, spec: &AggregateSpec) -> Result<RecordBatch, EngineError> {
    let keys = int64_column(batch, spec.group_by, "group key")?;
    let mut inputs = Vec::with_capacity(spec.aggregates.len());
    for (_, idx) in &spec.aggregates {
        let col = column(batch, *idx)?;
        let input = match col.data_type() {
            DataType::Int64 => col.as_any().downcast_ref::<Int64Array>().map(Input::I),
            DataType::Float64 => col.as_any().downcast_ref::<Float64Array>().map(Input::F),
            _ => None,
        };
        inputs.push(input.ok_or_else(|| {
            EngineError::unsupported(
                "cpu.aggregate",
                format!(
                    "aggregate input column {idx} has type {:?}; v1 supports Int64 and Float64",
                    col.data_type()
                ),
            )
        })?);
    }

    let mut index: HashMap<Option<i64>, usize> = HashMap::new();
    let mut group_keys: Vec<Option<i64>> = Vec::new();
    let mut accs: Vec<Vec<Acc>> = Vec::new();
    for row in 0..batch.num_rows() {
        let key = keys.is_valid(row).then(|| keys.value(row));
        let group = match index.get(&key) {
            Some(&g) => g,
            None => {
                let mut fresh = Vec::with_capacity(spec.aggregates.len());
                for (input, (_, idx)) in inputs.iter().zip(&spec.aggregates) {
                    fresh.push(Acc::for_type(
                        match input {
                            Input::I(_) => &DataType::Int64,
                            Input::F(_) => &DataType::Float64,
                        },
                        *idx,
                    )?);
                }
                group_keys.push(key);
                accs.push(fresh);
                index.insert(key, group_keys.len() - 1);
                group_keys.len() - 1
            }
        };
        for (acc, input) in accs[group].iter_mut().zip(&inputs) {
            match (acc, input) {
                (
                    Acc::I64 {
                        count,
                        sum,
                        min,
                        max,
                    },
                    Input::I(arr),
                ) if arr.is_valid(row) => {
                    let v = arr.value(row);
                    *count += 1;
                    *sum += i128::from(v);
                    *min = (*min).min(v);
                    *max = (*max).max(v);
                }
                (
                    Acc::F64 {
                        count,
                        sum,
                        min,
                        max,
                    },
                    Input::F(arr),
                ) if arr.is_valid(row) => {
                    let v = arr.value(row);
                    *count += 1;
                    *sum += v;
                    *min = min.min(v);
                    *max = max.max(v);
                }
                _ => {}
            }
        }
    }

    let mut order: Vec<usize> = (0..group_keys.len()).collect();
    order.sort_by(|&a, &b| match (group_keys[a], group_keys[b]) {
        (Some(x), Some(y)) => x.cmp(&y),
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, None) => std::cmp::Ordering::Equal,
    });

    let key_field = batch.schema().field(spec.group_by).clone();
    let mut fields = vec![Field::new(key_field.name(), DataType::Int64, true)];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(
        order.iter().map(|&g| group_keys[g]).collect::<Vec<_>>(),
    ))];

    for (a, (func, idx)) in spec.aggregates.iter().enumerate() {
        let name = format!("{}({})", func.name(), batch.schema().field(*idx).name());
        let (dt, array): (DataType, ArrayRef) = match (func, &inputs[a]) {
            (AggregateFunction::Count, _) => (
                DataType::Int64,
                Arc::new(Int64Array::from(
                    order
                        .iter()
                        .map(|&g| Some(acc_count(&accs[g][a])))
                        .collect::<Vec<_>>(),
                )),
            ),
            (AggregateFunction::Sum, Input::I(_)) => {
                let mut out = Vec::with_capacity(order.len());
                for &g in &order {
                    out.push(match &accs[g][a] {
                        Acc::I64 { count, sum, .. } if *count > 0 => Some(
                            i64::try_from(*sum)
                                .map_err(|_| EngineError::execution("SUM overflowed Int64"))?,
                        ),
                        _ => None,
                    });
                }
                (DataType::Int64, Arc::new(Int64Array::from(out)))
            }
            (AggregateFunction::Sum, Input::F(_)) => (
                DataType::Float64,
                Arc::new(Float64Array::from(
                    order
                        .iter()
                        .map(|&g| match &accs[g][a] {
                            Acc::F64 { count, sum, .. } if *count > 0 => Some(*sum),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                )),
            ),
            (AggregateFunction::Min | AggregateFunction::Max, Input::I(_)) => (
                DataType::Int64,
                Arc::new(Int64Array::from(
                    order
                        .iter()
                        .map(|&g| match &accs[g][a] {
                            Acc::I64 {
                                count, min, max, ..
                            } if *count > 0 => Some(if *func == AggregateFunction::Min {
                                *min
                            } else {
                                *max
                            }),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                )),
            ),
            (AggregateFunction::Min | AggregateFunction::Max, Input::F(_)) => (
                DataType::Float64,
                Arc::new(Float64Array::from(
                    order
                        .iter()
                        .map(|&g| match &accs[g][a] {
                            Acc::F64 {
                                count, min, max, ..
                            } if *count > 0 => Some(if *func == AggregateFunction::Min {
                                *min
                            } else {
                                *max
                            }),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                )),
            ),
        };
        fields.push(Field::new(name, dt, true));
        columns.push(array);
    }
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

fn acc_count(acc: &Acc) -> i64 {
    match acc {
        Acc::I64 { count, .. } | Acc::F64 { count, .. } => *count,
    }
}

/// Appends a `Float32` distance column between `query` and each row's vector.
pub fn vector_distance(
    batch: &RecordBatch,
    column_index: usize,
    query: &[f32],
    metric: DistanceMetric,
    output_name: &str,
) -> Result<RecordBatch, EngineError> {
    let col = column(batch, column_index)?;
    let list = col
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| {
            EngineError::unsupported(
                "cpu.vector_distance",
                format!(
                    "column {column_index} has type {:?}; expected FixedSizeList<Float32>",
                    col.data_type()
                ),
            )
        })?;
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            EngineError::unsupported(
                "cpu.vector_distance",
                format!(
                    "vector element type is {:?}; expected Float32",
                    list.value_type()
                ),
            )
        })?;
    let dim = usize::try_from(list.value_length())
        .map_err(|_| EngineError::plan("negative list size"))?;
    if query.len() != dim {
        return Err(EngineError::plan(format!(
            "query vector has {} dimensions but the column has {dim}",
            query.len()
        )));
    }
    let raw = values.values();
    let distances: Vec<Option<f32>> = (0..batch.num_rows())
        .into_par_iter()
        .map(|row| {
            if list.is_null(row) {
                return None;
            }
            let start = usize::try_from(list.value_offset(row)).unwrap_or(0);
            let v = &raw[start..start + dim];
            Some(match metric {
                DistanceMetric::L2 => l2(v, query),
                DistanceMetric::Cosine => cosine(v, query),
            })
        })
        .collect();

    let mut fields: Vec<_> = batch.schema().fields().iter().cloned().collect();
    fields.push(Arc::new(Field::new(output_name, DataType::Float32, true)));
    let mut columns = batch.columns().to_vec();
    columns.push(Arc::new(Float32Array::from(distances)));
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

/// Euclidean distance.
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Cosine distance `1 - cos(θ)`; `NaN` when either norm is zero.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        f32::NAN
    } else {
        1.0 - dot / (na * nb)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use arrow::array::{Array, StringArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::SchemaRef;

    use super::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
            Field::new("s", DataType::Utf8, true),
        ]))
    }

    fn batch() -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(vec![
                    Some(1),
                    Some(2),
                    None,
                    Some(2),
                    Some(5),
                ])),
                Arc::new(Float64Array::from(vec![
                    Some(0.5),
                    None,
                    Some(2.5),
                    Some(3.5),
                    Some(-1.0),
                ])),
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    Some("c"),
                    None,
                    Some("e"),
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn filter_drops_null_predicates_and_projects() {
        let pred = Predicate::and(
            Predicate::compare(0, Comparison::GtEq, Literal::Int64(2)),
            Predicate::compare(1, Comparison::Lt, Literal::Float64(4.0)),
        );
        let out = filter_project(&batch(), &pred, &[2, 0]).unwrap();
        // row1: v is NULL → dropped; row2: k NULL → dropped; row3 kept; row4: v < 4 but k=5 ≥ 2 → kept
        assert_eq!(out.num_columns(), 2);
        assert_eq!(out.schema().field(0).name(), "s");
        let ks = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(ks.values(), &[2, 5]);
    }

    #[test]
    fn filter_type_mismatch_is_unsupported() {
        let pred = Predicate::compare(1, Comparison::Eq, Literal::Int64(1));
        assert!(
            filter_project(&batch(), &pred, &[0])
                .unwrap_err()
                .is_unsupported()
        );
    }

    #[test]
    fn join_skips_null_keys_and_multiplies_matches() {
        let right = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("rk", DataType::Int64, true),
                Field::new("name", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![Some(2), Some(2), None, Some(9)])),
                Arc::new(StringArray::from(vec!["two-a", "two-b", "null", "nine"])),
            ],
        )
        .unwrap();
        let out = hash_join(&batch(), &right, 0, 0).unwrap();
        assert_eq!(out.num_columns(), 5);
        // left rows with k=2 are rows 1 and 3; each matches two right rows → 4 rows
        assert_eq!(out.num_rows(), 4);
        let names = out
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let mut got: Vec<&str> = names.iter().flatten().collect();
        got.sort_unstable();
        assert_eq!(got, vec!["two-a", "two-a", "two-b", "two-b"]);
    }

    #[test]
    fn aggregate_groups_nulls_last_and_skips_null_values() {
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![
                (AggregateFunction::Sum, 1),
                (AggregateFunction::Count, 1),
                (AggregateFunction::Min, 1),
                (AggregateFunction::Max, 0),
            ],
        };
        let out = aggregate(&batch(), &spec).unwrap();
        let keys = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(
            keys.iter().collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(5), None]
        );
        let sums = out
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(
            sums.iter().collect::<Vec<_>>(),
            vec![Some(0.5), Some(3.5), Some(-1.0), Some(2.5)]
        );
        let counts = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(counts.values(), &[1, 1, 1, 1]);
        assert_eq!(out.schema().field(1).name(), "SUM(v)");
        assert_eq!(out.schema().field(4).name(), "MAX(k)");
        let maxk = out.column(4).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(
            maxk.iter().collect::<Vec<_>>(),
            vec![Some(1), Some(2), Some(5), None]
        );
    }

    #[test]
    fn aggregate_over_empty_batch_has_no_groups() {
        let empty = RecordBatch::new_empty(schema());
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![(AggregateFunction::Count, 1)],
        };
        let out = aggregate(&empty, &spec).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.num_columns(), 2);
    }

    #[test]
    fn vector_distances() {
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let vectors = FixedSizeListArray::try_new(
            field,
            2,
            Arc::new(Float32Array::from(vec![1.0, 0.0, 0.0, 1.0, 3.0, 4.0])),
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "vec",
            vectors.data_type().clone(),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(vectors)]).unwrap();
        let out = vector_distance(&batch, 0, &[1.0, 0.0], DistanceMetric::L2, "d").unwrap();
        let d = out
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(
            d.iter().collect::<Vec<_>>(),
            vec![Some(0.0), None, Some(20f32.sqrt())]
        );
        let out = vector_distance(&batch, 0, &[1.0, 0.0], DistanceMetric::Cosine, "d").unwrap();
        let d = out
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!((d.value(0)).abs() < 1e-6);
        assert!((d.value(2) - (1.0 - 3.0 / 5.0)).abs() < 1e-6);
        assert!(vector_distance(&batch, 0, &[1.0], DistanceMetric::L2, "d").is_err());
        assert!(cosine(&[0.0, 0.0], &[1.0, 0.0]).is_nan());
    }
}
