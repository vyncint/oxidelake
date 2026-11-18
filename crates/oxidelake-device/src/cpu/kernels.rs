//! CPU reference implementations of the four v1 operators, over host
//! `RecordBatch`es. These define the semantics the GPU kernels must match:
//!
//! * filter: a comparison against `NULL` is `NULL`, and `NULL` predicates drop
//!   the row (SQL `WHERE` semantics);
//! * join: `NULL` keys never match;
//! * aggregate: `NULL` keys form their own group; `SUM/MIN/MAX` skip `NULL`s
//!   and are `NULL` for groups with no values; `COUNT` counts non-`NULL`s;
//!   `SUM` over `Int64` wraps on overflow, as stock DataFusion's does;
//!   `MIN`/`MAX` over `Float64` follow IEEE 754 totalOrder, the order Arrow's
//!   comparison and sort kernels use (`NaN` greatest, a `NaN` with the sign
//!   bit set least, `-0.0` below `0.0`); output is sorted by key with the
//!   `NULL` group last;
//! * vector distance: a `NULL` vector yields a `NULL` distance; cosine distance
//!   is `1 - cos(θ)` and is `NaN` when either norm is zero.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
    PrimitiveArray, RecordBatch, Scalar, UInt32Array,
};
use arrow::compute::kernels::cmp;
use arrow::compute::{TakeOptions, and_kleene, filter_record_batch, take};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Float64Type, Int64Type, Schema};
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

    /// `(start, count)` of `key`'s run in [`Self::row_indices`].
    fn run(&self, key: i64) -> Option<(u32, u32)> {
        self.ranges.get(&key).copied()
    }

    /// Every distinct key with its run in [`Self::row_indices`], as
    /// `(key, start, count)`, in no particular order. A backend builds its
    /// device index from these rather than hashing the build side again
    /// (see `GpuBackend::prepare_join_build`).
    pub fn runs(&self) -> impl Iterator<Item = (i64, u32, u32)> + '_ {
        self.ranges
            .iter()
            .map(|(&key, &(start, count))| (key, start, count))
    }

    /// The build rows with a non-`NULL` key, grouped by key and in build
    /// order within a key: the array every run indexes.
    pub fn row_indices(&self) -> &[u32] {
        &self.rows
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

/// Probes `left` against a prebuilt [`JoinTable`] over `right` and returns
/// every match in one batch. The output is `left ++ right` columns, or the
/// `projection` subset (indices into that concatenation) in projection order
/// — columns outside it are never gathered.
///
/// One batch holds the whole product, however large: a key that repeats on
/// both sides makes it the product of the two counts. A caller that must
/// stay within a batch size probes with [`JoinMatches`] and gathers ranges
/// with [`hash_join_gather`].
pub fn hash_join_probe(
    left: &RecordBatch,
    left_key: usize,
    right: &RecordBatch,
    table: &JoinTable,
    projection: Option<&[usize]>,
) -> Result<RecordBatch, EngineError> {
    let matches = JoinMatches::probe(left, left_key, table)?;
    let rows = usize::try_from(matches.len())
        .map_err(|_| EngineError::execution("join output exceeds usize::MAX rows"))?;
    hash_join_gather(left, right, table, &matches, 0, rows, projection)
}

/// One probe batch's matches against a [`JoinTable`]: a sequence in
/// probe-row order, and in build order within a probe row, kept as each
/// row's run in the table and the running total of the matches before it.
/// Nothing is materialised: [`hash_join_gather`] turns any range of the
/// sequence into output rows, so a probe batch's output can be emitted in
/// batches of a bounded size however often its keys repeat (#95).
#[derive(Debug, Clone, Default)]
pub struct JoinMatches {
    /// `offsets[i]` is probe row `i`'s first match; one entry more than
    /// there are rows, the last being the total. `u64`: one batch's matches
    /// can pass `u32::MAX` however its output is cut.
    offsets: Vec<u64>,
    /// Where probe row `i`'s run starts in the table's row array (zero when
    /// its key matches nothing).
    starts: Vec<u32>,
}

impl JoinMatches {
    /// Looks every row of `left`'s `Int64` column `left_key` up in `table`.
    /// `NULL` keys match nothing.
    pub fn probe(
        left: &RecordBatch,
        left_key: usize,
        table: &JoinTable,
    ) -> Result<Self, EngineError> {
        let keys = int64_column(left, left_key, "left join key")?;
        // Output rows are gathered with `u32` probe-row indices.
        index_u32(keys.len())?;
        let mut offsets = Vec::with_capacity(keys.len() + 1);
        let mut starts = Vec::with_capacity(keys.len());
        let mut total = 0u64;
        for key in keys.iter() {
            offsets.push(total);
            let (start, count) = key.and_then(|k| table.run(k)).unwrap_or((0, 0));
            starts.push(start);
            total += u64::from(count);
        }
        offsets.push(total);
        Ok(Self { offsets, starts })
    }

    /// Total matches: the rows this probe batch joins to.
    pub fn len(&self) -> u64 {
        self.offsets.last().copied().unwrap_or(0)
    }

    /// `true` when no probe row matched.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Probe rows looked up.
    pub fn probe_rows(&self) -> usize {
        self.starts.len()
    }

    /// The probe row and the build row of matches `first..first + rows`.
    pub fn pairs(
        &self,
        table: &JoinTable,
        first: u64,
        rows: usize,
    ) -> Result<(UInt32Array, UInt32Array), EngineError> {
        let end = u64::try_from(rows)
            .ok()
            .and_then(|r| first.checked_add(r))
            .filter(|&end| end <= self.len())
            .ok_or_else(|| {
                EngineError::execution(format!(
                    "matches {first}..+{rows} are outside a probe with {} matches",
                    self.len()
                ))
            })?;
        let mut left = Vec::with_capacity(rows);
        let mut right = Vec::with_capacity(rows);
        // The probe row holding match `first` is the last one whose offset
        // is at most `first`: a row with no matches shares its offset with
        // the row after it, so it never wins.
        let mut row = self
            .offsets
            .partition_point(|&offset| offset <= first)
            .saturating_sub(1);
        let mut at = first;
        while at < end {
            // `at < end <= total == offsets[probe_rows]`, so `row` is a probe
            // row and `row + 1` an offset.
            let (lo, hi) = (self.offsets[row], self.offsets[row + 1]);
            let upto = hi.min(end);
            if upto > at {
                let start = self.starts[row] as usize;
                let run = table
                    .rows
                    .get(start + (at - lo) as usize..start + (upto - lo) as usize)
                    .ok_or_else(|| {
                        EngineError::execution("join matches were probed against another table")
                    })?;
                left.extend(std::iter::repeat_n(index_u32(row)?, run.len()));
                right.extend_from_slice(run);
                at = upto;
            }
            row += 1;
        }
        Ok((UInt32Array::from(left), UInt32Array::from(right)))
    }
}

/// Gathers matches `first..first + rows` of `matches` (probed from `left`
/// against `table`, built over `right`) as output rows. The output is
/// `left ++ right` columns, or the `projection` subset (indices into that
/// concatenation) in projection order — columns outside it are never
/// gathered.
pub fn hash_join_gather(
    left: &RecordBatch,
    right: &RecordBatch,
    table: &JoinTable,
    matches: &JoinMatches,
    first: u64,
    rows: usize,
    projection: Option<&[usize]>,
) -> Result<RecordBatch, EngineError> {
    if matches.probe_rows() != left.num_rows() {
        return Err(EngineError::execution(format!(
            "join matches cover {} probe rows but the probe batch has {}",
            matches.probe_rows(),
            left.num_rows()
        )));
    }
    let (left_idx, right_idx) = matches.pairs(table, first, rows)?;
    // A table built over another batch must not index past this one: checked
    // here, where it costs one pass over a chunk, rather than trusted.
    let checked = Some(TakeOptions { check_bounds: true });

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
            take(left.column(index).as_ref(), &left_idx, checked.clone())?
        } else {
            take(
                right.column(index - width).as_ref(),
                &right_idx,
                checked.clone(),
            )?
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

/// One aggregate's per-group state: a slot per group, `None` until the group
/// sees a non-`NULL` value — which is what makes `SUM/MIN/MAX` `NULL` for a
/// group that has none.
#[derive(Debug)]
enum Accumulator {
    Count(Vec<i64>),
    SumI64(Vec<Option<i64>>),
    SumF64(Vec<Option<f64>>),
    MinI64(Vec<Option<i64>>),
    MaxI64(Vec<Option<i64>>),
    MinF64(Vec<Option<f64>>),
    MaxF64(Vec<Option<f64>>),
}

fn primitive<'a, T: ArrowPrimitiveType>(
    values: &'a ArrayRef,
    what: &str,
) -> Result<&'a PrimitiveArray<T>, EngineError> {
    values
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| {
            EngineError::execution(format!(
                "{what} input has type {:?}, expected {:?}",
                values.data_type(),
                T::DATA_TYPE
            ))
        })
}

/// Folds every valid `values[row]` into `slots[ids[row]]`: `first` starts a
/// group's value, `next` combines it with the value the group holds.
fn fold<T: ArrowPrimitiveType>(
    values: &PrimitiveArray<T>,
    ids: &[usize],
    slots: &mut [Option<T::Native>],
    first: impl Fn(T::Native) -> T::Native,
    next: impl Fn(T::Native, T::Native) -> T::Native,
) {
    for (row, &g) in ids.iter().enumerate() {
        if values.is_valid(row) {
            let v = values.value(row);
            slots[g] = Some(match slots[g] {
                Some(held) => next(held, v),
                None => first(v),
            });
        }
    }
}

/// `MIN` in IEEE 754 totalOrder (#101). `f64::min` is IEEE minNum: it skips
/// NaN, so an all-NaN group used to keep the +inf it started from, and it
/// leaves the sign of a zero result unspecified.
fn total_min(held: f64, v: f64) -> f64 {
    if v.total_cmp(&held).is_lt() { v } else { held }
}

/// `MAX` in IEEE 754 totalOrder (#101); see [`total_min`].
fn total_max(held: f64, v: f64) -> f64 {
    if v.total_cmp(&held).is_gt() { v } else { held }
}

impl Accumulator {
    fn new(func: AggregateFunction, input: &DataType, index: usize) -> Result<Self, EngineError> {
        Ok(match (func, input) {
            (AggregateFunction::Count, DataType::Int64 | DataType::Float64) => {
                Self::Count(Vec::new())
            }
            (AggregateFunction::Sum, DataType::Int64) => Self::SumI64(Vec::new()),
            (AggregateFunction::Sum, DataType::Float64) => Self::SumF64(Vec::new()),
            (AggregateFunction::Min, DataType::Int64) => Self::MinI64(Vec::new()),
            (AggregateFunction::Max, DataType::Int64) => Self::MaxI64(Vec::new()),
            (AggregateFunction::Min, DataType::Float64) => Self::MinF64(Vec::new()),
            (AggregateFunction::Max, DataType::Float64) => Self::MaxF64(Vec::new()),
            (_, other) => {
                return Err(EngineError::unsupported(
                    "cpu.aggregate",
                    format!(
                        "aggregate input column {index} has type {other:?}; v1 supports Int64 and Float64"
                    ),
                ));
            }
        })
    }

    fn grow(&mut self, groups: usize) {
        match self {
            Self::Count(v) => v.resize(groups, 0),
            Self::SumI64(v) | Self::MinI64(v) | Self::MaxI64(v) => v.resize(groups, None),
            Self::SumF64(v) | Self::MinF64(v) | Self::MaxF64(v) => v.resize(groups, None),
        }
    }

    /// Folds input values: `values[row]` belongs to group `ids[row]`.
    fn update(&mut self, values: &ArrayRef, ids: &[usize]) -> Result<(), EngineError> {
        match self {
            Self::Count(counts) => {
                for (row, &g) in ids.iter().enumerate() {
                    if values.is_valid(row) {
                        counts[g] += 1;
                    }
                }
            }
            // Wrapping, like every stock DataFusion `SUM` accumulator
            // (`add_wrapping`), rather than an error (#100).
            Self::SumI64(slots) => fold(
                primitive::<Int64Type>(values, "SUM")?,
                ids,
                slots,
                |v| v,
                i64::wrapping_add,
            ),
            // From 0.0, not from the first value, as DataFusion's accumulator
            // starts: the SUM of {-0.0} is 0.0.
            Self::SumF64(slots) => fold(
                primitive::<Float64Type>(values, "SUM")?,
                ids,
                slots,
                |v| 0.0 + v,
                |held, v| held + v,
            ),
            Self::MinI64(slots) => fold(
                primitive::<Int64Type>(values, "MIN")?,
                ids,
                slots,
                |v| v,
                i64::min,
            ),
            Self::MaxI64(slots) => fold(
                primitive::<Int64Type>(values, "MAX")?,
                ids,
                slots,
                |v| v,
                i64::max,
            ),
            Self::MinF64(slots) => fold(
                primitive::<Float64Type>(values, "MIN")?,
                ids,
                slots,
                |v| v,
                total_min,
            ),
            Self::MaxF64(slots) => fold(
                primitive::<Float64Type>(values, "MAX")?,
                ids,
                slots,
                |v| v,
                total_max,
            ),
        }
        Ok(())
    }

    /// Folds partial results — this accumulator's own output for other rows.
    /// Partial counts add up; the SUM, MIN or MAX of partial SUM, MIN or MAX
    /// values is the same function again, so those fold like input.
    fn merge(&mut self, partial: &ArrayRef, ids: &[usize]) -> Result<(), EngineError> {
        if let Self::Count(counts) = self {
            let partial = primitive::<Int64Type>(partial, "partial COUNT")?;
            for (row, &g) in ids.iter().enumerate() {
                if partial.is_valid(row) {
                    counts[g] = counts[g].wrapping_add(partial.value(row));
                }
            }
            return Ok(());
        }
        self.update(partial, ids)
    }

    /// The output column, groups in `order`.
    fn finish(&self, order: &[usize]) -> (DataType, ArrayRef) {
        fn pick<T: Copy>(slots: &[T], order: &[usize]) -> Vec<T> {
            order.iter().map(|&g| slots[g]).collect()
        }
        match self {
            Self::Count(counts) => (
                DataType::Int64,
                Arc::new(Int64Array::from(pick(counts, order))),
            ),
            Self::SumI64(slots) | Self::MinI64(slots) | Self::MaxI64(slots) => (
                DataType::Int64,
                Arc::new(Int64Array::from(pick(slots, order))),
            ),
            Self::SumF64(slots) | Self::MinF64(slots) | Self::MaxF64(slots) => (
                DataType::Float64,
                Arc::new(Float64Array::from(pick(slots, order))),
            ),
        }
    }
}

/// Grouped `SUM/COUNT/MIN/MAX` over one `Int64` key, fed a batch at a time.
///
/// [`Self::update`] folds input rows; [`Self::merge`] folds a batch laid out
/// as this aggregation's own output — what [`aggregate`] or a device kernel
/// returned for other rows — so a device result for one batch and the CPU
/// reference for the next accumulate into the same groups, and so does the
/// final half of a two-phase aggregate. State is one slot per group and
/// aggregate however many rows pass through (#29); [`Self::finish`] emits the
/// groups sorted by key, `NULL` last.
#[derive(Debug)]
pub struct GroupedAggregator {
    group_by: usize,
    key_name: String,
    inputs: Vec<usize>,
    names: Vec<String>,
    index: HashMap<i64, usize>,
    null_group: Option<usize>,
    keys: Vec<Option<i64>>,
    accumulators: Vec<Accumulator>,
}

impl GroupedAggregator {
    /// Accumulators for `spec` over batches of `schema`, with the key and
    /// input types checked up front.
    pub fn new(schema: &Schema, spec: &AggregateSpec) -> Result<Self, EngineError> {
        let field = |index: usize| {
            schema.fields().get(index).ok_or_else(|| {
                EngineError::plan(format!(
                    "column index {index} out of range for a batch with {} columns",
                    schema.fields().len()
                ))
            })
        };
        let key = field(spec.group_by)?;
        if key.data_type() != &DataType::Int64 {
            return Err(EngineError::unsupported(
                "cpu.int64_column",
                format!(
                    "group key column {} has type {:?}; v1 supports Int64",
                    spec.group_by,
                    key.data_type()
                ),
            ));
        }
        let mut accumulators = Vec::with_capacity(spec.aggregates.len());
        let mut names = Vec::with_capacity(spec.aggregates.len());
        for (func, index) in &spec.aggregates {
            let input = field(*index)?;
            accumulators.push(Accumulator::new(*func, input.data_type(), *index)?);
            names.push(format!("{}({})", func.name(), input.name()));
        }
        Ok(Self {
            group_by: spec.group_by,
            key_name: key.name().clone(),
            inputs: spec.aggregates.iter().map(|(_, index)| *index).collect(),
            names,
            index: HashMap::new(),
            null_group: None,
            keys: Vec::new(),
            accumulators,
        })
    }

    /// Groups seen so far.
    pub fn groups(&self) -> usize {
        self.keys.len()
    }

    /// The group of every row of `keys`, opening groups for keys not seen yet.
    fn group_ids(&mut self, keys: &Int64Array) -> Vec<usize> {
        let mut ids = Vec::with_capacity(keys.len());
        for row in 0..keys.len() {
            let next = self.keys.len();
            let id = if keys.is_valid(row) {
                let key = keys.value(row);
                let groups = &mut self.keys;
                *self.index.entry(key).or_insert_with(|| {
                    groups.push(Some(key));
                    next
                })
            } else if let Some(g) = self.null_group {
                g
            } else {
                self.keys.push(None);
                self.null_group = Some(next);
                next
            };
            ids.push(id);
        }
        for accumulator in &mut self.accumulators {
            accumulator.grow(self.keys.len());
        }
        ids
    }

    /// Folds the rows of one input batch.
    pub fn update(&mut self, batch: &RecordBatch) -> Result<(), EngineError> {
        let keys = int64_column(batch, self.group_by, "group key")?;
        let ids = self.group_ids(keys);
        for (accumulator, &input) in self.accumulators.iter_mut().zip(&self.inputs) {
            accumulator.update(column(batch, input)?, &ids)?;
        }
        Ok(())
    }

    /// Folds a partial result: a batch laid out as this aggregation's output
    /// (the key, then one column per aggregate) that covers other rows.
    pub fn merge(&mut self, partial: &RecordBatch) -> Result<(), EngineError> {
        if partial.num_columns() != 1 + self.accumulators.len() {
            return Err(EngineError::execution(format!(
                "a partial aggregate has {} columns; this aggregation's output has {}",
                partial.num_columns(),
                1 + self.accumulators.len()
            )));
        }
        let keys = int64_column(partial, 0, "partial group key")?;
        let ids = self.group_ids(keys);
        for (i, accumulator) in self.accumulators.iter_mut().enumerate() {
            accumulator.merge(partial.column(1 + i), &ids)?;
        }
        Ok(())
    }

    /// The groups, sorted by key with the `NULL` group last.
    pub fn finish(self) -> Result<RecordBatch, EngineError> {
        let mut order: Vec<usize> = (0..self.keys.len()).collect();
        // `None` sorts first as an `Option`; the reference puts it last.
        order.sort_unstable_by_key(|&g| (self.keys[g].is_none(), self.keys[g]));
        let mut fields = vec![Field::new(&self.key_name, DataType::Int64, true)];
        let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(
            order.iter().map(|&g| self.keys[g]).collect::<Vec<_>>(),
        ))];
        for (accumulator, name) in self.accumulators.iter().zip(&self.names) {
            let (data_type, array) = accumulator.finish(&order);
            fields.push(Field::new(name, data_type, true));
            columns.push(array);
        }
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }
}

/// Grouped `SUM/COUNT/MIN/MAX` over one `Int64` key, in one pass over `batch`.
pub fn aggregate(batch: &RecordBatch, spec: &AggregateSpec) -> Result<RecordBatch, EngineError> {
    let mut groups = GroupedAggregator::new(&batch.schema(), spec)?;
    groups.update(batch)?;
    groups.finish()
}

/// `x` as a signed integer that orders like IEEE 754 totalOrder: comparing
/// two keys as `i64` is `f64::total_cmp` on the values. A device has 64-bit
/// integer atomic min/max but no totalOrder on doubles, so the CUDA kernel
/// reduces these keys (`oxide_total_order_key` in `aggregation.cu` is the
/// same transform) and the host decodes them with [`from_total_order_key`].
pub const fn total_order_key(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    // Flip every bit but the sign of a negative value: negative values then
    // order backwards by magnitude, below every positive one.
    bits ^ ((((bits >> 63) as u64) >> 1) as i64)
}

/// The value whose [`total_order_key`] is `key`; the transform is its own
/// inverse. `i64::MAX` decodes to the greatest value (a positive `NaN`) and
/// `i64::MIN` to the least (a negative one), which is what makes them the
/// device's starting values for `MIN` and `MAX`.
pub const fn from_total_order_key(key: i64) -> f64 {
    f64::from_bits((key ^ ((((key >> 63) as u64) >> 1) as i64)) as u64)
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

    /// Gathering a probe's matches in ranges of any size, cut inside a probe
    /// row's matches or across rows with none, yields exactly the one-batch
    /// result, in the same order (#95).
    #[test]
    fn join_matches_gather_in_ranges_equal_one_gather() {
        let right = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("rk", DataType::Int64, true),
                Field::new("name", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![
                    Some(2),
                    Some(2),
                    None,
                    Some(9),
                    Some(2),
                    Some(5),
                ])),
                Arc::new(StringArray::from(vec![
                    "a", "b", "null", "nine", "c", "five",
                ])),
            ],
        )
        .unwrap();
        let left = batch();
        let table = JoinTable::build(&right, 0).unwrap();
        let matches = JoinMatches::probe(&left, 0, &table).unwrap();
        let whole = hash_join_probe(&left, 0, &right, &table, None).unwrap();
        // k = 1, 2, NULL, 2, 5 against three 2s and one 5.
        assert_eq!(matches.len(), 7);
        assert_eq!(whole.num_rows(), 7);
        let projection = [0, 1, 3, 4];
        for chunk in 1..=8 {
            let mut parts = Vec::new();
            let mut first = 0;
            while first < matches.len() {
                let rows = chunk.min(usize::try_from(matches.len() - first).unwrap());
                let part = hash_join_gather(
                    &left,
                    &right,
                    &table,
                    &matches,
                    first,
                    rows,
                    Some(&projection),
                )
                .unwrap();
                assert_eq!(part.num_rows(), rows);
                parts.push(part);
                first += rows as u64;
            }
            let joined = arrow::compute::concat_batches(&parts[0].schema(), &parts).unwrap();
            assert_eq!(joined, whole.project(&projection).unwrap(), "chunk {chunk}");
        }
        // Out of range, or probed against another table: an error, never a panic.
        assert!(matches.pairs(&table, 7, 1).is_err());
        assert!(matches.pairs(&table, 0, usize::MAX).is_err());
        let other = JoinTable::build(&right.slice(0, 2), 0).unwrap();
        assert!(hash_join_gather(&left, &right, &other, &matches, 0, 7, None).is_err());
        assert!(hash_join_gather(&left.slice(0, 2), &right, &table, &matches, 0, 1, None).is_err());
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

    /// `k Int64?`, `i Int64?`, `f Float64?`.
    fn edge_batch(rows: &[(Option<i64>, Option<i64>, Option<f64>)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("i", DataType::Int64, true),
            Field::new("f", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(
                    rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(
                    rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    /// Every column as bit patterns (`None` for `NULL`), so a comparison
    /// tells `-0.0` from `0.0` and a NaN's sign.
    fn bits(batch: &RecordBatch) -> Vec<Vec<Option<u64>>> {
        batch
            .columns()
            .iter()
            .map(|c| {
                if let Some(ints) = c.as_any().downcast_ref::<Int64Array>() {
                    ints.iter().map(|v| v.map(|v| v as u64)).collect()
                } else {
                    let floats = c.as_any().downcast_ref::<Float64Array>().unwrap();
                    floats.iter().map(|v| v.map(f64::to_bits)).collect()
                }
            })
            .collect()
    }

    /// Folding batch after batch, or merging each batch's own result in
    /// either order, gives exactly what one pass over all the rows gives —
    /// the streaming aggregate's whole correctness argument (#29). The rows
    /// carry every value the accumulators treat specially: `NULL` keys and
    /// values, a group with no values, Int64 sums that wrap, NaN of both
    /// signs, signed zeros and an infinity. (A float `SUM` over NaNs of both
    /// signs would depend on the order of addition, in DataFusion too, so no
    /// group mixes them.)
    #[test]
    fn batch_by_batch_and_merged_partials_equal_one_pass() {
        use AggregateFunction::{Count, Max, Min, Sum};
        let nan = f64::NAN;
        let rows = [
            (Some(1), Some(i64::MAX - 1), Some(1.0)),
            (Some(2), Some(5), Some(nan)),
            (None, Some(i64::MAX), Some(-0.0)),
            (Some(1), Some(i64::MAX - 1), Some(nan)),
            (Some(3), None, None),
            (Some(4), Some(i64::MIN), Some(-nan)),
            (None, Some(i64::MAX), Some(0.0)),
            (Some(1), Some(-3), Some(3.0)),
            (Some(4), Some(7), Some(f64::NEG_INFINITY)),
            (Some(2), Some(-1), Some(0.25)),
        ];
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![
                (Sum, 1),
                (Count, 1),
                (Min, 1),
                (Max, 1),
                (Sum, 2),
                (Count, 2),
                (Min, 2),
                (Max, 2),
            ],
        };
        let whole = aggregate(&edge_batch(&rows), &spec).unwrap();
        for split in 1..rows.len() {
            let (a, b) = rows.split_at(split);
            let (a, b) = (edge_batch(a), edge_batch(b));
            let mut streamed = GroupedAggregator::new(&a.schema(), &spec).unwrap();
            streamed.update(&a).unwrap();
            streamed.update(&b).unwrap();
            let mut merged = GroupedAggregator::new(&a.schema(), &spec).unwrap();
            merged.merge(&aggregate(&b, &spec).unwrap()).unwrap();
            merged.merge(&aggregate(&a, &spec).unwrap()).unwrap();
            assert_eq!(
                bits(&streamed.finish().unwrap()),
                bits(&whole),
                "update, split at {split}"
            );
            assert_eq!(
                bits(&merged.finish().unwrap()),
                bits(&whole),
                "merge, split at {split}"
            );
        }
        // And the answers themselves: wrapped sums, totalOrder extremes.
        let sums = whole
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(
            sums.iter().collect::<Vec<_>>(),
            vec![Some(-7), Some(4), None, Some(i64::MIN + 7), Some(-2)]
        );
        let float = |c: usize| bits(&whole)[c].clone();
        assert_eq!(
            float(7),
            [Some(1.0), Some(0.25), None, Some(-nan), Some(-0.0)]
                .map(|v| v.map(f64::to_bits))
                .to_vec()
        );
        assert_eq!(
            float(8),
            [
                Some(nan),
                Some(nan),
                None,
                Some(f64::NEG_INFINITY),
                Some(0.0)
            ]
            .map(|v| v.map(f64::to_bits))
            .to_vec()
        );
    }

    /// A partial result that is not laid out as this aggregation's output is
    /// refused, not folded into the wrong accumulators.
    #[test]
    fn merging_a_mislaid_partial_is_an_error() {
        let spec = AggregateSpec {
            group_by: 0,
            aggregates: vec![(AggregateFunction::Count, 1)],
        };
        let mut groups = GroupedAggregator::new(&schema(), &spec).unwrap();
        assert!(groups.merge(&batch()).is_err());
        assert_eq!(groups.groups(), 0);
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

    /// The integer image the CUDA kernel reduces orders exactly as
    /// `f64::total_cmp` and decodes to the same bits (#101): signed zeros,
    /// subnormals, infinities and NaNs of either sign and any payload.
    #[test]
    fn total_order_keys_order_like_total_cmp_and_round_trip() {
        let values = [
            f64::from_bits(u64::MAX),
            -f64::NAN,
            f64::NEG_INFINITY,
            f64::MIN,
            -1.0,
            -f64::from_bits(1),
            -0.0,
            0.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            1.0,
            f64::MAX,
            f64::INFINITY,
            f64::from_bits(0x7FF0_0000_0000_0001),
            f64::NAN,
            f64::from_bits(0x7FFF_FFFF_FFFF_FFFF),
        ];
        for a in values {
            assert_eq!(
                from_total_order_key(total_order_key(a)).to_bits(),
                a.to_bits()
            );
            for b in values {
                assert_eq!(
                    total_order_key(a).cmp(&total_order_key(b)),
                    a.total_cmp(&b),
                    "{a:?} ({:#x}) vs {b:?} ({:#x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
        assert_eq!(
            from_total_order_key(i64::MAX).to_bits(),
            0x7FFF_FFFF_FFFF_FFFF
        );
        assert_eq!(from_total_order_key(i64::MIN).to_bits(), u64::MAX);
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
