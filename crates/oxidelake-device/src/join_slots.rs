//! The device index of a hash join's build side (#45, #108).
//!
//! One open-addressing slot per *distinct* build key, holding `(start,
//! count)` into the row-index array of the CPU reference's [`JoinTable`] —
//! the layout the CPU path probes. A duplicated key therefore costs a device
//! probe nothing, and a key's rows come back in build order exactly as the
//! CPU path emits them, which is what lets one probe batch move to the CPU
//! reference between two output chunks without repeating or losing a row. A
//! key starts its walk at [`slot_hash`]; an empty slot has count zero, which
//! no key in the table can have.
//!
//! The CUDA backend builds this once per operator on the host, from the
//! table the operator already holds, and uploads it; the kernels in
//! `kernels/cuda/hash_join.cu` only read it, so the device needs no atomics.
//! It replaced a table that gave every build *row* a slot, claimed with
//! `atomicCAS` and rebuilt for every probe batch: a key repeated D times cost
//! O(D²) CAS attempts per batch to build, and O(D) slot visits for every
//! probe row whose hash landed in its run.
//!
//! Backend-independent, so it is unit-tested on the CPU — including a
//! sequential replica of the device's probe loop — where no CUDA job runs.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use oxidelake_core::EngineError;

use crate::cpu::kernels::JoinTable;

/// The most slots the index may have: the kernels address it with an
/// `unsigned int`, and twice the distinct keys must still fit.
const MAX_CAPACITY: usize = 1 << 31;

/// The slot `key` starts its walk at in a table of `capacity` slots (a power
/// of two). Bit for bit the kernel's `oxide_hj_hash`: the host places each
/// key with this and the device probes with that, so a difference would send
/// every probe to a slot its key was never put in.
pub(crate) fn slot_hash(key: i64, capacity: u32) -> u32 {
    let mut h = key as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    // The mask keeps the value below `capacity`, so it fits.
    (h & u64::from(capacity.wrapping_sub(1))) as u32
}

/// A hash join's build side as the device probes it: four parallel arrays of
/// `capacity` slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JoinSlots {
    pub(crate) capacity: u32,
    pub(crate) keys: Vec<i64>,
    pub(crate) starts: Vec<u32>,
    pub(crate) counts: Vec<u32>,
    distinct: usize,
}

impl JoinSlots {
    /// Places every distinct key of `table`, at a load of at most one half.
    ///
    /// `build_rows` is the length of the batch the table indexes. Every row
    /// index and every run is checked against it here, once, so the device
    /// can gather at the indices it reads without checking them per row.
    pub(crate) fn build(table: &JoinTable, build_rows: usize) -> Result<Self, EngineError> {
        let rows = table.row_indices();
        if let Some(&bad) = rows.iter().find(|&&r| r as usize >= build_rows) {
            return Err(EngineError::execution(format!(
                "join table row {bad} is out of range for a {build_rows}-row build side"
            )));
        }
        let distinct = table.distinct_keys();
        let capacity = distinct
            .saturating_mul(2)
            .max(16)
            .checked_next_power_of_two()
            .filter(|&c| c <= MAX_CAPACITY)
            .and_then(|c| u32::try_from(c).ok())
            .ok_or_else(|| {
                EngineError::unsupported(
                    "device.join_table",
                    format!("{distinct} distinct join keys exceed the device index"),
                )
            })?;
        let slots = capacity as usize;
        let mut keys = vec![0i64; slots];
        let mut starts = vec![0u32; slots];
        let mut counts = vec![0u32; slots];
        for (key, start, count) in table.runs() {
            let end = start as usize + count as usize;
            if count == 0 || end > rows.len() {
                return Err(EngineError::execution(format!(
                    "join table run {start}+{count} for key {key} is outside its {} rows",
                    rows.len()
                )));
            }
            let mut slot = slot_hash(key, capacity);
            let mut placed = false;
            for _ in 0..capacity {
                let s = slot as usize;
                if counts[s] == 0 {
                    keys[s] = key;
                    starts[s] = start;
                    counts[s] = count;
                    placed = true;
                    break;
                }
                slot = (slot + 1) & (capacity - 1);
            }
            if !placed {
                // Unreachable at a load of at most one half; a typed error
                // rather than a silently missing key if that ever changes.
                return Err(EngineError::execution("device join index is full"));
            }
        }
        Ok(Self {
            capacity,
            keys,
            starts,
            counts,
            distinct,
        })
    }

    /// Number of keys placed: one slot each, however many rows share it.
    pub(crate) fn distinct(&self) -> usize {
        self.distinct
    }

    /// The device's probe loop (`oxide_hj_count`), run sequentially: the
    /// `(start, count)` of `key`'s run, or `None`, and how many slots the
    /// walk read. A replica for the tests, where no device runs it.
    #[cfg(test)]
    pub(crate) fn probe(&self, key: i64) -> (Option<(u32, u32)>, u32) {
        let mut slot = slot_hash(key, self.capacity);
        let mut visits = 0u32;
        for _ in 0..self.capacity {
            visits += 1;
            let s = slot as usize;
            let count = self.counts[s];
            if count == 0 {
                break;
            }
            if self.keys[s] == key {
                return (Some((self.starts[s], count)), visits);
            }
            slot = (slot + 1) & (self.capacity - 1);
        }
        (None, visits)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn table(keys: Vec<Option<i64>>) -> (JoinTable, usize) {
        let rows = keys.len();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(keys))]).unwrap();
        (JoinTable::build(&batch, 0).unwrap(), rows)
    }

    /// The kernel's hash, evaluated by compiling `kernels/cuda/hash_join.cu`
    /// with g++ against a shim of the CUDA builtins (there is no NVRTC on the
    /// machines this suite runs on). The textual check below catches an edit
    /// to the kernel's constants; these values catch one to this side.
    #[test]
    fn slot_hash_is_the_kernels_hash() {
        assert_eq!(slot_hash(0, 16), 0);
        assert_eq!(slot_hash(1, 16), 12);
        assert_eq!(slot_hash(-1, 1024), 801);
        assert_eq!(slot_hash(42, 131_072), 22_988);
        assert_eq!(slot_hash(i64::MIN, 1 << 20), 107_667);
        let kernel = include_str!("../kernels/cuda/hash_join.cu");
        for needle in [
            "h ^= h >> 33;",
            "h *= 0xff51afd7ed558ccdULL;",
            "h *= 0xc4ceb9fe1a85ec53ULL;",
            "(h & (unsigned long long)(capacity - 1u))",
        ] {
            assert!(kernel.contains(needle), "hash_join.cu lost `{needle}`");
        }
    }

    #[test]
    fn every_key_finds_its_run_and_absent_keys_find_nothing() {
        let keys: Vec<Option<i64>> = (0..500i64)
            .map(|r| (r % 11 != 4).then_some((r * 7919) % 37 - 18))
            .chain([Some(i64::MIN), Some(i64::MAX), None])
            .collect();
        let (table, rows) = table(keys);
        let slots = JoinSlots::build(&table, rows).unwrap();
        assert_eq!(slots.distinct(), table.distinct_keys());
        assert!(slots.capacity as usize >= 2 * slots.distinct());
        for (key, start, count) in table.runs() {
            assert_eq!(slots.probe(key).0, Some((start, count)), "key {key}");
            let run = &table.row_indices()[start as usize..(start + count) as usize];
            assert_eq!(run, table.matches(key));
        }
        for absent in [19, -19, 1 << 40, i64::MIN + 1] {
            assert_eq!(slots.probe(absent).0, None, "key {absent}");
        }
    }

    /// #108's measurement, on the layout the device now probes: the
    /// reviewer's four build sides of 100 000 rows, probed by a 65 536-row
    /// batch whose keys are all absent. The per-row slot layout this
    /// replaced read 1.8, 7 246, 19 050 and 5 324 slots per probe row for
    /// them (and paid 1.3e5 to 5.0e9 CAS attempts per batch to build);
    /// here a probe is bounded by the load factor, not by how often a key
    /// repeats.
    #[test]
    fn a_repeated_build_key_costs_a_probe_nothing() {
        fn splitmix(s: &mut u64) -> u64 {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        let mut state = 7u64;
        let probe: Vec<i64> = (0..65_536)
            .map(|_| ((splitmix(&mut state) >> 1) as i64) | 1 << 62)
            .collect();
        let n = 100_000i64;
        let shapes: [(&str, Vec<Option<i64>>); 4] = [
            ("unique", (0..n).map(Some).collect()),
            (
                "half key 0",
                (0..n)
                    .map(|i| Some(if i % 2 == 0 { 0 } else { i }))
                    .collect(),
            ),
            ("all key 0", vec![Some(0); 100_000]),
            ("10 keys", (0..n).map(|i| Some(i % 10)).collect()),
        ];
        for (name, keys) in shapes {
            let (table, rows) = table(keys);
            let slots = JoinSlots::build(&table, rows).unwrap();
            assert_eq!(
                slots.counts.iter().filter(|&&c| c > 0).count(),
                table.distinct_keys(),
                "{name}: one slot per distinct key"
            );
            let visits: u64 = probe
                .iter()
                .map(|&k| {
                    let (found, visits) = slots.probe(k);
                    assert_eq!(found, None);
                    u64::from(visits)
                })
                .sum();
            let mean = visits as f64 / probe.len() as f64;
            // Linear probing at load <= 1/2 expects 2.5 reads per miss.
            assert!(mean <= 2.5, "{name}: {mean:.1} slot reads per probe row");
        }
    }

    #[test]
    fn rows_outside_the_build_side_are_refused() {
        let (table, rows) = table(vec![Some(1), Some(2), Some(1)]);
        assert!(JoinSlots::build(&table, rows).is_ok());
        let err = JoinSlots::build(&table, rows - 1).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
    }

    #[test]
    fn an_empty_or_all_null_build_side_has_no_keys() {
        for keys in [vec![], vec![None, None]] {
            let (table, rows) = table(keys);
            let slots = JoinSlots::build(&table, rows).unwrap();
            assert_eq!(slots.distinct(), 0);
            assert_eq!(slots.capacity, 16);
            assert_eq!(slots.probe(0), (None, 1));
        }
    }
}
