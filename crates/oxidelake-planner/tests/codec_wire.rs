//! The plan codec's wire format, pinned (#42).
//!
//! `postcard` is not self-describing: a field added to `GpuNode`, or two
//! fields of the same width reordered, decodes without complaint into the
//! wrong parameters. A mixed-build fleet then does not fail — it computes a
//! confident wrong answer from a plausible-looking plan.
//!
//! Two things guard that. At run time the header carries a fingerprint
//! derived from the crate version and the compute layer's feature mask, so a
//! plan from a different build is refused at decode. At review time this
//! snapshot is the other half: changing `GpuNode`'s encoding moves a
//! committed file, so it cannot land unnoticed, and whoever moves it has to
//! say why in the same pull request.
//!
//! If this test fails and the change to `GpuNode` was deliberate: bump
//! `VERSION` in `codec.rs` and accept the snapshot with `cargo insta review`
//! — never blind-accept it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use oxidelake_core::BackendKind;
use oxidelake_core::params::{
    AggregateFunction, AggregateSpec, Comparison, DistanceMetric, Literal, Predicate,
};
use oxidelake_planner::codec::GpuNode;

/// Every variant, with values chosen so a reordering of same-width fields
/// shows up: no two numeric fields share a value.
fn every_variant() -> Vec<(&'static str, GpuNode)> {
    vec![
        (
            "filter",
            GpuNode::Filter {
                target: BackendKind::Cuda,
                predicate: Predicate::Compare {
                    column: 3,
                    op: Comparison::Gt,
                    literal: Literal::Int64(17),
                },
                projection: vec![5, 8],
            },
        ),
        (
            "hash_join",
            GpuNode::HashJoin {
                target: BackendKind::Metal,
                left_key: 1,
                right_key: 2,
                projection: Some(vec![4, 6, 9]),
            },
        ),
        (
            "aggregate",
            GpuNode::Aggregate {
                target: BackendKind::CpuSimd,
                spec: AggregateSpec {
                    group_by: 7,
                    aggregates: vec![(AggregateFunction::Sum, 11)],
                },
                output: vec![("k".to_owned(), false), ("sv".to_owned(), true)],
            },
        ),
        (
            "vector_distance",
            GpuNode::VectorDistance {
                target: BackendKind::Cuda,
                column: 13,
                query: vec![0.5, 0.25],
                metric: DistanceMetric::Cosine,
                output_name: "dist".to_owned(),
            },
        ),
    ]
}

/// The bytes each variant encodes to, as hex. A diff here is a wire-format
/// change; there is no other reason for these to move.
#[test]
fn every_gpu_node_variant_has_a_pinned_encoding() {
    let mut rendered = String::new();
    for (name, node) in every_variant() {
        let bytes = postcard::to_allocvec(&node).expect("a GpuNode encodes");
        let hex: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
        rendered.push_str(&format!("{name}: {}\n", hex.join(" ")));
    }
    insta::assert_snapshot!("gpu_node_encodings", rendered);
}

/// A round trip through the encoding every variant survives, so the snapshot
/// above pins something that is also correct rather than merely stable.
#[test]
fn every_variant_round_trips() {
    for (name, node) in every_variant() {
        let bytes = postcard::to_allocvec(&node).expect("encodes");
        let back: GpuNode = postcard::from_bytes(&bytes).expect("decodes");
        assert_eq!(back, node, "{name} did not survive a round trip");
    }
}
