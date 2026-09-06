//! `predict` end to end: train-shaped weights on disk, a SQL query over a
//! Parquet-shaped batch, and the answer checked against oxmera's own forward
//! pass rather than against a number someone wrote down.
//!
//! The reference is the point. A UDF that computes *something* plausible is
//! the failure mode here — the architecture is inferred from tensor names, so
//! a wrong inference produces confident wrong numbers rather than an error.
#![cfg(feature = "predict")]
// The house convention for test files (see conformance.rs and siblings): a
// failed setup should be a loud panic with a reason, not a `?` that turns
// into an opaque test-harness error.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, Int32Array, ListArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use oxidelake_compute::predict_udf;
use oxmera::Tensor;
use oxmera::nn::{Linear, Module, Sequential};

const IN: usize = 4;
const HIDDEN: usize = 3;
const OUT: usize = 2;

/// A two-layer MLP with fixed seeds, saved where the test can point SQL at it.
fn model_on_disk(dir: &std::path::Path) -> (Sequential, String) {
    let model = Sequential::new()
        .push(Linear::new(IN, HIDDEN, 11))
        .push(Linear::new(HIDDEN, OUT, 22));
    let path = dir.join("scorer.safetensors");
    oxmera::nn::serialize::save(&model, &path).expect("save");
    (model, path.to_string_lossy().into_owned())
}

fn features() -> Vec<Option<[f32; IN]>> {
    vec![
        Some([1.0, 0.0, -2.0, 0.5]),
        Some([0.25, 0.25, 0.25, 0.25]),
        None, // a null row: the model must never see it, and it must survive
        Some([-1.0, 3.0, 0.0, 2.0]),
    ]
}

fn batch() -> RecordBatch {
    let rows = features();
    let mut values = Vec::new();
    let mut validity = Vec::new();
    for row in &rows {
        match row {
            Some(v) => {
                values.extend_from_slice(v);
                validity.push(true);
            }
            None => {
                values.extend(std::iter::repeat_n(0.0, IN));
                validity.push(false);
            }
        }
    }
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let list = FixedSizeListArray::new(
        Arc::clone(&field),
        IN as i32,
        Arc::new(Float32Array::from(values)),
        Some(validity.into()),
    );
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("features", DataType::FixedSizeList(field, IN as i32), true),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from((0..rows.len() as i32).collect::<Vec<_>>())),
            Arc::new(list),
        ],
    )
    .expect("batch")
}

async fn run(sql: &str, path: &str) -> Vec<RecordBatch> {
    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).expect("register");
    ctx.sql(&sql.replace("{model}", path))
        .await
        .expect("plan")
        .collect()
        .await
        .expect("execute")
}

#[tokio::test]
async fn predict_agrees_with_the_model_it_loaded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (model, path) = model_on_disk(dir.path());

    let out = run(
        "SELECT id, predict('{model}', features) AS y FROM t ORDER BY id",
        &path,
    )
    .await;
    let batch = &out[0];
    let y = batch
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("predict returns a list");

    for (row, expected_in) in features().iter().enumerate() {
        let Some(input) = expected_in else {
            assert!(
                y.is_null(row),
                "row {row}: a null feature vector must stay null"
            );
            continue;
        };
        assert!(
            !y.is_null(row),
            "row {row}: a real feature vector must produce a value"
        );

        // The reference: the same model, called directly.
        let want = oxmera::no_grad(|| {
            model.forward(&Tensor::from_vec_f32(input.to_vec(), [1, IN]).unwrap())
        })
        .expect("reference forward")
        .to_vec_f32()
        .expect("reference values");

        let got = y.value(row);
        let got = got
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("float output");
        assert_eq!(got.len(), OUT, "row {row}: output width");
        for (i, expected) in want.iter().enumerate().take(OUT) {
            assert!(
                (got.value(i) - expected).abs() < 1e-5,
                "row {row}, output {i}: SQL gave {} and the model gives {expected}",
                got.value(i),
            );
        }
    }
}

/// The batched path has to agree with the row-at-a-time path.
///
/// The UDF builds one `[rows, in]` matmul for the whole batch and masks nulls
/// afterwards. That is only sound if a null row's zeros cannot leak into a
/// real row's answer — which they cannot for a matmul, but the whole point of
/// the null column above is that this is asserted rather than reasoned about.
#[tokio::test]
async fn a_single_row_gives_the_same_answer_as_the_batch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_, path) = model_on_disk(dir.path());

    let whole = run(
        "SELECT predict('{model}', features) AS y FROM t ORDER BY id",
        &path,
    )
    .await;
    let one = run(
        "SELECT predict('{model}', features) AS y FROM t WHERE id = 3",
        &path,
    )
    .await;

    let from_batch = whole[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let alone = one[0]
        .column(0)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let a = from_batch.value(3);
    let b = alone.value(0);
    assert_eq!(
        a.as_any().downcast_ref::<Float32Array>().unwrap().values(),
        b.as_any().downcast_ref::<Float32Array>().unwrap().values(),
        "row 3 read differently in a batch of four than on its own"
    );
}

#[tokio::test]
async fn a_missing_model_is_an_error_naming_the_path() {
    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).unwrap();
    let err = ctx
        .sql("SELECT predict('/nonexistent/model.safetensors', features) FROM t")
        .await
        .expect("plans fine — the file is only read at execution")
        .collect()
        .await
        .expect_err("executing must fail");
    let msg = err.to_string();
    assert!(msg.contains("predict"), "{msg}");
    assert!(msg.contains("/nonexistent/model.safetensors"), "{msg}");
}

#[tokio::test]
async fn a_feature_width_the_model_does_not_accept_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A model that wants 7 features, against the 4-wide column.
    let wrong = Sequential::new().push(Linear::new(7, OUT, 1));
    let path = dir.path().join("wrong.safetensors");
    oxmera::nn::serialize::save(&wrong, &path).unwrap();

    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).unwrap();
    let err = ctx
        .sql(&format!(
            "SELECT predict('{}', features) FROM t",
            path.to_string_lossy()
        ))
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("4 features into a 7-feature model must fail");
    let msg = err.to_string();
    assert!(msg.contains("expects 7 features"), "{msg}");
    assert!(msg.contains("has 4"), "{msg}");
}

/// A per-row model path is refused at planning, not discovered at scale.
#[tokio::test]
async fn a_column_of_model_paths_is_refused() {
    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).unwrap();
    let result = ctx
        .sql("SELECT predict(CAST(id AS VARCHAR), features) FROM t")
        .await;
    let err = match result {
        Err(e) => e.to_string(),
        Ok(df) => df
            .collect()
            .await
            .expect_err("a column of paths must not run")
            .to_string(),
    };
    assert!(err.contains("constant string"), "{err}");
}
