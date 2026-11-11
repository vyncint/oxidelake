//! `predict` end to end: train-shaped weights on disk, a SQL query over a
//! Parquet-shaped batch, and the answer checked against oxmera's own forward
//! pass rather than against a number someone wrote down.
//!
//! The reference is the point. A UDF that computes *something* plausible is
//! the failure mode here — the architecture is inferred from tensor names, so
//! a wrong inference produces confident wrong numbers rather than an error.
//! That is not hypothetical: before 0.2.0 the loader built bare `Linear`
//! layers while every document said "ReLU between them", and the only test of
//! the numbers compared them against the same bare stack. So the activation
//! is now read from the file's `__metadata__` (#47), and these tests check it
//! against a reference that does *not* share the loader's loop: the `none`
//! case is compared with oxmera's own `Sequential::forward`, and `relu` and
//! `gelu` must disagree with it and with each other.
#![cfg(feature = "predict")]
// The house convention for test files (see conformance.rs and siblings): a
// failed setup should be a loud panic with a reason, not a `?` that turns
// into an opaque test-harness error.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, Int32Array, ListArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use oxidelake_compute::{ACTIVATION_KEY, predict_udf};
use oxmera::Tensor;
use oxmera::nn::{Linear, Module, Sequential};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};

const IN: usize = 4;
const HIDDEN: usize = 3;
const OUT: usize = 2;

fn two_layer_mlp() -> Sequential {
    Sequential::new()
        .push(Linear::new(IN, HIDDEN, 11))
        .push(Linear::new(HIDDEN, OUT, 22))
}

/// Saves `model` at `path`, adding `oxidelake.activation` to the safetensors
/// `__metadata__` when `activation` is given.
///
/// Written here rather than taken from a helper because `oxmera`'s
/// `nn::serialize::save` writes no metadata (0.5): the file is saved, read
/// back, and re-serialized with the header this engine asks for. That is
/// exactly what a user of a third-party checkpoint has to do, so the test
/// doing it is the documentation of it working.
fn save_with_activation(model: &Sequential, path: &Path, activation: Option<&str>) {
    oxmera::nn::serialize::save(model, path).expect("save");
    let Some(activation) = activation else { return };
    let bytes = std::fs::read(path).expect("read back");
    let file = SafeTensors::deserialize(&bytes).expect("deserialize");
    let tensors: Vec<_> = file.tensors();
    let metadata = HashMap::from([(ACTIVATION_KEY.to_owned(), activation.to_owned())]);
    let out = safetensors::serialize(
        tensors.iter().map(|(name, view)| (name.clone(), view)),
        Some(metadata),
    )
    .expect("serialize with metadata");
    std::fs::write(path, out).expect("write");
}

/// A two-layer MLP with fixed seeds, declaring `activation`, saved where the
/// test can point SQL at it.
fn model_on_disk_with(dir: &Path, name: &str, activation: Option<&str>) -> (Sequential, String) {
    let model = two_layer_mlp();
    let path = dir.join(name);
    save_with_activation(&model, &path, activation);
    (model, path.to_string_lossy().into_owned())
}

/// The default fixture: a ReLU model.
fn model_on_disk(dir: &Path) -> (Sequential, String) {
    model_on_disk_with(dir, "scorer.safetensors", Some("relu"))
}

/// The expected output for one row, computed by stepping the layers and
/// applying `activation` between them — never after the last.
fn reference(model: &Sequential, input: &[f32; IN], activation: &str) -> Vec<f32> {
    oxmera::no_grad(|| -> oxmera::Result<Tensor> {
        let mut x = Tensor::from_vec_f32(input.to_vec(), [1, IN])?;
        let last = model.len() - 1;
        for (i, layer) in model.iter().enumerate() {
            x = layer.forward(&x)?;
            if i != last {
                x = match activation {
                    "relu" => x.relu()?,
                    "gelu" => x.gelu()?,
                    "sigmoid" => x.sigmoid()?,
                    "tanh" => x.tanh()?,
                    "none" => x,
                    other => panic!("the test does not model {other}"),
                };
            }
        }
        Ok(x)
    })
    .expect("reference forward")
    .to_vec_f32()
    .expect("reference values")
}

/// The `y` column of the first batch, as a list array.
fn outputs(out: &[RecordBatch], column: usize) -> ListArray {
    out[0]
        .column(column)
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("predict returns a list")
        .clone()
}

fn row_values(list: &ListArray, row: usize) -> Vec<f32> {
    let value = list.value(row);
    value
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("float output")
        .values()
        .to_vec()
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
    let y = outputs(&out, 1);

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

        let want = reference(&model, input, "relu");
        let got = row_values(&y, row);
        assert_eq!(got.len(), OUT, "row {row}: output width");
        for (i, expected) in want.iter().enumerate().take(OUT) {
            assert!(
                (got[i] - expected).abs() < 1e-5,
                "row {row}, output {i}: SQL gave {} and the model gives {expected}",
                got[i],
            );
        }
    }
}

/// `none` is the plain affine stack, so oxmera's own `Sequential::forward` —
/// which knows nothing about this crate's loop — is the reference.
#[tokio::test]
async fn an_activation_of_none_is_oxmeras_own_sequential() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (model, path) = model_on_disk_with(dir.path(), "linear.safetensors", Some("none"));

    let out = run(
        "SELECT predict('{model}', features) AS y FROM t ORDER BY id",
        &path,
    )
    .await;
    let y = outputs(&out, 0);

    for (row, input) in features().iter().enumerate() {
        let Some(input) = input else { continue };
        let want = oxmera::no_grad(|| {
            model.forward(&Tensor::from_vec_f32(input.to_vec(), [1, IN]).unwrap())
        })
        .expect("reference forward")
        .to_vec_f32()
        .expect("reference values");
        for (i, expected) in want.iter().enumerate().take(OUT) {
            assert!(
                (row_values(&y, row)[i] - expected).abs() < 1e-5,
                "row {row}, output {i}"
            );
        }
    }
}

/// The same weights under three activations must give three answers. This is
/// what a stubbed or ignored activation fails: it would make all three equal.
#[tokio::test]
async fn the_declared_activation_changes_the_numbers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut answers = Vec::new();
    for activation in ["relu", "gelu", "none"] {
        let (model, path) = model_on_disk_with(
            dir.path(),
            &format!("{activation}.safetensors"),
            Some(activation),
        );
        let out = run(
            "SELECT predict('{model}', features) AS y FROM t ORDER BY id",
            &path,
        )
        .await;
        let got = row_values(&outputs(&out, 0), 3);
        let want = reference(
            &model,
            &features()[3].expect("row 3 is not null"),
            activation,
        );
        for (i, expected) in want.iter().enumerate().take(OUT) {
            assert!(
                (got[i] - expected).abs() < 1e-5,
                "{activation}, output {i}: SQL gave {} and the model gives {expected}",
                got[i]
            );
        }
        answers.push((activation, got));
    }
    for pair in answers.windows(2) {
        let [(a, x), (b, y)] = pair else { continue };
        assert_ne!(x, y, "{a} and {b} produced identical output");
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
    save_with_activation(&wrong, &path, Some("relu"));

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

// --- the activation contract (#47) -----------------------------------------

/// A model that declares nothing is refused, and the message says both ways
/// of fixing it. The old behaviour — load it and apply no activation at all —
/// is the one outcome that must not happen.
#[tokio::test]
async fn a_model_without_the_activation_key_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_, path) = model_on_disk_with(dir.path(), "bare.safetensors", None);

    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).unwrap();
    let err = ctx
        .sql(&format!("SELECT predict('{path}', features) FROM t"))
        .await
        .expect("plans: the file is only read at execution")
        .collect()
        .await
        .expect_err("a header-less model must not run");
    let msg = err.to_string();
    assert!(msg.contains(ACTIVATION_KEY), "{msg}");
    assert!(msg.contains("__metadata__"), "{msg}");
    assert!(msg.contains("features, 'relu'"), "{msg}");
    assert!(msg.contains("relu, gelu, sigmoid, tanh, none"), "{msg}");
}

/// The third argument is how a checkpoint someone else wrote is used.
#[tokio::test]
async fn the_query_may_state_the_activation_the_file_does_not() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (model, path) = model_on_disk_with(dir.path(), "stated.safetensors", None);

    let out = run(
        "SELECT predict('{model}', features, 'gelu') AS y FROM t ORDER BY id",
        &path,
    )
    .await;
    let want = reference(&model, &features()[0].expect("row 0"), "gelu");
    let got = row_values(&outputs(&out, 0), 0);
    for (i, expected) in want.iter().enumerate().take(OUT) {
        assert!((got[i] - expected).abs() < 1e-5, "output {i}");
    }
}

/// When both say something, they must say the same thing: silently letting
/// the query override the file would be the original bug with extra steps.
#[tokio::test]
async fn a_query_that_contradicts_the_file_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_, path) = model_on_disk_with(dir.path(), "relu-model.safetensors", Some("relu"));

    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).unwrap();
    let err = ctx
        .sql(&format!(
            "SELECT predict('{path}', features, 'gelu') FROM t"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("a contradiction must not run");
    let msg = err.to_string();
    assert!(msg.contains("relu"), "{msg}");
    assert!(msg.contains("gelu"), "{msg}");
}

/// Agreeing with the file is fine, and gives the file's answer.
#[tokio::test]
async fn a_query_that_repeats_the_file_is_accepted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (model, path) = model_on_disk_with(dir.path(), "agree.safetensors", Some("relu"));
    let out = run(
        "SELECT predict('{model}', features, 'relu') AS y FROM t ORDER BY id",
        &path,
    )
    .await;
    let want = reference(&model, &features()[0].expect("row 0"), "relu");
    let got = row_values(&outputs(&out, 0), 0);
    for (i, expected) in want.iter().enumerate().take(OUT) {
        assert!((got[i] - expected).abs() < 1e-5, "output {i}");
    }
}

#[tokio::test]
async fn an_unknown_activation_names_the_ones_that_exist() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_, path) = model_on_disk_with(dir.path(), "unknown.safetensors", None);

    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).unwrap();
    let err = ctx
        .sql(&format!(
            "SELECT predict('{path}', features, 'swish') FROM t"
        ))
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("an unknown activation must not run");
    let msg = err.to_string();
    assert!(msg.contains("swish"), "{msg}");
    assert!(msg.contains("relu, gelu, sigmoid, tanh, none"), "{msg}");
}

/// A file whose header says something this engine does not know is refused
/// with the file named — the alternative is ignoring the key and computing
/// the wrong model, which is what the key exists to prevent.
#[tokio::test]
async fn an_unknown_activation_in_the_file_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_, path) = model_on_disk_with(dir.path(), "swish.safetensors", Some("swish"));

    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).unwrap();
    let err = ctx
        .sql(&format!("SELECT predict('{path}', features) FROM t"))
        .await
        .unwrap()
        .collect()
        .await
        .expect_err("an unknown declared activation must not run");
    let msg = err.to_string();
    assert!(msg.contains("swish.safetensors"), "{msg}");
    assert!(msg.contains(ACTIVATION_KEY), "{msg}");
}

#[tokio::test]
async fn a_column_of_activations_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_, path) = model_on_disk_with(dir.path(), "col.safetensors", Some("relu"));

    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).unwrap();
    let result = ctx
        .sql(&format!(
            "SELECT predict('{path}', features, CAST(id AS VARCHAR)) FROM t"
        ))
        .await;
    let err = match result {
        Err(e) => e.to_string(),
        Ok(df) => df
            .collect()
            .await
            .expect_err("a column of activations must not run")
            .to_string(),
    };
    assert!(err.contains("constant string"), "{err}");
}

#[tokio::test]
async fn four_arguments_are_refused_naming_the_shape() {
    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).unwrap();
    let result = ctx
        .sql("SELECT predict('m', features, 'relu', 'extra') FROM t")
        .await;
    let err = match result {
        Err(e) => e.to_string(),
        Ok(df) => df.collect().await.expect_err("four arguments").to_string(),
    };
    assert!(err.contains("2 or 3 arguments"), "{err}");
}

// --- a malformed model file (#96) ------------------------------------------

/// `run`, with the error kept as a value rather than a panic.
///
/// Owned arguments so it can be spawned: a panic inside the UDF then fails
/// that one task, and the test still reaches the assertions that say which
/// guard gave way.
async fn try_run(sql: String, path: String) -> Result<Vec<RecordBatch>, String> {
    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch())
        .map_err(|e| e.to_string())?;
    let frame = ctx
        .sql(&sql.replace("{model}", &path))
        .await
        .map_err(|e| e.to_string())?;
    frame.collect().await.map_err(|e| e.to_string())
}

/// A model file whose one layer is `0.weight: [rows, 0]` and nothing else.
///
/// Zero elements means zero bytes of data, so safetensors accepts the file
/// whatever `rows` is — the reviewer's was 136 bytes long. What made it
/// dangerous is that `Linear::new(0, rows)` sizes a bias of `rows` f32s
/// before a single value is read from the file.
fn hollow_model(dir: &Path, name: &str, rows: usize) -> String {
    let path = dir.join(name);
    let view = TensorView::new(Dtype::F32, vec![rows, 0], &[]).expect("a zero-element view");
    let metadata = HashMap::from([(ACTIVATION_KEY.to_owned(), "none".to_owned())]);
    let bytes =
        safetensors::serialize([("0.weight".to_owned(), view)], Some(metadata)).expect("serialize");
    assert!(bytes.len() < 256, "the file is tiny: {} bytes", bytes.len());
    std::fs::write(&path, bytes).expect("write");
    path.to_string_lossy().into_owned()
}

const ONE_ROW: &str = "SELECT predict('{model}', features) AS y FROM t WHERE id = 0";

/// Runs a hollow model, then the same path once the file is fixed, then a
/// second valid model, and checks all three.
///
/// The order of the assertions is the point. "The path loads once the file
/// is fixed" and "another model still loads" come first, because they are
/// what a poisoned cache breaks — every later `predict` in the process
/// answered `model cache poisoned` until a restart. "The malformed file is a
/// clean error" comes last: it is what the shape check breaks, and checking
/// it first would hide whether the cache survived.
async fn malformed_then_valid(dir: &Path, rows: usize) {
    let hollow = hollow_model(dir, &format!("hollow-{rows}.safetensors"), rows);
    let first = tokio::spawn(try_run(ONE_ROW.to_owned(), hollow.clone())).await;

    let fixed = two_layer_mlp();
    save_with_activation(&fixed, Path::new(&hollow), Some("none"));
    let again = try_run(ONE_ROW.to_owned(), hollow.clone()).await;
    let (other, other_path) = model_on_disk_with(dir, "other.safetensors", Some("relu"));
    let elsewhere = try_run(ONE_ROW.to_owned(), other_path).await;

    let input = features()[0].expect("row 0 is not null");
    for (what, result, model, activation) in [
        (
            "the same path, once the file is fixed",
            &again,
            &fixed,
            "none",
        ),
        ("another model", &elsewhere, &other, "relu"),
    ] {
        let out = result
            .as_ref()
            .unwrap_or_else(|e| panic!("{what} must load after a malformed file: {e}"));
        let got = row_values(&outputs(out, 0), 0);
        let want = reference(model, &input, activation);
        for (i, expected) in want.iter().enumerate() {
            assert!((got[i] - expected).abs() < 1e-5, "{what}, output {i}");
        }
    }

    let err = match first {
        Ok(Err(e)) => e,
        Ok(Ok(_)) => panic!("a model with [{rows}, 0] weights and no data ran"),
        Err(e) => panic!("a malformed model file must be an error, not a panic: {e}"),
    };
    assert!(err.contains(&hollow), "the error names the file: {err}");
    assert!(err.contains("cannot fill"), "{err}");
}

/// `[2^62, 0]`: the bias's byte count overflows `usize`, so nothing can be
/// allocated even if the bound regresses — which is why this case runs
/// in-process, on every platform.
#[tokio::test]
async fn a_layer_whose_size_overflows_is_an_error_and_predict_keeps_working() {
    let dir = tempfile::tempdir().expect("tempdir");
    malformed_then_valid(dir.path(), 1 << 62).await;
}

/// Set in the child [`a_layer_larger_than_its_file_is_refused_before_allocating`]
/// spawns, which is the process that may run the malformed load.
#[cfg(target_os = "linux")]
const CAPPED_CHILD: &str = "OXIDELAKE_TEST_CAPPED_CHILD";

/// `[2^40, 0]`: a 4 TiB bias, which overflows nothing. Only a bound tied to
/// the file's own size refuses it, so this is the case that proves there is
/// one.
///
/// Run in a child whose address space is capped at 4 GiB, never in this
/// process. If the bound ever regresses, `Linear::new` reserves 4 TiB and
/// then writes zeros into it; on a host with `vm.overcommit_memory = 1` the
/// reservation succeeds and the writes take the machine down. Inside the cap
/// the reservation fails instead, and the test reports it. Linux only:
/// macOS does not enforce `RLIMIT_AS`, so there is no cap to run it under.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_layer_larger_than_its_file_is_refused_before_allocating() {
    const NAME: &str = "a_layer_larger_than_its_file_is_refused_before_allocating";
    if std::env::var_os(CAPPED_CHILD).is_some() {
        let dir = tempfile::tempdir().expect("tempdir");
        malformed_then_valid(dir.path(), 1 << 40).await;
        return;
    }
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new("/bin/sh")
        .arg("-c")
        // `ulimit -v` counts KiB: 4 GiB, ample for one test and a thousand
        // times less than the bias.
        .arg(r#"ulimit -v 4194304 && exec "$0" "$@""#)
        .arg(exe)
        .args([NAME, "--exact", "--test-threads=1"])
        .env(CAPPED_CHILD, "1")
        .output()
        .expect("spawn the capped child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "the capped child failed ({}):\n{stdout}\n{stderr}",
        out.status
    );
    // A filter that matched nothing would pass too.
    assert!(
        stdout.contains("1 passed"),
        "the child ran no test:\n{stdout}"
    );
}

// --- NULL elements (#97) ----------------------------------------------------

/// `id` and `features` from rows whose elements may each be NULL, as well as
/// the row itself.
fn batch_with_elements(rows: &[Option<[Option<f32>; IN]>]) -> RecordBatch {
    let mut values = Vec::new();
    let mut validity = Vec::new();
    for row in rows {
        match row {
            Some(elements) => {
                values.extend_from_slice(elements);
                validity.push(true);
            }
            None => {
                values.extend(std::iter::repeat_n(Some(0.0), IN));
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

/// A NULL element is as missing as a NULL row (#97).
///
/// Under a NULL element the values buffer holds whatever the builder left
/// there — 0.0 from every Arrow builder — and the model used to read it as
/// data: `[-1.0, NULL, 0.0, 2.0]` scored exactly like `[-1.0, 0.0, 0.0, 2.0]`.
/// The rows after the NULL ones are checked too, so masking a row cannot
/// shift where the next one is read from.
#[tokio::test]
async fn a_null_element_gives_a_null_output() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (model, path) = model_on_disk(dir.path());
    let rows = [
        Some([Some(-1.0), Some(0.0), Some(0.0), Some(2.0)]),
        Some([Some(-1.0), None, Some(0.0), Some(2.0)]),
        None,
        Some([None, None, None, None]),
        Some([Some(-1.0), Some(3.0), Some(0.0), Some(2.0)]),
    ];
    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch_with_elements(&rows))
        .expect("register");
    let out = ctx
        .sql(&format!(
            "SELECT predict('{path}', features) AS y FROM t ORDER BY id"
        ))
        .await
        .expect("plan")
        .collect()
        .await
        .expect("execute");
    let y = outputs(&out, 0);
    assert_eq!(y.len(), rows.len());

    for (row, input) in rows.iter().enumerate() {
        let complete = input.and_then(|elements| {
            let mut values = [0.0; IN];
            for (value, element) in values.iter_mut().zip(elements) {
                *value = element?;
            }
            Some(values)
        });
        let Some(values) = complete else {
            assert!(
                y.is_null(row),
                "row {row} ({input:?}) has a NULL in it, so its output is NULL, got {:?}",
                row_values(&y, row)
            );
            continue;
        };
        assert!(!y.is_null(row), "row {row} is complete and must score");
        let want = reference(&model, &values, "relu");
        let got = row_values(&y, row);
        for (i, expected) in want.iter().enumerate() {
            assert!(
                (got[i] - expected).abs() < 1e-5,
                "row {row}, output {i}: SQL gave {} and the model gives {expected}",
                got[i]
            );
        }
    }
}

// --- message text (#98) -----------------------------------------------------

/// The error `sql` fails with, whether planning or execution refuses it.
async fn error_of(sql: &str) -> String {
    let ctx = SessionContext::new();
    ctx.register_udf(predict_udf());
    ctx.register_batch("t", batch()).expect("register");
    match ctx.sql(sql).await {
        Err(e) => e.to_string(),
        Ok(df) => df.collect().await.expect_err(sql).to_string(),
    }
}

/// Each of these four messages used to carry the source's indentation, runs
/// of 14 to 22 spaces, where a string literal had lost its `\` line
/// continuations (#98). Every test above matches a substring, and passed.
#[tokio::test]
async fn messages_read_as_written_with_no_runs_of_spaces() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_, declared) = model_on_disk_with(dir.path(), "declared.safetensors", Some("relu"));
    let (_, bare) = model_on_disk_with(dir.path(), "bare.safetensors", None);
    let messages = [
        error_of(&format!(
            "SELECT predict('{declared}', features, 'gelu') FROM t"
        ))
        .await,
        error_of(&format!("SELECT predict('{bare}', features) FROM t")).await,
        error_of("SELECT predict('m', features, 'relu', 'extra') FROM t").await,
        error_of("SELECT predict('m', features, 1) FROM t").await,
    ];
    for message in &messages {
        assert!(!message.contains("  "), "a run of spaces in: {message}");
    }
    // One pinned whole, so the sentence itself is under test and not only
    // its whitespace.
    let contradiction = format!(
        "predict: {declared} declares `{ACTIVATION_KEY}: relu` but the query asks for \
         'gelu'; the model was trained with one of them, so change the query or the \
         file rather than running the other"
    );
    assert!(
        messages[0].contains(&contradiction),
        "expected\n  {contradiction}\nin\n  {}",
        messages[0]
    );
}
