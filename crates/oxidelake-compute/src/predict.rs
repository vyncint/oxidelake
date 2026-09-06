//! The `predict` scalar UDF: run a trained model over a column, in SQL.
//!
//! ```sql
//! SELECT id, predict('models/scorer.safetensors', features) AS score
//! FROM t ORDER BY score DESC LIMIT 10
//! ```
//!
//! OxideLake could already *search* embeddings — [`crate::GpuVectorDistanceExec`]
//! and the `l2_distance`/`cosine_distance` UDFs work over
//! `FixedSizeList<Float32>` — and could not *produce* a score from a model.
//! This is the other half: inference at the UDF layer, where a batch is a
//! column of feature vectors and the answer is a column of outputs.
//!
//! ## Why the UDF layer and not an operator
//!
//! [`crate::GpuFilterExec`] and friends work on Arrow batches with null
//! bitmaps and selection semantics behind OxideLake's own object-safe
//! `GpuBackend`. A tensor library's array is dense, strided and carries an
//! autograd tape; routing batches through one would cost a copy each way,
//! lose nulls, and replace kernels already conformance-tested against stock
//! DataFusion. A UDF is the seam where the two models genuinely meet: one
//! feature vector in, one output vector out, nulls handled by this file.
//!
//! ## What a model file is
//!
//! `safetensors` stores *named tensors*, not a graph, so the architecture has
//! to come from somewhere. Rather than invent a sidecar format, this reads the
//! naming convention `oxmera::nn::Sequential` already writes — `0.weight`,
//! `0.bias`, `1.weight`, … — and rebuilds a stack of `Linear` layers from the
//! shapes. `N.weight` of shape `[out, in]` is layer `N`.
//!
//! **The activation is the prototype's one real assumption**: ReLU between
//! layers, nothing after the last. That covers an MLP scorer and nothing else,
//! and a model whose activations differ will silently produce wrong numbers —
//! which is why [`ModelSpec`] exists as the place to put an explicit
//! description when this graduates. See the module tests for what is pinned.
//!
//! ## Cost model
//!
//! Models are loaded once per path and cached for the life of the process
//! (`MODELS`). Inference runs on the CPU: `oxmera` is depended on with
//! `default-features = false`, so no CUDA context is created and no
//! pre-`main` constructor runs — see the note in the workspace manifest.
//! Batching is per Arrow batch: one `[rows, in_features]` matmul chain rather
//! than one per row.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use arrow::array::{Array, FixedSizeListArray, Float32Array};
use arrow::datatypes::{DataType, Field};
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use oxmera::Tensor;
use oxmera::nn::{Linear, Module, Sequential};
use safetensors::SafeTensors;

/// Name of the inference UDF.
pub const PREDICT: &str = "predict";

/// How a model file is interpreted.
///
/// One variant today. It is an enum rather than a bare `load_mlp` so that the
/// activation assumption documented above has somewhere to go when a second
/// shape is needed: the model path stays the SQL surface, and what changes is
/// this description of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSpec {
    /// A stack of `Linear` layers named `0.weight`/`0.bias`, `1.weight`/…,
    /// ReLU between them and nothing after the last.
    SequentialMlpRelu,
}

/// A loaded model and the shapes it accepts.
///
/// No `Debug`: `oxmera::nn::Sequential` does not implement it (a module is a
/// vector of trait objects), and printing weights would be useless anyway.
/// The two shape fields are the part worth seeing, and they are public.
pub struct Model {
    module: Sequential,
    /// Columns the first layer's weight expects.
    pub in_features: usize,
    /// Rows the last layer's weight produces.
    pub out_features: usize,
}

impl Model {
    /// Run `rows x in_features` through the stack.
    fn forward(&self, input: &Tensor) -> oxmera::Result<Tensor> {
        self.module.forward(input)
    }
}

/// Process-wide model cache, keyed by path.
///
/// A UDF is invoked per batch, and a scan has many batches; reading and
/// parsing a weights file each time would dominate every query. The lock is
/// held across the load, so two batches racing on a cold path do the work
/// once. It is never invalidated: a model file that changes under a running
/// process keeps serving the old weights, which is the right default for a
/// query engine (a plan should not change answers halfway) and the wrong one
/// for a notebook, so it is stated rather than assumed.
static MODELS: OnceLock<Mutex<HashMap<String, Arc<Model>>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<String, Arc<Model>>> {
    MODELS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load `path`, or return the cached model for it.
///
/// # Errors
///
/// [`DataFusionError::Execution`] when the file is missing, is not
/// safetensors, or does not describe a shape [`ModelSpec`] recognizes.
pub fn model(path: &str, spec: ModelSpec) -> Result<Arc<Model>> {
    let mut guard = cache()
        .lock()
        .map_err(|_| DataFusionError::Execution(format!("{PREDICT}: model cache poisoned")))?;
    if let Some(found) = guard.get(path) {
        return Ok(Arc::clone(found));
    }
    let loaded = Arc::new(load(path, spec)?);
    guard.insert(path.to_string(), Arc::clone(&loaded));
    Ok(loaded)
}

/// Rebuild the architecture from the tensor names and shapes, then fill it.
fn load(path: &str, spec: ModelSpec) -> Result<Model> {
    let ModelSpec::SequentialMlpRelu = spec;
    let shapes = layer_shapes(path)?;
    // Destructured together so the "not empty" check happens once and the
    // compiler carries it, rather than a first()? followed by a last() that
    // is unreachable-but-fallible.
    let ([(first_in, _), ..], [.., (_, last_out)]) = (&shapes[..], &shapes[..]) else {
        return Err(exec(format!(
            "{PREDICT}: {path} declares no `N.weight` tensors, so there is no \
             architecture to rebuild"
        )));
    };
    let (first_in, last_out) = (*first_in, *last_out);

    let mut stack = Sequential::new();
    for (i, &(in_features, out_features)) in shapes.iter().enumerate() {
        // The seed is irrelevant — every value is about to be overwritten by
        // the file. Using the index keeps construction deterministic anyway,
        // so a load that silently fails to overwrite is reproducible rather
        // than random.
        stack = stack.push(Linear::new(in_features, out_features, i as u64));
    }
    oxmera::nn::serialize::load(&stack, path)
        .map_err(|e| exec(format!("{PREDICT}: loading {path}: {e}")))?;

    Ok(Model {
        module: stack,
        in_features: first_in,
        out_features: last_out,
    })
}

/// `(in_features, out_features)` per layer, in layer order.
///
/// Read from the file rather than configured: `N.weight` has shape
/// `[out, in]`, so the whole architecture is recoverable from the header.
/// Layers must be numbered `0..n` with no gaps — a gap means this is not the
/// `Sequential` layout, and guessing at that point would build a model that
/// loads and computes something else.
fn layer_shapes(path: &str) -> Result<Vec<(usize, usize)>> {
    let bytes = std::fs::read(path).map_err(|e| exec(format!("{PREDICT}: reading {path}: {e}")))?;
    let file = SafeTensors::deserialize(&bytes)
        .map_err(|e| exec(format!("{PREDICT}: {path} is not a safetensors file: {e}")))?;

    let mut by_index: HashMap<usize, (usize, usize)> = HashMap::new();
    for (name, view) in file.tensors() {
        let shape = view.shape();
        let Some((index, "weight")) = name.split_once('.') else {
            continue; // biases and anything else carry no shape information we need
        };
        let Ok(index) = index.parse::<usize>() else {
            continue;
        };
        let [out_features, in_features] = shape[..] else {
            return Err(exec(format!(
                "{PREDICT}: {path}: `{name}` has {} dimensions, expected 2",
                shape.len()
            )));
        };
        by_index.insert(index, (in_features, out_features));
    }

    let mut shapes = Vec::with_capacity(by_index.len());
    for i in 0..by_index.len() {
        let found = by_index.get(&i).ok_or_else(|| {
            exec(format!(
                "{PREDICT}: {path}: layers are numbered with a gap at {i}; \
                 expected 0..{} as `oxmera::nn::Sequential` writes them",
                by_index.len()
            ))
        })?;
        shapes.push(*found);
    }

    // Consecutive layers have to compose, or the file is not one stack.
    for pair in shapes.windows(2) {
        let [(_, out), (next_in, _)] = pair else {
            continue;
        };
        if out != next_in {
            return Err(exec(format!(
                "{PREDICT}: {path}: a layer produces {out} features and the next \
                 expects {next_in}"
            )));
        }
    }
    Ok(shapes)
}

fn exec(message: String) -> DataFusionError {
    DataFusionError::Execution(message)
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct PredictUdf {
    signature: Signature,
}

impl PredictUdf {
    fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for PredictUdf {
    fn name(&self) -> &str {
        PREDICT
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    /// `predict(model_path, features)`: a Utf8 literal and a list of numbers.
    ///
    /// The features are coerced to `FixedSizeList<Float32>` exactly as the
    /// distance UDFs do, so a column written as `List<Float64>` still works
    /// and the dimension is known before execution.
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [path, features] = arg_types else {
            return Err(DataFusionError::Plan(format!(
                "{PREDICT} takes 2 arguments (model path, features), got {}",
                arg_types.len()
            )));
        };
        if !matches!(
            path,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) {
            return Err(DataFusionError::Plan(format!(
                "{PREDICT}: the first argument is the model path as a string, got {path:?}"
            )));
        }
        let dim = match features {
            DataType::FixedSizeList(_, n) => *n,
            other => {
                return Err(DataFusionError::Plan(format!(
                    "{PREDICT}: features must be a FixedSizeList so the width is \
                     known before execution, got {other:?}"
                )));
            }
        };
        Ok(vec![
            DataType::Utf8,
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
        ])
    }

    /// A vector out, one element per output feature — the model's width is
    /// not known until the file is read, so this is a plain `List` rather
    /// than a `FixedSizeList`.
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Float32,
            true,
        ))))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [path, features] = args.args.as_slice() else {
            return Err(exec(format!(
                "{PREDICT} takes 2 arguments, got {}",
                args.args.len()
            )));
        };
        let path = model_path(path)?;
        let model = model(&path, ModelSpec::SequentialMlpRelu)?;
        let list = feature_list(features)?;
        let (values, dim) = feature_values(&list)?;
        if dim != model.in_features {
            return Err(exec(format!(
                "{PREDICT}: {path} expects {} features but the column has {dim}",
                model.in_features
            )));
        }

        // One matmul chain for the whole batch, not one per row. Null rows are
        // fed zeros and masked afterwards: the shapes stay rectangular, which
        // is what makes the batched path worth having, and no null ever
        // reaches the model as a number a user could mistake for data.
        let rows = args.number_rows;
        let mut flat = vec![0.0f32; rows * dim];
        let mut null = vec![false; rows];
        for row in 0..rows {
            if list.is_null(row) {
                null[row] = true;
                continue;
            }
            let start = usize::try_from(list.value_offset(row)).unwrap_or(0);
            flat[row * dim..(row + 1) * dim].copy_from_slice(&values[start..start + dim]);
        }

        let input = Tensor::from_vec_f32(flat, [rows, dim])
            .map_err(|e| exec(format!("{PREDICT}: building the input batch: {e}")))?;
        let out = oxmera::no_grad(|| model.forward(&input))
            .map_err(|e| exec(format!("{PREDICT}: {path}: {e}")))?;
        let out = out
            .to_vec_f32()
            .map_err(|e| exec(format!("{PREDICT}: reading the output: {e}")))?;

        let width = model.out_features;
        let mut builder = arrow::array::ListBuilder::new(Float32Array::builder(rows * width));
        for (row, is_null) in null.iter().enumerate() {
            if *is_null {
                builder.append_null();
                continue;
            }
            builder
                .values()
                .append_slice(&out[row * width..(row + 1) * width]);
            builder.append(true);
        }
        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

fn model_path(value: &ColumnarValue) -> Result<String> {
    match value {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) => Ok(s.clone()),
        // Deliberately refused rather than supported. A per-row model path
        // means a per-row model load, and the cache would grow without bound
        // on a column of distinct paths — a query that quietly consumes the
        // machine is worse than one that will not plan.
        _ => Err(DataFusionError::Plan(format!(
            "{PREDICT}: the model path must be a constant string, not a column"
        ))),
    }
}

fn feature_list(value: &ColumnarValue) -> Result<FixedSizeListArray> {
    match value {
        ColumnarValue::Array(array) => array
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .cloned()
            .ok_or_else(|| {
                exec(format!(
                    "{PREDICT}: expected FixedSizeList features, got {:?}",
                    array.data_type()
                ))
            }),
        ColumnarValue::Scalar(ScalarValue::FixedSizeList(array)) => Ok(array.as_ref().clone()),
        ColumnarValue::Scalar(other) => Err(exec(format!(
            "{PREDICT}: expected FixedSizeList features, got {:?}",
            other.data_type()
        ))),
    }
}

fn feature_values(list: &FixedSizeListArray) -> Result<(&[f32], usize)> {
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            exec(format!(
                "{PREDICT}: feature elements are {:?}, expected Float32",
                list.value_type()
            ))
        })?;
    let dim = usize::try_from(list.value_length())
        .map_err(|_| exec(format!("{PREDICT}: negative list size")))?;
    Ok((values.values(), dim))
}

/// The `predict` scalar UDF.
pub fn predict_udf() -> ScalarUDF {
    ScalarUDF::from(PredictUdf::new())
}
