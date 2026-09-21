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
//! **The activation comes from the file, never from a guess** (#47). A
//! `safetensors` header can carry free-form metadata, so a model declares its
//! own activation there:
//!
//! ```json
//! {"__metadata__": {"oxidelake.activation": "relu"}}
//! ```
//!
//! and a file without that key is refused rather than assumed. The activation
//! is applied *between* the `Linear` layers and never after the last, so the
//! output stays logits. A query may state the activation itself —
//! `predict(path, features, 'relu')` — which is how a file someone else wrote
//! is used; when the file also declares one, the two must agree.
//!
//! This is the difference between an honest error and a confident wrong
//! number: until 0.2.0 the loader built a stack of bare `Linear` layers and
//! documented ReLU between them, so every non-linear model it was pointed at
//! returned the answer of a linear one. See the module tests for what is
//! pinned.
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
use oxmera::nn::{Linear, Sequential};
use safetensors::SafeTensors;

/// Name of the inference UDF.
pub const PREDICT: &str = "predict";

/// The safetensors `__metadata__` key a model declares its activation under.
pub const ACTIVATION_KEY: &str = "oxidelake.activation";

/// The activation applied between a model's `Linear` layers.
///
/// Never after the last one: the output of `predict` is logits, which is what
/// a caller composes with `[1]`, a threshold or a sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Activation {
    /// `max(0, x)`.
    Relu,
    /// The Gaussian error linear unit.
    Gelu,
    /// The logistic function.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// No activation: the stack is one affine map. Spelled out rather than
    /// left to a missing key, so a linear model is a statement and not an
    /// omission.
    None,
}

impl Activation {
    /// Every activation, in the order the error messages list them.
    pub const ALL: [Activation; 5] = [
        Activation::Relu,
        Activation::Gelu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::None,
    ];

    /// The lowercase name used in `__metadata__` and in SQL.
    pub const fn as_str(self) -> &'static str {
        match self {
            Activation::Relu => "relu",
            Activation::Gelu => "gelu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::None => "none",
        }
    }

    /// The names this accepts, for an error that tells the reader what to write.
    fn names() -> String {
        Self::ALL
            .iter()
            .map(|a| a.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn parse(name: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|a| a.as_str() == name.trim().to_ascii_lowercase())
            .ok_or_else(|| {
                exec(format!(
                    "{PREDICT}: unknown activation '{name}'; expected one of {}",
                    Self::names()
                ))
            })
    }

    fn apply(self, x: &Tensor) -> oxmera::Result<Tensor> {
        match self {
            Activation::Relu => x.relu(),
            Activation::Gelu => x.gelu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Tanh => x.tanh(),
            Activation::None => Ok(x.clone()),
        }
    }
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a model file is interpreted.
///
/// One shape today. It is an enum rather than a bare `load_mlp` so that a
/// second architecture has somewhere to go: the model path stays the SQL
/// surface, and what changes is this description of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelSpec {
    /// A stack of `Linear` layers named `0.weight`/`0.bias`, `1.weight`/…,
    /// with an activation between them and nothing after the last.
    SequentialMlp {
        /// The activation the caller asserts. `None` means "take it from the
        /// file's `__metadata__`", which is the only way that does not guess.
        /// When both are present they must agree — a query that states the
        /// wrong activation for a file is a mistake worth an error, not a
        /// silent override of what the model was trained with.
        activation: Option<Activation>,
    },
}

impl ModelSpec {
    /// The spec that reads everything, including the activation, from the file.
    pub const FROM_FILE: ModelSpec = ModelSpec::SequentialMlp { activation: None };
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
    /// The activation between layers, as the file (or the query) declared it.
    pub activation: Activation,
}

impl Model {
    /// Run `rows x in_features` through the stack.
    ///
    /// The layers are stepped by hand rather than through
    /// `Sequential::forward` so the activation lands *between* them and not
    /// after the last. The activation is not a child of the `Sequential`
    /// either: children are numbered `0.`, `1.`, … in `named_parameters`, so
    /// inserting a parameterless module between the layers would renumber
    /// every weight and stop the file loading at all.
    fn forward(&self, input: &Tensor) -> oxmera::Result<Tensor> {
        let last = self.module.len().saturating_sub(1);
        let mut x = input.clone();
        for (i, layer) in self.module.iter().enumerate() {
            x = layer.forward(&x)?;
            if i != last {
                x = self.activation.apply(&x)?;
            }
        }
        Ok(x)
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
///
/// Keyed by the path *and* the spec: two queries over the same file, one
/// asserting an activation and one not, must both be checked against the
/// header rather than the second silently inheriting the first's answer.
static MODELS: OnceLock<Mutex<ModelCache>> = OnceLock::new();

/// The cache's contents: a loaded model per path and asserted activation.
type ModelCache = HashMap<(String, ModelSpec), Arc<Model>>;

fn cache() -> &'static Mutex<ModelCache> {
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
    let key = (path.to_string(), spec);
    if let Some(found) = guard.get(&key) {
        return Ok(Arc::clone(found));
    }
    let loaded = Arc::new(load(path, spec)?);
    guard.insert(key, Arc::clone(&loaded));
    Ok(loaded)
}

/// Rebuild the architecture from the tensor names and shapes, then fill it.
fn load(path: &str, spec: ModelSpec) -> Result<Model> {
    let ModelSpec::SequentialMlp { activation } = spec;
    let bytes = std::fs::read(path).map_err(|e| exec(format!("{PREDICT}: reading {path}: {e}")))?;
    let activation = resolve_activation(path, &bytes, activation)?;
    let shapes = layer_shapes(path, &bytes)?;
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
        activation,
    })
}

/// The activation to use, from the file's `__metadata__`, the query, or both.
///
/// A file that declares nothing and a query that asserts nothing is the one
/// case with no answer, and it is an error. Guessing there is what made every
/// non-linear model load and return a linear model's numbers before 0.2.0, so
/// the message names the key and both ways of supplying it.
fn resolve_activation(
    path: &str,
    bytes: &[u8],
    asserted: Option<Activation>,
) -> Result<Activation> {
    let declared = declared_activation(path, bytes)?;
    match (declared, asserted) {
        (Some(declared), Some(asserted)) if declared != asserted => Err(exec(format!(
            "{PREDICT}: {path} declares `{ACTIVATION_KEY}: {declared}` but the query              asks for '{asserted}'; the model was trained with one of them, so              change the query or the file rather than running the other"
        ))),
        (Some(declared), _) => Ok(declared),
        (None, Some(asserted)) => Ok(asserted),
        (None, None) => Err(exec(format!(
            "{PREDICT}: {path} does not declare `{ACTIVATION_KEY}` in its              safetensors `__metadata__`, so the activation between its Linear              layers is unknown and assuming one would return confident wrong              numbers. Either save the file with that key (`safetensors::serialize`              takes a metadata map; `oxmera::nn::serialize::save` does not write              one as of oxmera 0.5), or state it in the query:              predict('{path}', features, 'relu'). Accepted values: {}",
            Activation::names()
        ))),
    }
}

/// The `oxidelake.activation` value in the file's header, if it has one.
fn declared_activation(path: &str, bytes: &[u8]) -> Result<Option<Activation>> {
    let (_, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|e| exec(format!("{PREDICT}: {path} is not a safetensors file: {e}")))?;
    metadata
        .metadata()
        .as_ref()
        .and_then(|entries| entries.get(ACTIVATION_KEY))
        .map(|name| {
            Activation::parse(name)
                .map_err(|e| exec(format!("{PREDICT}: {path}: `{ACTIVATION_KEY}`: {e}")))
        })
        .transpose()
}

/// `(in_features, out_features)` per layer, in layer order.
///
/// Read from the file rather than configured: `N.weight` has shape
/// `[out, in]`, so the whole architecture is recoverable from the header.
/// Layers must be numbered `0..n` with no gaps — a gap means this is not the
/// `Sequential` layout, and guessing at that point would build a model that
/// loads and computes something else.
fn layer_shapes(path: &str, bytes: &[u8]) -> Result<Vec<(usize, usize)>> {
    let file = SafeTensors::deserialize(bytes)
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

    /// `predict(model_path, features)` or `predict(model_path, features,
    /// activation)`: a Utf8 literal, a list of numbers, and optionally the
    /// activation as a string.
    ///
    /// The features are coerced to `FixedSizeList<Float32>` exactly as the
    /// distance UDFs do, so a column written as `List<Float64>` still works
    /// and the dimension is known before execution.
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let (path, features, activation) = match arg_types {
            [path, features] => (path, features, None),
            [path, features, activation] => (path, features, Some(activation)),
            other => {
                return Err(DataFusionError::Plan(format!(
                    "{PREDICT} takes 2 or 3 arguments (model path, features                      [, activation]), got {}",
                    other.len()
                )));
            }
        };
        if let Some(activation) = activation
            && !is_string(activation)
        {
            return Err(DataFusionError::Plan(format!(
                "{PREDICT}: the third argument is the activation as a string                  ({}), got {activation:?}",
                Activation::names()
            )));
        }
        if !is_string(path) {
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
        let mut coerced = vec![
            DataType::Utf8,
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
        ];
        if activation.is_some() {
            coerced.push(DataType::Utf8);
        }
        Ok(coerced)
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
        let (path, features, activation) = match args.args.as_slice() {
            [path, features] => (path, features, None),
            [path, features, activation] => (path, features, Some(activation)),
            other => {
                return Err(exec(format!(
                    "{PREDICT} takes 2 or 3 arguments, got {}",
                    other.len()
                )));
            }
        };
        let path = model_path(path)?;
        let activation = activation.map(activation_argument).transpose()?;
        let activation = activation.as_deref().map(Activation::parse).transpose()?;
        let model = model(&path, ModelSpec::SequentialMlp { activation })?;
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

fn is_string(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

/// A literal string argument, or `None` when it is not one.
fn constant_string(value: &ColumnarValue) -> Option<String> {
    match value {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(s)))
        | ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) => Some(s.clone()),
        _ => None,
    }
}

fn model_path(value: &ColumnarValue) -> Result<String> {
    // Deliberately refused rather than supported. A per-row model path means a
    // per-row model load, and the cache would grow without bound on a column
    // of distinct paths — a query that quietly consumes the machine is worse
    // than one that will not plan.
    constant_string(value).ok_or_else(|| {
        DataFusionError::Plan(format!(
            "{PREDICT}: the model path must be a constant string, not a column"
        ))
    })
}

/// The activation argument, which must be a literal for the same reason the
/// path must: it selects which model is built and cached.
fn activation_argument(value: &ColumnarValue) -> Result<String> {
    constant_string(value).ok_or_else(|| {
        DataFusionError::Plan(format!(
            "{PREDICT}: the activation must be a constant string, not a column"
        ))
    })
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
