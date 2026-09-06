# ADR-0015: In-database inference belongs at the UDF layer, not the operator layer

- Status: accepted (prototype)
- Date: 2026-09-06

## Context

OxideLake can *search* embeddings — `GpuVectorDistanceExec` and the
`l2_distance`/`cosine_distance` UDFs work over `FixedSizeList<Float32>` — and
cannot *score* a row against a model. The obvious missing feature is
in-database inference: `predict(model, features)` in SQL, as BigQuery ML and
DuckDB's extensions offer it.

`oxmera` is a Rust-native tensor and deep-learning library by the same
maintainer, with CPU, Metal and CUDA backends, safetensors weights, and
`nn::Sequential`/`Linear`. The question is where — if anywhere — it belongs
in this engine.

## Decision

**Inference is a scalar UDF. The operator layer is not touched.**

`predict(path, features)` loads a safetensors model once per path, runs the
batch through it on the CPU, and returns `List<Float32>`. It is behind a
`predict` cargo feature, off by default, forwarded by `oxidelake-runtime` and
`oxidelake-api` like every other optional capability (ADR-0006).

## Rejected: oxmera as a kernel or operator backend

`GpuFilterExec`, `GpuHashJoinExec` and `GpuAggregateExec` work on Arrow
batches with null bitmaps and selection semantics, behind the object-safe
`GpuBackend` of ADR-0005. A tensor library's array is dense, strided and
carries an autograd tape. Routing batches through one would:

- cost a copy in each direction per batch,
- lose null semantics, which Arrow has and a tensor does not,
- and replace kernels that are conformance-tested against stock DataFusion
  with kernels that are not.

The two projects both have a CUDA backend on `cudarc` and a Metal backend in
MSL. That is duplication, and it is the kind worth keeping: the memory models
differ, so the kernels are not interchangeable even though the arithmetic is.

## `default-features = false` is load-bearing

oxmera registers its CUDA backend from a `#[ctor]` that runs **before
`main`** and calls `CudaContext::new(0)`. Depending on it with default
features would give every `oxide` process — CLI, scheduler and each worker —
a `dlopen` of `libcuda` and a CUDA context at load time, beside the one
OxideLake creates itself, for every query including pure-CPU ones.

oxmera 0.4.0 made the CUDA backend an optional (default-on) feature precisely
so a consumer can decline it. `cargo tree -p oxidelake-compute --features predict`
shows no `cudarc` and no `ctor`, which is the property that makes this
dependency acceptable at all.

## The architecture is read, not configured

safetensors stores named tensors, not a graph, so the shape has to come from
somewhere. Rather than invent a sidecar format, `predict` reads the naming
convention `oxmera::nn::Sequential` already writes — `0.weight`, `0.bias`,
`1.weight`, … — and rebuilds a stack of `Linear` layers, where `N.weight` of
shape `[out, in]` is layer `N`. Layers must be numbered `0..n` with no gaps
and consecutive widths must compose, or the file is refused.

**The activation is the one real assumption**: ReLU between layers, nothing
after the last. That covers an MLP scorer and nothing else, and a model whose
activations differ will produce confidently wrong numbers rather than an
error. `ModelSpec` exists as the enum where an explicit description goes when
a second shape is needed; today it has one variant and the assumption is
documented at the call site.

## Consequences

- Inference composes with the whole SQL surface. Measured: `predict(...)` in a
  projection over `GpuFilterExec[cuda]`, with array indexing, `ORDER BY` and
  `LIMIT`, and in the same query as `l2_distance` — search and score together,
  which was the gap.
- The default build is unchanged: no oxmera, no safetensors, nothing new in
  `cargo tree`.
- The feature is a **cluster-wide** decision. `oxide_udfs()` only includes
  `predict` when the feature is on, so an executor built without it would fail
  a query the client planned with it. Build the cluster uniformly, as the GPU
  features already require.
- Correctness is checked against the reference rather than a recorded number:
  the test runs the same model through `oxmera` directly and asserts the SQL
  answer matches within `1e-5`. A wrong architecture inference is the failure
  mode this guards.
- Not addressed: GPU inference (the model runs on the CPU), model cache
  invalidation (a file that changes under a running process keeps serving the
  old weights — right for a query engine, wrong for a notebook), and any model
  that is not an MLP.
