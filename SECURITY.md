# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub's
*Security → Report a vulnerability* form on this repository rather than in a
public issue. Include the commit or version, a reproduction, and the impact you
believe it has. You will get an acknowledgement, and a fix or a written
assessment, as fast as a small project can manage.

## Posture

What the project does continuously, enforced by required CI on every change:

- **`unsafe` is confined and justified.** Allocator/FFI internals only
  (`oxidelake-memory`, `oxidelake-device`, the io_uring worker in
  `oxidelake-storage`); every block carries a `// SAFETY:` comment stating the
  invariant it relies on (`clippy.toml` requires it), every raw resource is
  owned by an RAII type, and `unsafe_op_in_unsafe_fn` is denied.
  `oxidelake-core` is `#![forbid(unsafe_code)]`.
- **No panics in library code.** `unwrap`, `expect`, `todo!` and
  `unimplemented!` are denied by the workspace lints; unimplemented paths
  return `EngineError::Unsupported`. A plan payload from the network cannot
  drive the predicate evaluators arbitrarily deep (`MAX_PREDICATE_LEAVES`).
- **Untrusted input is validated before it becomes a node.** The plan codec
  checks magic, version, column indices, types and dimensions; generated SQL
  quotes every identifier; a model path given to `predict` must be a constant.
- **Dependency policy** (`deny` job): RUSTSEC advisories, a license allow-list,
  banned crates and crates.io-only sources on every PR; weekly grouped
  Dependabot updates that leave the DataFusion/Ballista chain's majors alone.
  Every accepted advisory in `deny.toml` carries a written reason.
- **Workflow security** (`zizmor` job at `--persona=pedantic`): every GitHub
  Action pinned to a full commit SHA, checkouts that do not persist
  credentials, a read-only workflow token by default, no build cache that
  could serve a stale artifact. Accepted findings are in `.github/zizmor.yml`.
- **Protected history.** `main` takes only squash-merged pull requests with
  `required-green` and `commit-policy` green — direct pushes are rejected, the
  maintainer's included. `v*` tags can only be created by an admin, and
  releases publish through crates.io Trusted Publishing from a `release`
  environment that deploys from `v*` tags alone.
- **GPU correctness is asserted, not assumed.** Every device operator is
  conformance-tested against stock DataFusion, and the Metal suite asserts
  device execution through transfer counters so a silent CPU fallback cannot
  pass as a GPU result.

## Supported versions

Only the `main` branch (and the most recent tagged release, if any) receives
security fixes.

## Trust model

OxideLake is an analytical engine intended to run **on infrastructure you
control**. Its security boundary is the machine or private network it runs on,
not the query interface.

- **Cluster mode has no authentication or transport encryption in v1.**
  `oxide-scheduler` and `oxide-worker` speak plain gRPC and Arrow Flight and
  will execute any plan a connected client submits, including reading any file
  the worker process can read. Both bind to `127.0.0.1` by default; expose them
  only on a trusted private network (or behind a mutually authenticated
  proxy). TLS/auth is tracked as out of scope for v1 in `STATUS.md`.
- **Workers trust their scheduler.** Physical plans arrive over the network and
  are decoded by `OxidePhysicalCodec`. Payloads are validated (magic, version,
  column indices, types, dimensions, a cap on predicate size) so a malformed
  payload yields an error rather than a panic, but a *malicious* scheduler is
  outside the threat model: it can already run arbitrary SQL.
- **The `oxide` CLI reads any Parquet path you give it** with your privileges;
  identifiers you pass (`--table NAME=PATH`) are quoted before they are used in
  generated SQL.
- **GPU kernels bounds-check every global access**; kernel sources are compiled
  from strings embedded in the binary at build time, never from user input.
- **Unsafe code** is confined to allocator and FFI internals and every block
  documents its invariant (`// SAFETY:`); `oxidelake-core` is `#![forbid(unsafe_code)]`.

## Dependencies

`cargo deny check` runs in CI against the RustSec advisory database. Advisories
that cannot be fixed by an upgrade are ignored only with a written reason in
`deny.toml` explaining why the affected code path is unreachable in OxideLake;
each ignore is revisited whenever the DataFusion/Ballista chain moves.
