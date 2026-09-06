# ADR-0013: Cluster mode runs on Apache DataFusion Ballista

- Status: accepted
- Date: 2026-08-25

## Context

Phase 5 originally specified our own coordinator (stage DAG, a tonic control plane with `RegisterWorker` / `AssignTask` / `ReportTaskStatus`) and our own Arrow Flight shuffle service (`oxide-flight`). That is a re-implementation of the core of Apache DataFusion Ballista. A landscape check on 2026-08-25 found Ballista active and moving fast — 53.0.0 (May 2026), 54.0.0 (July 2026), 54.1.0 (2026-08-09) — shipping Python wheels, accepting pre-built physical plans (`submit_physical_plan`), carrying retry settings (`task_max_failures`, `stage_max_failures`) and listing adaptive query execution as its next priority. It exposes exactly the extension points OxideLake needs: `SchedulerConfig::override_physical_codec` and `override_session_builder`, `ExecutorProcessConfig::override_physical_codec` and `override_function_registry`, and `SessionContext::remote` / `SessionContext::standalone` on the client. The scheduler is not where OxideLake is unique; the GPU operator layer is.

## Decision

Delete `oxide-flight`. `oxidelake-runtime` wraps Ballista's scheduler and executor processes and installs our `OxidePhysicalCodec` (a `datafusion_proto::physical_plan::PhysicalExtensionCodec` for the four `Gpu*Exec` nodes) plus the `HardwarePlacementRule` through those hooks. On the scheduler the rule targets the declared cluster capability (`OXIDE_CLUSTER_BACKEND`); every `Gpu*Exec` selects its real local backend at execute time and runs the CPU reference path when the device is absent, so heterogeneous clusters stay correct. Ballista's documentation requires `datafusion` to match Ballista's version, so **Ballista's DataFusion major becomes the root of the dependency chain**: today Ballista 54 → DataFusion 54 / datafusion-proto 54 → arrow / parquet 58 → arrow-flight 58 → tonic / prost 0.14, with `rust-version = "1.88"`. DataFusion 55 is not used until Ballista moves.

## Consequences

Phase 5 shrinks to placement, codec, thin wrappers and the equivalence test. We inherit Ballista's Arrow Flight shuffle, retries, REST endpoint and release cadence — and lag DataFusion by one major whenever Ballista does. `datafusion-proto` becomes a direct dependency (the single exception recorded in ADR-0009); `arrow-flight`, `tonic` and `prost` are no longer declared directly. The Phase 7 end-to-end test spawns the real `oxide-scheduler` and `oxide-worker` binaries.
