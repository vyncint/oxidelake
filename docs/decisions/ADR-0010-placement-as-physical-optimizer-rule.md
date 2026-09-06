# ADR-0010: Hardware placement is a DataFusion `PhysicalOptimizerRule`

- Status: accepted
- Date: 2026-08-25

## Context

The draft said to "analyze the query AST" to route operators to GPU or CPU. At the AST/logical level the physical operator shapes, resolved data types and partitioning are not yet known, so eligibility cannot be decided correctly there.

## Decision

`HardwarePlacementRule` implements `PhysicalOptimizerRule` and is registered into the session state. It rewrites `FilterExec` / `ProjectionExec` / `HashJoinExec` / `AggregateExec` into `Gpu*Exec` when the selected backend is not CPU and the node fits the bounded v1 coverage; everything else stays on CPU. `EXPLAIN` shows placement tags such as `GpuFilterExec[cuda]`.

## Consequences

Decisions are unit-testable with a mocked GPU-present detector and visible to users. The rule is the single place where coverage rules live. In cluster mode it is installed into the Ballista scheduler's session through `SchedulerConfig::override_session_builder` and targets the declared cluster capability, while each `Gpu*Exec` re-selects the real local backend at execute time (ADR-0013).
