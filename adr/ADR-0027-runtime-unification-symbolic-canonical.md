# ADR 0027: Runtime Unification — Symbolic Pipeline as Canonical Runtime

## Status
Accepted (v0.4.1)

## Context
THEO v0.4.1 ships two execution paths: the v0.2 12-stage `CognitiveEngine` and
the v0.4 `SymbolicCognitivePipeline` (9 stages). Dual paths created ambiguity
about which runtime is "THEO", where language generation lives, and which
contract carries the determinism guarantee.

## Decision
- **Symbolic pipeline is canonical.** `SymbolicCognitivePipeline` (9 stages:
  PERCEPTION → ACTIVATION → REVISION → INFERENCE → HYPOTHESIS →
  CONFLICT_RESOLUTION → DECISION → REALIZATION → LEARNING) is the single
  reference implementation of Canon Edition C1.
- **Legacy engine demoted.** The v0.2 12-stage `CognitiveEngine` remains as a
  legacy compatibility path, reachable only via `theo chat --engine legacy`, and
  is excluded from the frozen contract.
- **Boundary runtime owns non-cognitive concerns.** `SymbolicRuntime` is the
  runtime boundary: it renders responses via `ResponseRendererPort`, persists
  committed state via `SymbolicStateStore`, and implements the kernel
  `Startable`/`Stoppable` lifecycle.
- **Canon Law 6 applies to the canonical path.** The pipeline never renders
  language; `GoldenTrace.response_text` carries the raw traceable interpretation,
  and rendering happens only at the boundary.
- **Replay is runtime-agnostic.** `ReplayEngine` depends on a `ReplayableEngine`
  protocol satisfied by both the legacy engine and `SymbolicRuntime`.

## Rationale
One canonical runtime gives the freeze posture a single enforceable contract:
benchmarks, determinism guarantees, and conformance all target the symbolic
pipeline. The legacy engine stays available for historical trace replay without
participating in the v0.4.1 architectural contract.

## Consequences
- The CLI defaults to `--engine symbolic`; `theo benchmark run` targets the
  canonical pipeline only.
- Determinism guarantees (fingerprint equality and cross-process determinism)
  are established for the symbolic runtime.
- Conformance tests, the benchmark corpus, and the state-machine suites verify
  the canonical path, including Canon Law 6 boundary behavior.
