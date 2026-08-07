# THEO v0.4.1 — Benchmark Corpus & Conformance Release Checkpoint

**Status**: FROZEN  
**Date**: 2026-08-08  
**Governance**: Canon Edition C1 (Immutable Baseline) + ADR-0026, ADR-0027

---

## Governing Specifications

The following governing contracts define THEO v0.4.1 and MUST NOT be altered without an ADR or a new Canon Edition:

- **Canon**: `CANON.md` (Edition C1)
- **Principles**: `PRINCIPLES.md`
- **Architectural Invariants**: `ARCHITECTURAL_INVARIANTS.md`
- **Implementation Guide**: `IMPLEMENTATION_GUIDE.md`
- **Living Roadmap**: `ROADMAP.md` (v0.4.1 Freeze section)
- **ADR-0026**: Canon Law 4 verdict — perception enters cognition as evidence; `BeliefSource` stays closed at MEMORY / KNOWLEDGE / INFERENCE.
- **ADR-0027**: Runtime unification — the 9-stage symbolic pipeline is the single canonical runtime; the v0.2 12-stage `CognitiveEngine` is demoted to a legacy compatibility path (`theo chat --engine legacy`).
- **ADR-0019 … ADR-0025**: Concept/Thought/Belief graph model, inference & hypothesis engines, conflict policy, boundary response rendering, and the cognitive ontology.

---

## What's New in v0.4.1

### Cognitive Benchmark Corpus & Governance CLI
- **26 benchmark cases** across 5 domains (commonsense, taxonomy, contradiction, causal reasoning, uncertainty), ≥5 per domain, in `evaluation/benchmarks/`.
- **`evaluation/harness.py`**: deterministic harness comparing expected vs actual Belief Graphs, Decisions, and GoldenTraces — asserting only explicitly specified fields.
- **`theo benchmark run` CLI**: executable governance over the canonical pipeline, with domain/case filters and exit code 1 on failure.
- **GoldenTrace**: complete per-cycle structural trace (retrieved memories, fired rules, derived beliefs, generated hypotheses, resolved conflicts, thought DAG).

### Conformance & Contract Enforcement
- Conformance suites for the belief laws, Canon Law 6 (no language generation inside cognition; boundary-only rendering), and canonical-vs-legacy isolation.
- **ADRs are test-enforced**: index completeness, sequential numbering, status validity, and ADR-to-symbol import mapping (`tests/adr/`).

### Kernel & Lifecycle Hardening
- **Legal transition graph**: `SubsystemRegistry.transition()` rejects illegal state jumps (`InvalidStateTransitionError`); `FAILED` subsystems may retry via `FAILED → STARTING`.
- **Idempotent lifecycle**: `Kernel.boot()` / `shutdown()` drive STARTING→RUNNING and STOPPING→STOPPED, and are safe to call repeatedly.

### Determinism Guarantees
- **Fingerprint-based determinism** over `CognitiveTraceFingerprint`; `ComputeBudget=None` → fully deterministic.
- **Cross-process determinism** verified via independent subprocesses producing byte-identical sha256 fingerprints.
- **ReplayEngine generalized** over a `ReplayableEngine` protocol satisfied by both the canonical runtime and the legacy engine.

### Engineering Hygiene
- **Single version source**: `theo_core/_version.py` feeds the package, the CLI banner, kernel boot events, trace metadata, and the build backend (hatchling `[tool.hatch.version]`).
- **No runtime state in the repository**: all bootstrap state routes to test-owned `tmp_path`; committed `data/` artifacts removed from the index.
- **Property suites**: identifier determinism, serialization idempotence, and edge-input behavior under Hypothesis.

---

## Verification

- **Test Suite**: 271/271 passing.
- **Type Checking**: `mypy src` clean (265 source files).
- **Linter**: `ruff check src tests` clean.
- **Benchmark CLI**: `theo benchmark run` → 26/26 PASS (all domains).
- **Determinism**: cross-process fingerprint equality (single and repeated runs).

---

## Frozen Contract (v0.4.1)

- 9-stage pipeline: PERCEPTION → ACTIVATION → REVISION → INFERENCE → HYPOTHESIS → CONFLICT_RESOLUTION → DECISION → REALIZATION → LEARNING.
- Decision model: `DecisionRecord` (Intent derived from a referenced Goal; `ActionSpec`), boundary `ResponseRenderer`.
- Belief sources: MEMORY / KNOWLEDGE / INFERENCE only (ADR-0026).
- Runtime: symbolic pipeline canonical, legacy engine excluded from the contract (ADR-0027).
- From v0.5 onward, releases deliver **content and compatibility only**.
