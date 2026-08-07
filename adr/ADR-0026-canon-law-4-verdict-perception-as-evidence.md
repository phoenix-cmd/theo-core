# ADR 0026: Canon Law 4 Verdict — Perception Enters Cognition as Evidence

## Status
Accepted (v0.4.1)

## Context
The v0.4.1 conformance audit surfaced a question about the status of perception
under Canon Law 4. Law 4 defines a closed set of three mechanical belief sources
(MEMORY, KNOWLEDGE, INFERENCE). An amendment was considered that would add
PERCEPTION as a fourth mechanical source, treating raw sensory input as a direct
source of belief.

## Decision
- **No Canon amendment.** `BeliefSource` retains exactly three mechanical sources:
  MEMORY, KNOWLEDGE, and INFERENCE.
- **Percepts enter as evidence.** Raw sensory input is structured into an
  immutable `Percept` (`percept://` identifier, deterministic from content) and
  is never itself a belief.
- **Inference derives perceptual beliefs.** Beliefs about percepts are mechanically
  derived by the Inference stage with `source=INFERENCE`, carrying an
  `EvidenceTrace(source_type="perception")` for provenance.
- **Traceability preserved.** Every belief, including percept-derived ones, remains
  traceable through its support evidence (Canon Invariant 5).

## Rationale
Adding a PERCEPTION source would widen Law 4's closed source set and blur the
distinction between raw evidence and epistemically-loaded belief. Keeping
perception as evidence preserves a single, verifiable derivation path: every
belief that is "about" a percept is an INFERENCE belief whose support references
the `percept://` evidence.

## Consequences
- Percept-derived beliefs carry the full provenance machinery of the Inference
  stage and are always produced inside the cognitive cycle, never injected
  outside it.
- Hypothesis generation self-matches the percept belief, yielding the universal
  interpretation action text "Interpretation based on belief '<percept content>'".
- Conformance is enforced by `tests/conformance/test_belief_laws.py` (source
  validity) and the benchmark corpus's percept-based cases.

## Ratification
Ratified 2026-08-08 under Canon §10 as a **minor amendment** (clarify wording, no
new Edition). Canon Law 4 was amended from:

> Beliefs MUST be derived from Perception, Memory, Knowledge, or Inference.

to:

> Beliefs MUST be derived from Memory, Knowledge, or Inference. Perception enters
> cognition as evidence and MUST NOT directly produce beliefs; beliefs about
> percepts are mechanically derived by Inference with `source=INFERENCE` and an
> `EvidenceTrace(source_type="perception")`.

This does not change the decision's verdict (three mechanical sources; perception
as evidence); it resolves the wording tension between the original Law 4 text and
this ADR.
