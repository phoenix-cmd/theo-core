# THEO SLM v0 — Phase 6B Design Review Packet

**Document:** `theo-core/docs/research/theo-slm-phase-6b-review-packet.md`
**Date:** 2026-08-25
**Status:** APPROVED — final human sign-off, 2026-08-25 (6B review packet / documentation stage). Documentation/review artifact only.

**Authorization boundary (this packet alone grants none of the following):** Packet approval does
**not** authorize model selection, model download, dataset mutation/creation, training, Phase 6C
build, or Phase 6D comparative evaluation. Each of those requires its own explicit, separate
authorization (see *Phase-distinction & authorization boundary* below).

**Source design (6B deliverable):** `theo-core/docs/research/theo-slm-design-v0.md`
**Context:** 6E.20.x fixed-subspace-readout experiment is CLOSED/inconclusive (outcome:
`docs/research/theo-slm-phase-6e20-outcome.md`). The roadmap now points to the product path:
**6B review → 6C build → 6D comparative evaluation → v0.5 Neural Runtime.**

---

## Cross-cutting lesson from 6E.20.x (applies to every gate)

> **`cos(H1)` is not a validated training/evaluation signal for THEO SLM v0.**

If any gate below would rely on hidden-state gradient-cosine compatibility, or any
gradient-based probe, it must be rejected or conditioned on a pre-registered sensitivity
control. THEO SLM v0 evaluation rests on **behavioral / structured-output gates** only.

---

## Gate 1 — Capability list (§A) is complete, narrow, and defensible

**Recommendation: APPROVE WITH NOTE.**

- The capability list is bounded: E0 structured output, grounded abstention, semantic
  novelty detection, decision-relevance explanation, schema-compliant rewriting. Each is a
  measurable behavioral function — not a hidden-state property.
- **Note for reviewer:** confirm no capability silently depends on the model exposing a
  "aligned readout subspace." Recommend the list stay at observable behavior only.

## Gate 2 — Exclusion list (§B) is enforced, especially the new gradient-compatibility item

**Recommendation: APPROVE.**

- The exclusion list now contains the binding clause: *no `cos(H1)`-style subspace-alignment
  training signals; the 6E.20.x experiment showed this observable is not a validated
  training or evaluation signal.*
- This closes the door that 6E.20.x left open. Confirm this clause is carried verbatim into
  the 6C implementation spec and any training config review.

## Gate 3 — Data contract (§C/§D) — symbolic ground truth, not Qwen outputs as labels

**Recommendation: APPROVE.**

- Primary labels are human/symbolic ground truth; Qwen outputs are negative examples and
  format-failure regressions only. The b/002 seed example is preserved as gold standard.
- **Note for reviewer:** this is the correct defense against label contamination; keep the
  6A.1 / 6A.2 provenance frozen.

## Gate 4 — Evaluation discipline (§G) — behavioral gates only, probes excluded

**Recommendation: APPROVE WITH NOTE.**

- Evaluation gates are behavioral: E0 structured-output failure rate, grounding rate,
  semantic-novelty rate, decision-relevance. Frozen 51-case benchmark + 15-case probe are
  evaluation-only forever.
- The new §G clause explicitly **excludes gradient-based compatibility probes** unless they
  pass a pre-registered sensitivity control.
- **Note for reviewer:** confirm the frozen benchmark/probe are byte-locked and that no
  future 6C eval run can silently add a `cos(H1)`-style metric.

## Gate 5 — Stopping / rescue discipline (folds the 6E.20.7 lesson)

**Recommendation: APPROVE.**

- §H encodes: when a measurement instrument fails its validation control, **stop** and
  redesign the measurement; do **not** increase training dose or iterate on the same metric.
- This is the formal guard against a 6E.20.x-style open-ended research loop. Confirm it is
  referenced in the 6C build plan and the 6D evaluation plan as a hard gate.

## Gate 6 — Roadmap realignment (do not let 6E.20.x become an open-ended loop)

**Recommendation: APPROVE.**

- Record that 6E.20.x is a closed sub-experiment; 6E.20.8 is a **separate, gated future
  branch** requiring a materially different measurement strategy or question — not an
  automatic continuation.
- The product path is explicit: **6B review → 6C build → 6D comparative evaluation →
  v0.5 Neural Runtime.** Recommend the ROADMAP.md reflect this ordering and remove any
  implication that 6E.20.x gates further THEO SLM v0 progress.

---

## (+1) Gate — Review sign-off mechanics

**Recommendation: REQUIRE EXPLICIT SIGN-OFF.**

- This packet is documentation/review only. It grants **no** authorization to build, train,
  modify source, checkpoints, or protocol.
- 6C (build) and 6D (eval) may **only** begin after this review is explicitly approved and
  the approval is recorded here in §Sign-off below.

---

## Crosswalk: seven review gates → 18 mandatory 6B approval items

**Source note (provenance / integrity):** A pre-existing enumerated *"dataset specification
§19.1 / 18 mandatory 6B approval items"* list was **not found** in the repository — `theo-slm-design-v0.md`
has a 6-item review gate, and `theo-slm-dataset-v0.md` runs sections 1–18 with no §19.1. To avoid
fabricating a list and presenting it as pre-existing, the **18 items below are consolidated directly
from the substantive approval content of the 6B design (`§A`–`§H`) and the dataset-spec governance
(`§18`)**, so the crosswalk is self-contained and auditable. If a separate authoritative §19.1 list
exists elsewhere, supply it and this crosswalk will be re-aligned to it.

**Disposition legend:** APPROVE · APPROVE WITH NOTE (condition attached) · REQUEST CHANGE (blocking).

| # | Mandatory 6B approval item (consolidated) | Source | Gate(s) | Disposition | Reviewer note |
|---|---|---|---|---|---|
| 1 | **A1** — E0 structured-output generation: derive conclusions absent from symbolic knowledge | §A | G1 | APPROVE WITH NOTE | Confirm no capability silently depends on an "aligned readout subspace"; stay at observable behavior. |
| 2 | **A2** — Grounded abstention: emit clean `ABSTAIN` on `!supported`/unsupported | §A | G1 | APPROVE WITH NOTE | Same as #1. |
| 3 | **A3** — Semantic-novelty detection + SN taxonomy remap | §A | G1 | APPROVE WITH NOTE | Same as #1. |
| 4 | **A4** — Decision-relevance explanation (`EVIDENCE`/`REASONING`/`CONCLUSION`) | §A | G1 | APPROVE WITH NOTE | Same as #1. |
| 5 | **A5** — Schema-compliant rewriting / repair | §A | G1 | APPROVE WITH NOTE | Same as #1. |
| 6 | **B1** — No hidden-state gradient-compatibility optimization (no `cos(H1)` training signals) | §B / §H | G2 | APPROVE | Binding clause; carry verbatim into 6C spec + training-config review. |
| 7 | **B2** — No replacement of deterministic verification / provider schema+grounding | §B | G2 | APPROVE | Untouched. |
| 8 | **C1** — `SemanticInterpretation` schema + adapter conversion adopted | §C | G3 | APPROVE | — |
| 9 | **D1** — Training-target doctrine: human/symbolic ground truth (Qwen = negatives only) | §D | G3 | APPROVE | Keep 6A.1/6A.2 provenance frozen. |
| 10 | **D2** — b/002 gold-standard seed preserved | §D | G3 | APPROVE | — |
| 11 | **D3** — derive-vs-repeat organizing dichotomy enforced | §D | G3 | APPROVE | — |
| 12 | **E1** — Dataset format fixed (`td://` namespace, full schema) | §E | G3 | APPROVE | — |
| 13 | **E2** — Train/eval separation; frozen 51-case + 15-case probe evaluation-only | §E / §G | G3, G4 | APPROVE | — |
| 14 | **F1** — Architecture options scored vs requirements table; selection deferred to 6C | §F | G1 | APPROVE | Final architecture choice is a 6C activity, not 6B. |
| 15 | **G1** — Behavioral evaluation gates (E0 failure rate, grounding, semantic-novelty, decision-relevance) | §G | G4 | APPROVE WITH NOTE | Confirm frozen benchmark/probe are byte-locked; no silent `cos(H1)`-style metric in 6C. |
| 16 | **G2** — Gradient-based compatibility probes excluded unless pre-registered sensitivity control passes | §G / §H | G4, G2 | APPROVE | Reinforces B1. |
| 17 | **H1** — 6E.20.x closure accepted; `cos(H1)` not validated; stopping/rescue discipline encoded | §H | G5 | APPROVE | — |
| 18 | **GOV1** — Governance stop: no training code, base-model selection, dataset creation, or model downloads precede this review | Design §Review-gate / dataset-spec §18 | G6, G7 | APPROVE | — |

**Disposition summary under CONDITIONAL APPROVAL:** items 1–5 and 14 carry the conditions
(APPROVE WITH NOTE); items 6–13, 15–18 are APPROVE. **No item is in blocking REQUEST CHANGE
status.** The conditions attached to the APPROVE WITH NOTE items are the substance of the
"conditional" approval and must be satisfied (or explicitly waived) at final sign-off.

---

## Phase-distinction & authorization boundary

These five stages are **distinct** and must not be conflated. Packet approval covers **only** the
first.

1. **6B design approval** — *this packet.* Approves the capability/exclusion/data/eval/stopping
   design recorded in `theo-slm-design-v0.md` (§A–§H). It is a design-review decision, not an
   execution authorization.
2. **6B.2 human review / GOLD adjudication** — a *separate downstream dataset activity*
   (dataset-spec §18 prerequisites; dual-reviewer GOLD curation). Not conferred by, and not
   required for, this 6B design packet — but it is a prerequisite for any later dataset use.
3. **Model selection** — choosing/downloading a base model (e.g., Qwen variant) is a **6C
   activity** requiring its own explicit authorization. Not conferred by this packet.
4. **6C build** — implementation/training of the THEO SLM. Requires explicit, separate
   authorization after 6B design approval **and** after 6B.2 GOLD adjudication as applicable.
5. **6D comparative evaluation** — comparative/end-to-end eval of the built model. Requires
   explicit, separate authorization after 6C.

**Explicit statement:** approval of this 6B review packet **alone** does **not** authorize
model selection, model download, dataset mutation/creation, training, Phase 6C build, or Phase 6D
comparative evaluation. Each requires its own explicit sign-off. The 6E.20.x branch remains
CLOSED; its frozen benchmark/probe, ADR-0028, provider contracts, and source code are untouched.
The `cos(H1)` exclusion (§H / B1 / G2) is preserved as a binding design constraint.

---

## Sign-off

| Role | Decision | Date | Notes |
|---|---|---|---|
| Reviewer (you) | ☑ APPROVE (final sign-off, 2026-08-25) | 2026-08-25 | Conditional items 1–5, 14 (APPROVE WITH NOTE) accepted; no blocking REQUEST CHANGE. |
| Next phase authorized | ☐ 6C build / ☐ 6D eval / ☑ neither | 2026-08-25 | Packet approval alone authorizes no further phase; 6C/6D/6B.2/model selection each require separate explicit authorization. |

**Until signed, the THEO SLM v0 program remains at 6B (review). No automatic advance.**
