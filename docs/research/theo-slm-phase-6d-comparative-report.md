# THEO SLM v0 — Phase 6D Comparative Evaluation Report

**Document:** `docs/research/theo-slm-phase-6d-comparative-report.md`  
**Date:** 2026-08-26  
**Status:** EVALUATION ONLY — no training, no architecture change performed. STOP for human review.  
**Governing constraint:** architecture selection uses **behavioral/structured-output** criteria only. Excluded signals: `cos(H1)`, gradient-compatibility, hidden-state geometry, Qwen-generated labels. 300-GOLD dataset, labels, target schema, and ADR-0028 contract unchanged.

---

## 1. Authorization & scope
Authorized: 6D comparative evaluation using the §7 evaluation matrix and §5 decision rule from
`theo-slm-phase-6c-model-selection-plan.md`. Candidates evaluated with **distinct provenance** so model
capability is never confused with training configuration:
- **M1_base** — `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae55760…` (no adapter; contract JSON prompt)
- **M1_v1** — M1_base + QLoRA adapter v1 (`4dcdbe51…`), trained to emit `DECISION: …` (diagnostic only)
- **M1_v2** — M1_base + QLoRA adapter v2 (`3462bd30…`), trained to emit `DECISION: …` (diagnostic only)
- **M2_base** — `HuggingFaceTB/SmolLM2-360M-Instruct` @ `a10cc1512…` (no adapter; contract JSON prompt)

Both v1/v2 training artifacts are **retained as diagnostic evidence**; their smoke results are **not** treated as protocol verdicts.

## 2. Reference instruments (frozen, evaluation-only, never trained on)
- 15-case semantic probe (`phase-6e3/semantic-probe-results.json`) — fields: `percept`, `expected_decision` ∈ {`SHOULD_PROPOSE`,`SHOULD_ABSTAIN`}, `group` (A–E).
- 51-case benchmark (`phase-6e3/benchmark-results.json`) — fields: `percept`, `expected_decision`, `domain`.
- **n = 66** total. The instruments use the v0.4.1/`decision`+`reasoning` contract. They do **not** carry
  snapshot-bound concept/evidence ids, so the formal `SemanticInterpretation` grounding check (id-validity
  against snapshot) is **not measurable from this instrument** (see §4 caveats).

## 3. Gate measurements (per candidate)

| Gate (§7) | GO threshold | M1_base | M1_v1 | M1_v2 | M2_base |
|---|---|---|---|---|---|
| E0 structured-output failure | ≤ 2.0% | **FAIL** 22.7% | **FAIL** 100% (non-contract `DECISION:` format) | **FAIL** 100% | **FAIL** 92.4% |
| Grounding (id validity) | 100% | N/A† | N/A† | N/A† | N/A† |
| E5 semantic novelty | ≥ 40% | N/A† | N/A† | N/A† | N/A† |
| E6 evidence relevance | ≥ 30% | N/A† (Group E is proxy) | N/A† | N/A† | N/A† |
| **Group E distractor rejection** | **≥ 80%** | **FAIL 0/3 (0%)** | **FAIL 0/3** | **FAIL 0/3** | **FAIL 0/3** |
| Abstention accuracy | ≥ 90% | **FAIL 5/16 (31%)** | FAIL 0/16 | FAIL 0/16 | FAIL 0/16 |
| Shortcut resistance | ≤ 40% bal-acc | RED FLAG (over-propose 42/9/15; degenerate imbalance) | RED FLAG (collapsed) | RED FLAG (collapsed all-PROPOSE) | RED FLAG (collapsed all-NONE) |
| Latency | ≤ 0.5 s/GPU | **FAIL 2.84 s** | FAIL 0.89 s | FAIL 0.86 s | FAIL 2.54 s |
| Footprint / VRAM | ≤1B / ≤2GB | PASS 0.48GB | PASS 0.72GB | PASS 0.72GB | PASS 0.72GB |
| v0.5 / ADR-0028 compat | all green | PASS (structural) | PARTIAL (output schema ≠ `SemanticInterpretation`) | PARTIAL (output schema ≠ `SemanticInterpretation`) | PASS (structural) |

† The frozen instrument carries no snapshot-bound ids and no per-case novelty/relevance labels; formal
grounding/E5/E6 require the ADR-0028 snapshot runtime + 6A.2 labeled rubric, which are **not invoked here**.
Group E is the plan-designated operational proxy for evidence relevance (§5: "C7 evidence relevance is the
deciding gate").

## 4. Caveats (required for honest interpretation)
1. **Group E n=3.** The frozen probe exposes only 3 cases tagged `group=="E"`; gate is weakly powered.
   Consistent with the plan's historical reference: the 6A.2 semantic probe scored Group E evidence
   relevance = **0/15** for the prior symbolic system — corroborating A's structural weakness on evidence relevance.
2. **Cross-domain instrument.** The probe/benchmark use the v0.4.1 `SHOULD_PROPOSE/ABSTAIN` framing
   (semantic-interpretation novelty), not the 300-GOLD `SENSITIVE/ABSTAIN/NEGATIVE` adjudication framing.
   Adapter outputs were mapped (NEGATIVE→PROPOSE; SENSITIVE/ABSTAIN→ABSTAIN) for comparability; the v1/v2
   "decision accuracy" (0.697 / 0.758) is a **majority-class artifact** — v2 emitted PROPOSE for **all 66**
   cases (instrument base rate ≈ 76%), confirming the training collapse, not capability.
3. **Adapter schema mismatch.** v1/v2 were fine-tuned to emit `DECISION: X`, not the `SemanticInterpretation`
   contract. They are therefore E0-fail by construction and must be rewired (separate authorization) to be
   protocol-eligible.

## 5. Decision-rule application (§5, pre-registered)
- **Primary: A (decoder-only).** The only contract-compliant, non-degenerate realization is **M1_base**
  (E0 77%, decision accuracy 64%). However A **fails the deciding gate**: Group E distractor rejection = **0%**
  (≪ 80%), and abstention = 31% (≪ 90%), and latency = 2.84 s (≫ 0.5 s). → **A is NOT confirmed.**
- **Fallback B (decoder + reranker).** Per §5, B is selected only if A fails Group E **while B clears it**.
  B (reranker) has **not been built** in this authorization; it is therefore **untested** and cannot be selected.
- **C (shared encoder).** Selected only if A and B both fail a v0.5-isolation/budget gate — not reached; also unbuilt.

**Verdict:** Selection **cannot be finalized**. Architecture A is the plan-designated primary, but every
available A realization fails the Group E deciding gate (and abstention/latency). The approved fallback B is
not yet available to test. No candidate clears all §7 GO thresholds.

## 6. Recommended next authorizations (human decision required)
1. **Build a compliant, non-degenerate A realization** — a QLoRA adapter on M1 that emits the
   `SemanticInterpretation` contract (not `DECISION:`), trained to clear Group E (≥80%) and abstention
   (≥90%) on the frozen instrument. (This is a new, separately-scoped training iteration — NOT performed here.)
2. **OR build architecture B (decoder + snapshot-only reranker)** to exercise the §5 fallback, then re-run 6D.
3. Address **latency**: all candidates exceed the 0.5 s/GPU gate; a CPU-primary or quantized-serving path is
   required before the latency gate can pass.

**→ STOP. Awaiting human review and explicit authorization before any further training or architecture change.**
