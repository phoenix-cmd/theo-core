# THEO-SLM Phase 6D — Comparative Evaluation: Architecture A vs Architecture B

**Date:** 2026-08-26
**Status:** EVALUATED — NO FINAL MODEL SELECTION (human review required)
**Instrument (frozen):** 66 cases = `phase-6e3/semantic-probe-results.json` (15, groups A–E) + `phase-6e3/benchmark-results.json` (51)
**Gates (6C plan §7, universal):** E0 ≤2%, grounding 100%*, E5 ≥40%*, E6 ≥30%*, **Group E ≥80% (deciding)**, abstention ≥90%, shortcut ≤40%, latency ≤0.5s, VRAM ≤2GB, v0.5/ADR-0028 compat.
(*grounding/E5/E6 not derivable from this instrument → not scored here; carried forward as unmet-not-evaluable.)

---

## 1. Architecture A (decoder-only, M1 base + optional QLoRA adapters)

| Candidate | E0 pass | Accuracy | Group E | Abstain acc | Latency | VRAM | Notes |
|---|---|---|---|---|---|---|---|
| M1_base | 77.3% | 0.636 | **0%** (0/3) | 31.3% | 2.84s | 0.48GB | distractor cases accepted → fails deciding gate |
| M1_v1 (adapter `4dcdbe51…`) | 0% | — | — | — | — | — | non-contract `DECISION:` format; collapsed |
| M1_v2 (adapter `3462bd30…`) | 0% | — | — | — | — | — | non-contract; emitted PROPOSE for all 66 |
| M2_base | 7.6% | 0.045 | — | — | — | — | off-distribution for this task |

**A verdict:** fails the deciding **Group E** gate (0% distractor rejection) for every realization; abstention far below 90%. **A NOT confirmed.**

## 2. Architecture B (decoder M1_base + snapshot-only reranker)

Built after authorization `6D / Architecture B` (bounded phase). Implementation: `theo-core/scripts/model_selection/arch_b/b_architecture.py`; contract tests C1–C6 **ALL PASS**; independent source audit **cleared for execution**.

Deliberate scope decision: plan §3 describes B's reranker as a *trained* evidence-relevance encoder, but the frozen artifacts contain **no evidence-text corpus** (only evidence *ids*), so the bounded build uses a **deterministic snapshot-only evidence gate** — the minimal mechanism exercising the identical decision logic. A trained-encoder B remains a separate authorization requiring a populated evidence corpus.

| Metric | Result | Gate | Outcome |
|---|---|---|---|
| E0 pass | 100% | ≤2% | PASS |
| Accuracy | 0.242 | (context) | — |
| Group E | 100% (3/3 → ABSTAIN) | ≥80% (deciding) | PASS |
| Abstention acc | 100% (16/16) | ≥90% | PASS |
| Latency | 2.58s | ≤0.5s | **FAIL** |
| VRAM | 0.48GB | ≤2GB | PASS |
| Shortcut/degenerate | 100% single-class (all ABSTAIN) | ≤40% | **FAIL** |
| v0.5/ADR-0028 compat | read-only, no firewall breach | — | PASS |

Because the frozen instrument carries **no per-case evidence**, B's reranker ran with an **empty snapshot** and therefore conservatively **abstained on all 66 cases** (42 proposals overridden, 24 abstentions preserved). This is correct conservative-degradation behavior of the gate, but it makes B a **degenerate classifier** (no decision capability on this instrument) and it still fails the latency gate.

**B verdict:** meets the conservative gates (E0, Group E, abstention, footprint) but is **degenerate** and fails latency + shortcut gates. **B NOT confirmed as a deployable selection.**

## 3. Decision rule application (§5/§7)

> If A is confirmed → select A; else if B is confirmed → select B; else if C is confirmed → select C; else REQUIRE HUMAN REVIEW.

- A confirmed? **No** (fails deciding Group E).
- B confirmed? **No** (degenerate; fails latency + shortcut).
- C built? **No.**

→ **Selection deferred to human review (REQUIRED). No model selected for Phase 6 target use.**

## 4. What this establishes

1. The frozen 66-case instrument exposes that **no current realization** of A or B satisfies the full gate set — primarily because (a) the instrument carries no evidence/snapshot content (so B degrades to all-abstain) and (b) latency/abstention gates are not met by the 0.5B/0.36B decoder architecture on CPU-offload GPU path.
2. Architecture B's **isolation properties are proven** by contract tests + audit: snapshot-only, read-only, no hidden-state/gradient/subspace/privileged signal, firewall intact. The gate logic itself is sound; its deployment value depends on a populated evidence snapshot.

## 5. Open decisions for human review

- **Accept B as a conservative fallback** (all-abstain-by-default) with the explicit caveat that it cannot propose under the current instrument? Or require a populated-snapshot evaluation before any B selection?
- **Authorize a populated-evidence evaluation** (provide an evidence corpus + snapshot ids) to validate B's non-degenerate behavior and re-score Group E/abstention/latency in a realistic setting?
- **Authorize Architecture C** (separate small encoder / hybrid) build+evaluation per §3?
- **Revisit the latency gate** (≤0.5s) given the 0.5B decoder on this hardware cannot meet it without a heavier quantization/engine change — is the gate hardware-relative or absolute?
- **Finalize the 6D record**: dataset, 6D results, A/B evaluations, and this report are complete; the *selection* decision is the only outstanding item.

## 6. Artifacts

- `6c-artifact-manifest.json` — updated with `architecture_b` block (build, audit, eval, decision).
- `6d_evaluation_results.json` — A candidates (raw 6D).
- `6d_evaluation_B_results.json` — B evaluation (raw 6D).
- `theo-slm-phase-6d-comparative-report.md` — A evaluation report (prior).
- `theo-slm-phase-6d-architecture-b-source-audit.md` — B pre-execution audit.
- `theo-core/scripts/model_selection/arch_b/b_architecture.py` + `test_b_contract.py`.

**L5 status:** `STAGE1_PASS_STAGE2_BUILT_REMEDIATION_FAILED_6D_EVALUATED_B_NOT_CONFIRMED_AWAIT_HUMAN_REVIEW` — **STOP. Awaiting human decision on the open items above before any model is finalized.**
