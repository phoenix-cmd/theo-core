# THEO-SLM Phase 6D — Final A / B / C Comparative Evaluation & Model Selection Report

**Date:** 2026-08-26
**Authorization:** `6D Architecture C: AUTHORIZED for bounded build + evaluation` (after A, B, and B populated-evidence evaluations).
**Status:** ALL THREE candidate architectures evaluated on the frozen fixture. **No architecture confirmed. Selection DEFERRED to human review** (NOT finalized).
**Scope preserved:** 300-GOLD unchanged; frozen 66-case instrument unchanged; populated-evidence corpus `f99ca025…ac6457` unchanged; all A/B artifacts and thresholds/decision rules unchanged; ADR-0028 / v0.5 interface constraints honored; ≤0.5s latency gate unchanged.

---

## 1. Candidates evaluated

| Arch | Description | Build | Training in this step |
|---|---|---|---|
| **A** | Decoder-only SLM (M1 Qwen2.5-0.5B) ± QLoRA adapters | prior | adapters v1/v2 (failed); base M1_base used as baseline |
| **B** | Decoder (M1) + snapshot-only reranker (evidence gate) | prior | none (deterministic gate); decoder consumes evidence |
| **C** | Shared encoder (M1) + decision head + evidence/relation head, read-only `Ports` gate | this step | **zero-shot** (no supervised heads; deferred — see §5) |

All three use the **same approved M1 binding** (`7ae55760…`), the same frozen 66-case fixture (15-probe + 51-benchmark), and the same §7 behavioral gates.

## 2. Results (populated-evidence mode; same §7 gates)

| Gate (threshold) | A (M1_base) | B (populated) | C (populated) | Required |
|---|---|---|---|---|
| E0 format reliability (≤2% fail) | 77.3% pass | 100% | 100% | **A fail / B,C pass** |
| Grounding validity (100%) | n/a | 100% | 100% | B,C pass |
| Distractor rejection Group E (≥80%, **deciding**) | 0% | 33.3% | 33.3% | **all FAIL** |
| Abstention accuracy (≥90%) | 31.3% | 37.5% | 6.25% | **all FAIL** |
| Latency (≤0.5s) | 2.84s | 3.26s | 4.63s | **all FAIL** |
| VRAM / footprint (≤2GB) | 0.48GB | 0.48GB | 0.48GB | all pass |
| Shortcut resistance (≤40% proxy) | — | 0.672 fail | 0.659 fail | **B,C fail** |
| E5/E6 novelty | N/A | N/A | N/A | not scored |
| v0.5 / ADR-0028 | baseline | pass | pass | B,C pass |

- **Accuracy:** A 0.636 · B 0.727 · C 0.742.
- **B empty-snapshot (historical):** all-ABSTAIN, degenerate (0.242 acc) — preserved as historical evidence, not overwritten.
- **C empty-snapshot:** degenerate-by-construction (all-ABSTAIN), same reasoning as B; only populated mode reported as the meaningful result.

## 3. Decision-rule application (§5/§7)

> If A confirmed → A; else if B confirmed → B; else if C confirmed → C; else REQUIRE HUMAN REVIEW.

- **A confirmed?** No — fails deciding **Group E (0%)**; abstention far below 90%.
- **B confirmed?** No — populated B fails **Group E (33.3%)**, **abstention (37.5%)**, latency, shortcut proxy; empty B is degenerate.
- **C confirmed?** No — populated C fails **Group E (33.3%)**, **abstention (6.25%)**, latency, shortcut proxy; most proposal-aggressive of the three.

→ **No architecture clears all gates. Selection DEFERRED to human review.**

## 4. Interpretation

1. **The deciding axis is Group E (distractor rejection) and abstention**, exactly as the 6C plan projected (C7 is THEO's empirically weakest axis). All three candidates — including the conservative gated B and the two-head C — fail it on the frozen instrument because the underlying 0.5B decoder proposes on distractors/unsupported cases and the gates (read-only, snapshot-only) can only *filter absent-evidence* proposals, not judge whether cited evidence *supports* the decision (that would require semantic/subspace judgment, which is excluded).
2. **B's verified value is the absent-evidence safety fallback** (empty-snapshot → conservative all-abstain). Under present evidence it does not exceed the decoder baseline.
3. **C's second head only adds candidate evidence refs and never countermands the decision head**, so it upholds nearly every PROPOSE → worst abstention. A genuinely corrective C would need the evidence head to *down-weight/override* unsupported proposals (supervised training on an evidence-relevance corpus, which does not exist in the frozen artifacts).
4. **Latency fails for all** on this GPU/offload path (0.5B decoder ≈ 3–5s/case), independent of architecture quality — a hardware/config issue, not a model-capability issue.

## 5. Caveats / honest limitations

- **C was evaluated zero-shot.** Supervised C (trained decision + evidence heads) was **not** built because (a) per-case GOLD input texts live in `candidate_records_batch*` under a different `case_id` namespace than `adjudication.json`, and faithfully reconstructing the prior join is fragile and risks misusing the frozen dataset; (b) no evidence-relevance corpus exists to supervise an evidence head. A supervised C (and a trained-encoder B) remain **separate authorizations**.
- **E5/E6** are not scored: the frozen instrument carries no novelty/grounding-support labels.
- **Shortcut resistance** is reported as an informational length-probe proxy (plan §7 formal probe requires representation access, which is excluded).
- **All evaluations are greedy/deterministic**; no temperature sampling.

## 6. Artifacts produced

- `arch_c/c_architecture.py` — Architecture C implementation.
- `arch_c/test_c_contract.py` — contract tests **C1–C7 ALL PASS** (Ports read-only; heads cannot mutate symbolic state; no prohibited signals; firewall; two-head-over-shared-encoder; gate logic).
- `theo-slm-phase-6d-architecture-c-source-audit.md` — independent pre-execution audit (cleared).
- `6d_evaluation_C_results.json` — C raw results (new).
- `6c-artifact-manifest.json` — updated with `architecture_c` + corrected `phase_6d` verdict.
- Prior: `6d_evaluation_results.json` (A), `6d_evaluation_B_results.json` (B empty, preserved), `6d_evaluation_B_populated_results.json`, populated corpus, B audit/report.

## 7. Open items for human review

1. **Finalize selection?** All three fail the deciding gates. Options: (a) defer/no-selection now; (b) select the least-bad on non-deciding gates (C has highest accuracy 0.742 but worst abstention; B best balances); (c) relax/qualify a gate (e.g., treat latency as hardware-relative).
2. **Latency gate:** absolute (≤0.5s) or hardware-relative? On this hardware the 0.5B decoder cannot meet it without separate optimization.
3. **Authorize a supervised C and/or trained-encoder B** requiring (i) a verified GOLD-input join and (ii) an evidence-relevance corpus — both out of scope for this bounded step.
4. **Group E / abstention investability:** given all candidates fail the deciding axis on this 0.5B base, is a larger/specialized base or a different training signal warranted (separate authorization, outside 6D selection)?

**L5 status:** `...6D_EVALUATED_ABC_AWAIT_HUMAN_REVIEW` — **STOP. Final A/B/C comparative report delivered; selection NOT finalized; awaiting human decision.**
