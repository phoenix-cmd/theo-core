# THEO-SLM Phase 6D — Architecture B: Populated-Evidence Evaluation

**Date:** 2026-08-26
**Authorization:** `6D follow-up: AUTHORIZE populated-evidence evaluation for Architecture B only` (no Architecture C build; no A retrain; latency gate unchanged; do not finalize 6D as "no selection" yet).
**Status:** EVALUATED — selection still **DEFERRED to human review** (neither A nor B confirmed).
**Prior result preserved:** the empty-snapshot B run (`6d_evaluation_B_results.json`, all-ABSTAIN) is retained as historical evidence and was **not overwritten**.

---

## 1. Populated-evidence corpus (frozen, hashed)

- **File:** `theo-data/datasets/theo_slm_v0_artifacts/phase-6e3/populated-evidence-corpus.json`
- **SHA-256:** `f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457`
- **Construction:** deterministically derived from the **frozen 66-case instrument** percepts (15-probe + 51-benchmark) via sentence extraction (1–4 evidence items per case). No synthesis, no fabrication of historical evidence fields.
- **Per case provides:** `case_id`, `evidence` (id + exact text), `snapshot_id`, `provenance` (derived-from / method / note), and the **existing** `expected_decision` contract (copied verbatim from the instrument).
- **Integrity:** 300-GOLD training corpus and the original probe/benchmark files are **unmodified**. The corpus is a separate, evaluation-time artifact only.
- **Distribution:** 66 cases → 50 `SHOULD_PROPOSE`, 16 `SHOULD_ABSTAIN` (matches instrument).

## 2. Architecture B refinement (bounded, backward-compatible)

- The `Decoder` (M1 Qwen2.5-0.5B-Instruct, 4-bit) was extended to **consume per-case evidence** and **cite snapshot IDs** in `evidence_refs`. Without evidence, `run()` reproduces the original empty-snapshot all-ABSTAIN behavior exactly (historical result reproducible).
- The **`EvidenceReranker` is unchanged**: still snapshot-only, read-only, deterministic evidence gate; isolation constraints (no hidden-state / gradient / subspace / privileged info / firewall breach) verified re-passed by contract tests **C1–C6**.
- Re-audit note: only the decoder prompt/interface changed; the reranker and all isolation guarantees from `theo-slm-phase-6d-architecture-b-source-audit.md` remain valid.

## 3. Results (same 66 cases, same §7 gates)

| Gate | Threshold | Populated B | Empty-snapshot B | Outcome |
|---|---|---|---|---|
| E0 format reliability | ≤2% fail | 100% pass | 100% pass | **PASS** |
| Grounding validity | 100% | 100% | (n/a) | **PASS** |
| Distractor rejection (Group E, deciding) | ≥80% | **0.333** (1/3) | 1.0 (degenerate) | **FAIL** |
| Abstention accuracy | ≥90% | **0.375** (6/16) | 1.0 (degenerate) | **FAIL** |
| Latency | ≤0.5s | **3.26s** | 2.58s | **FAIL** |
| VRAM / footprint | ≤2GB | 0.48GB | 0.48GB | **PASS** |
| Shortcut resistance (proxy) | ≤40% | **0.672** | (degenerate) | **FAIL (proxy)** |
| E5 semantic novelty | ≥40% | N/A (no labels) | N/A | not scored |
| E6 decision relevance | ≥30% | N/A (no labels) | N/A | not scored |
| v0.5 / ADR-0028 compatibility | all green | read-only reranker; no firewall breach | same | **PASS** |

- **Accuracy:** 0.727 (48/66) — up from 0.242 in empty-snapshot mode.
- **Non-degenerate:** predictions = 52 `SHOULD_PROPOSE` / 14 `SHOULD_ABSTAIN` (was 66/0).
- **Reranker activity:** `upheld`=52, `overridden_no_evidence`=12, `kept_abstain`=2. The gate correctly caught 12 unsupported proposals; but 52 decoder proposals (many on expected-ABSTAIN / distractor cases) were upheld because the decoder cited valid (present) evidence.

## 4. Interpretation

With evidence present, the reranker verifies the decoder's citations and **upholds proposals** instead of overriding them. Architecture B therefore becomes non-degenerate and accuracy improves (0.242 → 0.727). However:

1. **B's decision quality is bounded by the underlying decoder (M1_base).** The reranker only removes *uncited* proposals; it cannot judge whether cited evidence actually *supports* the decision (that would require semantic matching / hidden states, which are excluded). Consequently B still proposes on distractor and unsupported cases → fails the deciding **Group E** and **abstention** gates, exactly as A did.
2. **B's verified, isolated value is the absent-evidence safety fallback** (empty-snapshot mode), where it conservatively abstains (already demonstrated). Under populated evidence it adds no accuracy beyond the decoder.
3. **Latency and shortcut gates remain unmet** (hardware-bound decoder path; predictions partly track percept length as a spurious feature).

## 5. Decision-rule application (§5/§7)

- A confirmed? **No** (fails Group E, 0%).
- B confirmed? **No** — empty-snapshot B is degenerate; populated B fails Group E (0.333), abstention (0.375), latency, and the shortcut proxy.
- C built? **No** (explicitly not authorized this step).

→ **No architecture confirmed. Selection remains DEFERRED to human review. 6D is NOT finalized as "no selection" per the authorization; the deferred state is preserved.**

## 6. Open items for human review

1. Accept the populated-eval finding that **B's benefit is the absent-evidence fallback only**, and that under present evidence it does not outperform the decoder baseline?
2. Is the **latency gate (≤0.5s)** absolute or hardware-relative? The 0.5B decoder on this GPU/offload path cannot meet it without heavier optimization (separate authorization).
3. Authorize **Architecture C** (shared-encoder / two-head) build+evaluation per §3, to test whether a separately-trained evidence encoder clears Group E where B's deterministic gate cannot?
4. Treat the populated corpus (`populated-evidence-corpus.json`, hash recorded) as a **standing evaluation fixture** for future B/C runs?

## 7. Artifacts

- `populated-evidence-corpus.json` — frozen corpus (SHA-256 above).
- `6d_evaluation_B_populated_results.json` — populated B raw results (new file).
- `6d_evaluation_B_results.json` — **historical empty-snapshot B result, preserved unchanged**.
- `6c-artifact-manifest.json` — updated with `architecture_b.populated_evidence_eval` (prior `phase_6d_b_eval` retained).
- `theo-slm-phase-6d-architecture-b-comparative-report.md` — prior A-vs-B report (unchanged).
- `theo-slm-phase-6d-architecture-b-source-audit.md` — pre-execution audit (reranker unchanged; still valid).

**L5 status:** `...6D_EVALUATED_B_POPULATED_AWAIT_HUMAN_REVIEW` — **STOP. Awaiting human decision on the open items; 6D not finalized.**
