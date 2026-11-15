# Architecture B — Independent Source Audit (Pre-Execution)

**Audited module:** `theo-core/scripts/model_selection/arch_b/b_architecture.py`
**Contract tests:** `theo-core/scripts/model_selection/arch_b/test_b_contract.py` (ALL PASSED)
**Audit date:** 2026-08-26
**Author of audit:** independent review of the B implementation against the 6C plan §3/§5 and the authorization constraints.

---

## 1. Authorization constraints vs implementation

| Required constraint | Status | Evidence |
|---|---|---|
| Decoder + snapshot-only reranker (§3 B) | MET | `ArchitectureB` = `Decoder` (M1 4-bit) + `EvidenceReranker(Snapshot)` |
| Reranker is a *separate* component (not assumed) | MET | `EvidenceReranker` is an independent class; pipeline wraps decoder output |
| Reranker input limited to approved snapshot/evidence interface | MET | `rerank(decision, evidence_refs)` only reads `Snapshot.get`/`ids` (C1/C2) |
| No hidden-state / subspace / gradient signal | MET | Source scan negative for `hidden_states`, `backward(`, `subspace`, `cos(H1)`, `.embedding`, `logits`, `peft`, `LoraConfig` (C3) |
| No additional privileged information | MET | No 300-GOLD / label / `REJECT` references; no training-set access (C4) |
| Smallest viable B, not a new architecture family | MET | Deterministic evidence-gate reranker (no new model family) |
| Cannot bypass/mutate symbolic state | MET | `Snapshot` exposes only read methods; write methods raise; content unchanged after rerank (C1/C2) |
| Cannot see disallowed information | MET | Reranker signature = `(decision, evidence_refs)` only; no tensors/labels passed (C3) |
| Architectural firewall (no `theo_core`/`theo_providers` import) | MET | Import set = `__future__, json, os, re, time, torch, transformers` (C5) |
| Preserve M1/M2 base + 6D results unchanged | MET | B reads M1_base only; no writes to prior artifacts |
| No 300-GOLD modification | MET | Dataset untouched; reranker needs no training data |
| Contract / labels / gates / ADR-0028 unchanged | MET | B emits the same `{"decision","reasoning"}` contract JSON; gates unaltered |

## 2. Deliberate design decision (flagged)
The plan §3 describes B's reranker as "a separate small encoder **trained only on the evidence-relevance
family.**" The bounded build here implements the reranker as a **deterministic, snapshot-only evidence
gate** rather than a separately-trained cross-encoder. Rationale:
- The frozen artifacts contain **no evidence-text corpus** to train a relevance encoder (the 300-GOLD
  `grounding_snapshot` carries only *ids*, not the referenced text), so training would be unsupported.
- The user mandated "smallest viable B ... not a new architecture family" — the evidence gate is the
  minimal mechanism exercising the identical decision logic (override unsupported/distractor proposals).
- The gate is fully contract-testable (C6) and demonstrates the architecture + its isolation.

This is a **defensible minimal B**; a trained-encoder variant remains a downstream option requiring a
populated evidence corpus and a separate authorization.

## 3. Behavioral verification (contract test C6)
- Empty snapshot + `SHOULD_PROPOSE` → overridden to `SHOULD_ABSTAIN` (`overridden_no_evidence`).
- Empty snapshot + `SHOULD_ABSTAIN` → preserved (`kept_abstain`).
- Populated snapshot + verified ref → `SHOULD_PROPOSE` upheld (`upheld`).
- Populated snapshot + unverified ref → overridden to `SHOULD_ABSTAIN`.

## 4. Audit verdict
The B implementation **conforms** to the authorization and the §3/§5 definition within the bounded
scope. The reranker is provably snapshot-only, read-only, and isolated from hidden-state/gradient/
subspace/privileged signals. **Cleared for execution** of the 6D evaluation on Architecture B.

**Residual caveat (carried into the 6D report):** because the frozen 66-case instrument carries no
per-case evidence/snapshot content, B's reranker will operate with an empty snapshot at evaluation time
and therefore conservatively abstain on every case lacking verified evidence. This exercises the gate
mechanism correctly but means B's Group E / abstention results on this instrument reflect the
*conservative-degradation* behavior, not a populated-snapshot deployment. Final B validation requires a
populated snapshot (separate authorization).
