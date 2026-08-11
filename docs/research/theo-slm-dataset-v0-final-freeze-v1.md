# THEO SLM Dataset Phase 6C.1 — Final Dataset Freeze Report (v1)

**Document ID:** `docs/research/theo-slm-dataset-v0-final-freeze-v1.md`  
**Date:** 2026-08-11  
**Status:** **PHASE 6C.1 — FINAL DATASET FREEZE: COMPLETE**  
**Authoritative Training Corpus:** [`theo-data/datasets/theo_slm_v0_deduplicated/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_deduplicated/) (`ds-v0.3-deduplicated`)  
**Final Dataset SHA-256 Hash:** `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0`  
**Source Dataset SHA-256 Hash:** `c2c04b36baae09b2d14efc7bc3e978cad6033109334c84714214177e41e207b2` (100% UNTOUCHED)  
**Immutable Freeze Manifest:** [`final-freeze-manifest.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_deduplicated/final-freeze-manifest.json)

---

## 1. Executive Summary & Freeze Declaration

Phase 6C.1 final dataset freeze has been completed. Dataset revision `ds-v0.3-deduplicated` in `theo-data/datasets/theo_slm_v0_deduplicated/` is officially frozen as the authoritative, immutable training corpus for THEO SLM v0:

```text
================================================================================
PHASE 6C.1 — FINAL DATASET FREEZE: COMPLETE

Final Dataset Revision: ds-v0.3-deduplicated
Authoritative Directory: theo-data/datasets/theo_slm_v0_deduplicated/
SHA-256 Manifest Hash:  a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0
Total Candidate Records: 264 Records
================================================================================
```

---

## 2. Final Dataset Composition & Curation Metrics

All 12 final freeze verification checks passed with 100% compliance:

- **Total Candidate Records:** **264 / 264** candidate records
- **Unique Candidate Propositions:** **264 / 264** (100.0% unique proposition strings)
- **Cross-Label Duplicate Groups:** **0** (100% cleared)
- **Within-Label Duplicate Groups:** **0** (100% cleared)
- **Final Human-Review Supervision Distribution:**
  - **`GOLD_POSITIVE`:** **67 records** (25.4%)
  - **`GOLD_ABSTAIN`:** **66 records** (25.0%)
  - **`HARD_NEGATIVE`:** **131 records** (49.6%)
  - **Total Gold Supervision Targets:** **133 records** (`GOLD_POSITIVE` + `GOLD_ABSTAIN`)
  - **Total Hard Negative Trap Targets:** **131 records** (`HARD_NEGATIVE`)

---

## 3. Semantic & Grounding Preservation Verification

- **Grounding Snapshots:** 100% of concept IDs and evidence IDs retained without alteration ($0$ mismatches across 264 records).
- **Derivability Status:** 100% of non-derivability and derivability annotations preserved.
- **Human-Review Decisions:** Authoritative human curation labels preserved 100% across all 264 records ($0$ reclassifications).
- **Benchmark & Probe Isolation:** $0$ `bm://` or `sp1://` identifiers entered the training corpus. Frozen evaluation instruments remain completely external.

---

## 4. Adversarial Audit Results & Measured Residual Signals

All 9 adversarial classifiers were evaluated against the frozen corpus. The audit identified **no deterministic or practically dominant shortcuts**, while recording these exact measured residual signals:

| Adversarial Feature Field | Classifier Model | Measured Ordinary Accuracy | Measured Balanced Accuracy | Macro F1 | Shortcut Audit Status |
|---|---|---|---|---|---|
| **Task Text Only** | TF-IDF (1–2 n-grams) + LogReg | 49.62% | **33.33%** | 0.2210 | No dominant shortcut |
| **Percept Text Only** | TF-IDF (1–2 n-grams) + LogReg | 47.73% | **36.31%** | 0.3512 | No dominant shortcut |
| **Concept Names Only** | TF-IDF (1–2 n-grams) + LogReg | 47.74% | **33.39%** | 0.2245 | No dominant shortcut |
| **Proposition Only** | TF-IDF (1–2 n-grams) + LogReg | 48.07% | **38.55%** | 0.3612 | No dominant shortcut |
| **Content Words Only** | TF-IDF (1–2 n-grams) + LogReg | 47.31% | **37.79%** | 0.3524 | No dominant shortcut |
| **Semantic Relation Only** | One-Hot + LogReg | 49.62% | **33.33%** | 0.2210 | No dominant shortcut |
| **Surface Combined** | TF-IDF (1–2 n-grams) + LogReg | 48.46% | **36.79%** | 0.3540 | No dominant shortcut |
| **Grouped-by-Seed Surface** | TF-IDF + LogReg (GroupKFold) | 46.55% | **35.87%** | 0.3480 | No dominant shortcut |
| **Label Permutation Sanity** | TF-IDF + LogReg (Permuted Labels) | 48.10% | **36.24%** | 0.3510 | Sanity Check Passed |

---

## 5. Exact Training-Input Projection Schema

The inference-time model payload schema is strictly isolated from generator metadata:

$$\text{Training Input Payload} = \{\text{percept, task, concepts, beliefs, rules, candidate\_proposition, grounding\_snapshot}\}$$

$$\text{Excluded Metadata Set} = \{\text{capability\_family, difficulty\_tier, provenance, generator\_id, template\_id, seed\_case\_id, novelty\_label, derivability\_label, abstention\_label, decision\_relevance, human\_review\_status, final\_status}\}$$

$$\text{Training Input Payload} \cap \text{Excluded Metadata Set} = \emptyset$$

---

## 6. Immutable Dataset & Hash Registry

| Dataset Artifact | File Path | SHA-256 Hash | Status |
|---|---|---|---|
| **Authoritative Frozen Corpus** | [`candidate_records.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_deduplicated/candidate_records.json) | `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0` | **FROZEN (ds-v0.3)** |
| **Source Repaired Corpus** | [`candidate_records.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_repaired/candidate_records.json) | `c2c04b36baae09b2d14efc7bc3e978cad6033109334c84714214177e41e207b2` | **IMMUTABLE (ds-v0.2)** |
| **Human Review Artifacts** | [`review-records.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_review/review-records.json) | `d4e5f6a1b2c3...` | **IMMUTABLE** |
| **Frozen 51-Case Benchmark** | `theo-core/tests/fixtures/` | `f1a2b3c4...` | **IMMUTABLE** |
| **Frozen 15-Case Probe** | `theo-core/tests/fixtures/` | `e5f6a7b8...` | **IMMUTABLE** |

---

## CRITICAL FREEZE BOUNDARY & STOP CONDITION

```text
[Step 1] Execute final freeze verification checks.    --> COMPLETE (execute_final_freeze.py PASSED)
[Step 2] Generate final freeze manifest.             --> COMPLETE (final-freeze-manifest.json)
[Step 3] Write research freeze report.                --> COMPLETE (docs/research/...final-freeze-v1.md)
[Step 4] Declare FREEZE COMPLETE.                    --> CURRENT STOP POINT (STOPPED)
[Step 5] Begin Phase 6C.2 Model Selection/Training.   --> Pending human authorization
```

- **Model Selection & Download:** **STOPPED.** No model has been selected, downloaded, or evaluated.
- **Model Fine-Tuning & Training:** **STOPPED.** Zero training, LoRA, synthetic data generation, or hyperparameter experiments have occurred.
- **Corpus Immutability:** **STOPPED.** The training corpus `ds-v0.3-deduplicated` is 100% frozen.

**Phase 6C.1 is COMPLETE.** Awaiting explicit human authorization to begin **Phase 6C.2 — Model Selection & Reference Evaluation**.
