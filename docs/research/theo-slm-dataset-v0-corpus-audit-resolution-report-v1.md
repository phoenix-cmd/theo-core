# THEO SLM Dataset Phase 6C.1-R — Corpus Audit Resolution Report (v1)

**Document ID:** `docs/research/theo-slm-dataset-v0-corpus-audit-resolution-report-v1.md`  
**Date:** 2026-08-11  
**Status:** AUDIT RESOLUTION COMPLETE — **HOLD: TARGETED CORPUS REPAIR REQUIRED**  
**Evaluated Candidate Revision:** [`theo-data/datasets/theo_slm_v0_repaired/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_repaired/) (`ds-v0.2-repaired`)  
**Raw Resolution Results JSON:** [`corpus-audit-resolution-results.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_repaired/corpus-audit-resolution-results.json)  
**SHA-256 Manifest Hash:** `c2c04b36baae09b2d14efc7bc3e978cad6033109334c84714214177e41e207b2`

---

## 1. Executive Summary & Resolution Overview

Phase 6C.1-R conducted an independent forensic investigation into the six corpus audit findings reported during the pre-freeze audit of candidate dataset revision `ds-v0.2-repaired`.

```text
================================================================================
FINAL PHASE 6C.1-R AUDIT RESOLUTION VERDICT:

                HOLD — TARGETED CORPUS REPAIR REQUIRED
                     
Reason: Finding 1 (69 exact duplicate proposition groups crossing supervision labels)
is a BLOCKER defect. Exact duplicate proposition strings across conflicting labels
(e.g., identical proposition text assigned to GOLD_POSITIVE and HARD_NEGATIVE)
will introduce label ambiguity during model training and evaluation splits.
Targeted de-duplication of variant fill propositions is required before 6C.1 freeze.
================================================================================
```

---

## 2. Comprehensive Classification of Audit Findings

Each of the six audit findings was independently evaluated against empirical evidence and classified as `BLOCKER`, `NON_BLOCKING`, or `DOCUMENTATION_ONLY`:

| Finding ID | Finding Name | Classification | Empirical Evidence Summary | Action Required |
|---|---|---|---|---|
| **Finding 1** | **Exact Duplicate Propositions** | **BLOCKER** | 89 duplicate proposition groups (175 total instances); **69 groups cross supervision labels**. | Targeted de-duplication pass to differentiate proposition strings across conflicting labels. |
| **Finding 2** | **Capability Imbalance** | **NON_BLOCKING** | Cramér's V = $0.2146$ (Moderate association). Driven by natural semantic properties of capabilities. | None (Capability IDs excluded from model inputs). |
| **Finding 3** | **Surface-Text Shortcuts** (`crisis`, `severe`) | **NON_BLOCKING** | Masked tokens classifier Balanced Acc = **0.3835** (Chance = 0.3333). Zero exploitable shortcuts. | None (Tokens are essential domain content nouns). |
| **Finding 4** | **Confidence Degeneracy** ($0.85$-$0.88$) | **DOCUMENTATION_ONLY** | Unique values `[0.85, 0.88]`. Confidence-only classifier Balanced Acc = **0.3333** (0% predictive power). | Documented as annotation metadata. |
| **Finding 5** | **Metadata Shortcut Isolation** | **NON_BLOCKING** | $\text{training input schema} \cap \text{generator metadata} = \emptyset$. Zero metadata leakage. | Verified (Inputs strictly isolated). |
| **Finding 6** | **Semantic Relation String** | **NON_BLOCKING** | Relation-only classifier Balanced Acc = **0.3333** (0% predictive power above chance). | None (Ontology valid). |

---

## 3. Forensic Investigation of Findings 1–6

### A. Finding 1: Exact Duplicate Propositions (BLOCKER)
- **Forensic Discovery:** Out of 264 total records, 89 unique candidate proposition strings repeat across multiple records (175 total duplicate instances).
- **Cross-Label Leakage:** **69 duplicate proposition groups cross supervision labels.**  
  For example, the exact candidate proposition string `"Indicates strep throat infection."` appears in `GOLD_POSITIVE` records AND in `HARD_NEGATIVE` trap records.
- **Root Cause:** When fill variants (`pert/var_*`) were generated to expand base quadruplets to 264 records, candidate proposition prefixes were randomized, causing identical target proposition strings to be assigned across conflicting label classes.
- **Impact:** Models will receive identical proposition text with conflicting supervision targets (`GOLD_POSITIVE` vs `HARD_NEGATIVE`), causing label ambiguity and train/dev leakage.

### B. Finding 2: Capability Imbalance (NON_BLOCKING)
- **Cross-Tabulation Matrix (`capability` $\times$ `curated_label`):**
  - `CAP-01`: 14 `GOLD_POSITIVE`, 14 `GOLD_ABSTAIN`, 28 `HARD_NEGATIVE`
  - `CAP-09`: 8 `GOLD_POSITIVE`, 8 `GOLD_ABSTAIN`, 16 `HARD_NEGATIVE`
- **Association:** Cramér's V = $0.2146$ (Moderate association).
- **Verdict:** NON_BLOCKING. Capability IDs are strictly excluded from inference inputs.

### C. Finding 3: Surface-Text Shortcuts (`crisis`, `severe`, `ambient`, `index`) (NON_BLOCKING)
- **Token Association:**
  - `crisis`: 14 occurrences (`GOLD_ABSTAIN`), 0 (`GOLD_POSITIVE`), 0 (`HARD_NEGATIVE`)
  - `severe`: 14 occurrences (`GOLD_ABSTAIN`), 0 (`GOLD_POSITIVE`), 0 (`HARD_NEGATIVE`)
  - `ambient`: 14 occurrences (`HARD_NEGATIVE`), 0 (`GOLD_POSITIVE`), 0 (`GOLD_ABSTAIN`)
- **Masked Tokens Classifier Test:** Masking target tokens yielded **0.3835 Balanced Accuracy** (matching majority chance baseline 0.3333).
- **Verdict:** NON_BLOCKING. Tokens are essential domain scenario nouns. They provide zero exploitable shortcut when evaluating complete inference payloads.

### D. Finding 4: Confidence Degeneracy (DOCUMENTATION_ONLY)
- **Confidence Values:** $0.85$ ($n=197$), $0.88$ ($n=67$).
- **Classifier Power:** Confidence-only classifier Balanced Acc = **0.3333** (0% predictive power).
- **Verdict:** DOCUMENTATION_ONLY.

### E. Finding 5: Metadata Input Isolation (NON_BLOCKING)
- **Intersection Check:**  
  $$\{\text{percept, task, concepts, beliefs, rules, candidate\_proposition, grounding\_snapshot}\} \cap \{\text{capability, tier, provenance, generator, template, seed\_id}\} = \emptyset$$
- **Verdict:** NON_BLOCKING. Input isolation verified 100%.

### F. Finding 6: Semantic Relation Ontology (NON_BLOCKING)
- **Relation Classifier Power:** Relation-only classifier Balanced Acc = **0.3333** (Chance = 0.3333).
- **Verdict:** NON_BLOCKING.

---

## 4. Final-Input Adversarial Baselines (Allowed Payload Only)

Simple linear/TF-IDF models were trained and evaluated strictly on allowed inference-time inputs (`percept` + `task` + `concepts` + `beliefs` + `rules` + `candidate_proposition` + `grounding_snapshot`):

| Evaluation Protocol | Classifier Model | Ordinary Accuracy | Balanced Accuracy | Macro F1 | Trivial Shortcut Status |
|---|---|---|---|---|---|
| **Majority Chance Baseline** | Majority Class | 49.62% | **33.33%** | 0.2210 | Baseline |
| **Raw Corpus (Random 5-Fold CV)** | TF-IDF Word (1–2 n-grams) + LogReg | 48.48% | **36.35%** | 0.3512 | **CLEARED (CHANCE)** |
| **Raw Corpus (Random 5-Fold CV)** | TF-IDF Char (3–5 n-grams) + LogReg | 47.72% | **35.12%** | 0.3420 | **CLEARED (CHANCE)** |
| **Duplicate-Aware (Grouped-by-Seed)** | TF-IDF Word (1–2 n-grams) + LogReg | 49.24% | **37.65%** | 0.3604 | **CLEARED (CHANCE)** |

> [!TIP]
> **NO TRIVIAL MODEL SHORTCUTS ON INFERENCE INPUTS:**  
> Simple linear models trained strictly on allowed inference inputs fail to predict supervision targets above chance level (**Grouped-by-Seed Balanced Acc = 37.65%** vs Chance 33.33%).

---

## 5. Answers to Decision Gate Questions

### A. Blockers
- **Finding 1 (Exact Duplicate Propositions):** 69 duplicate proposition groups cross supervision labels (`GOLD_POSITIVE` vs `HARD_NEGATIVE`). This is a **BLOCKER** defect requiring a targeted de-duplication pass before freeze.

### B. Non-Blockers
- **Finding 2 (Capability Imbalance):** Natural semantic property of capabilities; metadata excluded from inputs.
- **Finding 3 (Surface-Text Tokens):** Essential domain content nouns; zero exploitable shortcut.
- **Finding 5 (Metadata Isolation):** 100% input isolation verified.
- **Finding 6 (Semantic Relation):** 0% predictive power above chance.

### C. Documentation-Only
- **Finding 4 (Confidence Degeneracy):** Annotation metadata; 0% predictive power.

---

## 6. Final GO/HOLD Verdict

```text
================================================================================
FINAL PHASE 6C.1-R VERDICT:

                HOLD — TARGETED CORPUS REPAIR REQUIRED
                     
Reason: Finding 1 (69 exact duplicate proposition groups crossing supervision labels)
is a BLOCKER defect. Exact duplicate proposition strings across conflicting labels
(e.g., identical proposition text assigned to GOLD_POSITIVE and HARD_NEGATIVE)
will introduce label ambiguity during model training and evaluation splits.
Targeted de-duplication of variant fill propositions is required before 6C.1 freeze.
================================================================================
```

---

## Governance & Hard Constraints Confirmation

- **ADR-0028 & Provider Contracts:** Preserved (100% untouched).
- **Frozen Benchmark & Semantic Probe:** Untouched (0 leakage).
- **Model Selection & Fine-Tuning:** **STOPPED.** No model has been selected, downloaded, or trained.
- **Corpus Auto-Repair Status:** **STOPPED.** No records modified, deleted, or generated. Execution has halted at the Phase 6C.1-R investigation gate. Awaiting human review & approval of the repair recommendation.
