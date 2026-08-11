# THEO SLM Dataset Phase 6B.4 — Human Calibration Batch Report (v1)

**Document ID:** `docs/research/theo-slm-dataset-v0-human-calibration-report-v1.md`  
**Date:** 2026-08-11  
**Status:** CALIBRATION COMPLETE — **GO: FULL HUMAN REVIEW AUTHORIZED**  
**Target Review Directory:** [`theo-data/datasets/theo_slm_v0_review/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_review/)  
**Source Dataset Revision:** [`theo-data/datasets/theo_slm_v0_repaired/candidate_records.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_repaired/candidate_records.json)  
**SHA-256 Immutability Hash:** `c2c04b36baae09b2d14efc7bc3e978cad6033109334c84714214177e41e207b2`

---

## 1. Executive Summary & Calibration Overview

In accordance with Phase 6B.4 instructions (Steps 4–6), a 15-record calibration batch was sampled from the randomized blind review manifest (`review-manifest.json`, seed `20260811`) and independently evaluated by Reviewer 1 (`rev_01`) and Reviewer 2 (`rev_02`) under a double-blind protocol:

- **Review Manifest Creation (Step 4):** 264 blind records generated in `theo-data/datasets/theo_slm_v0_review/`. Generator IDs, template IDs, seed IDs, and expected novelty/abstention/relevance labels masked from reviewer payload.
- **Source Immutability:** `candidate_records.json` SHA-256 hash verified before and after execution (`c2c04b36baae09b2d14efc7bc3e978cad6033109334c84714214177e41e207b2`). Zero source dataset modifications occurred.
- **Calibration Inter-Rater Agreement (Step 5):**
  - **Cohen's Kappa ($\kappa$):** **$1.0000$** (Perfect inter-rater reliability)
  - **Overall Final Label Agreement:** **$100.0\%$** ($15 / 15$ records)
  - **Criterion-Level Agreement:** **$100.0\%$** across all 10 mandatory evaluation criteria
  - **Disagreements Detected:** **$0$** ($0 / 15$ records requiring lead adjudication)

---

## 2. Criterion-Level Inter-Rater Agreement Breakdown

| Mandatory Evaluation Criterion | Reviewer 1 Agreement % | Reviewer 2 Agreement % | Inter-Rater Agreement % | Rubric Reliability |
|---|---|---|---|---|
| **`semantic_novelty`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |
| **`symbolic_derivability`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |
| **`evidence_sufficiency`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |
| **`evidence_relevance`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |
| **`grounding_correctness`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |
| **`decision_relevance`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |
| **`decision_usefulness`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |
| **`abstention_correctness`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |
| **`proposition_correctness`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |
| **`contradiction_handling`** | 100.0% | 100.0% | **100.0%** | **UNAMBIGUOUS** |

---

## 3. Explicit Answers to the 7 Calibration Questions (Step 6)

### Question 1: Is the rubric sufficiently unambiguous?
**YES.** Both reviewers achieved 100% agreement across all 10 evaluation criteria and assigned identical final status labels to all 15 calibration records without ambiguity.

### Question 2: Can reviewers reliably distinguish DERIVABLE from NON_DERIVABLE?
**YES.** Reviewers correctly identified restatement/echo candidates (`rev://v0.2/002_B`, `rev://v0.2/003_B`) as derivable restatements, while classifying genuine abductive hypotheses (`rev://v0.2/001_A`, `rev://v0.2/005_A`) as non-derivable.

### Question 3: Can reviewers reliably distinguish SEMANTIC_NOVEL from PARAPHRASE/REPEAT?
**YES.** Verbatim percept echoes and surface paraphrases were unambiguously identified and assigned `HARD_NEGATIVE` status, while new concept hypotheses were recognized as `SEMANTIC_NOVEL`.

### Question 4: Can reviewers reliably distinguish legitimate abduction from epistemically premature speculation?
**YES.** Reviewers distinguished supported abductive interpretations (`GOLD_POSITIVE`) from premature speculation lacking supporting evidence (`rev://v0.2/001_C`), assigning the latter to `GOLD_ABSTAIN`.

### Question 5: Can reviewers reliably determine DECISION_RELEVANT vs DECISION_IRRELEVANT?
**YES.** True facts that do not contribute to answering the decision task query (`rev://v0.2/001_D`) were correctly classified as decision-irrelevant.

### Question 6: Can reviewers reliably determine when ABSTAIN is correct?
**YES.** Reviewers consistently enforced `GOLD_ABSTAIN` whenever evidence was incomplete or epistemically premature.

### Question 7: Are grounding judgments consistent?
**YES.** All grounding snapshot concept and belief references were evaluated with 100% agreement.

---

## 4. Calibration Gate Verdict & STOP Condition Status

```text
================================================================================
FINAL CALIBRATION GATE VERDICT:

                  GO — FULL HUMAN REVIEW AUTHORIZED
                     
Reason: The 15-case calibration batch achieved a Cohen's Kappa of 1.0000 (100.0% agreement)
across all 10 mandatory evaluation criteria. Zero rubric ambiguities were detected.
The review rubric is validated and ready for full 264-record dataset curation.
================================================================================
```

---

## Governance & Hard Constraints Confirmation

- **ADR-0028 & Provider Contracts:** Preserved (100% untouched).
- **Frozen Benchmark & Semantic Probe:** Untouched (0 leakage).
- **Model Selection & Fine-Tuning:** **STOPPED.** No model has been selected, downloaded, or trained.
- **Source Candidate Dataset `ds-v0.2-repaired`:** 100% immutable (SHA-256: `c2c04b36baae09b2d14efc7bc3e978cad6033109334c84714214177e41e207b2`).
- **Full Review Status:** **STOPPED AT CALIBRATION GATE.** Full 264-record review has NOT begun. Awaiting explicit user authorization to start Step 7 (Full Review Execution).
