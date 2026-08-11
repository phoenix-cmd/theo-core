# THEO SLM Dataset Phase 6B.4 — Human Review & Curation Specification (v1)

**Document ID:** `docs/research/theo-slm-dataset-v0-human-review-v1.md`  
**Date:** 2026-08-11  
**Status:** SPECIFICATION COMPLETE — **AWAITING HUMAN APPROVAL (STOPPED)**  
**Candidate Dataset Revision:** [`theo-data/datasets/theo_slm_v0_repaired/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_repaired/) (`ds-v0.2-repaired`)  
**Target Review Directory:** [`theo-data/datasets/theo_slm_v0_review/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_review/)  
**SHA-256 Manifest Hash (`ds-v0.2-repaired`):** `6d028c2c78a0eebebc567bf854fe45d5e5cf6bd13b5bf98c569ff466f22e8ec5`

---

## 1. Review Objective & Scope

The purpose of Phase 6B.4 is to independently evaluate each candidate record in dataset revision `ds-v0.2-repaired` to determine whether it is genuinely worthy of becoming a training label for the THEO SLM.

Reviewers must evaluate candidates against the fundamental THEO hierarchy:

$$\text{DERIVABLE} \longrightarrow \text{NON_DERIVABLE} \longrightarrow \text{SEMANTIC_NOVEL} \longrightarrow \text{DECISION_RELEVANT} \longrightarrow \text{DECISION_USEFUL}$$

### The Core Evaluation Question:
> *Is this candidate a genuinely useful semantic interpretation that THEO cannot currently derive, or is it merely a restatement, paraphrase, stored-belief echo, rule echo, unsupported speculation, or decision-irrelevant fact?*

---

## 2. Review Status Labels & Taxonomies

Reviewers and adjudicators must assign candidates strictly to one of 5 final statuses:

| Status Label | Definition | Required Conditions |
|---|---|---|
| **`GOLD_POSITIVE`** | Confirmed high-quality positive training target | Non-derivable + Semantically Novel + Supported + Relevant + Useful |
| **`GOLD_ABSTAIN`** | Confirmed high-quality abstention target | Premature speculation or insufficient evidence for responsible assertion |
| **`HARD_NEGATIVE`** | High-value negative training candidate | Plausible trap, echo, paraphrase, or distractor that model must reject |
| **`REJECT`** | Malformed or flawed candidate record | Structurally broken, ungrounded entity, or logically defective |
| **`NEEDS_REVISION`** | Borderline candidate requiring editing | Valid concept but requires proposition rephrasing or grounding edit |

---

## 3. Mandatory 10-Point Review Criteria

For every candidate record, reviewers must independently score 10 evaluation criteria (binary or Likert scale) and provide a written justification:

1. **`semantic_novelty`:** Does the proposition introduce new conceptual state beyond input percepts?
2. **`symbolic_derivability`:** Is the proposition non-derivable by the symbolic engine alone?
3. **`evidence_sufficiency`:** Does the cited evidence actually support the proposition?
4. **`evidence_relevance`:** Is the cited evidence decision-relevant to the task question?
5. **`grounding_correctness`:** Are all referenced concept and belief IDs valid in the grounding snapshot?
6. **`decision_relevance`:** Does the proposition help answer the decision query?
7. **`decision_usefulness`:** Does the proposition add actionable utility?
8. **`abstention_correctness`:** Is abstention appropriate when evidence is incomplete?
9. **`proposition_correctness`:** Is the proposition syntactically clean and grammatically sound?
10. **`contradiction_handling`:** Does the proposition properly account for any environmental contradictions?

---

## 4. Positive Gold Standard Protocol (`GOLD_POSITIVE`)

A candidate record may be marked **`GOLD_POSITIVE`** ONLY when BOTH reviewers independently agree that ALL 7 conditions are satisfied:

1. **Semantic Accuracy:** The candidate interpretation is factually and logically correct in the domain context.
2. **Symbolic Non-Derivability:** The interpretation cannot be derived by symbol matching, rule execution, or taxonomy traversal.
3. **Evidence Support:** Cited evidence in `supporting_evidence_ids` provides sufficient grounding support.
4. **Decision Relevance:** The interpretation directly informs the decision query.
5. **Information Addition:** Adds genuine semantic value rather than paraphrasing existing percepts.
6. **Grounding Integrity:** All referenced concepts exist in the snapshot.
7. **Appropriate Epistemic Posture:** Confidence score is calibrated appropriately (not overconfident).

> [!CAUTION]
> **NO SINGLE-REVIEWER GOLD LABELS:**  
> A candidate CANNOT be marked `GOLD_POSITIVE` based on a single reviewer's judgment. Dual-reviewer independent consensus is strictly required.

---

## 5. Abstention Gold Standard Protocol (`GOLD_ABSTAIN`)

A candidate record may be marked **`GOLD_ABSTAIN`** when available evidence is insufficient for a responsible interpretation:

- **Crucial Distinction:** Reviewers must explicitly distinguish between:
  - *"I cannot derive this symbolically"* (Symbolic engine limitation $\rightarrow$ candidate for SLM inference).
  - *"I should not assert this because evidence is insufficient"* (Epistemic boundary $\rightarrow$ candidate for `GOLD_ABSTAIN`).
- **NEG-14 Preservation:** Epistemically premature candidates (e.g. single wet road observation without rain evidence) must be preserved as `GOLD_ABSTAIN` to train the SLM in calibrated abstention.

---

## 6. Blind Dual-Reviewer Protocol & Independence

Reviewers work completely independently under a double-blind protocol:

```text
               Blind Review Record
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
  Reviewer 1                      Reviewer 2
(Blind to R2)                   (Blind to R1)
       │                               │
       └───────────────┬───────────────┘
                       ▼
          Independent Agreement Engine
```

- **Reviewer 1 Fields:** `reviewer_1_id`, `reviewer_1_timestamp`, `reviewer_1_label`, `reviewer_1_evaluations`, `reviewer_1_reason`.
- **Reviewer 2 Fields:** `reviewer_2_id`, `reviewer_2_timestamp`, `reviewer_2_label`, `reviewer_2_evaluations`, `reviewer_2_reason`.
- Neither reviewer can view the other reviewer's scores or written reasons before submitting their own evaluation.

---

## 7. Disagreement & Adjudication Protocol

When Reviewer 1 and Reviewer 2 assign different status labels:

1. **No Automatic Majority Vote:** Disagreements are NEVER resolved by automatic algorithm or majority rule.
2. **Adjudication Engine:** Disagreements trigger an explicit adjudication step logged in `adjudication.json`.
3. **Adjudicator Requirements:** An expert lead adjudicator inspects both reviewer arguments, evaluates the candidate against the 10-point rubric, records an explicit `adjudication_reason`, and assigns the final status label.

---

## 8. Deterministic Candidate Randomization & Blindness

To prevent reviewers from inferring labels from neighboring records or generator sequence:

- **Review Ordering Seed:** Review order is deterministically shuffled using random seed `20260811`.
- **Metadata Masking:** The review interface hides:
  - `generator_id`
  - `template_id`
  - `seed_case_id`
  - `original_negative_family_id`
  - `capability_family`
  - `expected_novelty_label`
  - `expected_abstention_label`
  - `expected_relevance_label`

---

## 9. Review Interface & Machine-Readable Schema Layout

The original dataset candidate file `ds-v0.2-repaired` remains 100% immutable.

All human review artifacts will be generated in a separate directory:  
[`theo-data/datasets/theo_slm_v0_review/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_review/)

```text
theo-data/datasets/theo_slm_v0_review/
  ├── review-manifest.json    (Deterministic randomized review sequence & blind records)
  ├── review-records.json     (Dual-reviewer evaluations & written justifications)
  ├── adjudication.json       (Logged disagreement resolutions & final statuses)
  └── review-summary.json     (Inter-rater agreement, Cohen's Kappa, & acceptance rates)
```

---

## 10. Reviewer Calibration Batch Protocol (15 Records)

Before reviewing the full 264 candidate pool:

1. **Sample Calibration Batch:** Select a 15-record calibration batch containing a deliberate mix of:
   - Genuine semantic interpretations (`b/002` type)
   - Percept echoes (`a/001` type)
   - Paraphrases (`NEG-02`)
   - Stored-belief echoes (`c/001`)
   - Rule echoes (`NEG-04`)
   - Unsupported speculation (`NEG-06`)
   - Epistemically premature candidates (`NEG-14`)
   - Irrelevant facts (`NEG-13`)
   - Grounding failures (`NEG-10`)
2. **Calibrate Reviewers:** Both reviewers independently evaluate the 15 calibration records.
3. **Measure Initial Agreement:** Compute Cohen's Kappa ($\kappa$).
4. **Ambiguity Gate:** If inter-rater agreement on calibration is $\kappa < 0.75$, **STOP** and refine rubric wording before proceeding to full dataset review.

---

## 11. Critical Protection Against 6A.2 False Positives

Reviewers must be trained on the 4 canonical Phase 6A.2 probe results:

| Case ID | Candidate Description | Expected Classification | Rationale |
|---|---|---|---|
| **`a/001`** | Verbatim percept restatement | **`HARD_NEGATIVE`** | Restatement / repeat; zero semantic novelty |
| **`b/002`** | Common power-related cause explanation | **`GOLD_POSITIVE`** | Canonical abductive interpretation |
| **`c/001`** | Echo of stored background belief | **`HARD_NEGATIVE`** | Stored belief match; non-novel |
| **`c/002`** | Conjunction of two existing beliefs | **`HARD_NEGATIVE`** | Simple conjunction; derivable |

---

## 12. Inter-Rater Agreement & Acceptance Metrics

Upon completion of human review, the system will compute:

1. **Cohen's Kappa ($\kappa$):** Inter-rater reliability statistic.
2. **Gold Positive Acceptance Rate:** Percentage of candidate positives confirmed as `GOLD_POSITIVE`.
3. **Gold Abstain Acceptance Rate:** Percentage of candidate abstentions confirmed as `GOLD_ABSTAIN`.
4. **Rejection & Revision Rates:** Percentages of candidates assigned to `REJECT` or `NEEDS_REVISION`.
5. **Disagreement Rate:** Percentage of records requiring lead adjudication.

---

## 13. Execution Order & Current Stop State

```text
[Step 1] Write human-review specification.             --> COMPLETE
[Step 2] Validate specification/schema.                --> COMPLETE (review_validator.py PASSED)
[Step 3] STOP for human approval.                     --> CURRENT STOP POINT
[Step 4] Create randomized blind review manifest.      --> Pending approval
[Step 5] Run reviewer calibration batch (15 records).   --> Pending approval
[Step 6] STOP if calibration exposes ambiguity.        --> Pending calibration
[Step 7] Conduct independent full review.              --> Pending approval
[Step 8] Run adjudication on disagreements.           --> Pending approval
[Step 9] Produce human-review report.                  --> Pending approval
[Step 10] STOP at the Phase 6B.4 gate.                 --> Pending review completion
```

---

## Governance Confirmation

- **ADR-0028 & Provider Contracts:** Preserved (100% untouched).
- **Frozen Benchmark & Semantic Probe:** Untouched (0 leakage).
- **Model Selection & Fine-Tuning:** **STOPPED.** No model has been selected, downloaded, or trained.
- **Human Review Execution Status:** **STOPPED FOR HUMAN APPROVAL.** Candidate dataset revision `ds-v0.2-repaired` remains 100% immutable.
