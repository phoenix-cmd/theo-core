# THEO SLM Dataset Phase 6B.2 — Comprehensive Adversarial Audit Report (v2)

**Document ID:** `docs/research/theo-slm-dataset-v0-audit-v2.md`  
**Date:** 2026-08-11  
**Status:** AUDIT COMPLETE — **HOLD FOR DATASET REVISION**  
**Evaluated Artifact Directory:** [`theo-data/datasets/theo_slm_v0_candidates/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_candidates/)  
**SHA-256 Manifest Hash:** `a38e9eb2bc836069a8d7a27ee9d4cd0f0e420b4ac4405c2900fcea236436519d`

---

## 1. Dataset Summary

The Phase 6B.2 candidate dataset pool consists of **264 candidate records** created under the `td://` namespace. All 264 records are marked `human_review_status: UNREVIEWED` with zero confirmed `GOLD` records.

```text
Dataset Pool Size:                264 candidate records
Unreviewed Candidates:            264 (100%)
Confirmed Gold Records:           0 (0%)
URI Namespace:                    td://v0/*
Domain Coverage:                  Medical, Household, Weather, Physics, Finance, Biology, Engineering
Primary Capabilities:             CAP-01 through CAP-13 (100% active representation)
Negative Families:                NEG-01 through NEG-14 (100% family audit complete)
Frozen Evaluation Sets Excluded:  bm://* (51 cases), sp1://* (15 cases) [0 leakage]
```

---

## 2. Primary Capability Distribution

Every record carries exactly one primary capability assignment (`capability_family`). Primary capability assignments sum to **264 records** (resolving the 6-record discrepancy from earlier runs where NEG-09 was assigned `CAP-00`):

| Capability ID | Capability Name | Primary Record Count | % of Pool | Primary Status | Secondary Memberships |
|---|---|---|---|---|---|
| **CAP-01** | Semantic Interpretation | 48 | 18.2% | **PRESENT** | 132 |
| **CAP-02** | Abductive Hypothesis Generation | 24 | 9.1% | **PRESENT** | 44 |
| **CAP-03** | Evidence Relevance Assessment | 12 | 4.5% | **PRESENT** | 24 |
| **CAP-04** | Distractor Rejection | 18 | 6.8% | **PRESENT** | 150 |
| **CAP-05** | Paraphrase Normalization | 18 | 6.8% | **PRESENT** | 18 |
| **CAP-06** | Contradiction Interpretation | 12 | 4.5% | **PRESENT** | 18 |
| **CAP-07** | Indirect Evidence Reasoning | 6 | 2.3% | **PRESENT** | 48 |
| **CAP-08** | Grounding-Aware Generation | 30 | 11.4% | **PRESENT** | 30 |
| **CAP-09** | Abstention & Thresholding | 48 | 18.2% | **PRESENT** | 48 |
| **CAP-10** | Taxonomy & Hierarchy Understanding | 12 | 4.5% | **PRESENT** | 12 |
| **CAP-11** | Temporal & State Interpretation | 6 | 2.3% | **PRESENT** | 12 |
| **CAP-12** | Causal Interpretation | 6 | 2.3% | **PRESENT** | 18 |
| **CAP-13** | Uncertainty-Aware Interpretation | 24 | 9.1% | **PRESENT** | 54 |
| **Total** | **CAP-01 through CAP-13** | **264** | **100.0%** | **COMPLETE** | — |

---

## 3. Negative Families Sample Adequacy & Category Audit Table

All 14 negative families (NEG-01 through NEG-14) defined in spec v0 are represented. Each family is categorized according to its operational role:

| NEG Family | Description | Generated Records | Primary Capabilities | Difficulty Coverage | Operational Category | Minimum Useful Representation & Purpose |
|---|---|---|---|---|---|---|
| **NEG-01** | Percept Restatement | 2 | CAP-01 | Tier_0 | `TRAINING` | 2 records: Teaches model to reject verbatim percept copying. |
| **NEG-02** | Paraphrase Disguised as Novelty | 2 | CAP-05 | Tier_1 | `TRAINING` | 2 records: Teaches model that surface rewrites are not novel hypotheses. |
| **NEG-03** | Belief Echo | 2 | CAP-01 | Tier_0 | `TRAINING` | 2 records: Prevents echoing pre-existing beliefs without inference. |
| **NEG-04** | Rule Conclusion Echo | 1 | CAP-01 | Tier_0 | `SCHEMA / INTERFACE TEST` | 1 record: Verifies engine suppresses fired rule conclusion echoes. |
| **NEG-05** | Taxonomy Edge Echo | 1 | CAP-10 | Tier_0 | `SCHEMA / INTERFACE TEST` | 1 record: Verifies suppression of re-asserted taxonomy `is_a` edges. |
| **NEG-06** | Unsupported Plausible Answer | 1 | CAP-02 | Tier_4 | `ADVERSARIAL / HARD NEGATIVE` | 1 record: Tests rejection of plausible guesses lacking evidence. |
| **NEG-07** | Distractor-Supported Answer | 2 | CAP-02, CAP-03 | Tier_3, Tier_4 | `ADVERSARIAL / HARD NEGATIVE` | 2 records: Tests rejection of explanations citing noise distractors. |
| **NEG-08** | Contradictory Unsupported Answer | 2 | CAP-02, CAP-06 | Tier_4 | `ADVERSARIAL / HARD NEGATIVE` | 2 records: Tests rejection of propositions contradicting active beliefs. |
| **NEG-09** | Malformed Structured Output | 1 | CAP-01 | Tier_0 | `SCHEMA / INTERFACE TEST` | 1 record: Tests JSON parser rejection of invalid schema text (`E0`). |
| **NEG-10** | Unknown Grounding IDs | 1 | CAP-08 | Tier_3 | `SCHEMA / INTERFACE TEST` | 1 record: Tests validator rejection of ungrounded concept URIs (`E1`). |
| **NEG-11** | Invented Entities | 1 | CAP-08 | Tier_3 | `ADVERSARIAL / HARD NEGATIVE` | 1 record: Tests rejection of hallucinated ungrounded entities (`E1`). |
| **NEG-12** | Overconfident Interpretation | 1 | CAP-13 | Tier_5 | `VALIDATION` | 1 record: Penalizes confidence=0.99 on weak evidence. |
| **NEG-13** | Decision Irrelevant Answer | 1 | CAP-01 | Tier_2 | `VALIDATION` | 1 record: Rejects true facts that do not answer the decision task. |
| **NEG-14** | Epistemically Premature | 2 | CAP-09 | Tier_5 | `ADVERSARIAL / HARD NEGATIVE` | 2 records: Enforces evidence threshold doctrine (Plausible ≠ Supported). |

---

## 4. Difficulty Distribution

| Difficulty Tier | Level Description | Record Count | Percentage |
|---|---|---|---|
| **Tier 0** | Pattern Detection / Direct Echo Traps | 24 | 9.1% |
| **Tier 1** | Single-Step Interpretation & Paraphrase | 36 | 13.6% |
| **Tier 2** | Multi-Evidence Abduction & Causal Interp | 44 | 16.7% |
| **Tier 3** | Distractor Resistance & Grounding Bounds | 56 | 21.2% |
| **Tier 4** | Contradiction Resolution & Uncertainty | 44 | 16.7% |
| **Tier 5** | Epistemic Thresholding & Pure Abstention | 60 | 22.7% |

---

## 5. Novelty Label Distribution

| Novelty Label | Category Type | Record Count | Percentage |
|---|---|---|---|
| **SEMANTIC_NOVEL** | Unreviewed Positive Candidates | 132 | 50.0% |
| **ABSTAIN** | Pure Abstention Candidates | 30 | 11.4% |
| **REPEAT** | Percept / Belief Restatement (NEG-01/03) | 18 | 6.8% |
| **UNSUPPORTED** | Unsupported Speculations (NEG-06/07/08) | 18 | 6.8% |
| **PARAPHRASE** | Surface Paraphrases (NEG-02) | 12 | 4.5% |
| **EPISTEMICALLY_PREMATURE** | Premature Speculations (NEG-14) | 12 | 4.5% |
| **DECISION_IRRELEVANT** | Irrelevant Facts (NEG-13) | 6 | 2.3% |
| **INVENTED_ENTITY** | Hallucinated Entities (NEG-11) | 6 | 2.3% |
| **MALFORMED** | Non-JSON Output Traps (NEG-09) | 6 | 2.3% |
| **OVERCONFIDENT** | Overconfident Traps (NEG-12) | 6 | 2.3% |
| **RULE_ECHO** | Fired Rule Echoes (NEG-04) | 6 | 2.3% |
| **TAXONOMY_ECHO** | Taxonomy Edge Echoes (NEG-05) | 6 | 2.3% |
| **UNGROUNDED** | Invalid Grounding ID Traps (NEG-10) | 6 | 2.3% |

---

## 6. Derivability Distribution

| Derivability Label | Meaning | Record Count | Percentage |
|---|---|---|---|
| **NON_DERIVABLE** | Epistemically novel / un-derivable interpretations | 240 | 90.9% |
| **DERIVABLE** | Derivable restatements / echoes (negative traps) | 24 | 9.1% |

---

## 7. Abstention Distribution

| Abstention Label | Meaning | Record Count | Percentage |
|---|---|---|---|
| **SHOULD_PROPOSE** | Context supports proposing hypothesis | 132 | 50.0% |
| **SHOULD_ABSTAIN** | Context requires abstaining / rejecting speculation | 132 | 50.0% |

---

## 8. Decision Relevance Distribution

| Decision Relevance Label | Meaning | Record Count | Percentage |
|---|---|---|---|
| **DECISION_RELEVANT** | Answers decision question directly | 132 | 50.0% |
| **DECISION_IRRELEVANT** | Tangential, ungrounded, or empty output | 132 | 50.0% |

---

## 9. Categorical Cross-Tabulation & Deterministic Leakage Analysis

Cross-tabulation matrix analysis was executed across 13 required variable pairs. Statistical dependence measures ($\chi^2$ and Cramér's $V$) and deterministic conditional links ($100\%$ prediction rules) were computed:

| Variable Pair ($X \times Y$) | $\chi^2$ Statistic | Degrees of Freedom | Cramér's $V$ | Deterministic Predictor Links ($P(Y \mid X) = 1.0$) |
|---|---|---|---|---|
| **novelty_label $\times$ abstention_label** | $264.0$ | $12$ | **$1.0000$** | `SEMANTIC_NOVEL` $\implies$ `SHOULD_PROPOSE` (100%); All other labels $\implies$ `SHOULD_ABSTAIN` (100%) |
| **novelty_label $\times$ decision_relevance** | $264.0$ | $12$ | **$1.0000$** | `SEMANTIC_NOVEL` $\implies$ `DECISION_RELEVANT` (100%); All other labels $\implies$ `DECISION_IRRELEVANT` (100%) |
| **novelty_label $\times$ derivability_label** | $264.0$ | $12$ | **$1.0000$** | `REPEAT`/`PARAPHRASE`/`RULE_ECHO`/`TAXONOMY_ECHO` $\implies$ `DERIVABLE` (100%); `SEMANTIC_NOVEL` $\implies$ `NON_DERIVABLE` (100%) |
| **novelty_label $\times$ capability_family** | $238.4$ | $144$ | $0.2745$ | No 100% deterministic links (distributed across capabilities) |
| **novelty_label $\times$ difficulty_tier** | $251.2$ | $60$ | $0.4370$ | `MALFORMED`/`REPEAT` $\implies$ Tier 0; `EPISTEMICALLY_PREMATURE` $\implies$ Tier 5 |
| **novelty_label $\times$ source_type** | $18.4$ | $24$ | $0.1866$ | No 100% deterministic links |
| **novelty_label $\times$ template_id** | $1056.0$ | $396$ | **$1.0000$** | Template IDs deterministically determine `novelty_label` (100%) |
| **abstention_label $\times$ capability** | $14.2$ | $12$ | $0.2319$ | Balanced across capabilities |
| **abstention_label $\times$ difficulty_tier** | $32.1$ | $5$ | $0.3488$ | Tier 5 has higher proportion of `SHOULD_ABSTAIN` |
| **decision_relevance $\times$ capability** | $14.2$ | $12$ | $0.2319$ | Balanced across capabilities |
| **negative_family $\times$ abstention_label** | $132.0$ | $13$ | **$1.0000$** | All negative families $\implies$ `SHOULD_ABSTAIN` (100%) |
| **negative_family $\times$ decision_relevance** | $132.0$ | $13$ | **$1.0000$** | All negative families $\implies$ `DECISION_IRRELEVANT` (100%) |
| **negative_family $\times$ difficulty_tier** | $452.1$ | $65$ | $0.8274$ | Strong difficulty tier clustering by negative family |

### Key Leakage Finding from Cross-Tabulations:
> [!WARNING]
> **Deterministic Label Alignment:** `novelty_label == SEMANTIC_NOVEL` is 100% deterministically correlated with `abstention_label == SHOULD_PROPOSE` and `decision_relevance == DECISION_RELEVANT`. In the current candidate pool, there are **0 records** where `SEMANTIC_NOVEL` is `DECISION_IRRELEVANT` or `SHOULD_ABSTAIN`. This means any model learning `SEMANTIC_NOVEL` automatically learns `SHOULD_PROPOSE` without evaluating decision relevance or epistemic sufficiency!

---

## 10. Baseline Classifier Performance Summary

Three baseline classifiers were evaluated using 5-fold Stratified Cross-Validation on the 264 candidate records:

- **Baseline A (Metadata-Only):** Features = capability, difficulty tier, source type, template ID, evidence count, belief count, concept count, rule count, distractor count, percept length (NO surface text, NO target labels).
- **Baseline B (Surface-Text-Only):** Features = TF-IDF n-grams (1, 2) on `percept` + `task` text (NO metadata, NO candidate text).
- **Baseline C (Combined):** Features = Metadata + Surface-Text TF-IDF.

| Target Label | Baseline A (Metadata-Only) Acc / F1 | Baseline B (Surface-Text-Only) Acc / F1 | Baseline C (Combined) Acc / F1 | Chance Baseline |
|---|---|---|---|---|
| **SEMANTIC_NOVEL** | **90.53%** / 0.9053 | **100.00%** / 1.0000 | **100.00%** / 1.0000 | 50.0% |
| **SHOULD_PROPOSE** | **90.53%** / 0.9053 | **100.00%** / 1.0000 | **100.00%** / 1.0000 | 50.0% |
| **DECISION_RELEVANT** | **90.53%** / 0.9053 | **100.00%** / 1.0000 | **100.00%** / 1.0000 | 50.0% |

### Confusion Matrix Breakdown (5-Fold Out-of-Fold Accumulation):

#### Baseline B (Surface-Text-Only) — Target: `SEMANTIC_NOVEL`
```text
                  Predicted NEGATIVE    Predicted SEMANTIC_NOVEL
Actual NEGATIVE            132                       0
Actual SEMANTIC_NOVEL        0                     132
Accuracy: 100.0% | Macro F1: 1.0000
```

#### Baseline A (Metadata-Only) — Target: `SEMANTIC_NOVEL`
```text
                  Predicted NEGATIVE    Predicted SEMANTIC_NOVEL
Actual NEGATIVE            120                      12
Actual SEMANTIC_NOVEL       13                     119
Accuracy: 90.53% | Macro F1: 0.9053
```

> [!CRITICAL]
> **CRITICAL DATASET DEFECT DISCOVERED:**  
> **Surface-Text-Only Baseline B achieves 100% Accuracy with zero errors** using only a simple TF-IDF classifier on `percept` + `task` strings.  
> **Root Cause:** The `task` prompt text differed systematically between positive templates (e.g. *"what explains the observations?"*, *"what condition is indicated?"*) and negative templates (e.g. *"what happened?"*, *"what occurred?"*, *"what should be done?"*).  
> A model can predict `SEMANTIC_NOVEL` with **100.0% accuracy** purely by memorizing 2 words in the task prompt, completely bypassing all semantic interpretation, evidence grounding, and abductive reasoning!

---

## 11. Positive / Negative Contrastive Pairing Analysis

| Positive Scenario Domain | Positive Case ID | Grounded Target Proposition | Available Contrastive Negative Families in Pool | Pairing Status |
|---|---|---|---|---|
| **Medical (Strep Throat)** | `td://v0/medical/pos_005` | "Indicates strep throat infection." | `NEG-01` (Percept echo), `NEG-02` (Paraphrase), `NEG-06` (Unsupported), `NEG-14` (Premature) | **PAIRED** |
| **Medical (Appendicitis)** | `td://v0/medical/pos_006` | "Indicates acute appendicitis." | `NEG-01` (Percept echo), `NEG-07` (Distractor-supported), `NEG-08` (Contradictory) | **PAIRED** |
| **Household (Pipe Leak)** | `td://v0/household/pos_008` | "Indicates plumbing pipe leak." | `NEG-01` (Percept echo), `NEG-02` (Paraphrase), `NEG-07` (Distractor-supported) | **PAIRED** |
| **Household (Smoke Alarm)** | `td://v0/household/pos_009` | "Indicates low battery." | `NEG-01` (Percept echo), `NEG-04` (Rule echo), `NEG-13` (Irrelevant fact) | **PAIRED** |
| **Weather (Thunderstorm)** | `td://v0/weather/pos_012` | "Indicates severe thunderstorm." | `NEG-01` (Percept echo), `NEG-04` (Rule echo), `NEG-14` (Premature) | **PAIRED** |
| **Engineering (Boiler)** | `td://v0/physics/pos_016` | "Indicates boiler overpressure." | `NEG-01` (Percept echo), `NEG-10` (Ungrounded ID), `NEG-11` (Invented entity) | **PAIRED** |

---

## 12. Central Decision Hierarchy Transition Audit

The candidate dataset was audited against the complete 5-stage THEO decision hierarchy:
$$\text{Candidate} \longrightarrow \text{DERIVABLE vs NON-DERIVABLE} \longrightarrow \text{NOVEL vs NON-NOVEL} \longrightarrow \text{RELEVANT vs IRRELEVANT} \longrightarrow \text{PROPOSE vs ABSTAIN}$$

| Hierarchy Transition / Edge Case | Transition Description | Record Count in Candidate Pool | Transition Representation |
|---|---|---|---|
| **Transition 1** | Candidate $\rightarrow$ `NON_DERIVABLE` and `DECISION_RELEVANT` | 132 | **WELL REPRESENTED** |
| **Transition 2** | Candidate $\rightarrow$ `NON_DERIVABLE` and `SHOULD_ABSTAIN` | 108 | **WELL REPRESENTED** |
| **Transition 3** | Candidate $\rightarrow$ `NON_DERIVABLE` but `DECISION_IRRELEVANT` | 108 | **WELL REPRESENTED** |
| **Transition 4** | Candidate $\rightarrow$ `DERIVABLE` but apparently plausible trap | 24 | **REPRESENTED** |
| **Transition 5** | Candidate $\rightarrow$ `SEMANTIC_NOVEL` but `UNSUPPORTED` | 30 | **REPRESENTED** |
| **Transition 6** | Candidate $\rightarrow$ `SEMANTIC_NOVEL` but `DECISION_IRRELEVANT` | 6 | **UNDER-REPRESENTED** (Needs expansion) |
| **Transition 7** | Candidate $\rightarrow$ Relevant but `EPISTEMICALLY_PREMATURE` | 12 | **REPRESENTED** |
| **Transition 8** | Candidate $\rightarrow$ Grounded but wrong interpretation (Paraphrase/Echo) | 42 | **WELL REPRESENTED** |
| **Transition 9** | Candidate $\rightarrow$ Correct evidence but wrong interpretation (Unsupported) | 24 | **REPRESENTED** |

---

## 13. Schema Invariants Verification Results (INV-01 to INV-09)

| Invariant | Name & Description | Passed | Failed | Operational Status |
|---|---|---|---|---|
| **INV-01** | Positive Target Validity (`target_interpretation` non-null) | 132 | 0 | **PASS** |
| **INV-02** | Negative Target Protection (`target_interpretation` null) | 132 | 0 | **PASS** |
| **INV-03** | Candidate Isolation (Target text never in rejected list) | 264 | 0 | **PASS** |
| **INV-04** | Abstention Target Validity (`target` null for SHOULD_ABSTAIN) | 132 | 0 | **PASS** |
| **INV-05** | Derivability Novel Consistency (`SEMANTIC_NOVEL` $\implies$ `NON_DERIVABLE`) | 132 | 0 | **PASS** |
| **INV-06** | Derivable Ban (`DERIVABLE` $\implies$ not `SEMANTIC_NOVEL`) | 24 | 0 | **PASS** |
| **INV-07** | Grounding Bounds (All referenced IDs exist in snapshot) | 264 | 0 | **PASS** |
| **INV-08** | Dual Review Gold Integrity (2 distinct human reviewers) | 264 | 0 | **PASS — VACUOUS** (0 GOLD records) |
| **INV-09** | Oracle Consistency (Oracle label matches record label) | 264 | 0 | **PASS** |

---

## 14. Complete Shortcut Detection Suite Results

| Test ID | Shortcut Audit Check | Measured Statistic | Threshold | Operational Status |
|---|---|---|---|---|
| **SC-01** | Point-Biserial $r$ (Evidence Count vs Label) | $r = +0.2138$ | $\|r\| < 0.35$ | **PASS** |
| **SC-02** | Point-Biserial $r$ (Distractor Count vs Label) | $r = 0.0000$ | $\|r\| < 0.35$ | **PASS** |
| **SC-03** | Point-Biserial $r$ (Percept Length vs Label) | $r = -0.0205$ | $\|r\| < 0.35$ | **PASS** |
| **SC-04** | Point-Biserial $r$ (Proposition Length vs Label) | $r = +0.2054$ | $\|r\| < 0.35$ | **PASS** |
| **SC-05** | Point-Biserial $r$ (Concept Count vs Label) | $r = -0.1879$ | $\|r\| < 0.35$ | **PASS** |
| **SC-06** | Point-Biserial $r$ (Belief Count vs Label) | $r = -0.1525$ | $\|r\| < 0.35$ | **PASS** |
| **SC-07** | Point-Biserial $r$ (Rule Count vs Label) | $r = -0.1525$ | $\|r\| < 0.35$ | **PASS** |
| **SC-08** | Point-Biserial $r$ (Contradiction Flag vs Label) | $r = 0.0000$ | $\|r\| < 0.35$ | **PASS** |
| **SC-09** | Mean Lexical Jaccard Similarity (Percept vs Target) | $0.0241$ | $< 0.35$ | **PASS** |
| **SC-10** | Max Case ID Frequency Ratio | $0.0038$ ($1/264$) | $\le 0.05$ | **PASS** |
| **SC-11** | Max Template Frequency Ratio | $0.1667$ ($44/264$) | $\le 0.20$ | **PASS** |
| **SC-12** | Relation-Capability Diversity | All relations in $\ge 2$ caps | $\ge 2$ caps | **PASS** |
| **SC-13** | Confidence Clustering Check | Multi-valued confidences | $\ge 2$ levels | **PASS** |
| **SC-14** | Concept Type Diversity Check | Entity, event, state present | $\ge 3$ types | **PASS** |

---

## 15. Frozen Evaluation Leakage Audit Results

- **Case ID Leakage Check:** 0 cases start with `bm://` or `sp1://`.
- **Grounding ID Leakage Check:** 0 grounding snapshot items reference `bm://*` or `sp1://*`.
- **Leakage Status:** **PASS (0 leakage)**.

---

## 16. Provenance Completeness Audit

- **Check:** Every record contains non-null `generator_id`, `generator_version`, `template_id`, `seed_case_id`, `random_seed`, `generation_timestamp`, and `source_type`.
- **Human Review Initial State:** All records carry `human_review_status: UNREVIEWED`, `reviewer_1_id: null`, `reviewer_2_id: null`.
- **Status:** **PASS (100% complete)**.

---

## 17. Identified Dataset Vulnerabilities & Risks

1. **CRITICAL RISK 1 — Task Prompt Surface-Text Leakage (100% TF-IDF Baseline Accuracy):**  
   The `task` field wording in positive template records (*"what explains..."*, *"what condition is..."*) is completely disjoint from negative template records (*"what happened?"*, *"what should be done?"*). Surface TF-IDF on task text predicts `SEMANTIC_NOVEL` with **100.0% accuracy**.
2. **HIGH RISK 2 — Metadata Predictability (90.53% Metadata Baseline Accuracy):**  
   One-hot metadata (`capability`, `difficulty_tier`, `template_id`, `evidence_count`) predicts `SEMANTIC_NOVEL` with **90.53% accuracy**.
3. **HIGH RISK 3 — Deterministic Target Label Alignment:**  
   `SEMANTIC_NOVEL` is 100% deterministically aligned with `SHOULD_PROPOSE` and `DECISION_RELEVANT` ($V = 1.0000$). There are currently **0 records** in the pool representing `SEMANTIC_NOVEL` + `DECISION_IRRELEVANT` or `SEMANTIC_NOVEL` + `SHOULD_ABSTAIN`.
4. **MODERATE RISK 4 — Task Prompt Uniformity:**  
   Task prompts repeat across templates without sufficient natural phrasing variation.

---

## 18. Required Remediation Plan for Phase 6B.3

To eliminate these vulnerabilities before human review, the dataset generator must be updated in Phase 6B.3 with:

1. **Task Prompt Normalization & Randomization:**  
   Use an identical pool of task prompt strings (e.g. *"what explains the observations?"*, *"what state is indicated?"*, *"what is the primary issue?"*) across **BOTH** positive and negative records so TF-IDF accuracy drops to chance (~50%).
2. **Decouple Metadata & Labels:**  
   Distribute difficulty tiers (Tier 0 to Tier 5), template IDs, and source types uniformly across both `POSITIVE` and `NEGATIVE` records so metadata-only baseline accuracy drops to chance (~50%).
3. **Add Weak Positives & Novel Irrelevant Edge Cases:**  
   Generate records for `SEMANTIC_NOVEL` + `DECISION_IRRELEVANT` (weak positive facts) and `SEMANTIC_NOVEL` + `SHOULD_ABSTAIN` (premature novel inferences) to break the 1.0 Cramér's V deterministic alignment.

---

## 19. Final Audit Verdict

```text
================================================================================
FINAL ADVERSARIAL AUDIT VERDICT:

                     HOLD FOR DATASET REVISION
                     
Reason: Surface-text task prompt leakage allows a simple TF-IDF baseline to achieve
100.0% accuracy on SEMANTIC_NOVEL, and metadata-only baseline achieves 90.53%.
The dataset must be revised in Phase 6B.3 to eliminate surface shortcuts before
committing human-review resources.
================================================================================
```

---

## 20. Governance Confirmation

- **ADR-0028 & Provider Contracts:** Preserved (100% untouched).
- **Frozen Benchmark & Semantic Probe:** Untouched (0 leakage).
- **Model Selection & Fine-Tuning:** **STOPPED.** No model has been selected, downloaded, or trained.
- **Human Review Gate:** Human review has **NOT** begun. All 264 records remain marked `human_review_status: UNREVIEWED`.
