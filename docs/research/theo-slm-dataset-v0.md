# THEO SLM Dataset Specification v0

**Status:** REVISED DRAFT FOR REVIEW — negative training semantics & oracle invariants locked; awaiting final human approval before Phase 6B.2.
**Date:** 2026-08-11
**Phase:** 6B.1 — Dataset Specification (Revised & Comprehensive)
**Origin:** Phase 6A.2 NEGATIVE/DIAGNOSTIC result
  (`docs/research/reference-slm/semantic-probe-v1/phase-6a2-report.md`);
  Phase 6B design (`docs/research/theo-slm-design-v0.md`, §D–E).
**Scope:** Define the training dataset *before* any dataset is generated, any
  model is selected, or any training begins.
**Governance:**
  - ADR-0028 is frozen. This document does not alter it.
  - The frozen 51-case benchmark and 15-case semantic probe are evaluation-only.
  - Qwen outputs are negative/diagnostic evidence (a failure oracle), never positive labels.
  - No model-specific assumptions (tokenizer, framework, architecture).

---

## 1. Dataset Objective & Decision Hierarchy

### 1.1 What THEO SLM is learning

THEO SLM learns to perform **semantic interpretation**: producing propositions
that go beyond what the symbolic runtime can derive from its current possessions
(percepts, beliefs, concepts, rules, taxonomy edges).

The central question the dataset teaches the model to answer is:

> **"Is this something THEO already knows/derives, or is this a genuinely useful interpretation that THEO cannot currently derive?"**

### 1.2 Mechanically Explicit Decision Hierarchy

The evaluation and annotation pipeline enforces a strict 4-stage decision hierarchy:

```text
                      Candidate Proposition P
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 1. DERIVABILITY CHECK │
                     └───────────┬───────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
            [DERIVABLE]                  [NON_DERIVABLE]
                 │                               │
                 ▼                               ▼
       REJECT as contribution       ┌──────────────────────────┐
      (REPEAT / PARAPHRASE /        │ 2. SEMANTIC NOVELTY CHECK│
       RULE_ECHO / TAXONOMY_ECHO)   │    (6 SN Criteria §1.3)  │
                                    └────────────┬─────────────┘
                                                 │
                                 ┌───────────────┴───────────────┐
                                 │                               │
                             [FAILS]                         [PASSES]
                                 │                               │
                                 ▼                               ▼
                       REJECT as contribution             SEMANTIC_NOVEL
                      (UNSUPPORTED / PREMATURE)                  │
                                                                 ▼
                                                    ┌──────────────────────────┐
                                                    │ 3. DECISION RELEVANCE    │
                                                    └────────────┬─────────────┘
                                                                 │
                                                 ┌───────────────┴───────────────┐
                                                 │                               │
                                            [IRRELEVANT]                     [RELEVANT]
                                                 │                               │
                                                 ▼                               ▼
                                            Weak Positive                  Gold Candidate
                                           (Tangential)                          │
                                                                                 ▼
                                                                    ┌──────────────────────────┐
                                                                    │ 4. DUAL HUMAN REVIEW     │
                                                                    └────────────┬─────────────┘
                                                                                 │
                                                                 ┌───────────────┴───────────────┐
                                                                 │                               │
                                                            [DISAGREE]                       [AGREE]
                                                                 │                               │
                                                                 ▼                               ▼
                                                             REJECTED                         GOLD POSITIVE
```

#### Hierarchy Validation Rules:
1. **DERIVABLE:** The Symbolic Derivability Oracle can produce the proposition via verbatim percept match, belief restatement/paraphrase, taxonomy edge echo (`is_a`, `related_to`), or rule-conclusion chain. Automatically rejected as a positive contribution.
2. **NON_DERIVABLE:** The symbolic engine cannot derive the proposition — a semantic gap exists.
3. **SEMANTIC_NOVEL:** Satisfies all six Semantic Novelty (SN) criteria from the frozen probe spec (semantic-probe-v1-spec.md §5).
4. **DECISION_RELEVANT:** Semantically novel, grounded in evidence, and directly answers or advances the decision task.
5. **GOLD POSITIVE:** `SEMANTIC_NOVEL` + `DECISION_RELEVANT` + 100% agreement across 2 human reviewers.

---

### 1.3 The 6 Semantic Novelty (SN) Invariants

A candidate proposition has **Semantic Novelty** if and only if **all** of the following hold:
1. **Not a textual duplicate:** normalized text is not equal to any stored percept sentence, belief, concept label, or rule conclusion.
2. **Not a paraphrase:** semantic content is not already present under a different surface form.
3. **Not a taxonomy restatement:** does not re-assert an existing `is_a`/`related_to` edge or concept definition.
4. **Not a rule echo:** is not the conclusion of a fired or derivable rule chain.
5. **Requires combining/interpreting evidence:** integrates at least two distinct grounding items (or maps surface text to a non-surface concept) to reach content no single item contains.
6. **All supporting claims are grounded:** every referenced ID resolves to an ID in the `GroundingSnapshot`.

---

### 1.4 What the model must NOT learn

The model must never learn behaviors that the symbolic runtime owns (from `theo-slm-design-v0.md` §B):
- final decision making;
- belief commitment;
- direct Thought/DecisionRecord/Rule creation;
- modification of symbolic state;
- bypassing grounding;
- replacing symbolic inference;
- replacing deterministic verification.

The model's output boundary is `SemanticInterpretation` (design doc §C); conversion to `HypothesisProposal` is performed deterministically by the provider adapter in `theo-providers`.

---

### 1.5 Abstention as a First-Class Training Behavior

Abstention is a **first-class core cognitive behavior**, not a label fallback.
The SLM explicitly learns the following decision logic:

```text
Cognitive Situation                                          Target Model Behavior
───────────────────────────────────────────────────────────  ────────────────────────────────────────────────────
Proposition is derivable from symbolic knowledge             → ABSTAIN / Do NOT propose
Proposition is semantically novel AND evidence supports it   → Emits SemanticInterpretation JSON
Proposition is plausible BUT evidence is unsupported         → ABSTAIN (produce null / empty output)
Available evidence is insufficient / ambiguous               → ABSTAIN (produce null / empty output)
```

A model that abstains when evidence is insufficient is far more valuable than a model that generates plausible hallucinations.

---

## 2. Learning Capabilities & Three-Tier Hierarchy

Capabilities are organized into three explicit levels to balance training priorities and prevent the model from becoming an expert paraphraser while failing at abductive reasoning.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. CORE CAPABILITIES (Must Learn — Primary Training Objectives)            │
│    • Semantic Interpretation (CAP-01)   • Abduction (CAP-02)                │
│    • Evidence Relevance (CAP-03)        • Distractor Rejection (CAP-04)    │
│    • Contradiction Interp. (CAP-06)     • Indirect Evidence (CAP-07)       │
│    • Abstention (CAP-09)                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. SUPPORTING CAPABILITIES (Secondary Reasoning Enhancers)                 │
│    • Paraphrase Normalization (CAP-05)  • Grounding-Awareness (CAP-08)      │
│    • Taxonomy Understanding (CAP-10)    • Temporal/State Interp. (CAP-11)   │
│    • Causal Interpretation (CAP-12)     • Uncertainty Calibration (CAP-13) │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. INFRASTRUCTURE BEHAVIOR (Non-Negotiable System Constraint)               │
│    • Schema Compliance & Structured Output (CAP-00)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Tier 1: Core Capabilities (Must Learn)

#### CAP-01: Semantic Interpretation
| Aspect | Definition |
|---|---|
| **Positive behavior** | Produce a proposition that goes beyond literal percept text and is supported by supplied evidence. The conclusion is not present in symbolic knowledge. |
| **Negative behavior** | Restate the percept, paraphrase a belief, or echo a concept label. |
| **Success** | Proposal satisfies all 6 SN criteria (§1.3) and references at least 2 distinct grounding items. |
| **False positive** | A surface-novel sentence that restates content already derivable by the runtime (e.g., a/001 "The container was broken by the impact on the floor" — a paraphrase). |
| **Evidence citation** | Must cite ≥2 `supporting_evidence_ids` from the grounding snapshot that jointly support the interpretation. |

#### CAP-02: Abductive Hypothesis Generation
| Aspect | Definition |
|---|---|
| **Positive behavior** | Infer an unobserved cause or explanation that accounts for observed facts (b/002 pattern: symptoms → "power outage"). |
| **Negative behavior** | Produce a plausible-sounding conclusion with no cited evidence, or cite evidence that does not support the conclusion. |
| **Success** | The inferred entity/event is not present in percept, beliefs, or rules. All supporting evidence is cited. |
| **False positive** | An unsupported plausible answer (e.g., "There was an earthquake" for b/002 with no earthquake-related evidence). |
| **Evidence citation** | Must cite specific observations that jointly support the abductive conclusion. |

#### CAP-03: Evidence Relevance
| Aspect | Definition |
|---|---|
| **Positive behavior** | Cite evidence that actually supports the interpretation, and ignore distractors. Group E pattern: cite rain evidence (wet, umbrella), not sky/car/dog. |
| **Negative behavior** | Cite distractor IDs, cite all available IDs indiscriminately, or cite no evidence. |
| **Success** | All cited IDs are relevant (precision = 1.0); all relevant IDs are cited (recall ≥ threshold). No distractor IDs cited. |
| **False positive** | Correct interpretation but citing a distractor alongside relevant evidence. |
| **Evidence citation** | `supporting_evidence_ids` must contain exactly the relevant IDs and no distractor IDs. |

#### CAP-04: Distractor Rejection
| Aspect | Definition |
|---|---|
| **Positive behavior** | Correctly identify and exclude irrelevant concepts/beliefs/evidence from the interpretation, even when they appear prominently in the percept. |
| **Negative behavior** | Reference distractor IDs, or produce an interpretation driven by distractor content. |
| **Success** | Zero distractor IDs in `supporting_evidence_ids` or `referenced_concept_ids`. |
| **False positive** | Correct interpretation that happens to exclude distractors by accident (e.g., model only cites the first ID). |
| **Evidence citation** | Only relevant IDs; distractor IDs must be absent. |

#### CAP-06: Contradiction Interpretation
| Aspect | Definition |
|---|---|
| **Positive behavior** | Detect contradictory stored beliefs and produce an explanatory interpretation (c/002: light_on ∧ room_dark → "The light is broken"). |
| **Negative behavior** | Echo the contradiction ("The room is dark and the light is on"), cite the percept as cause, or ignore the contradiction entirely. |
| **Success** | Proposal resolves the contradiction into a new explanatory proposition not present in beliefs. Both contradicting beliefs are cited. |
| **False positive** | A conjunction of two beliefs presented as if it were an explanation. |
| **Evidence citation** | Must cite both contradicting belief IDs plus the concept that resolves them. |

#### CAP-07: Indirect Evidence Reasoning
| Aspect | Definition |
|---|---|
| **Positive behavior** | Bridge gaps where the conclusion is not stated, only hinted at (Group B/D patterns). Combine multiple observations into one unifying cause. |
| **Negative behavior** | Concatenate observations without new content, or produce a conclusion from a single observation. |
| **Success** | Proposal integrates ≥2 distinct evidence items into content not present in any single item. |
| **False positive** | Concatenation disguised as interpretation (e.g., "The umbrella was taken and clouds covered the sky"). |
| **Evidence citation** | Must cite ≥2 distinct grounding items from different knowledge sources. |

#### CAP-09: Abstention
| Aspect | Definition |
|---|---|
| **Positive behavior** | When evidence is insufficient to support any interpretation, produce no proposal (empty output) rather than an unsupported guess or plausible speculation. |
| **Negative behavior** | Produce a confident interpretation when evidence is absent, ambiguous, or epistemically premature. |
| **Success** | Empty output on cases designed to have insufficient evidence. Non-empty output on cases with sufficient evidence. |
| **False positive** | Abstaining on a case where evidence is sufficient (over-abstention). |
| **Evidence citation** | No citations required for abstention; the absence of output is the target. |

---

### Tier 2: Supporting Capabilities

#### CAP-05: Paraphrase Normalization
| Aspect | Definition |
|---|---|
| **Positive behavior** | Recognize that differently-worded propositions denote the same content and classify the paraphrase as REPEAT, not NOVEL. Map surface variation to stored concepts. |
| **Negative behavior** | Classify a paraphrase as novel (e.g., a/001 "broken by the impact on the floor" treated as novel). |
| **Success** | Paraphrases are labeled REPEAT/PARAPHRASE; genuine novel interpretations are labeled SEMANTIC_NOVEL. |
| **False positive** | A genuinely novel interpretation mis-classified as paraphrase due to surface word overlap. |
| **Evidence citation** | For paraphrase detection: cite the stored item that the proposal paraphrases. |

#### CAP-08: Grounding-Aware Output
| Aspect | Definition |
|---|---|
| **Positive behavior** | Reference only IDs present in the grounding snapshot. All `referenced_concept_ids` and `supporting_evidence_ids` resolve against the `GroundingSnapshot`. |
| **Negative behavior** | Reference unknown IDs, invent entity URIs, or use non-URI-form identifiers. |
| **Success** | 100% of referenced IDs are present in the grounding snapshot. |
| **False positive** | Model cites a valid ID that does not actually support the proposition (grounded but irrelevant). |
| **Evidence citation** | Every ID must resolve against the case's `GroundingSnapshot`. |

#### CAP-10: Taxonomy Understanding
| Aspect | Definition |
|---|---|
| **Positive behavior** | Use taxonomic knowledge (is_a edges) to classify observations (d/001: fur + nurses young → mammal) without echoing the taxonomy edges themselves. |
| **Negative behavior** | Echo a taxonomy edge as if it were novel ("A mammal is an animal" — E4), or classify without evidence. |
| **Success** | Correct classification using feature evidence, citing the classification concept. The taxonomy edge is used for reasoning, not reproduced as output. |
| **False positive** | Echoing the taxonomy edge verbatim. |
| **Evidence citation** | Must cite features in the percept and target classification concept. |

#### CAP-11: Temporal/State Interpretation
| Aspect | Definition |
|---|---|
| **Positive behavior** | Interpret temporal cues or state changes from observations (e.g., "The towel still feels damp after hanging all night" → ongoing moisture retention). |
| **Negative behavior** | Ignore temporal markers or produce time-independent interpretations when temporal reasoning is required. |
| **Success** | Proposal incorporates temporal/state information from the percept into the interpretation. |
| **False positive** | Restating temporal information without interpretation. |
| **Evidence citation** | Must cite evidence containing temporal/state markers. |

#### CAP-12: Causal Interpretation
| Aspect | Definition |
|---|---|
| **Positive behavior** | Identify causal relationships between observations and produce causal interpretations (not just correlational). |
| **Negative behavior** | State a correlation as if it were causal, or assert causation with no supporting evidence. |
| **Success** | `semantic_relation` is "causal" or "explanation" and the causal chain is supported by cited evidence. |
| **False positive** | A post-hoc-ergo-propter-hoc interpretation where temporal sequence is mistaken for causation. |
| **Evidence citation** | Must cite evidence for both cause and effect. |

#### CAP-13: Uncertainty-Aware Interpretation
| Aspect | Definition |
|---|---|
| **Positive behavior** | Calibrate confidence based on evidence strength. Hedge appropriately when evidence is ambiguous ("likely", "possibly"). |
| **Negative behavior** | Assert with high confidence when evidence is weak, or understate confidence when evidence is strong. |
| **Success** | Confidence value correlates with evidence quality (more/stronger evidence → higher confidence; sparse/conflicting evidence → lower confidence). |
| **False positive** | Universally low or universally high confidence regardless of evidence. |
| **Evidence citation** | Must cite evidence that supports the stated confidence level. |

---

### Tier 3: Infrastructure Behavior

#### CAP-00: Schema Compliance & Structured Output
| Aspect | Definition |
|---|---|
| **Positive behavior** | Produce syntactically valid JSON matching `SemanticInterpretation` schema exactly. No literal `...` elision, no missing keys, no trailing fences. |
| **Negative behavior** | Output unparseable text, truncated JSON, placeholder syntax, or extra markdown text outside the JSON object (6A.2 produced 53.3% format failures). |
| **Success** | 100% schema validation pass rate by the provider parser adapter. |
| **False positive** | Syntactically valid JSON containing non-grounded or hallucinated data. |
| **Evidence citation** | All required JSON schema fields must be present and valid. |

---

## 3. Training Semantics of Negative Examples & Representation

### 3.1 The Negative Training Semantics Problem

If a negative candidate string (e.g. a percept restatement "The lights went out") is stored inside `target_interpretation.proposition`, standard supervised fine-tuning (SFT) would train the model via teacher forcing to **generate the exact string we want it to reject**.

Conversely, if negative cases simply set `target_interpretation = null`, negative cases become indistinguishable from genuine `ABSTAIN` cases unless the training representation explicitly separates generation targets from candidate annotations.

### 3.2 Dual Representation Architecture

To resolve this ambiguity, every dataset record separates **Supervised Generation Targets** from **Candidate Rejection Supervision Artifacts**:

```text
Record Field                     Purpose in Dataset Specification
───────────────────────────────  ────────────────────────────────────────────────────────────────────────────
target_interpretation            SUPERVISED GENERATION TARGET. Contains ONLY valid Gold Positive
                                 propositions or null (empty JSON output). Negative candidate text MUST
                                 NEVER appear in this field.

rejected_candidates              SUPERVISION ARTIFACTS. An array of invalid candidate objects (REPEAT,
                                 PARAPHRASE, EPISTEMICALLY_PREMATURE, etc.) stored as metadata.
                                 
                                 Principle: rejected_candidates are supervision artifacts. The eventual
                                 training objective (e.g. Unlikelihood Loss, DPO/ORPO, contrastive preference,
                                 candidate ranking, or reranker supervision) is a Phase 6C implementation
                                 decision. The dataset specification remains 100% model- and algorithm-
                                 independent.
```

### 3.3 Explicit Target & Candidate Mapping by Taxonomy Class

| Taxonomy Class | `positive_negative` | `target_interpretation` (Generation Target) | `rejected_candidates` (Supervision Artifacts) | `abstention_label` |
|---|---|---|---|---|
| **Gold Positive** (`SEMANTIC_NOVEL` + `DECISION_RELEVANT`) | `"POSITIVE"` | Valid `SemanticInterpretation` JSON | Array of hard negative traps (percept echoes, premature guesses) | `SHOULD_PROPOSE` |
| **Weak Positive** (`SEMANTIC_NOVEL` + `DECISION_IRRELEVANT`) | `"POSITIVE"` | Valid `SemanticInterpretation` JSON (or `null` if task-gated) | Array of derivable / tangential traps | `SHOULD_PROPOSE` |
| **Derivable Repeat** (`REPEAT` / `PARAPHRASE` / `RULE_ECHO` / `TAXONOMY_ECHO`) | `"NEGATIVE"` | `null` (empty output: `{"hypotheses": []}`) | `[{"proposition": "<derivable string>", "novelty_label": "REPEAT", ...}]` | `SHOULD_ABSTAIN` |
| **Unsupported / Premature** (`UNSUPPORTED` / `EPISTEMICALLY_PREMATURE`) | `"NEGATIVE"` | `null` (empty output: `{"hypotheses": []}`) | `[{"proposition": "<premature string>", "novelty_label": "EPISTEMICALLY_PREMATURE", ...}]` | `SHOULD_ABSTAIN` |
| **Genuine Abstention** (`ABSTAIN`) | `"NEGATIVE"` | `null` (empty output: `{"hypotheses": []}`) | `[]` (or array of rejected speculations if prompted) | `SHOULD_ABSTAIN` |

---

### 3.4 Technical Distinction: Abstention vs. Candidate Rejection

| Dimension | Genuine Abstention (`ABSTAIN`) | Negative Candidate Rejection |
|---|---|---|
| **Context State** | Evidence in context is sparse, ambiguous, or absent. | Evidence exists in context, but candidate text is invalid (derivable, paraphrase, premature). |
| **`target_interpretation`** | `null` | `null` (if no gold positive exists for case) OR Gold Positive (if gold positive exists alongside rejected candidate). |
| **`abstention_label`** | `SHOULD_ABSTAIN` | `SHOULD_ABSTAIN` (if no gold positive) or `SHOULD_PROPOSE` (if gold positive exists). |
| **`rejected_candidates`** | `[]` | `[{"proposition": "<rejected string>", "reason": "..."}]` |
| **Loss Function Role** | Supervises model to output `{"hypotheses": []}` on sparse context. | Supervises unlikelihood / preference loss against specific rejected string while training SFT on gold positive or empty output. |

---

## 4. Complete Record Schema & Symbolic Derivability Oracle

```jsonc
{
  // ═══════════════════════════════════════════
  // INPUT — what the model receives at inference
  // ═══════════════════════════════════════════
  "case_id": "string",                    // URI-style: td://v0/<family>/<index>
  "percept": "string",                    // raw observation text
  "task": "string",                       // the decision question the interpretation feeds
  "concepts": [                           // ConceptSnapshot[]
    {
      "id": "string",                     // concept://...
      "label": "string",
      "definition": "string",
      "concept_type": "string"
    }
  ],
  "concept_edges": [                      // taxonomy/relation edges
    {
      "source": "string",                 // concept://...
      "target": "string",                 // concept://...
      "relation": "string"               // is_a | related_to | ...
    }
  ],
  "beliefs": [                            // BeliefSnapshot[]
    {
      "id": "string",                     // belief://...
      "proposition": "string",
      "confidence": 0.0                   // number, 0.0–1.0
    }
  ],
  "belief_edges": [                       // contradiction/support edges
    {
      "source": "string",
      "target": "string",
      "relation": "string"               // contradicts | supports | ...
    }
  ],
  "rules": [                              // RuleSnapshot[]
    {
      "id": "string",                     // rule://...
      "name": "string",
      "premise_text": "string",
      "conclusion_text": "string"
    }
  ],
  "grounding_snapshot": {                 // GroundingSnapshot
    "concept_ids": ["string"],
    "belief_ids": ["string"],
    "rule_ids": ["string"],
    "evidence_ids": ["string"]
  },

  // ═══════════════════════════════════════════
  // TARGET — Supervised SFT generation target
  // (MUST BE null FOR NEGATIVE / ABSTAIN CASES)
  // ═══════════════════════════════════════════
  "target_interpretation": {              // SemanticInterpretation OR null
    "proposition": "string",              // ≤120 chars, single declarative sentence
    "supporting_evidence_ids": ["string"],// grounding IDs the interpretation rests on
    "referenced_concept_ids": ["string"], // grounding IDs used in the proposition
    "semantic_relation": "string",        // causal|explanation|paraphrase|contrast|category|state|other
    "confidence": 0.0                     // 0.0–1.0, target strength
  },

  // ═══════════════════════════════════════════
  // ANNOTATION — Oracle verification & rejected candidate targets
  // ═══════════════════════════════════════════
  "derivability": {
    "label": "NON_DERIVABLE",             // DERIVABLE | NON_DERIVABLE
    "derivation_type": "none",            // none | percept_match | belief_echo | rule_chain | taxonomy_edge
    "derivation_trace": null,             // array of strings e.g. ["rule://causal/rain -> belief://inf/1"] or null
    "checked_by": "symbolic_oracle_v1"
  },
  "rejected_candidates": [                // Array of candidates for Contrastive/Unlikelihood training
    {
      "candidate_id": "string",
      "proposition": "string",
      "supporting_evidence_ids": ["string"],
      "referenced_concept_ids": ["string"],
      "novelty_label": "string",         // REPEAT|PARAPHRASE|TAXONOMY_ECHO|RULE_ECHO|UNSUPPORTED|EPISTEMICALLY_PREMATURE
      "rejection_reason": "string",
      "oracle_derivation_trace": null
    }
  ],
  "novelty_label": "string",             // REPEAT|PARAPHRASE|TAXONOMY_ECHO|RULE_ECHO|UNSUPPORTED|EPISTEMICALLY_PREMATURE|SEMANTIC_NOVEL|ABSTAIN
  "derivability_label": "string",        // DERIVABLE|NON_DERIVABLE
  "decision_relevance": "string",        // DECISION_RELEVANT|DECISION_IRRELEVANT
  "abstention_label": "string",          // SHOULD_ABSTAIN|SHOULD_PROPOSE
  "difficulty_tier": 0,                  // integer, 0–5 (§7)
  "capability_family": "string",         // CAP-01..CAP-13 primary capability tested
  "capability_families_secondary": [],   // additional capabilities tested
  "positive_negative": "string",         // POSITIVE|NEGATIVE
  "evidence_count": 0,                   // count of relevant evidence items
  "distractor_count": 0,                 // count of distractor items
  "contradiction_present": false,        // whether beliefs contain contradictions
  "trap_propositions": ["string"],       // known traps for this case

  // ═══════════════════════════════════════════
  // PROVENANCE — lineage and dual human review
  // ═══════════════════════════════════════════
  "provenance": {
    "generator_id": "string",            // generator identifier
    "generator_version": "string",       // semver
    "template_id": "string",             // template ID
    "seed_case_id": "string",            // conceptual seed
    "random_seed": 0,                    // for reproducibility
    "generation_timestamp": "string",    // ISO 8601
    "human_review_status": "string",     // UNREVIEWED|REVIEWED|GOLD|REJECTED
    "reviewer_1_id": "string",           // first reviewer ID
    "reviewer_2_id": "string",           // second reviewer ID (mandatory for gold positives)
    "review_timestamp": "string",       // ISO 8601
    "review_notes": "string",           // free-text rationale
    "source_type": "string"             // HUMAN_AUTHORED|DETERMINISTIC_TEMPLATE|SYNTHETIC|QWEN_NEGATIVE
  }
}
```

---

## 5. Machine-Checkable Schema Invariants

To prevent any implementation bug or dataset generator defect, every record in the dataset must pass 9 machine-checkable validation invariants prior to dataset release:

```text
Invariant Identifier       Validation Rule & Assertion
─────────────────────────  ─────────────────────────────────────────────────────────────────────────────────────────────
INV-01: Positive Target    IF positive_negative == "POSITIVE" AND novelty_label == "SEMANTIC_NOVEL",
                           THEN target_interpretation MUST NOT BE null.

INV-02: Negative Target    IF positive_negative == "NEGATIVE" OR novelty_label != "SEMANTIC_NOVEL",
                           THEN target_interpretation MUST BE null. (Prevents negative generation target).

INV-03: Candidate Isolation target_interpretation.proposition MUST NEVER equal any proposition string in rejected_candidates[*].

INV-04: Abstention Target  IF abstention_label == "SHOULD_ABSTAIN",
                           THEN target_interpretation MUST BE null.

INV-05: Derivability Novel  IF novelty_label == "SEMANTIC_NOVEL",
                           THEN derivability.label MUST BE "NON_DERIVABLE".

INV-06: Derivable Ban      IF derivability.label == "DERIVABLE",
                           THEN novelty_label MUST NOT BE "SEMANTIC_NOVEL".

INV-07: Grounding Bounds   ALL IDs in target_interpretation AND rejected_candidates MUST exist in grounding_snapshot.

INV-08: Dual Review Gold   IF human_review_status == "GOLD",
                           THEN reviewer_1_id AND reviewer_2_id MUST BE present AND reviewer_1_id != reviewer_2_id.

INV-09: Oracle Consistency derivability.label MUST EQUAL derivability_label.
```

---

## 6. Concrete Instantiated Examples

### Example 1: Gold Positive Record (`SEMANTIC_NOVEL` + `DECISION_RELEVANT`)

```json
{
  "case_id": "td://v0/abduction/002",
  "percept": "The lights went out. The microwave clock was blinking. The fridge hummed to life.",
  "task": "what explains the observations?",
  "concepts": [
    {"id": "concept://power", "label": "power", "definition": "electrical power supply", "concept_type": "entity"},
    {"id": "concept://electricity", "label": "electricity", "definition": "electric current", "concept_type": "substance"},
    {"id": "concept://light", "label": "light", "definition": "illumination fixture", "concept_type": "entity"},
    {"id": "concept://outage", "label": "outage", "definition": "temporary loss of power", "concept_type": "event"}
  ],
  "concept_edges": [],
  "beliefs": [],
  "belief_edges": [],
  "rules": [],
  "grounding_snapshot": {
    "concept_ids": ["concept://power", "concept://electricity", "concept://light", "concept://outage"],
    "belief_ids": [],
    "rule_ids": [],
    "evidence_ids": ["concept://power", "concept://electricity", "concept://light"]
  },
  "target_interpretation": {
    "proposition": "There is a power outage.",
    "supporting_evidence_ids": ["concept://power", "concept://light", "concept://electricity"],
    "referenced_concept_ids": ["concept://power", "concept://outage"],
    "semantic_relation": "explanation",
    "confidence": 0.95
  },
  "derivability": {
    "label": "NON_DERIVABLE",
    "derivation_type": "none",
    "derivation_trace": null,
    "checked_by": "symbolic_oracle_v1"
  },
  "rejected_candidates": [
    {
      "candidate_id": "cand://002/neg1",
      "proposition": "The lights went out.",
      "supporting_evidence_ids": ["concept://light"],
      "referenced_concept_ids": ["concept://light"],
      "novelty_label": "REPEAT",
      "rejection_reason": "Verbatim percept restatement",
      "oracle_derivation_trace": ["percept_input: line 1"]
    }
  ],
  "novelty_label": "SEMANTIC_NOVEL",
  "derivability_label": "NON_DERIVABLE",
  "decision_relevance": "DECISION_RELEVANT",
  "abstention_label": "SHOULD_PROPOSE",
  "difficulty_tier": 2,
  "capability_family": "CAP-02",
  "capability_families_secondary": ["CAP-01", "CAP-07"],
  "positive_negative": "POSITIVE",
  "evidence_count": 3,
  "distractor_count": 0,
  "contradiction_present": false,
  "trap_propositions": ["The lights went out."],
  "provenance": {
    "generator_id": "gen_abduction_v1",
    "generator_version": "0.1.0",
    "template_id": "tmpl_outage_pattern",
    "seed_case_id": "sp1://b/002",
    "random_seed": 42,
    "generation_timestamp": "2026-08-11T14:00:00Z",
    "human_review_status": "GOLD",
    "reviewer_1_id": "rev_human_01",
    "reviewer_2_id": "rev_human_02",
    "review_timestamp": "2026-08-11T14:15:00Z",
    "review_notes": "Unobserved cause, 100% agreement between reviewers",
    "source_type": "HUMAN_AUTHORED"
  }
}
```

---

### Example 2: Negative Candidate Rejection (`REPEAT` / `PARAPHRASE`)

```json
{
  "case_id": "td://v0/paraphrase/001",
  "percept": "The container shattered after hitting the floor.",
  "task": "what happened?",
  "concepts": [
    {"id": "concept://container", "label": "container", "definition": "receptacle", "concept_type": "entity"},
    {"id": "concept://floor", "label": "floor", "definition": "surface", "concept_type": "entity"},
    {"id": "concept://impact", "label": "impact", "definition": "collision event", "concept_type": "event"},
    {"id": "concept://break", "label": "break", "definition": "fracture event", "concept_type": "event"}
  ],
  "concept_edges": [],
  "beliefs": [],
  "belief_edges": [],
  "rules": [],
  "grounding_snapshot": {
    "concept_ids": ["concept://container", "concept://floor", "concept://impact", "concept://break"],
    "belief_ids": [],
    "rule_ids": [],
    "evidence_ids": ["concept://container", "concept://floor"]
  },
  "target_interpretation": null,
  "derivability": {
    "label": "DERIVABLE",
    "derivation_type": "percept_match",
    "derivation_trace": ["percept_input: 'shattered after hitting the floor'"],
    "checked_by": "symbolic_oracle_v1"
  },
  "rejected_candidates": [
    {
      "candidate_id": "cand://p001/neg1",
      "proposition": "The container was broken by the impact on the floor.",
      "supporting_evidence_ids": ["concept://container", "concept://floor"],
      "referenced_concept_ids": ["concept://container", "concept://break"],
      "novelty_label": "PARAPHRASE",
      "rejection_reason": "Surface rephrasing of percept content; no new semantic proposition",
      "oracle_derivation_trace": ["percept_input -> verbatim token mapping"]
    }
  ],
  "novelty_label": "PARAPHRASE",
  "derivability_label": "DERIVABLE",
  "decision_relevance": "DECISION_IRRELEVANT",
  "abstention_label": "SHOULD_ABSTAIN",
  "difficulty_tier": 0,
  "capability_family": "CAP-05",
  "capability_families_secondary": [],
  "positive_negative": "NEGATIVE",
  "evidence_count": 2,
  "distractor_count": 0,
  "contradiction_present": false,
  "trap_propositions": ["The container was broken by the impact on the floor."],
  "provenance": {
    "generator_id": "gen_paraphrase_v1",
    "generator_version": "0.1.0",
    "template_id": "tmpl_paraphrase_trap",
    "seed_case_id": "sp1://a/001",
    "random_seed": 101,
    "generation_timestamp": "2026-08-11T14:05:00Z",
    "human_review_status": "REVIEWED",
    "reviewer_1_id": "rev_human_01",
    "reviewer_2_id": null,
    "review_timestamp": "2026-08-11T14:20:00Z",
    "review_notes": "Confirmed derivability echo. Target interpretation set to null.",
    "source_type": "DETERMINISTIC_TEMPLATE"
  }
}
```

---

### Example 3: Epistemically Premature Abstention (`NEG-14`)

```json
{
  "case_id": "td://v0/premature/014",
  "percept": "The road is wet. The sky is dark.",
  "task": "what weather is present?",
  "concepts": [
    {"id": "concept://road", "label": "road", "definition": "thoroughfare", "concept_type": "entity"},
    {"id": "concept://wet", "label": "wet", "definition": "moist condition", "concept_type": "property"},
    {"id": "concept://sky", "label": "sky", "definition": "atmosphere", "concept_type": "entity"},
    {"id": "concept://rain", "label": "rain", "definition": "precipitation", "concept_type": "weather"}
  ],
  "concept_edges": [],
  "beliefs": [],
  "belief_edges": [],
  "rules": [],
  "grounding_snapshot": {
    "concept_ids": ["concept://road", "concept://wet", "concept://sky", "concept://rain"],
    "belief_ids": [],
    "rule_ids": [],
    "evidence_ids": ["concept://road", "concept://wet", "concept://sky"]
  },
  "target_interpretation": null,
  "derivability": {
    "label": "NON_DERIVABLE",
    "derivation_type": "none",
    "derivation_trace": null,
    "checked_by": "symbolic_oracle_v1"
  },
  "rejected_candidates": [
    {
      "candidate_id": "cand://n14/premature1",
      "proposition": "It is raining.",
      "supporting_evidence_ids": ["concept://wet", "concept://sky"],
      "referenced_concept_ids": ["concept://rain"],
      "novelty_label": "EPISTEMICALLY_PREMATURE",
      "rejection_reason": "Plausible but unsupported: wet road + dark sky lacks direct rain evidence (could be street cleaner or post-rain clouds). Asserting rain is premature.",
      "oracle_derivation_trace": null
    }
  ],
  "novelty_label": "EPISTEMICALLY_PREMATURE",
  "derivability_label": "NON_DERIVABLE",
  "decision_relevance": "DECISION_IRRELEVANT",
  "abstention_label": "SHOULD_ABSTAIN",
  "difficulty_tier": 5,
  "capability_family": "CAP-09",
  "capability_families_secondary": ["CAP-13"],
  "positive_negative": "NEGATIVE",
  "evidence_count": 2,
  "distractor_count": 0,
  "contradiction_present": false,
  "trap_propositions": ["It is raining."],
  "provenance": {
    "generator_id": "gen_premature_v1",
    "generator_version": "0.1.0",
    "template_id": "tmpl_premature_rain",
    "seed_case_id": "neg14_seed",
    "random_seed": 202,
    "generation_timestamp": "2026-08-11T14:10:00Z",
    "human_review_status": "REVIEWED",
    "reviewer_1_id": "rev_human_02",
    "reviewer_2_id": null,
    "review_timestamp": "2026-08-11T14:25:00Z",
    "review_notes": "Plausible != Supported test case. Target interpretation set to null.",
    "source_type": "HUMAN_AUTHORED"
  }
}
```

---

## 7. Negative Example Families & Tempting Wrong Answers

### Standard Negatives (NEG-01 to NEG-13)
- **NEG-01:** Percept Restatement (`REPEAT`)
- **NEG-02:** Paraphrase Disguised as Novelty (`PARAPHRASE`)
- **NEG-03:** Belief Echo (`REPEAT`)
- **NEG-04:** Rule-Conclusion Echo (`RULE_ECHO`)
- **NEG-05:** Taxonomy Echo (`TAXONOMY_ECHO`)
- **NEG-06:** Unsupported Plausible Answer (`UNSUPPORTED`)
- **NEG-07:** Distractor-Supported Answer (`UNSUPPORTED`)
- **NEG-08:** Contradictory Unsupported Answer (`UNSUPPORTED`)
- **NEG-09:** Malformed Structured Output (`E0`)
- **NEG-10:** Unknown Grounding IDs (`E1`)
- **NEG-11:** Invented Entities (`E1`)
- **NEG-12:** Overconfident Interpretation (`UNSUPPORTED`)
- **NEG-13:** Answer That Does Not Answer the Task (`DECISION_IRRELEVANT`)

---

### NEG-14: Correct but Epistemically Premature (Plausible ≠ Supported)

**What:** Proposal is plausible or even likely in the real world, but the available evidence in the percept/beliefs is insufficient to establish it.
**Label:** `EPISTEMICALLY_PREMATURE`, `SHOULD_ABSTAIN`
**Example:**
- *Percept:* "The road is wet. The sky is dark."
- *Candidate Proposal:* "It is raining."
- *Rationale:* While rain is plausible, wet road + dark sky could be caused by street cleaners, melted snow, or storm clouds prior to rain. Without direct evidence (e.g. umbrella, falling drops), asserting rain is epistemically premature. Correct behavior: ABSTAIN or hedge explicitly.
**Teaching signal:** Plausible ≠ Supported. Never assert an ungrounded inference simply because it is common.

#### Explicit Evidence Threshold Doctrine (b/002 vs NEG-14):
To ensure NEG-14 does not punish legitimate abductive reasoning (like b/002):
- **Legitimate Abduction (b/002):** Multiple co-occurring anomalies (lights out + microwave blinking + fridge humming) *strongly support a common power-related cause* (power outage) because alternative explanations do not readily account for all three simultaneously. The distinction remains: deduction is logically derivable; abduction is a strongly supported explanation that may remain uncertain.
- **Epistemically Premature (NEG-14):** Observations (wet road + dark sky) have multiple independent, equally plausible explanations (street cleaning, melted snow, dusk). Asserting rain without a deciding evidence item (umbrella, raindrops) is premature.

---

### Adversarial "Tempting Wrong Answers" Collection

The dataset explicitly incorporates hard adversarial negatives designed to break superficial heuristics:

1. **Strongly plausible but unsupported:** (NEG-14 / NEG-06)
2. **Semantically close but wrong:** Target concept is adjacent to the correct explanation (e.g., inferring "hurricane" when evidence only supports "thunderstorm").
3. **Correct conclusion with wrong evidence:** Valid proposition citing distractor evidence IDs.
4. **Correct evidence with wrong conclusion:** Relevant evidence IDs cited to support a non-sequitur proposition.
5. **Correct interpretation but goal-irrelevant:** Valid E5 interpretation that does not address the case's task (NEG-13).
6. **Already derivable but reworded:** Paraphrased rule conclusion or belief (NEG-02/04).
7. **Contradiction presented as novelty:** Conjunction of conflicting beliefs without resolution (NEG-08).

---

## 8. Difficulty Curriculum

Examples progress across Tiers 0–5:
- **Tier 0:** Pattern Detection (Discriminate REPEAT from non-REPEAT)
- **Tier 1:** Single-Step Semantic Interpretation
- **Tier 2:** Multi-Evidence Interpretation
- **Tier 3:** Distractor Resistance + Evidence Selection
- **Tier 4:** Contradiction + Indirect Evidence
- **Tier 5:** Ambiguous/Uncertain Cases Requiring Abstention & Hedging

---

## 9. Data Generation Strategy & Qwen Failure Oracle

### 9.1 Generation Methods
1. **Human-Authored Cases:** Highest reliability for gold seeds.
2. **Deterministic Templates:** High reliability for structured variations.
3. **Symbolic Perturbation:** Systematic masking, distractor injection, and contradiction insertion.
4. **Controlled Synthetic Generation:** Rule-based generators under strict validation.

### 9.2 Qwen Failure Oracle Doctrine
Outputs from the Qwen reference run (Phase 6A.1 & 6A.2) serve strictly as a **Failure Oracle**:
- Used to construct hard negatives (format failures, premise echoes, distractor citations).
- **PROHIBITED:** Qwen outputs MUST NEVER be used as positive training labels under any circumstances.

---

## 10. Ground-Truth Doctrine & Dual Human Review

### 10.1 Human Review Protocol Matrix

Dataset quality is paramount. Positive labels must undergo strict dual human review.

| Case Category | Review Requirement | Minimum Reviewers | Approval Rule |
|---|---|---|---|
| **Gold Positive** (`SEMANTIC_NOVEL` + `DECISION_RELEVANT`) | **100% Dual Human Review** | **2 Reviewers** | 100% Agreement required → GOLD |
| **Positive** (`SEMANTIC_NOVEL` + `DECISION_IRRELEVANT`) | Dual Review Preferred | 2 Reviewers (min 1) | Agreement required |
| **Abstention** (`SHOULD_ABSTAIN`) | Full Human Review | 1 Reviewer | Confirm evidence is truly insufficient |
| **Repeat / Paraphrase / Echo** | Automated + Sampled Review | 1 Reviewer | Automated oracle check + ≥20% human spot-check |
| **Unsupported / Premature** | Automated + Sampled Review | 1 Reviewer | Oracle check + sampled review |
| **Synthetic Data Batch** | Tiered by label importance | Per matrix above | Batch gate pass |

---

## 11. Train/Validation/Evaluation Separation

- **Evaluation Instruments Frozen:** 51-case benchmark and 15-case semantic probe are evaluation-only, forever.
- **Strict Leakage Prevention:** Automated n-gram and ID checks prevent evaluation cases or exact paraphrases from entering training/validation.

---

## 12. Dataset Scale Target (Quality over Quantity)

### Sizing Principle: Coverage × Quality × Variation

Dataset scale is specified as a **target range**, not a rigid quota. The project explicitly prefers **~400 excellent, dual-reviewed examples** over 1,000 synthetic examples with repetitive templates.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ TARGET DATASET SCALE RANGE                                                  │
│                                                                             │
│  • Target Training Records:    350 – 500 records                            │
│  • Target Validation Records:   80 – 120 records                            │
│  • Total Target Range:         430 – 620 records                            │
│                                                                             │
│  Optimization metric: Maximize semantic diversity and reviewer agreement    │
│  rather than raw record count.                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Balance Requirements & Anti-Shortcut Measures

- **Positive/Negative Balance:** ~40% Positive / 60% Negative & Abstention.
- **Anti-Shortcut Testing:** Automated correlation checks ensure evidence count, sentence length, or concept type do not act as label shortcuts.

---

## 14. Data Versioning & Manifest

Dataset releases carry full semver lineage (`ds-v0.0`, `spec-v0.1`, `gen-v0.1.0`) with checksummed manifests (`dataset-manifest.json`).

---

## 15. Evaluation Mapping

Maps each training capability directly to frozen evaluation instruments (Probe Groups A–E & 51-case benchmark).

---

## 16. Model Independence

Specification remains 100% model-independent. Logical JSON records convert to model-specific token formats only at fine-tuning execution time.

---

## 17. Target v0 Composition Summary

```text
Dimension                          Target Range
─────────────────────────────────  ──────────────────────────────────────────
Total Training Records             350 – 500
Total Validation Records           80 – 120
Gold Positive (Dual Reviewed)      140 – 180 (100% dual human review)
Negatives & Tempting Traps         180 – 240 (Automated oracle + spot-check)
First-Class Abstention Cases       40 – 60 (100% human reviewed)
```

---

## 18. Governance & Review Gate Status

**Status:** REVISED DRAFT FOR REVIEW — negative training semantics & schema invariants locked; ready for human review gate.

### Prerequisites for Phase 6B.2 (Dataset Generation):
1. Dataset specification locked (this document).
2. Derivability Oracle tool/module integrated for generator annotation.
3. Dual-reviewer workflow setup.
4. Schema invariant validator implemented (`scripts/validate_dataset_schema.py`).
5. **STOP:** No dataset generation or model selection until human explicitly triggers Phase 6B.2 execution.
