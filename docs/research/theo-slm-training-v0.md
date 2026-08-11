# THEO SLM Training Objective v0 — Phase 6C.1

**Status:** DRAFT FOR REVIEW — training objective, supervision semantics, and
model-selection gate; no training, no base-model selection, no downloads until
reviewed.
**Date:** 2026-08-11
**Scope:** Phase 6C.1 — Steps 2, 3, 4, 5, 6, 8, 9 of the 6C.1 plan.
**Depends on:**
- `docs/research/theo-slm-design-v0.md` (6B design; sections A–G accepted)
- `docs/research/theo-slm-dataset-v0-gold-freeze.md` (6C.1 Steps 1 & 7; corpus
  freeze and corpus audit)
- `theo-data/datasets/theo_slm_v0_gold/` (the frozen, human-reviewed corpus:
  67 GOLD_POSITIVE / 66 GOLD_ABSTAIN / 131 HARD_NEGATIVE)

This document is deliberately **algorithm-agnostic**. It specifies *what* the
model must learn and *how the labels are derived*, but commits to no training
procedure (no SFT, DPO, ORPO, unlikelihood, contrastive, or reranker is chosen
here). The procedure is decided after this objective and the frozen evaluation
instruments are approved.

---

## 1. Training objective (WHAT to learn)

### 1.1 Objective statement

Train a small language model that produces a **grounded semantic interpretation**
of an input percept — a proposition that goes beyond literal percept text and is
supported by supplied evidence — **useful precisely when the symbolic runtime
cannot derive it**, and that **reliably abstains when evidence is insufficient**.

The defining behavioral properties, in decreasing priority:

| # | property | meaning |
|---|---|---|
| P1 | **derive-vs-repeat discrimination** | Propose only what the symbolic runtime cannot already derive; never re-assert percept text, paraphrases, taxonomy facts, rule conclusions, or conjunctions of inputs as novel |
| P2 | **epistemic thresholding** | Distinguish "plausible" from "sufficiently supported"; abstain rather than propose on insufficient evidence |
| P3 | **evidence relevance** | Cite only evidence that actually supports the interpretation; ignore distractors |
| P4 | **grounding validity** | Reference only ids present in the supplied grounding snapshot; reject/avoid unknowns |
| P5 | **abductive interpretation** | Infer an unobserved cause/explanation that accounts for the observed facts (b/002 pattern) |
| P6 | **contradiction interpretation** | Interpret conflicting stored beliefs (light on ∧ room dark) without asserting the contradiction as a fact or resolving it without support |
| P7 | **decision relevance** | The interpretation answers the decision task it feeds (highest-value proposals) |
| P8 | **format reliability** | Emit a single well-formed `SemanticInterpretation`, or nothing; no partial/malformed output |

P1–P4 are hard gates: a model that fails any of them is not useful. P5–P8 are
graded.

### 1.2 Derivability awareness (P1)

For every input, the model must determine whether the symbolic runtime could
already derive the interpretation from its possessions (beliefs, rules,
concepts). The abstract dichotomy is:

- **DERIVABLE** — a rule or belief already yields the conclusion. Proposing it
  as new interpretation is a *repeat/echo* and must be suppressed.
- **NON_DERIVABLE** — no rule/belief yields the conclusion; it must come from
  interpretation of evidence. Only these are eligible for proposal.

This is a *classification of the interpretation relative to the supplied
symbolic knowledge* — not a property of the surface text alone. Percept
paraphrases ("broken by the impact on the floor" for "shattered after hitting
the floor") are derivable-in-content and must be recognized as repeats even when
the wording differs (a/001 pattern).

### 1.3 Semantic novelty taxonomy

The model must distinguish these semantic classes of a candidate proposition
(ordered from not-novel to novel):

| class | example | verdict |
|---|---|---|
| `REPEAT` | verbatim percept echo | HARD_NEGATIVE |
| `PARAPHRASE` | same content, different wording | HARD_NEGATIVE |
| `RULE_ECHO` | conclusion already implied by a rule | HARD_NEGATIVE |
| `TAXONOMY_ECHO` | re-asserting an is-a/category fact as if new | HARD_NEGATIVE |
| `DECISION_IRRELEVANT` | non-derivable but does not answer the task (e.g., ambient instrumental observations) | HARD_NEGATIVE |
| `UNSUPPORTED` | new but not supported by supplied evidence | GOLD_ABSTAIN |
| `EPISTEMICALLY_PREMATURE` | plausible but below the sufficiency threshold | GOLD_ABSTAIN |
| `SEMANTIC_NOVEL` | non-derivable, evidence-supported, beyond literal text | GOLD_POSITIVE |

The corpus realizes every class except `PARAPHRASE` and `RULE_ECHO` (masked
expected-novelty distribution: GOLD_POSITIVE = SEMANTIC_NOVEL 67;
GOLD_ABSTAIN = UNSUPPORTED 64 / EPISTEMICALLY_PREMATURE 2; HARD_NEGATIVE =
DECISION_IRRELEVANT 64 / REPEAT 65 / SEMANTIC_NOVEL 2 human-override). Where a
class has no realized examples, the frozen evaluation instruments supply the
behavioral target; corpus coverage of `PARAPHRASE`/`RULE_ECHO` is a documented
data-coverage gap, not a license to skip the capability.

### 1.4 Evidence reasoning (P3, P5)

- **Support** — an interpretation must be supported by evidence actually present
  in the percept/grounding, not merely thematically related.
- **Distractor rejection** — with distractor ids present, cite only the relevant
  ones (Group E pattern: cite rain evidence, not `sky`/`wet`).
- **Multi-evidence abduction** — several observations → one unifying cause,
  rather than concatenation of observations.
- **Indirect evidence** — bridge gaps where the conclusion is hinted at, not
  stated.

### 1.5 Epistemic thresholding (P2)

- `SHOULD_PROPOSE` — evidence-sufficient, non-derivable, decision-relevant.
- `SHOULD_ABSTAIN` — otherwise. **Plausible ≠ sufficiently supported**; a
  conclusion that might be true but is not supported by the supplied evidence is
  an abstain, not a proposal.
- The abstain class is first-class: the model must be able to *decline*, and
  declining must never be scored as an error equal to malformed output. Empty
  output (no `SemanticInterpretation`) is the legal, preferred abstention
  representation.

### 1.6 Contradiction interpretation (P6)

When the supplied beliefs are contradictory (light on ∧ room dark), the model
must detect the conflict and produce an interpretation that accounts for it
(e.g., light is broken) — **without** echoing the contradiction as a conjunction
and **without** unilaterally resolving it beyond what evidence supports.

### 1.7 Grounding (P4)

Every id in `supporting_evidence_ids` and `referenced_concept_ids` must resolve
against the supplied grounding snapshot. Unknown/non-URI ids are rejected by the
adapter (E0-class failure) and must be treated as a training failure whenever
they appear in training output.

### 1.8 Output

The model's output is a minimal, schema-constrained `SemanticInterpretation`
(design doc C, unchanged):

```text
SemanticInterpretation
    proposition: string               # one declarative sentence, <= 120 chars
    supporting_evidence_ids: string[] # grounding ids the interpretation rests on
    referenced_concept_ids: string[]  # grounding ids used in the proposition
    semantic_relation: string         # causal | explanation | paraphrase |
                                      # contrast | category | state | other
    confidence: number                # 0.0..1.0
```

Empty output is legal and preferred over garbage. The deterministic adapter
(schema + grounding validation → `HypothesisProposal`) is unchanged from the
design doc; malformed output is rejected whole (E0), never partially recovered.

---

## 2. Exclusions (what the model must NOT learn)

Restated from design doc B and made sharper for training data construction.
The model must never be trained to:

1. make final decisions — the runtime decides;
2. commit to or modify beliefs — no belief writes, no `ThoughtRecord` DTOs;
3. create `DecisionRecord` DTOs or fire rules — no `RuleRecord` DTOs;
4. mutate symbolic state (beliefs, rules, memory) in any form;
5. bypass grounding verification — no `grounded=False` fallback;
6. replace symbolic inference/verification — deterministic reasoning stays in
   the runtime;
7. act as an authority source — the model proposes, the runtime disposes;
8. emit symbolic runtime objects directly — only `SemanticInterpretation`, via
   the adapter.

These behaviors must also be **absent from the training data** (no example ever
demonstrates a model making a decision, writing a belief, or emitting a DTO), so
they are never learned. The provider boundary (snapshot DTOs in, interpreted
proposal out) is mechanically enforced as before.

---

## 3. Supervision semantics (how the frozen corpus becomes labels)

### 3.1 Primary supervision

The three-way human `final_status` is the primary label, taken verbatim from the
accepted corpus (264 records):

- **GOLD_POSITIVE** (67) — the record's `candidate_proposition` is the target
  output proposition.
- **GOLD_ABSTAIN** (66) — the target is "no supported interpretation" (empty
  output).
- **HARD_NEGATIVE** (131) — the target is "no supported interpretation" (empty
  output); additionally the candidate must be *recognized as non-proposable*.

### 3.2 Abstract supervision categories (no algorithm commitment)

Labels are organized into abstract categories so any training procedure can
consume them without prescribing a loss:

| category | source | fields |
|---|---|---|
| positive target | corpus `candidate_proposition` | proposition text |
| output structured fields | candidate `target_interpretation` (read-only join) | supporting_evidence_ids, referenced_concept_ids, semantic_relation, confidence |
| abstention | `final_status = GOLD_ABSTAIN` | empty output |
| hard negative | `final_status = HARD_NEGATIVE` | empty output; candidate non-proposability |
| grounding | corpus `grounding_snapshot` + candidate `target_interpretation` | every target id must resolve |
| derivability | masked `_masked_expected_derivability` | DERIVABLE / NON_DERIVABLE |
| evidence relevance | record booleans (`evidence_relevance`, `evidence_sufficiency`) + candidate `supporting_evidence_ids` | which ids support the label |
| epistemic threshold | masked `_masked_expected_abstention` + novelty | SHOULD_PROPOSE / SHOULD_ABSTAIN; UNSUPPORTED / EPISTEMICALLY_PREMATURE |

### 3.3 Read-only joins

The gold corpus is immutable and carries review fields only. Three deterministic,
read-only joins supply supplementary supervision and **alter nothing**:

1. `provenance.source_case_id` → `candidate_records.json` `target_interpretation`
   (positive output structured fields) and `rejected_candidates` (negative
   rejection context, e.g., oracle trace "percept_match").
2. `review_id` → `review-records.json` masked expected fields
   (`_masked_expected_novelty`, `_masked_expected_derivability`,
   `_masked_expected_abstention`) for sub-label semantics.
3. Record-level boolean evaluations (`proposition_correctness`,
   `evidence_relevance`, `evidence_sufficiency`, `grounding_correctness`,
   `semantic_novelty`, `symbolic_derivability`, `decision_relevance`,
   `decision_usefulness`, `contradiction_handling`, `abstention_correctness`)
   gate which generator-authored fields may be promoted to gold.

Rule: a generator-authored field is promoted to gold only if the corresponding
human boolean is true. Fields with no human gate are not promoted.

### 3.4 Human-override signal

Human reviewers agreed with the generator on 262/264 records. Two records are
documented overrides: masked `SEMANTIC_NOVEL`/`SHOULD_ABSTAIN` candidates that
humans labeled **HARD_NEGATIVE**. These are supervision signals, not errors:
they teach that novelty alone is insufficient for proposal. They must be
preserved verbatim.

### 3.5 Semantic-novelty vs decision-relevance

Both are human labels. `decision_relevance`/`decision_usefulness` are recorded
per record and must be usable as upweighting/positive-only flags without being
model inputs. A proposal that is novel but decision-irrelevant is still
negative; the corpus realizes this via the DECISION_IRRELEVANT class (64
records).

---

## 4. Inference-time input schema and no-leakage rules

### 4.1 Input fields (the only things the model may see at inference)

```text
percept            : str      # raw observation text
concepts           : [{id, label, definition, type}]
beliefs            : [{id, proposition}]
rules              : [{id, antecedent, consequent}]
grounding snapshot : {concept_ids, belief_ids, rule_ids, evidence_ids}
task               : str      # decision question the interpretation feeds
```

### 4.2 Explicitly excluded from model inputs (supervision-only, never runtime features)

```text
final_status                     # the label itself
candidate_proposition            # the gold target, never an input
target_interpretation            # the gold target's structured fields
SEMANTIC_NOVEL / NON_DERIVABLE / SHOULD_PROPOSE / DECISION_RELEVANT
GOLD_POSITIVE / GOLD_ABSTAIN / HARD_NEGATIVE
capability IDs (CAP-01 .. CAP-13)      # generator bookkeeping
difficulty tier / source_type          # generator bookkeeping
generator_id / template_id / review ids
masked expected labels (_masked_*)
```

These values may exist in training metadata (for audit, diagnostics, or label
construction) but **must never appear in any model input**. The input schema is
a hard contract: constructing a training example from a record requires removing
every excluded field, and any code path that feeds an excluded field is a
training-data defect, not a modeling choice.

### 4.3 Output-target construction (no corpus alteration)

The training target is built at data-load time from the immutable corpus plus
the read-only joins of §3.3. The gold corpus files are never rewritten. The
builder must:
- produce the input from §4.1 only;
- produce the output from §3.2, gated by §3.3;
- record the provenance chain (review_id → source_case_id → corpus row) so every
  training example traces to its human decision;
- re-verify on every build that no §4.2 field leaks into the input side.

---

## 5. Train/eval separation

### 5.1 Training data

Only the frozen gold corpus (plus read-only joins). No additional synthetic
data is generated until this document is approved (Step 10). No template,
proposition, concept, grounding id, or derived variant from the frozen
evaluation instruments ever enters training.

### 5.2 Evaluation-only instruments (forever)

- Frozen **51-case benchmark** (`bm://*`, 6A.1; AST-percepts) — untouched.
- Frozen **15-case semantic probe** (`sp1://*`, 6A.2) — untouched.
- Newly discovered regression cases → appended to evaluation, never to training.

These are the **model-selection instruments** (§7). No decision to promote or
reject a candidate model is made from within-corpus metrics alone.

### 5.3 Within-corpus split discipline (diagnostic only)

When a train/dev split of the corpus is needed for training monitoring, it must
be **grouped by seed family**: a base case and every perturb variant sharing its
`seed_case_id` stay in the same split. Rationale: the corpus contains near-duplicate
records by construction (1396 percept-similar pairs; exact-duplicate propositions
across perturb variants), and a naive random split would let the dev set be
solved by surface memory. Within-corpus metrics are diagnostic; selection uses
the frozen instruments.

### 5.4 Determinism

Every training/eval build is versioned (schema version, corpus hash, seed,
generator id, provenance). Greedy decoding with fixed seed; byte-stable replay
required on the probe (the 6A.2 rationale-ordering defect is a regression test).

---

## 6. Success metrics

Defined before training, measured on the frozen instruments (§5.2) unless noted.
Per-capability scores must be reported per probe group (A–E, E0–E6), not only as
macro averages.

| metric | definition | target (v0) |
|---|---|---|
| semantic novelty precision | proposed propositions that are genuinely novel / all proposed | > 0 with ≥ 1 human-verified E5/E6 per group |
| derivability discrimination | DERIVABLE-vs-NON_DERIVABLE accuracy on probe cases | high; no derivable proposal admitted |
| evidence relevance accuracy | proposed supporting_evidence_ids match relevant ids (Group E) | ≥ threshold (design G) |
| grounding validity | 100% of accepted-proposal ids resolve against snapshot | 100% |
| distractor rejection | proposal cites no distractor when distractors present | ≥ threshold |
| abstention precision/recall | abstain when it should, propose when it should | high; abstain never penalized as malformed |
| hard-negative rejection | non-proposable candidates produce no proposal | ≈ 100% |
| decision-relevance precision | proposals that answer the decision task | headline metric (design G) |
| useful-proposal rate | see below | see below |
| false-positive rates | proposal emitted for DERIVABLE / epistemically-premature input | ≈ 0 |
| E0 (malformed) rate | structurally invalid output / all outputs | ≈ 0 |
| latency | per-case CPU seconds (reference: 51.6 s/case is unacceptable) | ≤ a few s CPU; ≤ ~0.5 s GPU |
| determinism | byte-stable replay on the probe | required |

**USEFUL_PROPOSAL_RATE** is the composite headline: a proposal is *useful* iff
it is simultaneously
1. **genuinely novel** (SEMANTIC_NOVEL, not repeat/echo/paraphrase),
2. **non-derivable** (symbolic runtime cannot derive it),
3. **evidence-supported** (supporting ids resolve and are relevant; no distractor
   reliance),
4. **decision-relevant** (answers the task it feeds),
5. **grounding-valid** (every referenced id resolves).

`USEFUL_PROPOSAL_RATE = useful proposals / total opportunities`. It is the metric
the model-selection gate optimizes, because it encodes P1–P4 + P7 jointly and
collapses to 0 for a chatty but unsupported model.

---

## 7. Model-selection gate

No base model is chosen in this document. The gate below must be satisfied and
scored before any download, fine-tune, LoRA, or quantization.

### 7.1 Scoring criteria (from design F, made decisive)

| criterion | requirement |
|---|---|
| parameter budget | ≤ 1B; smallest tier clearing the eval gates wins (100M–300M / 300M–600M / 600M–1B) |
| context length | ≥ 2k tokens (full input schema must fit) |
| tokenizer | small vocab; must round-trip entity/grounding ids losslessly |
| structured output | strict schema + deterministic adapter; E0 ≈ 0 |
| latency | ≤ a few s/case CPU; ≤ ~0.5 s/case GPU |
| device | CPU-primary; single small GPU optional; no datacenter |
| quantization | int8/int4 without regression on the eval gates |
| training method | full fine-tune vs LoRA/QLoRA vs from-scratch — decided after objective + instruments approved |
| architecture | decoder-only baseline; evidence-reranker only if single-model gate fails on Group E |
| determinism | greedy, fixed seed, byte-stable replay |
| licensing | reproducible, license-compatible with THEO distribution |
| inference independence | `theo_core` never imports `torch`/`transformers`; provider boundary intact |

### 7.2 Reference experiment — Qwen3-0.6B

Qwen3-0.6B is a **reference experiment for the evaluation harness**, not an
automatic training base. The 6A.2 measurements (51.6 s/case, 53.3% structured
failure, capability misses) are the baseline every candidate must beat. Qwen3-0.6B
is scored like any other candidate under §7.1 and §6; it gets no default slot.

### 7.3 Deliverables before any selection

1. This objective approved (this document).
2. Corpus freeze + audit report accepted (Steps 1 & 7).
3. A candidate matrix scoring ≥2 base models against §7.1.
4. An eval-harness dry run on the frozen instruments with a non-trained
   reference, to prove the harness is deterministic and reproducible.

---

## 8. STOP gate

Phase 6C.1 completes when:
- the corpus is frozen and audited (Steps 1 & 7, companion report); and
- this training objective is reviewed and accepted (Steps 2, 3, 4, 5, 6, 8, 9).

Until review accepts both deliverables: **no model downloads, no training
initialization, no LoRA adapters, no quantization runs, and no additional
synthetic data generation.**
