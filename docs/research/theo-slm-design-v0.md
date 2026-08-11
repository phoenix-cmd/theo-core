# THEO SLM Design v0 — Specification for the THEO Semantic SLM

**Status:** DRAFT FOR REVIEW — design only; no training, no base-model selection,
no dataset, no downloads until this document is reviewed.
**Date:** 2026-08-11
**Origin:** Phase 6A.2 result (**NEGATIVE / DIAGNOSTIC** —
`docs/research/reference-slm/semantic-probe-v1/phase-6a2-report.md`).
**Scope:** Phase 6B — THEO SLM Design.

---

## 0. Governing principle

> **Build the smallest model that provides the semantic capability THEO's
> symbolic architecture demonstrably lacks.**

The objective is **not** "a small version of Qwen." 6A.2 showed the reference
SLM generates only partially usable, poorly constrained output at ~51.6 s/case,
and fails exactly the capabilities the probe measures. THEO SLM v0 is specified
to supply those capabilities and nothing else. Everything the symbolic runtime
already does reliably — inference, verification, decisions, commitment, rule
firing — is explicitly excluded from the model's remit. Where the architecture
conflicts with "small," the architecture is subordinate to the capability
requirement, not the reverse.

The model is an **intelligence provider**, not THEO itself.

---

## A. What is THEO SLM supposed to learn?

6A.2 identified, empirically, the capabilities the symbolic runtime lacks and
Qwen did not reliably provide. Each is stated as a measurable behavior so it can
be trained, tested, and gated.

| # | capability | meaning | probe evidence / target pattern |
|---|---|---|---|
| 1 | **semantic interpretation** | Produce a proposition that goes beyond literal percept text and is supported by supplied evidence | b/002: symptoms → "power outage" |
| 2 | **evidence relevance** | Cite the evidence that actually supports the interpretation, and ignore distractors | Group E failed (0.0); target: cite rain evidence, not `sky`/`wet` |
| 3 | **abductive hypothesis generation** | Infer an unobserved cause/explanation that accounts for observed facts | b/002 (outage); conclusion must not already exist in symbolic knowledge |
| 4 | **semantic paraphrase normalization** | Recognize that differently-worded propositions denote the same content | a/001: "shattered after hitting the floor" ≡ "broken by the impact on the floor" → must be classified derive/repeat, not novel |
| 5 | **contradiction interpretation** | Detect contradictory stored beliefs and interpret them (e.g., light on ∧ room dark → light broken) | c/001, c/002: model must resolve, not echo the contradiction |
| 6 | **candidate generation from indirect evidence** | Bridge gaps where the conclusion is not stated, only hinted at | Group B/D style cases |
| 7 | **grounding-aware proposal generation** | Reference only ids present in the grounding snapshot; reject/avoid unknowns | 7/7 Qwen proposals grounded, but the model must be *reliable* at this, not accidental |

**Design consequence:** capabilities 2 and 7 are as much *output-format and
attention* properties as language properties. That argues for a task-specific
structured-output model rather than a general chat model.

---

## B. What must THEO SLM NOT learn?

The model must never be trained, prompted, or wired to do any of the following.
These are mechanically enforced by the provider boundary (snapshot DTOs in,
`SemanticInterpretation`/`HypothesisProposal` out), and must also be absent from
the training data so they are never *learned* as behaviors:

- **final decision making** — the runtime decides, using its own confidence machinery
- **belief commitment** — the model never asserts or modifies beliefs
- **direct Thought creation** — no Thought DTOs emitted by the model
- **direct DecisionRecord creation** — no DecisionRecord DTOs emitted by the model
- **autonomous rule creation** — the model never proposes new rules
- **modification of symbolic state** — no writes to beliefs, rules, or memory
- **bypassing grounding** — every output id must resolve against the supplied grounding snapshot
- **replacing symbolic inference** — deterministic inference/verification stays in the runtime
- **replacing deterministic verification** — the provider pipeline's schema and grounding validation is non-negotiable

**Boundary contract (unchanged from ADR-0028 / 6A.1):** model proposes;
runtime disposes. `theo_core` never imports `torch`/`transformers`.

---

## C. Define the THEO SLM output

Minimal structured output, defined by THEO (not inherited from Qwen's
free-form `{"hypotheses": [...]}` JSON, which produced the 53.3% structured-output
failure rate in 6A.2).

```text
SemanticInterpretation
    proposition: string                 # one declarative sentence, <= 120 chars
    supporting_evidence_ids: string[]   # grounding ids the interpretation rests on
    referenced_concept_ids: string[]    # grounding ids used in the proposition
    semantic_relation: string           # one of: causal | explanation | paraphrase |
                                        # contrast | category | state | other
    confidence: number                  # 0.0..1.0, model's own stated strength
```

Conversion (provider adapter, deterministic, in `theo-providers`):

```text
SemanticInterpretation
    → schema + grounding validation (reject → E0, never partial)
    → HypothesisProposal
        proposal_id = proposal://theoslm/<case>/<index>
        content = proposition
        referenced_ids = supporting_evidence_ids ∪ referenced_concept_ids
        rationale = semantic_relation
```

Rules:

- The model **never directly emits symbolic runtime objects** (`HypothesisProposal`,
  beliefs, rules, decisions). `HypothesisProposal` is constructed by the adapter
  after validation.
- Malformed output → rejected as a whole (E0). No partial recovery, no
  `grounded=False` fallback. This is what makes the 53.3% failure rate an
  unacceptable baseline and a first-class training/eval metric for v0.
- Empty output is legal and better than garbage: a model that says "no
  supported interpretation" scores as a clean abstention, not as a malformed E0.

---

## D. Define the training target

**Primary labels are human/symbolic ground truth — not Qwen outputs.** Qwen's
outputs (frozen in 6A.2) are a source of *negative examples* and format-failure
regressions, never of positive labels.

**Seed positive example — b/002 (preserve as a gold standard):**

```text
observations:   "The lights went out. The microwave clock was blinking.
                The fridge hummed to life."
concepts:       power, electricity, light, outage
beliefs/rules:  (none)
decision task:  what explains the observations?
target:         power outage          # conclusion NOT present in symbolic knowledge
```

The defining property of a positive example: **the conclusion is not derivable
from the supplied symbolic knowledge** — the model must supply it from the
evidence plus interpretation.

**Construct positive and negative example families around the derive-vs-repeat
distinction** (the entire point of the 6A.2 funnel):

| family | positive example (target class) | negative example (must be labeled repeat/echo) |
|---|---|---|
| derive vs repeat | symptoms → unobserved cause (b/002 pattern) | paraphrase of the percept (a/001 pattern: "broken by the impact on the floor") |
| evidence relevance | interpretation citing the relevant ids | same proposition citing only distractor ids (Group E pattern) |
| distractor rejection | correct evidence, distractors present | cites a distractor as support |
| multi-fact interpretation | several observations → one unifying cause | concatenation of observations (no new content) |
| contradiction interpretation | light on ∧ room dark → light is broken | "the room is dark and the light is on" (c/002 pattern — conjunction of beliefs) |
| taxonomy understanding | correct is-a/category use | re-asserting a taxonomy fact as if novel (D pattern) |
| abductive reasoning | best-explanation cause among several plausible | unsupported plausible conclusion with no cited evidence |
| grounding-awareness | all referenced ids in the snapshot | any unknown/non-URI id |

Labeling protocol: for every candidate interpretation, apply the 6A.2 test —
**could the symbolic runtime already derive this from its possessions?** If yes
→ repeat/echo (negative). If no but supported by evidence → novel (positive).
If the novel interpretation answers the task → decision-relevant (positive,
higher value). Semantic novelty and decision relevance are human labels on top
of deterministic measurement; the 6A.2 evaluator and its label schema carry over.

---

## E. Define the dataset structure

One record per case. Format designed to later feed both decoder-only
sequence training and encoder/reranker evidence selection:

```text
percept                    : str          # raw observation text
beliefs                    : [{id, proposition}]
concepts                   : [{id, label, definition, type}]
rules                      : [{id, antecedent, consequent}]
grounding snapshot         : {concept_ids, belief_ids, rule_ids, evidence_ids}
task                       : str          # decision question the interpretation feeds
candidate interpretations  : [SemanticInterpretation ...]
positive/negative labels   : bool         # derive-vs-repeat verdict
supporting evidence        : id[]         # the ids that actually support the label
semantic novelty label     : enum         # novel | paraphrase | repeat | taxonomy-echo | unsupported
decision relevance label   : bool         # does it answer the task?
```

Splits and governance:

- **Training data** — newly constructed synthetic/templated cases (b/002-pattern
  seeds, the D-families above, plus distractor-perturbed variants). Trained on
  *nothing* from the two frozen instruments.
- **Evaluation data (held out, never trained on):**
  - the frozen 51-case benchmark (6A.1) — untouched;
  - the frozen 15-case semantic probe (6A.2) — untouched;
  - newly discovered regression cases added over time (e.g., a/002-style
    `...`-malformed outputs, Group E distractor variants), stored separately
    from training data.
- Versioned dataset manifest (schema version, generator seed, provenance of each
  label, human-review status), mirroring the probe's freeze discipline.

---

## F. Model architecture requirements

No model is chosen here. First the requirements the selection must satisfy.

| requirement | target | rationale |
|---|---|---|
| parameter budget | ≤ 1B (tiered: 100M–300M / 300M–600M / 600M–1B) | capability-first; smallest tier that clears the eval gates wins |
| context length | ≥ 2k tokens | percept + concepts + beliefs + rules + task + grounding snapshot must fit; Qwen ran at 2048 in 6A.1 |
| tokenizer | small vocab, efficient for short structured text; must round-trip entity ids losslessly | grounding ids are opaque tokens that must be emitted exactly |
| structured output | strict schema (C); deterministic adapter converts | the 53.3% E0 rate in 6A.2 is a format problem the adapter must make structurally impossible to regress silently |
| inference latency | ≤ a few seconds/case CPU; ≤ ~0.5 s/case GPU | 6A.2 reference = 51.6 s/case is unacceptable; runtime cycle budget dominates |
| target device | CPU (primary), single small GPU (optional) | THEO must work without any model; the model must not require a datacenter |
| quantization | int8/int4 capable without evaluation-gate regression | same eval gates apply post-quantization |
| training method | candidates: (a) full fine-tune from a small open base, (b) LoRA/QLoRA adapters, (c) from-scratch small LM | decided after dataset + eval gates are fixed |
| decoder-only? | baseline candidate: yes | simple, matches SLM role |
| small encoder/reranker for evidence selection? | candidate: separate tiny encoder trained on the evidence-relevance family | Group E failure suggests attention-level evidence selection may need explicit supervision; a reranker is evaluated against a single-model baseline, not assumed |
| one model for interpretation + proposal generation? | default: one model, one adapter | only split if the single-model gate fails on evidence relevance |

**Architecture comparison (to be scored, not decided here):** single
decoder-only model vs decoder + evidence-reranker vs shared-encoder/two-head.
Scoring criteria: eval-gate pass on the 15-case probe (novelty rate > 0 with
human-verified E5/E6 per group), Group E evidence-relevance ≥ threshold, E0 rate
≈ 0, grounded rate = 100%, latency, size, trainability within budget.

---

## G. THEO SLM evaluation

Future evaluation compares providers end-to-end on the same frozen instruments:

```text
v0.4.1 symbolic (baseline, no provider)
    ↓
heuristic proposals + heuristic calibration (Phase 4)
    ↓
Qwen reference (6A.1 / 6A.2 — frozen, archived)
    ↓
THEO SLM v0          ← this phase
    ↓
THEO SLM v1 ...
```

- Primary gates (from the frozen probe): semantic novelty rate > 0 with at
  least one **human-verified** E5/E6 per group; Group E evidence-relevance
  ≥ threshold; structured-output (E0) rate ≈ 0; 100% grounding on accepted
  proposals.
- Decision-relevant interpretation rate is the headline metric (6A.1 measured
  decision impact directly; the probe measures the interpretation capability
  that precedes it).
- Determinism: greedy, fixed seed; byte-stable replay required on the probe
  (the 6A.2 rationale-ordering defect is a regression test).
- The frozen 51-case benchmark and 15-case probe are evaluation-only, forever;
  newly discovered regression cases are appended to evaluation, never to training.

---

## Review gate (before any implementation)

This document is the 6B deliverable. **Stop here** until review decides:

1. approve the capability list (A) and the exclusion list (B);
2. approve the `SemanticInterpretation` schema and adapter conversion (C);
3. approve the training-target doctrine: human/symbolic labels only, b/002 as
   the gold-standard seed, derive-vs-repeat as the organizing dichotomy (D);
4. approve the dataset format and the train/eval separation (E);
5. score the architecture options against the requirements table (F);
6. approve the evaluation gates (G).

No training code, no base-model selection, no dataset creation, and no model
downloads precede this review.
