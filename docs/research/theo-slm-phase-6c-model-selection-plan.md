# THEO SLM v0 — Phase 6C Model-Selection Plan

**Document ID:** `docs/research/theo-slm-phase-6c-model-selection-plan.md`  
**Date:** 2026-08-25  
**Status:** PLANNING / READ-ONLY SELECTION ANALYSIS — **NOT a build authorization.**  
**Authoritative Training/Selection Corpus:** `theo-data/datasets/theo_slm_v0_review/adjudication.json`
(canonical SHA-256 `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4`; GOLD = 300
records: 84 `GOLD_POSITIVE` / 66 `GOLD_ABSTAIN` / 150 `HARD_NEGATIVE`; 44 `REJECT` excluded).

---

## 0. Authorization boundary & scope

| Out-of-scope (explicitly NOT done here) | Status in this document |
|---|---|
| Download model weights | **not performed** |
| Begin 6C build / training / fine-tuning | **not performed** |
| Execute 6C/6D build or evaluation | **not performed** |
| Modify the canonical dataset (`adjudication.json`) | **not performed** |
| Generate Batch 3 | **deferred** — only authorized later if a 6C/6D evaluation demonstrates a concrete *blocking* deficiency |

This document is the **6C model-selection plan**: it defines the selection criteria, scores the
approved architecture candidates against the 6B §F requirements using **behavioral / structured-output**
criteria only, and records the selection decision rule. It does **not** execute the selection. Stop
after this plan + the independent audit (`theo-slm-phase-6c-model-selection-audit.md`).

Governing constraints carried verbatim from the 6B review packet and design doc:
- §B / §H / Gate 2/4: **no `cos(H1)`-style subspace-alignment training or evaluation signals.**
- §G / Gate 4: **gradient-based compatibility probes excluded** unless a pre-registered sensitivity
  control passes.
- §D / Gate 3: **primary labels are human/symbolic ground truth; Qwen outputs are negatives/format
  regressions only.**
- v0.5 plan Constraint 4: **THEO's architecture SHALL NOT depend on any particular neural model.**

---

## 1. Authoritative corpus (frozen input to selection)

| Slice | Count | Role in selection/training |
|---|---|---|
| `GOLD_POSITIVE` | 84 | positive (derive/novel) examples |
| `GOLD_ABSTAIN` | 66 | grounded-abstention examples |
| `HARD_NEGATIVE` | 150 | negative (repeat/echo/distractor/unsupported) examples |
| `REJECT` | 44 | **excluded** — never train or select on these |
| **GOLD total** | **300** | authoritative training/selection corpus |
| Canonical hash | `97bc6bc6…` | verified MATCH; corpus is 100% UNTOUCHED |

Class balance (from the post-Batch-2 readiness review): positive fraction = **28%** vs the §13
~40% guideline. Per spec §12 this is a **TARGET RANGE shortfall, RECOMMENDED not blocking** for 6C.
Batch-3 generation is **deferred**; it is authorized only if a later 6C/6D evaluation surfaces a
*concrete blocking* deficiency (e.g., a capability with zero positive coverage that cannot be gated).

---

## 2. Selection criteria (the approved 6B §F set, re-stated)

Each criterion below is **behavioral / structured-output** and maps to a design-doc §F requirement
and/or a user-specified selection axis. No criterion relies on a hidden-state, gradient, or label-leak
observable.

| # | Criterion | Source | How it is measured (behavioral only) |
|---|---|---|---|
| C1 | single decoder-only SLM candidate | §F (baseline) | evaluated as architecture option A |
| C2 | decoder + evidence reranker candidate | §F (evidence-relevance family) | evaluated as architecture option B |
| C3 | shared encoder / alternative architecture candidate | §F (shared-encoder/two-head) | evaluated as architecture option C, only if within approved set |
| C4 | parameter size / memory footprint | §F (≤1B tiered; VRAM ≤2GB) | model-config inspection + memory profiler |
| C5 | semantic interpretation capability | §A.1 / §G | novelty rate E5 + decision-relevance E6 on frozen probe |
| C6 | grounding | §C / §G | 100% id round-trip + 100% grounding on accepted outputs |
| C7 | evidence relevance | §A.2 / §F (Group E) | Group E distractor-rejection rate on frozen probe |
| C8 | abstention behavior | §A.2 / design §C | abstention accuracy on `GOLD_ABSTAIN` / unsupported cases |
| C9 | semantic novelty | §A.3 / §G | E5 novelty rate (non-derivable, evidence-supported) |
| C10 | E0 / structured-output failure rate | §C / §G | proportion failing `SemanticInterpretation` schema/adapter |
| C11 | latency & deployment practicality | §F | ≤0.5 s/case GPU, ≤ few s/case CPU; CPU-primary |
| C12 | compatibility w/ frozen neural-symbolic interface & v0.5 constraints | ADR-0028 / v0.5 plan | baseline invariance, model independence, determinism/replay, firewall |

---

## 3. Candidate architecture set (all within the approved §F set)

> Per design §F, the architecture comparison is **scored, not pre-decided**. The three candidates
> below are exactly the §F-approved comparison set: *"single decoder-only model vs decoder +
> evidence-reranker vs shared-encoder/two-head."*

| ID | Architecture | §F basis | Notes / dependency on base model |
|---|---|---|---|
| **A** | Single decoder-only SLM (one model, one adapter) | §F baseline; "one model default" | Minimal; emits `SemanticInterpretation` directly |
| **B** | Decoder-only SLM **+ tiny evidence reranker** | §F (evidence-relevance family) | Reranker is a *separate* small encoder trained only on the evidence-relevance (Group E) family; evaluated against A, **not assumed** |
| **C** | Shared encoder / two-head (alt architecture) | §F (shared-encoder/two-head) | Encoder for evidence/relation encoding + decoder head; only if it stays within the ≤1B budget and v0.5 isolation |

All three assume a base model chosen from the ≤1B open-weight tier (e.g., the Qwen2.5-0.5B-Instruct
class previously identified in `model-selection-v0.md`); **the base-model choice is an implementation
detail owned by the design doc, never frozen into ADR-0028** (v0.5 Constraint 4).

---

## 4. Scoring matrix (projected — to be confirmed at 6C.3 build / 6D eval)

> Ratings are **pre-build projections** derived from §F reasoning and the 6A.1/6A.2 reference outcomes
> (architecture PASS, cognitive NULL; Group E evidence selection = 0/15; E0 = 53.3% baseline). They are
> **not measured results**. Measured values are filled only at 6C/6D, against the frozen instruments.

| Criterion | A: decoder-only | B: +reranker | C: shared-encoder | Projected basis |
|---|---|---|---|---|
| C4 footprint (≤1B / ≤2GB VRAM) | Strong | Moderate (adds reranker params) | Moderate | A smallest; B/C add a small encoder |
| C5 semantic interpretation (E5/E6) | Moderate (6A.2 weak) | Moderate–Strong (reranker aids grounding) | Moderate | 6A.2 showed interpretation is the hard axis |
| C6 grounding (100%) | Strong (adapter-enforced) | Strong | Strong | Deterministic adapter governs all |
| C7 evidence relevance (Group E) | **Weak** (6A.2 = 0/15) | **Strong** (explicit supervision) | Moderate | This is the deciding axis |
| C8 abstention | Moderate | Moderate | Moderate | Trainable from 66 abstain examples |
| C9 semantic novelty (E5) | Moderate | Moderate–Strong | Moderate | Coupled to C5/C7 |
| C10 E0 (≈0%) | Strong (schema-guided) | Strong | Strong | Adapter makes E0 structurally near-zero |
| C11 latency (CPU-primary) | Strong | Moderate (2-stage) | Moderate | A is single-pass |
| C12 v0.5 compatibility | Strong | Strong (reranker isolated) | Strong (if snapshot-only) | All via ADR-0028 hooks only |
| **Overall projection** | **Primary** | **Conditional fallback** | Candidate (lower priority) | see §5 |

---

## 5. Recommended selection & decision rule

1. **Primary candidate: A (single decoder-only SLM).** It is the §F default ("one model, one adapter …
   only split if the single-model gate fails on evidence relevance"), has the smallest footprint, the
   strongest E0/grounding/latency profile, and full v0.5 compatibility.
2. **Conditional fallback: B (decoder + reranker).** §F states the reranker is *evaluated against a
   single-model baseline, not assumed*. Because 6A.2 measured **Group E evidence relevance = 0/15** —
   THEO's empirically weakest axis — **C7 (evidence relevance) is the deciding gate**:
   - If 6C.3/6D shows architecture **A** clears the Group E distractor-rejection gate
     (≥80% per design §G), **A is selected**.
   - If **A** fails Group E while **B** clears it (with no regression on E0/grounding/novelty), **B is
     selected**. The reranker must remain a *separate, snapshot-only* provider component (no symbolic
     state writes).
3. **C (shared encoder)** is retained as a lower-priority candidate; it is selected only if A and B both
   fail a v0.5-isolation or budget gate, which is not projected.
4. **Base-model family** (≤1B open-weight decoder) is chosen in the design doc, not here; selection of
   the *architecture* (A/B/C) is independent of the specific weights, per v0.5 Constraint 4.

**Selection is finalized only at 6C.3/6D against measured behavioral gates — not in this planning doc.**

---

## 6. Prohibited selection signals — explicit exclusion verification

The selection uses **only** the behavioral/structured-output criteria in §2. The following are
**verified absent** from the scoring matrix, the decision rule, and any future 6C gate:

| Prohibited signal | Used in selection? | Binding authority | Verification |
|---|---|---|---|
| `cos(H1)` (hidden-state gradient cosine) | **NO** | design §B/§H, review-packet Gate 2/4; 6E.20.x `cos_H1(SENS)=0.322966<0.95` → INVALID | Not present in §2 criteria; excluded by §H.1 |
| Gradient-compatibility metrics | **NO** | design §B ("no hidden-state gradient-compatibility optimization"), §G | Not in §2; excluded by §G/§H |
| Hidden-state subspace alignment | **NO** | design §B/§H; review-packet Gate 1 note ("no aligned readout subspace") | Not in §2; capability list is observable behavior only |
| Qwen-generated labels as positive labels | **NO** | design §D/D1, review-packet Gate 3 | Qwen outputs are negatives/format-regressions only; GOLD = human/symbolic |

No candidate is ranked, scored, or rejected on any of the four prohibited signals.

---

## 7. Evaluation gates (behavioral only) — applied at 6C.3 / 6D

| Gate | Definition | GO threshold (from design §F/§G) |
|---|---|---|
| E0 format reliability | outputs failing `SemanticInterpretation` schema/adapter | ≤ 2.0% |
| Grounding validity | referenced concept/evidence ids valid in snapshot | 100% |
| Semantic novelty (E5) | non-derivable, evidence-supported novel interpretations | ≥ 40% |
| Decision relevance (E6) | novel interpretation directly answers the task | ≥ 30% |
| Distractor rejection (Group E) | distractors ignored on Group E cases | ≥ 80% |
| Abstention accuracy | incomplete/premature → model abstains | ≥ 90% |
| Shortcut resistance | balanced-acc classifier on predictions | ≤ 40% (chance) |
| Latency | s/case | ≤0.5 GPU / few s CPU |
| v0.5 compatibility | baseline invariance, determinism, replay, firewall | all green |

**Reference instruments (evaluation-only, byte-locked, never trained on):** the frozen 51-case
benchmark (`v0.4.1` corpus, expanded in v0.5 Phase 2) and the frozen 15-case semantic probe (6A.2).
**Training corpus:** the 300-record GOLD above, using a **grouped-by-seed-family** train/dev split that
preserves the intentional `/005`∼`/050` near-duplicate group (split-leakage = 0).

---

## 8. v0.5 / ADR-0028 interface compatibility

- **Constraint 1 (baseline invariance):** the SLM provider is wired only through the five ADR-0028
  hooks; with no provider configured the runtime is byte-identical to v0.4.1 (35/35 PASS). The selected
  architecture must not alter symbolic cognition.
- **Constraint 3 (determinism / replay / grounding):** greedy, fixed-seed decoding; byte-stable replay
  on the probe; the 6A.2 rationale-ordering defect is a regression test; every output grounded against
  the snapshot boundary before influencing cognition.
- **Constraint 4 (model independence):** the architecture selection (A/B/C) and base-model family are
  recorded in the design doc / research notes, **never** written into ADR-0028. THEO "supports neural
  providers," it does not "uses \<model\>."
- **Architectural firewall:** `theo_core` MUST NOT import `theo_providers`; providers import
  `theo_core.models.ports` only. This permanent test gates any 6C build.

---

## 9. Residual data caveats (do not block selection; inform 6C/6D)

- **Positive-ratio shortfall (28% vs ~40%):** RECOMMENDED, non-blocking per §12. Monitored at 6C/6D;
  Batch-3 generation authorized **only** if a concrete blocking deficiency appears.
- **220 historical GOLD have empty `capability_family`/`domain`:** capability-stratified coverage can be
  measured only on the 80 Batch-1+Batch-2 records; behavioral gates rely on the fully-tagged frozen
  probe/benchmark, so selection is unaffected.
- **Within-batch structural correlations** (best-available probe: percept length r=0.61, proposition
  length r=0.50) are on features that are **never model inputs** (design §G) and are **not** selection
  signals — documented, not used.

---

## 10. STOP condition

This is a **planning / read-only** deliverable. No weights are downloaded, no training begins, the
dataset is unmodified, and 6C build/eval is not executed. Awaiting the independent audit
(`theo-slm-phase-6c-model-selection-audit.md`) and separate human authorization before any 6C.3 build
or 6D evaluation.

---

## 11. Human Approval Record (2026-08-25)

- **Status:** **APPROVED.**
- **Approved artifacts:** `theo-slm-phase-6c-model-selection-plan.md` (this document) **and** its
  independent audit `theo-slm-phase-6c-model-selection-audit.md`.
- **Preserved constraints (binding on all downstream 6C work):**
  1. Canonical dataset remains **immutable** at SHA-256
     `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4`.
  2. **Batch 3 remains deferred** — generated only if a later 6C/6D evaluation demonstrates a concrete
     *blocking* deficiency.
  3. **No `cos(H1)` / gradient-compatibility / hidden-state-subspace** criteria in selection or evaluation.
  4. **No Qwen-generated labels** as training positives (Qwen outputs = negatives / format-regressions only).
  5. Candidate architectures remain limited to the **approved §F set** (A: decoder-only, B: decoder+reranker,
     C: shared-encoder).
  6. **Projected §4 scores remain provisional** and must be replaced by measured 6C.3/6D results before any
     final architecture selection.
- **This approval does NOT authorize** weight download, training/fine-tuning, or 6D evaluation. Each
  requires separate explicit human sign-off (see the 6C execution/readiness package).
