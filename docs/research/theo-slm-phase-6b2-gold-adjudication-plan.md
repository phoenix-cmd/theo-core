# THEO SLM v0 — Phase 6B.2 GOLD Adjudication Plan

**Status:** DRAFT FOR REVIEW — documentation only. No execution, no mutation.
**Date:** 2026-08-25
**Authoritative specification:** `theo-core/docs/research/theo-slm-dataset-v0.md`
(the approved Phase 6B.1 dataset spec, §4/§5/§10/§11/§13/§16/§17/§18) together with the
approved 6B design `theo-core/docs/research/theo-slm-design-v0.md` (§A–§H).
**Governance anchor:** the 6B review packet (`theo-slm-phase-6b-review-packet.md`) approval
covers *design* only; it explicitly does **not** authorize model selection, model download,
dataset mutation, training, 6C, or 6D. This plan inherits that boundary.

> **Inspection-only notice:** Every fact in §1 was obtained by *reading* artifacts. No
> candidate dataset, review manifest, gold corpus, or `adjudication.json` was modified during
> planning. Execution (populating `adjudication.json`, freezing GOLD) requires a separate,
> explicit human authorization beyond the 6B design approval.

---

## 0. Scope and relationship to existing artifacts

This plan governs **6B.2 GOLD adjudication** — converting the dual-human-reviewed candidate
pool into a formally adjudicated, eventually-immutable GOLD corpus. It follows the approved
spec exactly; it does not invent new criteria.

**Reconciliation note (transparency):** the repository already contains:
- a `theo_slm_v0_gold/` corpus of 264 records (67 GOLD_POSITIVE / 66 GOLD_ABSTAIN /
  131 HARD_NEGATIVE) with a `corpus-manifest.json` carrying mandated hashes (gold-freeze
  draft reports SHA-256 `6377dd60…`); and
- a `theo-slm-dataset-v0-gold-freeze.md` draft describing a Phase 6C.1 freeze from
  "264 final human-review decisions (κ = 1.0000, 0 disagreements)."

However, the **canonical `theo_slm_v0_review/adjudication.json` is currently `[]` (empty)**,
and the gold-freeze document is itself a DRAFT. Therefore this 6B.2 plan treats the existing
materials as *prior evidence / audit trail*, **not** as the authoritative, human-signed GOLD
decision. The immutable GOLD point (§10) is reached only after the procedure below populates
`adjudication.json` and receives explicit human sign-off.

---

## 1. Candidate-record inventory and current counts (INSPECTION ONLY)

| Artifact | Path | Count | Role |
|---|---|---|---|
| Source candidate pool (approved) | `theo-data/datasets/theo_slm_v0_repaired/candidate_records.json` | **264** | Authoritative input to adjudication (`ds-v0.2-repaired`) |
| Deduplicated pool | `theo-data/datasets/theo_slm_v0_deduplicated/candidate_records.json` | 264 | Prior dedup output (not the adjudication source) |
| Original candidate pool | `theo-data/datasets/theo_slm_v0_candidates/candidate_records.json` | 264 | Upstream source of the repaired pool |
| Dual-review manifest | `theo-data/datasets/theo_slm_v0_review/review-manifest.json` | **264** | Per-record `reviewer_1` / `reviewer_2` / `adjudication` fields |
| Canonical adjudication artifact | `theo-data/datasets/theo_slm_v0_review/adjudication.json` | **0 (empty `[]`)** | **To be populated by this plan** |
| Pre-existing frozen gold corpus | `theo-data/datasets/theo_slm_v0_gold/theo_slm_v0_gold.jsonl` | 264 (67 / 66 / 131) | Prior draft freeze; reconciled, not authoritative until re-signed |
| Reviewer decision copies | `theo_slm_v0_gold/reviewer-1-decisions.json`, `reviewer-2-decisions.json` | 264 each | Review trail |

**Scale note:** the approved spec targets 350–500 training records (§12/§17). The available
approved pool is 264 records. This plan adjudicates the 264 available records; reaching the
target range requires *additional generation*, which is a separate, authorized activity and is
out of scope for 6B.2 adjudication.

---

## 2. Exact human-review workflow

1. **Inputs frozen:** the 264 records in `theo_slm_v0_repaired/candidate_records.json` and
   their `reviewer_1` / `reviewer_2` decisions in `review-manifest.json` are read-only.
2. **Per-record independent review (already performed; verified here):** two distinct human
   reviewers (`rev_01`, `rev_02`) each assigned, per the spec §10.1 matrix:
   - `novelty_label` (REPEAT | PARAPHRASE | TAXONOMY_ECHO | RULE_ECHO | UNSUPPORTED |
     EPISTEMICALLY_PREMATURE | SEMANTIC_NOVEL | ABSTAIN)
   - `derivability_label` (DERIVABLE | NON_DERIVABLE)
   - `decision_relevance` (DECISION_RELEVANT | DECISION_IRRELEVANT)
   - `abstention_label` (SHOULD_ABSTAIN | SHOULD_PROPOSE)
   - a per-record recommendation (GOLD / REJECT / NEEDS-REVIEW) + free-text `review_notes`.
3. **Agreement computation:** for every record, compare the two reviewers on all four labels;
   compute corpus-wide Cohen's κ (target κ = 1.0000 for the GOLD set, replicating the
   prior κ = 1.0000 expectation).
4. **Adjudication decision:** apply §3 rules to assign each record a `final_status`
   (GOLD / REJECT / NEEDS-REVIEW) and write it into the per-record `adjudication` field of
   `review-manifest.json` **and** into the canonical `adjudication.json`.
5. **Escalation:** any NEEDS-REVIEW record is routed to a tie-break/third-reviewer step;
   it remains NEEDS-REVIEW until resolved and is excluded from GOLD.
6. **No auto-repair:** records are never silently edited. A rejected/unusable record is
   excluded from GOLD; any later fix is a documented exception on a new corpus revision.

---

## 3. Dispositions: GOLD / REJECT / NEEDS-REVIEW

A record receives **GOLD** only if it satisfies **all** of:
- the spec GOLD criteria (§10.1 / line 90):
  - *Gold Positive:* `SEMANTIC_NOVEL` + `DECISION_RELEVANT` + **100% agreement across 2
    distinct human reviewers**;
  - *Gold Abstain:* `SHOULD_ABSTAIN` confirmed by full human review (≥1 reviewer; INV-08
    requires 2 distinct reviewers if marked GOLD);
  - *Hard Negative:* REPEAT/PARAPHRASE/UNSUPPORTED/EPISTEMICALLY_PREMATURE with oracle check
    + sampled review;
- schema invariants INV-01…INV-09 all pass;
- frozen-source provenance present (generator_id, template_id, seed_case_id, timestamp).

A record receives **REJECT** if:
- it fails a schema invariant irrecoverably (e.g., positive target on a derivable/non-novel
  record — INV-02/INV-04/INV-05/INV-06), or
- reviewers agree the record is unusable (malformed beyond scope, contamination, or label
  leakage), or
- it violates grounding bounds (INV-07) with no resolvable IDs.

A record receives **NEEDS-REVIEW** if:
- reviewer disagreement (<100% agreement) on any of the four labels, or
- `reviewer_1_id == reviewer_2_id` (violates INV-08) for a would-be GOLD record, or
- a near-duplicate/leakage check requires a human judgment call.

---

## 4. Adjudication criteria for `SemanticInterpretation`

`target_interpretation` is the supervised SFT target and **MUST be null for NEGATIVE and
ABSTAIN cases** (schema §4; INV-02 / INV-04). For GOLD positives it is adjudicated as:

- **Form:** single declarative sentence, ≤120 chars (schema §4).
- **Grounding (INV-07):** every id in `supporting_evidence_ids` and `referenced_concept_ids`
  must resolve within `grounding_snapshot`; all 67 prior positives resolved 0 issues — this
  is re-asserted, not assumed.
- **Relation enum:** `semantic_relation` ∈ {causal, explanation, paraphrase, contrast,
  category, state, other}.
- **Non-derivability:** the proposition must be genuinely `NON_DERIVABLE` (not a verbatim
  percept match, belief echo, taxonomy edge, or rule conclusion) — INV-05/INV-06.
- **Decision relevance:** semantically novel, grounded in evidence, and directly answers or
  advances the decision task (spec line 89).
- **Isolation (INV-03):** `target_interpretation.proposition` must never equal any
  `rejected_candidates[*].proposition`.
- **Confidence:** 0.0–1.0; treated as a format field, not a learned quantity (per
  gold-freeze finding 6).

---

## 5. Handling of malformed / schema-invalid records

- **Pre-check:** run the invariant validator `scripts/dataset_generator/quality_gates.py`
  (`check_invariants`, INV-01…INV-09 — the actual implementation; the spec §18 nominally names
  `scripts/validate_dataset_schema.py`, which does not exist) on every one of the 264 records
  **before** adjudication finalization.
- **Malformed → REJECT** (or NEEDS-REVIEW if the defect is ambiguous and human-judgment
  dependent). No repair occurs inside 6B.2; repair belongs to a separate 6B.3-style
  activity and would require its own authorization.
- **INV-08 failure** (missing or identical reviewer ids on a GOLD candidate) → NEEDS-REVIEW;
  the record cannot become GOLD until two distinct reviewer ids are present and agreeing.
- Every rejection records the failing invariant id / reason in `review_notes`.

---

## 6. Duplicate and near-duplicate handling

- **Exact-duplicate propositions** (normalized): if two GOLD records share a normalized
  proposition, retain one as GOLD and flag the duplicate REJECT (or NEEDS-REVIEW per §3),
  recorded in a dedup ledger.
- **Near-duplicate percepts** (Jaccard ≥ 0.8), including *opposite-status pairs on identical
  inputs* (e.g., a GOLD_POSITIVE and a HARD_NEGATIVE sharing a percept): these are deliberate,
  high-value discriminative supervision (gold-freeze finding 1). They are **preserved**, but
  their `seed_case_id` / template grouping is recorded so that any future train/validation
  split uses **grouped-by-seed-family splits** (training spec §5.3) and never naively
  separates a pair across train/eval.
- **Duplicate evidence structures** (≥10 groups observed previously): logged, not removed.

---

## 7. Leakage / shortcut checks

- **Frozen-evaluation leakage (spec §11):** automated n-gram + ID checks assert **0** leaked
  case ids, **0** leaked grounding ids, and **0** percept/candidate overlap with the
  frozen 51-case benchmark (6A.1) and 15-case probe (6A.2). These instruments are
  evaluation-only, forever.
- **Shortcut probes (spec §13):** verify evidence count, sentence length, concept type,
  capability/tier/relation/confidence metadata are **not** predictable of status in a way
  that substitutes for reasoning. These fields are **supervision-only / metadata** and must
  never be model inputs (training spec no-leakage input schema). Any record whose label is
  readable from surface vocabulary alone is **kept but documented** for human decision — no
  silent removal.
- All checks are recorded in the adjudication provenance; findings are reported, never
  auto-repaired.

---

## 8. Provenance and frozen-source requirements

- Every GOLD record traces to its source `case_id` in `theo_slm_v0_repaired/
  candidate_records.json`, with `generator_id`, `template_id`, `seed_case_id`,
  `generation_timestamp` preserved.
- `adjudication.json` records per record: `final_status`, `reviewer_1_id`, `reviewer_2_id`,
  agreement (per-label + κ), decision rule applied, `review_timestamp`, `review_notes`.
- Source files (`candidate_records.json`, `review-manifest.json`) are **read-only inputs**;
  their SHA-256 is recorded in `corpus-manifest.json`. They are never modified by adjudication.
- **No Qwen outputs become labels** (6B design §D1): Qwen is a source of negative examples
  and format-failure regressions only, never positive labels.

---

## 9. Reviewer agreement requirements

- **Gold Positive:** 2 **distinct** reviewers, **100% agreement** on all four labels → GOLD.
- **Corpus-wide κ:** computed over all 264 records; the GOLD set must reach κ = 1.0000
  (replicating the prior κ = 1.0000 expectation). If κ < 1.0, **every** disagreement is
  resolved (tie-break/third reviewer) before any record is marked GOLD.
- **INV-08:** `human_review_status == "GOLD"` ⇒ `reviewer_1_id` and `reviewer_2_id` present
  and `reviewer_1_id != reviewer_2_id`.
- Disagreement on any label ⇒ NEEDS-REVIEW (escalate); not GOLD.

---

## 10. Exact point at which the corpus becomes immutable GOLD

The corpus is **mutable** throughout adjudication. It becomes **immutable GOLD only at the
GOLD FREEZE step** (the Phase 6C.1 Step-1 analog), which occurs **after all** of:

1. `adjudication.json` is populated from the per-record `adjudication` fields and **human-
   signed**;
2. all schema invariants (INV-01…INV-09) pass on the accepted records;
3. leakage / shortcut / duplicate checks are recorded with findings reported;
4. the reviewer-agreement gate (κ = 1.0000 on GOLD) is satisfied;
5. a human explicitly authorizes the freeze.

At that point `theo_slm_v0_gold/theo_slm_v0_gold.jsonl` is materialized **verbatim** from the
accepted review records (never modified), the six mandated SHA-256 hashes are computed, and
the corpus version is tagged (e.g., `ds-v0.3-gold`). **After freeze: no record may be added,
removed, or replaced** except as a documented exception on a new corpus revision.

---

## 11. Gate separating GOLD adjudication from model selection and 6C

- **6B.2 GOLD adjudication** completes → corpus frozen GOLD → **then** model selection
  (separate authorization) → **then** 6C build (separate authorization) → 6D evaluation.
- This plan grants **no** authority to: select or download a base model; mutate the candidate
  dataset; train; or begin 6C/6D. Those each require their own explicit authorization,
  consistent with the 6B review packet's phase-distinction boundary.
- Dataset adjudication is **human/symbolic only**; it does not engage any model. The
  `cos(H1)` exclusion (6B design §H / B1 / G2) governs 6C/6D training and evaluation signals
  and is not implicated by GOLD adjudication.

---

## 12. Execution authorization & STOP

This document is a plan. Executing it — populating `adjudication.json`, running the invariant
validator, or freezing GOLD — requires a separate, explicit human authorization. **Until
then: inspection only; no mutation of any dataset, manifest, or gold artifact.**
