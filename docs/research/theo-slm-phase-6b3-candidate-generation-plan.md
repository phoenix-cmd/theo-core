# THEO SLM v0 — Phase 6B.3 Candidate Generation Plan

**Status:** DRAFT FOR REVIEW — documentation only. No execution, no mutation.
**Date:** 2026-08-25
**Authoritative specification:** `theo-core/docs/research/theo-slm-dataset-v0.md`
(the approved Phase 6B.1 dataset spec, §3/§5/§10/§11/§12/§13/§14/§16/§17/§18).
**Phase context:** 6B.2 is **complete/successful** — it answered "what does the reviewed pool
contain?": **226 valid GOLD records (29 POS / 66 ABSTAIN / 131 NEG), 0 unresolved, with a
positive-volume/coverage shortfall.** 6B.3 is the forward step: controlled candidate
generation to fill the *actual* missing scientific/data requirements. It does **not** revisit or
re-relax 6B.2.

> **No-execution notice:** This is a plan. Generating records, mutating `adjudication.json` or
> the gold corpus, model selection/download, training, 6C, or 6D all require *separate, explicit
> authorization* beyond this plan's approval. Until then: planning only.

---

## 1. Objective & scope

Generate **new candidate records** to close the adjudicated-GOLD gaps identified in the 6B.2
coverage/gap report, with strict uniqueness, coverage, provenance, and leakage/shortcut
safeguards. Output is a **separate new candidate pool** — the existing 226 GOLD records and
`adjudication.json` are never modified.

**In scope:** generation of raw candidates + their provenance/manifest; a uniqueness oracle;
near-duplicate seed-family grouping metadata for future splitting.
**Out of scope (separate authorized steps):** human dual-review and 6B.3 adjudication of the
new candidates; GOLD freeze; model selection/download; 6C/6D.

---

## 2. Current state (from 6B.2, preserved unchanged)

| Status | Count |
|---|---|
| GOLD_POSITIVE | 29 |
| GOLD_ABSTAIN | 66 |
| HARD_NEGATIVE | 131 |
| **GOLD total** | **226** |
| REJECT (exact-dup, valid) | 38 |
| NEEDS_REVIEW | 0 |

Approved target range (spec §12/§17): **350–500 total GOLD**; **GOLD_POSITIVE 140–180**;
**GOLD_ABSTAIN 40–60** (already met at 66); **HARD_NEGATIVE 180–240**.

---

## 3. Generation targets (adjudicated-GOLD based, not raw-candidate based)

The **stopping criterion (§9)** is expressed in *final adjudicated GOLD* counts, never raw
candidate count. The raw generation budget below is a planning estimate that includes an
attrition buffer; generation continues in batches until §9 is met.

| Component | Current adjudicated | Target (adjudicated) | Net needed | Raw-gen budget (with attrition buffer) |
|---|---|---|---|---|
| GOLD_POSITIVE | 29 | 140–180 | **+111 to +151** | generate **~200–240** raw positive candidates |
| GOLD_ABSTAIN | 66 | 40–60 (met) | 0 (monitor) | generate **0–20** if coverage requires |
| HARD_NEGATIVE | 131 | 180–240 | **+49 to +109** | generate **~90–130** raw negative candidates |
| **Total GOLD** | 226 | **350–500** | **+124 to +274** | total raw-gen **~290–390** |

**Priority order:** (1) GOLD_POSITIVE volume first; (2) HARD_NEGATIVE to reach ≥180; (3)
GOLD_ABSTAIN only if a capability/domain gap appears.

---

## 4. Uniqueness requirement (no re-creation of rejected duplicates)

New candidates **must not** recreate the 38 rejected exact-duplicate propositions nor any
existing 226 GOLD proposition. Mechanism:

- **Uniqueness oracle:** before acceptance, normalize each candidate's
  `target_interpretation.proposition` (lowercase, whitespace-collapsed) and compare against:
  (a) the 226 existing GOLD propositions, and (b) the 38 rejected propositions (from
  `dedup_ledger.jsonl` + `adjudication.json` REJECT entries). Any match → candidate rejected
  at generation time (not admitted to the pool).
- **Intra-batch uniqueness:** the oracle also checks the growing new-pool set, preventing
  exact duplicates *within* 6B.3 generation.
- **Semantic novelty:** each new positive must satisfy the six SN criteria (spec §1.3 /
  semantic-probe-v1-spec §5) and be `NON_DERIVABLE` (INV-05/06) — i.e., genuinely new
  interpretation content, not a paraphrase variant of an existing proposition.
- This directly answers the 6B.2 lesson: the 38 rejections came from perturb-family content
  reuse; 6B.3 must produce *new* propositions, not re-emissions.

---

## 5. Capability & domain coverage

- New candidates must span **all 13 capability families (CAP-01…CAP-13)** and **all 8 domains**
  (medical, weather, physics, finance, biology, engineering, household, conflict) for
  GOLD_POSITIVE in particular (positives were concentrated; spread them).
- Prioritize under-covered capability/domain cells for positives using the 6B.2 coverage map.
- Negative/trap families (REPEAT, PARAPHRASE, UNSUPPORTED, EPISTEMICALLY_PREMATURE) generated
  deliberately to exercise each capability's tempting-wrong-answer surface.

---

## 6. Attrition accounting

Expected attrition channels and planning buffers:
- **Exact-duplicate rejection:** ~0 by construction (§4 oracle).
- **Schema-invariant failure (INV-01…09):** low (6B.2 pool had 0 failures); buffer 5–10%.
- **Reviewer disagreement / NEEDS-REVIEW:** new diverse content may raise this above the 6B.2
  κ=1.0; plan 15–30% for positives, 10–20% for negatives.
- **Near-duplicate-too-aggressive:** some near-dup (Jaccard≥0.8) pairs may be judged
  undesirable; handled by grouping (§8), not rejection.
- **Net:** raw-gen budgets in §3 already embed ~1.6–2.0× overshoot on positives to absorb
  attrition and still land GOLD_POSITIVE ≥140.

---

## 7. Preservation of existing 226 GOLD & non-mutation

- `theo_slm_v0_review/adjudication.json` and the `theo_slm_v0_gold/` corpus are **read-only
  inputs** to 6B.3 planning; they are **not modified**.
- New candidates are written to a **new pool** (e.g., `theo_slm_v0_candidates_6b3/
  candidate_records.json`), versioned separately. No merge into GOLD until a later, authorized
  6B.3 adjudication + freeze.
- The 38 REJECT dispositions are **valid and preserved**; they are not restored.

---

## 8. Near-duplicate handling for future train/eval splitting

The 6B.2 report found **1011 near-duplicate percept pairs (Jaccard ≥ 0.8)** among GOLD. These
are preserved as high-value discriminative supervision but must never be split across
train/eval. Define:

- **Group key:** `seed_case_id` for perturb-family records (base case + its `pert/var_*`),
  else the authored-case family prefix of `case_id`. All records sharing a group key are
  co-located in one split.
- **Explicit pair manifest:** emit `split_group_manifest.json` mapping every GOLD record
  (existing 226 + future new) to its `group_key`, and additionally force-group every one of
  the 1011 near-dup pairs into the same `group_key`.
- **Split rule (training spec §5.3):** grouped-by-seed-family splits only — a group is
  assigned wholly to train or to eval, never partitioned. This prevents near-duplicate leakage
  that would inflate eval metrics.
- New 6B.3 candidates receive `group_key` at generation time so they integrate with the same
  manifest.

---

## 9. Stopping criterion (adjudicated-GOLD based)

Generation (and its separate adjudication step) **stops** when **all** hold:
1. **Total GOLD ∈ [350, 500]** (approved target range).
2. **GOLD_POSITIVE ∈ [140, 180]** (the primary gap).
3. **Coverage:** every capability family CAP-01…CAP-13 and all 8 domains represented in GOLD,
   with GOLD_POSITIVE present in each.
4. **Balance:** positives ≈ 40% of GOLD, negatives+abstain ≈ 60% (spec §13).
5. **Quality:** all INV-01…09 pass; 0 frozen-evaluation leaks; reviewer agreement κ at target.

Raw-candidate count is **not** a stopping metric — only final adjudicated GOLD counts and
coverage trigger stop. If attrition is higher than modeled, generation continues (with
re-budgeting) until §9 is met; if lower, it stops earlier.

---

## 10. Provenance, deterministic seeds, versioning, leakage/shortcut safeguards

- **Versioning:** new pool `ds-v0.4-candidates` (semver `gen-v0.4.0`); `dataset-manifest.json`
  with full checksum lineage (spec §14). Existing `ds-v0.3-gold`/adjudication unchanged.
- **Deterministic seeds:** a fixed `SEED_DATA` (e.g., 43, consistent with 6B.2 repaired pool)
  drives all generation/ordering; each record carries `provenance.random_seed`,
  `generator_id`, `generator_version`, `template_id`, `seed_case_id`, `generation_timestamp`,
  and `human_review_status = "UNREVIEWED"` at creation.
- **Reproducibility:** generator is re-runnable to reproduce the exact pool from the seed + spec.
- **Leakage safeguards (spec §11):** automated n-gram + ID checks ensure no overlap with the
  frozen 51-case benchmark (6A.1) / 15-case probe (6A.2); these instruments remain
  evaluation-only forever.
- **Shortcut safeguards (spec §13):** evidence count, sentence length, concept type, and
  metadata (capability/tier/relation/confidence) are **never model inputs**; surface-shortcut
  correlation checks are run and reported (not auto-repaired).
- **Uniqueness safeguard:** §4 oracle prevents exact-duplicate propositions vs existing GOLD
  and the 38 rejected.
- **No Qwen-as-labels (6B design §D1):** Qwen outputs remain negative examples / format-failure
  regressions only.
- **cos(H1) exclusion:** irrelevant to dataset generation (human/symbolic only); the 6B design
  §H/B1/G2 exclusion governs 6C/6D and is not implicated here.

---

## 11. Governance & STOP

Execution of this plan (actual generation) requires a **separate, explicit human
authorization** distinct from 6B.2 approval. Until then: no record generation, no mutation of
`adjudication.json`/gold corpus, no model selection/download, no training, no 6C/6D. New
candidates, once generated, enter their own dual-review + adjudication (reusing the approved
6B.2 procedure) before any merge into GOLD.
