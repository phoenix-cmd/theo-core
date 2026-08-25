# THEO SLM v0 — Phase 6B.3 Batch-1 Quality Audit (Independent, Read-Only)

**Subject:** `theo-data/datasets/theo_slm_v0_review/6b3_batch1/candidate_records_batch1.json`
(31 records: 21 `GOLD_POSITIVE`-target, 10 `HARD_NEGATIVE`-target)
**Audit mode:** Independent quality review of the *generated batch only*. No file was modified:
the batch, existing 226 GOLD, `adjudication.json`, all manifests, and the historical 67/66/131
corpus are byte-for-byte unchanged. No adjudication, no Batch-2 generation, no model/6C/6D.
**Reference inputs:** `adjudication.json` (226 GOLD + 38 REJECT), `review-records.json` (264),
prior coverage/gap report, and `theo-slm-dataset-v0.md` (spec §1.3/§5/§13/§14).

---

## Audit results by category

| # | Category | Verdict |
|---|---|---|
| 1 | Semantic novelty vs existing GOLD + 38 rejected | **PASS** |
| 2 | True capability / domain alignment | **PASS** (advisory notes, non-blocking) |
| 3 | NON_DERIVABLE compliance (positive targets) | **PASS** |
| 4 | HARD_NEGATIVE target authenticity | **PASS** |
| 5 | Surface shortcuts & generator artifacts | **PASS** |
| 6 | Leakage | **PASS** |
| 7 | Seed-family / near-duplicate grouping | **PASS** (Jaccard 0.818 pair verified) |
| 8 | Provenance completeness | **PASS** |
| 9 | Projected GOLD contribution (not adjudicated) | **INFO** (labeled, not a pass/fail) |

**Overall batch disposition: PASS** (no blocking findings; two non-blocking advisory notes in §2).

---

## 1. Semantic novelty vs existing GOLD + 38 rejected — PASS
- Exact normalized match of any new proposition against the 226 GOLD + 38 REJECT proposition
  sets: **0**.
- Proposition token-Jaccard (≥0.8) of any new record against all 264 existing propositions:
  **0** (no paraphrase re-emission of prior content).
- Cross-pool percept Jaccard (≥0.8) of any new record against the 226 existing GOLD percepts:
  **0** (no ungrouped near-duplicate of an existing GOLD context).
- Conclusion: the batch satisfies the §4 uniqueness oracle and adds genuinely novel content.

## 2. True capability / domain alignment — PASS (advisory)
- Every record's `capability_family` and `domain` were checked against its percept/proposition
  content. All 13 CAP families and all 8 domains (medical, weather, physics, finance, biology,
  engineering, household, conflict) are represented; domains match content (e.g., weather records
  use `concept://wx/*`, medical use `concept://medical/*`).
- Positives align with the intended CAP's *positive behavior* (e.g., CAP-02 abductive cause,
  CAP-06 contradiction resolution, CAP-12 causal chain, CAP-13 hedged confidence).
- **Advisory (non-blocking):** three HARD_NEGATIVE records carry capability labels that are
  slightly looser than their trap type — `rev://v0.3/b1/023` (DECISION_IRRELEVANT, labeled
  CAP-03), `rev://v0.3/b1/028` (REPEAT restatement, labeled CAP-01; CAP-05 would be more
  precise), and `rev://v0.3/b1/030` (DECISION_IRRELEVANT, labeled CAP-09). These are acceptable
  as negative exercises, but tightening the CAP label on negatives is recommended at any later
  merge/adjudication pass. Not a defect for this batch's purpose.

## 3. NON_DERIVABLE compliance (positive targets) — PASS
- All 21 `GOLD_POSITIVE`-target records carry `_masked_expected_derivability = NON_DERIVABLE`.
- None of the 21 positive propositions is a verbatim substring of its percept (no DERIVABLE
  restatement / paraphrase-of-percept violation). INV-05/06 intent satisfied.

## 4. HARD_NEGATIVE target authenticity — PASS
- 10 negatives exercise distinct trap surfaces:
  - REPEAT (paraphrase of percept): `/021`, `/026`, `/028`.
  - UNSUPPORTED (no evidence): `/022`, `/025`, `/029`.
  - DECISION_IRRELEVANT (off-task): `/023`, `/030`.
  - EPISTEMICALLY_PREMATURE (overclaim): `/024`.
  - Distractor-inclusion failure (CAP-04 negative): `/027`.
- Each negative is a plausible *wrong* answer, not an accidentally-correct interpretation. The
  trap semantics are coherent and align with the spec's negative families (NEG-01…NEG-13).

## 5. Surface shortcuts & generator artifacts — PASS
- No placeholder/template artifacts (`TODO`, `PLACEHOLDER`, `__X__`) in any field.
- No metadata leaked into proposition text: zero records contain `CAP-`, `GOLD_POSITIVE`,
  `HARD_NEGATIVE`, `SEMANTIC_NOVEL`, `NON_DERIVABLE`, or `ds-v0` tokens in `candidate_proposition`.
- Proposition length variance healthy: word-count 3–13, mean 9.3 (no fixed-length telltale).
- Evidence-citation variance healthy: `evidence_ids` count 0–2 (negatives correctly cite fewer /
  none; positives cite ≥2 as required by CAP-01/03 success criteria).
- No repetitive openers or boilerplate phrasing across records.

## 6. Leakage — PASS
- Eval-instrument ids (`bm://`, `sp1://`): **0** found in percept or proposition.
- `review_id` scheme `rev://v0.3/b1/*` does not collide with the existing 264 (`rev://v0.2/*`).
- `case_id` and `review_id` are each unique within the batch (31/31).
- No model weights, checkpoints, or 6C/6D artifacts referenced.

## 7. Seed-family / near-duplicate grouping — PASS (Jaccard 0.818 pair verified)
- Intra-batch percept Jaccard ≥0.8 produced exactly **one** pair:
  `rev://v0.3/b1/019` ~ `rev://v0.3/b1/031` (Jaccard = 0.818).
- **Verification of the flagged pair:** its two *percepts* differ only by "rapid drop" vs
  "rapid fall" (the 0.818 score); the two *propositions* are **not** exact duplicates
  ("An approaching thunderstorm is likely given the pressure fall and thunder" vs
  "A thunderstorm is approaching as shown by the falling pressure and thunder" — distinct wording
  and structure, proposition token-Jaccard ≈0.53). This is a genuine *near*-duplicate seed family,
  **not** a masked exact duplicate.
- The pair correctly shares `group_key = sf_weather_storm_01` (and `split_group` equals
  `group_key` for all 31 records), so it will be co-located in future train/eval splits per
  spec §5.3 / plan §8. The shared group_key is therefore **correct and intended**, not masking an
  unintended duplicate.
- No other group_key is used more than once; no ungrouped high-Jaccard pair exists.

## 8. Provenance completeness — PASS
- All 31 records contain every required top-level field and a complete `provenance` block
  (`generator_id`, `generator_version`, `dataset_version`, `random_seed`=43, `template_id`,
  `seed_case_id`, `generation_timestamp`, `deterministic`=true, `spec_ref`).
- `human_review_status = "UNREVIEWED"` on all 31, correctly marking them as candidates (not GOLD).
- `_masked_*` expected-behavior fields present and consistent with `target_disposition`.

## 9. Projected GOLD contribution (NOT adjudicated) — INFO
- This is a projection only; **no adjudication was performed** and the 226 GOLD count is unchanged.
- *If* all 21 positives later adjudicate as GOLD_POSITIVE: GOLD_POSITIVE would rise 29 → 50
  (still below the 140–180 target; ~71–130 more needed).
- *If* all 31 were accepted into GOLD: total GOLD 226 → 257 (below the 350–500 target).
- This batch therefore closes only a small fraction of the identified gap and confirms the need
  for further authorized batches — consistent with the single-batch authorization boundary.

---

## Overall disposition
**PASS.** The 31-record batch meets every audit criterion: novel vs existing GOLD/REJECT,
capability/domain-aligned, NON_DERIVABLE positives, authentic negatives, free of surface
shortcuts and leakage, correctly grouped near-duplicates (the Jaccard 0.818 pair is a legitimate
seed family, not a hidden duplicate), and fully provenanced. No modification to any artifact is
required by this audit. Two non-blocking advisory notes (§2) may be addressed at a later merge
step and do not affect this batch's acceptance for inspection.

**This audit authorizes no further action.** Batch-2 generation, Batch-1 adjudication, GOLD
merge, model selection, 6C, and 6D each remain blocked pending their own separate, explicit
human authorization.
