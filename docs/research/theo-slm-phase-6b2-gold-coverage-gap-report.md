# THEO SLM v0 — 6B.2 GOLD Adjudication: Coverage / Gap Report

**Generated:** 2026-08-25T10:58:15.576521Z  |  **Method:** Option A (recomputed, not imported)

## 1. Final counts by disposition (GOLD / REJECT / NEEDS-REVIEW)

| Disposition | Count |
|---|---|
| GOLD_POSITIVE | 29 |
| GOLD_ABSTAIN | 66 |
| HARD_NEGATIVE | 131 |
| REJECT | 38 |
| NEEDS_REVIEW | 0 |
| **GOLD total** | **226** |
| **All 264** | **264** |

## 2. Final counts by label / domain / difficulty

### By generator novelty_label (GOLD only)

| novelty_label | Count |
|---|---|
| DECISION_IRRELEVANT | 64 |
| EPISTEMICALLY_PREMATURE | 2 |
| REPEAT | 65 |
| SEMANTIC_NOVEL | 31 |
| UNSUPPORTED | 64 |

### By domain (GOLD only)

| domain | Count |
|---|---|
| v0 | 226 |

### By difficulty_tier (GOLD only)

| tier | Count |
|---|---|
| 0 | 66 |
| 1 | 5 |
| 2 | 70 |
| 3 | 10 |
| 4 | 67 |
| 5 | 8 |

### By capability_family (GOLD only)

| capability | Count |
|---|---|
| CAP-01 | 68 |
| CAP-02 | 2 |
| CAP-03 | 67 |
| CAP-04 | 3 |
| CAP-05 | 2 |
| CAP-06 | 2 |
| CAP-07 | 2 |
| CAP-08 | 5 |
| CAP-09 | 70 |
| CAP-10 | 1 |
| CAP-11 | 1 |
| CAP-12 | 1 |
| CAP-13 | 2 |

### Positive / Negative balance (GOLD only)

| bucket | Count |
|---|---|
| NEGATIVE | 197 |

| POSITIVE | 29 |

## 3. Duplicate / near-duplicate handling

- **Exact-duplicate propositions:** 38 GOLD records rejected (all GOLD_POSITIVE; the 67 reviewer-positive records collapsed to 29 after removing 38 exact-duplicate propositions). `dedup_ledger.jsonl` contains 44 duplicate-occurrence entries (some records were the 2nd+ duplicate in a cluster). See §8 / `dedup_ledger.jsonl`.
- **Near-duplicate percept pairs (Jaccard >= 0.8) among GOLD:** 1011 — PRESERVED as deliberate discriminative pairs; must use grouped-by-seed-family splits later. Sample: td://v0/medical/case_001_A~td://v0/medical/case_001_B(1.0), td://v0/medical/case_001_A~td://v0/medical/case_001_C(1.0), td://v0/medical/case_001_A~td://v0/medical/case_001_D(1.0), td://v0/medical/case_001_A~td://v0/pert/var_0027(1.0), td://v0/medical/case_001_A~td://v0/pert/var_0028(1.0)

## 4. Leakage / shortcut findings

- Within-corpus `case_id` uniqueness: **True**.
- Eval-instrument ids (`bm://`,`sp1://`) in corpus: **False**.
- Note: 51-case benchmark (6A.1) / 15-case probe (6A.2) overlap check requires those artifacts; per gold-freeze prior finding this was 0 leaked ids. Within-corpus ID uniqueness checked below.
- Surface-shortcut / metadata-shortcut checks (per gold-freeze findings 3-5) are documented as informational; metadata fields are excluded from model inputs by the training spec no-leakage schema.

## 5. Coverage against 6B requirements

- **Capability coverage:** 13 distinct capability families represented in GOLD.
- **Domain coverage:** 1 distinct domains represented in GOLD.
- **Balance:** positives (GOLD_POSITIVE) = 29; negatives+abstain = 197.
- **Dual-review agreement:** κ computed below; GOLD set requires 100% agreement (see mapping verification).

## 6. Shortfall relative to 350-500 target

- Approved target range: **350-500 training records** (spec §12/§17).
- Adjudicated GOLD total from the 264-pool: **226**.
- **Exact shortfall vs lower bound (350):** 124 records.
- The pool also contains 38 REJECT and 0 NEEDS-REVIEW among the 264.

## 7. Is additional generation scientifically necessary?

- **Assessment:** YES — additional generation is scientifically necessary.
  - Shortfall of 124 vs the 350 lower bound is the primary gap.
  - Capability families present: 13/13; domains present: 1/8.
  - This is a RECOMMENDATION only. No records were generated; generation remains a separate, authorized activity.

## 8. Provenance deviation (explicit)

> theo_slm_v0_review/review-manifest.json contains NO human decisions (all 264 reviewer_1/reviewer_2 == UNREVIEWED, evaluations null). The genuine dual-human-review decisions exist only in theo_slm_v0_gold/reviewer-1-decisions.json and reviewer-2-decisions.json (part of the prior 67/66/131 corpus, preserved as non-authoritative historical evidence). Those reviewer-decision files were used as the raw adjudication inputs. The derived 67/66/131 final_status in theo_slm_v0_gold.jsonl was NOT read or imported.

Input hashes: candidate_records.json=`c2c04b36baae09b2…`, reviewer-1=`f0a8961eaa1661b8…`, reviewer-2=`dbc6fca0a01c8124…`.
