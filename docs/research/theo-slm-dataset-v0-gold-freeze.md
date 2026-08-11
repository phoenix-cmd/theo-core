# THEO SLM Gold Corpus Freeze & Audit — Phase 6C.1 (Steps 1 & 7)

**Status:** DRAFT FOR REVIEW — corpus frozen from final human decisions; audit
complete with residual findings reported (no auto-repair). Training objective is
specified in the companion `theo-slm-training-v0.md`.
**Date:** 2026-08-11
**Scope:** Phase 6C.1, Steps 1 (freeze) and 7 (corpus-level audit).

---

## 1. Scope

- **Step 1 — freeze:** materialize the immutable gold corpus from the 264 final
  human-review decisions (Phase 6B.4, κ = 1.0000, 0 disagreements) and only from
  them. Do not modify `ds-v0.2-repaired`. Preserve provenance to every reviewed
  candidate and review artifact. Compute the six mandated SHA-256 hashes.
- **Step 7 — audit:** adversarial surface-text classifiers, shortcut probes,
  duplicate/near-duplicate/evidence-structure checks, capability/domain
  coverage, and frozen-evaluation-leakage checks against the final corpus.
  Report findings for human decision; never auto-repair.

## 2. Inputs and hashes

| artifact | path | SHA-256 |
|---|---|---|
| source candidate dataset (untouched) | `theo-data/datasets/theo_slm_v0_repaired/candidate_records.json` | `c2c04b36baae09b2d14efc7bc3e978cad6033109334c84714214177e41e207b2` |
| review manifest | `theo-data/datasets/theo_slm_v0_review/review-manifest.json` | `f07d30b9affe53869ad4e59c0b055e02e34e6ede5e58bba8daee655dc1f1b41b` |
| reviewer-1 decisions | `theo-data/datasets/theo_slm_v0_review/reviewer-1-decisions.json` | `f0a8961eaa1661b84ecc55bde14f320843f46a40c8a726d4a1500bce59359ff6` |
| reviewer-2 decisions | `theo-data/datasets/theo_slm_v0_review/reviewer-2-decisions.json` | `dbc6fca0a01c81243ecfdb3ee33de5d61328770d0e3d57a954778be18d56f886` |
| adjudication artifact | `theo-data/datasets/theo_slm_v0_review/adjudication.json` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| **frozen gold corpus** | `theo-data/datasets/theo_slm_v0_gold/theo_slm_v0_gold.jsonl` | **`6377dd6016fbfa9d2fdede682f24b876834e9555092c2cf1065fc441ed6ec13e`** |

Split hashes (in `corpus-manifest.json`): GOLD_POSITIVE `f80eb300…`,
GOLD_ABSTAIN `c0e8cfc0…`, HARD_NEGATIVE `4501bc81…`. Corpus version
`ds-v0.3-gold`, schema `gold-schema-v0.1`.

## 3. Freeze process (Step 1)

Every corpus record is derived **verbatim** from the accepted review records:

- `final_status` copied from `adjudication.final_status`;
- all reviewed fields byte-identical to both the review manifest and
  `review-records.json` (asserted by the freeze script);
- provenance chain verified with 0 missing, 0 duplicate, 0 order mismatches
  (`review-order.json` rank ↔ `review-records.json` `_masked_original_case_id`
  ↔ `candidate_records.json` `case_id`);
- generator expected labels (`_masked_*`), capability IDs, difficulty tiers,
  and source types are **not** propagated into the corpus (label-leakage rule);
- evaluation-only instruments excluded (`bm://*`, `sp1://*`); no corpus record
  derives from them.

Artifacts produced in `theo-data/datasets/theo_slm_v0_gold/`:

| artifact | contents |
|---|---|
| `theo_slm_v0_gold.jsonl` | 264 records, one compact-JSON object per line |
| `gold_positive.jsonl` / `gold_abstain.jsonl` / `hard_negative.jsonl` | status splits (67 / 66 / 131) |
| `reviewer-1-decisions.json`, `reviewer-2-decisions.json` | per-record reviewer labels |
| `provenance-index.json` | review_id → source_case_id / generator / template |
| `corpus-manifest.json` | all mandated hashes + integrity assertions |
| `corpus-audit.json` | full audit output (this report's source) |

## 4. Corpus composition

- **Status:** 67 GOLD_POSITIVE / 66 GOLD_ABSTAIN / 131 HARD_NEGATIVE
  (25.4% / 25.0% / 49.6%).
- **Domains (PASS):** medical, weather, physics, finance, biology, engineering,
  household, conflict — every domain contains all three statuses.
- **Case families:** 173 perturb variants (`td://v0/pert/var_*`) + 91 authored
  cases; perturb variants dominate each status (43 GOLD_ABSTAIN / 44
  GOLD_POSITIVE / 86 HARD_NEGATIVE). Details in `case_family_x_status`.
- **Human vs generator:** reviewers agreed with the generator's expected
  category on 262/264 records. Two documented overrides: generator
  `SEMANTIC_NOVEL` + `SHOULD_ABSTAIN` candidates that humans labeled
  **HARD_NEGATIVE** — preserved verbatim as supervision signal (novelty alone is
  not proposal-worthiness).
- **Masked-expected distribution (supervision-only, not model inputs):**
  - GOLD_POSITIVE (67): SEMANTIC_NOVEL 67; NON_DERIVABLE 67; SHOULD_PROPOSE 67.
  - GOLD_ABSTAIN (66): UNSUPPORTED 64, EPISTEMICALLY_PREMATURE 2; all
    NON_DERIVABLE; all SHOULD_ABSTAIN.
  - HARD_NEGATIVE (131): DECISION_IRRELEVANT 64, REPEAT 65, SEMANTIC_NOVEL 2
    (the overrides); DERIVABLE 15 / NON_DERIVABLE 116; all SHOULD_ABSTAIN.
- **Grounding:** 0 beliefs / 0 rules in all 264 records; 3–4 concepts; 64
  records carry 1 distractor, 200 carry none. All 67 positive
  `target_interpretation` referenced ids resolve within their snapshot
  (0 issues).

## 5. Audit methodology (Step 7)

Adversarial surface classifiers: TF-IDF (max 250 features, 1–2 grams)
LogisticRegression, 5-fold StratifiedKFold (seed `20260811`), one-vs-rest per
status, over 8 feature sets (proposition only, task only, percept only, concept
names only, relation, content words only, surface combined, metadata only), plus
length-only and confidence probes, near-duplicate detection (percept Jaccard ≥
0.8), exact-duplicate proposition detection (normalized), duplicate evidence
structures, capability/domain/tier balance, and frozen-instrument leakage checks
(case ids, grounding ids, percept/candidate overlap vs the 51-case benchmark and
15-case probe).

## 6. Findings

**Overall verdict: REVIEW** — every automated PASS check passed; every residual
finding is informational and was **not** auto-repaired.

### 6.1 PASS

- **review_status_integrity** — κ recomputed 1.0000; reviewer-1 = reviewer-2 on
  all 264; `final_status` matches adjudication; split counts match the accepted
  67/66/131; human-vs-generator 262/2 as above.
- **frozen_evaluation_leakage** — 0 leaked case ids, 0 leaked grounding ids, 0
  percept/candidate overlaps with the 51-case benchmark or 15-case probe.
- **grounding** — 0 unresolvable target references (67/67 positive targets ok).
- **domain_coverage** — all 8 domains have all three statuses.
- **split_balance** — 67/66/131 with supervision targets (positive + abstain) =
  133 and hard negatives = 131.
- **proposition length** — length-only classifier shows *negative* excess
  (cannot predict status); not a shortcut.

### 6.2 REVIEW — findings for human decision

1. **Duplicate propositions / template reuse.** Exact normalized duplicates
   across records, e.g. `points to strep bacterial infection` ×3 (GOLD_POSITIVE),
   `points to severe strep bacterial infection crisis` ×3 (GOLD_ABSTAIN),
   `points to ambient storm clouds` ×6 (HARD_NEGATIVE). Root cause: the debiased
   `pert/var_*` family re-emits base-case content with status-specific lexical
   framing; the base case and its variants are literally the same percept and
   same proposition (percept Jaccard = 1.0 pairs across statuses exist, e.g.
   rev 001 GOLD_POSITIVE pairs with rev 018 HARD_NEGATIVE and rev 120/121
   GOLD_ABSTAIN). **Implication:** these are *opposite-status pairs on identical
   inputs* — strong discriminative supervision — but the corpus must never be
   split naively (see §7.3 of the training spec: grouped-by-seed-family splits).
2. **Capability coverage imbalance.** Every capability has GOLD_POSITIVE, but
   GOLD_ABSTAIN exists only in CAP-09 (+66) and HARD_NEGATIVE only in CAP-01
   (+65) and CAP-03 (+66). CAP-02/04–08/10–13 have no negative or abstain
   examples. This reflects the generator's per-capability perturbation plan, not
   a labeling decision; it limits in-corpus supervision for those capabilities.
3. **Surface-text shortcut vocabulary.** The content-words classifier reaches
   balanced-accuracy excess +0.47 (HARD_NEGATIVE) and the top discriminating
   tokens are exactly the status-specific framings: `crisis`/`severe` → abstain
   (e.g. `infection crisis`, `failure crisis`), `ambient`/`index`/`ruptured`/
   `shelled eggs` → negative (instrumental/ambient readings), hazard verbs →
   positive. A linear model can read the status off the wording.
4. **Metadata shortcut.** The metadata-only classifier (capability, difficulty
   tier, source type, counts) reaches excess +0.48. These fields are generator
   bookkeeping, **not** present in the corpus and excluded from model inputs by
   the no-leakage contract (§4 of the training spec); they must never be added
   back as features.
5. **Relation is definitional.** `semantic_relation = UNKNOWN` for all 66
   abstains + 131 negatives (no proposed interpretation exists); positives use
   cause (18), state_observation (18), association (14), explanation (14),
   indication (3). This is a label property, not a leak; it is excluded from
   model inputs and documented in `relation_shortcuts`.
6. **Confidence is degenerate.** Only GOLD_POSITIVE has confidence; range
   0.85–0.88, stdev 0.005 (mean 0.879). The corpus cannot supervise a rich
   confidence scale; confidence should be treated as a format field, not a
   learned quantity (noted in the training spec §6 caveats).
7. **INFO — near-duplicates (1396 pairs at Jaccard ≥ 0.8)** and **INFO —
   duplicate evidence structures (10 groups)**. Expected by the perturb-family
   design; no action, but they drive the grouped-split rule.

## 7. Training implications (mapped to `theo-slm-training-v0.md`)

- §4 no-leakage input schema is the direct response to findings 3–5: capability,
  tier, relation, confidence-target, and masked labels must never be inputs.
- §5.3 grouped-by-seed-family splits are the direct response to findings 1 and 7.
- §3.4 human-override signal preserves the 2 overrides as supervision.
- The frozen instruments (never trained on) are the only guard against the
  surface-vocabulary shortcut: a model may exploit it on the corpus, but the
  selection gates (§6/§7 of the training spec) are evaluated on instruments with
  different wording.

## 8. No auto-repair policy

Per the 6C.1 plan, zero records were added, removed, or replaced. All residual
findings above are reported for human decision. If review decides any record is
unusable (e.g., a status-specific framing is unacceptable), the fix is a
documented exception on the next corpus revision — not a silent edit of this
frozen version.

## 9. Reproduction

Script: `theo-core/scripts/dataset_generator/run_gold_corpus_freeze.py`
(stdlib + numpy + scikit-learn only; no `uv` needed). Run from the repo root:

```
python theo-core/scripts/dataset_generator/run_gold_corpus_freeze.py
```

It rebuilds the corpus, recomputes all hashes, and regenerates
`corpus-audit.json`. The corpus SHA-256 must remain
`6377dd6016fbfa9d2fdede682f24b876834e9555092c2cf1065fc441ed6ec13e`.
