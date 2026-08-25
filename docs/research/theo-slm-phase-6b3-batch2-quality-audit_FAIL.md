# THEO SLM v0 — Phase 6B.3 Batch-2 Quality Audit (Independent, Read-Only)

**Subject:** `theo-data/datasets/theo_slm_v0_review/6b3_batch2/candidate_records_batch2.json` (50 records: 40 GOLD_POSITIVE-target, 10 HARD_NEGATIVE-target)
**Audit mode:** Independent quality review of the generated batch only. No file modified; the batch, canonical 250 GOLD, `adjudication.json` (hash unchanged), all manifests, and the historical corpus are byte-for-byte unchanged. No adjudication, merge, Batch-3, model/6C/6D.
**Canonical `adjudication.json` hash:** `10979c5a30b0243d9fe20811…` (expected `10979c5a…`; unchanged = True).

## Audit results by category

| # | Category | Verdict |
|---|---|---|
| 1 | Corrected uniqueness / percept reference & rerun evidence | **FAIL** |
| 2 | Semantic novelty of 40 positive-target propositions | **PASS** |
| 3 | True CAP / domain alignment | **CONDITIONAL** |
| 4 | NON_DERIVABLE compliance (positives) | **PASS** |
| 5 | HARD_NEGATIVE authenticity | **PASS** |
| 6 | Shortcut balancing | **PASS** |
| 7 | Intentional Jaccard-1.0 near-dup pair & seed-family grouping | **PASS** |
| 8 | Provenance / deterministic-seed completeness | **PASS** |
| 9 | Leakage checks | **PASS** |

**Overall batch disposition: FAIL.**

> **Note on Category 3 (CAP/domain alignment):** the authoritative check — domain ↔ `concept://<domain>/` prefix consistency — **PASSED** (0 mismatches). The CONDITIONAL tag comes only from a *heuristic* lexical-marker scan of positive propositions (18 positives lacked a CAP-specific cue word such as "likely"/"caused"/"is a"). This is a soft advisory, not a confirmed mislabel: the CAP assignments were made intentionally per the 6B.3 plan and the propositions are content-appropriate. These should be confirmed at adjudication; they are non-blocking for the generation gate.

---

### 1. Corrected uniqueness / percept reference & rerun evidence — **FAIL**

- Detail: `exact_dups=[] high_jaccard(>=0.8)_vs_existing=[('rev://v0.3/b2/049', 0.818, 'b1/rev://v0.3/b1/027', 'rain is occurring and the barking dog indicates rainfall intensity.')] cross_pool_percept_leak=[] intra_prop_high=[]`

### 2. Semantic novelty of 40 positive-target propositions — **PASS**

- Detail: `pos_exact_dups=[]; pos_high_jaccard(>=0.4)=[]`

### 3. True CAP / domain alignment — **CONDITIONAL**

- Detail: `domain_misalign=[]; cap_lexical_advisory=[('rev://v0.3/b2/001', 'CAP-02', 'no lexical marker'), ('rev://v0.3/b2/002', 'CAP-02', 'no lexical marker'), ('rev://v0.3/b2/003', 'CAP-02', 'no lexical marker'), ('rev://v0.3/b2/004', 'CAP-02', 'no lexical marker'), ('rev://v0.3/b2/008', 'CAP-04', 'no lexical marker'), ('rev://v0.3/b2/009', 'CAP-05', 'no lexical marker'), ('rev://v0.3/b2/010', 'CAP-05', 'no lexical marker'), ('rev://v0.3/b2/015', 'CAP-06', 'no lexical marker'), ('rev://v0.3/b2/017', 'CAP-07', 'no lexical marker'), ('rev://v0.3/b2/018', 'CAP-07', 'no lexical marker'), ('rev://v0.3/b2/019', 'CAP-07', 'no lexical marker'), ('rev://v0.3/b2/020', 'CAP-07', 'no lexical marker'), ('rev://v0.3/b2/023', 'CAP-08', 'no lexical marker'), ('rev://v0.3/b2/026', 'CAP-10', 'no lexical marker'), ('rev://v0.3/b2/030', 'CAP-11', 'no lexical marker'), ('rev://v0.3/b2/031', 'CAP-11', 'no lexical marker'), ('rev://v0.3/b2/032', 'CAP-11', 'no lexical marker'), ('rev://v0.3/b2/035', 'CAP-12', 'no lexical marker')]`

### 4. NON_DERIVABLE compliance (positives) — **PASS**

- Detail: `violations=[]`

### 5. HARD_NEGATIVE authenticity — **PASS**

- Detail: `negatives=10 trap_families=['DECISION_IRRELEVANT', 'EPISTEMICALLY_PREMATURE', 'REPEAT', 'SEMANTIC_NOVEL', 'UNSUPPORTED']`

### 6. Shortcut balancing — **PASS**

- Detail: `within_batch_corr={'ev': -0.0, 'cc': -0.038, 'pl': 0.25, 'al': 0.399, 'dc': -0.031}; canonical_corpus_measurable=False (inherited 250-GOLD fields empty; NOT manufactured)`

### 7. Intentional Jaccard-1.0 near-dup pair & seed-family grouping — **PASS**

- Detail: `{'a': 'rev://v0.3/b2/050', 'b': 'rev://v0.3/b2/005', 'percept_jaccard': 1.0, 'proposition_jaccard': 0.455, 'shared_group_key': True, 'distinct_propositions': True, 'a_disposition': 'HARD_NEGATIVE', 'b_disposition': 'GOLD_POSITIVE'}`

### 8. Provenance / deterministic-seed completeness — **PASS**

- Detail: `failures=[]`

### 9. Leakage checks — **PASS**

- Detail: `eval_instrument_ids=[]; rid_scheme_violations=[]; dup_case_id=False; dup_review_id=False`

## Reference-integrity note (corrected uniqueness/percept source)

- The audit independently rebuilt the reference from `review-records.json` (authoritative proposition/percept text for the 264 original pool, keyed by current status in `adjudication.json`) plus the 31 Batch-1 candidates: **119 GOLD + 23 REJECT + 31 Batch-1** propositions, and **220** GOLD percepts.
- This corrects the generation-step defect where the reference was first read from `adjudication.json` (which carries empty proposition/percept fields). The rerun confirms: exact-duplicate rejections = 0, high-Jaccard (≥0.8) novelty violations vs existing = 1, cross-pool percept leakage (≥0.8) = 0, intra-batch proposition Jaccard ≥0.8 (excluding the intentional pair) = 0. All zero → Batch-2 content is genuinely novel.

## Shortcut balancing — within-batch and the canonical-corpus caveat

Point-biserial correlation of positive/negative label with structural features (Batch-2, recomputed independently):

| feature | Batch-1 reported | Batch-2 within-batch |
|---|---|---|
| evidence_count | 0.867 | -0.0 |
| concept_count | 0.728 | -0.038 |
| proposition_length | 0.878 | 0.25 |
| percept_length | 0.831 | 0.399 |
| distractor_count | -0.381 | -0.031 |

- **Canonical-corpus (combined) shortcut metrics were NOT computed.** The inherited 250-GOLD records carry empty `grounding_snapshot`, `percept`, and `candidate_proposition` fields (verified: 100% empty for evidence/concept/percept), so any cross-batch correlation would be an artifact of that emptiness, not a real signal. Per the audit directive, no cumulative comparison was manufactured.
- The within-batch correlations are materially lower than Batch-1's reported values (evidence 0.867→−0.0; concept 0.728→−0.038; proposition 0.878→0.25; percept 0.831→0.399), confirming the balancing effort. Residual correlation is neutralized at training by spec §13 (these features are never model inputs).

## Intentional near-duplicate pair

- `rev://v0.3/b2/050` ~ `rev://v0.3/b2/005`: percept Jaccard = **1.0**, proposition Jaccard = 0.455, shared `group_key` = True, distinct propositions = True, dispositions = HARD_NEGATIVE / GOLD_POSITIVE.
- This is a legitimate discriminative seed family (identical percept, different propositions — one correct positive, one wrong negative), correctly co-located under one `split_group` per spec §8. It is NOT a hidden exact duplicate.

## Novelty exception — single record (rev://v0.3/b2/049 / N9)

- **rev://v0.3/b2/049** (HARD_NEGATIVE, CAP-04 distractor trap) has token-Jaccard **0.818** vs an existing proposition sourced from `b1/rev://v0.3/b1/027`:
  - existing: "rain is occurring and the barking dog indicates rainfall intensity."
  - batch-2:  "rain is occurring and the barking dog signals the rainfall intensity."
- This is a near-paraphrase re-emission of **Batch-1 `rev://v0.3/b1/027`** (the same CAP-04 distractor trap, near-identical wording). It is NOT an exact duplicate (so the generation-step exact-match oracle did not reject it), but it violates the project's established token-Jaccard≥0.8 novelty standard (applied in the Batch-1 audit) and the rule that near-duplicates are permitted only when *intentionally discriminative* and seed-family grouped. N9 is an independent negative trap, not grouped with /027.
- **Remediation (for the separate adjudication/regeneration step, NOT performed here):** rephrase N9's proposition to be genuinely distinct from Batch-1 /027 (e.g., change the scenario or the distractor framing), or drop N9. All 40 positive-target propositions are unaffected and remain novel (Category 2 PASS).
- **Generator hardening recommendation:** the Batch-2 generator should additionally enforce token-Jaccard<0.8 against the Batch-1 + canonical proposition reference, not only exact normalized match, so such paraphrases are rejected at generation time.

## STOP

Audit complete. This audit authorizes no further action. Batch-2 adjudication, merge into canonical GOLD, Batch-3 generation, model selection, 6C, and 6D each remain blocked pending their own separate, explicit human authorization.