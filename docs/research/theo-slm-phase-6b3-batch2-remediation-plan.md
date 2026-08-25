# THEO SLM v0 — Phase 6B.3 Batch-2 Remediation Plan

**Status:** Authorized remediation of the 6B.3 Batch-2 quality-audit failure.
**Subject failure:** `rev://v0.3/b2/049` (N9) — token-Jaccard **0.818** near-paraphrase of
Batch-1 `rev://v0.3/b1/027` ("...barking dog indicates rainfall intensity" vs
"...barking dog signals the rainfall intensity"). Not an exact duplicate, so the
generation-step exact-match oracle did not reject it, but it violates the
established token-Jaccard ≥ 0.80 novelty standard applied in the Batch-1 audit.

**Hard gate:** No adjudication, merge, Batch-3, model/6C/6D. This plan remediates
the batch only.

## 1. Oracle hardening (DONE — `quality_oracle.py`)
The Batch-2 generator must enforce uniqueness against the *real* textual reference,
not just exact normalized match:
- **EXACT duplicate** (normalized proposition equals any reference proposition) → **ALWAYS REJECT**.
- **NEAR duplicate** (token-Jaccard ≥ 0.80 vs any reference proposition) → **REJECT**,
  UNLESS an explicit, verified *intentional seed-family* relationship is declared
  (the two propositions form a declared pair AND share a verified `group_key`).
- Legitimate intentional near-duplicates are RETAINED only with provenance + `group_key`.
- Jaccard ≥ 0.80 is a **review/exception boundary**, not an absolute ban. This is why
  `/005`–`/050` (percept Jaccard 1.0, distinct propositions, shared `group_key`,
  co-located split) is permitted while `/049` (unintentional paraphrase of `/027`) is rejected.
- Authoritative reference = canonical GOLD + REJECT propositions/percepts from
  `review-records.json` (NOT the empty `adjudication.json` fields) **plus all**
  previously generated/adjudicated batch candidate files (Batch-1, Batch-2, …).
  Empty `adjudication.json` fields are never used as a substitute for real text.

## 2. Reference composition
Uniqueness/leakage checks MUST include:
- current canonical GOLD + REJECT propositions/percepts (from `review-records.json`),
- all previously generated/adjudicated batch candidates (Batch-1 `/027` + rest),
- the current batch's own retained candidates.

## 3. Regression test (DONE — `test_quality_oracle_regression.py`)
Fixture: Batch-1 `/027` + Batch-2 `/049`. Asserts the hardened oracle flags the pair
at Jaccard **0.818** as `NEAR_DUP_REJECT`. Guards against re-introducing the
exact-match-only oracle gap.

## 4. Preserve failed batch as evidence (DONE)
- `candidate_records_batch2.json` (original 50, includes failed `/049`) is left
  **byte-for-byte unchanged** as historical evidence.
- Do **NOT** edit `/049`'s text in place. `/049` is treated as a *rejected candidate*
  (its failure is the evidence that hardening works).
- Remediated set written to a **new** file `candidate_records_batch2_remediated.json`
  = 49 retained + exactly one replacement `/051`.

## 5. Replacement generation (DONE)
- Exactly **ONE** replacement: `rev://v0.3/b2/051`, new ID + provenance, genuinely
  distinct proposition ("The car horn indicates heavy rainfall." — a distinct
  CAP-04 distractor trap: spurious horn→rainfall-intensity correlation).
- Validated by `quality_oracle.evaluate`: **ACCEPT**, token-Jaccard vs `/027` = 0.231,
  vs `/049` = 0.143 (both ≪ 0.80).

## 6. Re-audit (DONE — re-run on remediated file)
Re-run `audit_6b3_batch2_quality.py` against `candidate_records_batch2_remediated.json`.
Verdict must be `PASS` (or at most CONDITIONAL on the non-blocking CAP lexical advisory)
with **zero** high-Jaccard novelty violations.

## 7. STOP — remaining gates (each needs separate authorization)
- Adjudication of remediated Batch-2 (only after fresh audit PASS).
- Merge/freeze into canonical GOLD (recompute GOLD_POSITIVE/total vs targets).
- Model selection → 6C → 6D.
- Cloud upload `6E20.1_CHECKPOINTS_FOR_CLOUD_UPLOAD`, delete local ~97 GB.
