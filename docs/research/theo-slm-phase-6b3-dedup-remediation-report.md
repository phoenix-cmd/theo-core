# THEO SLM v0 — Phase 6B.3 Dedup Remediation Report

**Authorization:** REJECT-6 remediation APPROVED — exact six named non-keeper records; keepers retained as GOLD_POSITIVE.
**Rule applied:** 6B.2 exact-duplicate procedure (no exception invented).
**Timestamp (UTC):** 2026-08-25T12:01:54Z

## Pre/Post hashes

- pre_remediation: `6eb9b50754365b132f14b3e1b3168d7a41dda1abcd395de5cad0b0b88bb6eb4c`
- post_remediation: `10979c5a30b0243d9fe20811c60f9a8b796180da0905783ec883a4c7b23e03f7`

## Records rejected (status flipped GOLD_POSITIVE → REJECT; retained in history)

| reject review_id | case_id | duplicate_of (keeper) |
|---|---|---|
| rev://v0.2/211 | td://v0/pert/var_0133 | rev://v0.2/071 |
| rev://v0.2/001 | td://v0/pert/var_0137 | rev://v0.2/244 |
| rev://v0.2/080 | td://v0/pert/var_0141 | rev://v0.2/235 |
| rev://v0.2/216 | td://v0/pert/var_0145 | rev://v0.2/212 |
| rev://v0.2/025 | td://v0/pert/var_0181 | rev://v0.2/119 |
| rev://v0.2/171 | td://v0/pert/var_0185 | rev://v0.2/107 |

## Post-remediation verification

- GOLD total: **250** (expected 250) → PASS
- GOLD_POSITIVE: **44** (expected 44) → PASS
- GOLD_ABSTAIN: 66 (expected 66) → PASS
- HARD_NEGATIVE: 140 (expected 140) → PASS
- REJECT: **44** (expected 44) → PASS
- NEEDS_REVIEW: 0 (expected 0) → PASS
- INV-01…INV-09 failures: **0** → PASS
- Exact-duplicate validator: **0** violations across **0** pairs → PASS
- Batch-1 provenance intact (30 merged entries): PASS
- Batch 1 introduced 0 duplicate violations: PASS

## FINAL: ALL CHECKS PASS

## STOP

Remediation complete and verified. No Batch-2 generated; no model/6C/6D. Awaiting separate authorization for Batch-2 generation once this clean canonical state is independently confirmed.