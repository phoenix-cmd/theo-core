# THEO SLM v0 — Phase 6B.3 Batch-2 Merge / Freeze Report

**Authorization:** 6B.3 Batch-2 merge/freeze APPROVED. 50 records (40 GOLD_POSITIVE + 10 HARD_NEGATIVE) merged from `candidate_records_batch2_remediated.json`.
**Timestamp (UTC):** 2026-08-25T13:05:16.814779+00:00

## Pre / Post canonical hashes

- pre_merge:  `10979c5a30b0243d9fe20811c60f9a8b796180da0905783ec883a4c7b23e03f7` (asserted == `10979c5a…`)
- post_merge: `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4`
- records: 294 -> **344** (+0)

## Post-merge verification

- GOLD total: **300** (expected 300) -> PASS
- GOLD_POSITIVE: **84** (expected 84) -> PASS
- GOLD_ABSTAIN: 66 (expected 66) -> PASS
- HARD_NEGATIVE: 150 (expected 150) -> PASS
- REJECT: **44** (expected 44) -> PASS
- NEEDS_REVIEW: 0 (expected 0) -> PASS
- INV-01…INV-09 failures: **0** -> PASS
- Exact-duplicate validator: **0** violations across **0** pairs -> PASS
- Frozen-eval leakage (bm://,sp1://): count=0, raw=0 -> PASS
- Batch-1 provenance intact (30 merged): PASS
- Batch-2 provenance intact (50 merged) + full provenance retained: PASS
- /051 replacement present (HARD_NEGATIVE, provenance retained): PASS
- /049 permanently excluded: PASS
- Intentional near-dup /005~/050 group preserved (percept Jaccard 1.0, shared group_key): PASS
- review_id / case_id uniqueness (no double-append): PASS

## FINAL: ALL CHECKS PASS

## Cumulative coverage (post-merge GOLD)

- **Recoverable real coverage** (Batch-1 + Batch-2 records carry real tags): 13 CAP families {'CAP-01': 5, 'CAP-02': 6, 'CAP-03': 4, 'CAP-04': 8, 'CAP-05': 6, 'CAP-06': 6, 'CAP-07': 6, 'CAP-08': 6, 'CAP-09': 3, 'CAP-10': 7, 'CAP-11': 6, 'CAP-12': 8, 'CAP-13': 9}; 8 domains {'biology': 10, 'conflict': 6, 'engineering': 8, 'finance': 9, 'household': 12, 'medical': 12, 'physics': 10, 'weather': 13}.
- The remaining **220** GOLD are pre-existing historical records (rev://v0.2/*) whose canonical `capability_family`/`domain` fields are empty (frozen-corpus limitation, NOT altered by this merge). Their proposition/percept text lives in `review-records.json`; capability/domain taxonomy is populated only for the 6B.3-generated batches. Full tag inventory (incl. historical `unknown`): CAP 14 -> {'CAP-01': 5, 'CAP-02': 6, 'CAP-03': 4, 'CAP-04': 8, 'CAP-05': 6, 'CAP-06': 6, 'CAP-07': 6, 'CAP-08': 6, 'CAP-09': 3, 'CAP-10': 7, 'CAP-11': 6, 'CAP-12': 8, 'CAP-13': 9, 'unknown': 220}; domains 9 -> {'biology': 10, 'conflict': 6, 'engineering': 8, 'finance': 9, 'household': 12, 'medical': 12, 'physics': 10, 'unknown': 220, 'weather': 13}.

## Remaining gaps to approved targets

- GOLD_POSITIVE: 84 vs target 140–180 -> need **+56–+96** more positives.
- HARD_NEGATIVE: 150 vs target 180–240 -> need **+30–+90** more hard negatives.
- GOLD total: 300 vs target 350–500 -> need **+50–+200** more GOLD.

## STOP

Batch-2 merge/freeze complete and verified. **No Batch-3 generated; no model/6C/6D; canonical dedup/near-dup rules unchanged.** Awaiting separate authorization for model selection (6C) and beyond.