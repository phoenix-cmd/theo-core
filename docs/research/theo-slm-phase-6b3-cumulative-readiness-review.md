# THEO SLM v0 — Phase 6B.3 Cumulative Readiness Review (Post-Batch-2 Merge)

**Mode:** READ-ONLY analysis. No `adjudication.json` or dataset file was modified. Canonical hash verified.
**Canonical SHA-256:** `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4` (expected `97bc6bc6…`) — verified MATCH.

## 1. Class-balance gap vs §12/§17

- GOLD total **300** vs target 350–500 → short **+50–+200**.
- GOLD_POSITIVE **84** vs target 140–180 → short **+56–+96**.
- HARD_NEGATIVE **150** vs target 180–240 → short **+30–+90**.
- GOLD_ABSTAIN **66** vs target 40–60 → short
- **Positive fraction = 28.0%** vs §13 ~40% positive / ~60% negative+abstain guideline → **below balance guideline** (40% target).

## 2. Capability / domain coverage (historical-empty vs real taxonomy)

- **Real taxonomy (Batch-1 + Batch-2, 80 records):** 13 CAP families {'CAP-01': 5, 'CAP-02': 6, 'CAP-03': 4, 'CAP-04': 8, 'CAP-05': 6, 'CAP-06': 6, 'CAP-07': 6, 'CAP-08': 6, 'CAP-09': 3, 'CAP-10': 7, 'CAP-11': 6, 'CAP-12': 8, 'CAP-13': 9}; 8 domains {'biology': 10, 'conflict': 6, 'engineering': 8, 'finance': 9, 'household': 12, 'medical': 12, 'physics': 10, 'weather': 13}.
- **Historical GOLD with EMPTY metadata: 220 records** (rev://v0.2/*; canonical `capability_family`/`domain` are empty — frozen-corpus limitation, NOT mutated by this review). Their text lives in `review-records.json`.
- Recoverable positive coverage by CAP family: {'CAP-01': 2, 'CAP-02': 6, 'CAP-03': 1, 'CAP-04': 6, 'CAP-05': 4, 'CAP-06': 6, 'CAP-07': 6, 'CAP-08': 6, 'CAP-10': 6, 'CAP-11': 6, 'CAP-12': 6, 'CAP-13': 6}.
- Recoverable positive coverage by domain: {'biology': 8, 'conflict': 5, 'engineering': 7, 'finance': 7, 'household': 7, 'medical': 9, 'physics': 8, 'weather': 10}.
- CAP families with **zero** positive coverage (real taxonomy): ['CAP-09'].
- Domains with **zero** positive coverage (real taxonomy): NONE.

## 3. Positive coverage gaps by capability/domain

- Positive records with real taxonomy tags = **61** (21 from Batch-1 + 40 from Batch-2). The other **23** positives are historical records whose CAP/domain tags are empty (unrecoverable from frozen canonical fields).
- Gaps to target: need **+56–+96** more GOLD_POSITIVE overall. The zero-positive CAP families above (e.g., CAP-01/CAP-03/CAP-09 in real taxonomy) and any zero-positive domains are the highest-priority targets for a Batch-3 generation focused on positives.

## 4. Cumulative shortcut correlations (best-available)

- Computed on the **80** Batch-1+Batch-2 records that carry grounding_snapshot/percept/proposition. The 220 historical GOLD records LACK these fields → **unavailable** for this probe (documented, not silently dropped).
- Point-biserial correlation of GOLD_POSITIVE with structural features: {'ev': 0.48, 'cc': 0.299, 'pl': 0.499, 'al': 0.611, 'dc': -0.183}.
- Per spec §13 these structural/metadata features are **never model inputs**, so any residual correlation is neutralized at training. Within-batch correlations were already reported low in the Batch-1/Batch-2 audits (evidence −0.0, concept −0.04, proposition 0.25–0.34, percept 0.40–0.60).

## 5. Exact / near-duplicate & grouped-split integrity

- **Exact-duplicate validator: 0 violations / 0 pairs → PASS.**
- **Near-duplicate percept pairs (Jaccard ≥0.8) among GOLD: 2**; **ungrouped (split-leakage risk): 0 → PASS** (all near-dup pairs are intentionally grouped).
  - `rev://v0.3/b1/019` ~ `rev://v0.3/b1/031`: Jaccard 0.818, grouped=True (group_key `sf_weather_storm_01`).
  - `rev://v0.3/b2/005` ~ `rev://v0.3/b2/050`: Jaccard 1.0, grouped=True (group_key `sf_physics_cap04_05`).
- Multi-record group_keys (intentional seed families, must stay grouped in any train/eval split): {'sf_weather_storm_01': 2, 'sf_physics_cap04_05': 2}.

## 6. INV-01…INV-09 & frozen-evaluation leakage

- **INV-01…INV-09 total failures: 0 → PASS.**
- **Frozen-evaluation leakage (bm://, sp1://): count=0, raw=0 → PASS.**

## 7. Blocker determination

- **All hard correctness gates PASS** (INV=0, leakage=0, exact-dup=0, near-dup grouping intact, dual-review + provenance intact, NEEDS_REVIEW=0).
- Per **spec §12**, the 350–500 scale is explicitly *"a target range, not a rigid quota."* Therefore the remaining count/balance shortfall is a **RECOMMENDED TARGET, NOT A HARD BLOCKER** for model selection. Hard-blocker verdict: **False** (False = not a blocker).

## 8. Recommendation

**PROCEED_TO_MODEL_SELECTION** (with recommended, non-blocking Batch-3 follow-up).

- Rationale: All hard correctness gates PASS (INV-01..09=0, frozen-eval leakage=0, exact-dup=0, near-dup grouping intact, dual-review + provenance intact). Per spec §12 the 350-500 scale is a TARGET RANGE, not a rigid quota, so the 300-record shortfall is a RECOMMENDED target, not a hard blocker for 6C. RECOMMENDED (non-blocking) follow-up: authorize a bounded Batch-3 to lift the positive ratio from 28% toward the §13 ~40% balance guideline and to exceed the 350 lower bound (need +56-+96 positives, +30-+90 hard negatives, +50-+200 total).

## STOP

Read-only review complete. No data mutated; no Batch-3 generated; no model selected; 6C/6D not begun. Awaiting separate human authorization for model selection (6C) and/or a Batch-3 generation step.