# THEO SLM v0 — Phase 6B.3 Cumulative Dedup Remediation Audit (READ-ONLY)

**Mode:** Read-only forensic audit. No records modified; canonical GOLD freeze, Batch-2, model, 6C/6D untouched.
**Canonical adjudication.json hash:** `6eb9b50754365b132f14b3e1…`

## 1. The six exact-duplicate groups (canonical GOLD_POSITIVE)

All 12 member records carry `rev://v0.2/*` review_ids → they **predate Batch 1** (`rev://v0.3/b1/*`). **Batch 1 contributed 0 of the six violations** (verified: no `rev://v0.3/b1/*` appears in any duplicate group).

| # | Proposition | Member review_id | case_id | Cap | Origin | Approved disposition |
|---|---|---|---|---|---|---|
| 1 | Points to severe thunderstorm approach.… | rev://v0.2/071 | td://v0/pert/var_0042 | CAP-05 | CANONICAL_6B2 | KEEP (GOLD_POSITIVE) |
| 1 | Points to severe thunderstorm approach.… | rev://v0.2/211 | td://v0/pert/var_0133 | CAP-05 | CANONICAL_6B2 | REJECT (exact duplicate) |
| 2 | Evidence shows nuclear radiation leak.… | rev://v0.2/107 | td://v0/pert/var_0094 | CAP-09 | CANONICAL_6B2 | KEEP (GOLD_POSITIVE) |
| 2 | Evidence shows nuclear radiation leak.… | rev://v0.2/171 | td://v0/pert/var_0185 | CAP-09 | CANONICAL_6B2 | REJECT (exact duplicate) |
| 3 | Points to blizzard weather alert.… | rev://v0.2/119 | td://v0/pert/var_0090 | CAP-08 | CANONICAL_6B2 | KEEP (GOLD_POSITIVE) |
| 3 | Points to blizzard weather alert.… | rev://v0.2/025 | td://v0/pert/var_0181 | CAP-08 | CANONICAL_6B2 | REJECT (exact duplicate) |
| 4 | Indicates thermal boiling state.… | rev://v0.2/212 | td://v0/pert/var_0054 | CAP-08 | CANONICAL_6B2 | KEEP (GOLD_POSITIVE) |
| 4 | Indicates thermal boiling state.… | rev://v0.2/216 | td://v0/pert/var_0145 | CAP-08 | CANONICAL_6B2 | REJECT (exact duplicate) |
| 5 | Observation suggests vibrational resonance insta… | rev://v0.2/235 | td://v0/pert/var_0050 | CAP-07 | CANONICAL_6B2 | KEEP (GOLD_POSITIVE) |
| 5 | Observation suggests vibrational resonance insta… | rev://v0.2/080 | td://v0/pert/var_0141 | CAP-07 | CANONICAL_6B2 | REJECT (exact duplicate) |
| 6 | Evidence shows foggy runway hazard.… | rev://v0.2/244 | td://v0/pert/var_0046 | CAP-06 | CANONICAL_6B2 | KEEP (GOLD_POSITIVE) |
| 6 | Evidence shows foggy runway hazard.… | rev://v0.2/001 | td://v0/pert/var_0137 | CAP-06 | CANONICAL_6B2 | REJECT (exact duplicate) |

## 2. Applied rule & recommended disposition

- **Approved rule (6B.2 exact-duplicate procedure):** among accepted GOLD with identical normalized propositions, keep the record with the lexicographically smallest `case_id` as GOLD_POSITIVE; REJECT the remaining exact duplicates.
- **No approved exception exists** in the 6B specification for exact duplicates (the single defined disposition is REJECT, consistent with the 38 already-REJECTed records). No new exception is invented here.
- **Recommended:** REJECT the 6 non-keeper duplicates: rev://v0.2/211, rev://v0.2/171, rev://v0.2/025, rev://v0.2/216, rev://v0.2/080, rev://v0.2/001.
- **Keeper (retained GOLD_POSITIVE):** rev://v0.2/071, rev://v0.2/107, rev://v0.2/119, rev://v0.2/212, rev://v0.2/235, rev://v0.2/244.

## 3. Quantified impact

### 3a. Option A — REJECT the 6 non-keepers (RECOMMENDED, spec-compliant)

- GOLD_POSITIVE: **50 → 44** (−6)
- GOLD_ABSTAIN: 66 (unchanged)
- HARD_NEGATIVE: 140 (unchanged)
- **Total GOLD: 256 → 250** (−6)
- REJECT: 38 → 44 (+6)
- Removed capabilities (GOLD_POSITIVE subset): {'CAP-05': 1, 'CAP-09': 1, 'CAP-08': 2, 'CAP-07': 1, 'CAP-06': 1}
- Removed domains: {'v0': 6} — canonical 226 used a single 'v0' tag, so this does not change the 8-domain real taxonomy contributed by Batch-1.
- After Option A, exact-duplicate validator would report **0** violations.
- All 13 capability families remain represented in GOLD; balance targets: GOLD_POSITIVE 44 (target 140–180, shortfall +96…+136); total GOLD 250 (target 350–500, shortfall +100…+250).

### 3b. Option B — retain both (NO remediation)

- GOLD_POSITIVE stays 50, total GOLD 256, REJECT 38.
- **Spec-noncompliant:** the exact-duplicate prohibition remains failed (6 violations). Not recommended; listed only for comparison.

### 3c. Other dispositions

- Demoting duplicates to GOLD_ABSTAIN/HARD_NEGATIVE or inventing a 'legacy exception' has **no basis in the 6B specification** and is not recommended.

## 4. Canonical integrity verification (independent)

- Total records: **294** (expected 264 original + 30 Batch-1 = 294).
- GOLD total: **256** (expected 256) | REJECT: **38** (expected 38).
- GOLD_POSITIVE 50 / GOLD_ABSTAIN 66 / HARD_NEGATIVE 140.
- Batch-1 merged entries: 30 (distinct review_ids: 30).
- Max review_id frequency: 1; max case_id frequency: 1 (both must be 1 → no double-append residue).
- **Integrity PASS:** True.

## 5. Re-run INV-01…INV-09 + exact-duplicate validator (read-only)

- INV-01…INV-09 total failures: **0** (PASS).
- Exact-duplicate validator (positive subset): **6** duplicate propositions across **6** pairs — these are the six pre-existing groups (Batch-1 introduced none). After applying Option A they would resolve to 0.

## 6. STOP

Audit complete. No records deleted, no status changed, no GOLD freeze changed, no Batch-2 generated, no model/6C/6D. Awaiting human decision on the recommended REJECT disposition (a separate authorized remediation step) and on Batch-2 authorization.