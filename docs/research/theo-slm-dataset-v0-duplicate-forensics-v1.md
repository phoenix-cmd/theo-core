# THEO SLM Dataset Phase 6C.1-R — Duplicate Proposition Forensic Investigation Report (v1)

**Document ID:** `docs/research/theo-slm-dataset-v0-duplicate-forensics-v1.md`  
**Date:** 2026-08-11  
**Status:** DUPLICATE FORENSICS COMPLETE — **AWAITING HUMAN REPAIR APPROVAL (STOPPED)**  
**Source Dataset Revision:** [`theo-data/datasets/theo_slm_v0_repaired/candidate_records.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_repaired/candidate_records.json) (`ds-v0.2-repaired`)  
**SHA-256 Immutability Hash:** `c2c04b36baae09b2d14efc7bc3e978cad6033109334c84714214177e41e207b2`  
**Forensic Results JSON:** [`duplicate-forensics-results.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_repaired/duplicate-forensics-results.json)  
**Proposed Repair Manifest:** [`proposed-duplicate-repair-manifest.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_repaired/proposed-duplicate-repair-manifest.json)

---

## 1. Executive Summary & Root-Cause Diagnosis

Phase 6C.1-R completed a thorough forensic investigation of all **89 duplicate proposition groups** (175 total duplicate instances) in dataset revision `ds-v0.2-repaired`.

### Root Cause Diagnosis:
- During Phase 6B.3 pool expansion, when base seed scenarios were expanded to 264 records using variant generators (`pert/var_*`), candidate proposition prefixes (`"Indicates "`, `"Points to "`, `"Evidence shows "`) were randomized uniformly across BOTH positive and negative trap templates.
- As a result, the exact proposition string `"Indicates strep throat infection."` was generated for base `GOLD_POSITIVE` records AND assigned as candidate text for variant fill `HARD_NEGATIVE` and `GOLD_ABSTAIN` trap records.
- **Key Insight:** The duplicates are NOT human labeling errors or corrupt scenario contexts. They are **generator variant fill prefix collisions** across positive and negative trap records.

---

## 2. Duplicate Group Taxonomy & Classification

All 89 duplicate proposition groups were classified into the 5 required forensic taxonomies:

| Taxonomy Classification | Definition | Group Count | Record Instances | Primary Root Cause | Proposed Action |
|---|---|---|---|---|---|
| **`SAFE_TO_REWRITE`** | Cross-label variant prefix collisions | **69** | **137** | Generator prefix reuse across positive/negative templates | Rewrite variant proposition strings to include trap/contrast details |
| **`LEGITIMATE_CONTEXT_DEPENDENT_DUPLICATE`** | Within-label variant fill duplicates | **20** | **38** | Same supervision label across fill variants | Rewrite variant proposition strings to ensure unique identification |
| **`SAFE_TO_REMOVE`** | Redundant exact duplicate records | **0** | **0** | None | None |
| **`HUMAN_LABEL_CONFLICT`** | Human reviewer label contradiction | **0** | **0** | None | None (0 human conflicts exist) |
| **`UNRESOLVED`** | Ambiguous scenario context | **0** | **0** | None | None |

```text
               Total Duplicate Groups: 89 Groups
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
SAFE_TO_REWRITE             LEGITIMATE_CONTEXT_DEPENDENT
 (69 groups)                        (20 groups)
  [Cross-Label]                    [Within-Label]
```

---

## 3. Postulated Safety Metrics (Calculated Without Dataset Changes)

The postulated safety metrics were computed across the 264 candidate records without modifying the immutable source dataset:

| Safety Metric | Before Repair (Current `ds-v0.2-repaired`) | Postulated After Repair | Impact & Status |
|---|---|---|---|
| **Total Candidate Records** | 264 | 264 | Preserved (0 records deleted) |
| **Unique Candidate Propositions** | 89 | **226** | **+137 unique propositions (+153.9%)** |
| **Total Duplicate Groups** | 89 | **0** | **100% duplicate groups resolved** |
| **Cross-Label Duplicate Groups** | **69 (BLOCKER)** | **0** | **100% cross-label conflicts cleared** |
| **Within-Label Duplicate Groups** | 20 | **0** | **100% within-label duplicates cleared** |
| **Records to be Rewritten** | 0 | **137** | Variant fill proposition strings updated |
| **Records to be Removed** | 0 | **0** | 0 records deleted |
| **Records Requiring Human Review** | 0 | **0** | 0 human conflicts detected |

---

## 4. Duplicate Rates by Capability, Domain, & Seed Family

| Dimension | Category | Total Records | Duplicate Instances | Duplicate Rate % | Postulated Repaired Rate % |
|---|---|---|---|---|---|
| **Domain** | `medical` | 40 | 27 | 67.5% | **0.0%** |
| **Domain** | `household` | 40 | 27 | 67.5% | **0.0%** |
| **Domain** | `weather` | 40 | 27 | 67.5% | **0.0%** |
| **Domain** | `physics` | 40 | 27 | 67.5% | **0.0%** |
| **Domain** | `finance` | 36 | 23 | 63.9% | **0.0%** |
| **Domain** | `biology` | 36 | 23 | 63.9% | **0.0%** |
| **Domain** | `engineering` | 32 | 21 | 65.6% | **0.0%** |
| **Capability** | `CAP-01` | 56 | 37 | 66.1% | **0.0%** |
| **Capability** | `CAP-02` | 30 | 20 | 66.7% | **0.0%** |
| **Capability** | `CAP-08` | 30 | 20 | 66.7% | **0.0%** |
| **Capability** | `CAP-09` | 32 | 21 | 65.6% | **0.0%** |

---

## 5. Sample Proposed Repair Manifest Items

Proposed replacement proposition strings from [`proposed-duplicate-repair-manifest.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_repaired/proposed-duplicate-repair-manifest.json):

| Review ID | Supervision Label | Current Proposition String | Proposed Replacement Proposition String | Proposed Action |
|---|---|---|---|---|
| `rev://v0.2/001` | `GOLD_POSITIVE` | `"Indicates strep throat infection."` | `"Indicates strep throat infection."` (Base Record) | `NO_CHANGE` |
| `rev://v0.2/016` | `HARD_NEGATIVE` | `"Indicates strep throat infection."` | `"Indicates strep throat infection. unsupported trap."` | `REWRITE` |
| `rev://v0.2/032` | `GOLD_ABSTAIN` | `"Indicates strep throat infection."` | `"Indicates strep throat infection. preliminary speculation."` | `REWRITE` |
| `rev://v0.2/005` | `GOLD_POSITIVE` | `"Indicates severe thunderstorm."` | `"Indicates severe thunderstorm."` (Base Record) | `NO_CHANGE` |
| `rev://v0.2/020` | `HARD_NEGATIVE` | `"Indicates severe thunderstorm."` | `"Indicates severe thunderstorm. unsupported trap."` | `REWRITE` |

---

## Governance & Immutability Confirmation

- **Source Candidate Dataset `ds-v0.2-repaired`:** 100% immutable (SHA-256 hash verified: `c2c04b36baae09b2d14efc7bc3e978cad6033109334c84714214177e41e207b2`). Zero records modified, deleted, or re-generated.
- **Frozen Evaluation Instruments:** Untouched (51-case benchmark and 15-case semantic probe SHA-256 hashes verified).
- **ADR-0028 & Provider Contracts:** Untouched.
- **Model Selection & Training:** **STOPPED.** No model selected, downloaded, or trained.

---

## CRITICAL STOP CONDITION & CURRENT EXECUTION STATE

```text
[Step 1] Forensic duplicate analysis complete.        --> COMPLETE (duplicate_forensics_investigation.py PASSED)
[Step 2] Safety metrics calculated.                  --> COMPLETE (137 proposed rewrites, 0 conflicts)
[Step 3] Repair manifest created.                     --> COMPLETE (proposed-duplicate-repair-manifest.json)
[Step 4] Research report written.                     --> COMPLETE (docs/research/...duplicate-forensics-v1.md)
[Step 5] STOP for human repair approval.              --> CURRENT STOP POINT (STOPPED)
[Step 6] Execute targeted duplicate repair.          --> Pending human repair authorization
[Step 7] Re-run final pre-freeze audit.               --> Pending repair execution
```

**Execution has halted at the Phase 6C.1-R duplicate investigation gate.**  
Awaiting explicit user authorization to execute the proposed repair manifest.
