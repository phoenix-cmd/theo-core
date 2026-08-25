# THEO SLM v0 — Phase 6B.3 Candidate Generation Plan: Independent Audit

**Document:** `theo-core/docs/research/theo-slm-phase-6b3-candidate-generation-plan-audit.md`
**Subject:** `theo-slm-phase-6b3-candidate-generation-plan.md` (2026-08-25 draft)
**Audit mode:** Independent review of the *plan*, not execution. No dataset, manifest, gold
artifact, or `adjudication.json` was modified. No execution authorization is created.
**Auditor stance:** Does the plan satisfy every requirement the reviewer set for 6B.3, follow
the approved 6B specification, preserve the 6B.2 outcome (226 GOLD, 38 valid REJECTs), and
respect all standing guardrails?

---

## A. Requirements coverage (reviewer's enumerated list)

| # | Required element | Plan § | Assessment |
|---|---|---|---|
| 1 | Target missing GOLD_POSITIVE volume first | §3, §9.2 | PRESENT. Positives are priority #1; budgets and stopping rule center on +111…+151 positives. |
| 2 | Require NEW unique propositions (not variants recreating rejected exact dups) | §4 | PRESENT. Uniqueness oracle blocks any proposition matching the 226 GOLD or the 38 rejected; semantic-novelty + NON_DERIVABLE required. |
| 3 | Preserve capability/domain coverage | §5 | PRESENT. All 13 CAP families + 8 domains required in GOLD_POSITIVE; under-covered cells prioritized. |
| 4 | Account for rejection/duplicate/NEEDS-REVIEW attrition | §6 | PRESENT. Explicit attrition channels with 1.6–2.0× overshoot buffer on positives. |
| 5 | Preserve existing 226 GOLD unchanged | §7 | PRESENT. Existing GOLD + `adjudication.json` declared read-only inputs; new pool is separate. |
| 6 | Do not modify adjudication.json / gold corpus | §7, §11 | PRESENT. Explicit non-mutation; merge only after a later authorized adjudication+freeze. |
| 7 | Define 1011 near-dup grouping by seed family for future splitting | §8 | PRESENT. `group_key` = `seed_case_id` (pert families) / authored-case prefix; `split_group_manifest.json` force-groups all 1011 pairs; grouped-by-seed-family splits (training spec §5.3). |
| 8 | Stopping criterion on final adjudicated GOLD counts + coverage, not raw candidate count | §9 | PRESENT. Five conditions, all adjudicated-GOLD/coverage based; raw count explicitly excluded as a stop metric. |
| 9 | Treat 350–500 as approved final GOLD target range, not a fixed raw-candidate number | §3, §9 | PRESENT. Targets are ranges; generation continues until adjudicated GOLD meets them. |
| 10 | Provenance, deterministic seeds, versioning, leakage/shortcut safeguards | §10 | PRESENT. `ds-v0.4-candidates` semver + manifest; fixed SEED_DATA; per-record provenance; §11/§13 leakage+shortcut safeguards; no-Qwen-labels; cos(H1) noted out-of-scope. |

**Verdict on requirements: PASS** — all 10 enumerated elements are present and substantive.

---

## B. Consistency with approved spec & 6B.2 outcome

- **6B.2 outcome honored:** 226 GOLD (29/66/131) and 38 REJECT accepted as valid; plan does
  **not** relax the dedup rule or restore rejected positives (§7, §4). ✔
- **Spec targets:** §12/§17 numbers (350–500 total; 140–180 positives; 40–60 abstain; 180–240
  negatives) cited correctly. ✔
- **Frozen-evaluation separation (§11)** and **model independence (§16)** honored. ✔
- **No Qwen-as-labels (6B design §D1)** carried into §10. ✔
- **cos(H1) exclusion (6B design §H/B1/G2)** correctly scoped out of dataset generation. ✔
- **Governance boundary** from the 6B review packet preserved: generation needs its own
  explicit authorization; no model/6C/6D. ✔

**Verdict on consistency: PASS.**

---

## C. Findings (clarifications; none blocking)

**F-1 (CLARIFICATION, non-blocking):** §9's stopping conditions reference *adjudicated* GOLD,
but §1 scopes adjudication of new candidates as a *separate authorized step*. The plan should
make explicit that generation executes **in batches interleaved with the (separately authorized)
6B.3 adjudication**, so §9 can be evaluated iteratively. Current text implies this but does not
state the batch↔adjudicate loop outright. Recommended wording addition at §9 or §11.

**F-2 (CLARIFICATION, non-blocking):** The uniqueness oracle (§4) correctly blocks **exact**
duplicates only; near-duplicates (Jaccard≥0.8) are intentionally *permitted* and handled by
grouping (§8). This distinction is correct but worth one explicit sentence in §4 to preempt
confusion ("near-duplicates are allowed and grouped, not blocked").

**F-3 (INFO):** Version tag `ds-v0.4-candidates` is a proposal; confirm it does not collide with
any existing `ds-v0.x` tag at execution time (spec §14). No collision expected.

**F-4 (INFO):** Raw-gen budgets (§3) are planning estimates only; the binding stop is §9. This
is correctly framed and needs no change.

---

## D. Guardrail preservation check

- 6E.20.x branch / cos(H1) exclusion: not implicated; preserved. ✔
- ADR-0028 / v0.4.1 contract: out of scope for dataset generation. ✔
- `adjudication.json` / gold corpus: declared read-only; not modified by this plan. ✔
- Source code / checkpoints / protocol: not modified. ✔
- No generation, model selection/download, training, 6C, 6D performed during planning. ✔

---

## E. Audit verdict

**APPROVE the 6B.3 Candidate Generation Plan for human sign-off**, with two optional
clarifications (F-1 batch↔adjudicate loop; F-2 explicit near-dup-allowed statement). Neither
blocks approval. The plan fully satisfies the reviewer's 10 enumerated requirements, honors the
6B.2 outcome (226 GOLD + 38 valid REJECTs), follows the approved spec, and preserves every
standing guardrail.

**This audit authorizes no execution.** Actual candidate generation remains blocked pending
the separate, explicit human authorization described in the plan's §11.
