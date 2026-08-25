# THEO SLM v0 — Phase 6B.2 GOLD Adjudication Plan: Independent Audit

**Document:** `theo-core/docs/research/theo-slm-phase-6b2-gold-adjudication-plan-audit.md`
**Subject under audit:** `theo-slm-phase-6b2-gold-adjudication-plan.md` (2026-08-25 draft)
**Audit mode:** Independent review of the *plan*, not execution. No dataset, manifest, or gold
artifact was modified. No execution authorization is created by this audit.
**Auditor stance:** Does the plan fully and correctly establish the 11 required elements,
follow the approved 6B specification, and preserve all standing guardrails?

---

## A. Completeness — the 11 required elements

| # | Required element | Present in plan | Assessment |
|---|---|---|---|
| 1 | Candidate-record inventory and current counts | §1 | PRESENT. 264 source / 264 review / 0 adjudicated / 264 prior-draft gold. Counts verified by direct read. |
| 2 | Exact human-review workflow | §2 | PRESENT. Two-distinct-reviewer model, four-label assignment, κ computation, escalation. |
| 3 | GOLD / REJECT / NEEDS-REVIEW dispositions | §3 | PRESENT. Each disposition has necessary-and-sufficient conditions tied to spec criteria + invariants. |
| 4 | Adjudication criteria for `SemanticInterpretation` | §4 | PRESENT. Form, grounding (INV-07), relation enum, non-derivability, decision relevance, isolation (INV-03), confidence handling. |
| 5 | Malformed / schema-invalid handling | §5 | PRESENT. Pre-check via `quality_gates.py`; malformed → REJECT; INV-08 failure → NEEDS-REVIEW; no in-plan repair. |
| 6 | Duplicate and near-duplicate handling | §6 | PRESENT. Exact-dup → one GOLD + flag other; near-dup/Jaccard≥0.8 opposite-status pairs preserved but grouped-by-seed-family for splits. |
| 7 | Leakage / shortcut checks | §7 | PRESENT. Frozen-eval leakage (51-case/15-case), shortcut probes, metadata-excluded-from-input, report-don't-auto-repair. |
| 8 | Provenance and frozen-source requirements | §8 | PRESENT. Per-record trace, `adjudication.json` fields, read-only source hashing, no-Qwen-labels rule. |
| 9 | Reviewer agreement requirements | §9 | PRESENT. 100% agreement for Gold Positive, κ=1.0000 gate, INV-08 distinct-reviewers, disagreement → NEEDS-REVIEW. |
| 10 | Exact point of immutability | §10 | PRESENT. Mutability throughout; immutability only at GOLD FREEZE after 5 enumerated prerequisites + explicit human authorization. |
| 11 | Gate separating adjudication from model selection / 6C | §11 | PRESENT. Explicit phase ordering; no model select/download/train/6C/6D authority; `cos(H1)` exclusion noted as 6C/6D-scoped. |

**Verdict on completeness: PASS** — all 11 elements are present and individually substantive.

---

## B. Spec-conformance

- **GOLD criteria** (§3, §10.1, line 90) are quoted verbatim and applied correctly:
  Gold Positive = `SEMANTIC_NOVEL` + `DECISION_RELEVANT` + 100% agreement across 2 distinct
  reviewers. ✔
- **Schema invariants INV-01…INV-09** (§5 of spec) are referenced in §3/§4/§5/§9 with correct
  semantics (INV-02/04 target-null for negative/abstain; INV-05/06 non-derivable; INV-07
  grounding bounds; INV-08 dual distinct reviewers; INV-03 proposition isolation). ✔
- **Frozen-evaluation separation** (§11) and **model independence** (§16) are honored: the 51-
  case / 15-case instruments are evaluation-only and excluded from the corpus. ✔
- **No Qwen-as-labels** (6B design §D1) is carried into §8. ✔
- **`cos(H1)` exclusion** (6B design §H/B1/G2) is correctly scoped to 6C/6D and explicitly
  stated as not implicated by human/symbolic dataset adjudication. ✔
- **Governance boundary** from the 6B review packet (design approval ≠ model selection/training/
  6C/6D) is preserved verbatim in §11 and §12. ✔

**Verdict on spec-conformance: PASS.**

---

## C. Findings (must be resolved or acknowledged before execution)

**F-1 (INFO, already addressed in plan):** The plan initially cited the spec's nominal
`scripts/validate_dataset_schema.py`, which does **not exist**. The real implementation is
`scripts/dataset_generator/quality_gates.py` (`check_invariants`, INV-01…INV-09). The plan's
§5 was corrected to reference the actual module. No blocking issue remains, but the spec §18
prerequisite #4 is itself inaccurate and should be corrected in the dataset spec at a convenient
time.

**F-2 (ATTENTION — reconciliation of prior gold):** `adjudication.json` is empty (`[]`) yet a
`theo_slm_v0_gold` corpus of 264 records (67/66/131) already exists with recorded hashes. The
plan's §0 correctly treats the prior gold as *non-authoritative draft evidence* and re-derives
GOLD from `review-manifest.json` under the spec criteria, reaching immutability only after
human sign-off. **Recommended human decision at approval:** explicitly confirm whether (a) the
existing 67/66/131 split is accepted as the adjudication basis (fast-path, still requires sign-
off + invariant re-validation), or (b) `adjudication.json` is independently recomputed from the
reviewers. The plan supports either; the choice is the reviewer's, not the plan's.

**F-3 (SCOPE — scale gap):** the approved spec targets 350–500 training records (§12/§17); the
available approved pool is 264. The plan correctly scopes 6B.2 to adjudicate the 264 available
records and flags additional generation as a separate authorized activity. No action needed
beyond acknowledging that the target range is not met by this pool.

**F-4 (CONSISTENCY — frozen-source integrity):** §8 requires SHA-256 of `candidate_records.json`
and `review-manifest.json` in `corpus-manifest.json`. The prior gold-freeze already computed
hashes for these (e.g., candidate `c2c04b36…`, review-manifest `f07d30b9…`). The plan should
reuse those exact hashes as the frozen-source anchor rather than recomputing ad hoc. Minor;
documentation-only.

**F-5 (NO BLOCKING):** No element requests model selection, download, dataset mutation,
training, 6C, or 6D. The plan is correctly confined to 6B.2.

---

## D. Guardrail preservation check

- 6E.20.x branch: not touched; closure and `cos(H1)` exclusion preserved in §11. ✔
- Frozen benchmark/probe (6A.1 / 6A.2): excluded from corpus, §7. ✔
- ADR-0028 / v0.4.1 contract: out of scope for dataset adjudication; not implicated. ✔
- Source code / checkpoints / protocol: not modified during planning. ✔
- Candidate dataset: inspected only; no mutation (§0, §12). ✔

---

## E. Audit verdict

**CONDITIONAL PASS — APPROVE the plan for human sign-off, subject to acknowledging F-2
(reconcile the prior gold corpus) and F-3 (scale gap) at approval.**

The plan is complete (all 11 elements), spec-conformant, and correctly bounded by the standing
governance boundary. No blocking defect was found. The only material open question is the
human decision called out in F-2: whether to accept the existing 67/66/131 draft split as the
adjudication basis or to recompute `adjudication.json` independently. Either path is supported
by the plan and gated on explicit human authorization before any mutation occurs.

**This audit authorizes no execution.** Population of `adjudication.json`, invariant
validation runs, and GOLD freeze remain blocked pending the separate, explicit human
authorization described in the plan's §12.
