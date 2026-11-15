# 6C.4 Independent Audit (Pre-Training)

**Audited artifacts:**
- `theo-data/datasets/theo_slm_v0_review/adjudication.json` (frozen 344 GOLD/REJECT records)
- `theo_slm_v0_deduplicated/candidate_records.json`, `6b3_batch1/candidate_records_batch1.json`, `6b3_batch2/candidate_records_batch2_remediated.json` (frozen case-text sources)
- `theo-data/datasets/theo_slm_v0_artifacts/phase-6e3/{semantic-probe-results.json, benchmark-results.json, populated-evidence-corpus.json}` (frozen 66-case fixture + corpus)
- `6c4_mapping_manifest.json` (read-only derived)

**Audit date:** 2026-08-26
**Auditor:** independent review of the 6C.4 alignment plan against the authorization's five audit points.
**Mode:** READ-ONLY. No source data, weights, labels, thresholds, or architecture definitions were modified.

---

## A. SemanticInterpretation training-target reconstruction

| Check | Result |
|---|---|
| Source lineage via `case_id` / `source_case_id` | Deterministic join over 3 frozen sources in fixed precedence. |
| 1:1 mapping | **Yes** — `6c4_mapping_manifest.json`: 344 adjudication records → 344 unique texts; **0 missing**, **0 text conflicts**. |
| Deterministic | **Yes** — precedence order is fixed; first match wins; re-running yields identical manifest (SHA-stable inputs). |
| No inferred/invented text | **Yes** — percept read verbatim from frozen source; manifest stores only `percept_sha256`, not copied text. |
| Mapping manifest with hashes | **Yes** — per-row `percept_sha256` + `evidence_ids_sha256`; source-file SHA-256 recorded (`deduplicated a7b4e845…`, `batch1 06fe6e59…`, `batch2_remediated 6b743d00…`). |

Verdict: **PASS.** A complete, auditable, 1:1 deterministic mapping exists for all 300 training-included records (44 `REJECT` excluded).

## B. Contract correctness

| Check | Result |
|---|---|
| Exact supervised output schema defined | **Yes** — `{"decision": "SHOULD_PROPOSE"|"SHOULD_ABSTAIN", "reasoning": "..."}` (+ `evidence_refs` for B/C). |
| Same contract as evaluation | **Yes** — matches the `SemanticInterpretation` envelope the 66-case fixture expects; train/eval parity confirmed. |
| Obsolete `DECISION:` format excluded | **Yes** — plan §2 explicitly forbids the `DECISION: <SENSITIVE|ABSTAIN|NEGATIVE>` string form used by M1 v1/v2 (root cause of their E0 failures). Only the JSON contract is permitted. |

Verdict: **PASS.**

## C. Evidence-relevance supervision

| Check | Result |
|---|---|
| Valid signal exists in frozen data? | **Yes** — `grounding_snapshot.evidence_ids` present in 100% of source records; derivable as positive relevance class vs available `concept_ids`/`belief_ids`/`rule_ids`. |
| Absence reported (not manufactured)? | **Yes** — signal read directly from frozen `grounding_snapshot`; no synthetic labels created. |
| Non-trivial where it matters | **Yes** — `distractor_count`/`trap_propositions` mark cases where `evidence_ids` ⊂ available symbols, giving genuine relevance contrast. |
| Minimum additional corpus | **None required** for a baseline evidence head. Optional enhancement: human-confirmed relevance labels on ~50–100 stratified distractor/conflict cases. |

Verdict: **PASS (signal present; no manufacturing).**

## D. Fair architecture comparison

| Check | Result |
|---|---|
| Same supervised data | **Yes** — all A/B/C use the 300 records from the manifest (identical texts + labels). |
| Same split groups | **Yes** — grouped by `capability_family`/case prefix, stratified by `final_status`. |
| Same seed policy | **Yes** — fixed `20260826`; greedy/de deterministic decode at eval. |
| Same evaluation fixture | **Yes** — frozen 66-case instrument + `f99ca025…` corpus; **fixture/training overlap = 0** (disjoint `case_id` namespaces). |
| Same behavioral gates | **Yes** — identical §7 gates; Group E ≥80%, abstention ≥90%, E0, grounding, E5/E6 (N/A on fixture), latency ≤0.5s all preserved unchanged. |

Leakage check: the 66-case fixture `case_id`s share no overlap with the 344 training `case_id`s (`fixture_overlap = 0` in the manifest). Verdict: **PASS.**

## E. Training eligibility decision

**READY_FOR_SUPERVISED_EVALUATION**

All five audit points pass. A fair supervised comparison of A, B, and C is constructible from the frozen data without modifying any source, weight, label, threshold, or architecture definition. Caveats (non-blocking):
1. `evidence_ids` is grounding-evidence, not independently human-labeled relevance (only non-trivial on distractor cases).
2. E5/E6 are not scorable on the fixture (no novelty labels there) — remain N/A, consistent with prior evals.
3. The ≤0.5s latency gate is hardware-bound and will fail on this GPU/offload path regardless of architecture; treat as a config decision.

## F. Constraint compliance

- 300-GOLD dataset: **unmodified**.
- 66-case fixture + populated corpus: **unmodified**.
- Model weights / labels / thresholds / architecture definitions: **unmodified**.
- No training of A/B/C performed in this step.
- Only new artifacts: this audit, the plan, and the read-only `6c4_mapping_manifest.json`.

**Audit verdict: READY_FOR_SUPERVISED_EVALUATION (caveats documented). No blocker.**
