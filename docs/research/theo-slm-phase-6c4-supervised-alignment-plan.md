# 6C.4 Supervised Contract / Data Alignment Plan

**Date:** 2026-08-26
**Status:** READ-ONLY PLANNING / AUDIT. No training, no dataset/weight/label/threshold/architecture changes.
**Goal:** Determine whether a *fair* supervised comparison of A, B, C can be constructed from the frozen 300-GOLD corpus and the frozen 66-case evaluation fixture.

---

## 1. SemanticInterpretation training-target reconstruction

**Join (auditable, deterministic, 1:1).** Each GOLD record in `adjudication.json` is joined to its case text by `case_id` across, in fixed precedence:
1. `theo_slm_v0_deduplicated/candidate_records.json` (264 records)
2. `theo_slm_v0_review/6b3_batch1/candidate_records_batch1.json` (31)
3. `theo_slm_v0_review/6b3_batch2/candidate_records_batch2_remediated.json` (50)

**Verified result (see `6c4_mapping_manifest.json`):**
- Adjudication records: **344**; covered by union: **344/344**; **missing: 0**; **text conflicts: 0**.
- Union has 345 unique `case_id`s; the 1 extra source record without an adjudication label is **excluded** (no label).
- Mapping is therefore **1:1 and deterministic** (case_id → first match in precedence order).
- **No text inferred or invented.** The percept is read verbatim from the frozen source; only SHA-256 hashes are stored in the manifest.
- `REJECT` (44) excluded → **300 training-included** records (84 `GOLD_POSITIVE`→PROPOSE, 66 `GOLD_ABSTAIN`→ABSTAIN, 150 `HARD_NEGATIVE`→ABSTAIN).
- Source-file SHA-256 recorded for provenance (`deduplicated` `a7b4e845…`, `batch1` `06fe6e59…`, `batch2_remediated` `6b743d00…`).

**Training target (decision):**
```
GOLD_POSITIVE       -> SHOULD_PROPOSE
GOLD_ABSTAIN        -> SHOULD_ABSTAIN
HARD_NEGATIVE       -> SHOULD_ABSTAIN
REJECT              -> excluded
```

## 2. Contract correctness

**Supervised output schema (identical to the evaluation contract):**
```json
{"decision": "SHOULD_PROPOSE" | "SHOULD_ABSTAIN", "reasoning": "<string>"}
```
For B/C evidence heads, the contract additionally carries `"evidence_refs": ["E1", ...]` verified against the snapshot/Ports.

**Verification:** the evaluation fixture (`semantic-probe-results.json`, `benchmark-results.json`) expects exactly this `SemanticInterpretation` envelope; training targets use the **same** schema, so train/eval contract parity holds.

**Explicitly excluded — obsolete `DECISION:` format:** the M1 v1/v2 adapters were trained on the string form `DECISION: <SENSITIVE|ABSTAIN|NEGATIVE>`, which is NOT the `SemanticInterpretation` contract and was the root cause of their E0 failures. **6C.4 supervised targets MUST NOT use `DECISION:`.** Only the JSON contract above is permitted.

## 3. Evidence-relevance supervision

**Finding: a valid signal EXISTS in the frozen data** — every candidate record carries `grounding_snapshot.evidence_ids` (a subset of the available `concept_ids`/`belief_ids`/`rule_ids`). This yields a derivable evidence-relevance target: given the percept and the available symbols, the positive class = `evidence_ids`.

- Present for all 264+80 source records; non-trivial when distractors exist (`distractor_count`, `trap_propositions` fields confirm distractor cases where `evidence_ids` ⊂ available symbols).
- **No labels manufactured.** The signal is read directly from frozen `grounding_snapshot`; it is a *grounding-evidence* signal, not an independently human-labeled *relevance* judgment.
- **Minimum additional corpus if stronger supervision is desired:** human-confirmed relevance labels on a stratified sample of distractor/conflict cases (e.g., 50–100 cases). This is **optional**, not a blocker — the existing `evidence_ids` are sufficient to train a baseline evidence head.
- Related labels already present in the frozen candidate records (`novelty_label`, `decision_relevance`, `abstention_label`, `derivability_label`) could supervise richer heads but are **not required** for the decision/evidence gates.

## 4. Fair architecture comparison

| Dimension | Shared specification (A = B = C) |
|---|---|
| Training data | The 300 training-included records from `6c4_mapping_manifest.json` (same texts, same labels) |
| Split policy | Deterministic, leakage-safe: group by `capability_family`/case prefix; stratify by `final_status`; fixed seed `20260826`; e.g. 90/10 train/dev for early-stop only. **The 66-case fixture is NEVER in train/dev.** |
| Seed policy | `20260826` for any stochastic step; greedy/Deterministic decode at eval (same as prior evals) |
| Evaluation fixture | The frozen 66-case instrument (15-probe + 51-benchmark) + frozen populated-evidence corpus (`f99ca025…`); **untouched** |
| Behavioral gates | Identical §7 gates: E0 ≤2%, grounding 100%, Group E ≥80% (deciding), abstention ≥90%, E5/E6 (N/A on fixture), latency ≤0.5s, VRAM ≤2GB, v0.5/ADR-0028, shortcut ≤40% |
| Architecture specifics | A: decoder fine-tune on decision targets. B: decoder/decision head + reranker trained on evidence-relevance (`evidence_ids`). C: shared encoder + two heads trained on decision + evidence-relevance. All consume the SAME training set/split/seed. |

This guarantees the only variable is the architecture; data, splits, seed, fixture, and gates are held constant.

## 5. Training eligibility decision

**READY_FOR_SUPERVISED_EVALUATION** — with documented caveats:
- Mapping is auditable, 1:1, deterministic, hash-recorded; no text invented.
- Contract schema defined; train/eval parity holds; `DECISION:` format excluded.
- Evidence-relevance signal exists in frozen data (no manufacturing); optional human labels would only strengthen it.
- Fair-comparison spec (data/split/seed/fixture/gates) is fully definable and identical across A/B/C.

**Caveats (do not block; inform design):**
1. `evidence_ids` is grounding-evidence, not human-labeled relevance — relevance is only non-trivial on distractor cases.
2. E5/E6 cannot be scored on the 66-case fixture (no novelty labels there); they remain N/A on the fixture (consistent with prior evals).
3. The ≤0.5s latency gate is hardware-bound (0.5B decoder ≈3–5s/case here) and will fail independent of architecture quality — treat as a config/hardware decision, not a model-capability gate.

## 6. What this step did NOT do (constraints honored)
- Did **not** modify 300-GOLD, the 66-case fixture, model weights, labels, thresholds, or architecture definitions.
- Did **not** train A/B/C. Produced only: this plan, the independent audit, and the read-only `6c4_mapping_manifest.json`.

## 7. Deliverables
- `theo-core/docs/research/theo-slm-phase-6c4-supervised-alignment-plan.md` (this file)
- `theo-core/docs/research/theo-slm-phase-6c4-independent-audit.md`
- `theo-data/datasets/theo_slm_v0_review/6b3_batch2/6c4_mapping_manifest.json`
