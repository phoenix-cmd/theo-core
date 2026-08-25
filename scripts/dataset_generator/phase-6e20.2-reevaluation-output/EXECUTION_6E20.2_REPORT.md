# Phase 6E.20.2 — Authorized Execution Report

**Date:** 2026-08-19
**Command:** `python run_phase_6e20_reevaluation.py --execute`
**Authorization:** ACTIVE `execution_authorization_6e20_2.json` (hash `9CC01618E3CD1CDC2364A07DF53E7BE3A803DEAA2D284D2EAA0761717A6B4B2A`), runner bound to `16BEF31E…`, test `55A71B41…`.
**Final status:** **HALTED at Stage 5 — B0 historical reproducibility anchor failure.** Execution NOT complete.

---

## 1. Authorization Verification Result

**PASS.** `check_authorization()` matched the on-disk runner SHA-256 `16BEF31E56D781170021092052CF708A92875EE68697FEB81FB77764E15E691C` against the active authorization's `runner_sha256` (identical), `decision == APPROVE`, `phase == 6E.20.2`. Execution proceeded past the guard.

## 2. Stage-by-Stage Status

| Stage | Name | Status | Detail |
|-------|------|--------|--------|
| 1 | Pre-execution frozen-artifact SHA-256 manifest | **PASS** | All 28 checkpoints + frozen inputs (protocol, plan, 6E.18 snapshot, runner bases, auth, corpus, step0 `*.npy`) verified. |
| 2 | Model/tokenizer identity preflight (corrected vocabulary semantics) | **PASS** | `Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775` verified. **The remediated invariant `model.config.vocab_size == 151936` passed**; tokenizer metrics (base 151643, full 151665) checked separately. This confirms the prior Stage-2 halt defect is resolved. |
| 3 & 4 | Checkpoint deserialization & non-silent LoRA key validation | **PASS** | All 28 checkpoints verified with **336 active LoRA keys**; missing/zero-norm detection active. |
| 5 | Historical B0/S8 reproducibility anchor gate | **FAIL (HALT)** | `Stage 5 HALT: B0 reproducibility failure at step 4! Recomputed=0.418701, Historical=0.428355, Diff=9.654000e-03 > 0.0001`. Gate raises on first failure; steps 6/14/15 not reached. |
| 6 | Condition-resolved 28-checkpoint evaluation | **NOT RUN** | Halted before Stage 6. |
| 7 | Post-execution frozen-artifact integrity audit | **NOT RUN** | Halted before Stage 7. Read-only post-halt verification performed instead (see §8). |
| 8 | Protocol hypothesis aggregation & verdicts | **NOT RUN** | Halted before Stage 8. |

## 3. Stage 2 Corrected Invariant — PASSED

The Classification-B defect is resolved. The corrected gate compares `model.config.vocab_size` (151936) to `VOCAB`, and checks `tokenizer.vocab_size` (151643) / `len(tokenizer)` (151665) only for internal consistency. Stage 2 PASS confirms the remediated runner executes past the point that previously halted.

## 4. B0/S8 Anchor Values vs Expected

`HISTORICAL_B0_S8_ANCHOR = {4: 0.428355, 6: 0.409713, 14: 0.42747, 15: 0.41745}`, tolerance `1e-4`.

| Step | Expected cos_H1 | Recomputed cos_H1 | Diff | Result |
|------|-----------------|-------------------|------|--------|
| 4 | 0.428355 | 0.418701 | 9.654e-03 | **FAIL** (halt) |
| 6 | 0.409713 | — (not reached) | — | not evaluated |
| 14 | 0.42747 | — (not reached) | — | not evaluated |
| 15 | 0.41745 | — (not reached) | — | not evaluated |

The gate fails closed on the first divergence; therefore only step 4 was computed. The recomputed B0/S8 cos_H1 at step 4 diverges from the frozen historical value by ~9.65e-3, far outside the 1e-4 reproducibility tolerance. This is a genuine environment/evaluation-parity divergence on the B0/S8 path, caught by design — **independent of the Stage 2 remediation**.

## 5. All 28 Evaluation Results

**NOT PRODUCED.** Stage 6 (the 28 condition×aggregation evaluations) did not execute because execution halted at Stage 5.

## 6. rho_c and sigma_c per Condition

**NOT PRODUCED.** `compute_live_rho_c` (which yields `rho_c` and `sigma_c`) runs within Stage 6 and was not reached.

## 7. H1 / H3 / H4 / H5 Verdicts (Protocol v3.1)

**NOT PRODUCED.** `evaluate_protocol_hypotheses` (Stage 8) did not run. No hypothesis verdicts were generated.

## 8. Output File Inventory and Sizes

| File | Size (bytes) | Origin |
|------|--------------|--------|
| `pre_execution_sha256_manifest.json` | 14,557 | Stage 1 (pipeline artifact) |
| `checkpoint_smoke_test_report.json` | 16,869 | Stage 3 & 4 (pipeline artifact) |
| `execution_run.log` | 4,502 | Operator shell redirect (not a pipeline artifact) |

No `post_execution_sha256_manifest.json`, no `hypothesis_verdicts.json`, and no per-condition telemetry were written (consistent with halt before Stages 6–8).

## 9. Post-Execution SHA-256 / Frozen-Integrity Result (read-only post-halt verification)

Stage 7 did not run (halt before). A **read-only** post-halt integrity check was performed: every file recorded in `pre_execution_sha256_manifest.json` was re-hashed from disk and compared to the Stage-1 recorded values.

- All **28 checkpoint** hashes: **byte-identical** to Stage-1 manifest (28/28 consistent).
- All frozen core inputs (protocol, plan_v40, frozen_6E18_snapshot, runner_6e20_base, runner_6e20_1_v7, auth_6e20_1_v7/v6, corpus) and all step0 `*.npy`: **match=True**.
- Frozen anchor re-confirmation: `protocol = E16CA4F1…`, `plan_v40 = 87C8E486…`, `frozen_6e18 = 6BCAF131…` (unchanged).

**Result: all protected inputs remain byte-identical post-halt.**

## 10. Protected 6E.20.1 Artifacts Byte-Identical?

**YES.** Verified via the post-halt re-hash above. No 6E.20.1 checkpoint, step0 artifact, protocol, plan, 6E.18 machinery, or provenance file was modified by this (partial) execution. The only newly written files are confined to `phase-6e20.2-reevaluation-output/` (manifest, smoke-test report, and the operator log).

## 11. Final Execution Status

**HALTED at Stage 5 — B0 historical reproducibility anchor failure.** The authorized execution proceeded normally through Stages 1–4, the Stage-2 vocabulary remediation was confirmed working (Stage 2 PASS), but a legitimate fail-closed B0/S8 reproducibility divergence (step-4 diff 9.654e-3 ≫ 1e-4) halted the pipeline before the 28 evaluations, post-execution integrity audit, and verdict aggregation. No retry, patch, setting change, artifact substitution, or gate bypass was performed. All frozen artifacts remain byte-identical.

**No follow-up execution, remediation, or new authorization action taken. Awaiting further instruction.**
