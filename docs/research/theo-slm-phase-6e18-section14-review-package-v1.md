# THEO Phase 6E.18 — Section 14 Sign-Off Review Package (v1, amended for O-1/O-2)

- **Purpose:** One consolidated package enabling the §14 human review to answer: *"Does the complete frozen protocol → implementation → preflight chain support §14 sign-off without unresolved contradictions or authorization-boundary violations?"*
- **Status:** Review input only. **No sign-off is issued by this document.** Execution authorization remains NONE.
- **Date:** 2026-08-16. **Amendment cycle:** O-1 and O-2 (from the prior v1 review) were resolved per the reviewer's AMEND decision; this package records the dispositions and the amended frozen state. The runner was re-verified (self-test + 11-step preflight) after the amendments.
- **Prior v1 package hash:** `A1132B7B1055FA583030522D31938C0ED7280D870CDB4B6A4994B82823C782B9` (superseded by this amended package).

## 1. Frozen State Record

| Artifact | SHA-256 | State |
|---|---|---|
| Protocol v2.1.3 (FINAL, review target) | `910796C879BBAFBA80C0D6829D382723CC4E89425003380FD4CC94E1E21312C9` | UNCHANGED (285 lines) |
| Protocol v2.1.2 (frozen, never edit) | `D13576BA019350266CC4551361CA07C0B20FE32CF462D768751461772B97EAF6` | UNCHANGED (243 lines) |
| Runner (pre-amendment build) | `322FC867086A74893C17D4D6E7AB126AD01CD020BE52072C8F6BEA8171F66501` | SUPERSEDED (2260 lines) |
| Runner (post-amendment, O-1/O-2 applied) | `6BCAF1319AC551848FFDB55AEA121D14D808408A192E05F2AA6171161DECEF96` | FROZEN (2560 lines) |
| 7-gate audit report | `299EEDFA5F2DF2885EDE277A6BB23B54BDBFF44EBA84C360C7C85E590B9A2D68` | UNCHANGED (89 lines) |
| Runner conformance/reconciliation v1 | `A982F26A0991BFE5692804534D99E6E0B8E541F8F1F6862247349FBDB2DD803E` | UPDATED this session (runner hash, O-1/O-2 evidence, §5/§6) |
| Execution-readiness report v1 | `2DA7BC2A2FCF8AB31C5297D4B1C1DEA658AA2E42C0E735033E632FADE381957C` | UPDATED this session (runner hash, O-1/O-2 evidence) |
| 6E.17 provenance | `runs_partial_B.json`, `B0_fingerprint_gate.json`, `verdict.json` in `phase-6e17-causal-intervention/` | PRESERVED, unmodified (§12) |

- 6E.18 artifact directory: **does not exist** (pristine). `execution_authorization.json`: **absent**.
- `--execute` was re-verified to refuse after the amendments (exit 1, `PRE_EXECUTION / NO_SCIENTIFIC_VERDICT / EXECUTION_NOT_AUTHORIZED`); the smoke-test directory it created was removed. The refusal manifest's `script_sha256` matched the post-amendment runner hash — self-consistent.

## 2. Chain Reconstruction and Evidence

```
Protocol v2.1.3 (910796C8, FINAL)
   ↓  7-gate literal-artifact audit (299EEDFA): PASS, no BLOCKING/MAJOR, 3 MINOR corrected
Runner conformance (69564355; amended this cycle → current A982F26A): 10/10 preflight + 25/25 targeted + 36/36 M-matrix PASS
   ↓  execution-phase build (runner 322FC867, 2260 lines)
Preflight 11/11 PASS (incl. §6.2 sign gate, §8 k∈[2,8], §3.8 init, V1 mirror, M-rules, zero-execution guard, verdict safeguard)
   ↓  zero-step self-test PASS (mock-backed; data, metrics, gradients, checkpoint routing + readout hash, 7-condition factory, P/M rules, verdict double gate)
Unauthorized --execute refusal PASS (exit 1, exact PRE_EXECUTION triplet, correct manifest)
   ↓  §14 REVIEW #1  →  DECISION: AMEND (O-1 + O-2 REQUIRED)
   ↓  AMEND CYCLE: O-1 (B0-only collapse guard + explicit VOID) and O-2 (POS/ABS accuracy + deterministic §5.2 row 7) implemented
   ↓  runner re-hashed 322FC867 → 6BCAF131 (2560 lines); targeted self-test sections added
Zero-step self-test re-run PASS (incl. new o1_* and o2_* sections) → preflight 11/11 re-run PASS
Unauthorized --execute refusal re-run PASS (exit 1, exact PRE_EXECUTION triplet, manifest script_sha256 = 6BCAF131)
   ↓  STOP HERE
§14 human re-review  ←  THIS PACKAGE (amended)
Separate explicit execution authorization
   ↓
ONLY THEN experimental execution
```

## 3. Conformance Cross-Check Results

Every requirement below was re-verified against the protocol text and the runner source during this review:

| # | Protocol requirement | Runner implementation | Status |
|---|---|---|---|
| 1 | §6.2 aggregate cos(H1) is the sole decision-rule input; mean-pairwise telemetry-only | `compute_cos_H1`; all P/M rules consume aggregate | CONFORMANT |
| 2 | ρ_CD = max(0, −agg) (audit note 6E18A-N1) | `compute_raw_gradient_cosine` L422–423 uses aggregate | CONFORMANT (resolved) |
| 3 | P1 ≥ −0.30; C2x ≥ +0.25; P3 d≥0.20/r_eff≤1.60; P4 [0.65,1.0]±0.10; P5 ≥0.30 | `P1_THRESH=-0.30`, `C2X_THRESH=0.25`, P3 0.20/1.60, P4 range/tol, `P5_THRESH=0.30` | CONFORMANT |
| 4 | M1–M5 dispositions, checkpoints {4,6,14,15}, mean aggregation, missing→VOID, M3 I1-void→NOT EVALUABLE, k∉[2,8]→VOID | `evaluate_m_rules` with `i1_void`, `t2_k` | CONFORMANT |
| 5 | §7.1 P4 comparator: {0,4,6} contemporaneous B0; {14,15,20} step-0 anchor; early-truncation fallback; missing→UNAVAILABLE (no implicit 0.0) | `evaluate_p4_comparator` + `evaluate_p_rules` P4 block | CONFORMANT (this session's fix) |
| 6 | T2 = k+1 params (§3.6/§9) | `T2ReadoutHead` (beta:k + b:1); fixture k=4→5 and §8 fixture k=3→4 both consistent | CONFORMANT |
| 7 | §9 readout accounting 0/897/897/0/2/k+1/1794 | `verify_condition_parameter_accounting` + preflight | CONFORMANT |
| 8 | §3.8 s_scale/b0/b0_c2/b_fixed/b_delta; V1 aligned mirror; T1 α(0)=1,b(0)=0; T2 β(0)=0,b(0)=0 | `compute_init_constants`, `derive_readout_initialization`, `V1ReadoutHead` | CONFORMANT |
| 9 | §8 SVD basis, energy ≥0.90, k∈[2,8] asserted pre-T2 | `derive_t2_basis_from_covariance` + `t2_basis.json` | CONFORMANT |
| 10 | §10 checkpoint set {0,4,6,7,14,15,17,20}; B0 truncation bound t ≤ 6 | `CHECKPOINTS`, guard break after step-6 checkpoint save | CONFORMANT |
| 11 | §6.1 items 4–9 runtime metrics (proposal rate, Cohen's d, r_eff, Resub k-NN1, LOO-CV, policy degradation) | `proposal_rate`, `cohens_d`, `compute_effective_rank`, `separability_suite`, `policy_degradation` | CONFORMANT |
| 12 | §6.2 synthetic sign gate, ε=0.001, VOID on failure | preflight step + `run_self_test` | CONFORMANT |
| 13 | 6E.17 parity: data split (seed 42, family 80/20, 134/67/67, panels 8/8/8), λ=10, grad_accum 2, batch 4, AdamW 1e-4/wd 0.01/clip 1.0, LoRA r16 α32 | `reconstruct_data`, `SFTDataset`, `compute_weighted_loss`, `run_condition` | CONFORMANT |
| 14 | Readout params trained (6E.17 readout-never-trained bug must not recur) | `assemble_trainable` + per-condition `readout_state_sha256` | CONFORMANT |
| 15 | §11 forensic packaging: `sha256_manifest.json`, `runs.json`, `pre_registration_snapshot.json` | `generate_sha256_manifest`, `_run_execution` | CONFORMANT |
| 16 | §14 hard stop; no verdict without authorization AND executed checkpoints | `emit_verdict` double gate; refusal path verified | CONFORMANT |
| 17 | §12 6E.17 provenance preserved | files present, unmodified | CONFORMANT |
| 18 | §5.2 outcome matrix rows 1–6 | `summarize_outcome` STRONG_H_CAUSAL / CAPACITY_EXPLANATION / H0_SURVIVES / H_CAUSAL_NOT_DECISIVE / H_CAUSAL_PARTIALLY_SUPPORTED + M3_M2_COMPOSITE / M4 | CONFORMANT (rows 1–6) |
| 19 | Base model revision `7ae557604adf…`; corpus binding; protocol binding | `EXPECTED_BASE_HASH`, `EXPECTED_CORPUS_HASH`, `EXPECTED_PROTOCOL_HASH` asserted pre-run | CONFORMANT |
| 20 | §10 collapse truncation scoped to B0 only; non-B0 collapse recorded and trajectory continues; cannot-continue → explicit VOID (never silent B0 semantics) | `collapse_guard_action` (TRUNCATE/CONTINUE/NO_TRIGGER) + `run_condition`; `status=VOID`/`void_reason` → rule evaluators mark VOID conditions NOT EVALUABLE | CONFORMANT (O-1 resolved, this amendment) |
| 21 | §5.2 row 7 machine-detectable; §6.1 item 9 POS/ABS accuracy registered; indeterminate → `ROW_7_NOT_DETERMINABLE_FROM_REGISTERED_METRICS`; no invented threshold | `evaluate_checkpoint` `panel16_pos_accuracy`/`panel16_abs_accuracy`/`pos_abs_accuracy`; `evaluate_row7` (strict-sign, anchors {0,4,6,14,15}); verdict + `outcome_summary` always surface the row | CONFORMANT (O-2 resolved, this amendment) |

## 4. Amendments Resolved (from §14 Review #1 → AMEND decision)

### O-1 — Collapse-guard scope: RESOLVED (scoped to B0, per the reviewer's AMEND requirement)

- **Disposition applied:** guard truncation is restricted to **B0 only**. Non-B0 conditions follow their explicit lifecycle (run to horizon); a cannot-continue condition is recorded as an explicit **VOID** state — never silently truncated under B0 collapse semantics.
- **Implementation (runner `6BCAF131`):**
  - `collapse_guard_action(condition, ever_proposed, consecutive_zero)` — pure decision helper: `NO_TRIGGER` (pattern absent), `TRUNCATE` (pattern present AND condition == "B0"), `CONTINUE` (pattern present on any non-B0 condition). `run_condition` truncates only on `TRUNCATE`.
  - Non-B0 collapse: sets `collapsed=True` (feeds P2's "no proposal collapse" rule → the §5.2 designed P2-fails path), records `non_b0_collapse_continued=True` at run level and `collapse_pattern=True` in the step telemetry, and continues to the full horizon so checkpoints 14/15/20 exist.
  - Numerical-invalid cannot-continue: non-finite training loss → `status="VOID"`, `void_reason="non_finite_loss"`, a void telemetry record, and early exit. Downstream, `build_metrics_store` carries `status`/`void_reason`/`non_b0_collapse_continued`; `evaluate_p_rules` marks every I1-dependent rule `evaluable=False` with detail `"I1 VOID: …"` (P4 additionally `not_evaluable=True`), C2x also handles a VOID C2; M3 → NOT EVALUABLE via the existing `i1_void` path.
- **Verification (zero-step self-test, new `o1_collapse_guard_scoping` + `o1_void_state_propagation` sections):** B0 with the pattern → `TRUNCATE`; I1 and C2 with the same pattern → `CONTINUE`; never-proposed → `NO_TRIGGER`; full store path shows an I1 run that recorded the collapse and still has checkpoints 14/15/20 (`collapsed=True`, `non_b0_collapse_continued=True`, `t*=5`) with **P1/P3/P5 evaluable and passed** while **P2 fails**; a VOID I1 → all I1-dependent rules NOT EVALUABLE, P4 `not_evaluable=True`, M3 NOT EVALUABLE, `collapsed=False` and `stopping_event=None` (explicitly not a B0-style collapse).

### O-2 — §5.2 row 7 machine-detectable: RESOLVED (per the reviewer's AMEND requirement)

- **Disposition applied:** POS/ABS accuracy is recorded at the required checkpoints; §5.2 row 7 ("One class improves while other degrades → Artifactual trade-off; NOT accepted as conflict resolution") is evaluated deterministically from **existing protocol language only** — no invented scientific threshold; indeterminate cases are explicit and Row 7 is never dropped.
- **Implementation (runner `6BCAF131`):**
  - `evaluate_checkpoint` emits per-class policy metrics (§6.1 item 9) over the N=16 panel using the registered proposal decision (margin > 0 predicts POS, margin ≤ 0 predicts ABS): `panel16_pos_accuracy`, `panel16_abs_accuracy`, and the balanced `pos_abs_accuracy`. These flow into checkpoint records → `runs.json` (hashed by `sha256_manifest.json`) and into `build_metrics_store` → the verdict bundle.
  - `evaluate_row7(metrics)` compares the I1 per-class accuracy deltas between consecutive registered anchors `{0, 4, 6, 14, 15}` (t=15 is the §6.1 item 9 policy anchor) using **strict sign comparisons only** (`> 0.0` / `< 0.0`; no magnitude threshold). Labels: `ARTIFACTUAL_TRADE_OFF` / `NO_ARTIFACTUAL_TRADE_OFF` / `ROW_7_NOT_DETERMINABLE_FROM_REGISTERED_METRICS` (VOID condition or missing registered checkpoint, with the specific gap documented).
  - Wiring: `compute_verdict` emits `Row7_artifactual_trade_off` and `summarize_outcome` always appends the Row-7 label to `outcome_summary.rows` — the row never disappears.
- **Verification (zero-step self-test, new `o2_row7_deterministic` section):** accuracy present at {0}+RULE_CHECKPOINTS incl. 15; flat accuracy → `NO_ARTIFACTUAL_TRADE_OFF`; POS-improving + ABS-degrading → `ARTIFACTUAL_TRADE_OFF` (interval 0→4); tiny opposite-sign deltas (+0.01/−0.01) still trigger (proves sign-only, no magnitude threshold); VOID I1 → NOT DETERMINABLE; checkpoint 15 recorded without accuracy → NOT DETERMINABLE with `missing_checkpoints=[15]`; both trade-off and flat fixtures surface their Row-7 label in `verdict.json` and `outcome_summary.rows`.

## 5. Authorization-Boundary Assessment

- **Verified boundary enforcement:** `--execute` refuses without `execution_authorization.json` (re-verified after the amendments: exit 1, exact `PRE_EXECUTION / NO_SCIENTIFIC_VERDICT / EXECUTION_NOT_AUTHORIZED`, correct manifest with `script_sha256 = 6BCAF131…`). No training, optimizer step, checkpoint, or scientific verdict produced in this build phase (zero-step proof in readiness report §14).
- **No verified contradictions:** every audited protocol requirement re-checked above is CONFORMANT (now 21 rows) with no conflicting thresholds, checkpoint sets, parameter counts, or disposition semantics.
- **Prior boundary question O-1 (guard scope) is RESOLVED** by the amendment: truncation is now B0-only per §10, non-B0 trajectories follow their explicit lifecycle, and cannot-continue conditions are explicitly VOID.

## 6. Recommendation

The frozen protocol → implementation → preflight chain is **internally consistent and CONFORMANT** on all 21 audited requirements. The two open items from §14 Review #1 were resolved by the AMEND decision and are implemented and verified:

- **O-1** (collapse guard scoped to B0 + explicit VOID for cannot-continue) — resolved, verified.
- **O-2** (§5.2 row 7 machine-detectable via registered POS/ABS accuracy; no invented threshold; explicit indeterminate) — resolved, verified.

**No §14 decision is rendered here.** The reviewer's options are:
- **APPROVE** (protocol v2.1.3 + amended runner ready), or
- **AMEND** (additional amendments required), or
- **REJECT** (no evidence supports this; all findings are minor/contained and the prior two are resolved).

Independent of the §14 decision, execution still requires a **separate, explicit execution authorization** (`execution_authorization.json`) afterward.

## 7. Sign-off Form (to be completed by the human reviewer)

- **Reviewer Name:** ____________________
- **Decision:** [ ] APPROVE   [ ] AMEND   [ ] REJECT
- **O-1 / O-2 dispositions (implemented and verified in this package):** [ ] accept as resolved
- **Date:** ____________________
- **Signature:** ____________________
