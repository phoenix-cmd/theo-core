# THEO Phase 6E.18 — Execution-Readiness Report v1

- **Scope:** Final execution-phase build report for the 6E.18 causal-intervention runner, covering harness, §6.1 runtime metrics, P/M/C2x rule wiring, manifest/verdict emission, §14 authorization gate, and the zero-step verification proof. Build-only; **no execution.**
- **Governing spec:** `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.3.md` (FINAL).
- **Runner:** `theo-core\scripts\dataset_generator\run_phase_6e18_causal_intervention.py`
- **Date:** 2026-08-16.

## 1. Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| Protocol v2.1.2 (frozen, unchanged) | `D13576BA019350266CC4551361CA07C0B20FE32CF462D768751461772B97EAF6` |
| Protocol v2.1.3 (FINAL, audit target) | `910796C879BBAFBA80C0D6829D382723CC4E89425003380FD4CC94E1E21312C9` |
| Runner (execution-phase build) | `322FC867086A74893C17D4D6E7AB126AD01CD020BE52072C8F6BEA8171F66501` (2260 lines) |
| Runner (post-amendment, O-1/O-2 applied) | `6BCAF1319AC551848FFDB55AEA121D14D808408A192E05F2AA6171161DECEF96` (2560 lines) |
| Base model `model.safetensors` (frozen at execution) | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` (bound via `EXPECTED_BASE_HASH`) |
| Corpus `candidate_records.json` (frozen at execution) | `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0` (bound via `EXPECTED_CORPUS_HASH`) |
| §14 review package (amended, reviewed) | `DEF8C4CD339A98D2217640EAE4CD2B31AD72D2CCC7E6D856D55D9DAAF0DFCB5A` (120 lines) |
| §14 sign-off record (APPROVED) | `00727716A0B88070CA22B0F45665DC5B7F37D9F53CF96852C23F63927DD877D3` (75 lines) |

The runner binds to all three expected hashes; `--execute` refuses if any mismatch (protocol assert, base-model assert, corpus assert).

## 2. Protocol↔Runner Conformance

All previously audited conformance items remain CONFORMANT (see `theo-slm-phase-6e18-runner-conformance-reconciliation-v1.md`), now extended by the execution-phase build:

| Protocol requirement | Runner implementation | Status |
|---|---|---|
| §6.1 items 1–9 runtime metrics | `compute_cos_H1`, `compute_raw_gradient_cosine`, `compute_effective_rank`, `get_top1_dominance`, `proposal_rate`, `cohens_d`, `resub_knn1_purity`, `loo_cv_accuracy_signed`, `policy_degradation`, `separability_suite`, `batched_decision_margins`, `per_sample_margin_jacobian` | CONFORMANT (verified via synthetic fixtures) |
| §6.1 item 9 POS/ABS accuracy (policy metric) | `evaluate_checkpoint` `panel16_pos_accuracy` (frac POS margins > 0), `panel16_abs_accuracy` (frac ABS margins ≤ 0), `pos_abs_accuracy` (balanced); surfaced via `build_metrics_store` | CONFORMANT (O-2 amendment, verified) |
| §10 collapse truncation B0-only + explicit VOID | `collapse_guard_action` (TRUNCATE/CONTINUE/NO_TRIGGER) + `run_condition`; non-B0 collapse recorded + continues; non-finite loss → `status=VOID`/`void_reason`; rules mark VOID conditions NOT EVALUABLE | CONFORMANT (O-1 amendment, verified) |
| §5.2 row 7 artifactual trade-off | `evaluate_row7` (strict-sign deltas over anchors {0,4,6,14,15}, no invented threshold; `ROW_7_NOT_DETERMINABLE_FROM_REGISTERED_METRICS` on gap; always surfaced in verdict + outcome_summary) | CONFORMANT (O-2 amendment, verified) |
| §7 P1–P5/C2x thresholds | `evaluate_p_rules` with constants `P1_THRESH=-0.30`, `C2X_THRESH=0.25`, `P3_COHEN_D_THRESH=0.20`, `P3_R_EFF_THRESH=1.60`, `P4_KN1_RANGE=(0.65,1.0)`, `P4_LOO_TOL=0.10`, `P5_THRESH=0.30` | CONFORMANT (matches §7 table exactly) |
| §7.1 P4 comparator + §10 truncation | `evaluate_p4_comparator` + `evaluate_p_rules` §7.1 universe `{0,4,6,14,15,20}`, step-0 anchor, early-truncation fallback, UNAVAILABLE recording | CONFORMANT (see §8) |
| §7.2 M1–M5 dispositions | `evaluate_m_rules` (SUPPORTED / NOT SUPPORTED / INDETERMINATE / NOT EVALUABLE / VOID) | CONFORMANT (verified) |
| §3.8 init arithmetic, V1 aligned mirror, §8 T2 basis | `compute_init_constants`, `derive_readout_initialization`, `derive_t2_basis_from_covariance` | CONFORMANT (verified) |
| 6E.17 data/loss/optimizer parity | `reconstruct_data`, `SFTDataset`, `data_collator`, `compute_weighted_loss`, `run_condition` | CONFORMANT (parity re-read; see §5) |
| §11 manifest hashing | `generate_sha256_manifest` | CONFORMANT (verified) |
| §14 hard stop / verdict safeguard | `check_execution_authorization`, `emit_verdict` double gate | CONFORMANT (verified end-to-end) |

## 3. §6.1 Metric Implementation Mapping

| §6.1 item | Metric | Implementation | Notes |
|---|---|---|---|
| 1 | Aggregate $\cos(H1)$ | `compute_cos_H1` → aggregate cosine over decision gradients at last-token decision position | Sole decision-rule input (aggregate); mean-pairwise emitted as telemetry only |
| 2 | $\rho_{CD} = \max(0, -\text{agg})$ | `compute_raw_gradient_cosine` / `evaluate_checkpoint` | Aggregated form |
| 3 | $F$ | fraction of POS×ABS dot products > 0 at checkpoint | Emitted per checkpoint |
| 4 | Proposal rate $P(t)$ | `proposal_rate` = fraction dev margins > 0 | Computed on dev set, per step |
| 5 | Cohen's $d(t)$ | `cohens_d` = (mean POS − mean ABS)/pooled sd, pooled sd = √((var POS + var ABS)/2), panels first-8 POS / last-8 ABS | Degenerate zero-variance panel is finite (eps floor) |
| 6 | $r_{\text{eff}}(t)$ | `compute_effective_rank` = tr²/tr2 of 16×16 margin-Jacobian Gram (`per_sample_margin_jacobian`) | Computed at t=0 and t=15; P3 uses $r_{\text{eff}}(15)$ |
| 7 | Resub k-NN1 | `resub_knn1_purity` / `separability_suite` (layer-24 last-token $h_L$, POS-vs-ABS) | P4 |
| 8 | LOO-CV($t$) | `loo_cv_accuracy_signed` (k=1 LOO) / `separability_suite` | P4/§7.1 comparator |
| 9 | Policy metrics / degradation | `policy_degradation` = $P(15) \le 0.0$ OR $d(15) < 0.00$; POS/ABS accuracy (balanced) + per-class accuracy recorded at every checkpoint | H5/M4; §5.2 row 7 (`evaluate_row7`) |

## 4. Determinism and Seeds

- Data split: `reconstruct_data` with `SEED_DATA=42` — family-level 80/20 train/dev split, balanced 134 POS / 67 ABS / 67 NEG, panels 8/8/8 (6E.17 parity).
- C2 random orientation: `generate_c2_random_orientation` with `SEED_C2_READOUT=20260816`, deterministic rejection sampler enforcing `|cos(n, u_hat_LDA)| ≤ 0.05`.
- LDA: `solver="lsqr", shrinkage="auto"` (scikit-learn; 6E.17 parity).
- §8 T2 basis: deterministic SVD of $\Sigma_{diff}$, minimum $k$ s.t. cumulative energy ≥ 0.90, asserted `k ∈ [2, 8]`; basis + hash frozen before T2 execution (`t2_basis.json`).
- `MockTokenizer`/`MockBackbone` (self-test only) are fully deterministic (SHA-256-derived tokens).

## 5. Data, Loss, Optimizer, and Training Loop (6E.17 Parity)

- Loss: weighted CE, `LAMBDA_DECISION=10` on the decision token via readout `forward_logits` patch (generalized to all readout heads); `GRAD_ACCUM=2`, `BATCH_SIZE=4`, `MAX_LEN=512`.
- Optimizer: AdamW `LR=1e-4`, `WD=0.01`, `CLIP_NORM=1.0`.
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`, targets `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`; body bf16, LDA/s_scale arithmetic fp32.
- **6E.18 critical fix:** readout parameters are included in the optimizer (`assemble_trainable`); per-condition `readout_state_sha256` is recorded in every checkpoint (6E.17 forensics: identical steps 0→20, readout never trained).
- Checkpoint set `[0, 4, 6, 7, 14, 15, 17, 20]`; `HORIZON=20` (I1/C2 = 32); B0 collapse guard per §10.

## 6. The Seven Conditions and Initialization Paths

| Condition | Description | Init path | Trainable params |
|---|---|---|---|
| B0 | Frozen baseline | `a0 = s_scale·u_hat_LDA`, `b0 = -median(a0@H16[:8].T)`, `b_fixed = b0` | 0 |
| I1 | Aligned adaptive readout | `a0`, `b0` | 897 |
| C2 | Random orientation | `c0 = s_scale·n` (random `n`, seed 20260816), `b0_c2 = -median(c0@H16[:8].T)` | 897 |
| F1 | Fixed geometry | `a0`, `b0` (frozen) | 0 |
| T1 | 1D gain | `alpha(0) = 1.0`, `b(0) = 0.0` | 2 |
| T2 | Low-rank SVD basis | `beta(0) = zeros(k)`, `b(0) = 0.0`, `k ∈ [2,8]` (§8) | 5 |
| V1 | Unconstrained 2-logit (aligned mirror) | `w_PRO(0)=a0`, `w_ABS(0)=-a0`, biases 0 (exact mirror, `cos=1.0`, `||w_PRO(0)||=s_scale`) | 1794 |

`derive_readout_initialization` is execution-time step-0 derivation (frozen-model fp32 inference) and is **not invoked in preflight/self-test** (zero-execution guard).

## 7. P-Rule and C2x Wiring (§7 / P-rule table)

Evaluated on condition I1 (C2x on I1 vs C2), mean over `RULE_CHECKPOINTS=[4,6,14,15]`:

| Rule | Criterion (protocol) | Runner constant | Notes |
|---|---|---|---|
| P1 | mean cos(H1)(I1) ≥ −0.30 | `P1_THRESH=-0.30` | |
| C2x | mean(I1) − mean(C2) ≥ +0.25 | `C2X_THRESH=0.25` | Mandatory alignment exclusion |
| P2 | proposal rate > 0.0% at all t ≤ 20 | `P2_THRESH=0.0`, telemetry `dev_proposal_rate` per step | Collapsed flag from B0/I1 run metadata |
| P3 | d(15) ≥ +0.20 AND r_eff(15) ≤ 1.60 | `P3_COHEN_D_THRESH=0.20`, `P3_R_EFF_THRESH=1.60` | margin-Jacobian Gram at t=15 |
| P4 | Resub k-NN1 ∈ [0.65, 1.0]; LOO-CV within ±0.10 of §7.1 comparator | `P4_KN1_RANGE`, `P4_LOO_TOL=0.10` | Comparator per §8 |
| P5 | max |cos(w_readout, u_hat_LDA)| ≥ 0.30 for t ≥ 4 | `P5_THRESH=0.30` | |

## 8. P4 Dual-Regime Comparator and §10 Truncation

`evaluate_p4_comparator` implements §7.1 exactly; `evaluate_p_rules` iterates the §7.1 universe `{0,4,6,14,15,20}`:

- **Shared evaluable checkpoints (t ∈ {0,4,6}):** `|LOO-CV_I1(t) − LOO-CV_B0(t)| ≤ 0.10` (contemporaneous).
- **Post-truncation (t ∈ {14,15,20}):** frozen step-0 anchor `|LOO-CV_I1(t) − LOO-CV_B0(0)| ≤ 0.10`.
- **Early-truncation fallback:** if `LOO-CV_B0(t)` is missing for t ∈ {0,4,6} (B0 guard triggered before evaluation), the step-0 anchor applies and the missing matched comparator is recorded as UNAVAILABLE in the manifest.
- Checkpoints outside the §7.1 universe (7, 17) are excluded from the comparator (not counted as unavailable).
- If no LOO pairs are evaluable at all, P4 is reported NOT EVALUABLE (`not_evaluable=True`).

## 9. Collapse Guard (§10)

- **Trigger (6E.17 dev-proposal parity):** the K-consecutive-zero detector fires when `ever_proposed` is true AND `COLLAPSE_K=3` consecutive steps have dev proposal rate 0.0%.
- **B0-only truncation (O-1 amendment):** `collapse_guard_action` returns `TRUNCATE` for B0 (evaluation ceases per §10; guard break) and `CONTINUE` for every other condition. Non-B0 runs record `non_b0_collapse_continued=True` + a `collapse_pattern` telemetry flag + `collapsed=True` (feeds P2's "no proposal collapse" rule, §5.2 P2-fails path) and continue to the full horizon, keeping checkpoints 14/15/20 for P1/P3/P5.
- **t\*:** first zero-after-positive step; `collapse_onset_t_star` is recorded for every condition.
- **VOID state (cannot-continue):** non-finite training loss → `status="VOID"`, `void_reason="non_finite_loss"`, void telemetry record; rule evaluators mark that condition's rules NOT EVALUABLE (P-rules `evaluable=False`, P4 `not_evaluable=True`; M3 `i1_void` → NOT EVALUABLE). A VOID run is never marked as a B0-style collapse (`collapsed=False`, `stopping_event=None`).
- LoRA grad-norm is recorded as **telemetry only** — it is not a collapse trigger.
- Truncated conditions' post-truncation checkpoints are absent from the store, which drives the P4 anchor/fallback and M-rule VOID semantics (§7.2: any missing checkpoint → VOID).

## 10. Checkpoint Routing and Hashing

- `save_checkpoint` persists LoRA (fp16) + readout (fp32) with `checkpoint_state_sha256` AND `readout_state_sha256` in each checkpoint meta (per-condition).
- `load_checkpoint` restores via `lora_*` keys + `__readout__{name}` keys.
- Self-test verified: perturbing readout params changes `readout_state_sha256`; restore matches step-0 state to `<1e-6`.
- `evaluate_checkpoint` emits: margins, cos(H1) aggregate + mean-pairwise, rho_CD, F, separability N16/N24 (resub k1/k3/k5, LOO k1/k3, SVM-CV), `lora_grad_norm`, `readout_cos_lda`/norm, margin-Jacobian Gram + r_eff/top1 (t=0,15), POS/ABS accuracy (`panel16_pos_accuracy`, `panel16_abs_accuracy`, `pos_abs_accuracy`).

## 11. Manifest Schema and Hashing

- `pre_registration_snapshot.json`: phase, timestamp, base/corpus/protocol/script SHA-256, authorization record, full frozen config (hyperparameters, LoRA, rule thresholds).
- `readout_initialization.json`: §3.8 derivation (`derive_readout_initialization`) incl. `w_delta_t` stash for T1.
- `t2_basis.json`: k, cumulative energy, spectrum top-10, `basis_sha256`.
- `runs.json`: `_clean`-serialized per-condition runs (checkpoints + telemetry).
- `verdict.json`: see §12.
- `sha256_manifest.json`: phase, timestamps, protocol/script hashes, elapsed seconds, and SHA-256 for **every** artifact file in the artifact directory (recursive, defensive).
- `code_snapshot/`: byte copy of the runner at execution time.

## 12. Verdict Safeguards (§14 Hard Stop)

- `check_execution_authorization` requires `execution_authorization.json` with `decision=APPROVE`, signature, date, `phase=6E.18` in the artifact directory.
- `emit_verdict` has a **double gate**: (a) signed authorization present AND (b) executed checkpoints exist. Otherwise the verdict is exactly `PRE_EXECUTION / NO_SCIENTIFIC_VERDICT / EXECUTION_NOT_AUTHORIZED`.
- `--execute` refusal path verified end-to-end (post-amendment): absent authorization → exit 1, exact triplet emitted, `verdict.json` + `sha256_manifest.json` written (protocol hash + `script_sha256 = 6BCAF131…`), no training/optimizer/checkpoint/verdict. Smoke-test directory removed afterward; real artifact dir left pristine.

## 13. Verification Evidence (zero-step, mock-backed)

| Verification | Result |
|---|---|
| `py_compile` | OK |
| Preflight (`--preflight`, 11 steps) | **11/11 PASS** — protocol hash binding; 7-condition universe; §6.2 aggregate sign gate; param accounting 0/897/897/0/2/5/1794; §7.1 P4 dual-regime + fallback; §8 SVD basis (k=3); §3.8 init arithmetic + invariants; V1 aligned mirror; M1–M5; zero-execution guard; verdict-emission/authorization-gate safeguard |
| Zero-step self-test (`--self-test`) | **PASS** — data layer (67/67/134, panels 8/8/8); §6.1 pure metrics incl. Cohen's d; gradient flow + readout connectivity; checkpoint routing + readout hash change + restore; all-7-condition factory; P1–P5/C2x incl. §7.1 P4 universe/anchor/UNAVAILABLE; M1–M5 (M5 NOT SUPPORTED for C2=−0.60 < −0.30 floor); verdict double gate; protocol hash. **Amendment sections:** O-1 (B0→TRUNCATE, I1/C2→CONTINUE, non-B0 collapse recorded + checkpoints 14/15/20 retained → P1/P3/P5 evaluable, P2 fails; VOID → rules NOT EVALUABLE, M3 NOT EVALUABLE, not collapsed) and O-2 (POS/ABS accuracy emitted at {0}+RULE_CHECKPOINTS incl. 15; flat→NO_ARTIFACTUAL_TRADE_OFF; improving-POS/degrading-ABS→ARTIFACTUAL_TRADE_OFF; tiny ±0.01 signs → sign-only no-threshold; VOID/missing→ROW_7_NOT_DETERMINABLE_FROM_REGISTERED_METRICS; Row-7 always surfaced) |
| `--execute` refusal smoke test | **PASS** (exit 1, exact PRE_EXECUTION triplet, correct manifest; `script_sha256 = 6BCAF131…`) |
| 6E.17 parity re-read | Byte-equivalent logic to 6E.17 L141–L371 (format_prompt, H16, margins, y16, LDA, s_scale/a0/b0/c0/b0_c2); no required telemetry omitted |

## 14. Zero-Step Proof

- `optimizer_steps = 0`; `training_steps = 0`; `experimental_checkpoints = 0` (preflight step 10 counter check).
- `derive_readout_initialization` NOT invoked in preflight/self-test.
- Self-test runs on `MockBackbone`/`MockTokenizer` only — no base model loaded, no optimizer step executed.
- `--execute` refused without authorization (verified).

## 15. Remaining Execution-Time Notes (not build blockers)

1. Real execution is still prohibited; `--execute` will run training only after (a) ~~§14 human sign-off~~ **DONE — APPROVED** (`DEF8C4CD…` reviewed; see `theo-slm-phase-6e18-section14-signoff-record-v1.md`), (b) explicit `execution_authorization.json`, (c) artifact-dir pre-registration confirmation.
2. On authorization, expected outputs: `pre_registration_snapshot.json`, `readout_initialization.json`, `t2_basis.json`, `runs.json`, `verdict.json`, `sha256_manifest.json`, `code_snapshot/`, 7 condition checkpoint dirs.
3. P4 in the self-test is exercised with synthetic separability; real P4 coverage comes from `evaluate_checkpoint` LOO outputs at execution time (values are unverifiable pre-execution — this is the intended pre-registration design).
4. Environment: Python 3.13.9, numpy 2.5.2, torch 2.13.0+cu126, sklearn 1.9.0; DEVICE = cuda:0 if available else cpu; benign `expandable_segments not supported` warning on this platform.

## 16. Verdict and Hard Stop

- **Build status:** Execution-phase infrastructure complete and verified via 11-step preflight, zero-step self-test, and refusal-path smoke test. **Amendment status:** §14 review #1 → AMEND; O-1 (B0-only collapse guard + explicit VOID) and O-2 (§5.2 row 7 deterministic + POS/ABS accuracy) implemented and verified; amended runner re-hashed (`6BCAF131…`, 2560 lines), self-test + preflight + refusal all re-passed. Conformance: **CONFORMANT**. **§14 status:** human re-review returned **APPROVED** on 2026-08-16 (sign-off record `theo-slm-phase-6e18-section14-signoff-record-v1.md`).
- **Execution authorization: NONE — NOT AUTHORIZED.**

- **Hard stop: ACTIVE.** Phase 6E.18 execution remains blocked pending explicit execution authorization (`execution_authorization.json`) and artifact-dir pre-registration confirmation. §14 human re-review/sign-off is **complete (APPROVED)**. This report produces no training, optimizer step, checkpoint, or scientific result.
