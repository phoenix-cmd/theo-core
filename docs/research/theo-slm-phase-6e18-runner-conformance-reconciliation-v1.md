# THEO Phase 6E.18 — Runner-Side Conformance and Protocol↔Runner Reconciliation Report

- **Scope:** Implementation and verification of the four documented runner-side conformance tasks required before §14 sign-off.
- **Governing spec:** `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.3.md` (FINAL, 7-gate audit PASS).
- **Runner modified:** `theo-core\scripts\dataset_generator\run_phase_6e18_causal_intervention.py`
- **Date:** 2026-08-16.

## 1. Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| Protocol v2.1.2 (frozen, unchanged) | `D13576BA019350266CC4551361CA07C0B20FE32CF462D768751461772B97EAF6` |
| Protocol v2.1.3 (FINAL, audit target) | `910796C879BBAFBA80C0D6829D382723CC4E89425003380FD4CC94E1E21312C9` |
| Runner AFTER conformance work | `706D75CED1716DB72183DD31B75A8BE6F4497C3C3DC57D62016F9BE718178E36` (845 lines) |
| Runner AFTER execution-phase build | `322FC867086A74893C17D4D6E7AB126AD01CD020BE52072C8F6BEA8171F66501` (2260 lines) |
| Runner AFTER O-1/O-2 amendment (§14 review #1 → AMEND) | `6BCAF1319AC551848FFDB55AEA121D14D808408A192E05F2AA6171161DECEF96` (2560 lines) |
| §14 review package (amended, reviewed) | `DEF8C4CD339A98D2217640EAE4CD2B31AD72D2CCC7E6D856D55D9DAAF0DFCB5A` (120 lines) |
| §14 sign-off record (APPROVED) | `00727716A0B88070CA22B0F45665DC5B7F37D9F53CF96852C23F63927DD877D3` (75 lines) |

The runner's `EXPECTED_PROTOCOL_HASH` now binds to the v2.1.3 FINAL hash; preflight step [1/11] verifies the on-disk protocol matches it.

## 2. Conformance Task Implementation Record

### Task 1 — Initialization Derivation (§3.8)
Implemented in the runner:
- `compute_init_constants(H16, margins_b, u_lda, n_c2)` — pure deterministic arithmetic: `s_scale = std(margins_B0)/(std(H16 @ u_hat_LDA) + 1e-12)`; `a0 = s_scale·u_hat_LDA`; `b0 = -median(a0 @ H16[:8].T)`; `c0 = s_scale·n_c2`; `b0_c2 = -median(c0 @ H16[:8].T)`; `b_delta = 0`; `b_fixed = b0`; T1 `(alpha0, b0) = (1.0, 0.0)`; T2 `beta0 = None -> zeros(k), b0 = 0.0`; V1 aligned mirror values.
- `generate_c2_random_orientation(u_lda, seed=20260816)` — deterministic torch.Generator rejection sampler enforcing `|cos(n, u_hat_LDA)| <= 0.05` (6E.17 parity).
- `derive_readout_initialization(data, tokenizer)` — execution-time step-0 derivation (frozen-model fp32 inference, layer-24 last-token `H16`, B0 margins, LDA fit `solver="lsqr", shrinkage="auto"`, all §3.8 constants). **INFERENCE-ONLY; defined but NOT invoked in preflight** (zero-execution guard).

### Task 2 — V1 Aligned-Mirror Initialization (the CHANGE, §3.7/§3.8)
- `compute_init_constants` produces `w_PRO(0) = a0`, `w_ABS(0) = -a0`, `b_PRO(0) = b_ABS(0) = 0.0`.
- `V1ReadoutHead` consumes these values at construction; 1794 trainable params asserted.
- Preflight step [8/10] and targeted checks verify: exact mirror (`w_ABS == -w_PRO`), `||w_PRO(0)|| == s_scale`, `cos(w_PRO(0), u_hat_LDA) == 1.0`, zero biases, mirrored step-0 logits.

### Task 3 — M1–M5 Rule Wiring (§7.2)
- New `evaluate_m_rules(metrics, t2_k, i1_void)` returns per-rule dispositions: **SUPPORTED / NOT SUPPORTED / INDETERMINATE / NOT EVALUABLE / VOID**, consuming only the §6.2 **aggregate** `cos(H1)` over `RULE_CHECKPOINTS = [4, 6, 14, 15]` (mean aggregation; any missing checkpoint -> VOID).
- M3 honors the composite-confound scoping: `k` outside [2, 8] -> VOID (§8); `i1_void` -> NOT EVALUABLE; MATCH requires `Δ ≤ 0.10 AND mean cos(T2) ≥ −0.30`.
- M4 applies the §6.1 item 9 policy-degradation test (`P(15) ≤ 0.0` OR `d(15) < 0.00`) with incomplete-telemetry annotation.

### Task 4 — ρ_CD Aggregate Binding (§6.1 item 2)
- `compute_raw_gradient_cosine` now computes `rho_CD = max(0, -agg_cos)` from the aggregate cosine (was mean-pairwise).
- The §6.2 synthetic sign gate now asserts on `cos_H1_aggregate` (identical for single synthetic pairs; binds the gate to the metric decision rules consume).

### Additional fix — Pre-existing preflight test fixture defect
- Preflight step 6 (existing, unmodified logic) used a synthetic panel whose energy profile derived `k = 13` (Marcenko–Pastur noise eigenvalues dominate the 3 artificially amplified directions), so the `[2, 8]` §8 assertion could never pass. Test fixture corrected: noise columns scaled `1e-3` so the §8 procedure derives `k = 3` in-range. This is a **test-data fix**, not a protocol change.

## 3. Verification Evidence

| Verification | Result |
|---|---|
| `py_compile` | OK |
| Preflight validation (11 steps) | **11/11 PASS** — protocol hash binding; 7-condition universe; §6.2 synthetic sign gate (aggregate); parameter accounting (0/897/897/0/2/5/1794); §7.1 P4 dual-regime + early-truncation fallback; §8 SVD basis (k=3 in [2,8]); §3.8 init arithmetic (hand-derived `b0 = -4.5·s_scale` anchor + invariants); V1 aligned mirror; M1–M5 dispositions; zero-execution guard; verdict-emission/authorization-gate safeguard |
| Targeted conformance script (25 checks) | **25/25 PASS** — ρ_CD discrimination case (`agg ≈ 0.7071` vs old pairwise `0.5`); init determinism; C2 sampler determinism + `|cos| ≤ 0.05`; all §3.8 invariants; V1 mirror exactness (bitwise) + logit mirror; M-rule disposition matrix |
| Exhaustive M1–M5 disposition matrix (36 checks) | **36/36 PASS** — every SUPPORTED / NOT SUPPORTED / INDETERMINATE / NOT EVALUABLE / VOID cell including exact thresholds (`-0.60`, `-0.30`, `Δ=0.10`, `k∈{2,8}`, `k∈{1,9}`, `P(15)=0.0`, `d(15)=−0.01`), boundary parity vs §7.2, missing/partial-checkpoint → VOID, incomplete-telemetry annotation, and k-range-before-I1-VOID precedence |
| Zero-step self-test (`--self-test`, mock-backed) | **PASS** — data layer (67/67/134, panels 8/8/8); §6.1 pure metrics incl. Cohen's d; gradient flow + readout connectivity; checkpoint routing + `readout_state_sha256` change-on-perturbation + restore; all-7-condition factory; P1–P5/C2x rules incl. §7.1 P4 comparator universe {0,4,6,14,15,20} + step-0 anchor + UNAVAILABLE recording; M1–M5 dispositions (M5 NOT SUPPORTED for C2 = −0.60 < −0.30 floor); verdict double gate; protocol hash; zero optimizer/training steps. **Amendment sections:** `o1_collapse_guard_scoping` + `o1_void_state_propagation` (B0→TRUNCATE; I1/C2 same condition→CONTINUE; non-B0 collapse recorded + checkpoints 14/15/20 retained → P1/P3/P5 evaluable, P2 fails; VOID → all I1 rules NOT EVALUABLE + M3 NOT EVALUABLE, not collapsed, no stopping_event) and `o2_row7_deterministic` (POS/ABS accuracy emitted at {0}+RULE_CHECKPOINTS incl. 15; flat→NO_ARTIFACTUAL_TRADE_OFF; improving-POS/degrading-ABS→ARTIFACTUAL_TRADE_OFF; tiny ±0.01 opposite signs still trigger → sign-only, no magnitude threshold; VOID/missing-checkpoint→ROW_7_NOT_DETERMINABLE_FROM_REGISTERED_METRICS; Row-7 always surfaced in verdict + outcome_summary) |
| `--execute` refusal path (smoke test) | **PASS** — absent `execution_authorization.json` → exit 1, exact verdict `PRE_EXECUTION / NO_SCIENTIFIC_VERDICT / EXECUTION_NOT_AUTHORIZED`, `verdict.json` + `sha256_manifest.json` written (protocol/script hashes correct; `script_sha256 = 6BCAF131…`); smoke-test dir removed afterward, real artifact dir left pristine |
| 6E.17 parity re-read | `format_prompt` + `'{"decision": "'`, layer-24 last-token `H16`, margins `z[PRO_ID]−z[AB_ID]`, `y16=[1×8,0×8]`, LDA `solver="lsqr", shrinkage="auto"`, `s_scale`/`a0`/`b0`/`c0`/`b0_c2` arithmetic — byte-equivalent logic to 6E.17 L141–L371 (no required telemetry omitted) |
| Zero execution | optimizer_steps = 0; training_steps = 0; experimental_checkpoints = 0; `derive_readout_initialization` not invoked; `--execute` refused without authorization |

## 4. Protocol↔Runner Reconciliation Table

| Protocol requirement | Runner implementation | Status |
|---|---|---|
| §3.8 `s_scale = std(margins_B0)/(std(H16@u_hat_LDA)+1e-12)` | `compute_init_constants` | CONFORMANT (verified) |
| §3.8 `b0 = -median(a0@H16[:8].T)`; `b''(0) = -median(c0@H16[:8].T)` | `compute_init_constants` | CONFORMANT (verified) |
| §3.8 `b_delta = 0`; `b_fixed = b0` | `compute_init_constants` | CONFORMANT (verified) |
| §3.3 C2 orientation `n`, seed `20260816`, `|cos(n,u_hat_LDA)| ≤ 0.05` | `generate_c2_random_orientation` | CONFORMANT (verified, deterministic) |
| §3.5 T1 `alpha(0)=1.0, b(0)=0.0` | `compute_init_constants` / `T1ReadoutHead` defaults | CONFORMANT (verified) |
| §3.6 T2 `beta(0)=zeros(k), b(0)=0.0` | `T2ReadoutHead` default | CONFORMANT (verified) |
| §3.7/§3.8 V1 aligned mirror (CHANGE) | `compute_init_constants` + `V1ReadoutHead` | CONFORMANT (verified) |
| §6.1 item 2 `rho_CD = max(0, -cos(H1)_agg)` | `compute_raw_gradient_cosine` | CONFORMANT (verified) |
| §7.2 M1–M5 dispositions | `evaluate_m_rules` | CONFORMANT (verified) |
| §8 SVD procedure + `k ∈ [2,8]` | `derive_t2_basis_from_covariance` | CONFORMANT (verified) |
| §7.1 P4 comparator + early-truncation fallback | `evaluate_p4_comparator` | CONFORMANT (verified) |
| Protocol artifact binding | `EXPECTED_PROTOCOL_HASH` = v2.1.3 FINAL | CONFORMANT (verified) |

## 5. Execution-Phase Build Status (previously "Remaining Execution-Time Items")

The execution-phase implementation items listed below have been **built and verified** (zero-step mock self-test, preflight, refusal-path smoke test) but **not executed**:

1. Full training/main harness for the 7 conditions (LoRA wrapping, optimizer incl. readout params, checkpoint evaluation, telemetry collection, **B0-only collapse guard** per §10 with recorded non-B0 collapse continuation and explicit VOID for cannot-continue, run manifest, verdict bundle) — implemented (`run_condition`, `collapse_guard_action`, `evaluate_checkpoint`, `save_checkpoint`/`load_checkpoint`), 0 optimizer steps taken.
2. Runtime metric implementations for §6.1 items 4–9 (proposal rate on dev set, Cohen's d, effective rank of margin-Jacobian Gram, Resub k-NN1, LOO-CV, **POS/ABS accuracy** incl. per-class) — implemented (6E.17 parity), wired into telemetry; verified via synthetic fixtures.
3. M1–M5 and P1–P5/C2x results emitted into `runs.json` / `verdict.json` with dispositions and checkpoint completeness at execution time; **§5.2 row 7 (artifactual trade-off) evaluated deterministically and always surfaced** (`evaluate_row7`, `Row7_artifactual_trade_off` in verdict) — implemented (`build_metrics_store`, `evaluate_p_rules`, `evaluate_m_rules`, `summarize_outcome`, `compute_verdict`, `emit_verdict`).
4. T2 `U_k`, `k`, basis hash, and all §3.8 constants recorded in the run manifest at execution time — implemented (`derive_t2_basis_from_covariance`, `t2_basis.json`, `pre_registration_snapshot.json`, `sha256_manifest.json`).

None of these are protocol defects; they are execution-phase implementation work, now complete and verified without executing.

## 6. Verdict and Hard Stop

- **Conformance status:** The four documented runner-side tasks are implemented and verified. Protocol↔runner reconciliation: **CONFORMANT** on all audited requirements.
- **Amendment status (§14 review #1 → AMEND):** O-1 (B0-only collapse guard + explicit VOID) and O-2 (§5.2 row 7 machine-detectable, no invented threshold, explicit indeterminate) are implemented and verified via targeted zero-step self-test sections; the amended runner is re-hashed (`6BCAF131…`, 2560 lines) and re-passed preflight 11/11 + refusal smoke test.
- **§14 status:** Human re-review of the amended package (`DEF8C4CD…`) returned **APPROVED** on 2026-08-16 (see `theo-slm-phase-6e18-section14-signoff-record-v1.md`). Approval is conditional: it makes the package eligible for, but does not itself grant, execution authorization.
- **Execution-phase build status:** All harness/metrics/manifest/verdict infrastructure is implemented and verified via zero-step self-test, 11-step preflight, and the `--execute` refusal-path smoke test. No training, optimizer step, checkpoint, or scientific result was produced.
- **Execution authorization: NONE — NOT AUTHORIZED.** No training, optimizer step, checkpoint, or scientific result was produced or authorized by this report.
- **Hard stop: ACTIVE.** Phase 6E.18 execution remains blocked pending: (1) ~~human §14 re-review/sign-off on the amended package~~ **DONE — APPROVED**, (2) explicit execution authorization (`execution_authorization.json`), and (3) post-review confirmation of the final execution-readiness report.
