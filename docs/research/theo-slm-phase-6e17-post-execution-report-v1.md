# Phase 6E.17 Post-Execution Report (v1)

- **Status:** 6E.17 **COMPLETE, ACCEPTED WITH LIMITATIONS**. HARD STOP PRESERVED. NO FURTHER EXECUTION WITHOUT A NEW HUMAN-AUTHORIZED PROTOCOL. No 6E.18, no follow-up interventions, no component tuning.
- **Date (UTC):** 2026-08-16
- **Human decisions applied in this report:**
  1. B0 step-0 profile mismatch and B0 collapse-onset detector mismatch declared **non-voiding**; I1/C2 executed (documented in `sha256_manifest.json`).
  2. **P1/C2x sign-convention discrepancy adjudicated: adopt the corrected (protocol-anchored) verdict** — `CAPACITY_EXPLANATION_STRENGTHENED` — with both value sets (machine vs corrected) reported.
- **Authoritative documents:** `theo-slm-causal-intervention-authorization-protocol-draft-v2.md` (frozen protocol); `theo-slm-phase-6e17-pre-execution-authorization-review-v1.md`; `theo-slm-phase-6e16r-forensic-reconciliation-v1.md`.
- **Framing (binding):** This was a **causal falsification test of H-Causal**, NOT a repair of a proven root cause. `readout misalignment ≺ gradient conflict` (temporal precedence, INFERRED) is not `⇒` (causation, NOT ESTABLISHED). No result below claims a proven cause.

---

## 1. Run article

- **Phase:** 6E.17, protocol v2 (§4, execution order I1 → C2; B0 run first as fingerprint reproduction).
- **Model:** Qwen2.5-0.5B-Instruct, hidden 896, vocab 151936, 24 layers, tied embeddings, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- **Conditions (within-pair trajectory identity only):**
  - **B0** — frozen `w_delta` readout (matched baseline). Partial: truncated at step 6 by the collapse-onset detector (t*=4), see §3.
  - **I1** — aligned trainable readout `s = aᵀh_L + b`, `ẑ_PRO = s`, `ẑ_ABS = −s`; `a(0) = s_scale·û_LDA`. Full run to horizon 20.
  - **C2** — random-orientation control, identical form/capacity, `c(0) = s_scale·n` (seed 20260816, `|cos(n, û_LDA)| ≤ 0.05`). Full run to horizon 20.
- **Intervention variable:** the readout initialization orientation ONLY. All other state (base weights, LoRA init seed, batch order, optimizer, corpus, hyperparameters) is byte-identical across conditions. `C2_readout_cos_n_lda = 0.026` (random control engaged); `I1 readout_cos_lda = 1.0` (aligned, P5).
- **Environment (frozen):** isolated venv `theo6e17`: torch 2.11.0+cu128, transformers 4.43.1, peft 0.13.0, sklearn 1.9.0, numpy 2.5.2, scipy 1.18.0; GPU GTX 1650 (CUDA 12.8, 4GB). LDA derivation in fp32 (bf16 quantizes cos to 0.0007); training bf16 per §4.12.
- **Hyperparameters (frozen, unchanged):** checkpoints {0,4,6,7,14,15,17,20}, horizon 20 (B0 horizon 20; I1/C2 horizon 32), collapse K=3, λ_decision=10, grad_accum=2, batch=4, AdamW lr=1e-4 wd=0.01 clip=1.0, LoRA r=16 α=32 dropout=0.05 targets q/k/v/o/gate/up/down, seeds data=42 / C2=20260816.

## 2. Artifact inventory and provenance

All artifacts in `theo-data/datasets/theo_slm_v0_artifacts/phase-6e17-causal-intervention/`. SHA-256 manifest: `sha256_manifest.json` (full per-file hashes; run elapsed 2752.8 s).

| Item | Value |
|---|---|
| Base model sha256 (pre/post) | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` (unchanged) |
| Corpus sha256 (pre/post) | `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0` (unchanged) |
| Executed script sha256 | `a66992fea17daadcfd49106ec1cd57501f75df146c063270b360d3ee0b11b3b4` |
| Protocol sha256 | `3319193e7794472ed91abd6401741cec119da90122e9f433a8a594e44f318e52` |
| Review sha256 | `400bb78e5cd089b0deaad51df7ba630f971ff3c93d43def2e536b59d74c117de` |
| Pre-registration snapshot | `pre_registration_snapshot.json` (frozen config: `checkpoints`, `horizon 20`, `collapse_K 3`, `lambda_decision 10`, LoRA r16/α32/dropout .05/qkvogatedown, lr 1e-4 wd 0.01) |
| Readout initialization | `readout_initialization.json` (see §2.1) |
| Run logs | `run.log` (B0), `run2.log` (I1→C2) |
| Metrics | `runs.json` (I1/C2 full), `runs_partial_B.json` (B0 t0/t4/t6) |
| Code snapshot | `code_snapshot/run_phase_6e17_causal_intervention.py` (hash above) |
| Checkpoint state | `condition_B/`, `condition_I1/`, `condition_C2/` — `.json` (metrics) + `.pt` (state) per step, hashed in manifest |
| Machine verdict (frozen code) | `verdict.json` — `H0_SURVIVES` (see §4; superseded by human-adjudicated corrected verdict) |
| Fingerprint gate | `B0_fingerprint_gate.json` (see §2.2) |
| B0 spectra | `B0_joint_spectra.json` (see §2.3) |

### 2.1 Readout initialization (`readout_initialization.json`)

- `w_delta_norm = 0.564183`; `cos(û_LDA, w_delta) = 0.005447` (target 0.0055 ± 0.001 ✓); `cos(û_LDA, u_centroid) = 0.377601`; `cos(SVM_axis, w_delta) = 0.034824`; `s_scale = 0.06683328`.
- Step-0: `b0 = −0.07326254` (I1), `b0_c2 = 0.17438322` (C2); `step0_margin_b_mean = −4.829`, `step0_margin_i1_mean = −0.355`; proposal rates B 0.0% vs I1 25% (see §3, STEP0_PROFILE_MISMATCH).
- `a0_hash`, `b0_hash`, `b0_c2_hash`, `u_hat_LDA_hash`, `c0_hash` recorded.

### 2.2 B0 fingerprint gate (`B0_fingerprint_gate.json`)

Frozen criteria: `cos_agg(t6) ≤ −0.90`; `t* ∈ {14,15,16}`; `scalar r_eff(15) ∈ [0.9, 1.8]`; `max|cos(w_delta, u_disc)| ≤ 0.06`.

| Criterion | Threshold | Observed | Pass |
|---|---|---|---|
| cos_agg(t6) | ≤ −0.90 | **−0.996879** | ✓ |
| t* (collapse onset) | ∈ {14,15,16} | 4 (detector fired) | ✗ |
| scalar r_eff(15) | ∈ [0.9, 1.8] | null (B truncated at t6) | ✗ |
| max\|cos(w_delta, u_disc)\| step0 | ≤ 0.06 | 0.034824 | ✓ |
| **Overall** | — | `gate_pass: false` | ✗ |

`cos_agg_t4 = −0.933085`, `cos_agg_t6 = −0.996879` — the cos(H1) conflict fingerprint **reproduces the historical lock-in (−0.93 to −0.997)** exactly. The gate failure is driven entirely by the collapse-onset detector firing early (t*=4) and the consequent truncation at step 6 (r_eff(15) never reached). See §3.

### 2.3 B0 joint spectra (`B0_joint_spectra.json`, step 0)

- raw joint effective rank 7.019; row-normalized joint effective rank **7.2439** (frozen target); row-normalized top-1 dominance 0.255; scalar-margin effective rank 1.1602; top-1 dominance 0.9279.
- Step-0 argmax token = 8590 ('SH').

## 3. Discrepancy decisions (all documented, none voiding)

1. **STEP0_PROFILE_MISMATCH** (pre-execution, human-adjudicated non-voiding): frozen b0 formula centers I1 step-0 panel proposal at 25% vs B 0% (dev split 11.5%). Formula kept byte-for-byte; not a deviation.
2. **B0_COLLAPSE_ONSET_DETECTOR_MISMATCH** (human-adjudicated non-voiding): margin-sign dev collapse detector fired at t*=4 (guard at step 6) vs frozen t* ∈ {14,15,16}; driven by 13/52 POS ceiling of margin-sign proposal on the ABS-heavy dev split. cos(H1) fingerprint reproduced (agg t6 = −0.997). B0 truncated at step 6; I1/C2 executed to horizon 20. B0 fingerprint gate therefore reports `gate_pass: false` on t* and r_eff(15) only.
3. **P1/C2x_SIGN_CONVENTION_DISCREPANCY** (adjudicated in this session — see §4): the frozen-code primary metric was computed with a sign-inverted label-signed convention. Corrected verdict adopted.

**B0 run limitation (stated honestly):** B0 is a *partial* reproduction (steps 0, 4, 6) terminated by its own collapse guard. It confirms the conflict fingerprint at t4/t6 but does not reach the t15 fingerprint window. I1 vs C2 trajectory identity and full-horizon comparisons are unaffected (shared step-0 state + LoRA init re-verified).

## 4. Verdict matrix (P1–P5, C2x) — two value sets

### 4.1 Sign-convention discrepancy (the material finding)

The protocol (§3.6) freezes `cos(H1)`: **negative = conflict**, historical lock-in ≈ **−0.93 to −0.997** (6E.13: −0.9968 at step 6; 6E.14: −0.97). The executed script computed the primary endpoint as `cos_H1_mean_pairwise` = mean pairwise **label-signed** (ABS-side negated) cross-cosine of decision gradients, which returns **+0.97 at B-conflict** — sign-inverted versus the protocol anchor. The script's second quantity, `cos_H1_aggregate_613` = `cos(mean g_POS, −mean g_ABS)`, returns **−0.997 at B-conflict**, matching the protocol lock-in exactly (this is the convention the B0 gate already uses). `ρ_CD` and `F` inherit the same inversion (B t6: machine `ρ_CD=0.0/F=1.0` vs protocol-correct `ρ_CD=0.997/F≈0.0`).

Corrected protocol-anchored values (negative = conflict) are obtained by the sign flip of the machine pairwise values and by the aggregate quantity directly:

| Quantity | B0 (conflict) | I1 mean {4,6,14,15} | C2 mean {4,6,14,15} |
|---|---|---|---|
| cos(H1) aggregate (protocol lock-in style) | t4 −0.933, t6 −0.997 | **+0.606** | **+0.651** |
| cos(H1) corrected pairwise (flip of machine) | t4 +0.909→−0.909, t6 +0.974→−0.974 | **+0.526** | **+0.565** |
| cos(H1) machine pairwise (as executed) | t4 +0.909, t6 +0.974 | −0.525 | −0.565 |

Under either corrected convention the endpoint conclusions are identical.

### 4.2 Criterion evaluation (corrected, protocol-anchored)

| ID | Criterion | Threshold | Value (I1 vs C2) | Verdict |
|---|---|---|---|---|
| P1 | Conflict reduction (primary) | mean cos(H1){4,6,14,15}, I1 ≥ −0.30 | I1 = +0.61 (agg) / +0.53 (pair) | **PASS** |
| C2x | Alignment-exclusion (primary, mandatory) | Δ = mean(I1) − mean(C2) ≥ +0.25 | Δ = −0.044 (agg) / −0.040 (pair) | **FAIL** |
| P2 | Collapse prevention | proposal rate > 0 all t ≤ 20 | I1: 0.19 (t0) → 1.0 (t15); never 0 | **PASS** |
| P3 | Margin maintenance | cohen_d(15) ≥ 0.2; r_eff(15) ≤ 1.6 | cohen_d = 1.894; r_eff = 1.2426 | **PASS** |
| P4 | No separability degradation (guard) | resub kNN1 ∈ [0.65,1.0]; LOO within ±0.1 of B | resub 0.75/0.7083 ✓; LOO t6_N16: I1 0.4375 vs B 0.3125 (Δ=0.125 > 0.1) | **FAIL** |
| P5 | Readout engagement (guard) | max\|cos(û_LDA, a)\| ≥ 0.3 at t ≥ 4 | 1.0 (I1); 0.026 (C2, as designed) | **PASS** |

### 4.3 Decision-rule application (frozen §7)

- H-Causal supported (P1 ∧ C2x ∧ P4): C2x fails → **NO**.
- H-Causal partially supported (P1 ∧ ¬P2 ∧ P4): P2 holds → **NO**.
- H0 survives (¬P1): P1 holds (corrected) → **NO**.
- **Capacity explanation strengthened (¬C2x): YES.**

### 4.4 Adjudicated verdict

> **CAPACITY_EXPLANATION_STRENGTHENED** (frozen-code machine label `H0_SURVIVES` preserved in `verdict.json` and superseded by this human-adjudicated correction).

**Canonical interpretation (final, human-accepted):**

- **The alignment-specific H-Causal hypothesis is NOT SUPPORTED.** The capacity-matched random-orientation control C2 resolved the conflict comparably to the aligned I1 readout, and C2x failed (Δ = −0.04 ≪ +0.25).
- **The leading conclusion is NOT "LDA alignment fixes the conflict."** The supported conclusion is narrower: **a trainable adaptive decision-readout parameterization eliminates the observed conflict under both aligned and random initial orientations**, strengthening an adaptive-readout / capacity-and-parameterization explanation.
- **"Capacity" is NOT upgraded to a proven sole root cause.** Capacity, trainability, adaptive axis rotation, and binary output reparameterization remain **confounded within this intervention family**. The evidence supports the readout-family explanation as a class; it does not isolate which member property is causal.

**Interpretation (§8 row "Both I1 and C2 improve similarly (C2x fails)"):** the aligned readout (I1) reduced the B0 gradient conflict — cos(H1) from −0.997 (B0, conflict) to +0.61 (I1, no conflict); P1 passes. But the random-orientation control (C2) achieved the same, cos(H1) = +0.65, and I1 does **not** beat C2 (Δ = −0.04 ≪ +0.25). Training a differentiable 897-param readout of the shared hidden state resolves the conflict regardless of orientation; alignment to û_LDA is **not** the operative variable. Trainability/capacity of the readout family is. The alignment-specific causal claim is therefore weakened; H-Causal is not supported.

**Constraints honored:** I1≈C2 on the primary endpoint is NOT treated as an alignment benefit; one-class improvement with equivalent other-class harm is NOT conflict resolution. P4 guard failure (LOO-kNN1 N16 t6, Δ=0.125 vs tolerance 0.1; resub unchanged at 0.75/0.7083) is a minor separability fluctuation on a single LOO cell and does not alter the decision row, but is reported for review.

## 5. Protocol amendments (all documented, config hashes frozen)

1. **BGR7_HORIZON_EARLY_STOP** — B0 horizon forced to 20 (then 32 for I1/C2) so the t15 fingerprint window is reachable; the early-stop patch is applied only for the experimental 6E.17 runs and reverted at each restore; protocol-config hashes remain frozen. Step-0 profile and b0 formula untouched.
2. **P1/C2x_SIGN_CONVENTION_ADJUDICATION** — this report's §4: metric sign convention reconciled to the protocol anchor ("negative = conflict", lock-in −0.93..−0.997); machine output preserved verbatim; corrected verdict adopted by human decision. No thresholds changed.

## 6. Provenance and verification summary

- Pre/post cryptographic verification: base model and corpus hashes identical before and after (`fdf756fa…`, `a7b4e845…`); full per-artifact SHA-256 manifest written (`sha256_manifest.json`).
- Trajectory identity: B0/I1/C2 share step-0 base weights (re-hashed) and identical LoRA adapter init; conditions differ only in the §1 intervention variable.
- Frozen-code output `verdict.json` (`H0_SURVIVES`) retained as executed for provenance; this report supersedes it via the recorded human adjudication.
- Hard stop: no 6E.18, no follow-up interventions, no component tuning. Return to human review.
