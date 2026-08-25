# FORENSIC 6E.20.1 — READ-ONLY CAUSAL ANALYSIS OF THE NEGATIVE SCIENTIFIC RESULT

**Phase:** 6E.20.1 — Fixed-Subspace Readout Execution Extension (v7)
**Analysis date:** 2026-08-18
**Mode:** READ-ONLY forensic causal analysis of the frozen 6E.20.1 artifacts
**Constrained by:** 6E.20.1 is CLOSED (PROCEDURALLY VALID · SCIENTIFICALLY NEGATIVE). No frozen/final artifact, runner, authorization, protocol, checkpoint, output, or provenance file was modified. No experiment was launched. Any proposed new experiment or code modification below is a **recommendation only** and requires separate review and explicit authorization.

---

## 1. Scope and Method

This report explains **why** the executed run produced `cos(H1) ≈ 0.39–0.46` across all eight mandated checkpoints, versus the protocol success threshold `cos(H1) ≥ +0.80` (P2 / H1). It uses only the frozen artifacts:

- `runs.json` (8 evaluation events + gate dispositions)
- `run_phase_6e20_execution_v7.py` (final runner, `C8E298C8…`)
- `run_phase_6e20_fixed_subspace_readout.py` (frozen base runner, `550FAA34…`)
- `run_phase_6e18_causal_intervention.py` (frozen F7 gradient machinery, `6BCAF131…`)
- `theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md` (protocol, `E16CA4F1…`)
- `pre_registration_snapshot.json`, `p5_loo_report.json`, `sha256_manifest.json`

No recomputation was performed; all quantitative claims are read directly from these files.

---

## 2. Status Recap

| Item | Value |
|---|---|
| Final verdict | `PARTIAL` for all 8 checkpoints (P1 passes, P2 fails) |
| `cos(H1)` observed | 0.461 (step 0) → 0.428 → 0.410 → 0.403 → 0.427 → 0.417 → 0.394 → 0.391 (step 20); mean ≈ 0.417 |
| Threshold P2 | `cos(H1) ≥ +0.80` |
| `rho_c` reported | 1.0 for all; `sigma_c` ≈ 1.5e-16 |
| Gates G1–G8 | All PASS (no VOID) |
| Conditions executed | 7 (B0, F1, S2, S4, S8, C4-Rand, R1-Rand); 56/56 checkpoints |

The negative result is procedurally valid (verified in `FORENSIC_6E20.1_FINAL_EXECUTION_REPORT.md`). This report asks *why* the metric landed where it did.

---

## 3. What `cos(H1)` Actually Measures (from the code)

`compute_cos_H1_from_live_model` (`run_phase_6e20_execution_v7.py:379`) computes, for the restored model, the cosine between:

- `mean POS gradient` = mean over 8 POS panel prompts of `per_sample_decision_gradient(model, readout, item, lora_params, …)`
- `mean ABS gradient` = mean over 8 ABS panel prompts of the same.

`per_sample_decision_gradient` (`run_phase_6e18_causal_intervention.py:1006`) returns `∇_{lora_params} dec_loss`, where `dec_loss = CrossEntropy(dec_logits, label)` at the decision position, with `dec_logits[PRO_ID]=z_pro`, `dec_logits[AB_ID]=z_abs` (the readout's predicted PRO/ABS logits), and the label is `PRO_ID` for POS items and `AB_ID` for ABS items.

`compute_raw_gradient_cosine` (`run_phase_6e18_causal_intervention.py:400`) returns the **raw** cosine between the two mean gradients, **with no label-sign inversion**. Its own sign-gate (`verify_synthetic_sign_gate`, lines 434–459) defines the convention unambiguously:
- aligned synthetic vectors → `cos = +1.0`
- opposing synthetic vectors → `cos = -1.0`

So **the protocol's success criterion `cos(H1) ≥ +0.80` means the mean POS and mean ABS decision-gradients must be ALIGNED (point in the same direction in LoRA-parameter space).** This is the 6E.18 "gradient compatibility" / low-conflict regime.

**Reference values from 6E.18 (protocol v3.1 §7, line 28–29):** T2 (truncated subspace) `cos(H1)=+0.9878`; F1 (fixed LDA) `cos(H1)=+0.5831`. Both are **positive (aligned)**, and the research question's target `+0.80` sits above the F1 baseline. The 6E.18 experience thus establishes that *positive, training-induced alignment is the expected, achievable regime* when a fixed readout conditions training.

---

## 4. Primary Empirical Findings

### F1 — The treatment conditions (S4/S8) were never evaluated for `cos(H1)` (HIGH confidence)
`run_evaluation` (`run_phase_6e20_execution_v7.py:858`) loops over `CHECKPOINT_SET` and for every checkpoint builds:
```python
cp_path = out_dir / "B0" / f"checkpoint_{cp_step}"     # line 866 — HARD-CODED B0
```
and `evaluate_checkpoint` computes `cos(H1)` with:
```python
s8_readout = reg["w_all"]["S8"]                          # line 805 — HARD-CODED S8 readout
```
There is **no outer loop over `CONDITIONS`**. Consequently, the only `cos(H1)` values ever produced are for the **B0 baseline model, probed through the S8 readout**.

But the protocol's primary hypothesis H1 (`v3.1` line 214) is: *"Both S4 and S8 achieve `mean cos(H1) ≥ +0.80`."* P2 (`v3.1` line 231, 245) is the acceptance metric for S4/S8. **The runner never computes `cos(H1)` for the S4 or S8 checkpoints.** Therefore the experiment's central hypothesis was **not measured**. The recorded "negative" value describes the B0 baseline, not the fixed-subspace-readout treatment whose success the protocol predicts.

### F2 — `rho_c` reported is a trivial constant, not a live measurement (HIGH confidence)
`evaluate_checkpoint` reports `rho_c` from `compute_rho_c_from_registration` (`run_phase_6e20_execution_v7.py:676`). That function computes the **readout-only analytic path** where, by construction (line 699):
```python
g_total_i = g_R_i.copy()      # readout-total alignment reduces to g_R itself
rho_i = rho_sample(g_R_i, g_total_i)   # == rho_sample(g_R_i, g_R_i) == 1.0
```
The in-code comment (lines 679–685) states this is the *"documented readout-only-path value … NOT the cos(H1) object"* and that `rho_c == 1.0` identically. So `rho_c = 1.0` in `runs.json` is a **fixed analytic constant**, not a measurement of the trained model. The protocol (§5.7, line 185, 392) explicitly defines `rho_c` as the *live readout–total alignment* and says `cos(H1)` "is never replaced by ρ_c or g_R." The runner therefore (a) reports a constant under a live-metric name, and (b) satisfies P1 (`rho_c ≥ 0.50`) trivially. Because the verdict logic is `PARTIAL = P1 ∧ ¬P2`, the trivial `rho_c=1.0` is the *sole* reason the verdict is `PARTIAL` rather than `FAIL`. The verdict is thus partially an artifact of this mis-implementation.

### F3 — `cos(H1)` is positive but below target and *declines* with training (HIGH confidence)
Observed: step 0 = 0.461 (highest) → step 20 = 0.391 (lowest). The untrained checkpoint already yields 0.461, and training **moved it downward** (net decrease; non-monotonic, with a local uptick at step 14). In the 6E.18 framework, *higher* `cos(H1)` = better gradient compatibility; training under a fixed readout *increased* alignment there (F1 +0.58, T2 +0.99). Here B0 training moved `cos(H1)` in the **opposite direction** (toward opposition / more conflict). The metric is responsive (it changed with restored weights), so this is a real signal, not noise — but its direction contradicts the 6E.18 reference regime.

### F4 — The protocol itself anticipates `cos(H1)` may be background-dominated (HIGH confidence, from protocol text)
Protocol §10.5 / §5.7 (line 185): *"cos(H1) … is explicitly background-dominated and condition-dependent; its sign/level is not interpreted as evidence of subspace confinement, and it is never replaced by ρ_c or g_R."* B-4 (lines 47, 373) establishes via chain rule that the total gradient = `g_bg + α·w`, and **only the readout contribution is subspace-confined**; the free LM-head / background term is not. B0's readout is `full ambient R^896 (unembedding rows)` (`run_phase_6e20_fixed_subspace_readout.py:275`), i.e., **not a confined subspace**. Therefore B0's total-gradient `cos(H1)` is, by the protocol's own analysis, dominated by the unconfined background term and is **not a clean test of the subspace-confinement hypothesis**. The protocol's own language for such values (re R1-Rand, line 218) is *"background-dominated and sign-indeterminate."* A `cos(H1)` of +0.4 for B0 is consistent with exactly this regime.

### F5 — Readout/measurement apparatus is functioning (HIGH confidence)
The F7 gradient object is the **hash-verified frozen 6E.18 machinery** (`6BCAF131…`), identical to the protocol reference. G6 sign-gate passes. G7 confinement residual ratios are ≈ 1e-15 (`run_phase_6e20_execution_v7.py` runs.json, line 117) — the readout fits the 16 hidden states **almost perfectly**, so the readout is not information-limited or broken. G3 κ-gates pass (κ2=0.057, κ4=0.086, κ8=0.125, κR=0.054; all ≥ 0.05 floor). So the measurement is sound; the problem is **target selection and interpretation**, not a broken readout.

### F6 — Provenance gaps limit the analysis (HIGH confidence that data is missing; MEDIUM on consequence)
`emit_provenance` (`run_phase_6e20_execution_v7.py:916–925`) writes only `cos_H1 / rho_c / sigma_c / verdict` per checkpoint. The per-condition margins, balanced-accuracy, and Cohen's d computed inside `evaluate_checkpoint_at_state` are **discarded**. Furthermore, the final valid run was the `evaluate-only` re-run (PID 27560), which passed **empty** `training_telemetry` to `emit_provenance` (`main`, line 1335), so the frozen `runs.json` contains **no training dynamics** (proposal_rate / balanced_accuracy / cohens_d trajectories). `pre_registration_snapshot.json` is a schema shell (no numeric κ/s_scale/residual values), and `p5_loo_report.json` contains only a description + hashes (the P5 LOO-CV diagnostic was never computed). Consequently we **cannot, from frozen provenance, determine whether B0 training moved the decision margin** — only that `cos(H1)` moved. The 56 on-disk checkpoints still exist and could recover this (read-only), but it is not in the preserved record.

### F7 — The intervention "dose" is very small (MEDIUM-HIGH confidence)
From `run_phase_6e20_fixed_subspace_readout.py`: `HORIZON=20`, `GRAD_ACCUM=2` → **10 optimizer steps**; `LR=1e-4`; `LORA_R=16`; `BATCH_SIZE=4`; panel `N=16`. The training loop (`run_phase_6e20_execution_v7.py:573`) processes all 16 prompts in one batched forward per step. So the entire B0 intervention is **10 gradient updates on a 16-prompt panel with LoRA rank 16 on a ~0.5B model**. This is minuscule relative to the scale at which 6E.18 established the +0.58/+0.99 baselines. Even with a correct design, such a dose may be inadequate to drive `cos(H1)` from ~0.4 to ≥ 0.80.

---

## 5. Competing Hypotheses

Each hypothesis is assessed for how well it explains the observed `cos(H1) ≈ 0.39–0.46` given the above evidence.

### H1 — Measurement-target mismatch (the experiment did not test its own hypothesis) — **PRIMARY**
**Claim:** The recorded negative `cos(H1)` is the B0 baseline probed through the S8 readout; the protocol's H1/P2 target (S4 and S8 achieving ≥ +0.80) was never evaluated because `run_evaluation` hard-codes B0 (F1) and S8-readout (F4).
**Supports:** F1, F4, F5. Directly explains why a "negative" result exists despite the hypothesis being about different conditions.
**Confidence:** **HIGH** that the measurement targets the wrong condition; the *actual* S4/S8 `cos(H1)` is **UNKNOWN (not measured)**.
**Implication:** The negative verdict does **not** refute H1. It is inconclusive with respect to the protocol's central claim.

### H2 — Under-powered intervention
**Claim:** 10 optimizer steps on 16 prompts cannot move `cos(H1)` to +0.80.
**Supports:** F7; F3 (metric is responsive but magnitude is small).
**Confidence:** **MEDIUM-HIGH** that the dose is small; **MEDIUM** that dose alone explains the gap (the direction also moved wrong, F3).
**Implication:** Even a correctly-targeted evaluation might still fail at this dose; dose is a confounder independent of H1.

### H3 — B0 training increased gradient conflict (opposite dynamical regime)
**Claim:** B0 training moved `cos(H1)` downward (0.46→0.39), i.e., toward *more opposed* POS/ABS gradients, whereas 6E.18 training under a fixed readout increased alignment. The B0 (full-unembedding) readout conditions training differently from the subspace readouts.
**Supports:** F3; F4 (B0 readout is unconfined, so its total gradient is background-dominated); reference contrast with 6E.18.
**Confidence:** **MEDIUM** (direction of change is factual; interpretation as "conflict" relies on accepting the 6E.18 alignment=good framing, which the protocol establishes).
**Implication:** If real, the B0 condition is simply the wrong probe for the subspace hypothesis — reinforcing H1.

### H4 — `rho_c` mis-implementation distorts the verdict
**Claim:** Reporting the readout-only constant `rho_c=1.0` under the live-metric name satisfies P1 trivially, making the verdict `PARTIAL` instead of `FAIL`, and masks the fact that the live readout–total alignment was never measured.
**Supports:** F2.
**Confidence:** **HIGH** that `rho_c` is a constant; **MEDIUM** on how it would change the verdict (a live `rho_c < 0.50` would yield `FAIL`, not `PARTIAL`).
**Implication:** The `PARTIAL` label is partly an artifact; the result's *interpretation* is weaker than the label suggests.

### H5 — `cos(H1)` is background-dominated for B0 (design-assumption stressor)
**Claim:** Because B0's readout is the full unembedding direction (unconfined), and only the readout contribution to the total gradient is subspace-confined (B-4), B0's `cos(H1)` is dominated by the background `g_bg` term. The protocol itself warns `cos(H1)` is "background-dominated and condition-dependent" (§10.5). The +0.4 may therefore reflect base-model background geometry, not a learned subspace effect.
**Supports:** F4, F5, protocol §10.5/B-4.
**Confidence:** **MEDIUM** (coherent with the protocol's own analysis; cannot be quantified from frozen artifacts because live `rho_c`/decomposition were not saved — F6).
**Implication:** Even a correct, adequately-dosed evaluation of B0 would not cleanly test the subspace hypothesis; S4/S8 (confined readouts) are the appropriate targets — which were never evaluated (H1).

### H6 — Registration geometry is near the κ-floor (secondary stressor)
**Claim:** κ2/κ4/κ8/κR are all near the 0.05 gate floor, meaning the LDA decision axis lives mostly in low-variance PCA directions of H16. Gradients along such near-orthogonal directions may be weak/noisy, contributing to a background-dominated, low-magnitude `cos(H1)`.
**Supports:** G3 values in runs.json; F4.
**Confidence:** **MEDIUM** (gates passed; effect on `cos(H1)` magnitude is plausible but unquantified).
**Implication:** A design property worth checking in any re-run, but not sufficient alone to explain the result.

---

## 6. Distinguishing the Three Failure Classes

The user's central question: **intervention failure vs readout/measurement failure vs deeper experimental-design assumption failure?**

| Class | Assessment | Key evidence |
|---|---|---|
| **Readout / measurement failure** | **Rejected as primary cause.** The F7 machinery is the hash-verified frozen 6E.18 code; G6 sign-gate passes; G7 residual ≈ 1e-15 (readout fits H16 near-perfectly). The readout is *not* broken. | F5 |
| **Intervention failure** | **Cannot be concluded.** We cannot tell whether the *intended* intervention (S4/S8 subspace readouts) failed, because those conditions were never evaluated (H1). For B0 specifically, the dose is tiny (H2) and the direction moved toward more conflict (H3) — but B0 is the baseline, not the treatment. | F1, F3, F7 |
| **Experimental-design / measurement-target assumption failure** | **PRIMARY.** The runner evaluates only B0 (baseline) through the S8 readout and never tests S4/S8; `rho_c` is a constant mislabeled as a live metric; `cos(H1)` for B0 is, by the protocol's own B-4/§10.5 analysis, background-dominated and not a clean subspace test. The negative verdict is most parsimoniously explained as *measuring the wrong target*. | F1, F2, F4, H1, H5 |

**Conclusion:** The evidence most strongly supports a **measurement-target / design-assumption failure**, not a confirmed intervention failure and not a readout breakdown. The experiment as executed does **not** constitute a valid test of H1. The "negative" label should be read as **"H1 untested; B0 baseline measured instead of S4/S8 treatment."**

---

## 7. Provenance Limitations Affecting This Analysis

- Per-condition margins/BA/Cohen's-d computed in-run were **not preserved** (F6).
- Training telemetry is **absent** from the frozen `runs.json` (evaluate-only re-run emitted empty telemetry).
- `pre_registration_snapshot.json` carries no numeric registration diagnostics; `p5_loo_report.json` carries no LOO results.
- The live `rho_c` (readout–total alignment) was never computed; only the readout-only constant was saved.
- These gaps mean the analysis cannot definitively separate "intervention had no effect" from "intervention moved the margin but not `cos(H1)`," nor quantify the background-vs-readout gradient decomposition. They should be closed in any authorized re-run.

---

## 8. Recommended Next Actions (RECOMMENDATIONS ONLY — require separate authorization)

1. **Evaluate the correct targets.** Modify `run_evaluation` to loop over `CONDITIONS` and compute `cos(H1)` for each condition's **own** checkpoint using that condition's **own** readout (S2/S4/S8 with their subspace readouts; F1 with LDA; B0 with unembedding). This directly tests H1/P2. *This is the single highest-value action.* (Current code: `run_phase_6e20_execution_v7.py:866,805`.)
2. **Compute live `rho_c`.** Replace the readout-only constant path (`compute_rho_c_from_registration`) with the protocol's §5.7 live readout–total alignment (project the actual total gradient onto the readout direction) so P1 is meaningfully evaluated and the `PARTIAL` verdict is no longer an artifact (F2, H4).
3. **Preserve full provenance.** Have `emit_provenance` retain per-condition margins/BA/Cohen's-d and the training telemetry (F6). Recover B0 training dynamics from the 56 on-disk checkpoints via a read-only script to determine whether the margin moved.
4. **Re-examine dose adequacy.** Compare the 10-step / 16-prompt intervention against the scale at which 6E.18 achieved +0.58/+0.99; consider whether more steps / more panel data are needed before concluding (H2, F7).
5. **Quantify background domination for B0.** Using B-4's decomposition (`g_total = g_bg + α·w`), measure the background share of B0's total gradient to confirm the F4/H5 hypothesis that B0's `cos(H1)` is background-dominated.
6. **Re-state the success criterion scope.** Confirm whether P2 is meant as a per-condition test of S4/S8 only, or a pooled mean; the current verdict logic applies it to B0, which the protocol does not require to pass.

None of the above may be executed without a new phase and explicit L5 authorization. They are listed for human review.

---

## 9. Bottom Line

The negative `cos(H1) ≈ 0.39–0.46` result is **trustworthy as a procedural execution** but **does not validly test the protocol's central hypothesis**. The experiment measured the **B0 baseline through the S8 readout** and never evaluated the **S4/S8 subspace-readout conditions** that H1/P2 actually concern. Combined with a mislabeled constant `rho_c`, a tiny intervention dose, and the protocol's own admission that `cos(H1)` is background-dominated and condition-dependent, the most defensible causal conclusion is a **measurement-target / design-assumption failure** — not a demonstrated failure of the fixed-subspace-readout intervention, and not a broken readout. **H1 remains untested.** A corrected, adequately-powered evaluation of the S4/S8 conditions is the necessary next step before any scientific conclusion about the intervention can be drawn.

*End of read-only causal analysis.*
