# THEO SLM Phase 6E.16-R: Forensic Reconciliation & Proximal Mechanism Confirmation (v1)

- **Phase ID:** 6E.16-R
- **Status:** COMPLETE (read-only audit)
- **Date (UTC):** 2026-08-16
- **Scope:** Reconciliation audit of Phases 6E.13, 6E.14, 6E.15, 6E.16
- **Report author:** theo-slm forensic audit (opencode agent), read-only
- **Machine-readable provenance:** `theo-data/datasets/theo_slm_v0_artifacts/phase-6e16r/phase-6e16r-forensic-reconciliation-provenance.json`

---

## 1. Governance & Scope Reconciliation

| Phase | Claims Made | Audit Verdict |
|------|-------------|---------------|
| 6E.12 | Collapse onset t\* = 15 (Run A and Run B) | **REPRODUCED.** `collapse_onset_analysis.json` records t\*=15 for both runs. |
| 6E.13 | POS/ABS decision-gradient conflict at module H1; conflict emerges step 4, locks by step 6 | **REPRODUCED.** `autoregressive_formulation_analysis.json` cos(H1): 0.2588 (t0) -> -0.9312 (t4) -> -0.9968 (t6); `temporal_event_ordering.json` first severe conflict step 4, near-perfect opposition step 6, margin oscillation step 7, damped transition step 14, collapse step 15. |
| 6E.14 | Scalar-margin Jacobian Gram is near-rank-1 across trajectory | **REPRODUCED** (see §2). Doc-level k-NN claim **NOT REPRODUCED** (see §5). |
| 6E.15 | Scalar-margin sensitivity bottleneck; readout w_delta geometry; Cohen's d inversion | **REPRODUCED** on the 6E.15 trajectory; inversion **NOT REPRODUCED** on the 6E.16 trajectory (see §5, D2). |
| 6E.16 | Readout/output-contrast bottleneck is proximal; scalar rank ~1 vs joint rank >> 1 | **REPRODUCED** (see §2). |

**Trajectory comparability caveat (D1).** Each phase re-simulates its own training
trajectory with a *different* configuration. Verified from `run_phase_6e14_*.py` /
`run_phase_6e15_*.py` / `run_phase_6e16_*.py`:

| Config | 6E.14 / 6E.15 | 6E.16 |
|--------|---------------|-------|
| dtype | torch.bfloat16 | torch.float32 |
| lora_dropout | 0.05 | 0.0 |
| weight_decay | 0.01 | 0.0 |
| grad_accum_steps | 2 | 1 |
| data order | shuffled (seeded) | sequential |
| batch_size | 4 | 4 |
| LoRA r / alpha | 16 / 32 | 16 / 32 |
| optimizer / lr | AdamW / 1e-4 | AdamW / 1e-4 |
| base model | Qwen2.5-0.5B-Instruct | Qwen2.5-0.5B-Instruct |

Checkpoints at the same nominal step index are therefore **different parameter states**
across phases. Only step 0 (base model) is hash-verified identical
(`fdf756fa...fb7fe`). All cross-phase comparisons below respect this; where trends
differ, the discrepancy is reported (§5), not silently harmonized.

---

## 2. Scalar-Margin vs Multi-Output Rank: Verification Report

Inputs: `phase-6e16/multi_output_jacobian_spectrum.json` (recorded spectra per checkpoint,
per modality) and `phase-6e16/metric_reconciliation_standard.json`. Recomputations used
recorded top-5 spectral distributions and the participation-ratio lower bound
`r_eff >= (sum p_i)^2 / sum p_i^2`.

| step | scalar margin r_eff | raw joint r_eff | row-normed joint r_eff | within-token r_eff range | across-token r_eff |
|------|--------------------:|----------------:|-----------------------:|-------------------------:|-------------------:|
| 0    | 1.1526 | 6.611 | **7.1705** | 1.08 – 1.18 | 5.9393 |
| 4    | 1.1535 | 2.1601 | **4.0602** | 1.09 – 1.80 | 2.0036 |
| 6    | 1.1993 | 1.7303 | **3.4062** | 1.10 – 1.75 | 1.5413 |
| 14   | 1.4820 | 1.8281 | **3.6204** | 1.20 – 2.15 | 1.4648 |
| 15   | 1.5026 | 1.9191 | **3.8874** | 1.23 – 2.42 | 1.5235 |

**Verdict: REPRODUCED.** The scalar PROPOSE–ABSTAIN margin sensitivity is near-rank-1
(r_eff 1.15–1.50; top-1 spectral share 0.81–0.93) at every checkpoint, while the
row-normalized joint multi-output sensitivity is materially higher-dimensional
(r_eff 3.41–7.17) at every checkpoint. The raw (non-normalized) joint r_eff drops toward
~1.7 by t=6, but this is dominated by token-norm magnitude imbalance (the PROPOSE token
dominates the total logit mass), which is exactly why row normalization is the correct
comparison and why the raw joint number must not be read as "the LoRA is globally rank-1."

Internal-consistency recomputation:
- 6E.14 Gram, t=15: recorded r_eff 1.0429, top-5 recomputed bound 1.0279 (consistent; remaining 11 eigenvalues raise r_eff slightly).
- 6E.16 joint, t=6: raw recorded 1.7303 vs top-5 bound 1.7323; row-normed recorded 3.4062 vs bound 3.4283 (consistent).

**Conclusion: scalar-margin rank-1 does NOT generalize to the multi-output tangent space.**
Claims built on scalar-margin rank (6E.14, 6E.15) are valid only for the scalar decision
contrast; claims that "LoRA is globally rank-1" are NOT supported by the joint spectrum.

---

## 3. Readout Geometry Reconfirmation (t = 0, 4, 6, 14, 15)

Three independent discriminative constructions (centroid difference, Fisher LDA, linear
SVM) were fit on the N=16 POS/ABS panel and aligned against the readout vector
`w_delta = w_PRO - w_ABS` (unit-normalized). PCA axis is not preserved in any phase
artifact and could not be recomputed; all other constructions were recomputed from
`phase-6e16/diagnostic_basis_alignment_trajectory.json`.

| step | cos(centroid, w_delta) | cos(LDA, w_delta) | cos(SVM, w_delta) | max \|cos\| |
|------|----------------------:|------------------:|------------------:|------------:|
| 0    | 0.0149 | 0.0055 | 0.0348 | 0.0348 |
| 4    | 0.0148 | 0.0117 | 0.0203 | 0.0203 |
| 6    | 0.0178 | 0.0096 | 0.0044 | 0.0178 |
| 14   | 0.0235 | 0.0131 | 0.0304 | 0.0304 |
| 15   | 0.0223 | 0.0133 | 0.0304 | 0.0304 |

**Verdict: ROBUST NEAR-ORTHOGONALITY.** Over 3 constructions x 5 checkpoints the maximum
|cos| is 0.0348. The misalignment is not an artifact of a single discriminant: three
independent directions that each separate POS from ABS are all nearly orthogonal to the
frozen readout vector. Readout vector norm is constant (0.5642) while centroid separation
norm grows 16.3 -> 48.8 (6E.15 `readout_vector_w_delta_geometry.json`), so the 
misalignment grows *relatively* even as absolute hidden separation grows.

---

## 4. Temporal Ordering

From `phase-6e16/trajectory_event_ordering_chronology.json` plus 6E.12/6E.13 event
telemetry:

| step | event | state at step |
|------|-------|---------------|
| 0    | Readout misalignment **already present** (cos up to 0.0348); no gradient conflict yet (cos(H1)=+0.26) | margin d=0.53; scalar r_eff 1.15; joint r_eff 7.17 |
| 4    | Decision-gradient conflict **emerges** (cos(H1) -> -0.93) | d=0.29; scalar 1.15; joint 4.06 |
| 6    | Conflict **locks** (cos(H1) = -0.997) | d=0.25; scalar 1.20; joint 3.41 |
| 7    | Margin oscillation begins (proposal rate 0% <-> 100%) | — |
| 14   | Damped transition (proposal rate 36.5%) | scalar 1.48; joint 3.62 |
| 15   | **Collapse onset t\*=15** (proposal rate 0.0%) | scalar 1.50; joint 3.89; max \|cos\| 0.0304 |

**Classification: READOUT MISALIGNMENT PRECEDES CONFLICT.**

- `first_readout_misalignment_step = 0` (present before any optimizer step and before any
  conflict appears).
- `first_margin_inversion_step = null` in the 6E.16 chronology (no margin inversion on this
  trajectory; the 6E.15 inversion is trajectory-specific, see D2).

**Causal caveat.** Misalignment at initialization does not *immediately* produce conflict:
step-0 decision gradients are aligned (cos=+0.26) and the conflict requires the
training-induced concentration of margin sensitivity (t=4–6). The causal chain
"pre-existing readout misalignment + margin-sensitivity concentration -> collapse" is
therefore **INFERRED from empirical evidence**, not demonstrated by intervention
(read-only constraint). The stronger claim "misalignment causes collapse" is not
established by this evidence alone.

---

## 5. Representation-Separability Reconciliation

Documented dimensions and values:

| Dimension | Value |
|-----------|-------|
| Space | full hidden H and orthogonal subspace h_perp (orthogonal to w_delta) |
| Panels | N=16 (8 POS + 8 ABS); N=24 (8 POS + 8 ABS + 8 NEG) |
| Checkpoints | t in {0, 4, 6, 14, 15} |
| Distance | Euclidean (k-NN) |
| k | {1, 3, 5} |
| Estimators | resubstitution k-NN; LOO-CV k-NN; 3-fold stratified linear SVM |

Reconciled values (all from preserved artifacts):

| Metric | 6E.14 artifact (N=24) | 6E.15 artifact (N=16) | 6E.16 artifact (N=16/N=24) |
|--------|----------------------:|----------------------:|---------------------------:|
| resub k-NN1 purity | 0.75 | 0.75 (h_perp) | 0.75 |
| LOO k-NN1 | ~0.33–0.38 | — | 0.31–0.44 |
| 3-fold SVM CV | 0.583–0.625 | — | 0.31–0.63 |
| centroid separation | 13.69 -> 3.19 | — | — |
| Fisher ratio | reproduced | reproduced | reproduced |

**D2 — 6E.14 doc-level claim (NOT supported by preserved artifacts).** The 6E.14 report
states k-NN purity = 100.0% at all steps and centroid distance 14.438 -> 16.850. The
preserved `temporal_decision_context_separability.json` records **k-NN1 = 0.75** at all
steps and centroid distance **13.6941 -> 3.1922** (decreasing). 6E.16's own attribution
of "100%" to 6E.14 repeats this unsupported value. **Classification: DOC-LEVEL
OVERSTATEMENT; the reproducible resubstitution value is 75%.** This value must NOT be
interpreted as generalization: LOO-CV is 31–44% (near/at chance), i.e., no genuine
generalizing separation is demonstrated at the output-representation level.

**D3 — Margin inversion discrepancy (6E.15 only).** 6E.15 `readout_vector_w_delta_geometry.json`
reports Cohen's d -> -0.21 (margin inversion) at t=14–15 on the 6E.15 trajectory. The
6E.16 trajectory shows Cohen's d staying positive (v1 basis, +0.30 at t=15). **Not a
contradiction:** trajectories differ in configuration (D1). **Classification: MEASURED IN
6E.15 TRAJECTORY ONLY; NOT REPRODUCED IN 6E.16 TRAJECTORY.**

**Verdict:** 6E.15 (0.75 orthogonal-subspace, N=16) and 6E.16 (0.75 resub, N=16/N=24)
**agree**. The 6E.14 doc "100%" is unsupported. The genuine, reproducible statement is:
separability is *weak* (resub 0.75, CV ~0.3–0.6) but **persists through the
conflict/collapse window** — it does not collapse.

---

## 6. Proximal Mechanism Determination (M1–M6)

| # | Candidate | Status | Basis |
|---|-----------|--------|-------|
| M1 | Global LoRA capacity bottleneck | **DOWNGRADED** | Row-normalized joint r_eff 3.41–7.17 disproves a globally rank-1 tangent space (§2). |
| M2 | Cross-module cancellation | **REJECTED** | Every individual LoRA group is independently near-rank-1 (max 1.23); progressive aggregation is rank-invariant (1.01–1.23). |
| M3 | Representation collapse | **NOT SUPPORTED** | Separability preserved through conflict/collapse: resub k-NN 0.75 stable; h_perp purity 0.75; layer-23 separability persists; SVM CV 0.31–0.63. |
| M4 | **Scalar decision-contrast bottleneck** | **LEADING PROXIMAL MECHANISM** | Joint output sensitivity higher-rank (3.4–7.2 vs 1.15–1.50); hidden discriminative directions exist (LDA/SVM/centroid all separate, all nearly orthogonal to w_delta, max \|cos\| 0.035). The fixed vocabulary contrast cannot read out the (available, higher-dimensional) discriminative geometry. |
| M5 | Broader architectural readout bottleneck | PARTIALLY SUPPORTED (subsidiary) | Per-sample output tangent rank collapses 5.87 -> ~1.5–1.8 by t=6; POS/ABS tangent subspaces nearly aligned (top principal cosine ~0.99). But across-token joint rank stays >3.4, so not all output directions are low-dimensional. |
| M6 | Mixed mechanism | NOT SELECTED | A single mechanism (M4) carries the independent support. |

**M4 is the leading proximal mechanism**: the measured facts are (a) higher-rank joint
sensitivity exists, (b) multiple valid discriminative directions exist in hidden space,
(c) those directions are nearly orthogonal to the frozen decision readout, and (d) this
structure is present at initialization and persists through collapse. The collapse is the
marginalization of the decision contrast, not a loss of discriminative information.

---

## 7. Evidence Table

| # | Claim | Evidence (artifact) | Status |
|---|-------|---------------------|--------|
| 1 | POS/ABS decision gradients strongly oppose | 6E.13 `autoregressive_formulation_analysis.json` cos(H1) -0.997; 6E.12 run telemetry | ACTUALLY MEASURED |
| 2 | Scalar margin sensitivity is near-rank-1 | 6E.14 Gram 1.0429–1.1601; 6E.15 groups 1.0113–1.2293; 6E.16 scalar 1.1526–1.5026 | ACTUALLY MEASURED |
| 3 | Joint multi-output sensitivity materially higher-rank | 6E.16 `multi_output_jacobian_spectrum.json` row-normed 3.4062–7.1705 vs scalar 1.1526–1.5026; recomputed | ACTUALLY MEASURED (REPRODUCED) |
| 4 | LoRA is globally rank-1 | contradicting joint spectrum r_eff >= 3.41 | NOT SUPPORTED |
| 5 | Cross-module cancellation causes rank collapse | 6E.15 `parameter_group_jacobian_decomposition.json`: per-group rank 1.01–1.23; aggregation rank-invariant | NOT SUPPORTED |
| 6 | Hidden representations collapse first | 6E.13 `representation_geometry.json`; 6E.14/15/16 separability preserved | NOT SUPPORTED |
| 7 | w_delta misaligned with hidden discriminative geometry | 6E.16 `diagnostic_basis_alignment_trajectory.json`; max \|cos\| 0.0348 over 3 constructions x 5 checkpoints; recomputed | ACTUALLY MEASURED (REPRODUCED) |
| 8 | Readout/output-contrast mismatch is proximal | synthesis of 6E.14–6E.16 | INFERRED FROM EMPIRICAL EVIDENCE |
| 9 | 6E.14 k-NN purity = 100%, centroid > 14.4 stable | contradicted by preserved artifact (0.75; 13.69 -> 3.19) | NOT SUPPORTED BY PRESERVED ARTIFACTS |
| 10 | Margin inversion (Cohen's d -> -0.21) at t=14–15 | 6E.15 trajectory only; absent on 6E.16 trajectory | MEASURED IN 6E.15 ONLY |
| 11 | Same-nominal-step checkpoints across phases are identical | phase scripts re-simulate with different configs; only step 0 hash-identical | NOT SUPPORTED |

---

## 8. Final Scientific Conclusion

**Two statements (replace the stronger claim with the better-supported one):**

> **Too strong:** "The LoRA module collapses to rank 1, so training capacity is globally
> exhausted; the decision-gate collapse is caused by the optimizer driving the model into
> a rank-limited basin, and margin inversion at t=15 proves the readout inverted."

> **Better-supported current statement:** "Within the evaluated trajectory and local LoRA
> parameterization, the PROPOSE–ABSTAIN scalar decision contrast exhibits a near-one-
> dimensional sensitivity (r_eff 1.15–1.50) that is poorly aligned with the measured
> POS/ABS discriminative geometry (max |cos| <= 0.035), while the broader local output
> sensitivity is materially higher-dimensional (row-normalized joint r_eff 3.4–7.2). This
> makes the fixed vocabulary decision contrast a leading proximal bottleneck: the model
> retains discriminative information (separability does not collapse) but cannot express it
> through the frozen decision readout, and repeated conflicting optimization (cos(H1) ->
> -0.997) marginalizes the PROPOSE margin until the gate locks at collapse onset t\*=15."

**Scope of conclusion:** Qwen2.5-0.5B-Instruct base (`fdf756fa...fb7fe`), r=16 LoRA,
checkpoint set {0,4,6,14,15}, N=16/N=24 diagnostic panels, fixed vocabulary readout
w_PRO - w_ABS. Generalization to other architectures, ranks, or readout constructions is
**NOT** established.

---

## 9. Provenance

- Input artifacts and their SHA-256 hashes: `phase-6e16r-forensic-reconciliation-provenance.json` (26 input artifacts from phases 6E.12–6E.16, all hash-verified).
- Base model sha256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
- Corpus sha256: `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0`.
- Audit method: read-only; no training, no model execution, no checkpoint access; all
  quantities recomputed from preserved JSON artifacts (spectral summaries, alignment
  trajectories, chronology, metric-reconciliation records).
- Data-loss note: raw Jacobian matrices were not persisted by phases 6E.13–6E.16;
  internal-consistency recomputations therefore bound, rather than exactly reproduce,
  the recorded r_eff values (all bounds consistent, §2).

## 10. Canonical Interpretation Record (v1)

This section is the canonical interpretation of Phases 6E.14–6E.16, as accepted by human
review of Phase 6E.16-R. It supersedes any narrative in the individual phase reports that
is inconsistent with it. Future phases and reports MUST cite this record as the
authoritative interpretation of 6E.14–6E.16 and MUST NOT silently reintroduce the
discrepancies recorded in §5 (D1–D3).

### 10.1 Corrected canonical interpretation

> The measured POS/ABS failure is NOT explained by a globally rank-1 LoRA sensitivity
> space. Instead, the specific PROPOSE − ABSTAIN scalar decision contrast has
> substantially lower effective dimensionality than the broader multi-output tangent
> geometry and is robustly misaligned with independently derived discriminative directions
> in hidden space.

Key figures (all checkpoints t in {0,4,6,14,15}):

| Quantity | Value |
|----------|-------|
| r_eff (scalar margin) | 1.15 – 1.50 |
| r_eff (row-normalized joint output) | 3.41 – 7.17 |
| max \|cos(w_delta, u_disc)\| (centroid/LDA/SVM x 5 checkpoints) | 0.0348 |

### 10.2 Three-tier classification of the evidentiary record

**ACTUALLY MEASURED**
1. Strong POS/ABS gradient opposition for the investigated scalar decision objective.
2. Near-low-dimensional scalar PROPOSE − ABSTAIN sensitivity.
3. Materially higher-dimensional multi-output local sensitivity.
4. Robust near-orthogonality between w_delta and the tested hidden-space discriminative
   directions.
5. Readout misalignment is present before the measured emergence of gradient conflict
   within the reconciled trajectory.
6. The earlier global rank-1 LoRA interpretation is not supported.

**INFERRED**
1. The fixed scalar decision contrast is the leading proximal bottleneck (M4).
2. Readout misalignment likely contributes to the later conflict mechanism.
3. M4 is currently the best explanatory mechanism.

**NOT ESTABLISHED**
1. That readout misalignment is causally sufficient for collapse.
2. That changing the output/readout formulation will fix training.
3. Which alternative decision formulation would work.
4. Whether the issue generalizes beyond this dataset, trajectory, tokenization, and model
   configuration.
5. Whether a learned classifier/readout, different verbalizer, multi-token decision
   formulation, or other intervention would improve anything.

### 10.3 Governance rules

- **Trajectory identity (D1):** Phases 6E.14/6E.15 and 6E.16 re-simulate the trajectory
  with different configurations (bfloat16 vs float32, lora_dropout 0.05 vs 0.0,
  weight_decay 0.01 vs 0.0, grad_accum 2 vs 1, shuffled vs sequential). Checkpoints at
  the same nominal step index are DIFFERENT parameter states across phases; only step 0
  (base model, hash-verified) is identical. Any future causal analysis MUST respect
  trajectory identity and MUST NOT merge historical metrics into a single seamless
  timeline keyed only by checkpoint number.
- **D2:** The 6E.14 report's "100% purity" claim is unsupported by the preserved artifact
  (resub k-NN1 = 0.75; LOO 31–44%). The reproducible resubstitution value is 75% and must
  NOT be cited as generalization.
- **D3:** The 6E.15 Cohen's d margin inversion (-0.21) is trajectory-specific and was NOT
  reproduced on the 6E.16 trajectory.
- **Causality labeling:** `READOUT MISALIGNMENT PRECEDES CONFLICT` remains an INFERRED
  temporal ordering (`≺`), not a demonstrated causal implication (`⇒`). The strength label
  must be carried in every future report.
- **Gate:** Phase 6E.17 (or any causal intervention) must not begin until a separate,
  human-reviewed intervention authorization protocol is approved.

### 10.4 First justified causal hypothesis (for future testing — NOT a conclusion)

> **H-Causal:** The failure arises substantially because the fixed PROPOSE/ABSTAIN
> vocabulary contrast is poorly aligned with the task-discriminative representation
> geometry; modifying the decision formulation to provide an appropriately aligned
> trainable or alternative output geometry should reduce the measured scalar conflict and
> improve simultaneous POS/ABS optimization.

Drafted at: `theo-slm-causal-intervention-authorization-protocol-draft-v2.md` (DRAFT,
AMENDED, NOT AUTHORIZED). Execution requires explicit human authorization under that
protocol. Current human review status: AMEND AND RESUBMIT (2026-08-16); pending final
authorization via `theo-slm-phase-6e17-pre-execution-authorization-review-v1.md`.

## 11. Open Items for Human Review

1. **D2** — 6E.14 report "100% purity" vs preserved artifact 0.75: now recorded in the
   Canonical Interpretation Record (§10.3). Recommended action: correct the narrative to
   "resubstitution 75%; LOO-CV 31–44%" wherever it is quoted.
2. **D3** — 6E.15 Cohen's d inversion not reproduced on 6E.16 trajectory: now recorded in
   §10.3. Recommended action: add an explicit trajectory-config footnote in both reports.
3. **Causality** — readout-misalignment-first is inferred, not manipulated. A causal
   (intervention) test is a natural candidate for a future phase; it was NOT run here
   (read-only).
4. **Intervention authorization** — the causal intervention protocol draft
   (`theo-slm-causal-intervention-authorization-protocol-draft-v1.md`) is DESIGN ONLY and
   NOT AUTHORIZED. The next human decision is whether the evidence threshold is sufficient
   to authorize one causal intervention experiment under that protocol. Phase 6E.17 must
   not start until that decision is made and the protocol is approved.
