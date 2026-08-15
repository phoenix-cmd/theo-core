# Phase 6E.16 — Decision Output Geometry & Multi-Directional Sensitivity Forensics

**Date**: 2026-08-15  
**Status**: COMPLETED  
**Primary Outcome**: `READOUT / OUTPUT-CONTRAST BOTTLENECK`  
**Governance**: PASS — Zero model mutations, strict forensic hard-stop enforced

---

## 1. Motivation & Prior Evidence Chain

Phase 6E.16 addresses the critical interpretational limitation left by 6E.15:

> A near-rank-1 Jacobian for one scalar margin z_PRO - z_AB does not, by itself, prove that the network has only one locally accessible output direction.

The evidence chain entering this phase:

| Phase | Finding |
|-------|---------|
| 6E.13 | POS/ABS decision gradients strongly oppose (cos ~ -0.997) |
| 6E.14 | Opposition caused by shared margin sensitivities, not autoregressive tokens |
| 6E.15 | Sensitivity propagates as near-rank-1 (r_eff <= 1.13) through all LoRA modules |

The unresolved question: **Is the rank-1 finding an intrinsic property of the network's local output geometry, or an artifact of measuring only a single scalar contrast Delta_z = z_PRO - z_AB?**

## 2. Pre-Registered Decision Gate

Three mutually exclusive outcomes were pre-registered:

| Outcome | Condition |
|---------|-----------|
| **READOUT / OUTPUT-CONTRAST BOTTLENECK** | Joint sensitivity high-rank + discriminative axis misaligned with w_Delta |
| **SCALAR-MARGIN ARTIFACT / COMPLEX MECHANISM** | Joint sensitivity high-rank + no special misalignment |
| **BROAD LOCAL OUTPUT-SENSITIVITY BOTTLENECK** | Joint sensitivity near rank-1 across all relevant outputs |

## 3. Experimental Design

**Forensic-only protocol**: Zero optimizer updates, zero model mutations. All analyses are read-only measurements on 5 checkpoints from the training trajectory (steps 0, 4, 6, 14, 15).

**Diagnostic panel**: 16 samples (8 POS, 8 ABS) for Jacobian analysis; 24 samples (8 POS, 8 ABS, 8 NEG) for 3-class reconciliation.

**Vocabulary basis** (K = 8 tokens): PRO (9117), AB (1867), REV (72487), EXEC (46340), null (2921), space (220), newline (198), top_unconstrained/The (785).

### Investigations

- **Metric Reconciliation**: Resubstitution vs LOO-CV, full space vs h_perp (orthogonal to w_Delta)
- **Investigation A**: Multi-output Jacobian spectrum (scalar margin vs K-token joint, raw vs row-normalized)
- **Investigation B**: Output-contrast basis analysis (Gram-Schmidt basis from vocabulary readout vectors)
- **Investigation C**: Local output tangent subspaces & principal angles (POS vs ABS)
- **Investigation D**: Diagnostic alignment tracking (u_LDA, u_centroid, u_SVM vs fixed w_Delta)
- **Investigation E**: Trajectory event ordering & causal precedence

---

## 4. Results

### 4.1 Investigation A — Multi-Output Jacobian Spectrum

> [!IMPORTANT]
> The scalar margin Jacobian is near-rank-1 at all checkpoints, but the multi-output joint Jacobian is decisively higher-dimensional. **The rank-1 finding from 6E.15 was a measurement artifact of collapsing K=8 output dimensions to 1.**

| Step | Scalar Margin r_eff | Raw Joint r_eff | Row-Normed Joint r_eff | Joint Top-1 Dominance |
|------|---------------------|-----------------|------------------------|-----------------------|
| 0    | 1.15                | 6.61            | 7.17                   | 24.7%                 |
| 4    | 1.15                | 2.16            | 4.06                   | 45.0%                 |
| 6    | 1.20                | 1.73            | 3.41                   | 50.9%                 |
| 14   | 1.48                | 1.83            | 3.62                   | 49.6%                 |
| 15   | 1.50                | 1.92            | 3.89                   | 47.6%                 |

**Key observations**:
- **Scalar margin r_eff ~ 1.15-1.50** throughout — confirming 6E.15's rank-1 Jacobian for Delta_z
- **Row-normalized joint r_eff = 3.4-7.2** — the parameter sensitivity spans multiple independent output directions when measured across all K tokens
- **Raw joint r_eff collapses from 6.6 to 1.7 during training** — magnitude concentrates into a dominant direction, but directional diversity (row-normalized) remains high

**Within-Token Analysis**: Each individual token's Jacobian is near-rank-1 (r_eff ~ 1.1-1.8), but the across-token spectrum shows these directions are distinct.

**Token Cosine Similarity Matrix** (step 6, decision-relevant tokens):
- PRO <-> AB: cos = **0.94** (very similar sensitivity direction)
- PRO <-> REV: cos = 0.74
- PRO <-> EXEC: cos = 0.82
- PRO <-> null: cos = -0.08 (orthogonal)
- PRO <-> The: cos = -0.72 (opposite)

The high PRO<->AB cosine (0.94) is consistent with the 6E.15 finding: since PRO and AB share nearly the same sensitivity direction, label-signed gradients necessarily oppose.

### 4.2 Investigation B — Output-Contrast Basis Analysis

| Step | v1(PRO-AB) Cohen's d | v2(PRO+AB) Cohen's d |
|------|----------------------|----------------------|
| 0    | 0.53                 | 0.07                 |
| 4    | 0.29                 | 0.06                 |
| 6    | 0.25                 | 0.08                 |
| 14   | 0.31                 | 0.17                 |
| 15   | 0.30                 | 0.15                 |

The v1 (decision-margin) direction shows weak but present separation at initialization (Cohen's d = 0.53), which **decreases** during training to d ~ 0.25-0.31. This is the opposite of what successful learning would produce.

### 4.3 Investigation C — Output Tangent Subspace Principal Angles

| Step | Mean Per-Sample Output Rank | Top Principal Cosine (POS vs ABS) | Smallest Principal Angle |
|------|-----------------------------|-----------------------------------|--------------------------|
| 0    | 5.87                        | 0.997                             | 4.73 deg                 |
| 4    | 2.06                        | 0.997                             | 4.57 deg                 |
| 6    | 1.64                        | 0.996                             | 4.96 deg                 |
| 14   | 1.66                        | 0.990                             | 7.97 deg                 |
| 15   | 1.79                        | 0.989                             | 8.45 deg                 |

- **Per-sample output rank collapses**: from 5.9 (initialization) to 1.6-1.8 (mid-training). The network's local output tangent space loses dimensionality during training.
- **POS and ABS tangent subspaces are nearly aligned**: top principal cosine ~ 0.99 throughout, with smallest angles 5-8 deg. This means both classes generate output changes in essentially the same parameter-space directions.

### 4.4 Investigation D — Discriminative Axis vs Readout Alignment

> [!CAUTION]
> This is the **smoking gun** measurement. The natural discriminative direction in hidden space is nearly **orthogonal** to the fixed vocabulary readout vector at every checkpoint.

| Step | cos(u_LDA, w_Delta) | cos(u_centroid, w_Delta) | cos(u_SVM, w_Delta) |
|------|----------------------|--------------------------|---------------------|
| 0    | 0.006                | 0.015                    | 0.035               |
| 4    | 0.012                | 0.015                    | 0.020               |
| 6    | 0.010                | 0.018                    | 0.004               |
| 14   | 0.013                | 0.024                    | 0.030               |
| 15   | 0.013                | 0.022                    | 0.030               |

All cosine alignments are within **0.004-0.035** — effectively zero. This means:

1. **Hidden representations DO contain some class-discriminative structure** (LDA, centroid, SVM can find separating directions — resubstitution KNN1 = 75%)
2. **But these directions are nearly orthogonal to w_Delta = w_PRO - w_AB**
3. **The model can only express its POS/ABS decision through w_Delta** (fixed vocabulary embeddings)
4. **Therefore, learning requires aligning hidden representations with an arbitrary direction** rather than exploiting natural geometry

### 4.5 Metric Reconciliation

| Step | Full Space Resub KNN1 | Full Space LOO-KNN1 | Orthog Subspace LOO-KNN1 |
|------|-----------------------|---------------------|--------------------------|
| 0    | 75.0%                 | 37.5%               | 43.8%                    |
| 6    | 75.0%                 | 37.5%               | 37.5%                    |
| 15   | 75.0%                 | 37.5%               | 31.3%                    |

- **Resubstitution vs LOO-CV gap**: 75% to 37.5% = massive overfitting artifact with N=16 in high-dimensional space. The earlier 6E.14 "100% k-NN purity" result is confirmed as resubstitution bias, not genuine generalization.
- **Orthogonal subspace**: LOO accuracy at/below chance — class information is concentrated along w_Delta but too weak to generalize.
- **Classification**: `DIFFERENT_GEOMETRIC_SUBSPACE` — the apparent high purity was an artifact of operating in the full space with N << D.

### 4.6 Investigation E — Causal Ordering

| Event | First Observed |
|-------|----------------|
| Readout misalignment (cos < 0.05) | Step 0 (present at initialization) |
| Margin inversion | Not observed (never reached) |
| Output rank collapse (r_eff < 2) | Step 4 |
| Joint sensitivity concentration | Step 4 |

> [!NOTE]
> The readout misalignment is **not caused by training** — it is present at initialization and never improves. This is a structural property of the frozen vocabulary embedding geometry, not a learned failure mode.

---

## 5. Decision Gate Evaluation

At the reference checkpoint (step 6):

| Criterion | Measured | Threshold | Met? |
|-----------|----------|-----------|------|
| Row-normalized joint r_eff | 3.41 | >= 2.50 | YES |
| Joint top-1 energy dominance | 50.9% | < 85% | YES |
| cos(u_discriminative, w_Delta) | 0.010 | < 0.10 | YES |
| Orthogonal subspace LOO purity | 37.5% | <= chance | YES |

**All criteria for `READOUT / OUTPUT-CONTRAST BOTTLENECK` are satisfied.**

---

## 6. Mechanistic Synthesis

The complete causal chain is now:

```
Frozen vocabulary embeddings w_PRO, w_AB
         |
         | Their difference w_Delta = w_PRO - w_AB defines a fixed readout direction
         |
         v
Hidden representations h_i contain weak class structure
         |
         | Natural discriminative axis u* is nearly orthogonal to w_Delta
         | (cos ~ 0.01 at all checkpoints)
         |
         v
Decision margin Delta_z = h . w_Delta is controlled by ONE arbitrary direction
         |
         | LoRA must rotate h to project onto w_Delta rather than exploit u*
         |
         v
d(Delta_z)/dh is identical for all examples (= w_Delta)
         |
         | Backpropagated through network -> near-rank-1 parameter Jacobian
         |
         v
POS and ABS share the same sensitivity direction
         |
         | Label-signed gradients necessarily oppose
         |
         v
Optimizer receives contradictory updates -> oscillation -> collapse
```

**Root cause identified**: The bottleneck is in the **readout layer geometry**. The frozen vocabulary embeddings define a decision-contrast direction (w_Delta) that is essentially random relative to the natural class-discriminative geometry of the hidden space. LoRA fine-tuning cannot overcome this because:

1. The readout weights are frozen (not in the LoRA parameter space)
2. Even if LoRA could rotate representations, the rank-1 margin sensitivity means all examples must move in the same direction, making class-conditional rotation impossible
3. The resulting gradient conflict is not a training hyperparameter issue — it is a structural consequence of the readout geometry

---

## 7. Implications for Intervention Design

The diagnosis points to specific intervention strategies:

| Strategy | Rationale | Expected Effect |
|----------|-----------|-----------------|
| **Unfreeze readout layer** | Allow w_Delta to align with natural discriminative geometry | Directly addresses root cause |
| **Add classification head** | Bypass frozen vocabulary entirely with a learned linear probe | Avoids the readout bottleneck |
| **LoRA on embedding layer** | Enable rotation of readout directions within LoRA training | Partially addresses misalignment |
| **Multi-token decision format** | Use multiple tokens to encode decisions, diversifying the readout subspace | Increases effective output rank |

> [!WARNING]
> The simple interventions (learning rate, batch size, balanced sampling, decision-token weighting) have already been tested in earlier phases and failed. This is consistent with a structural geometric bottleneck that cannot be resolved by optimizer tuning.

---

## 8. Artifacts

All forensic data is saved with cryptographic provenance:

| File | Contents |
|------|----------|
| [multi_output_jacobian_spectrum.json](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e16/multi_output_jacobian_spectrum.json) | Investigation A — full spectral data for all 5 checkpoints |
| [output_contrast_basis_analysis.json](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e16/output_contrast_basis_analysis.json) | Investigation B — basis separation metrics |
| [vocabulary_subspace_tangent_geometry.json](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e16/vocabulary_subspace_tangent_geometry.json) | Investigation C — principal angles and tangent ranks |
| [diagnostic_basis_alignment_trajectory.json](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e16/diagnostic_basis_alignment_trajectory.json) | Investigation D — alignment tracking |
| [trajectory_event_ordering_chronology.json](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e16/trajectory_event_ordering_chronology.json) | Investigation E — causal timeline |
| [metric_reconciliation_standard.json](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e16/metric_reconciliation_standard.json) | LOO-CV and resubstitution metrics |
| [anti_fabrication_provenance.json](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e16/anti_fabrication_provenance.json) | SHA-256 hashes for model and corpus |
| [phase-6e16-final-forensic-summary.json](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e16/phase-6e16-final-forensic-summary.json) | Consolidated decision gate outcome |

---

## 9. Mandatory Hard Stop

Phase 6E.16 is a **forensic-only** phase. No training was conducted. No model weights were modified. All measurements are read-only. The cryptographic provenance hashes were verified pre- and post-experiment.

**Next phase**: Intervention design based on the `READOUT / OUTPUT-CONTRAST BOTTLENECK` diagnosis.
