# THEO SLM Fixed Subspace Readout Authorization Protocol — DRAFT v1.0

- **Status:** DRAFT — DESIGN ONLY. **NOT AUTHORIZED FOR EXECUTION.**
- **Phase:** Phase 6E.20 (Fixed Low-Dimensional & Subspace Readout Architectures).
- **Authority:** Requires formal pre-execution audit and explicit human authorization. No model execution, training, fine-tuning, optimizer steps, or checkpoint generation is authorized by this document.
- **Basis & Provenance:** Phase 6E.18 Execution Findings (`phase-6e18-causal-intervention`) and Phase 6E.19 Post-Execution Forensic Adjudication Record (`theo-slm-phase-6e18-post-execution-forensic-analysis-v1.md`).
- **Core Problem Addressed:** Decoupling of gradient compatibility ($\cos(H1)$) and downstream decision utility (margin separation $d(15)$, balanced proposal rate) established in Phase 6E.18.

---

## 1. Scientific Motivation and Theoretical Framing

### 1.1 Empirical Baseline from Phase 6E.18
Phase 6E.18 established three fundamental empirical facts:
1. **Decoupling of Gradient Compatibility and Task Utility:** Maximizing gradient cosine ($\cos(H1)$) is mathematically distinct from maintaining discriminative decision utility. Condition T2 achieved near-perfect gradient alignment ($\cos(H1) = +0.9878$) by restricting updates to a rank-8 SVD subspace, but suffered from near-zero decision margin ($d(15) = +0.101$) and proposal decay ($19.2\%$).
2. **Superiority of Fixed Aligned Geometry (F1):** Condition F1 (fixed 1D Fisher LDA projection, $0$ trainable parameters) completely resolved gradient conflict ($\cos(H1) = +0.5831$), preserved strong margin ($d(15) = +1.813$), and maintained balanced proposal rates ($98.1\%$), outperforming trainable readouts (I1, C2).
3. **Trainable Readout Instability:** Trainable heads (I1, C2, V1) introduce non-stationarity, bias inflation, and parameter drift:
   - I1 drifted from balanced classification to $100\%$ proposal over-prediction.
   - C2 achieved positive gradient cosine ($+0.3066$) via capacity, but its margin inverted to $-0.169$ at step 20.
   - V1 experienced a late-training polar reversal, dropping from $100\%$ to $0.0\%$ proposal rate at step 20.

### 1.2 The Phase 6E.20 Research Question
> **Can a fixed, low-dimensional subspace readout architecture combine the superior decision utility and margin separation of Condition F1 ($d > 1.50$, balanced proposal rate) with the orthogonal noise filtering of Condition T2 ($\cos(H1) \ge +0.80$), without introducing trainable head confounds, bias drift, or optimization non-stationarity?**

---

## 2. Experimental Condition Matrix ($N=7$)

All candidate conditions in Phase 6E.20 carry **EXACTLY ZERO ($0$) TRAINABLE READOUT PARAMETERS**, completely isolating the representation-readout geometry from trainable head adaptation.

| Condition ID | Architecture Name | Readout Subspace Construction | Trainable Params | Fixed Orientation Form | Primary Mechanistic Isolation |
|---|---|---|---|---|---|
| **B0** | Matched Frozen Baseline | Full ambient space $\mathbb{R}^{896}$ | **0** | Frozen vocabulary margin $w_{\delta}$ | Baseline regression reference |
| **F1** | 1D Fixed Fisher LDA | 1D line spanned by $\hat{u}_{\text{LDA}}$ | **0** | $s = (s_{\text{scale}} \hat{u}_{\text{LDA}})^T h_L + b_{\text{fixed}}$ | 1D discriminative target replication |
| **S2** | Fixed Subspace ($k=2$) | $\text{span}\{\hat{u}_{\text{LDA}}, v_1^{\perp}\}$ | **0** | $s = (P_{S2} \hat{u}_{\text{LDA}})^T h_L + b_{\text{fixed}}$ | 2D subspace noise reduction |
| **S4** | Fixed Subspace ($k=4$) | $\text{span}\{\hat{u}_{\text{LDA}}, v_1^{\perp}, v_2^{\perp}, v_3^{\perp}\}$ | **0** | $s = (P_{S4} \hat{u}_{\text{LDA}})^T h_L + b_{\text{fixed}}$ | 4D subspace noise reduction |
| **S8** | Fixed Subspace ($k=8$) | $\text{span}\{\hat{u}_{\text{LDA}}, v_1^{\perp}, \dots, v_7^{\perp}\}$ | **0** | $s = (P_{S8} \hat{u}_{\text{LDA}})^T h_L + b_{\text{fixed}}$ | 8D subspace noise reduction (T2 dimensionality match) |
| **S8-Proj** | SVD Basis Projected LDA | Top-8 SVD subspace $U_8$ | **0** | $s = (\text{proj}_{U_8}(\hat{u}_{\text{LDA}}))^T h_L + b_{\text{fixed}}$ | Projection of LDA onto pure representation SVD basis |
| **C4-Rand** | Random Fixed Subspace Control | Random 4D subspace $R_4$ with $n_{\text{rand}} \in R_4$ | **0** | $s = (s_{\text{scale}} n_{\text{rand}})^T h_L + b_{\text{rand}}$ | Subspace restriction without discriminative alignment |

---

## 3. Mathematical Formulations & Prospective Derivation

### 3.1 Step-0 Discriminative Axis ($\hat{u}_{\text{LDA}}$)
Computed on the frozen step-0 $N=16$ panel hidden states $H_{16} \in \mathbb{R}^{16 \times 896}$:
$$\hat{u}_{\text{LDA}} = \frac{\Sigma_{\text{within}}^{-1} (\mu_{\text{POS}} - \mu_{\text{ABS}})}{\|\Sigma_{\text{within}}^{-1} (\mu_{\text{POS}} - \mu_{\text{ABS}})\|_2}$$
Scaling factor $s_{\text{scale}}$ and bias $b_{\text{fixed}}$ are computed identically to Phase 6E.18 (§3.8).

### 3.2 Gram-Schmidt Orthogonal Subspace Construction ($S_k$)
To construct a $k$-dimensional subspace containing the exact discriminative axis $\hat{u}_{\text{LDA}}$:
1. Compute the empirical covariance matrix $\Sigma_{\text{step0}}$ of centered hidden states $H_{16}$.
2. Project out the LDA component from $\Sigma_{\text{step0}}$:
   $$\Sigma^{\perp} = (I - \hat{u}_{\text{LDA}} \hat{u}_{\text{LDA}}^T) \Sigma_{\text{step0}} (I - \hat{u}_{\text{LDA}} \hat{u}_{\text{LDA}}^T)$$
3. Compute eigenvectors of $\Sigma^{\perp}$: $\{v_1^{\perp}, v_2^{\perp}, \dots, v_{k-1}^{\perp}\}$ corresponding to the largest eigenvalues.
4. The orthonormal basis matrix $V_k \in \mathbb{R}^{896 \times k}$ is defined as:
   $$V_k = [\hat{u}_{\text{LDA}} \quad v_1^{\perp} \quad v_2^{\perp} \quad \dots \quad v_{k-1}^{\perp}]$$
5. The fixed projection operator is $P_{Sk} = V_k V_k^T$.
6. Because $\hat{u}_{\text{LDA}} \in \text{col}(V_k)$, the effective readout vector is:
   $$w_{Sk} = s_{\text{scale}} P_{Sk} \hat{u}_{\text{LDA}} = s_{\text{scale}} \hat{u}_{\text{LDA}}$$
7. **Mechanistic Action:** In the forward pass, $w_{Sk}^T h_L = (s_{\text{scale}} \hat{u}_{\text{LDA}})^T h_L$. In the backward pass, gradients propagating from the readout are restricted to the subspace $\text{col}(V_k)$, filtering out all gradient components in the orthogonal complement $\text{col}(V_k)^{\perp}$.

### 3.3 Condition S8-Proj (Projection onto SVD Principal Subspace)
Let $U_8 \in \mathbb{R}^{896 \times 8}$ be the top-8 eigenvectors of $\Sigma_{\text{step0}}$ (the exact T2 basis from 6E.18):
$$w_{\text{S8-Proj}} = s_{\text{scale}} \frac{U_8 U_8^T \hat{u}_{\text{LDA}}}{\|U_8 U_8^T \hat{u}_{\text{LDA}}\|_2}$$
Evaluates the effect of restricting the discriminative axis strictly to the principal variance manifold.

---

## 4. Formal Hypotheses

- **H-Subspace-Utility (Primary):** Fixed discriminative subspace projection (S2, S4, S8) achieves both high gradient alignment ($\cos(H1) \ge +0.70$) AND strong decision margin ($d(15) \ge +1.50$, balanced proposal rate $\in [60\%, 95\%]$).
- **H-Dimensionality-Tradeoff:** Lower subspace rank $k$ increases gradient alignment ($\cos(H1)$) by filtering more noise dimensions, without degrading Cohen's $d$.
- **H-SVD-Manifold-Sufficiency:** Projecting LDA onto the top-8 SVD subspace (S8-Proj) preserves $\ge 90\%$ of F1's margin while increasing gradient cosine above $+0.80$.
- **H-Control-Exclusion:** Random fixed subspace control (C4-Rand) fails margin maintenance ($d(15) < +0.50$), proving that subspace restriction without discriminative alignment is insufficient for task utility.

---

## 5. Dual-Objective Acceptance Criteria (P1–P6)

To prevent the decoupling observed in Phase 6E.18, a condition is considered a **SUCCESS** if and only if it satisfies **BOTH** gradient compatibility and task utility:

```
===================================================================================================
CRITERION ID | METRIC NAME                   | THRESHOLD / REQUIREMENT          | OBJECTIVE GOAL
===================================================================================================
P1           | Gradient Conflict Reduction   | mean cos(H1){4,6,14,15} >= +0.50 | Gradient Compatibility
P2           | Subspace Noise Reduction      | mean cos(H1)(Sk) >= mean(F1)+0.10| Geometric Filtering
P3           | Margin Maintenance            | Cohen's d(15) >= +1.50           | Task Discriminability
P4           | Proposal Balance              | Dev Proposal Rate in [60%, 95%]  | Anti-Collapse / Anti-Drift
P5           | Balanced Accuracy             | (pos_acc + abs_acc)/2 >= 75%     | Dual-Class Precision
P6           | Anti-Degradation Guard        | Cohen's d(20) >= +1.00           | Late-Training Stability
===================================================================================================
```

---

## 6. Pre-Registered Outcome Interpretation Matrix

| Outcome Pattern | Primary Mechanistic Inference |
|---|---|
| **S4 or S8 passes P1–P6; C4-Rand fails P3** | **Decisive Support for Fixed Subspace Architecture.** Subspace restriction successfully filters gradient noise while fixed LDA alignment enforces task discriminability. |
| **S2/S4/S8 matches F1 on margin, but $\cos(H1)$ is unchanged** | Subspace restriction provides no additional gradient compatibility over pure 1D LDA. 1D LDA remains optimal. |
| **S2/S4/S8 improves $\cos(H1)$ to $>+0.90$, but margin degrades ($d < 1.0$)** | Fundamental trade-off between gradient alignment and representation separation; subspace filtering discards discriminative variance. |
| **S8-Proj matches S8** | The discriminative axis naturally lies within the top-8 representation covariance manifold. |
| **All conditions fail P4/P5 (polar collapse)** | Gradient conflict is driven by loss function / optimizer dynamics independent of readout geometry. |

---

## 7. Operational Discipline & Execution Preflight Guard

1. **Zero Readout Parameters:** Runtime assertions MUST verify `trainable_readout_params == 0` for all 7 conditions.
2. **Fixed Protocol Hash Binding:** Execution script MUST bind and verify this document's SHA-256 hash.
3. **No Retrospective Modification:** Phase 6E.18 execution logs and Phase 6E.19 adjudication reports remain permanently immutable.
4. **Preflight Mode Required:** A non-training validation mode (`--preflight`) must verify all basis projections, Gram-Schmidt orthogonalizations, parameter counts, and metric computations prior to execution.

---

## 8. Hard Stop and Human Authorization Gate

```
================================================================================
HARD STOP — NON-NEGOTIABLE
================================================================================
This document is a DESIGN-ONLY specification and does NOT authorize execution.
No model execution, training, fine-tuning, optimizer steps, or checkpoint
generation may occur without formal human review and explicit signed authorization.

Sign-off Gate:
  Reviewer: ______________________________
  Decision: [ ] APPROVE   [ ] AMEND   [ ] REJECT
  Date:     ______________________________
  Signature: ______________________________
================================================================================
```
