# THEO SLM Causal Intervention Authorization Protocol — DRAFT v2.1.1 (AMENDED)

- **Status:** DRAFT — DESIGN ONLY. **NOT AUTHORIZED FOR EXECUTION.**
- **Human review decision:** AMEND AND RESUBMIT FOR FINAL AUTHORIZATION (2026-08-16).
- **Supersedes:** `theo-slm-causal-intervention-authorization-protocol-draft-v1.md` (v1) and `theo-slm-causal-intervention-authorization-protocol-draft-v2.md` (v2).
- **Companion document (required reading):** `theo-slm-phase-6e17-post-execution-report-v1.md`.
- **Authority:** Requires explicit human authorization. Phase 6E.18 must not begin until this protocol is approved. No model execution, training, optimizer step, fine-tuning, or checkpoint creation is authorized by this document.
- **Basis:** Phase 6E.16-R Canonical Interpretation Record and Phase 6E.17 Post-Execution Findings.

---

## 1. Causal Framing and Falsification Mandate

The relationship between readout geometry and gradient conflict remains a **hypothesis**, not an established causal conclusion:

> Readout misalignment $\prec$ gradient conflict (temporal precedence — INFERRED)

The empirical evidence establishes temporal precedence and geometric association. It does **not** establish:

> Readout misalignment $\Rightarrow$ gradient conflict (causation — NOT ESTABLISHED)

All Phase 6E.18 documentation MUST frame the experiment as a **causal falsification test of H-Causal**, not as a repair of a proven root cause. Any report language stating or implying "the root cause is confirmed" is strictly prohibited.

- **H-Causal (falsifiable):** The proposal/abstention gradient conflict arises substantially because the fixed PROPOSE/ABSTAIN vocabulary contrast is misaligned with the task-discriminative representation geometry; replacing the fixed readout with an appropriately aligned trainable output geometry eliminates or materially reduces the measured scalar conflict and improves simultaneous POS/ABS optimization.
- **H0 (null hypothesis):** The scalar decision conflict and gate collapse are caused by training dynamics / objective interactions independent of readout alignment; an aligned trainable readout produces no measurable reduction in scalar conflict over matched controls, or any observed benefit is attributable purely to added parameter capacity.

---

## 2. Intervention Scope and Deferred Family Exclusions

The authorized conceptual intervention family is strictly restricted to the **7 pre-registered conditions** defined in §3.

The following are DEFERRED, not part of Phase 6E.18, and explicitly PROHIBITED during evaluation:
- Verbalizer changes or token remapping.
- Multi-token decision labels.
- Standalone classifier heads detached from LM backbone.
- Gradient surgery methods (PCGrad, MGDA).
- Backbone LoRA rank changes or target module modifications.
- Optimizer substitutions or learning rate schedule modifications.
- Objective loss reweighting ($\lambda_{\text{decision}} \ne 10$).
- Combinatorial multi-component modifications.

---

## 3. Mathematical Definitions of the 7 Conditions

### 3.1 Target Model Architecture & Baseline Margin (B0)
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, $d = 896$, vocabulary size $151,936$, tied embeddings, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Margin: $\Delta z_b = z_{\text{PRO}} - z_{\text{ABS}} = w_{\delta}^T h_L + b_{\delta}$, where $w_{\delta} = W_{\text{emb}}[\text{PROPOSE}] - W_{\text{emb}}[\text{ABSTAIN}]$ is frozen.
- Condition **B0**: Frozen original readout ($0$ added parameters, $0$ trainable parameters).

### 3.2 Aligned Trainable Readout (I1)
At the decision position only, the two decision logits are replaced:
$$\hat{z}_{\text{PRO}} = s, \quad \hat{z}_{\text{ABS}} = -s, \quad s = a^T h_L + b$$
- $a \in \mathbb{R}^{896}, b \in \mathbb{R}$ ($897$ trainable parameters).
- Initialized at $a(0) = s_{\text{scale}} \hat{u}_{\text{LDA}}$, where $\hat{u}_{\text{LDA}}$ is the unit Fisher LDA axis computed on the frozen step-0 $N=16$ panel.

### 3.3 Random-Orientation Trainable Control (C2)
Identical functional form to I1:
$$\hat{z}_{\text{PRO}}'' = s'', \quad \hat{z}_{\text{ABS}}'' = -s'', \quad s'' = c^T h_L + b''$$
- $c \in \mathbb{R}^{896}, b'' \in \mathbb{R}$ ($897$ trainable parameters).
- Initialized at $c(0) = s_{\text{scale}} n$, where $n$ is a unit random vector with $|\cos(n, \hat{u}_{\text{LDA}})| \le 0.05$ (seed `20260816`).

### 3.4 Fixed LDA-Axis Projection (F1)
Evaluates fixed projection onto the empirical discriminative axis without trainability:
$$\hat{z}_{\text{PRO}} = s_{\text{fixed}}, \quad \hat{z}_{\text{ABS}} = -s_{\text{fixed}}, \quad s_{\text{fixed}} = (s_{\text{scale}} \hat{u}_{\text{LDA}})^T h_L + b_{\text{fixed}}$$
- $0$ trainable parameters. Fixed projection evaluated across training.

### 3.5 1D Trainable Scalar Gain on Frozen Axis (T1)
Evaluates 1D scalar gain adaptation along the historical frozen margin axis $w_{\delta}$:
$$s = \alpha (w_{\delta}^T h_L) + b$$
- $\alpha \in \mathbb{R}, b \in \mathbb{R}$ ($1$ trainable gain parameter, $1$ bias; $0$ axis orientation degrees of freedom).

### 3.6 Truncated Subspace Trainable Readout (T2)
Evaluates trainable readout restricted to the top-$k$ discriminative subspace:
$$s = (U_k \beta)^T h_L + b$$
- $U_k \in \mathbb{R}^{896 \times k}$ is the frozen orthogonal basis of the top-$k$ SVD components of the step-0 discriminative panel (§8).
- $\beta \in \mathbb{R}^k, b \in \mathbb{R}$ ($k+1$ trainable parameters).

### 3.7 Unconstrained Affine Readout (V1)
Tests the 2-logit unconstrained cross-entropy formulation:
$$\hat{z}_{\text{PRO}} = w_{\text{PRO}}^T h_L + b_{\text{PRO}}, \quad \hat{z}_{\text{ABS}} = w_{\text{ABS}}^T h_L + b_{\text{ABS}}$$
- $w_{\text{PRO}}, w_{\text{ABS}} \in \mathbb{R}^{896}; b_{\text{PRO}}, b_{\text{ABS}} \in \mathbb{R}$ ($1794$ trainable parameters).

---

## 4. Experimental Condition and Confound Control Matrix

| ID | Condition Name | Trainable Params | Init Orientation | Parameterization Form | Primary Confound Controlled |
|---|---|---|---|---|---|
| **B0** | Matched Frozen Baseline | 0 | Frozen $w_{\delta}$ | Tied difference | Base trajectory comparison |
| **I1** | Aligned Readout | 897 | $\hat{u}_{\text{LDA}}$ | Symmetric scalar ($s, -s$) | Alignment test candidate |
| **C2** | Random Control | 897 | Random ($|\cos| \le 0.05$) | Symmetric scalar ($s'', -s''$) | Capacity confound vs I1 |
| **F1** | Fixed LDA Projection | 0 | $\hat{u}_{\text{LDA}}$ | Symmetric scalar ($s_{\text{fixed}}, -s_{\text{fixed}}$) | Fixed geometry vs trainability |
| **T1** | 1D Gain on $w_{\delta}$ | 1 | $w_{\delta}$ | Scaled frozen axis | Minimal capacity adaptation |
| **T2** | Subspace Readout | $k+1$ | Top-$k$ SVD | Truncated subspace ($U_k \beta$) | Low-rank axis subspace |
| **V1** | Unconstrained Affine | 1794 | Random / zero | Independent 2-logit | Single-scalar constraint confound |

### Confound Controls & Isolation Boundaries
1. **Capacity Confound (I1 vs C2):** I1 and C2 carry identical parameter count ($897$) and optimizer treatment, isolating initialization orientation.
2. **Trainability vs Geometry Confound (T1 vs F1):** T1 vs F1 does **not** isolate trainability because trainability ($1$ param vs $0$) and orientation ($w_{\delta}$ vs $\hat{u}_{\text{LDA}}$) differ simultaneously. No trainability-only claim may be drawn.
3. **Subspace Constraint (T2 vs I1):** Evaluates whether full $896$-dimensional adaptation is necessary relative to a $k$-dimensional subspace.
4. **Logit Independence (V1 vs I1):** Isolates the symmetric constraint ($\hat{z}_{\text{PRO}} = -\hat{z}_{\text{ABS}}$) against unconstrained independent logits.

---

## 5. Hypotheses and Mechanistic Interpretation Matrix

### 5.1 Formal Hypotheses
- **H1 (Alignment Specificity):** $\cos(H1)(I1) - \cos(H1)(C2) \ge +0.25$ (C2x pass).
- **H2 (Capacity Sufficiency):** $\cos(H1)(C2) \ge -0.30$ and C2x fails.
- **H3 (Fixed Geometry Insufficiency):** F1 fails to prevent conflict ($\cos(H1)(F1) \le -0.60$).
- **H4 (Subspace Rank Sufficiency):** T2 matches I1 within $\Delta \le 0.10$ on $\cos(H1)$.
- **H5 (Logit Independence Irrelevance):** V1 matches I1 on primary conflict reduction without superior performance.

### 5.2 Pre-Registered Outcome Interpretation Matrix

| Outcome Pattern | Primary Mechanistic Inference |
|---|---|
| I1 passes P1 & C2x; F1 fails H3 | Strong support for H-Causal (alignment-specific mechanism). |
| I1 passes P1; C2 passes P1; C2x fails | Capacity/trainability explanation strengthened; alignment claim rejected. |
| I1, C2, F1 all fail P1 | Readout intervention ineffective; H0 survives. |
| T2 matches I1; T1 fails | Low-rank discriminative subspace sufficient; 1D gain insufficient. |
| V1 ≫ I1 on policy metrics | Unconstrained logit degrees of freedom provide independent advantage. |
| One class improves while other degrades | Artifactual trade-off; NOT accepted as conflict resolution. |
| Protocol deviation in any condition | Run VOID. |

---

## 6. Normative Metric Implementation and Sign Conventions

### 6.1 Metric Formulations
1. **Pairwise Decision-Gradient Cosine:** Raw cosine between mean POS and mean ABS gradients.
2. **Common Descent Infeasibility:** $\rho_{\text{CD}}(t) = \max(0, -\cos(H1)(t))$.
3. **Simultaneous Descent Fraction:** $F(t) = \text{fraction of } (i, j) \in \text{POS} \times \text{ABS} \text{ with } \langle g_i, g_j \rangle > 0$.

### 6.2 Mandatory Metric Binding and Pre-Verdict Assertion Gate

$\cos(H1)$ **SHALL** be computed as the raw cosine between the mean $\text{POS}$ and $\text{ABS}$ decision gradients, without label-sign inversion or $\text{ABS}$ negation:

$$\cos(H1) = \frac{\langle \bar{g}_{\text{POS}}, \bar{g}_{\text{ABS}} \rangle}{\|\bar{g}_{\text{POS}}\|_2 \|\bar{g}_{\text{ABS}}\|_2}$$

Before verdict generation, the implementation **SHALL** verify agreement with the protocol-anchored convention within the pre-registered tolerance (where negative values strictly denote oppositional gradient conflict). Failure **SHALL** void the run.

---

## 7. Frozen Decision Rules and Acceptance Criteria (P1–P5, C2x)

| ID | Criterion | Threshold | Metric Target |
|---|---|---|---|
| **P1** | Primary Conflict Reduction | $\text{mean}_{t \in \{4,6,14,15\}} \cos(H1) \ge -0.30$ | Condition I1 |
| **C2x** | Alignment Exclusion (Mandatory) | $\text{mean } \cos(H1)(I1) - \text{mean } \cos(H1)(C2) \ge +0.25$ | I1 vs C2 |
| **P2** | Collapse Prevention | Proposal rate $> 0.0\%$ at all $t \le 20$ | Condition I1 |
| **P3** | Margin Maintenance | Cohen's $d(15) \ge +0.20$; $r_{\text{eff}}(15) \le 1.60$ | Condition I1 |
| **P4** | Separability Guard | Resub $k\text{-NN1} \in [0.65, 1.0]$; LOO-CV within $\pm 0.10$ of B0 | Condition I1 |
| **P5** | Readout Engagement | $\max |\cos(w_{\text{readout}}, \hat{u}_{\text{LDA}})| \ge 0.30$ for $t \ge 4$ | Condition I1 |

---

## 8. T2 Basis Rank Prospective Verification Procedure

Prior to executing condition T2:
1. Compute the empirical covariance matrix $\Sigma_{\text{diff}}$ of the step-0 hidden states on the $N=16$ panel.
2. Perform SVD: $\Sigma_{\text{diff}} = U S V^T$.
3. Select minimum $k$ such that cumulative energy $\sum_{i=1}^k S_{ii} / \sum_{j=1}^d S_{jj} \ge 0.90$.
4. Assert $k \in [2, 8]$. Record $k$ and basis hash in the run manifest.

---

## 9. Parameter-Count and Degree-of-Freedom Accounting

| Condition | Trainable Weights | Trainable Biases | Total Trainable Params | Readout Subspace DOF |
|---|---|---|---|---|
| **B0** | 0 | 0 | **0** | 0 (fixed) |
| **I1** | 896 | 1 | **897** | 896 |
| **C2** | 896 | 1 | **897** | 896 |
| **F1** | 0 | 0 | **0** | 0 (fixed) |
| **T1** | 1 | 1 | **2** | 0 (fixed axis) |
| **T2** | $k$ | 1 | **$k+1$** | $k$ |
| **V1** | 1792 | 2 | **1794** | 1792 |

---

## 10. Trajectory Identity and Baseline Discipline

- Comparisons are valid ONLY within matched experimental batches sharing step-0 seed state.
- Historical B0 from 6E.17 is designated strictly as **truncated supporting context** ($t \le 6$).
- Cross-phase synthetic trajectory concatenation is strictly prohibited.

---

## 11. Forensic Packaging and Cryptographic Provenance

All execution outputs must produce:
1. `sha256_manifest.json` recording hashes of code, base model, corpus, and checkpoints.
2. Step-by-step telemetry logged to `runs.json`.
3. Pre-registration hash recorded in `pre_registration_snapshot.json`.

---

## 12. 6E.17 Provenance Preservation

Historical Phase 6E.17 records (`runs_partial_B.json`, `B0_fingerprint_gate.json`, `verdict.json`) remain permanently preserved in their original form. Adjudication notes are appended as companion metadata without retrospective mutation of raw execution logs.

---

## 13. Amendment and Reconciliation Log

- **v1 $\rightarrow$ v2:** Added capacity control (C2), temporal precedence framing, and primary outcome isolation.
- **v2 $\rightarrow$ v2.1:** Resolved section numbering collisions (§1–§14 sequential normalization), eliminated phantom section references, and bound normative sign calculation clause.
- **v2.1 $\rightarrow$ v2.1.1:** Reconciled canonical 7-condition universe (formally retained F1 across §§3, 4, 5, 9); refined T1 vs F1 dual-confound qualification; verified complete internal cross-reference integrity.

---

## 14. Hard Stop and Human Authorization Gate

This document does **NOT** authorize model execution, training, fine-tuning, optimizer steps, or checkpoint generation. Execution remains strictly blocked until formal human sign-off.

### Sign-off Form
- **Reviewer Name:** ____________________
- **Decision:** [ ] APPROVE   [ ] AMEND   [ ] REJECT
- **Date:** ____________________
- **Signature:** ____________________
