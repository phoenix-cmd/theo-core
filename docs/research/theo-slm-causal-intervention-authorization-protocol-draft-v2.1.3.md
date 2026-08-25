# THEO SLM Causal Intervention Authorization Protocol — DRAFT v2.1.3 (AMENDED)

- **Status:** DRAFT — DESIGN ONLY. **NOT AUTHORIZED FOR EXECUTION.**
- **Human review decision:** AMEND AND RESUBMIT FOR FINAL AUTHORIZATION (2026-08-16).
- **Supersedes:** `theo-slm-causal-intervention-authorization-protocol-draft-v1.md` (v1), `theo-slm-causal-intervention-authorization-protocol-draft-v2.md` (v2), `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.1.md` (v2.1.1), and `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.2.md` (v2.1.2).
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
- Initialization of orientation: $a(0) = s_{\text{scale}} \hat{u}_{\text{LDA}}$, where $\hat{u}_{\text{LDA}}$ is the unit Fisher LDA axis computed on the frozen step-0 $N=16$ panel.
- Initialization of bias: $b(0) = -\text{median}(s_{\text{scale}} \hat{u}_{\text{LDA}} @ H_{16[:8]})$ (reference 6E.17 value: $-0.07326254$; see §3.8).

### 3.3 Random-Orientation Trainable Control (C2)
Identical functional form to I1:
$$\hat{z}_{\text{PRO}}'' = s'', \quad \hat{z}_{\text{ABS}}'' = -s'', \quad s'' = c^T h_L + b''$$
- $c \in \mathbb{R}^{896}, b'' \in \mathbb{R}$ ($897$ trainable parameters).
- Initialization of orientation: $c(0) = s_{\text{scale}} n$, where $n$ is a unit random vector with $|\cos(n, \hat{u}_{\text{LDA}})| \le 0.05$ (seed `20260816`).
- Initialization of bias: $b''(0) = -\text{median}(c_0 @ H_{16[:8]})$ with $c_0 = s_{\text{scale}} n$ (reference 6E.17 value: $0.17438322$; see §3.8).

### 3.4 Fixed LDA-Axis Projection (F1)
Evaluates fixed projection onto the empirical discriminative axis without trainability:
$$\hat{z}_{\text{PRO}} = s_{\text{fixed}}, \quad \hat{z}_{\text{ABS}} = -s_{\text{fixed}}, \quad s_{\text{fixed}} = (s_{\text{scale}} \hat{u}_{\text{LDA}})^T h_L + b_{\text{fixed}}$$
- $0$ trainable parameters. Fixed projection evaluated across training.

### 3.5 1D Trainable Scalar Gain on Frozen Axis (T1)
Evaluates 1D scalar gain adaptation along the historical frozen margin axis $w_{\delta}$:
$$s = \alpha (w_{\delta}^T h_L) + b$$
- $\alpha \in \mathbb{R}, b \in \mathbb{R}$ ($1$ trainable gain parameter, $1$ trainable bias; **$2$ total trainable parameters**; $0$ axis orientation degrees of freedom).
- Initialization: $\alpha(0) = 1.0$, $b(0) = 0.0$ (formalized from the 6E.18 runner default). Deterministic; recorded in the manifest.

### 3.6 Truncated Subspace Trainable Readout (T2)
Evaluates trainable readout restricted to the top-$k$ discriminative subspace:
$$s = (U_k \beta)^T h_L + b$$
- $U_k \in \mathbb{R}^{896 \times k}$ is the frozen orthogonal basis of the top-$k$ SVD components of the empirical covariance of the step-0 panel as defined in §8.
- $\beta \in \mathbb{R}^k, b \in \mathbb{R}$ ($k+1$ trainable parameters).
- Initialization: $\beta(0) = 0$, $b(0) = 0.0$ (formalized from the 6E.18 runner default). Deterministic; recorded in the manifest.

### 3.7 Unconstrained Affine Readout (V1)
Tests the 2-logit unconstrained cross-entropy formulation:
$$\hat{z}_{\text{PRO}} = w_{\text{PRO}}^T h_L + b_{\text{PRO}}, \quad \hat{z}_{\text{ABS}} = w_{\text{ABS}}^T h_L + b_{\text{ABS}}$$
- $w_{\text{PRO}}, w_{\text{ABS}} \in \mathbb{R}^{896}; b_{\text{PRO}}, b_{\text{ABS}} \in \mathbb{R}$ ($1794$ trainable parameters).
- Initialization (**CHANGE** — genuine protocol choice introduced in v2.1.3; requires 6E.18 runner implementation): $w_{\text{PRO}}(0) = s_{\text{scale}} \hat{u}_{\text{LDA}}$, $w_{\text{ABS}}(0) = -s_{\text{scale}} \hat{u}_{\text{LDA}}$, $b_{\text{PRO}}(0) = b_{\text{ABS}}(0) = 0.0$. Deterministic aligned mirror of the I1 symmetric scalar in 2-logit form; no RNG. V1's starting geometry is identical to I1's, isolating parameterization and capacity.

### 3.8 Shared Initialization Constants
- **$H_{16}$** denotes the step-0 $16 \times 896$ hidden-state matrix whose rows are the layer-24 last-token states $h_L$ of the frozen base model (before LoRA adaptation) for the $N=16$ panel; $H_{16[:8]}$ are its first 8 rows (POS). Same object used by the §6.1 $k$-NN metrics.
- **$s_{\text{scale}}$** $= \text{std}(\text{margins}_{B0}) / (\text{std}(H_{16} @ \hat{u}_{\text{LDA}}) + \varepsilon)$ with $\varepsilon = 10^{-12}$ (numerical-stability guard), computed at step 0 on the frozen $N=16$ panel (deterministic; no RNG); recorded in the run manifest. Reference value from 6E.17: $0.06683328$.
- **$b_{\delta}$** $= 0$ (the B0 margin is a difference of two logits; the shared final-layer bias cancels).
- **$b(0)$ for I1** $= -\text{median}(s_{\text{scale}} \hat{u}_{\text{LDA}} @ H_{16[:8]})$ (reference 6E.17 value: $-0.07326254$).
- **$b''(0)$ for C2** $= -\text{median}(c_0 @ H_{16[:8]})$ with $c_0 = s_{\text{scale}} n$ (reference 6E.17 value: $0.17438322$).
- **$b_{\text{fixed}}$** $= b(0)$ for I1 (F1 is the frozen twin of I1).

---

## 4. Experimental Condition and Confound Control Matrix

| ID | Condition Name | Trainable Params | Init Orientation | Parameterization Form | Primary Confound Controlled |
|---|---|---|---|---|---|
| **B0** | Matched Frozen Baseline | 0 | Frozen $w_{\delta}$ | Tied difference | Base trajectory comparison |
| **I1** | Aligned Readout | 897 | $\hat{u}_{\text{LDA}}$ | Symmetric scalar ($s, -s$) | Alignment test candidate |
| **C2** | Random Control | 897 | Random ($|\cos| \le 0.05$) | Symmetric scalar ($s'', -s''$) | Capacity confound vs I1 |
| **F1** | Fixed LDA Projection | 0 | $\hat{u}_{\text{LDA}}$ | Symmetric scalar ($s_{\text{fixed}}, -s_{\text{fixed}}$) | Fixed geometry vs trainability |
| **T1** | 1D Gain on $w_{\delta}$ | 2 | $w_{\delta}$ | Scaled frozen axis | Minimal capacity adaptation |
| **T2** | Subspace Readout | $k+1$ | Top-$k$ SVD | Truncated subspace ($U_k \beta$) | Low-rank axis subspace |
| **V1** | Unconstrained Affine | 1794 | $s_{\text{scale}} \cdot (\hat{u}_{\text{LDA}}, -\hat{u}_{\text{LDA}})$ | Independent 2-logit | Output form & unconstrained capacity |

### 4.1 Confound Controls & Isolation Boundaries
1. **Capacity Confound (I1 vs C2):** I1 and C2 carry identical parameter count ($897$) and optimizer treatment, isolating initialization orientation.
2. **Trainability vs Geometry Confound (T1 vs F1):** T1 vs F1 does **not** isolate trainability because trainability ($2$ params vs $0$) and orientation ($w_{\delta}$ vs $\hat{u}_{\text{LDA}}$) differ simultaneously. No trainability-only claim may be drawn.
3. **Subspace Constraint (T2 vs I1):** T2 vs I1 differs simultaneously in (a) rank ($k$ vs 896), (b) basis construction (top-$k$ SVD of the total step-0 panel covariance, §8, vs the Fisher LDA axis $\hat{u}_{\text{LDA}}$), and (c) trainable-vector initialization ($\beta(0) = 0$ vs $a(0) = s_{\text{scale}} \hat{u}_{\text{LDA}}$). M3/H4 evaluates a composite low-rank-adapted-readout contrast; no rank-only causal claim may be drawn from M3 alone.
4. **Logit Parameterization & Capacity Confound (V1 vs I1):** V1 vs I1 simultaneously alters both the logit parameterization (independent 2-logit vs symmetric single scalar) AND trainable capacity ($1794$ vs $897$ parameters). The comparison evaluates the combined effect of unconstrained multi-logit adaptation rather than isolating logit independence at matched capacity.

---

## 5. Hypotheses and Mechanistic Interpretation Matrix

### 5.1 Formal Hypotheses
- **H1 (Alignment Specificity):** $\cos(H1)(I1) - \cos(H1)(C2) \ge +0.25$ (C2x pass).
- **H2 (Capacity Sufficiency):** $\cos(H1)(C2) \ge -0.30$ and C2x fails.
- **H3 (Fixed Geometry Insufficiency):** F1 exhibits persistent conflict and satisfies the insufficiency criterion ($\cos(H1)(F1) \le -0.60$). The $-0.60$ threshold is calibrated against the 6E.17 anchored B0 conflict ($-0.933$ at $t=4$, $-0.997$ at $t=6$; see companion post-execution report).
- **H4 (Subspace-Readout Match, composite):** T2 matches I1 within $\Delta \le 0.10$ on $\cos(H1)$ under the acknowledged rank $\times$ basis $\times$ initialization composite contrast (M3). Support for H4 indicates that a low-rank adapted readout (principal-subspace SVD basis, zero $\beta$ init) suffices; it does not isolate rank from basis construction or initialization.
- **H5 (Unconstrained Multi-Logit Advantage):** V1 matches I1 on primary conflict reduction ($\cos(H1)(V1) \ge -0.30$) without exhibiting policy degradation.

### 5.2 Pre-Registered Outcome Interpretation Matrix

| Outcome Pattern | Primary Mechanistic Inference |
|---|---|
| I1 passes P1 & C2x; F1 satisfies H3 ($\cos(F1) \le -0.60$) [M1 SUPPORTED] | Strong support for H-Causal (alignment-specific mechanism requires trainable adaptation). |
| I1 passes P1; C2 meets M5 floor; C2x fails | Capacity/trainability explanation strengthened; alignment claim rejected. |
| I1 fails P1; C2 fails M5 floor; F1 remains conflictual ($\cos(F1) < -0.30$, i.e., H3 SUPPORTED or INDETERMINATE) | Readout intervention ineffective; H0 survives. |
| I1 passes P1 & C2x; F1 in indeterminate band ($-0.60 < \cos(F1) \le -0.30$) [M1 INDETERMINATE] | H-Causal not decisively supported; F1 fixed-geometry insufficiency neither established nor refuted. |
| M3 SUPPORTED; M2 NOT SUPPORTED | Low-rank adapted readout matches aligned full-rank readout; 1D gain insufficient (rank, basis, and init remain confounded). |
| M4 SUPPORTED | Unconstrained 2-logit parameterization and expanded capacity provide independent advantage. |
| One class improves while other degrades | Artifactual trade-off; NOT accepted as conflict resolution. |
| Protocol deviation in any condition | Run VOID. |

---

## 6. Normative Metric Implementation and Sign Conventions

### 6.1 Metric Formulations
1. **Decision-Gradient Cosine $\cos(H1)$:** Raw cosine between the mean POS and mean ABS decision gradients, as bound in §6.2. All decision rules (P1–P5, C2x, M1–M5) consume this aggregate cosine; the mean-pairwise variant is telemetry-only and does not enter any decision rule.
2. **Common Descent Infeasibility:** $\rho_{\text{CD}}(t) = \max(0, -\cos(H1)(t))$.
3. **Simultaneous Descent Fraction:** $F(t) = \text{fraction of } (i, j) \in \text{POS} \times \text{ABS} \text{ with } \langle g_i, g_j \rangle > 0$.
4. **Proposal Rate:** $P(t) = \text{fraction of dev-set items with margin } \hat{z}_{\text{PRO}} - \hat{z}_{\text{ABS}} > 0$ at checkpoint $t$. Used by P2 and M4 policy degradation.
5. **Margin Cohen's $d(t)$:** $(\text{mean } m_{\text{POS}} - \text{mean } m_{\text{ABS}}) / \text{pooled sd}$ over the $N=16$ panel margins at checkpoint $t$, with pooled sd $= \sqrt{(\text{var}(m_{\text{POS}}) + \text{var}(m_{\text{ABS}})) / 2}$; panel first-8 items POS, last-8 ABS. P3 uses $d(15)$.
6. **Effective Rank $r_{\text{eff}}(t)$:** Participation-ratio effective rank $\text{tr}(G)^2 / \text{tr}(G^2)$ of the margin-Jacobian Gram matrix $G$ at checkpoint $t$ ($N=16 \times N=16$ over the panel; margin gradients with respect to LoRA parameters). P3 uses $r_{\text{eff}}(15)$.
7. **Resub $k$-NN1:** Resubstitution accuracy of a $k=1$ Euclidean nearest-neighbor classifier over the layer-24 last-token hidden state $h_L$, POS-vs-ABS labels. Used by P4.
8. **LOO-CV($t$):** Leave-one-out accuracy of the same $k=1$ nearest-neighbor classifier on $h_L$ over the $N=16$ panel at checkpoint $t$. Used by P4/§7.1.
9. **Policy Metrics / Policy Degradation:** Policy metrics $= \{P(t), d(t), \text{POS/ABS accuracy}\}$. A condition exhibits policy degradation if $P(15) \le 0.0\%$ OR $d(15) < 0.00$. Used by H5/M4.

### 6.2 Mandatory Metric Binding and Pre-Verdict Assertion Gate

$\cos(H1)$ **SHALL** be computed as the raw cosine between the mean $\text{POS}$ and $\text{ABS}$ decision gradients, without label-sign inversion or $\text{ABS}$ negation:

$$\cos(H1) = \frac{\langle \bar{g}_{\text{POS}}, \bar{g}_{\text{ABS}} \rangle}{\|\bar{g}_{\text{POS}}\|_2 \|\bar{g}_{\text{ABS}}\|_2}$$

#### Pre-Verdict Deterministic Implementation Gate
Prior to generating any verdict or evaluating decision criteria, the runtime implementation **SHALL** execute a synthetic assertion verifying the sign convention:
1. **Synthetic Unit Test Vector Assertion:** The metric calculation function shall be evaluated on deterministic synthetic vectors:
   - Synthetic Oppositional Pair: $u_1 = [1.0, 0.0]^T, u_2 = [-1.0, 0.0]^T \implies \cos_{\text{synth}} = -1.0$.
   - Synthetic Aligned Pair: $v_1 = [1.0, 0.0]^T, v_2 = [1.0, 0.0]^T \implies \cos_{\text{synth}} = +1.0$.
2. **Numerical Tolerance:** The computed cosine on the synthetic pair must satisfy:
   $$|\cos_{\text{calc}}(u_1, u_2) - (-1.0)| \le \epsilon, \quad |\cos_{\text{calc}}(v_1, v_2) - (+1.0)| \le \epsilon, \quad \text{with } \epsilon = 0.001$$
3. **Voiding Condition:** If the implementation fails this synthetic assertion ($\Delta > \epsilon$), or if label-sign inversion is detected, the run is immediately declared **VOID** and no verdict may be generated.

*(Note: This assertion gate validates implementation arithmetic and sign correctness exclusively; it does NOT gate or filter actual model gradient values against historical outcome ranges.)*

---

## 7. Frozen Decision Rules and Acceptance Criteria (P1–P5, C2x, M1–M5)

| ID | Criterion | Threshold | Evaluated Condition & Comparator |
|---|---|---|---|
| **P1** | Primary Conflict Reduction | $\text{mean}_{t \in \{4,6,14,15\}} \cos(H1) \ge -0.30$ | Condition I1 |
| **C2x** | Alignment Exclusion (Mandatory) | $\text{mean}_{t \in \{4,6,14,15\}} \cos(H1)(I1) - \text{mean}_{t \in \{4,6,14,15\}} \cos(H1)(C2) \ge +0.25$ | I1 vs C2 |
| **P2** | Collapse Prevention | Proposal rate $> 0.0\%$ at all $t \le 20$ | Condition I1 |
| **P3** | Margin Maintenance | Cohen's $d(15) \ge +0.20$; $r_{\text{eff}}(15) \le 1.60$ | Condition I1 |
| **P4** | Separability Guard | Resub $k\text{-NN1} \in [0.65, 1.0]$; $\text{LOO-CV}(t)$ within $\pm 0.10$ of defined baseline comparator | Condition I1 (see §7.1 for deterministic comparator) |
| **P5** | Readout Engagement | $\max |\cos(w_{\text{readout}}, \hat{u}_{\text{LDA}})| \ge 0.30$ for $t \ge 4$ | Condition I1 |

### 7.1 Deterministic P4 Baseline Comparator Specification
To ensure unambiguous evaluation in the presence of baseline collapse truncation (§10):
- **For shared evaluable checkpoints ($t \in \{0, 4, 6\}$):** $\text{LOO-CV}_{\text{I1}}(t)$ is compared against the matched contemporaneously evaluated baseline: $|\text{LOO-CV}_{\text{I1}}(t) - \text{LOO-CV}_{\text{B0}}(t)| \le 0.10$.
- **For post-truncation checkpoints ($t \in \{14, 15, 20\}$):** Because condition B0 truncates at $t \le 6$ upon collapse detection, the frozen reference comparator is fixed to the step-0 baseline separability: $|\text{LOO-CV}_{\text{I1}}(t) - \text{LOO-CV}_{\text{B0}}(0)| \le 0.10$.
- **Early-truncation fallback:** If $\text{LOO-CV}_{\text{B0}}(t)$ is unavailable for any shared checkpoint (including $t \in \{4, 6\}$) because the B0 guard triggered before evaluation, the step-0 anchor applies: $|\text{LOO-CV}_{\text{I1}}(t) - \text{LOO-CV}_{\text{B0}}(0)| \le 0.10$, and the missing matched comparator is recorded as UNAVAILABLE in the run manifest.

### 7.2 Secondary Mechanistic Probe Decision Rules (M1–M5)

The M-rules formalize the non-I1 conditions (F1, T1, T2, V1, C2). All use the §6.2 aggregate $\cos(H1)$; checkpoints missing from an evaluated condition render the corresponding rule non-evaluable.

| ID | Probe / Hypothesis | Metric | Checkpoints | Aggregation | SUPPORT | NOT SUPPORTED | AMBIGUOUS / INTERMEDIATE | UNAVAILABLE / TRUNCATED |
|---|---|---|---|---|---|---|---|---|
| **M1** | F1 / H3 | $\cos(H1)(F1)$ | {4, 6, 14, 15} | mean | mean $\le -0.60$ → H3 SUPPORTED | mean $> -0.30$ → H3 NOT SUPPORTED | $-0.60 < \text{mean} \le -0.30$ → H3 INDETERMINATE | any checkpoint missing → rule VOID (§5.2) |
| **M2** | T1 / 1D-Gain | $\cos(H1)(T1)$ | {4, 6, 14, 15} | mean | mean $\ge -0.30$ → 1D gain SUFFICIENT | mean $< -0.30$ → 1D gain INSUFFICIENT | none (binary rule) | any checkpoint missing → rule VOID |
| **M3** | T2 / H4 | $\Delta = \|\text{mean } \cos(H1)(T2) - \text{mean } \cos(H1)(I1)\|$ | {4, 6, 14, 15} | mean | $\Delta \le 0.10$ AND mean $\cos(H1)(T2) \ge -0.30$ → MATCH | $\Delta > 0.10$ OR mean $\cos(H1)(T2) < -0.30$ → NO MATCH | not evaluable if I1 VOID → M3 NOT EVALUABLE | $k \notin [2, 8]$ → T2 VOID (§8); missing checkpoint → rule VOID |
| **M4** | V1 / H5 | mean $\cos(H1)(V1)$ AND policy degradation status | {4, 6, 14, 15} | mean | $\cos \ge -0.30$ AND no degradation → SUPPORTED | $\cos < -0.30$ OR degradation → NOT SUPPORTED | $\cos \ge -0.30$ with degradation → NOT SUPPORTED (degradation recorded) | any checkpoint missing → rule VOID |
| **M5** | C2 / H2 | $\cos(H1)(C2)$ | {4, 6, 14, 15} | mean | mean $\ge -0.30$ → capacity sufficiency SUPPORTED | mean $< -0.30$ → NOT SUPPORTED | none (binary rule) | any checkpoint missing → rule VOID |

---

## 8. T2 Basis Rank Prospective Verification Procedure

Prior to executing condition T2:
1. Compute the total sample covariance matrix $\Sigma_{\text{diff}}$ of the step-0 hidden states on the $N=16$ panel: center $X_{\text{panel}}$ on its global mean and form $\Sigma_{\text{diff}} = X_{\text{centered}}^T X_{\text{centered}} / (N - 1)$. Deterministic; rank $\le 15$; construction recorded in the run manifest.
2. Perform SVD: $\Sigma_{\text{diff}} = U S V^T$.
3. Select minimum $k$ such that cumulative energy $\sum_{i=1}^k S_{ii} / \sum_{j=1}^d S_{jj} \ge 0.90$.
4. Assert $k \in [2, 8]$. Record $k$ and basis hash in the run manifest. If $k \notin [2, 8]$, condition T2 is declared VOID and must be resubmitted for prospective review.
5. The T2 basis $U_k$ and initialization $\beta(0) = 0$ are frozen before T2 execution; $k$ and the basis hash are recorded in the run manifest.

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
- Cross-phase synthetic trajectory concatenation is strictly prohibited. Active comparisons utilize fresh matched B0 trajectories whose evaluation ceases upon detector guard trigger.
- If the B0 guard triggers before checkpoint 6, contemporaneous matched comparators at $t \in \{4, 6\}$ that were never evaluated are reported UNAVAILABLE and the §7.1 step-0 anchor is used; no implicit back-filling of B0 trajectories is permitted. The canonical evaluation checkpoint set is $\{0, 4, 6, 7, 14, 15, 17, 20\}$; rule aggregation sets are specified in §7.

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
- **v2.1 $\rightarrow$ v2.1.1:** Reconciled canonical 7-condition universe (formally retained F1 across §§3, 4, 5, 9); refined T1 vs F1 dual-confound qualification.
- **v2.1.1 $\rightarrow$ v2.1.2:** 
  1. Resolved **CANON-F01**: Bound exact synthetic unit test assertion and numerical tolerance $\epsilon = 0.001$ in §6.2.
  2. Resolved **CANON-F02**: Reconciled T1 parameter count to $2$ ($1$ gain $+ 1$ bias) across §§3.5, 4, and 9.
  3. Resolved **CANON-F03**: Corrected H3 logical polarity in §5.2 outcome matrix to explicit observable formulation (`F1 satisfies H3`).
  4. Resolved **CANON-F04**: Codified deterministic P4 comparator in §7.1 (matched baseline for $t \le 6$; step-0 anchor for $t > 6$).
  5. Resolved **CANON-F05**: Explicitly codified dual logit-parameterization and $2\times$ capacity confound for V1 vs I1 in §4.1 (item 4).
- **v2.1.2 $\rightarrow$ v2.1.3:**
  1. Resolved **REC-01 (BLOCKING)**: Repaired phantom §4.4 reference to §4.1 (item 4) and formally numbered the §4.1 confound subsection; corrected the CANON-F05 log pointer.
  2. Resolved **F-03 (MAJOR)**: Added §7.2 frozen secondary probe rules M1–M5 (F1/T1/T2/V1/C2) with metric, checkpoint set, aggregation, and SUPPORT/NOT SUPPORTED/AMBIGUOUS/UNAVAILABLE dispositions.
  3. Resolved **F-02 (MAJOR)**: Defined the F1 indeterminate band ($-0.60 < \cos(H1)(F1) \le -0.30$) as M1 INDETERMINATE in §5.2 and removed the non-I1 "fails P1" naming.
  4. Resolved **F-04 (MAJOR)**: Scoped the H4/T2 claim to the acknowledged rank $\times$ basis $\times$ initialization composite (§4.1 item 3, §5.1 H4, §5.2 row 4).
  5. Resolved **F-05 (MAJOR)**: Froze $s_{\text{scale}}$, $b_{\delta}$, $b(0)$ (I1), $b''(0)$ (C2), and $b_{\text{fixed}} = b(0)$ (§3.8) — FORMALIZATION (6E.17 derivation); T1 $\alpha(0) = 1.0$, $b(0) = 0.0$ — FORMALIZATION (6E.18 runner defaults); T2 $\beta(0) = 0$, $b(0) = 0.0$ — FORMALIZATION (6E.18 runner default); V1 initialization $s_{\text{scale}} \cdot (\hat{u}_{\text{LDA}}, -\hat{u}_{\text{LDA}})$ — CHANGE (deterministic aligned mirror; requires 6E.18 runner implementation).
  6. Resolved **MINOR set**: Defined proposal rate, margin Cohen's $d$, effective rank $r_{\text{eff}}$, Resub $k$-NN1, LOO-CV, policy metrics/degradation, and C2x aggregation set (§6.1 items 4–9, §7 C2x) — FORMALIZATION (6E.17 runner definitions); $\Sigma_{\text{diff}}$ fixed as total sample covariance (§8) — FORMALIZATION; P4 early-truncation fallback (§7.1, §10) — FORMALIZATION (6E.18 comparator behavior).

---

## 14. Hard Stop and Human Authorization Gate

This document does **NOT** authorize model execution, training, fine-tuning, optimizer steps, or checkpoint generation. Execution remains strictly blocked until formal human sign-off.

### Sign-off Form
- **Reviewer Name:** ____________________
- **Decision:** [ ] APPROVE   [ ] AMEND   [ ] REJECT
- **Date:** ____________________
- **Signature:** ____________________
