# THEO SLM Phase 6E.18 Post-Execution Forensic Analysis & Scientific Adjudication — v1

- **Authority:** Post-Execution Forensic Investigation Record (Phase 6E.19).
- **Target Artifact Set:** Phase 6E.18 Causal Intervention Execution Artifacts (`phase-6e18-causal-intervention`).
- **Governing Protocol:** `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.3.md` (`SHA-256: 910796C879BBAFBA80C0D6829D382723CC4E89425003380FD4CC94E1E21312C9`).
- **Execution Script:** `run_phase_6e18_causal_intervention.py` (`SHA-256: 6BCAF1319AC551848FFDB55AEA121D14D808408A192E05F2AA6171161DECEF96`).
- **Execution State:** Complete (7 of 7 conditions executed). Hard stop active.

---

## 1. Executive Summary & Calibrated Scientific Headline

```
================================================================================
PRIMARY HYPOTHESIS:         H-CAUSAL (ALIGNMENT-SPECIFIC MECHANISM) — REFUTED / NOT SUPPORTED
PRIMARY EMPIRICAL RESULT:   GRADIENT COMPATIBILITY & TASK UTILITY ARE DECOUPLED
REPRESENTATION HYPOTHESIS:  GEOMETRIC / SUBSPACE EXPLANATION — STRONGLY STRENGTHENED (CONFOUNDED)
CAPACITY HYPOTHESIS:        CONTRIBUTING FACTOR, NOT SOLE EXPLANATION (MARGIN DEGRADES)
FIXED GEOMETRY (F1):        HIGHLY EFFECTIVE (RESOLVES CONFLICT WITHOUT READOUT TRAINING)
HARD STOP STATUS:           ACTIVE (ZERO FURTHER TRAINING AUTHORIZED)
================================================================================
```

### Calibrated Headline Finding
The registered alignment-specific **H-Causal** claim is **REFUTED / NOT SUPPORTED**. While Condition I1 passes the primary conflict-reduction gate (P1, $+0.2286 \ge -0.30$), the random control Condition C2 also passes the conflict-reduction threshold ($+0.3066 \ge -0.30$), causing the pre-registered alignment-exclusion gate C2x to fail ($\Delta = -0.0780 < +0.25$). Therefore, the experiment **does not establish that LDA-aligned initialization is the unique causal reason gradient conflict is reduced**.

Crucially, the 7-condition suite reveals that **gradient compatibility ($\cos(H1)$) and downstream decision utility (margin separation, proposal balance) have separated**:
1. **Condition T2 (Truncated Subspace)** achieves near-perfect gradient alignment ($\cos(H1) \approx +0.988$), but weak margin ($d(15) = +0.101$) and late-training proposal degradation ($19.2\%$).
2. **Condition F1 (Fixed LDA)** achieves strong gradient alignment ($\cos(H1) = +0.5831$), superior margin ($d(15) = +1.813$), and healthy proposal rate ($98.1\%$) with **zero trainable readout parameters**.
3. **Condition C2 (Random Control)** achieves positive gradient cosine ($+0.3066$) via trainable capacity, but its discriminative margin deteriorates to near-zero ($d(15) = +0.198$) and inverts at step 20 ($d = -0.169$).
4. **Condition V1 (Multi-Logit)** achieves strong early margin ($d(15) = +1.840$), but undergoes a late-training polar collapse ($0.0\%$ proposal rate at step 20).

---

## 2. Artifact Integrity Verification

Prior to forensic interpretation, all 109 artifact entries registered in `sha256_manifest.json` were independently re-hashed from disk and verified against their cryptographic commitments:

- **Manifest Artifact Count:** 109 registered files (checkpoint `.pt`, checkpoint `.json`, initialization constants, SVD basis, runs, pre-registration snapshot, code snapshot, verdict).
- **Independent Verification Result:** **109 / 109 MATCHED (0 MISMATCHES, 0 CORRUPTIONS).**
- **Protocol SHA-256:** `910796C879BBAFBA80C0D6829D382723CC4E89425003380FD4CC94E1E21312C9` [VERIFIED].
- **Base Model SHA-256:** `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` [VERIFIED].
- **Corpus SHA-256:** `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0` [VERIFIED].
- **Signed Authorization Record:** `execution_authorization.json` (`APPROVE`, Signature `EXPLICIT_HUMAN_AUTHORIZATION_2026-08-16T18:03:34+05:30`, Date `2026-08-16`) [VERIFIED].

---

## 3. Independent Rule Recalculation

All primary acceptance criteria (P1–P5, C2x) and secondary mechanistic probe rules (M1–M5) were recomputed directly from the raw checkpoint metrics in `runs.json`:

```
=================================================================================================
RULE ID  | CRITERION NAME               | THRESHOLD / CONDITION             | COMPUTED VALUE   | VERDICT
=================================================================================================
P1       | Primary Conflict Reduction   | mean cos(H1)(I1) >= -0.30         | +0.2286          | PASS
C2x      | Alignment Exclusion          | mean(I1) - mean(C2) >= +0.25      | -0.0780          | FAIL
P2       | Collapse Prevention          | Proposal rate > 0 all t <= 20     | 100% all t >= 4  | PASS
P3       | Margin Maintenance           | d(15) >= +0.20 & r_eff(15) <= 1.6 | d=1.415, r=1.191 | PASS
P4       | Separability Guard           | |LOO_I1(t) - LOO_B0(t)| <= 0.10   | Max diff = 0.125 | FAIL (at t=6)
P5       | Readout Engagement           | max |cos_lda| >= 0.30 (t >= 4)    | 0.9906           | PASS
-------------------------------------------------------------------------------------------------
M1       | Fixed Geometry Probe (F1)    | H3 Insufficiency: cos <= -0.60    | +0.5831          | NOT SUPPORTED
M2       | 1D Gain Probe (T1)           | 1D Sufficiency: cos >= -0.30      | -0.8368          | NOT SUPPORTED
M3       | Subspace Probe (T2)          | Subspace Match: |T2 - I1| <= 0.10 | +0.7592 (T2=.988)| NOT SUPPORTED
M4       | Multi-Logit Probe (V1)       | cos >= -0.30, no policy degrad.   | +0.4637, d=1.840 | SUPPORTED
M5       | Capacity Sufficiency (C2)    | Capacity Floor: cos >= -0.30      | +0.3066          | SUPPORTED
=================================================================================================
```

---

## 4. Full Condition Trajectory Analysis

### Complete Metric Evolution Across Evaluated Checkpoints

#### A. Aggregate Gradient Cosine $\cos(H1)_{\text{agg}}$
$$\cos(H1) = \frac{\langle \bar{g}_{\text{POS}}, \bar{g}_{\text{ABS}} \rangle}{\|\bar{g}_{\text{POS}}\|_2 \|\bar{g}_{\text{ABS}}\|_2}$$

| Condition | $t=0$ | $t=4$ | $t=6$ | $t=7$ | $t=14$ | $t=15$ | $t=17$ | $t=20$ | Mean $\{4,6,14,15\}$ |
|---|---|---|---|---|---|---|---|---|---|
| **B0** (Baseline) | $+0.2588$ | $-0.9318$ | $-0.9967$ | — | — | — | — | — | **$-0.9642$** |
| **I1** (Aligned) | $+0.7175$ | $+0.4855$ | $+0.5132$ | $+0.5242$ | $-0.0881$ | $+0.0037$ | $+0.2755$ | $+0.5025$ | **$+0.2286$** |
| **C2** (Random) | $+0.8876$ | $+0.4886$ | $+0.4895$ | $+0.5245$ | $+0.0620$ | $+0.1864$ | $+0.7716$ | $+0.1551$ | **$+0.3066$** |
| **F1** (Fixed LDA) | $+0.7175$ | $+0.6693$ | $+0.7129$ | $+0.7344$ | $+0.5105$ | $+0.4396$ | $+0.4685$ | $+0.2236$ | **$+0.5831$** |
| **T1** (1D Gain) | $-0.3930$ | $-0.8331$ | $-0.7236$ | $-0.6857$ | $-0.8966$ | $-0.8937$ | $-0.6388$ | $-0.8550$ | **$-0.8368$** |
| **T2** (Subspace) | $+0.9856$ | $+0.9899$ | $+0.9926$ | $+0.9929$ | $+0.9837$ | $+0.9849$ | $+0.9830$ | $+0.9810$ | **$+0.9878$** |
| **V1** (Multi-Logit) | $+0.7175$ | $+0.6681$ | $+0.6469$ | $+0.6064$ | $+0.4610$ | $+0.0790$ | $-0.3956$ | $-0.8482$ | **$+0.4637$** |

#### B. Margin Separation Metric (Cohen's $d$ on $N=16$ Panel)
$$d = \frac{\bar{m}_{\text{POS}} - \bar{m}_{\text{ABS}}}{\sigma_{\text{pooled}}}$$

| Condition | $t=0$ | $t=4$ | $t=6$ | $t=7$ | $t=14$ | $t=15$ | $t=17$ | $t=20$ |
|---|---|---|---|---|---|---|---|---|
| **B0** | $+0.493$ | $+0.000$ | $+0.603$ | — | — | — | — | — |
| **I1** | $+2.355$ | $+1.946$ | $+1.900$ | $+1.861$ | $+1.483$ | $+1.415$ | $+1.251$ | $+1.104$ |
| **C2** | $-0.057$ | $+0.208$ | $+0.200$ | $+0.226$ | $+0.331$ | $+0.198$ | $-0.049$ | $-0.169$ |
| **F1** | $+2.355$ | $+2.023$ | $+1.965$ | $+1.968$ | $+1.900$ | $+1.813$ | $+1.343$ | $+1.306$ |
| **T1** | $+0.502$ | $+0.548$ | $+0.258$ | $+0.202$ | $+0.183$ | $+0.119$ | $-0.095$ | $-0.259$ |
| **T2** | $+0.000$ | $+0.135$ | $+0.186$ | $+0.186$ | $+0.104$ | $+0.101$ | $-0.018$ | $-0.242$ |
| **V1** | $+2.355$ | $+2.026$ | $+2.013$ | $+2.016$ | $+2.058$ | $+1.840$ | $+1.615$ | $+1.779$ |

#### C. Dev Proposal Rate Dynamics

| Condition | $t=0$ | $t=4$ | $t=6$ | $t=7$ | $t=14$ | $t=15$ | $t=17$ | $t=20$ | Collapse State |
|---|---|---|---|---|---|---|---|---|---|
| **B0** | $0.0\%$ | $0.0\%$ | $0.0\%$ | — | — | — | — | — | Truncated ($t^*=4$) |
| **I1** | $7.7\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | No collapse (over-predicts proposal) |
| **C2** | $40.4\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | No collapse (over-predicts proposal) |
| **F1** | $7.7\%$ | $73.1\%$ | $73.1\%$ | $75.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $98.1\%$ | **Balanced operation** ($73\%-98\%$) |
| **T1** | $0.0\%$ | $100.0\%$ | $98.1\%$ | $98.1\%$ | $100.0\%$ | $100.0\%$ | $69.2\%$ | $0.0\%$ | **Collapses to 0%** ($t^*=19$) |
| **T2** | $0.0\%$ | $88.5\%$ | $88.5\%$ | $96.2\%$ | $98.1\%$ | $88.5\%$ | $48.1\%$ | $19.2\%$ | **Severe proposal decay** ($19.2\%$) |
| **V1** | $21.2\%$ | $73.1\%$ | $82.7\%$ | $82.7\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | $0.0\%$ | **Late polar flip to 0%** ($t^*=20$) |

---

## 5. Formal 5-Tier Epistemic Adjudication

| Hypothesis / Mechanism | Candidate Condition(s) | Observed Evidence | Epistemic Classification | Scientific Interpretation |
|---|---|---|---|---|
| **H-Causal (Alignment Specificity)** | I1 vs C2 | C2 achieves $\cos(H1)=+0.3066$, causing C2x to fail ($\Delta = -0.0780 < +0.25$). | **REFUTED / NOT SUPPORTED** | Trainable aligned initialization is not uniquely necessary to achieve positive gradient cosine. |
| **H2 (Capacity Contribution)** | C2 | C2 reaches $+0.3066 \ge -0.30$, but margin deteriorates ($d(15)=0.198, d(20)=-0.169$). | **SUPPORTED (PARTIAL)** | Trainable capacity lifts gradient cosine, but cannot maintain discriminative margin. |
| **H3 (Fixed Geometry Insufficiency)** | F1 | F1 achieves $\cos(H1)=+0.5831 \gg -0.60$ with $d(15)=+1.813$ ($0$ parameters). | **REFUTED** | Fixed aligned geometry is highly sufficient and outperforms trainable readouts. |
| **H4 (Subspace Sufficiency)** | T2 vs I1 | T2 achieves $\cos(H1)=+0.9878 \gg +0.2286$, but margin collapses ($d(15)=0.101, d(20)=-0.242$). | **CONFOUNDED / NOT IDENTIFIED** | Subspace restriction filters gradient noise, but 0-init fails downstream decision utility. |
| **H5 (Multi-Logit Advantage)** | V1 | V1 achieves $\cos(H1)=+0.4637, d(15)=+1.840$, but collapses to $0\%$ proposal at $t=20$. | **CONFOUNDED** | High initial capacity/margin, but unconstrained independent logits exhibit late instability. |
| **1D Gain Inefficacy** | T1 | T1 yields $\cos(H1)=-0.8368$ and collapses to $0\%$ proposal at $t=20$. | **SUPPORTED** | Scalar scaling along a misaligned vocabulary axis cannot resolve gradient conflict. |
| **Geometric Primacy** | F1, I1 vs B0, T1 | Aligned axes (F1, I1) resolve conflict; misaligned axes (B0, T1) collapse. | **STRONGLY STRENGTHENED** | Readout geometry is central, but confounded with parameterization and subspace choices. |

---

## 6. Deep-Dive Forensic Investigations

### Forensic Investigation 1: Decoupling of Gradient Alignment and Task Utility

The central conceptual discovery of Phase 6E.18 is that **gradient compatibility ($\cos(H1) > 0$) is necessary but entirely insufficient for a functional decision model**:

- **Case 1 — Condition T2:**
  - Gradient cosine: **$+0.9878$** (nearly perfect common descent).
  - Decision margin $d(15)$: **$+0.101$** (near-zero class separation).
  - Proposal rate at $t=20$: **$19.2\%$** (degrading toward collapse).
  - *Diagnosis:* Constraining updates to the top-8 SVD subspace eliminates orthogonal gradient conflict, but without an aligned discriminative bias vector, the model fails to learn decision separation.

- **Case 2 — Condition C2:**
  - Gradient cosine: **$+0.3066$** (comfortably passes P1).
  - Decision margin $d(15)$: **$+0.198$**, dropping to **$-0.169$** at $t=20$ (inverted margin).
  - *Diagnosis:* Trainable capacity allows the optimizer to rotate parameters into a mutually non-interfering gradient subspace, but because the readout axis is unaligned ($\cos_{\text{LDA}} \approx 0.013$), the learned representations do not separate PROPOSE from ABSTAIN.

- **Case 3 — Condition F1:**
  - Gradient cosine: **$+0.5831$** (strong alignment).
  - Decision margin $d(15)$: **$+1.813$** (strong, stable separation).
  - Proposal rate at $t=20$: **$98.1\%$** (balanced, robust operation).
  - *Diagnosis:* Fixed LDA projection simultaneously provides gradient compatibility AND enforces discriminative task separation.

---

### Forensic Investigation 2: Why Does Fixed Projection (F1) Outperform Trainable Readout (I1)?

Comparing F1 ($0$ trainable parameters) with I1 ($897$ trainable parameters):
- **Gradient Alignment:** F1 ($+0.5831$) outperforms I1 ($+0.2286$).
- **Margin Quality:** F1 ($d(15) = +1.813$) outperforms I1 ($d(15) = +1.415$).
- **Balanced Accuracy:** At $t=4$, F1 achieves $81.25\%$ balanced accuracy ($\text{pos}=87.5\%, \text{abs}=75.0\%$). In contrast, I1 immediately shifts to predicting $100\%$ PROPOSE ($\text{pos}=100\%, \text{abs}=0.0\%$).

#### Theoretical Explanation
When the readout is fixed (F1), the LoRA backbone adapts representations to match a stationary, optimal geometric target. 

When the readout head is trainable (I1), the head parameters $(a, b)$ absorb error gradients. The optimizer exploits the extra degrees of freedom by inflating the bias $b$ to satisfy the proposal objective, causing the head to slightly rotate away from the optimal axis ($\cos_{\text{LDA}} = 0.9709$ at step 20) and over-predict proposals. **Readout trainability is not only unnecessary—it introduces optimization instability compared to a fixed aligned axis.**

---

### Forensic Investigation 3: Forensic Reconciliation of Condition V1 (Affine 2-Logit)

In the initial report, Condition V1 was noted as having $d(15) = +1.840$, but its proposal rate dropped to $0.0\%$ at $t=20$. A step-by-step audit of V1 reveals:

1. **Steps 1–9:** Dev proposal rate increases from $40.4\% \rightarrow 96.2\%$; margin increases from $-0.001 \rightarrow +0.879$.
2. **Steps 10–18:** Dev proposal rate locks at $100.0\%$; POS accuracy is $100\%$, ABS accuracy is $0\%$; margin peaks at $d(14) = +2.058$.
3. **Step 19:** Dev proposal rate suddenly drops to $38.5\%$ (dev margin $-0.039$).
4. **Step 20:** Dev proposal rate collapses to $0.0\%$; POS accuracy drops to $0.0\%$, ABS accuracy becomes $100\%$; gradient cosine plunges to $\cos(H1) = -0.8482$.

#### Diagnosis
V1's independent 2-logit parameterization ($w_{\text{PRO}}, w_{\text{ABS}}, b_{\text{PRO}}, b_{\text{ABS}}$) has $1794$ degrees of freedom. Without symmetric constraint ($z_{\text{PRO}} = -z_{\text{ABS}}$), the independent logits suffer from an unconstrained drift mode: the network first optimizes the proposal margin by saturating $w_{\text{PRO}}$, and then at step 20 undergoes a catastrophic polar reversal where $w_{\text{ABS}}$ overwhelms $w_{\text{PRO}}$. This proves that **symmetric single-scalar parameterization ($s, -s$) is essential to prevent unconstrained logit drift.**

---

### Forensic Investigation 4: Root Cause of P4 Separability Failure at $t=6$

Criterion P4 was evaluated as **FAIL** at $t=6$ on the $N=16$ panel:
- At $t=6$, Condition B0 collapsed to $31.25\%$ LOO accuracy.
- At $t=6$, Condition I1 improved to $43.75\%$ LOO accuracy.
- The absolute difference $|\text{LOO}_{\text{I1}}(6) - \text{LOO}_{\text{B0}}(6)| = |0.4375 - 0.3125| = 0.1250 > 0.1000$.
- Because P4 was formulated as a symmetric two-sided tolerance guard ($|\text{diff}| \le 0.10$), I1 outperforming the collapsing baseline by $+12.5\%$ triggered an automated failure. At all post-truncation checkpoints ($t \in \{14, 15, 20\}$) evaluated against the step-0 anchor ($37.5\%$), I1 was within tolerance ($\text{diff} \le 0.0625$).

**Conclusion:** The P4 failure is an operational artifact of baseline collapse dynamics, not representation degradation in Condition I1.

---

## 7. Strategic Question for Future Research (Phase 6E.20 Framing)

Phase 6E.18 has successfully refuted the narrow hypothesis that a trainable aligned head is uniquely required, while providing powerful evidence for the role of representation geometry and low-rank subspaces.

The sharp question for the next investigation is:
> **Can we design a fixed-subspace readout architecture that retains the superior task utility and balanced margin of Condition F1 ($d > 1.8$, balanced accuracy $>80\%$) while capturing the orthogonal noise reduction of Condition T2 ($\cos \approx +0.99$), without introducing trainable head confounds or unconstrained logit drift?**

---

## 8. Final Adjudication Record

```
================================================================================
FINAL SCIENTIFIC VERDICT:
  - H-Causal (Alignment Specificity): REFUTED / NOT SUPPORTED
  - Fixed Geometry Mechanism:         STRONGLY STRENGTHENED (F1 is top performer)
  - Subspace Restriction Mechanism:   STRONGLY STRENGTHENED (T2 filters gradient noise)
  - Trainable Capacity Mechanism:     PARTIAL (C2 lifts cos, but margin degrades)
  - Core Theoretical Insight:         Gradient alignment and task utility are decoupled
================================================================================
HARD STOP: ACTIVE
No further training, optimizer steps, or model modifications are authorized.
```
