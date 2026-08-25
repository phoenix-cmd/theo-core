# THEO SLM Fixed Subspace Readout Authorization Protocol — DRAFT v2.0

- **Status:** DRAFT v2.0 — DESIGN ONLY. **NOT AUTHORIZED FOR EXECUTION.**
- **Supersedes:** `theo-slm-fixed-subspace-readout-authorization-protocol-draft-v1.md` (v1.0), which was **REJECTED** following the Phase 6E.20 adversarial artifact-level audit (blocking findings B-1, B-2, B-3). This draft is the materialization of the design-level redesign (deliverables A–E); it makes no claim to have fixed the audit findings by cosmetics — the readout geometry itself is redefined in §5.
- **Provisional decisions:** Design decisions **E-1 … E-10** (§12) carry recommended defaults. They are **PROVISIONAL — PENDING HUMAN REVIEW**. Adoption is required before this document is finalized for any pre-execution audit.
- **Phase:** Phase 6E.20 (Fixed Low-Dimensional & Subspace Readout Architectures).
- **Authority:** Requires formal pre-execution audit and explicit human authorization (execution_authorization.json, decision=APPROVE, phase=6E.20). No model execution, training, fine-tuning, optimizer steps, or checkpoint generation is authorized by this document.
- **Basis & Provenance:** Phase 6E.18 Execution Findings (`phase-6e18-causal-intervention`); Phase 6E.19 Post-Execution Forensic Adjudication Record (`theo-slm-phase-6e18-post-execution-forensic-analysis-v1.md`); Phase 6E.18 authoritative protocol `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.3.md` (source of all inherited operational rules in §10).
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

## 2. Design Rationale: Why v1 Was Rejected and How the Geometry Is Redefined

The adversarial audit of v1.0 identified three blocking defects. Each is resolved structurally, not cosmetically:

| Blocking Finding | v1.0 Defect | v2.0 Resolution |
|---|---|---|
| **B-1** | $S_k$ was constructed to contain $\hat{u}_{\text{LDA}}$, so $P_{Sk}\hat{u}_{\text{LDA}} = \hat{u}_{\text{LDA}}$ and every subspace condition $S2/S4/S8$ was **mathematically identical to F1** (verified numerically: $\|P_k\hat{u}_{\text{LDA}} - \hat{u}_{\text{LDA}}\|_2 \approx 2 \times 10^{-16}$; forward outputs and decision gradients identical; readout decision-gradient rank $= 1$). | $S_k$ is redefined as the **top-$k$ principal subspace $U_k$ of $\Sigma_{\text{diff}}$** (§3.4), which does **not** contain $\hat{u}_{\text{LDA}}$ (enforced by the registration gate $\kappa_k \le 0.995$, §6). The readout is a **representation projection** $h' = U_kU_k^{\mathsf{T}}h_L$ feeding a fixed, gain-normalized LDA direction (§5.3). $k$ is genuinely active in the forward pass and in the backward decision-gradient path. |
| **B-2** | Criterion P2 was `mean cos(H1)(Sk) >= mean(F1) + 0.10` — defined **post hoc** from the 6E.18 F1 observation and **unsatisfiable** because $S_k \equiv F1$. The joint P1–P6 gate was unpassable by any condition. | P2 is replaced by an **absolute, independently justified threshold**: $\text{mean }\cos(H1) \ge +0.80$ — the research question's own target (§8). No Phase 6E.18/6E.19 observed value enters any threshold (§8, §13 M-1). |
| **B-3** | Condition C4-Rand used only a single random vector $n_{\text{rand}}$ in its forward equation ($s = (s_{\text{scale}}n_{\text{rand}})^{\mathsf{T}}h_L + b_{\text{rand}}$); the declared random 4D subspace $R_4$ never entered the mathematics, $b_{\text{rand}}$ was undefined, and no seed was specified. | C4-Rand is redefined as a genuine 4-dimensional random-subspace readout with the **same functional form as S4**: $w_{C4} = (s_{\text{scale}}/\kappa_R)R_4R_4^{\mathsf{T}}\hat{u}_{\text{LDA}}$, with $R_4$ deterministic from an explicit seed and $\kappa_R$ registered (§5.5, §6). |

**Two candidate geometries were evaluated and rejected as provably degenerate:**
- $S_k \ni \hat{u}_{\text{LDA}}$ (v1's construction): reduces to F1 — see B-1.
- $S_k \perp \hat{u}_{\text{LDA}}$: the projection $P_{Sk}\hat{u}_{\text{LDA}} = 0$, so the readout weight vanishes and the condition carries no discriminative signal — a degenerate null control, not a subspace hypothesis.

**The retained geometry** — $\hat{u}_{\text{LDA}} \notin S_k$ but $\hat{u}_{\text{LDA}} \not\perp S_k$ — places $S_k = U_k$ (top-$k$ principal subspace), the only construction satisfying (a) meaningful forward signal, (b) a backward decision-gradient in $\text{span}(U_k)$, and (c) genuine $k$-dependence.

---

## 3. Shared Construction Objects (Frozen at Step 0)

All objects below are computed on the **frozen step-0 panel** and are deterministic (no RNG, except where explicitly seeded). All are recorded in the run manifest.

### 3.1 Step-0 Panel $H_{16}$
$H_{16} \in \mathbb{R}^{16 \times 896}$: the layer-24 last-token hidden states $h_L$ of the frozen base model (before LoRA adaptation) over the $N = 16$ panel; rows $0{:}8$ are POS, rows $8{:}16$ are ABS. $H_{16[:8]}$ denotes the first 8 rows (POS). Same object used by the Phase 6E.18 §6.1 $k$-NN metrics. A panel hash is recorded in the run manifest (E-6).

### 3.2 Discriminative Axis $\hat{u}_{\text{LDA}}$
The unit Fisher LDA axis computed on $H_{16}$ — **the exact Phase 6E.18 object** (sklearn LDA, `solver="lsqr"`, `shrinkage="auto"`), bound by `u_hat_LDA_hash == ff3cab496f8c219da96a9f59e144459cad6497113e4eaef0b1c0cd26525f3cc8` (registered from `readout_initialization.json`, §6).

### 3.3 Scaling Factor $s_{\text{scale}}$
$$s_{\text{scale}} = \frac{\text{std}(\text{margins}_{B0})}{\text{std}(H_{16} @ \hat{u}_{\text{LDA}}) + \varepsilon}, \quad \varepsilon = 10^{-12}$$
Identical to Phase 6E.18 §3.8. Reference value from 6E.18: $0.06681545$ (recorded; recomputed deterministically and matched at preflight).

### 3.4 Principal Basis $U_k$ (the Subspace $S_k$)
1. Center $H_{16}$ on its global mean and form the sample covariance $\Sigma_{\text{diff}} = X_{\text{centered}}^{\mathsf{T}} X_{\text{centered}} / (N - 1)$ (rank $\le 15$), exactly as Phase 6E.18 §8.
2. SVD: $\Sigma_{\text{diff}} = U S V^{\mathsf{T}}$, columns ordered by descending singular values.
3. **$U_k$ = the first $k$ columns of $U$**, for the nested set $k \in \{2, 4, 8\}$. Because the columns derive from a single SVD, $U_2 = U_8[:, :2]$ and $U_4 = U_8[:, :4]$ are **nested**: $\text{span}(U_2) \subset \text{span}(U_4) \subset \text{span}(U_8)$.
4. **$S_k := \text{span}(U_k)$** for $k \in \{2, 4, 8\}$. $\hat{u}_{\text{LDA}} \notin S_k$ is enforced by the $\kappa_k \le 0.995$ gate (§6); $\hat{u}_{\text{LDA}} \not\perp S_k$ is enforced by $\kappa_k \ge 0.05$ (§6).
5. **Hash binding:** $U_8$ must reproduce the Phase 6E.18 T2 basis: `basis_sha256(U_8) == b8dfbd6ae71277d5350b7ed2a513ccfb771b8e263102b843433c96cab8645060` (registered from `t2_basis.json`, k=8, cumulative energy $0.9277434349060059 \ge 0.90$; §6). Condition S8 is thus the **fixed-gradient twin of T2**: identical subspace, zero trainable capacity.

### 3.5 Projection Norms $\kappa_k$ and $\kappa_R$
$$\kappa_k = \|U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}}\|_2, \quad k \in \{2,4,8\}; \qquad \kappa_R = \|R_4R_4^{\mathsf{T}}\hat{u}_{\text{LDA}}\|_2$$
Registration gates in §6.

### 3.6 Frozen Vocabulary Margin $w_{\delta}$ (B0)
$w_{\delta} = W_{\text{emb}}[\text{PROPOSE}] - W_{\text{emb}}[\text{ABSTAIN}]$ (frozen embedding difference), $b_{\delta} = 0$.

### 3.7 Random Objects (Deterministic from Seed)
- **$R_4 \in \mathbb{R}^{896 \times 4}$:** orthonormal basis from the QR decomposition of a standard-normal $896 \times 4$ matrix drawn with `torch.Generator(seed = 20260820)` (E-8). Fixed and recorded.
- **$n_{\text{rand}} \in \mathbb{R}^{896}$:** unit random vector with $|\cos(n_{\text{rand}}, \hat{u}_{\text{LDA}})| \le 0.05$, drawn with the same seed (rejection sampling; deterministic). Same convention as Phase 6E.18 condition C2.

---

## 4. Experimental Condition Matrix ($N = 7$)

All seven conditions carry **EXACTLY ZERO ($0$) TRAINABLE READOUT PARAMETERS** and a **common readout gain** $\|w\|_2 = s_{\text{scale}}$ (normalization convention E-1), isolating direction (representation geometry) from gain.

| Cond | Architecture | Readout subspace | Forward readout $s = w^{\mathsf{T}}h_L + b$ | Trainable params | Primary isolation |
|---|---|---|---|---|---|
| **B0** | Matched Frozen Baseline | full ambient $\mathbb{R}^{896}$ | $w = w_{\delta}$, $b = 0$ | **0** | baseline regression reference |
| **F1** | 1D Fixed Fisher LDA | $\text{span}(\hat{u}_{\text{LDA}})$ | $w_{F1} = s_{\text{scale}}\hat{u}_{\text{LDA}}$, $b_{F1}$ | **0** | 1D discriminative replication |
| **S2** | Fixed Subspace $k{=}2$ | $\text{span}(U_2)$ | $w_{S2} = \frac{s_{\text{scale}}}{\kappa_2}U_2U_2^{\mathsf{T}}\hat{u}_{\text{LDA}}$, $b_{S2}$ | **0** | 2D principal-subspace filtering |
| **S4** | Fixed Subspace $k{=}4$ | $\text{span}(U_4)$ | $w_{S4} = \frac{s_{\text{scale}}}{\kappa_4}U_4U_4^{\mathsf{T}}\hat{u}_{\text{LDA}}$, $b_{S4}$ | **0** | 4D principal-subspace filtering |
| **S8** | Fixed Subspace $k{=}8$ | $\text{span}(U_8)$ (T2 basis) | $w_{S8} = \frac{s_{\text{scale}}}{\kappa_8}U_8U_8^{\mathsf{T}}\hat{u}_{\text{LDA}}$, $b_{S8}$ | **0** | 8D filtering; T2 dimensionality match, fixed gradient |
| **C4-Rand** | Random Fixed Subspace Control | $\text{span}(R_4)$ | $w_{C4} = \frac{s_{\text{scale}}}{\kappa_R}R_4R_4^{\mathsf{T}}\hat{u}_{\text{LDA}}$, $b_{C4}$ | **0** | random 4D subspace without alignment |
| **R1-Rand** | Random Fixed 1D Control | $\text{span}(n_{\text{rand}})$ | $w_{R1} = s_{\text{scale}}n_{\text{rand}}$, $b_{R1}$ | **0** | null-model reference; matched 1D control for F1 |

Bias convention (E-7): $b_c = -\text{median}(w_c @ H_{16[:8]})$ for every scalar condition $c$ (generalizing the Phase 6E.18 F1 bias rule; F1 reference value $-0.07310398$). For B0, $b_{\delta} = 0$ (difference of two logits).

---

## 5. Mathematical Formulations, Backward Geometry, and Pairwise Distinctness

### 5.1 Shared Forward/Backward Form
For any fixed readout weight $w$ with bias $b$:
- Forward: $s = w^{\mathsf{T}}h_L + b$.
- Backward: $\dfrac{\partial L}{\partial h_L} = w \cdot \dfrac{\partial L}{\partial s}$, so the decision-gradient contribution lies in $\text{span}(w)$, with **zero** contribution in $\text{span}(w)^{\perp}$.

### 5.2 Condition B0
$s = w_{\delta}^{\mathsf{T}}h_L$, $\dfrac{\partial L}{\partial h_L} = w_{\delta}\dfrac{\partial L}{\partial s} \in \text{span}(w_{\delta})$.

### 5.3 Conditions F1, S2, S4, S8 (Representation-Projection Formulation)
**F1:** $s = (s_{\text{scale}}\hat{u}_{\text{LDA}})^{\mathsf{T}}h_L + b_{F1}$; backward in $\text{span}(\hat{u}_{\text{LDA}})$.

**S$k$ ($k \in \{2,4,8\}$) — the representation projection:**
$$h' = U_kU_k^{\mathsf{T}}h_L, \qquad s = \frac{s_{\text{scale}}}{\kappa_k}\,\hat{u}_{\text{LDA}}^{\mathsf{T}} h' + b_{Sk}$$
with the algebraically equivalent fixed-weight form
$$w_{Sk} = \frac{s_{\text{scale}}}{\kappa_k}U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}}, \qquad s = w_{Sk}^{\mathsf{T}}h_L + b_{Sk}, \qquad b_{Sk} = -\text{median}(w_{Sk} @ H_{16[:8]}).$$
- $\|w_{Sk}\|_2 = s_{\text{scale}}$ (unit direction, common gain).
- Backward: $\dfrac{\partial L}{\partial h_L} = w_{Sk}\dfrac{\partial L}{\partial s} \in \text{span}(U_k) = S_k$; **every component of the decision gradient in $\text{span}(U_k)^{\perp}$ is exactly zero.** This is the precise, non-degenerate form of "gradient filtering in the subspace" that v1 claimed but never implemented.
- Because $\kappa_k \le 0.995$ (§6), $U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}} \neq \hat{u}_{\text{LDA}}$, so $w_{Sk} \neq w_{F1}$ in direction; the normalized readout directions $\hat{w}_{Sk} = U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}}/\kappa_k$ and $\hat{w}_{F1} = \hat{u}_{\text{LDA}}$ differ.

### 5.4 Condition C4-Rand
$$s = \frac{s_{\text{scale}}}{\kappa_R}\,\hat{u}_{\text{LDA}}^{\mathsf{T}} (R_4R_4^{\mathsf{T}}h_L) + b_{C4} \;\equiv\; w_{C4}^{\mathsf{T}}h_L + b_{C4}, \qquad w_{C4} = \frac{s_{\text{scale}}}{\kappa_R}R_4R_4^{\mathsf{T}}\hat{u}_{\text{LDA}}$$
- Backward in $\text{span}(R_4)$; $\|w_{C4}\|_2 = s_{\text{scale}}$; $b_{C4} = -\text{median}(w_{C4} @ H_{16[:8]})$.
- The random 4D subspace $R_4$ enters the mathematics (fixing B-3); $\kappa_R \le 0.30$ (§6) guarantees the aligned component of $\hat{u}_{\text{LDA}}$ in $R_4$ is small, so C4-Rand is a **weakly aligned random-subspace control matched in form and rank to S4** (like C2 is to I1 in 6E.18).

### 5.5 Condition R1-Rand
$$s = (s_{\text{scale}}n_{\text{rand}})^{\mathsf{T}}h_L + b_{R1}, \qquad b_{R1} = -\text{median}(w_{R1} @ H_{16[:8]}), \quad w_{R1} = s_{\text{scale}}n_{\text{rand}}$$
- Backward in $\text{span}(n_{\text{rand}})$; $|\cos(n_{\text{rand}}, \hat{u}_{\text{LDA}})| \le 0.05$; $\|w_{R1}\|_2 = s_{\text{scale}}$.

### 5.6 Pairwise Distinctness (Prospective)
- **B0 vs all:** $w_{\delta}$ is an embedding-derived vocabulary-margin vector, generically not in any $\text{span}(\hat{u}_{\text{LDA}}), \text{span}(U_k), \text{span}(R_4), \text{span}(n_{\text{rand}})$.
- **F1 vs S$k$:** $\hat{u}_{\text{LDA}} \notin S_k$ (gate $\kappa_k \le 0.995$) $\Rightarrow$ direction differs; also $\text{span}$ differ.
- **S2, S4, S8 mutually:** nested $U_2 \subset U_4 \subset U_8$ and the increment gates $\|(U_4U_4^{\mathsf{T}} - U_2U_2^{\mathsf{T}})\hat{u}_{\text{LDA}}\|_2 \ge 0.01$, $\|(U_8U_8^{\mathsf{T}} - U_4U_4^{\mathsf{T}})\hat{u}_{\text{LDA}}\|_2 \ge 0.01$ (§6) guarantee the projected directions, hence $w_{Sk}$, are pairwise distinct.
- **C4-Rand vs all:** $\text{span}(R_4)$ is a seeded random 4-plane; deterministically, $\hat{u}_{\text{LDA}} \notin \text{span}(R_4)$ (as $\kappa_R \le 0.30 < 0.995$) and $\text{span}(R_4) \neq \text{span}(U_4)$ (generically, verified at preflight); directions differ from F1, S$k$, R1, B0.
- **R1-Rand vs all:** $n_{\text{rand}}$ has $|\cos| \le 0.05$ with $\hat{u}_{\text{LDA}}$ (so $\neq$ F1, and generically $\notin$ any low-dimensional $\text{span}(U_k)$ or $\text{span}(R_4)$), and is 1D while $w_{C4} \in \text{span}(R_4)$ is a 4D projection.

All distinctness facts are verified numerically at preflight (§10.5) and recorded in the run manifest.

---

## 6. Registration Gates (Prospective, Deterministic, Non-Negotiable)

If any gate fails, the affected condition is declared **VOID** and must be resubmitted for human review. Silent re-derivation with altered inputs is prohibited.

| Gate | Requirement | Rationale |
|---|---|---|
| **G1** | `u_hat_LDA_hash == ff3cab49…` (exact, §3.2) | $\hat{u}_{\text{LDA}}$ reproduces the Phase 6E.18 object; conditions are cross-phase comparable |
| **G2** | `basis_sha256(U_8) == b8dfbd6a…` (exact, §3.4) | $U_8$ is the Phase 6E.18 T2 basis; S8 is T2's fixed-gradient twin |
| **G3** | $\kappa_k \in [0.05,\, 0.995]$ for $k \in \{2,4,8\}$ | $\hat{u}_{\text{LDA}} \notin S_k$ (non-degeneracy vs F1) and $\hat{u}_{\text{LDA}} \not\perp S_k$ (non-degenerate signal); $\kappa_8 > 0.995$ triggers the **S8 near-F1 VOID** (E-5) |
| **G4** | $\|(U_4U_4^{\mathsf{T}} - U_2U_2^{\mathsf{T}})\hat{u}_{\text{LDA}}\|_2 \ge 0.01$; $\|(U_8U_8^{\mathsf{T}} - U_4U_4^{\mathsf{T}})\hat{u}_{\text{LDA}}\|_2 \ge 0.01$ | ensures S2, S4, S8 are pairwise distinct in readout direction |
| **G5** | $\kappa_R \in [0.05,\, 0.30]$ | C4-Rand is weakly aligned (control), not near-F1 or null |
| **G6** | $\cos(H1)$ sign-assertion gate (synthetic vectors, $\varepsilon = 0.001$) passes | inherited Phase 6E.18 §6.2; violation VOIDs the run |

Reference 6E.18 values (informational, not thresholds): $s_{\text{scale}} = 0.06681545$, $b_{F1} = -0.07310398$; cumulative energy of $U_8 = 0.9277434349060059$.

---

## 7. Formal Hypotheses (Provisional; see §12 E-set and review)

- **H1 (Subspace Gradient Compatibility, Primary):** Both S4 and S8 achieve $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1) \ge +0.80$.
- **H2 (Subspace Utility Preservation, Primary):** Both S4 and S8 achieve $d(15) \ge +1.50$ and balanced accuracy $\ge 70\%$ at $t = 15$.
- **H3 (Dimensionality Gradient Filtering):** $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1)$ is non-decreasing in $k$: $S8 \ge S4 \ge S2$ (subspace restriction filters gradient noise).
- **H4 (Aligned vs Random Subspace):** S4 exceeds C4-Rand on $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1)$ by $\ge +0.30$, and C4-Rand fails utility ($d(15) < +0.50$) — alignment, not mere subspace restriction, drives both objectives.
- **H5 (Random-1D Null):** R1-Rand achieves $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1) \le +0.20$ (near-null reference), supporting the P1 null-model justification and F1 alignment specificity.
- **H6 (T2 Confound Resolution):** S8 matches or exceeds T2's gradient alignment ($\text{mean}_{t \in \{4,6,14,15\}}\cos(H1)(S8) \ge +0.80$; reference T2 $= +0.9878$) **while** $d(15)(S8) \ge +1.50$ (reference T2 $= +0.101$) — demonstrating that fixed, gain-normalized projection, not trainable capacity, resolves the alignment/utility decoupling.
- **H7 (Anti-Collapse / Anti-Drift):** Every scalar condition maintains $P(t) > 0.0\%$ for all $t \le 20$ and $d(20) \ge 0.00$ (no polar reversal, no late-training collapse).

---

## 8. Dual-Objective Acceptance Criteria (P1–P6)

All thresholds are **absolute constants** justified independently below; **none is a function of any Phase 6E.18/6E.19 observed value** (fixes audit finding M-1). Aggregation set is $\{4, 6, 14, 15\}$ (inherited).

| ID | Criterion | Metric | Threshold | Independent Justification |
|---|---|---|---|---|
| **P1** | Gradient Conflict — null tier | $\text{mean }\cos(H1)$ | $\ge +0.50$ | Null-model bound: a random readout direction (R1-Rand) is expected to yield mean $\cos \approx 0$; $+0.50$ is a round, strict-improvement constant |
| **P2** | Gradient Compatibility — RQ target | $\text{mean }\cos(H1)$ | $\ge +0.80$ | The research question's own target (§1.2) |
| **P3** | Margin Maintenance — large effect | Cohen's $d(15)$ | $\ge +0.80$ | Cohen's conventional "large" effect-size constant |
| **P4** | Margin Strength — RQ target | Cohen's $d(15)$ | $\ge +1.50$ | The research question's margin target (§1.2) |
| **P5** | Balanced Accuracy | $(a_{\text{POS}} + a_{\text{ABS}})/2$ at $t=15$ | $\ge 70\%$ | Chance $= 50\%$; $+20$ pp is a round, substantial margin; replaces v1's raw proposal-rate window $[60\%, 95\%]$ (which excluded F1's own $98.1\%$) |
| **P6** | Anti-Collapse / Anti-Drift | $P(t)$; $d(20)$ | $P(t) > 0.0\%\ \forall t \le 20$; $d(20) \ge 0.00$ | inherited Phase 6E.18 anti-degradation guard |

**JOINT SUCCESS GATE (the RQ demands both objectives simultaneously):** a condition is **SUCCESS** iff $P2 \wedge P4 \wedge P5 \wedge P6$.
**PARTIAL** iff $P1 \wedge P3$ (and not SUCCESS). Otherwise **FAIL**. Metrics follow the Phase 6E.18 §6.1 normative definitions, with $d(t)$ and balanced accuracy computed over the $N = 16$ panel at checkpoint $t$.

---

## 9. Pre-Registered Outcome Interpretation Matrix

### 9.1 Primary 2×2 (GC × TU)
- **GC** (gradient compatibility) $=$ P2: $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1) \ge +0.80$.
- **TU** (task utility) $=$ P4 $\wedge$ P5: $d(15) \ge +1.50$ and balanced accuracy $\ge 70\%$.

| | **TU high** ($d(15) \ge +1.50$, bal. acc. $\ge 70\%$) | **TU low** (otherwise) |
|---|---|---|
| **GC high** ($\cos(H1) \ge +0.80$) | **SUCCESS** — H1 $\wedge$ H2 supported. The fixed subspace readout resolves the 6E.18 decoupling: T2-grade alignment with F1-grade utility, with zero trainable capacity. | **GC-only** — decoupling persists within subspace geometry: replicates the T2 pattern ($+0.9878$ / $+0.101$) without trainable capacity. H2 rejected; subspace filtering discards discriminative variance. |
| **GC low** ($\cos(H1) < +0.80$) | **TU-only** — subspace restriction buys no gradient benefit over 1D LDA: replicates the F1 pattern ($+0.5831$ / $+1.813$). H1 rejected; F1 remains optimal. | **Both fail** — fixed subspace readout is inferior on both axes. H1 $\wedge$ H2 rejected. |

### 9.2 Secondary Rows
- **C4-Rand fails P3** (i.e., $d(15) < +0.80$) and S4 passes P1: **H4 supported** — alignment specificity confirmed.
- **R1-Rand** mean $\cos(H1) \le +0.20$: **H5 supported** — P1 null model confirmed; F1's $+0.5831$ is attributable to its aligned direction.
- **S8 $\approx$ T2 in alignment, S8 $>$ T2 in margin:** **H6 supported**.
- **B0 guard triggers** (collapse detection, §10.7): B0 truncates; all other conditions continue on the full horizon; the B0 reference is the §10.7 step-0 anchor.
- **Any protocol deviation in any condition:** that run is **VOID**; no verdict is generated from it.

---

## 10. Operational Discipline and Inherited Phase 6E.18 Rules

### 10.1 Zero Readout Parameters
Runtime assertion: `trainable_readout_params == 0` for all seven conditions (verified at every checkpoint).

### 10.2 Protocol Hash Binding
The execution script MUST bind and verify this document's SHA-256 hash before any checkpoint evaluation.

### 10.3 Data and Training Configuration (verbatim inheritance; E-10)
- Data: THEO SLM v0; train/dev split with `seed = 42` — identical to Phase 6E.18.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, $d = 896$, revision `7ae557604adf67be50417f59c2c2f167def9a775`; LoRA $r = 16$, $\alpha = 32$ (backbone LoRA only — no readout LoRA, no target-module changes).
- Optimizer: AdamW $\text{lr} = 10^{-4}$, weight decay $0.01$, gradient clip $1.0$; `grad_accum = 2`, `batch = 4`; $\lambda = 10$; horizon $T = 20$ steps.
- The readout is the ONLY difference between conditions; every other training input is identical across conditions.

### 10.4 Checkpoints and Aggregation
Canonical evaluation checkpoint set: $\{0, 4, 6, 7, 14, 15, 17, 20\}$. Decision-rule aggregation set: $\{4, 6, 14, 15\}$.

### 10.5 Gradient Cosine $\cos(H1)$ — Normative Binding
$\cos(H1)$ is the raw cosine between the mean POS and mean ABS **decision gradients with respect to the LoRA parameters** (`torch.autograd.grad(dec_loss, lora_param_list)`), without label-sign inversion:

$$\cos(H1) = \frac{\langle \bar{g}_{\text{POS}}, \bar{g}_{\text{ABS}} \rangle}{\|\bar{g}_{\text{POS}}\|_2\,\|\bar{g}_{\text{ABS}}\|_2}$$

Inherited §6.2 pre-verdict gate: a synthetic sign-assertion test ($u_1 = [1,0]^{\mathsf{T}}, u_2 = [-1,0]^{\mathsf{T}} \Rightarrow -1$; $v_1 = [1,0]^{\mathsf{T}}, v_2 = [1,0]^{\mathsf{T}} \Rightarrow +1$; tolerance $\varepsilon = 0.001$) MUST pass before any verdict; failure $\Rightarrow$ run VOID.

### 10.6 Preflight Mode (Required, Non-Training)
A non-training `--preflight` mode MUST verify, prior to any execution: all basis constructions and hashes (G1–G2), $\kappa$ gates (G3–G5), parameter counts, pairwise-distinctness checks (§5.6), sign-assertion gate (G6), and metric arithmetic.

### 10.7 Baseline Collapse Guard (B0 only)
If B0's detector guard triggers (collapse criterion, K = 3), B0 truncates at $t \le 6$; no other condition truncates. When a matched B0 comparator is unavailable, the frozen step-0 anchor applies and the missing comparator is recorded `UNAVAILABLE` (inherited Phase 6E.18 §7.1/§10). No trajectory back-filling and no cross-phase synthetic concatenation.

### 10.8 Void and Deviation Rules
Non-finite loss, a failed assertion gate, or a missing rule-mandated checkpoint in any condition $\Rightarrow$ that run is VOID; no verdict is generated from it. Phase 6E.18 execution logs and the Phase 6E.19 adjudication record remain permanently immutable.

### 10.9 Verdict Double Gate and Refusal Triplet
Scientific evaluation alone does NOT produce a verdict. Any attempt to issue a verdict without the technical evaluation AND human authorization returns the exact refusal triplet:
`PRE_EXECUTION` / `NO_SCIENTIFIC_VERDICT` / `EXECUTION_NOT_AUTHORIZED`.

### 10.10 Provenance and Authorization Artifacts
- `sha256_manifest.json` (hashes of code, base model, corpus, checkpoints, all §3 objects).
- `runs.json` (step-by-step telemetry), `pre_registration_snapshot.json` (this document's hash + provisional-decision log).
- `execution_authorization.json` schema: `{"decision": "APPROVE", "signature": ..., "date": ..., "phase": "6E.20"}`. Absent this file, no execution may occur.

---

## 11. Hard Stop and Human Authorization Gate

```
================================================================================
HARD STOP — NON-NEGOTIABLE
================================================================================
This document is a DESIGN-ONLY specification and does NOT authorize execution.
No model execution, training, fine-tuning, optimizer steps, or checkpoint
generation may occur without: (1) resolution of provisional decisions E-1..E-10,
(2) formal pre-execution audit of the finalized draft, and (3) explicit signed
human authorization (execution_authorization.json, decision=APPROVE, phase=6E.20).

Sign-off Gate:
  Reviewer: ______________________________
  Decision: [ ] APPROVE   [ ] AMEND   [ ] REJECT
  Date:     ______________________________
  Signature: ______________________________
================================================================================
```

---

## 12. Provisional Design Decisions (E-1 … E-10) — PENDING HUMAN REVIEW

Each carries the recommended default used in this draft. Adoption requires explicit human confirmation; deviation requires a documented amendment.

| ID | Decision | Recommended Default (used in this draft) |
|---|---|---|
| **E-1** | Readout normalization convention | Unit direction, common gain: $\|w\|_2 = s_{\text{scale}}$ for all scalar conditions ($w_{Sk} = (s_{\text{scale}}/\kappa_k)U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}}$). Isolates direction from gain. |
| **E-2** | C4-Rand definition | $w_{C4} = (s_{\text{scale}}/\kappa_R)R_4R_4^{\mathsf{T}}\hat{u}_{\text{LDA}}$ — 4D random-subspace readout, same functional form as S4. |
| **E-3** | R1-Rand inclusion | **Include** as the 7th condition: matched 1D random control for F1 and the null reference for P1/H5. |
| **E-4** | $\kappa$-gate bands | $\kappa_k \in [0.05, 0.995]$; inter-$k$ increments $\ge 0.01$; $\kappa_R \in [0.05, 0.30]$. |
| **E-5** | S8 near-F1 VOID | $\kappa_8 > 0.995 \Rightarrow$ S8 VOID (subsumed by G3). Residual $\|(I - U_8U_8^{\mathsf{T}})\hat{u}_{\text{LDA}}\| \ge \sqrt{1 - 0.995^2} \approx 0.0999$ guarantees direction separation. |
| **E-6** | Step-0 panel hash binding | Bind $\hat{u}_{\text{LDA}}$ via `u_hat_LDA_hash` (G1), $U_8$ via `basis_sha256` (G2), and additionally record an `H_16` panel hash in the run manifest. |
| **E-7** | Bias convention | $b_c = -\text{median}(w_c @ H_{16[:8]})$ for all scalar conditions (generalizes Phase 6E.18 F1 rule). |
| **E-8** | Random-seed value | `20260820` for $R_4$ and $n_{\text{rand}}$ via explicit `torch.Generator`. |
| **E-9** | P5 checkpoint | Balanced accuracy at $t = 15$ (matches the $d(15)$ evaluation point). |
| **E-10** | LoRA/training inheritance | Verbatim Phase 6E.18 training configuration (§10.3). |

---

## 13. Amendment and Reconciliation Log (v1.0 → v2.0)

- **B-1 (BLOCKING) — resolved structurally:** $S_k$ redefined as $U_k$ (top-$k$ principal subspace of $\Sigma_{\text{diff}}$, §3.4); $\hat{u}_{\text{LDA}} \notin S_k$ enforced by G3; readout is a representation projection (§5.3) with backward gradient in $\text{span}(U_k)$. The v1 identity $P_{Sk}\hat{u}_{\text{LDA}} = \hat{u}_{\text{LDA}}$ (verified $\approx 2\times10^{-16}$) can no longer arise.
- **B-2 (BLOCKING) — resolved:** P2 is now the absolute RQ target $\ge +0.80$; the post-hoc `mean(F1) + 0.10` criterion and the unpassable joint gate are deleted. Thresholds are absolute constants (§8, fixes M-1).
- **B-3 (BLOCKING) — resolved:** C4-Rand now uses $R_4$ in its mathematics with a defined seed and bias; $\kappa_R$ gate (G5) guards non-degeneracy.
- **M-2 (MAJOR) — resolved:** JOINT SUCCESS GATE matches the RQ: $P2 \wedge P4 \wedge P5 \wedge P6$ (§8).
- **M-3 (MAJOR) — resolved:** complete operational section (§10) inheriting checkpoints/horizon, training config, data/split, hashes, RNG rules, VOID rules, and the explicit P2 reference — no F1 ambiguity.
- **M-4 (MAJOR) — resolved:** T2-basis provenance pinned via `basis_sha256` (G2), construction convention (§3.4), and the near-F1 guard G3/G5.
- **M-5 (MAJOR) — resolved:** basis hashing registered (G2), plus $\hat{u}_{\text{LDA}}$ hash (G1) and panel hash (E-6).
- **Structure:** S8-Proj subsumed into S8 (identical subspace, now fixed-gradient); the deflated-covariance variant is dropped as degenerate (§2); R1-Rand added as the 7th condition (E-3); hypotheses restructured as H1–H7 (§7); outcome matrix replaced by the GC × TU 2×2 (§9.1).

---

*End of document — DRAFT v2.0. Provisional decisions E-1 … E-10 pending human review. Execution remains HARD-STOPPED.*
