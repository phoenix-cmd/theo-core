# THEO SLM Fixed Subspace Readout Authorization Protocol — DRAFT v3.0

- **Status:** DRAFT v3.0 — DESIGN ONLY. **NOT AUTHORIZED FOR EXECUTION.**
- **Supersedes:** `theo-slm-fixed-subspace-readout-authorization-protocol-draft-v2.md` (v2.0, SHA-256 `07E796AF15E462CB11C77FEF01CC8D6144E5472EA50276E5250F68BD482AF6E1`, 305 lines). v2.0 was audited at artifact level (read-only; v2 unmodified) with verdict **AMEND AND RESUBMIT**: one blocking finding (B-4), 13 MAJOR (M-6 … M-17), 12 MINOR (m-1 … m-13), 4 INFO (i-1 … i-4). v3.0 is the materialization of that verdict.
- **Decisions E-1 … E-10:** **ADOPTED** by explicit human decision dated 2026-08-16 (§12). Adoption is a *human* decision, not an agent recommendation; §13 records the adoption basis per decision.
- **Phase:** Phase 6E.20 (Fixed Low-Dimensional & Subspace Readout Architectures).
- **Authority:** Requires formal pre-execution audit of this finalized draft and explicit human authorization (`execution_authorization.json`, decision=APPROVE, phase=6E.20). No model execution, training, fine-tuning, optimizer steps, or checkpoint generation is authorized by this document.
- **Basis & Provenance:** Phase 6E.18 Execution Findings (`phase-6e18-causal-intervention`); Phase 6E.19 Post-Execution Forensic Adjudication Record (`theo-slm-phase-6e18-post-execution-forensic-analysis-v1.md`); Phase 6E.18 authoritative protocol `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.3.md` (source of all inherited operational rules in §10); the v2.0 artifact-level audit finding register (B-4, M-6 … M-17, m-1 … m-13, i-1 … i-4).
- **Core Problem Addressed:** Decoupling of gradient compatibility ($\cos(H1)$) and downstream decision utility (margin separation $d(15)$, balanced proposal rate) established in Phase 6E.18.

---

## 1. Scientific Motivation and Theoretical Framing

### 1.1 Empirical Baseline from Phase 6E.18
Phase 6E.18 established three fundamental empirical facts:
1. **Decoupling of Gradient Compatibility and Task Utility:** Maximizing gradient cosine ($\cos(H1)$) is mathematically distinct from maintaining discriminative decision utility. Condition T2 achieved near-perfect gradient alignment ($\cos(H1) = +0.9878$) by restricting updates to a rank-8 SVD subspace, but suffered from near-zero decision margin ($d(15) = +0.101$) and proposal decay ($19.2\%$).
2. **Superiority of Fixed Aligned Geometry (F1):** Condition F1 (fixed 1D Fisher LDA projection, $0$ trainable parameters) resolved gradient conflict ($\cos(H1) = +0.5831$), preserved strong margin ($d(15) = +1.813$), and maintained balanced proposal rates ($98.1\%$), outperforming trainable readouts (I1, C2).
3. **Trainable Readout Instability:** Trainable heads (I1, C2, V1) introduce non-stationarity, bias inflation, and parameter drift:
   - I1 drifted from balanced classification to $100\%$ proposal over-prediction.
   - C2 achieved positive gradient cosine ($+0.3066$) via capacity, but its margin inverted to $-0.169$ at step 20.
   - V1 experienced a late-training polar reversal, dropping from $100\%$ to $0.0\%$ proposal rate at step 20.

All values above are cited as informational context only (Phase 6E.19 re-verified them). **No Phase 6E.18/6E.19 observed value enters any Phase 6E.20 threshold** (§8, §12 E-1).

### 1.2 The Phase 6E.20 Research Question
> **Can a fixed, low-dimensional subspace readout architecture combine the superior decision utility and margin separation of Condition F1 ($d > 1.50$, balanced proposal rate) with the gradient compatibility of Condition T2 ($\cos(H1) \ge +0.80$), without introducing trainable head confounds, bias drift, or optimization non-stationarity?**

The phrase "orthogonal noise filtering" is deliberately **not** used in this document. Phase 6E.18's T2 result was a *restriction of the trainable update to a subspace*, not a measured property of the total decision gradient. v2.0's claim that the *total* decision gradient is confined to $\text{span}(U_k)$ is **false** and is withdrawn in §5.3 (finding B-4). The retained, testable hypothesis is that the *readout contribution to the decision gradient* is confined to $\text{span}(U_k)$, and that this confined contribution measurably steers the total decision gradient (§5.7, §7 H3, §8 P1).

---

## 2. Design Rationale: Why v1 Was Rejected, What v2 Fixed, and What v3 Corrects

### 2.1 v1.0 Blocking Findings (Fixed in v2, retained in v3)
| Blocking Finding | v1.0 Defect | Resolution (v2, retained) |
|---|---|---|
| **B-1** | $S_k$ contained $\hat{u}_{\text{LDA}}$, so every subspace condition $S2/S4/S8$ was mathematically identical to F1 (verified numerically: $\|P_k\hat{u}_{\text{LDA}} - \hat{u}_{\text{LDA}}\|_2 \approx 2 \times 10^{-16}$). | $S_k$ redefined as the top-$k$ principal subspace $U_k$ of $\Sigma_{\text{diff}}$ (§3.4), which does not contain $\hat{u}_{\text{LDA}}$ (gate $\kappa_k \le 0.995$, §6). Readout is a representation projection (§5.3). |
| **B-2** | Criterion P2 was `mean cos(H1)(Sk) >= mean(F1) + 0.10` — post hoc and unsatisfiable. | P2 replaced by the absolute RQ target $\ge +0.80$ (§8). No observed value enters any threshold. |
| **B-3** | Condition C4-Rand's 4D subspace $R_4$ never entered the mathematics; $b_{\text{rand}}$ undefined; no seed. | C4-Rand is a genuine 4D random-subspace readout with the same functional form as S4, deterministic from an explicit seed, with $\kappa_R$ registered (§5.4, §6). |

Two candidate geometries were evaluated and rejected as provably degenerate:
- $S_k \ni \hat{u}_{\text{LDA}}$ (v1's construction): reduces to F1 — see B-1.
- $S_k \perp \hat{u}_{\text{LDA}}$: the projection $P_{Sk}\hat{u}_{\text{LDA}} = 0$, so the readout weight vanishes — a degenerate null control, not a subspace hypothesis.

### 2.2 v2.0 Audit Findings (Fixed in v3)
- **B-4 (BLOCKING) — resolved by re-specification (human decision 2026-08-16, §12 E-11):** v2 §5.3 claimed "every component of the decision gradient in $\text{span}(U_k)^{\perp}$ is exactly zero" and §5.1 claimed the *decision-gradient contribution* is the whole gradient. Under §10.5's literal object — `torch.autograd.grad(dec_loss, lora_param_list)` where `dec_loss` is the full-vocabulary CE at the decision position with only PRO/AB patched (Phase 6E.18 runner, lines 1019–1027) — the measured gradient is a *sum* of a free LM-head term and the readout term, so it is **not** confined to $\text{span}(U_k)$. The readout-only contribution *is* confined. v3 re-specifies the primary gradient object as the readout-only decision gradient $g_R$ (§5.7) and restates every confinement claim on that object; $\cos(H1)$ remains the 6E.18-compatible total-gradient cosine as the auxiliary metric (§10.5). Numerical check on synthetic vectors: readout-only contribution off-span $\approx 4\times10^{-8}$; total-gradient off-span $1.63$ vs readout-span norm $0.21$; S2-vs-F1 total-gradient difference norm $0.246$.
- **M-6/M-7 (H3 mechanism, honest characterization):** H3 no longer claims "subspace restriction filters gradient noise." The family is characterized honestly: seven *fixed, zero-trainable-parameter linear readouts* that differ only in weight direction; the mechanism tested is whether the readout contribution dominates and steers the total decision gradient (§7 H3, §8 P1).
- **M-8 (G1/G2 verifiability):** Phase 6E.18 artifacts store only hashes/spectra (no $U_8$ or $\hat{u}_{\text{LDA}}$ tensors; SVD column sign unpinned). v3 pins a canonical sign convention and hashing rule and mandates storing the full reference tensors (`.npy`) at step 0; G1/G2 additionally verify deterministic **reconstruction** from the frozen panel (§3.4, §6, §10.6).
- **M-9 (gate criteria not frozen; E-4 embedded):** E-4 is now adopted and the gate criteria are frozen (§12). An explicit VOID contingency exists for $\kappa_4/\kappa_8$ (and $\kappa_2$) falling outside the bands (§6 G3).
- **M-10 (H6 composite confound):** H6 is restated as a joint-effect hypothesis; C4-Rand and R1-Rand are the matching-capacity fixed-gradient controls that anchor interpretation (§7 H6).
- **M-11 (P1 null):** P1 no longer rests on "random readout yields $\cos \approx 0$" (unsound: readout-dominant gradients approach $\pm 1$, background-dominant gradients are indeterminate). P1 is re-specified as the readout-steering dominance test (§8 P1).
- **M-12 (procedurally post-hoc thresholds):** Thresholds are absolute constants with independent justifications; no 6E.18/6E.19 observed value appears in any criterion or justification (§8).
- **M-13 (P5 resubstitution):** P5 keeps balanced accuracy at $t=15$ on the $N=16$ panel as the primary gate and adds a **leave-one-panel-out (LOO-CV) comparator** reported alongside (non-gating) (§8 P5).
- **M-14 (P6 omitted from 2×2):** The SUCCESS cell of the outcome matrix now explicitly requires P6 (§9.1).
- **M-15 (H4 criteria mismatch):** H4 and §9.2 now both use P3's threshold: C4-Rand fails utility iff $d(15) < +0.80$ (§7 H4, §9.2).
- **M-16 (§10.3 verbatim inheritance incomplete):** §10.3 now enumerates the LoRA target modules, dropout, decision-position prompt convention, decision index, and batch RNG rule verbatim (§10.3).
- **M-17 (gradient object undefined):** $\cos(H1)$'s gradient object is now defined by §10.5's dual-object specification; no free text remains.
- **m-1 … m-13 (minor):** fixed throughout (§3, §4, §6, §7, §8, §9, §10) — see §13 for the itemized log.

---

## 3. Shared Construction Objects (Frozen at Step 0)

All objects below are computed on the **frozen step-0 panel** and are deterministic (no RNG, except where explicitly seeded). All are recorded in the run manifest, and the full reference tensors are stored as artifacts (`.npy` + SHA-256) so every gate is verifiable against stored objects, not only hashes.

### 3.1 Step-0 Panel $H_{16}$ and Label Vector $y_{16}$
- $H_{16} \in \mathbb{R}^{16 \times 896}$: the layer-24 last-token hidden states $h_L$ of the frozen base model (before LoRA adaptation) over the $N = 16$ panel; rows $0{:}8$ are POS, rows $8{:}16$ are ABS. $H_{16[:8]}$ denotes the first 8 rows (POS). Same object used by the Phase 6E.18 §6.1 $k$-NN metrics.
- **Label vector $y_{16}$ is pinned (fixes m-3):** $y_{16} = [+1]^8 \oplus [-1]^8$ (POS $= +1$, ABS $= -1$), recorded in the run manifest as `labels` with its own hash. Decision rule: margin $s > 0 \Rightarrow$ POS, $s \le 0 \Rightarrow$ ABS, exactly as Phase 6E.18.
- A panel hash of $H_{16}$ (float64 canonical serialization, §3.4 rule) is recorded in the run manifest (E-6).

### 3.2 Discriminative Axis $\hat{u}_{\text{LDA}}$
The unit Fisher LDA axis computed on $H_{16}$ — **the exact Phase 6E.18 object** (sklearn LDA, `solver="lsqr"`, `shrinkage="auto"`), bound by `u_hat_LDA_hash == ff3cab496f8c219da96a9f59e144459cad6497113e4eaef0b1c0cd26525f3cc8` (registered from `readout_initialization.json`). The full tensor is stored as `u_lda.npy` with SHA-256 `u_lda_tensor_hash` (fixes M-8: gate G1 verifies both the registered hash and the reconstructed tensor match).

### 3.3 Scaling Factor $s_{\text{scale}}$
$$s_{\text{scale}} = \frac{\text{std}(\text{margins}_{B0})}{\text{std}(H_{16} @ \hat{u}_{\text{LDA}}) + \varepsilon}, \quad \varepsilon = 10^{-12}$$
Identical to Phase 6E.18 §3.8. Reference value from 6E.18: $0.06681545$ (recorded; recomputed deterministically and matched at preflight).
**Degenerate guard (fixes i-3):** if $s_{\text{scale}}$ is non-finite or $\le 0$, registration fails; the run is VOID (§6 G8).

### 3.4 Principal Basis $U_k$ (the Subspace $S_k$)
1. Center $H_{16}$ on its global mean and form the sample covariance $\Sigma_{\text{diff}} = X_{\text{centered}}^{\mathsf{T}} X_{\text{centered}} / (N - 1)$ (rank $\le 15$), exactly as Phase 6E.18 §8.
2. SVD: $\Sigma_{\text{diff}} = U S V^{\mathsf{T}}$, columns ordered by descending singular values.
3. **Canonical sign convention (fixes M-8):** for each column of $U$, multiply by $-1$ if the largest-magnitude element is negative (ties broken by the lowest row index). This pins the SVD column signs and makes $U$ reproducible byte-for-byte.
4. **Canonical hashing rule (fixes M-8):** all tensor hashes are SHA-256 of the float64 canonical serialization: tensor → numpy float64, C-contiguous, little-endian raw bytes. (Phase 6E.18's `t2_basis.json` stored only `basis_sha256` plus the spectrum — no $U_8$ tensor; v3 stores the full tensors going forward.)
5. **$U_k$ = the first $k$ columns of $U$**, for the nested set $k \in \{2, 4, 8\}$. Because the columns derive from a single SVD, $U_2 = U_8[:, :2]$ and $U_4 = U_8[:, :4]$ are **nested**: $\text{span}(U_2) \subset \text{span}(U_4) \subset \text{span}(U_8)$.
6. **$S_k := \text{span}(U_k)$** for $k \in \{2, 4, 8\}$. $\hat{u}_{\text{LDA}} \notin S_k$ is enforced by the $\kappa_k \le 0.995$ gate (§6 G3); $\hat{u}_{\text{LDA}} \not\perp S_k$ is enforced by $\kappa_k \ge 0.05$ (§6 G3).
7. **Hash binding:** $U_8$ must reproduce the Phase 6E.18 T2 basis: `basis_sha256(U_8) == b8dfbd6ae71277d5350b7ed2a513ccfb771b8e263102b843433c96cab8645060` (registered from `t2_basis.json`, k=8, cumulative energy $0.9277434349060059 \ge 0.90$). Condition S8 is thus the **fixed-gradient twin of T2**: identical subspace, zero trainable capacity. The full $U_8$ tensor is stored as `U_8.npy` with `U_8_tensor_hash` (G2 verifies registered hash **and** reconstruction).
8. All seven conditions store their readout weight tensors at step 0: `w_<cond>.npy` with per-tensor hashes (fixes m-4 analog for readout weights).

### 3.5 Projection Norms $\kappa_k$ and $\kappa_R$
$$\kappa_k = \|U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}}\|_2, \quad k \in \{2,4,8\}; \qquad \kappa_R = \|R_4R_4^{\mathsf{T}}\hat{u}_{\text{LDA}}\|_2$$
Registration gates in §6. All $\kappa$ values must be finite and in $(0, 1)$ (G8).

### 3.6 Frozen Vocabulary Margin $w_{\delta}$ (B0)
$w_{\delta} = W_{\text{emb}}[\text{PROPOSE}] - W_{\text{emb}}[\text{ABSTAIN}]$ (frozen embedding difference), $b_{\delta} = 0$. Excluded from the common-gain claim (fixes m-1): B0 is the vocabulary-margin baseline reference, not a gain-normalized readout.

### 3.7 Random Objects (Deterministic from Seed)
- **$R_4 \in \mathbb{R}^{896 \times 4}$:** orthonormal basis from the QR decomposition of a standard-normal $896 \times 4$ matrix drawn with `torch.Generator(seed = 20260820)` (E-8). Fixed, recorded, and stored as `R_4.npy` with `R_4_tensor_hash` (fixes m-4).
- **$n_{\text{rand}} \in \mathbb{R}^{896}$:** unit random vector with $|\cos(n_{\text{rand}}, \hat{u}_{\text{LDA}})| \le 0.05$, drawn with the same seed (rejection sampling; deterministic). Same convention as Phase 6E.18 condition C2. Stored as `n_rand.npy` with `n_rand_tensor_hash` (fixes m-4).

---

## 4. Experimental Condition Matrix ($N = 7$)

All seven conditions carry **EXACTLY ZERO ($0$) TRAINABLE READOUT PARAMETERS** (verified at every checkpoint, §10.1). The **common readout gain** $\|w\|_2 = s_{\text{scale}}$ (E-1) applies to the **six scalar readouts** F1, S2, S4, S8, C4-Rand, R1-Rand, isolating direction (representation geometry) from gain; **B0 is excluded** from the common-gain claim ($w = w_{\delta}$, $b = 0$; fixes m-1).

| Cond | Architecture | Readout subspace | Forward readout $s = w^{\mathsf{T}}h_L + b$ | Trainable params | Primary isolation |
|---|---|---|---|---|---|
| **B0** | Matched Frozen Baseline | full ambient $\mathbb{R}^{896}$ | $w = w_{\delta}$, $b = 0$ | **0** | baseline regression reference |
| **F1** | 1D Fixed Fisher LDA | $\text{span}(\hat{u}_{\text{LDA}})$ | $w_{F1} = s_{\text{scale}}\hat{u}_{\text{LDA}}$, $b_{F1}$ | **0** | 1D discriminative replication |
| **S2** | Fixed Subspace $k{=}2$ | $\text{span}(U_2)$ | $w_{S2} = \frac{s_{\text{scale}}}{\kappa_2}U_2U_2^{\mathsf{T}}\hat{u}_{\text{LDA}}$, $b_{S2}$ | **0** | 2D principal-subspace readout |
| **S4** | Fixed Subspace $k{=}4$ | $\text{span}(U_4)$ | $w_{S4} = \frac{s_{\text{scale}}}{\kappa_4}U_4U_4^{\mathsf{T}}\hat{u}_{\text{LDA}}$, $b_{S4}$ | **0** | 4D principal-subspace readout |
| **S8** | Fixed Subspace $k{=}8$ | $\text{span}(U_8)$ (T2 basis) | $w_{S8} = \frac{s_{\text{scale}}}{\kappa_8}U_8U_8^{\mathsf{T}}\hat{u}_{\text{LDA}}$, $b_{S8}$ | **0** | 8D principal-subspace readout; T2 dimensionality match, fixed gradient |
| **C4-Rand** | Random Fixed Subspace Control | $\text{span}(R_4)$ | $w_{C4} = \frac{s_{\text{scale}}}{\kappa_R}R_4R_4^{\mathsf{T}}\hat{u}_{\text{LDA}}$, $b_{C4}$ | **0** | random 4D subspace without alignment |
| **R1-Rand** | Random Fixed 1D Control | $\text{span}(n_{\text{rand}})$ | $w_{R1} = s_{\text{scale}}n_{\text{rand}}$, $b_{R1}$ | **0** | matched 1D random control for F1 |

Bias convention (E-7): $b_c = -\text{median}(w_c @ H_{16[:8]})$ for every scalar condition $c$ (generalizing the Phase 6E.18 F1 bias rule; F1 reference value $-0.07310398$). For B0, $b_{\delta} = 0$ (difference of two logits).

---

## 5. Mathematical Formulations, Backward Geometry, and Pairwise Distinctness

### 5.1 Shared Forward/Backward Form
For any fixed readout weight $w$ with bias $b$:
- Forward: $s = w^{\mathsf{T}}h_L + b$; the readout emits logits $z_{\text{PRO}} = s$, $z_{\text{ABS}} = -s$ (exactly Phase 6E.18: runner lines 331–345, 375–379).
- Backward (readout-only contribution): $\dfrac{\partial L}{\partial h_L}\Big|_{\text{readout}} = \alpha\, w$, with $\alpha$ defined in §5.7; this contribution lies in $\text{span}(w)$, with **zero** contribution in $\text{span}(w)^{\perp}$.

**Scope note (fixes B-4):** the confinement statement above is about the **readout-only contribution** to the decision gradient, not the total gradient. The total decision gradient also contains a free LM-head term (§5.7) and is **not** confined to $\text{span}(w)$.

### 5.2 Condition B0
$s = w_{\delta}^{\mathsf{T}}h_L$, readout-only contribution $\dfrac{\partial L}{\partial h_L}\Big|_{\text{readout}} = \alpha\, w_{\delta} \in \text{span}(w_{\delta})$.

### 5.3 Conditions F1, S2, S4, S8 (Representation-Projection Formulation)
**F1:** $s = (s_{\text{scale}}\hat{u}_{\text{LDA}})^{\mathsf{T}}h_L + b_{F1}$; readout-only contribution in $\text{span}(\hat{u}_{\text{LDA}})$.

**S$k$ ($k \in \{2,4,8\}$) — the representation projection:**
$$h' = U_kU_k^{\mathsf{T}}h_L, \qquad s = \frac{s_{\text{scale}}}{\kappa_k}\,\hat{u}_{\text{LDA}}^{\mathsf{T}} h' + b_{Sk}$$
with the algebraically equivalent fixed-weight form
$$w_{Sk} = \frac{s_{\text{scale}}}{\kappa_k}U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}}, \qquad s = w_{Sk}^{\mathsf{T}}h_L + b_{Sk}, \qquad b_{Sk} = -\text{median}(w_{Sk} @ H_{16[:8]}).$$
- $\|w_{Sk}\|_2 = s_{\text{scale}}$ (unit direction, common gain).
- **Confinement theorem (B-4 resolution):** $w_{Sk} \in \text{span}(U_k)$ by construction, hence the readout-only decision gradient $\alpha\, w_{Sk} \in \text{span}(U_k) = S_k$, and **every component of the readout-only decision gradient in $\text{span}(U_k)^{\perp}$ is exactly zero.** This is a computed registration invariant, verified numerically at preflight (gate G7, §6): $\|(I - U_kU_k^{\mathsf{T}})\,( \alpha_{c}\, w_{Sk})\|_2 \le 10^{-6}\,\|\alpha_{c}\, w_{Sk}\|_2$ for $c \in \{\text{POS}, \text{ABS}\}$.
- **Withdrawn claim (B-4):** v2.0's statement that "every component of the *decision gradient* in $\text{span}(U_k)^{\perp}$ is exactly zero" (where the object is the total $\texttt{autograd.grad(dec\_loss, lora\_param\_list)}$) is **false** and is not made in v3. The total decision gradient is $\alpha\, w_{Sk}$ **plus** a free LM-head background term (§5.7).
- Because $\kappa_k \le 0.995$ (§6), $U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}} \neq \hat{u}_{\text{LDA}}$, so $w_{Sk} \neq w_{F1}$ in direction; the normalized readout directions $\hat{w}_{Sk} = U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}}/\kappa_k$ and $\hat{w}_{F1} = \hat{u}_{\text{LDA}}$ differ.

### 5.4 Condition C4-Rand
$$s = \frac{s_{\text{scale}}}{\kappa_R}\,\hat{u}_{\text{LDA}}^{\mathsf{T}} (R_4R_4^{\mathsf{T}}h_L) + b_{C4} \;\equiv\; w_{C4}^{\mathsf{T}}h_L + b_{C4}, \qquad w_{C4} = \frac{s_{\text{scale}}}{\kappa_R}R_4R_4^{\mathsf{T}}\hat{u}_{\text{LDA}}$$
- Readout-only contribution in $\text{span}(R_4)$; $\|w_{C4}\|_2 = s_{\text{scale}}$; $b_{C4} = -\text{median}(w_{C4} @ H_{16[:8]})$.
- The random 4D subspace $R_4$ enters the mathematics (fixing B-3); $\kappa_R \le 0.30$ (§6) guarantees the aligned component of $\hat{u}_{\text{LDA}}$ in $R_4$ is small, so C4-Rand is a **weakly aligned random-subspace control matched in form and rank to S4** (like C2 is to I1 in 6E.18).

### 5.5 Condition R1-Rand
$$s = (s_{\text{scale}}n_{\text{rand}})^{\mathsf{T}}h_L + b_{R1}, \qquad b_{R1} = -\text{median}(w_{R1} @ H_{16[:8]}), \quad w_{R1} = s_{\text{scale}}n_{\text{rand}}$$
- Readout-only contribution in $\text{span}(n_{\text{rand}})$; $|\cos(n_{\text{rand}}, \hat{u}_{\text{LDA}})| \le 0.05$; $\|w_{R1}\|_2 = s_{\text{scale}}$.

### 5.6 Pairwise Distinctness (Prospective)
- **B0 vs all:** $w_{\delta}$ is an embedding-derived vocabulary-margin vector, generically not in any $\text{span}(\hat{u}_{\text{LDA}}), \text{span}(U_k), \text{span}(R_4), \text{span}(n_{\text{rand}})$.
- **F1 vs S$k$:** $\hat{u}_{\text{LDA}} \notin S_k$ (gate $\kappa_k \le 0.995$) $\Rightarrow$ direction differs; also $\text{span}$ differ.
- **S2, S4, S8 mutually:** nested $U_2 \subset U_4 \subset U_8$ and the increment gates $\|(U_4U_4^{\mathsf{T}} - U_2U_2^{\mathsf{T}})\hat{u}_{\text{LDA}}\|_2 \ge 0.01$, $\|(U_8U_8^{\mathsf{T}} - U_4U_4^{\mathsf{T}})\hat{u}_{\text{LDA}}\|_2 \ge 0.01$ (§6 G4) guarantee the projected directions, hence $w_{Sk}$, are pairwise distinct.
- **C4-Rand vs all:** $\text{span}(R_4)$ is a seeded random 4-plane; deterministically, $\hat{u}_{\text{LDA}} \notin \text{span}(R_4)$ (as $\kappa_R \le 0.30 < 0.995$) and $\text{span}(R_4) \neq \text{span}(U_4)$ (generically, verified at preflight); directions differ from F1, S$k$, R1, B0.
- **R1-Rand vs all:** $n_{\text{rand}}$ has $|\cos| \le 0.05$ with $\hat{u}_{\text{LDA}}$ (so $\neq$ F1, and generically $\notin$ any low-dimensional $\text{span}(U_k)$ or $\text{span}(R_4)$), and is 1D while $w_{C4} \in \text{span}(R_4)$ is a 4D projection.

All distinctness facts are verified numerically at preflight (§10.6) and recorded in the run manifest.

### 5.7 Gradient Decomposition and Metric Objects (fixes M-17, B-4)
This section defines the two gradient objects used throughout. The reference computation is the Phase 6E.18 runner, `per_sample_decision_gradient` (lines 1006–1033): `dec_loss` is the full-vocabulary cross-entropy at the decision position $d = v_{\text{idx}}[4]$ with only `dec_logits[PRO_ID]` and `dec_logits[AB_ID]` replaced by the readout logits $(s, -s)$.

**Object 1 — readout-only decision gradient $g_R$ (primary, new in v3):** the gradient of the readout contribution to `dec_loss` with respect to the LoRA parameters,
$$g_R^{\theta} \;=\; \alpha_{c}\, \nabla_{\theta} s, \qquad \nabla_{\theta} s = \texttt{torch.autograd.grad}(s,\; \texttt{lora\_param\_list})$$
with scalar coefficients derived by the chain rule from the patched full-vocabulary CE (verified on synthetic vectors):
$$\alpha_{\text{POS}} = p_{\text{PRO}} - p_{\text{ABS}} - 1, \qquad \alpha_{\text{ABS}} = p_{\text{PRO}} - p_{\text{ABS}} + 1,$$
where $p_{\text{PRO}}, p_{\text{ABS}}$ are the model's (pre-patch) softmax probabilities at the decision position. Equivalently, in hidden space, $g_R^{h} = \alpha_{c}\, w_c$, which lies in $\text{span}(w_c)$ and, for $S_k$, in $\text{span}(U_k)$ (§5.3 confinement theorem). Because $z_{\text{ABS}} = -s$, the readout contribution to the two class-conditional gradients is exactly $\alpha_{c}\, \nabla_{\theta} s$; this is verifiable by comparing `dec_loss` backward against the readout-scalar backward.

**Object 2 — total decision gradient $g_{\text{total}}$ (auxiliary, 6E.18-compatible):** the exact Phase 6E.18 object,
$$g_{\text{total}}^{\theta} = \texttt{torch.autograd.grad}(\texttt{dec\_loss},\; \texttt{lora\_param\_list}).$$
By the chain rule it decomposes as
$$g_{\text{total}} = g_{\text{bg}} + g_R, \qquad g_{\text{bg}} = \sum_{v \notin \{\text{PRO},\text{ABS}\}} p_v\, \nabla_{\theta} \ell_v + (\text{patched-class residual terms}),$$
where $g_{\text{bg}}$ is the free LM-head background term. $g_{\text{total}}$ is **not** confined to any readout subspace; only $g_R$ is.

**Primary metric — readout-dominance ratio $\rho_c$ (fixes M-6, M-7, M-11):** for condition $c$ and class $c \in \{\text{POS}, \text{ABS}\}$,
$$\rho_c^{(x)} = \frac{\|\, g_{R}^{(x)}\,\|_2}{\|\, g_{\text{total}}^{(x)}\,\|_2}, \qquad \rho_c = \operatorname*{mean}_{t \in \{4,6,14,15\},\; x \in \{\text{POS},\text{ABS}\}} \rho_c^{(x)}.$$
$\rho_c \in [0,1]$ measures the fraction of the decision gradient attributable to the readout path (exact from the decomposition, no approximation). It is the operational statistic for H3 and P1.

**Auxiliary metric — gradient cosine $\cos(H1)$ (6E.18-compatible):** the raw cosine between the mean POS and mean ABS **total** decision gradients, without label-sign inversion (§10.5). It is background-dominated and condition-dependent; it is reported for cross-phase comparability and used by H1/H4/H5/H6. Its sign/level is *not* interpreted as evidence of subspace confinement.

**Note on the readout-only cosine:** the raw cosine between the paired class-conditional readout-only gradients is the sign identity $\operatorname{sign}(\alpha_{\text{POS}}\alpha_{\text{ABS}}) \in \{-1,+1\}$; it carries no graded information and is not used as a metric. This is why v3 does not define $\cos(H1)$ on $g_R$ alone (fixes M-17's open definition while honoring E-11's re-specification).

---

## 6. Registration Gates (Prospective, Deterministic, Non-Negotiable)

If any gate fails, the affected condition is declared **VOID** and must be resubmitted for human review. Silent re-derivation with altered inputs is prohibited. **Scope of VOID (fixes m-5):** failure of G1–G5, G7, G8 for a *single* condition VOIDs that condition (and any derived hypotheses referencing it); failure of G6 (sign-assertion) VOIDs the entire run (inherited Phase 6E.18 semantics).

| Gate | Requirement | Rationale |
|---|---|---|
| **G1** | `u_hat_LDA_hash == ff3cab49…` (exact, §3.2) **and** reconstructed $\hat{u}_{\text{LDA}}$ from the frozen panel matches `u_lda.npy` (cosine $\ge 0.9999$, elementwise max-abs $\le 10^{-6}$) | $\hat{u}_{\text{LDA}}$ reproduces the Phase 6E.18 object and is verifiable against a stored tensor, not only a hash (fixes M-8) |
| **G2** | `basis_sha256(U_8) == b8dfbd6a…` (exact, §3.4) **and** reconstructed $U_8$ (canonical sign, §3.4) matches `U_8.npy` (max-abs $\le 10^{-6}$) | $U_8$ is the Phase 6E.18 T2 basis and is reproducible byte-for-byte under the pinned convention (fixes M-8) |
| **G3** | $\kappa_k \in [0.05,\, 0.995]$ for $k \in \{2,4,8\}$ | $\hat{u}_{\text{LDA}} \notin S_k$ (non-degeneracy vs F1) and $\hat{u}_{\text{LDA}} \not\perp S_k$ (non-degenerate signal). **VOID contingency (fixes M-9):** $\kappa_8 > 0.995 \Rightarrow$ S8 VOID (E-5); $\kappa_4 \notin [0.05, 0.995] \Rightarrow$ S4 VOID; $\kappa_2 \notin [0.05, 0.995] \Rightarrow$ S2 VOID. Each VOID requires human resubmission; no endpoint is silently dropped |
| **G4** | $\|(U_4U_4^{\mathsf{T}} - U_2U_2^{\mathsf{T}})\hat{u}_{\text{LDA}}\|_2 \ge 0.01$; $\|(U_8U_8^{\mathsf{T}} - U_4U_4^{\mathsf{T}})\hat{u}_{\text{LDA}}\|_2 \ge 0.01$; **and** cosine between reconstructed and registered $w_{Sk}$ $\ge 0.9999$ (fixes m-2) | ensures S2, S4, S8 are pairwise distinct in readout direction and the stored weights are reproducible |
| **G5** | $\kappa_R \in [0.05,\, 0.30]$ | C4-Rand is weakly aligned (control), not near-F1 or null |
| **G6** | $\cos(H1)$ sign-assertion gate (synthetic vectors, $\varepsilon = 0.001$) passes ($u_1 = [1,0]^{\mathsf{T}}, u_2 = [-1,0]^{\mathsf{T}} \Rightarrow -1$; $v_1 = [1,0]^{\mathsf{T}}, v_2 = [1,0]^{\mathsf{T}} \Rightarrow +1$) | inherited Phase 6E.18 §6.2; failure VOIDs the run |
| **G7** | Confinement residual (fixes B-4): $\|(I - U_kU_k^{\mathsf{T}})\,(\alpha_{c} w_{Sk})\|_2 \le 10^{-6}\,\|\alpha_{c} w_{Sk}\|_2$ for $k \in \{2,4,8\}$, $c \in \{\text{POS},\text{ABS}\}$; and identity check $g_R^{\theta} = \alpha_{c}\nabla_{\theta}s$ against the full `dec_loss` backward (relative error $\le 10^{-4}$) | verifies the readout-only decision gradient is exactly confined to $\text{span}(U_k)$ and that the decomposition of §5.7 holds numerically |
| **G8** | Degenerate-guard (fixes i-3): $s_{\text{scale}}$ finite and $> 0$; all $\kappa$ finite and in $(0,1)$; all readout weights finite with $\|w_c\|_2 = s_{\text{scale}}$ within $10^{-6}$ relative error (six scalar conditions) | rejects degenerate scaling or weight construction before any execution |

Reference 6E.18 values (informational, not thresholds): $s_{\text{scale}} = 0.06681545$, $b_{F1} = -0.07310398$; cumulative energy of $U_8 = 0.9277434349060059$.

---

## 7. Formal Hypotheses (Frozen)

All hypotheses reference the metrics of §5.7 and the acceptance criteria of §8. Aggregation set for gradient metrics: $\{4, 6, 14, 15\}$ (inherited).

- **H1 (Subspace Gradient Compatibility, Primary):** Both S4 and S8 achieve $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1) \ge +0.80$.
- **H2 (Subspace Utility Preservation, Primary):** Both S4 and S8 achieve $d(15) \ge +1.50$ and balanced accuracy $\ge 70\%$ at $t = 15$.
- **H3 (Readout Steering, fixes M-6/M-7):** The readout-dominance ratio $\rho_c$ (§5.7) satisfies $\rho_{S8} \ge 0.50$ and is non-decreasing in $k$: $\rho_{S8} \ge \rho_{S4} \ge \rho_{S2}$. Interpretation: for the fixed subspace readout to "steer" the decision gradient, the readout contribution must be a substantial fraction of the total decision gradient, and increasing subspace dimension must increase that fraction. No claim of "noise filtering" of the total gradient is made.
- **H4 (Aligned vs Random Subspace, fixes M-15):** S4 exceeds C4-Rand on $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1)$ by $\ge +0.30$, **and** C4-Rand fails P3 ($d(15) < +0.80$). Alignment, not mere subspace restriction, drives both objectives.
- **H5 (Random-1D Alignment Specificity, fixes M-11):** R1-Rand fails P3 ($d(15) < +0.80$) while F1 passes P3. Direction alignment to $\hat{u}_{\text{LDA}}$ (F1) or to $U_k$ (S$k$) — not 1D fixedness per se — carries the utility. R1-Rand's $\cos(H1)$ level is **not** hypothesized: with a random direction the readout contribution is small ($\rho_{R1}$ expected $\le 0.30$) and the measured cosine is background-dominated and sign-indeterminate; R1-Rand is therefore **not** a null model for $\cos(H1)$.
- **H6 (T2 Confound Resolution, joint-effect framing, fixes M-10):** S8 achieves $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1) \ge +0.80$ **while** $d(15) \ge +1.50$. **Explicit joint-effect caveat:** S8 differs from T2 in both trainable capacity (0 vs $k+1$) and normalization (gain-matched fixed vs learned); a difference is a joint effect of both. C4-Rand and R1-Rand — fixed, zero-trainable-capacity, non-aligned — anchor the interpretation: if the joint effect exceeds both random controls, capacity removal + gain normalization plus alignment are jointly implicated.
- **H7 (Anti-Collapse / Anti-Drift, fixes m-9/m-10):** Every scalar condition maintains $P(t) > 0.0\%$ for all $t \le 20$ and $d(20) \ge 0.00$ (the inherited anti-degradation guard; the B0 collapse guard triggers at $K = 3$ consecutive zero-proposal steps, §10.7).

---

## 8. Dual-Objective Acceptance Criteria (P1–P6)

All thresholds are **absolute constants** justified independently below; **none is a function of any Phase 6E.18/6E.19 observed value** (fixes M-1, M-12). Aggregation set is $\{4, 6, 14, 15\}$ (inherited).

| ID | Criterion | Metric | Threshold | Independent Justification |
|---|---|---|---|---|
| **P1** | Readout Steering — null tier (fixes M-11) | $\rho_c$ (§5.7) | $\ge 0.50$ for S4 and S8 | The subspace hypothesis requires the readout contribution to dominate the decision gradient; $\rho \ge 0.50$ is a round, substantial "more than half" constant. The random-1D control R1-Rand is expected $\le 0.30$; that contrast is reported, not gating |
| **P2** | Gradient Compatibility — RQ target | $\text{mean }\cos(H1)$ | $\ge +0.80$ | The research question's own target (§1.2) |
| **P3** | Margin Maintenance — large effect | Cohen's $d(15)$ | $\ge +0.80$ | Cohen's conventional "large" effect-size constant |
| **P4** | Margin Strength — RQ target | Cohen's $d(15)$ | $\ge +1.50$ | The research question's margin target (§1.2) |
| **P5** | Balanced Accuracy (fixes M-13) | $(a_{\text{POS}} + a_{\text{ABS}})/2$ at $t=15$ | $\ge 70\%$ | Chance $= 50\%$; $+20$ pp is a round, substantial constant margin. **LOO-CV comparator (non-gating):** the readout is re-fit on 15 of the 16 panels (rolling over all 16 splits: fit $u_{\text{LDA}}$ and $b_c$ on 15, evaluate balanced accuracy on the held-out 16th); the resulting mean $\pm$ SD is reported alongside the primary full-panel value as a generalization check and is explicitly not part of the PASS/FAIL decision |
| **P6** | Anti-Collapse / Anti-Drift | $P(t)$; $d(20)$ | $P(t) > 0.0\%\ \forall t \le 20$; $d(20) \ge 0.00$ | inherited Phase 6E.18 anti-degradation guard |

**JOINT SUCCESS GATE (the RQ demands both objectives simultaneously):** a condition is **SUCCESS** iff $P2 \wedge P4 \wedge P5 \wedge P6$ (fixes M-14: P6 is required, and is included in the §9.1 matrix).
**PARTIAL** iff $P1 \wedge P3$ (and not SUCCESS). Otherwise **FAIL**. Metrics follow the Phase 6E.18 §6.1 normative definitions, with $d(t)$ and balanced accuracy computed over the $N = 16$ panel at checkpoint $t$.

---

## 9. Pre-Registered Outcome Interpretation Matrix

### 9.1 Primary Outcome Matrix (GC × TU), with PARTIAL tier (fixes M-14, m-8)
- **GC** (gradient compatibility) $=$ P2: $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1) \ge +0.80$.
- **GC-PARTIAL** $=$ $P1 \wedge \neg P2$: readout steering confirmed ($\rho \ge 0.50$) but alignment below the RQ target.
- **TU** (task utility) $=$ P4 $\wedge$ P5: $d(15) \ge +1.50$ and balanced accuracy $\ge 70\%$.
- **SUCCESS** additionally requires **P6** (anti-collapse); a condition passing GC $\wedge$ TU but failing P6 is **GC-only/TU-only per row** and its result is annotated "P6 FAILED — not SUCCESS".

| | **TU high** ($d(15) \ge +1.50$, bal. acc. $\ge 70\%$) | **TU low** (otherwise) |
|---|---|---|
| **GC high** ($\cos(H1) \ge +0.80$) | **SUCCESS** (if P6) — H1 $\wedge$ H2 supported. The fixed subspace readout resolves the 6E.18 decoupling: T2-grade alignment with F1-grade utility, with zero trainable capacity. | **GC-only** — decoupling persists within subspace geometry: T2-like alignment without utility. H2 rejected; subspace readout discards discriminative variance. |
| **GC-PARTIAL** ($P1 \wedge \neg P2$) | **PARTIAL** (if P3, i.e., $d(15) \ge +0.80$) — readout steers the gradient but alignment stays below the RQ target; H3 supported, H1 rejected, H2 supported. | **PARTIAL** only if P3 holds ($d(15) \ge +0.80$ but below $+1.50$ or bal. acc. $< 70\%$): steering without full utility, H3 supported, H1 $\wedge$ H2 rejected. Otherwise **FAIL**. |
| **GC low** ($\neg P1$) | **TU-only** — no readout steering; replicates the F1 utility pattern. H3 rejected; H1 rejected; F1 remains the utility-optimal fixed readout. | **Both fail** — fixed subspace readout is inferior on both axes. H1 $\wedge$ H2 $\wedge$ H3 rejected. |

### 9.2 Secondary Rows (fixes M-15)
- **S4 exceeds C4-Rand on $\text{mean}_{t \in \{4,6,14,15\}}\cos(H1)$ by $\ge +0.30$ and C4-Rand fails P3** ($d(15) < +0.80$): **H4 supported** — alignment, not mere subspace restriction, drives both objectives.
- **R1-Rand fails P3 while F1 passes P3:** **H5 supported** — direction alignment carries the utility; the random-1D control separates alignment from fixedness. (R1-Rand's $\cos(H1)$ is reported but not interpreted as a null — fixes M-11.)
- **S8 reaches $\cos(H1) \ge +0.80$ with $d(15) \ge +1.50$:** **H6 supported** — the decoupling is resolved without trainable capacity; the result is interpreted jointly with C4-Rand/R1-Rand (fixes M-10).
- **S8 $\rho_{S8} \ge \rho_{S4} \ge \rho_{S2}$ and $\rho_{S8} \ge 0.50$:** **H3 supported** — subspace dimension increases readout steering (fixes M-6/M-7).
- **B0 guard triggers** (collapse detection, §10.7): B0 truncates; all other conditions continue on the full horizon; the B0 reference is the §10.7 step-0 anchor.
- **Any protocol deviation in any condition:** that run is **VOID**; no verdict is generated from it.

---

## 10. Operational Discipline and Inherited Phase 6E.18 Rules

### 10.1 Zero Readout Parameters
Runtime assertion: `trainable_readout_params == 0` for all seven conditions (verified at every checkpoint).

### 10.2 Protocol Hash Binding
The execution script MUST bind and verify this document's SHA-256 hash before any checkpoint evaluation.

### 10.3 Data and Training Configuration (verbatim inheritance, fully enumerated — fixes M-16; E-10)
- Data: THEO SLM v0; train/dev split with `seed = 42` — identical to Phase 6E.18.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, $d = 896$, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- LoRA: $r = 16$, $\alpha = 32$, `dropout = 0.05`, `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (7 modules), `bias = "none"`, `task_type = "CAUSAL_LM"` — backbone LoRA only; **no readout LoRA, no target-module changes** (verbatim Phase 6E.18 runner lines 92–93, 1149–1150).
- Optimizer: AdamW $\text{lr} = 10^{-4}$, weight decay $0.01$, gradient clip $1.0$; `grad_accum = 2`, `batch = 4`; $\lambda = 10$; horizon $T = 20$ steps.
- Decision position: the prompt is `format_prompt(...) + '{"decision": "'`; the decision logits are taken at $d = v_{\text{idx}}[4]$ (the 5th valid token), `dec_logits[PRO_ID] = s`, `dec_logits[AB_ID] = -s`, with $\text{PRO\_ID} = 9117$, $\text{AB\_ID} = 1867$ (verbatim Phase 6E.18 runner lines 73–74, 265, 982, 1006–1027).
- Batch RNG (fixes M-16): per-epoch batch sampling and ordering use a single deterministic generator (inherited Phase 6E.18 RNG rule seeded from `SEED_DATA = 42`), byte-identical across all seven conditions within the run. The readout is the ONLY difference between conditions; every other training input is identical across conditions.

### 10.4 Checkpoints and Aggregation
Canonical evaluation checkpoint set: $\{0, 4, 6, 7, 14, 15, 17, 20\}$. Decision-rule aggregation set: $\{4, 6, 14, 15\}$.

### 10.5 Gradient Cosine $\cos(H1)$ — Normative Binding (fixes M-17, B-4)
Two gradient objects are computed at each aggregation checkpoint over the $N = 16$ panel (8 POS, 8 ABS), using the exact Phase 6E.18 `per_sample_decision_gradient` computation (runner lines 1006–1033):
- **Total decision gradient** $g_{\text{total}} = \texttt{torch.autograd.grad}(\texttt{dec\_loss},\; \texttt{lora\_param\_list})$, mean per class: $\bar{g}_{\text{POS}}$, $\bar{g}_{\text{ABS}}$.
- **Readout-only decision gradient** $g_R = \alpha_c\, \nabla_{\theta} s$ (§5.7), used for the confinement residual (G7) and the dominance ratio $\rho_c$.

$\cos(H1)$ is the raw cosine between $\bar{g}_{\text{POS}}$ and $\bar{g}_{\text{ABS}}$, **without label-sign inversion** (6E.18-compatible):
$$\cos(H1) = \frac{\langle \bar{g}_{\text{POS}}, \bar{g}_{\text{ABS}} \rangle}{\|\bar{g}_{\text{POS}}\|_2\,\|\bar{g}_{\text{ABS}}\|_2}$$

**Interpretation binding:** $\cos(H1)$ is measured on the total gradient and is **background-dominated and condition-dependent**; it is **not** a subspace-confinement statistic. Subspace confinement is tested by G7 (readout-only residual) and by P1/H3 ($\rho_c$). Inherited §6.2 pre-verdict gate: the synthetic sign-assertion test ($u_1 = [1,0]^{\mathsf{T}}, u_2 = [-1,0]^{\mathsf{T}} \Rightarrow -1$; $v_1 = [1,0]^{\mathsf{T}}, v_2 = [1,0]^{\mathsf{T}} \Rightarrow +1$; tolerance $\varepsilon = 0.001$) MUST pass before any verdict; failure $\Rightarrow$ run VOID.

### 10.6 Preflight Mode (Required, Non-Training)
A non-training `--preflight` mode MUST verify, prior to any execution: reconstruction of all bases and axes from the frozen panel against stored tensors and hashes (G1–G2, §3.2/§3.4), the canonical sign convention (§3.4), all $\kappa$ gates and degeneracy guards (G3–G5, G8), confinement residual and decomposition identity (G7), parameter counts, pairwise-distinctness checks (§5.6), sign-assertion gate (G6), and the metric arithmetic of §5.7/§10.5 including the $\alpha$ coefficients and $\rho_c$ on synthetic vectors.

### 10.7 Baseline Collapse Guard (B0 only; fixes m-9/m-10)
The B0 collapse guard triggers when the detector observes $K = 3$ **consecutive zero-proposal steps** (inherited criterion; observed $t^{*} = 4$ in Phase 6E.18 — reported as context, not a rule). When triggered, B0 truncates at that point and no other condition truncates. When a matched B0 comparator is unavailable, the frozen step-0 anchor applies and the missing comparator is recorded `UNAVAILABLE` (inherited Phase 6E.18 §7.1/§10). No trajectory back-filling and no cross-phase synthetic concatenation.

### 10.8 Void and Deviation Rules
Non-finite loss, a failed assertion gate, or a missing rule-mandated checkpoint in any condition $\Rightarrow$ that run is VOID; no verdict is generated from it. Gate failures VOID their condition (or the run, for G6) per §6 without silent re-derivation. Phase 6E.18 execution logs and the Phase 6E.19 adjudication record remain permanently immutable.

### 10.9 Verdict Double Gate and Refusal Triplet
Scientific evaluation alone does NOT produce a verdict. Any attempt to issue a verdict without the technical evaluation AND human authorization returns the exact refusal triplet:
`PRE_EXECUTION` / `NO_SCIENTIFIC_VERDICT` / `EXECUTION_NOT_AUTHORIZED`.

### 10.10 Provenance and Authorization Artifacts
- `sha256_manifest.json` (hashes of code, base model, corpus, checkpoints, all §3 objects).
- Reference tensors (fixes M-8, m-4): `u_lda.npy`, `U_8.npy`, `R_4.npy`, `n_rand.npy`, `w_<cond>.npy` (7), `H_16.npy`, `y16` label vector — each with its SHA-256.
- `runs.json` (step-by-step telemetry), `pre_registration_snapshot.json` (this document's hash + adopted-decision log).
- P5 LOO-CV report (fixes M-13): `p5_loo_report.json` (16 rolling splits, mean ± SD, non-gating).
- `execution_authorization.json` schema: `{"decision": "APPROVE", "signature": ..., "date": ..., "phase": "6E.20"}`. Absent this file, no execution may occur. (The Phase 6E.18 `execution_authorization.json` carries `phase = "6E.18"` and does **not** authorize Phase 6E.20 — info note i-2.)

---

## 11. Hard Stop and Human Authorization Gate

```
================================================================================
HARD STOP — NON-NEGOTIABLE
================================================================================
This document is a DESIGN-ONLY specification and does NOT authorize execution.
No model execution, training, fine-tuning, optimizer steps, or checkpoint
generation may occur without: (1) formal pre-execution audit of this finalized
draft, and (2) explicit signed human authorization
(execution_authorization.json, decision=APPROVE, phase=6E.20).

Decisions E-1..E-11 have been adopted by explicit human decision (2026-08-16),
so the earlier "provisional decisions" prerequisite is satisfied; the remaining
two prerequisites (pre-execution audit, explicit authorization) are still
mandatory and have NOT occurred.

Sign-off Gate:
  Reviewer: ______________________________
  Decision: [ ] APPROVE   [ ] AMEND   [ ] REJECT
  Date:     ______________________________
  Signature: ______________________________
================================================================================
```

---

## 12. Adopted Design Decisions (E-1 … E-11)

**ADOPTED** by explicit human decision dated 2026-08-16, following the v2.0 artifact-level audit. Each entry records the adopted resolution and its adoption basis. Deviation from any adopted decision requires a documented amendment and re-adoption.

| ID | Decision | Adopted Resolution | Adoption Basis |
|---|---|---|---|
| **E-1** | Readout normalization convention | Unit direction, common gain: $\|w\|_2 = s_{\text{scale}}$ for the six scalar readouts ($w_{Sk} = (s_{\text{scale}}/\kappa_k)U_kU_k^{\mathsf{T}}\hat{u}_{\text{LDA}}$); B0 excluded | Forced by the direction-vs-gain isolation goal; audit-verified |
| **E-2** | C4-Rand definition | $w_{C4} = (s_{\text{scale}}/\kappa_R)R_4R_4^{\mathsf{T}}\hat{u}_{\text{LDA}}$ — 4D random-subspace readout, same functional form as S4 | Forced by B-3 resolution |
| **E-3** | R1-Rand inclusion | **Include** as the 7th condition: matched 1D random control for F1 and the steering contrast for P1/H3/H5 | Human choice; adopted 2026-08-16 |
| **E-4** | $\kappa$-gate bands | $\kappa_k \in [0.05, 0.995]$; inter-$k$ increments $\ge 0.01$; $\kappa_R \in [0.05, 0.30]$; with S2/S4/S8 VOID contingency (§6 G3) | Human choice; adopted 2026-08-16. Arbitrariness accepted as a design property; bands are frozen and no endpoint is silently dropped |
| **E-5** | S8 near-F1 VOID | $\kappa_8 > 0.995 \Rightarrow$ S8 VOID (subsumed by G3). Residual $\|(I - U_8U_8^{\mathsf{T}})\hat{u}_{\text{LDA}}\| \ge \sqrt{1 - 0.995^2} \approx 0.0999$ guarantees direction separation | Forced by B-1; math audit-verified |
| **E-6** | Step-0 panel hash binding | Bind $\hat{u}_{\text{LDA}}$ via `u_hat_LDA_hash` (G1), $U_8$ via `basis_sha256` (G2), and record `H_16` panel + label-vector hashes and full reference tensors | Provenance practice; audit-verified as supported |
| **E-7** | Bias convention | $b_c = -\text{median}(w_c @ H_{16[:8]})$ for all scalar conditions (generalizes Phase 6E.18 F1 rule) | Literally inherited from Phase 6E.18; audit-verified |
| **E-8** | Random-seed value | `20260820` for $R_4$ and $n_{\text{rand}}$ via explicit `torch.Generator`; tensors stored with hashes | Human choice; adopted 2026-08-16 |
| **E-9** | P5 checkpoint | Balanced accuracy at $t = 15$ (matches the $d(15)$ evaluation point), with non-gating LOO-CV comparator | Human choice; adopted 2026-08-16 |
| **E-10** | LoRA/training inheritance | Verbatim Phase 6E.18 training configuration, fully enumerated (§10.3) | Forced by cross-phase comparability; audit-verified |
| **E-11** | B-4 metric re-specification (new) | Primary gradient object is the readout-only decision gradient $g_R = \alpha_c \nabla_\theta s$ (§5.7); confinement is a registration invariant (G7); $\cos(H1)$ remains the 6E.18-compatible total-gradient metric, explicitly background-dominated (§10.5) | Human choice; adopted 2026-08-16, resolving B-4 |

---

## 13. Amendment and Reconciliation Log (v1.0 → v2.0 → v3.0)

### v1.0 → v2.0 (retained)
- **B-1 (BLOCKING) — resolved structurally:** $S_k$ redefined as $U_k$ (top-$k$ principal subspace of $\Sigma_{\text{diff}}$, §3.4); $\hat{u}_{\text{LDA}} \notin S_k$ enforced by G3; readout is a representation projection (§5.3). The v1 identity $P_{Sk}\hat{u}_{\text{LDA}} = \hat{u}_{\text{LDA}}$ (verified $\approx 2\times10^{-16}$) can no longer arise.
- **B-2 (BLOCKING) — resolved:** P2 is the absolute RQ target $\ge +0.80$; the post-hoc `mean(F1) + 0.10` criterion and the unpassable joint gate are deleted. Thresholds are absolute constants (§8, fixes M-1).
- **B-3 (BLOCKING) — resolved:** C4-Rand now uses $R_4$ in its mathematics with a defined seed and bias; $\kappa_R$ gate (G5) guards non-degeneracy.
- **M-2 … M-5 (MAJOR) — resolved:** JOINT SUCCESS GATE matches the RQ (§8); complete operational section (§10); T2-basis provenance pinned (G2/G3/G5); basis hashing registered (G1/G2/E-6).

### v2.0 → v3.0 (this revision; maps the v2 artifact-level audit register)
- **B-4 (BLOCKING) — resolved by re-specification (E-11):** the total decision gradient is shown (chain rule from the literal runner) to be $g_{\text{bg}} + \alpha_c w_c$; only the readout contribution is subspace-confined. §5.1/§5.3/§10.5 restated; confinement is now an exact, preflight-verified invariant (G7); $\cos(H1)$ is re-bound as the background-dominated 6E.18-compatible metric (§10.5); the false total-gradient confinement claim is withdrawn (§5.3).
- **M-6 (MAJOR) — resolved:** H3 rewritten as the readout-steering hypothesis with the exact dominance ratio $\rho_c$ (§5.7, §7 H3, §8 P1).
- **M-7 (MAJOR) — resolved:** the family is honestly characterized as seven fixed 1D linear readouts differing only in weight direction (§2.2, §4); no "noise filtering" mechanism is claimed.
- **M-8 (MAJOR) — resolved:** canonical SVD sign convention + canonical float64 hashing (§3.4); full reference tensors stored and hashed (§3.2/§3.4/§3.7, §10.10); G1/G2 verify reconstruction (§6).
- **M-9 (MAJOR) — resolved:** E-set adopted and gates frozen (§12); S2/S4/S8 VOID contingency added to G3 (§6).
- **M-10 (MAJOR) — resolved:** H6 restated as a joint-effect hypothesis with explicit caveat and random fixed-gradient controls (§7 H6, §9.2).
- **M-11 (MAJOR) — resolved:** P1 re-specified as the readout-steering dominance test; the unsound "random $\Rightarrow \cos \approx 0$" null is deleted (§8 P1, §7 H5).
- **M-12 (MAJOR) — resolved:** all thresholds are absolute constants with independent justifications; no observed value appears in any criterion or justification (§8); procedural post-hoc claims removed.
- **M-13 (MAJOR) — resolved:** P5 keeps the $t=15$ full-panel gate and adds a non-gating LOO-CV comparator (§8 P5, §10.10).
- **M-14 (MAJOR) — resolved:** SUCCESS explicitly requires P6 and P6 appears in the outcome matrix (§8, §9.1).
- **M-15 (MAJOR) — resolved:** H4 and §9.2 aligned on the single P3 threshold $d(15) < +0.80$ (§7 H4, §9.2).
- **M-16 (MAJOR) — resolved:** §10.3 fully enumerates LoRA dropout/targets, decision-position prompt convention, decision index, PRO/AB IDs, and batch RNG (§10.3).
- **M-17 (MAJOR) — resolved:** the gradient object for $\cos(H1)$ is defined by the dual-object spec (§5.7, §10.5).
- **m-1 … m-13 (MINOR) — resolved:** B0 excluded from common-gain (§4, §3.6); G4 normalized-direction bound (§6); $y_{16}$ pinned (§3.1); $R_4$/$n_{\text{rand}}$ hashes (§3.7); G6 vs condition-VOID semantics (§6, §10.8); H3 label corrected (§7 H3); $d(20)$ guard literal (§8 P6, §10.7); PARTIAL tier added to the outcome matrix (§9.1); H6 row tightened (§7 H6, §9.2); collapse trigger reworded to $K=3$ consecutive zero-proposal steps (§10.7); dangling §6.2 reference removed (§6 G6, §10.5); "registered from t2_basis.json" replaced by reconstruction verification (§3.4, §6 G2); P5 vs P6 scope clarified (§8 P5/P6).
- **i-1 … i-4 (INFO) — incorporated:** baseline values re-verified (forensic record) cited as context only (§1.1); 6E.18 auth file does not authorize 6E.20 (§10.10); $s_{\text{scale}}$ degenerate guard added (§3.3, §6 G8); joint gate $P2 \wedge P4 \wedge P5 \wedge P6$ retained (§8).

---

*End of document — DRAFT v3.0. Decisions E-1 … E-11 ADOPTED (human, 2026-08-16). Execution remains HARD-STOPPED pending pre-execution audit and explicit human authorization.*
