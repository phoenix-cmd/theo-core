# THEO SLM 6E.18 — v2.1.3 Amendment Specification (VERIFIED)

- **Status:** SPECIFICATION ONLY. Does not authorize execution. Does not modify any protocol artifact.
- **Date:** 2026-08-16.
- **Frozen audited artifact:** `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.2.md` — **REJECTED FOR EXECUTION — PRESERVED FORENSIC ARTIFACT.** Not to be edited.
  - SHA-256: `D13576BA019350266CC4551361CA07C0B20FE32CF462D768751461772B97EAF6`
  - Size / lines: 16,747 bytes / 243 lines. Last write: 2026-08-16 13:14:34.
- **Method:** surgical amendments only; every change maps to an exact section of the frozen v2.1.2 and states a before/after requirement. Every mathematical choice is labeled **FORMALIZATION** (existing runner/protocol behavior being made explicit) or **CHANGE** (a genuine protocol/experimental choice introduced in v2.1.3). No casual restructuring.
- **Controlled sequence:** v2.1.2 frozen → this specification → v2.1.3 (single implementation pass) → protocol/runner conformance check → independent forensic audit → authorization decision.

---

## 0. Pre-Implementation Conformance Verification Record

Sources: 6E.17 runner snapshot `phase-6e17-causal-intervention/code_snapshot/run_phase_6e17_causal_intervention.py` (1160 lines); canonical `readout_initialization.json`; 6E.18 runner draft `theo-core/scripts/dataset_generator/run_phase_6e18_causal_intervention.py` (514 lines, preflight-only, bound to v2.1.2 hash `d13576ba…`).

| # | Proposed choice | Verification result | Ruling |
|---|---|---|---|
| 1 | `s_scale = ‖w_δ‖₂` | **REFUTED.** Runner derives `s_scale = np.std(margins_b) / (np.std(H16 @ u_lda) + 1e-12)` (6E.17 L356); canonical value `s_scale = 0.06683328` vs `w_delta_norm = 0.564183` (json). ‖w_δ‖₂ is not the scale. | Freeze the runner formula (FORMALIZATION). |
| 2 | V1 aligned-mirror init | **CHANGE.** 6E.18 `V1ReadoutHead` takes `w_pro_0, w_abs_0` as args; preflight passes zero dummies (6E.18 L454). No real V1 init construction exists anywhere. | Genuine protocol choice; requires runner implementation. Label CHANGE (user conditionally approved). |
| 3 | T2 `β(0) = s_scale·U_k^T û_LDA` | **CONTRADICTS existing runner.** 6E.18 `T2ReadoutHead` defaults `beta0 = np.zeros(k)` (L203–205). Projection init is a CHANGE, not a formalization. | Recommend formalizing `β(0) = 0` (FORMALIZATION). Projection init available as explicit CHANGE if rank isolation is prioritized. |
| 4 | `Σ_diff` = pooled-within-class | **REFUTED.** 6E.18 L330–331 computes the **total sample covariance** (global-mean centered, `/(N-1)`), matching §8 L177 literal. | Formalize total covariance (FORMALIZATION); reconcile §3.6 wording. No math change. |
| 5 | `r_eff(15)` = participation ratio of decision-gradient matrix | **PARTIALLY REFUTED.** Formula IS participation ratio `tr²/tr²` (6E.17 L121–125), but the object is the **margin-Jacobian Gram** `G_margin` at t=15 (N×N over panel, margin gradients w.r.t. LoRA params; 6E.17 L584–591, L787, L929). | Correct the object to margin-Jacobian Gram (FORMALIZATION). |
| 6 | (additional) `b0_I1`, `b0_C2` | 6E.17 L358: `b0 = -median(a0 @ H16[:8])` → −0.07326254; L371: `b0_c2 = -median(c0 @ H16[:8])` → 0.17438322 (json). Protocol §3.2/§3.3 underspecify these. | Freeze both rules (FORMALIZATION). |
| 7 | (additional) `b_δ` | `b_δ = 0` confirmed: margin = `z_PRO − z_ABS` directly from logits (6E.17 L571); shared head bias cancels. | FORMALIZATION. |
| 8 | (additional) proposal rate | Dev-set based: `dev_proposal_rate` = fraction of dev items with margin > 0 (6E.17 L678, L733, L766, L925); P2 = all dev rates > 0 AND not collapsed. | Define as dev-set fraction (FORMALIZATION). |
| 9 | (additional) Cohen's d operands | Confirmed: `margin_cohen_d` = (mean POS margins − mean ABS margins) / pooled sd, pooled = sqrt((var_POS+var_ABS)/2), panel16 first 8 POS / last 8 ABS, at t=15 (6E.17 L619–622, L928). | FORMALIZATION (specify panel16, first-8/last-8 split). |
| 10 | (additional) Resub k-NN1 / LOO-CV | Confirmed: `KNeighborsClassifier(n_neighbors=1)` resub + LeaveOneOut on X = layer-24 last-token hidden state `h_L` (6E.17 L519–552). | FORMALIZATION. |
| 11 | (additional) P4 early-truncation fallback | 6E.18 `evaluate_p4_comparator` (L299–319) already falls back to the step-0 anchor whenever contemporaneous B0 is None — including t ∈ {4, 6}. | Formalize this behavior (FORMALIZATION). No runner change needed. |
| 12 | (additional) checkpoint set | 6E.18 `CHECKPOINTS = [0, 4, 6, 7, 14, 15, 17, 20]` (L68) — a superset of the rule sets; P1 uses {4,6,14,15}, P4 uses {0,4,6}/{14,15,20}. No conflict. | Conformance note; record the evaluation schedule in the manifest. |

**Open decisions requiring human confirmation:** (a) T2 β(0) = zeros (recommended) vs LDA projection; (b) b_fixed = 0 vs b0 (recommend b0, F1 is the frozen twin of I1).

---

## Amendment A — BLOCKING: Repair phantom §4.4 reference (REC-01)

**Section:** §13 amendment log, entry CANON-F05, line 231 of frozen v2.1.2. Also §4 heading, line 99.

**Before (frozen L231):** `Resolved CANON-F05: Explicitly codified dual logit-parameterization and 2× capacity confound for V1 vs I1 in §4.4.`

**After (v2.1.3):** `Resolved CANON-F05: Explicitly codified dual logit-parameterization and 2× capacity confound for V1 vs I1 in §4.1 (item 4).`

**Requirement A1:** Formally number the subsection heading at L99:
- Before: `### Confound Controls & Isolation Boundaries`
- After: `### 4.1 Confound Controls & Isolation Boundaries`
No renumbering of any other §4 element. Items 1–4 inside §4.1 remain numbered 1–4 (unmodified text).

**Requirement A2:** Do not rewrite the historical v2→v2.1 entry (L224). The REC-01 repair is recorded in the new v2.1.2→v2.1.3 log entry (Amendment H).

**Void:** Do not flatten §4 items into subsections §4.1–§4.4; that collides with the chosen `§4.1` numbering.

---

## Amendment B — MAJOR (F-03): Operationalize F1, T1, T2, V1 (and C2) as frozen executable rules

**Section:** §7 (title line 156) and new subsection §7.2 inserted after §7.1 (after line 170).

**Requirement B1:** Retitle §7:
- Before: `## 7. Frozen Decision Rules and Acceptance Criteria (P1–P5, C2x)`
- After: `## 7. Frozen Decision Rules and Acceptance Criteria (P1–P5, C2x, M1–M5)`

**Requirement B2:** Add `### 7.2 Secondary Mechanistic Probe Decision Rules (M1–M5)` with the following table. All thresholds reuse pre-registered values (−0.30 from P1; −0.60 from H3; Δ ≤ 0.10 from H4). Cos(H1) is defined by §6.2 (aggregate cosine of mean POS vs mean ABS gradients).

| ID | Probe | Metric | Checkpoints | Aggregation | PASS / SUPPORT | FAIL / NON-SUPPORT | AMBIGUOUS / INTERMEDIATE | UNAVAILABLE / TRUNCATED |
|---|---|---|---|---|---|---|---|---|
| **M1** | F1 / H3 | cos(H1)(F1) | {4, 6, 14, 15} | mean | mean ≤ −0.60 → H3 SUPPORTED | mean > −0.30 → H3 NOT SUPPORTED | −0.60 < mean ≤ −0.30 → H3 INDETERMINATE | any checkpoint missing → run VOID (§5.2 row 7, L126) |
| **M2** | T1 / 1D-Gain | cos(H1)(T1) | {4, 6, 14, 15} | mean | mean ≥ −0.30 → 1D gain SUFFICIENT | mean < −0.30 → 1D gain INSUFFICIENT | none (binary rule) | any checkpoint missing → run VOID |
| **M3** | T2 / H4 | Δ = \|mean cos(H1)(T2) − mean cos(H1)(I1)\| | {4, 6, 14, 15} | mean | Δ ≤ 0.10 AND mean cos(H1)(T2) ≥ −0.30 → MATCH | Δ > 0.10 OR mean cos(H1)(T2) < −0.30 → NO MATCH | not evaluable if I1 VOID → M3 NOT EVALUABLE | k ∉ [2, 8] → T2 VOID (§8 L180); missing checkpoint → VOID |
| **M4** | V1 / H5 | mean cos(H1)(V1) AND no policy degradation | {4, 6, 14, 15} | mean | cos ≥ −0.30 AND no degradation → SUPPORTED | cos < −0.30 OR degradation → NOT SUPPORTED | cos ≥ −0.30 with degradation → NOT SUPPORTED, degradation recorded | any checkpoint missing → run VOID |
| **M5** | C2 / H2 | cos(H1)(C2) | {4, 6, 14, 15} | mean | mean ≥ −0.30 → capacity sufficiency SUPPORTED | mean < −0.30 → NOT SUPPORTED | none (binary rule) | any checkpoint missing → run VOID |

**Requirement B3:** Define "policy degradation" in §6.1 (Amendment F, item 9): V1 degrades if ProposalRate(15) ≤ 0.0% OR d(15) < 0.00.

**Requirement B4:** P2 operands bound (FORMALIZATION, verified 6E.17 L678/733/766/925): proposal rate = dev-set fraction of items with margin > 0; P2 passes iff dev proposal rate > 0 at all evaluated t ≤ 20 AND condition not collapsed.

---

## Amendment C — MAJOR (F-02): Resolve the F1 threshold band and rule naming

**Sections:** §5.1 H3 (line 112); §5.2 outcome matrix (lines 118–126); §7.2 M1.

**Requirement C1 — band disposition:** Add an outcome-matrix row:
- After (new row): `| I1 passes P1 & C2x; F1 in indeterminate band (−0.60 < mean cos(F1) ≤ −0.30) [M1 INDETERMINATE] | H-Causal not decisively supported; F1 fixed-geometry insufficiency neither established nor refuted. |`
- Append a one-line rationale to H3 in §5.1: −0.60 calibrated against 6E.17 anchored B0 conflict (−0.933/−0.997; see companion post-execution report).

**Requirement C2 — rule naming:** Reword outcome-matrix rows to M-rule IDs (no non-I1 condition SHALL be described as passing/failing P1):
- Row 2 (L121): `| I1 passes P1; C2 meets M5 floor; C2x fails | Capacity/trainability explanation strengthened; alignment claim rejected. |`
- Row 3 (L122): `| I1 fails P1; C2 fails M5 floor; F1 fails M1 floor (mean cos < −0.30) | Readout intervention ineffective; H0 survives. |`
- Row 4 (L123): `| M3 SUPPORTED; M2 NOT SUPPORTED | Low-rank adapted readout matches aligned full-rank readout; 1D gain insufficient (rank, basis, and init remain confounded). |`
- Row 5 (L124): `| M4 SUPPORTED | Unconstrained 2-logit parameterization and expanded capacity provide independent advantage. |`

---

## Amendment D — MAJOR (F-04): Limit the T2 causal claim (Option 1 — acknowledge, do not redesign)

**Sections:** §4.1 confound item 3 (line 102); §5.1 H4 (line 113); §5.2 row 4 (line 123).

**Requirement D1 — §4.1 item 3 after:**
`3. **Subspace Constraint (T2 vs I1):** T2 vs I1 differs simultaneously in (a) rank (k vs 896), (b) basis construction (top-k SVD of the total step-0 panel covariance, §8, vs the Fisher LDA axis û_LDA), and (c) trainable-vector initialization (β(0) = 0 vs a(0) = s_scale·û_LDA). M3/H4 evaluates a composite low-rank-adapted-readout contrast; no rank-only causal claim may be drawn from M3 alone.`

**Requirement D2 — H4 (line 113) after:**
`- **H4 (Subspace-Readout Match, composite):** T2 matches I1 within Δ ≤ 0.10 on cos(H1) under the acknowledged rank × basis × initialization composite contrast (M3). Support for H4 indicates a low-rank adapted readout (principal-subspace SVD basis, zero β init) suffices; it does not isolate rank from basis construction or initialization.`

**Requirement D3 — §5.2 row 4 inference:** per Amendment C row 4 text (rank, basis, and init remain confounded).

**Requirement D4:** Add one sentence to §8 (after line 180): "The T2 basis and β(0) initialization are frozen before T2 execution; k and the basis hash are recorded in the run manifest."

---

## Amendment E — MAJOR (F-05): Freeze all initializations and shared constants

**Sections:** new subsection `### 3.8 Shared Initialization Constants` after §3.7; per-condition init bullets (§3.2, §3.3, §3.4, §3.5, §3.6, §3.7); §4 table V1 cell (line 97).

**Requirement E1 — §3.8 (new):**
`### 3.8 Shared Initialization Constants (all FORMALIZED from the 6E.17/6E.18 canonical derivation)`
`- **s_scale** = std(margins_B0) / std(H16 @ û_LDA), computed at step 0 on the frozen N=16 panel (deterministic; no RNG); recorded in the run manifest. Reference value from 6E.17: 0.06683328.`
`- **b_δ** = 0 (margin = z_PRO − z_ABS; shared head bias cancels).`
`- **b(0) for I1** = −median(s_scale·û_LDA @ H16[:8]) (reference 6E.17 value: −0.07326254).`
`- **b''(0) for C2** = −median(c0 @ H16[:8]) where c0 = s_scale·n (reference 6E.17 value: 0.17438322).`
`- **b_fixed** = b(0) for I1 (F1 is the frozen twin of I1; decision-confirmable).`

**Requirement E2 — §3.5 T1 after (append):**
`- Initialization: α(0) = 1.0, b(0) = 0.0 (FORMALIZATION — matches 6E.18 runner T1ReadoutHead defaults). Deterministic; recorded in manifest.`

**Requirement E3 — §3.6 T2 after (append):**
`- Initialization: β(0) = 0, b(0) = 0.0 (FORMALIZATION — matches 6E.18 runner T2ReadoutHead zero-init default). Deterministic; recorded in manifest.`
- NOTE: β(0) = s_scale·U_k^T û_LDA remains an available CHANGE (requires runner modification) if rank-isolation priority demands aligned init; not recommended under Amendment D option 1.

**Requirement E4 — §3.7 V1 after (append):**
`- Initialization (CHANGE — genuine protocol choice, requires 6E.18 runner implementation): w_PRO(0) = s_scale·û_LDA, w_ABS(0) = −s_scale·û_LDA, b_PRO(0) = b_ABS(0) = 0.0. Deterministic aligned mirror of the I1 symmetric scalar in 2-logit form; no RNG. This makes V1's starting geometry identical to I1's, isolating parameterization and capacity.`

**Requirement E5 — §4 table V1 cell (line 97):**
- Before: `| **V1** | Unconstrained Affine | 1794 | Random / zero | Independent 2-logit | Output form & unconstrained capacity |`
- After: `| **V1** | Unconstrained Affine | 1794 | s_scale·(û_LDA, −û_LDA) | Independent 2-logit | Output form & unconstrained capacity |`

**Requirement E6:** Delete "Random / zero" for V1 wherever it appears. No other §4 table cell changes.

**Reproducibility test:** a cold run from the amended §3.1–§3.8 alone must fully determine every initial weight, bias, basis, and scale without consulting the runner source.

---

## Amendment F — MINOR: Define or normatively reference all decision-rule operands

**Section:** §6.1 (after line 135), new metric items 4–9. All FORMALIZED from the 6E.17 runner definitions.

| # | Operand | After (new §6.1 text) |
|---|---|---|
| 4 | ProposalRate(t) | Fraction of dev-set items with margin ẑ_PRO − ẑ_ABS > 0 at checkpoint t. Used by P2 (L162) and M4 degradation. (FORMALIZATION, 6E.17 L678/733/766/925.) |
| 5 | Cohen's d(t) | (mean m_POS − mean m_ABS) / pooled sd over panel16 margins at t, pooled = sqrt((var_POS + var_ABS)/2), first 8 items POS, last 8 ABS. P3 uses d(15). (FORMALIZATION, 6E.17 L619–622.) |
| 6 | r_eff(t) | Participation-ratio effective rank tr²/tr² of the margin-Jacobian Gram G_margin at checkpoint t (N×N over the panel; margin gradients w.r.t. LoRA parameters). P3 uses r_eff(15). (FORMALIZATION, 6E.17 L121–125, L584–591, L929.) |
| 7 | Resub k-NN1 | Resubstitution accuracy of k=1 Euclidean nearest-neighbor classifier over the layer-24 last-token hidden state h_L, POS-vs-ABS labels. Used by P4. (FORMALIZATION, 6E.17 L530–548.) |
| 8 | LOO-CV(t) | Leave-one-out accuracy of the same k=1 NN classifier on h_L over the N=16 panel at checkpoint t. Used by P4/§7.1. (FORMALIZATION, 6E.17 L534–539.) |
| 9 | Policy metrics / degradation | Policy metrics = {ProposalRate(t), d(t), POS/ABS accuracy}. V1 degrades if ProposalRate(15) ≤ 0.0% OR d(15) < 0.00. Used by H5/M4. |

**Requirement F1:** Amend §7 C2x row (line 161) to bind aggregation:
- After: `mean over t ∈ {4, 6, 14, 15} of cos(H1)(I1) − cos(H1)(C2) ≥ +0.25`.

**Requirement F2:** Bind cos(H1) to §6.2 aggregate definition (mean-POS vs mean-ABS cosine). Conformance note for the 6E.18 runner: P1/M-rules must consume the aggregate cosine, not the mean-pairwise variant.

---

## Amendment G — MINOR: Σ_diff construction and P4 early-truncation fallback

**Sections:** §8 (line 177); §3.6 wording (line 77); §7.1 (after line 170); §10 (after line 202).

**Requirement G1 — §8 line 177 after (FORMALIZATION — matches 6E.18 L330–331 and the §8 literal):**
`1. Compute the total sample covariance Σ_diff of the N=16 step-0 panel hidden states: center X_panel on its global mean and form Σ_diff = X_centered^T X_centered / (N − 1). Deterministic; rank ≤ 15; construction recorded in the run manifest.`
- §3.6 (L77) SHALL be reworded to reference the amended §8 verbatim ("top-k SVD components of the empirical covariance of the step-0 panel as defined in §8"), removing the standalone term "discriminative panel" (the SVD basis is a principal subspace, not class-discriminative).

**Requirement G2 — §7.1 new bullet 3 (FORMALIZATION — matches 6E.18 evaluate_p4_comparator L299–319):**
`- **Early-truncation fallback:** If LOO-CV_B0(t) is unavailable for any shared checkpoint (including t ∈ {4, 6}) because the B0 guard triggered before evaluation, the step-0 anchor applies: |LOO-CV_I1(t) − LOO-CV_B0(0)| ≤ 0.10, and the missing matched comparator is recorded as UNAVAILABLE in the run manifest.`

**Requirement G3 — §10 new bullet:**
`- If the B0 guard triggers before checkpoint 6, contemporaneous matched comparators at t ∈ {4, 6} that were never evaluated are reported UNAVAILABLE and the §7.1 step-0 anchor is used; no implicit back-filling of B0 trajectories is permitted.`

**Requirement G4 — checkpoint schedule:** Record the evaluation checkpoint set (canonical: {0, 4, 6, 7, 14, 15, 17, 20} per 6E.18 runner) in the run manifest; rule aggregation sets remain as specified (P1/M-rules: {4,6,14,15}; P4: {0,4,6} and {14,15,20}).

---

## Amendment H — Amendment log entry (with FORMALIZATION / CHANGE labels)

**Section:** §13 (append after line 231).

**After (new entries):**
`- **v2.1.2 → v2.1.3:**`
`  1. Resolved REC-01 (BLOCKING): repaired phantom §4.4 reference to §4.1 (item 4); formally numbered §4.1 confound subsection.`
`  2. Resolved F-03 (MAJOR): added §7.2 frozen probe rules M1–M5 (F1/T1/T2/V1/C2).`
`  3. Resolved F-02 (MAJOR): defined F1 indeterminate band disposition and removed non-I1 "fail P1" naming in §5.2.`
`  4. Resolved F-04 (MAJOR): limited H4/T2 claim to the acknowledged rank × basis × initialization composite (§4.1 item 3, §5.1 H4, §5.2 row 4).`
`  5. Resolved F-05 (MAJOR): froze s_scale, b_δ, b(0) I1, b''(0) C2, b_fixed (§3.8) — FORMALIZATION (6E.17 derivation); T1 α(0)/b(0) — FORMALIZATION (6E.18 runner defaults); T2 β(0) = 0 — FORMALIZATION (6E.18 runner default); V1 init — CHANGE (aligned mirror; requires 6E.18 runner implementation; approved).`
`  6. Resolved MINOR set: defined proposal rate, d, r_eff, Resub k-NN1, LOO-CV, policy metrics/degradation, C2x aggregation (§6.1 items 4–9, §7 C2x) — FORMALIZATION (6E.17 runner definitions); Σ_diff fixed as total sample covariance (§8) — FORMALIZATION; P4 early-truncation fallback (§7.1, §10) — FORMALIZATION (6E.18 comparator).`

---

## Post-implementation audit checklist (for v2.1.3)

Run in order; each gate must pass before the next:
1. **Literal cross-reference audit** — every `§X.Y` citation resolves to an existing heading (no phantoms; re-check §4.1, §7.2, §8).
2. **Math / parameter reconciliation** — §3 ↔ §4 table ↔ §9 totals unchanged (B0 0, I1 897, C2 897, F1 0, T1 2, T2 k+1, V1 1794).
3. **Condition-by-condition executability audit** — each condition has metric → checkpoints → aggregation → threshold → PASS/FAIL/intermediate/unavailable dispositions.
4. **Causal-confound audit** — every comparison clean, or explicitly acknowledged with a limited claim.
5. **Reproducibility audit** — a cold run from §3 alone determines all initializations/constants without the runner source.
6. **Runner ↔ protocol conformance audit** — 6E.18 runner implements §6.2 aggregate cos(H1), §7.1 comparators + fallback, §8 total-covariance SVD, §3.8 constants, and the V1 aligned-mirror init (runner change), exactly as written.
7. **Independent adversarial re-audit** — fresh forensic review of the literal v2.1.3 artifact (new SHA-256), findings classified per the standing audit standard.

Only after all seven pass is the protocol eligible for human authorization. Hard stop remains active throughout.
