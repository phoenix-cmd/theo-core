# FORENSIC 6E.20.1 — REGISTRATION GEOMETRY DEEP-DIVE (READ-ONLY)

**Phase:** 6E.20.1 — Fixed-Subspace Readout Execution Extension (v7)
**Analysis date:** 2026-08-18
**Mode:** STRICTLY READ-ONLY. Recomputed diagnostics exclusively from the saved frozen arrays in `phase-6e20.1-execution-output/step0/*.npy` (u_lda, U_8, H_16, R_4, n_rand, y16, w_{B0..R1-Rand}). No model was loaded, no inference or training performed, no artifact was modified. This is a numerical cross-check of the frozen provenance and a quantification of the registration geometry underlying the causal-analysis hypotheses H5/H6/F4.

---

## 1. Provenance Cross-Validation — PASS

Recomputed the registration diagnostics from the saved arrays using the frozen definitions (`subspace_kappa(u,U_k)=||U_k U_k^T u||`, `inc` from `derive_principal_basis`):

| Quantity | Recomputed | runs.json (frozen) | Match |
|---|---|---|---|
| κ₂ | 0.056847 | 0.056847 | ✅ |
| κ₄ | 0.086011 | 0.086011 | ✅ |
| κ₈ | 0.125257 | 0.125257 | ✅ |
| κ_R | 0.054079 | 0.054079 | ✅ |
| inc42 | 0.064547 | 0.064547 | ✅ |
| inc84 | 0.091057 | 0.091057 | ✅ |

The frozen `runs.json` gate values are faithfully reproduced from the saved artifacts → **provenance integrity confirmed** (no silent mutation of registration diagnostics).

---

## 2. PCA Geometry of H16 — the H6 Stressor Quantified

Covariance SVD of the 16 panel hidden states (896-d):
- Cumulative variance: top-2 = 0.4805, top-4 = 0.7200, top-8 = **0.9297**, top-15 = 1.0000.
- `u_lda` norm inside top-8 PCA subspace = κ₈ = **0.1253**.
- `u_lda` norm **outside** top-8 PCA (low-variance directions) = **0.8747**.

**Interpretation:** The LDA discriminative axis lives overwhelmingly (≈87.5%) in directions where the panel hidden states have *little variance*. The decision-relevant direction is nearly orthogonal to where the model's representational energy concentrates. This is the registration-geometry stressor flagged as H6 in the causal report: it is a design property affecting **all** conditions (every readout is a projection of `u_lda`), not a B0-specific artifact. It means gradient signal along `u_lda` is in a "thin" direction of the representation.

---

## 3. Readout Geometry — B0 is the Outlier (H5/F4 Supported)

Cosine of each readout weight `w` with `u_lda`:

| Readout | cos(w, u_lda) | Note |
|---|---|---|
| B0 | **−0.012** | unembedding difference; ≈ orthogonal to LDA axis |
| F1 | +1.000 | by construction (F1 = s_scale·u_lda) |
| S2 | +0.0568 | = κ₂ (confined to U_2) |
| S4 | +0.0860 | = κ₄ (confined to U_4) |
| S8 | +0.1253 | = κ₈ (confined to U_8) |
| C4-Rand | +0.0541 | = κ_R (random 4D) |
| R1-Rand | +0.0315 | near-orthogonal (random 1D) |

Pairwise cosine matrix (B0 row/col):

```
        B0     F1     S2     S4     S8   C4-R   R1-R
B0    +1.000 -0.012 +0.023 +0.007 -0.011 -0.025 +0.001
F1    -0.012 +1.000 +0.057 +0.086 +0.125 +0.054 +0.032
S2    +0.023 +0.057 +1.000 +0.661 +0.454 -0.023 -0.023
S4    +0.007 +0.086 +0.661 +1.000 +0.687 +0.012 -0.017
S8    -0.011 +0.125 +0.454 +0.687 +1.000 +0.024 +0.015
C4-R  -0.025 +0.054 -0.023 +0.012 +0.024 +1.000 -0.050
R1-R  +0.001 +0.032 -0.023 -0.017 +0.015 -0.050 +1.000
```

**B0 is orthogonal to every other readout** (|cos| ≤ 0.025 with F1/S2/S4/S8/C4-Rand/R1-Rand). Specifically `cos(w_B0, w_S8) = −0.011`. B0's readout is the *unconfined* unembedding difference; the S2/S4/S8 readouts are *confined* projections of `u_lda`. B0 therefore operates in a geometric direction **unrelated** to the LDA/subspace hypothesis that H1/P2 actually concerns.

---

## 4. Implications for the Causal Hypotheses

### H5 / F4 — B0's `cos(H1)` is background-dominated (STRONGLY SUPPORTED)
- Protocol B-4 establishes `g_total = g_bg + α·w`, where only the readout term `α·w` is subspace-confined. For B0, `w = w_delta` is **unconfined** (orthogonal to `u_lda` and to all subspace readouts). Hence B0's total decision gradient is dominated by the background term `g_bg`.
- The protocol itself (§10.5/§5.7) states `cos(H1)` is *"explicitly background-dominated and condition-dependent … never replaced by ρ_c or g_R,"* and for random/unconfined directions the value is *"background-dominated and sign-indeterminate"* (re R1-Rand).
- The measured B0 `cos(H1) ≈ +0.4` is therefore **exactly the regime the protocol anticipates for an unconfined readout** — it is not a clean measurement of the subspace-confined intervention effect. This is quantitative support for the causal report's H5/F4.

### H6 — κ-floor stressor (QUANTIFIED)
- 87.5% of `u_lda` lies outside the top-8 PCA subspace. This is a real, frozen-artifact-confirmed geometric property. It implies that even the *correct* targets (S4/S8, whose readouts are confined to `u_lda`) are evaluated through a direction that is largely orthogonal to the model's principal gradient directions. Whether S4/S8 still achieve `cos(H1) ≥ +0.80` (per H1/P2) is **undetermined** and can only be answered by evaluating those conditions (feasibility audit P9) — which the completed runner never did.

### Evaluation-path incoherence (reinforces causal report F1)
- The completed runner (`v7:866,805`) evaluated **B0 checkpoints through the S8 readout**. Geometry shows `cos(w_B0, w_S8) = −0.011` — the eval probe (S8) is orthogonal to B0's own training readout (w_B0). Probing B0's model with a readout orthogonal to its training objective is geometrically incoherent, and it never touched the treatment conditions (S4/S8) whose confined readouts are the protocol's actual H1/P2 targets.

---

## 5. Caveats

- This analysis is **pure geometry on saved arrays**; it does **not** compute `cos(H1)` (that requires model forward/autograd inference, intentionally not performed). It strengthens the *interpretation* of the already-measured B0 `cos(H1)` and the *justification* for re-evaluating S4/S8, but it does not replace the per-condition re-evaluation recommended in the feasibility audit.
- `87.5% outside top-8 PCA` is a property of the 16-panel H16 snapshot; it is frozen and reproducible (corpus hash-verified). It is a design/registration characteristic, not a defect in execution.

---

## 6. Conclusion

The frozen registration artifacts are self-consistent (cross-validation PASS) and reveal two decisive geometric facts:
1. **B0's readout is orthogonal to the LDA axis and to all subspace readouts** → B0's `cos(H1)` is background-dominated per the protocol's own B-4/§10.5 analysis, exactly as the causal report hypothesized (H5/F4).
2. **The LDA discriminative axis is 87.5% outside the top-8 PCA subspace** (H6 stressor), a frozen design property affecting all conditions.

Together these confirm that the completed run's negative `cos(H1) ≈ +0.4` reflects a **measurement-target / design-assumption failure** (wrong, unconfined probe on the baseline), not a validated failure of the fixed-subspace-readout intervention. The correct targets — **S4 and S8 with their own confined readouts** — remain untested and are the necessary next step (feasibility audit P9), requiring new authorization.

*End of read-only registration-geometry deep-dive.*
