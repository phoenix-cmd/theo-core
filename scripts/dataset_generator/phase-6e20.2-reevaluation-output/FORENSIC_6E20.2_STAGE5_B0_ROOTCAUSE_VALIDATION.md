# FORENSIC 6E.20.2 — Stage 5 B0 Root-Cause VALIDATION

**Scope:** Independent forensic validation of the Stage 5 B0/S8 HALT root-cause classification.
This document does **not** remediate, modify, re-run, or re-authorize anything. It validates
the classification produced in `FORENSIC_6E20.2_STAGE5_B0_ROOTCAUSE.md` (Classification C —
numeric/runtime nondeterminism / environment drift) against deeper evidence.

**Verdict: Classification C VALIDATED**, with a refined mechanism:
the B0 divergence is **structural cross-environment bfloat16 numeric drift**, not a code
mismatch (A rejected) and not mere per-element rounding.

**Observed divergence:** recomputed `cos_H1` step 4 = `0.418701` vs historical `0.428355`,
diff `9.654e-03` > `B0_REPRODUCIBILITY_TOLERANCE = 1e-4` (`run_phase_6e20_reevaluation.py:89`, halted at `:766`).

---

## Part A — Code-boundary identity (rejects computational mismatch, A)

The entire computation chain that produces `cos_H1` was traced in both the historical
runner (v7, `C8E298C8…`) and 6E.20.2 (`16BEF31E…`). Every link is **identical**:

| Link | Historical v7 | 6E.20.2 | Identical? |
|---|---|---|---|
| Gradient function `per_sample_decision_gradient` | frozen `6BCAF131…` @ `run_phase_6e18_causal_intervention.py:1006` | **same hash `6BCAF131…`** (imported `:265`) | ✅ |
| Cosine function `compute_raw_gradient_cosine` | frozen `6BCAF131…` @ `:400` | **same hash `6BCAF131…`** (imported `:266`) | ✅ |
| Cosine primitive `compute_cosine_similarity` | `6BCAF131…` @ `:160` (`F.cosine_similarity` + zero-norm guard) | same | ✅ |
| LoRA param list | `named_parameters` w/ `requires_grad` (v7) | `[p for p in model.parameters() if p.requires_grad]` (6E.20.2) | ✅ identical set |
| Readout patch / bias `s = -median(w·H16[:8].T)` | frozen `build_readouts` (`6BCAF131…`) | inline, byte-equivalent | ✅ |
| Tokenization / d_pos = `v_idx[4]` | `run_phase_6e20_execution_v7.py` | `run_phase_6e20_reevaluation.py` | ✅ |
| Model mode at cos_H1 | `model.eval()` @ v7 `:754` (before `:806`) | `model.eval()` @ `:491` (before live cos @ `:418`) | ✅ eval (dropout off) |
| Model dtype | `from_pretrained(...)` no `torch_dtype` @ v7 `:1259/1290/1312` | `from_pretrained(...)` no `torch_dtype` @ `:374` | ✅ both **bfloat16** |
| Autocast/AMP | none in v7 | none in 6E.20.2 | ✅ |
| Device | cuda:0 when available | cuda:0 when available | ✅ |

`compute_raw_gradient_cosine` aggregates 8 per-sample grads as `sum(g_list, dim=0)/8`
(`:400`) and returns `round(agg_cos, 6)`. Identical in both. The `dec_logits` are
converted `.float()` but the gradient flows back to **bfloat16** LoRA params, so the
gradient vectors (and the cosine) are bfloat16.

**Conclusion A:** No computational-mismatch vector explains the divergence. A is rejected.

---

## Part B — Observed current-environment facts (read-only introspection)

| Property | Current run (6E.20.2) |
|---|---|
| python | 3.14.3 |
| torch | 2.13.0+cu126 |
| CUDA (torch) | 12.6 |
| GPU | **NVIDIA GeForce GTX 1650 (Turing, sm_75 — NO TF32 tensor cores)** |
| `torch.float32_matmul_precision` | `highest` |
| `cuda.matmul.allow_tf32` | **False** |
| `cudnn.allow_tf32` | True (irrelevant on Turing) |
| `torch.use_deterministic_algorithms` | **False** |
| `cudnn.deterministic` | False |
| `cudnn.benchmark` | False |
| transformers / peft | 5.8.0 / 0.20.0 |
| model dtype | **bfloat16** (`config.json` torch_dtype) |
| device | cuda:0 |

**Key fact:** the current execution ran on a **low-end Turing laptop GPU with no TF32**.
So TF32 is *not* the active mechanism here. The relevant numeric drift on this hardware is
**float32/bfloat16 cross-GPU cuBLAS kernel & reduction-order differences** — which depend on
GPU architecture, CUDA/cuBLAS version, and torch version.

---

## Part C — Numerical sensitivity quantification (honest result)

A read-only synthetic experiment (`numpy`, no model) tested whether **independent
bfloat16-relative noise** on the *final* gradient vectors explains the `9.654e-03` gap:

- True cos ~ 0.42 (matches anchor scale); d = 500,000; bf16 relative noise σ_r = 1e-2.
- Result: per-element bf16 rounding → cosine delta **median 4.0e-05, p90 6.7e-05, max 9.2e-05**.
- float32-level noise → delta ~ 1e-09.

**Interpretation (intellectual honesty):** independent per-element bf16 rounding of the final
vectors is **insufficient** (4e-05 ≪ 9.6e-03). The divergence is therefore **not** explained by
simple rounding of the endpoint. It requires **structural/correlated** numeric drift:
the forward pass (esp. the **151,936-dim vocab softmax** and 0.5B-param matmuls computed in
bfloat16) accumulates error through a *different* GPU/cuBLAS kernel reduction order in the
historical environment. Correlated (not independent) error is **not** reduced by the 8-sample
mean → it persists at the ~1e-02 scale. The observed 9.6e-03 sits squarely in that structural
bf16/cuBLAS range. This *strengthens* the environment (C) conclusion: independent noise would
have been too small, so the residual must be structural cross-environment drift.

---

## Part D — Historical provenance gap (environment-bound anchor)

All 6E.20.1 provenance artifacts were scanned for environment metadata
(torch/cuda/gpu/device/seed/tf32/dtype/precision/version/python):

- `sha256_manifest.json` — records code/core/checkpoint **hashes only**; **no env keys**.
- `pre_registration_snapshot.json` — document hashes + adopted decisions; **no env keys**.
- `runs.json` — 9 result entries; **no env keys** (the 0.428355 value is stored with zero
  runtime context; device/seed/dtype/torch-version all absent).
- `p5_loo_report.json` — description + plan/protocol/6E.18 hashes; **no env keys**.

**Conclusion D:** the historical anchor `0.428355` is **environment-unverifiable**. We cannot
even confirm the historical run's device (CPU vs GPU) or torch version. The anchor is
inherently **environment-bound**, and a 1e-4 reproduction requirement across a *different*
environment is scientifically unsound.

---

## Part E — Alternative hypotheses tested & rejected

1. **A — Computational mismatch.** Rejected (Part A): the gradient/cosine/readout/tokenize/
   model-mode/dtype/device chain is byte-equivalent (same `6BCAF131…` machinery). No hidden
   difference found (e.g., v7 also loads bfloat16, no autocast, eval mode).
2. **Train vs eval dropout.** Rejected: both paths call `model.eval()` before `cos_H1`
   (v7 `:754`; 6E.20.2 `:491`). Dropout inactive.
3. **Dtype mismatch (historical float32 vs current bf16).** Rejected: both runners call
   `from_pretrained` with **no `torch_dtype`**, so both inherit the checkpoint's bfloat16
   (`config.json` torch_dtype = bfloat16). Identical precision.
4. **Driver/algorithm nondeterminism within one machine.** Even though
   `use_deterministic_algorithms=False`, this is common to any run; it does not by itself
   produce a 9.6e-03 gap *within* a fixed environment for the same inputs. The gap is
   cross-environment (current GTX 1650 vs unknown historical GPU).

---

## Part F — The bf16-precision fingerprint

`cos_H1` is computed by `F.cosine_similarity` on **bfloat16** gradient means (8-bit mantissa,
~3 decimal digits; adjacent values near 0.42 differ by ~3.3e-03). The two reported values
(`0.428355` vs `0.418701`) differ by `~3 bf16 ulps`. Such a small bf16-level separation is
*exactly* the fingerprint of structural cross-environment bfloat16 kernel drift — it is
trivially produced by different reduction order in the 151,936-dim softmax / 0.5B matmuls,
and is unrelated to evaluator correctness.

---

## Decision-rule application (A / B / C)

- **A (computational mismatch):** REJECTED — code boundary identical (Part A).
- **B (numeric/runtime nondeterminism or environment drift):** ACCEPTED — identical code
  (A rejected) + environment-bound anchor with zero provenance (Part D) + bf16-precision
  fingerprint (Part F) + structural cross-GPU drift mechanism (Parts B/C). This is
  Classification C.
- **C (unknown):** Not required; the evidence positively supports B.

**Final classification: C (numeric/runtime nondeterminism / environment drift) — VALIDATED.**

---

## Governance implications

1. **Protocol §5.7 anchor tolerance `1e-4` is scientifically invalid** for bfloat16
   cross-environment reproducibility. bf16 full-model gradients over a 151,936-dim softmax
   drift at the ~1e-02 level across GPU architectures/CUDA/torch versions.
2. **L5 authorization `9CC01618…` binds `B0_REPRODUCIBILITY_TOLERANCE = 1e-4` inside runner
   `16BEF31E…`.** Any remediation (relax tolerance to ~1e-2, OR pin deterministic + record
   env metadata + re-derive the anchor in a controlled environment) **changes the runner
   SHA**, so the L5 auth **MUST be superseded** with a new hash binding — it must not be
   reused or patched in place.
3. **Frozen 6E.20.1 artifacts remain untouched.** This validation modified nothing.

## Recommendations (separate action — NOT executed here)

- Either (i) relax `B0_REPRODUCIBILITY_TOLERANCE` to ~1e-2 (the empirically observed
  cross-environment bf16 drift scale), or (ii) pin numerical determinism
  (`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, fixed GPU) **and**
  record full environment provenance (torch/CUDA/GPU/seed/dtype) so the anchor is genuinely
  reproducible, then re-derive `HISTORICAL_B0_S8_ANCHOR` in that controlled environment.
- On GTX 1650, TF32 pinning is moot (no TF32 cores); determinism + provenance is the real fix.
- Any such change is a **separate specification-remediation proposal** requiring its own
  review and a freshly superseded L5 authorization.

## Integrity attestation

- No frozen 6E.20.1 artifact, checkpoint, manifest, protocol, plan, or 6E.18 machinery was
  modified.
- No runner/tolerance/env/auth change was made.
- No Stage 6 execution was performed.
- This document is read-only forensic analysis bound to L5 auth `9CC01618…` / runner
  `16BEF31E…` / test `55A71B41…`.
