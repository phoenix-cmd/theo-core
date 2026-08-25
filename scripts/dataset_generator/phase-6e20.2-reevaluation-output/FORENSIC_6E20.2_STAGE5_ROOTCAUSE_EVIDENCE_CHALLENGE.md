# FORENSIC 6E.20.2 — STAGE 5 ROOT-CAUSE REPORT CHALLENGE & EVIDENCE GAP AUDIT

**Mode:** READ-ONLY FORENSIC REVIEW ONLY.
No modification of runner, tolerance, protocol, plan, frozen artifacts, authorization,
environment settings, or checkpoints. No Stage 6 execution. No authorization changes.

**Scope:** Independently challenge the conclusion "Classification C is VALIDATED as
structural cross-environment bfloat16 numeric drift." Determine whether available evidence
supports classification A, B, or C.

**Verdict: B — Runtime drift plausible, not proven.**

---

## 1. Challenge the BF16 mechanism claim

### What the prior report asserted
"Independent endpoint BF16 perturbation yields ~4e-05, therefore the 9.654e-03 gap must
be structural/correlated BF16/cuBLAS drift from different GPU architectures."

### What the evidence actually supports

**Independent bf16 rounding of final gradient vectors** was tested via synthetic experiment
(500,000-dim random vectors, bf16-level relative noise σ_r = 1e-2, cosine delta
median = 4.0e-05, max = 9.2e-05). This is well below the observed 9.654e-03. The prior
report correctly identified this insufficiency, then **inferred** the residual must be
structural/correlated drift. This inference is **not automatically valid**.

The following direct evidence is **absent**:

| Factor | Directly known? | Evidence |
|---|---|---|
| Historical GPU architecture | **UNKNOWN** — no provenance | — |
| Historical CUDA version | **UNKNOWN** | — |
| Historical cuBLAS kernel selection | **UNKNOWN** | — |
| Historical PyTorch kernel dispatch | **UNKNOWN** | — |
| Historical reduction algorithm | **UNKNOWN** | — |
| Historical BF16 accumulation behavior | **UNKNOWN** | — |
| Historical model loading dtype | **INFERRED** (from source + config; not runtime-verified) | — |
| Historical gradient accumulation behavior | **UNKNOWN** | — |
| Current vs historical attention backend | **UNKNOWN** — see §4 | — |

**Assessment:** A correlated mechanism *could* exist. The evidence does not demonstrate
that it *did* produce the observed divergence. "Plausible mechanism" ≠ "proven cause."

---

## 2. Challenge the "~3 BF16 ULP fingerprint" claim

### The arithmetic
BF16 ULP near x ≈ x · 2⁻⁷. Near 0.42: ULP ≈ 0.0033.
|0.428355 − 0.418701| / 0.0033 ≈ 2.9 → ~3 ULPs.

### Problems with this as forensic evidence

1. **Circular:** ANY small numerical difference falls within *some* number of ULPs.
   If the values had differed by 0.001, that would be ~0.3 ULPs; by 0.01, ~3 ULPs;
   by 0.05, ~15 ULPs. Choosing "3 ULPs" as meaningful is retrospective pattern-matching,
   not a predicted fingerprint.

2. **Spurious precision:** `compute_raw_gradient_cosine` returns `round(agg_cos, 6)`.
   The cosine is computed on bf16 tensors via `F.cosine_similarity` on bf16 inputs —
   the *underlying* precision is bf16 (~3 decimal digits). The `round(x, 6)` call
   converts the bf16 scalar to a Python float (float64) and formats to 6 decimals,
   but the trailing 3 digits carry no information beyond bf16 precision. The 6-decimal
   output is **spurious precision**, not enhanced resolution.

3. **Mixed accumulation:** The cosine is `F.cosine_similarity(pos_mean, abs_mean)` where
   pos_mean = `sum(g_pos_list, dim=0) / 8` (bf16). `torch.linalg.vector_norm` inside
   `F.cosine_similarity` may internally promote to float32 for the norm computation
   (PyTorch promotes for linalg). The dot product may also use mixed accumulation.
   The result is not a pure bf16 operation — it's bf16 with possible float32 internal
   promotion — making bf16 ULP analysis unreliable as a diagnostic.

4. **Not predictive:** The "~3 ULP" observation was not predicted *a priori*; it was
   computed *after* the fact to match the observed gap. A valid fingerprint would predict
   a specific gap magnitude *before* observing it.

### Verdict
**WEAK HEURISTIC, not probative.** The ULP observation is consistent with bf16 drift
but does not distinguish it from other small numerical causes. It has no independent
forensic power.

---

## 3. Verify historical dtype claims

### Layered audit of the bf16 chain

| Layer | What | How known | Status |
|---|---|---|---|
| config.json declares `torch_dtype = bfloat16` | File read | **KNOWN** |
| Checkpoint storage dtype | Not inspected (safetensors headers not read) | **INFERRED** (matches config) |
| `from_pretrained` loads bf16 (no `torch_dtype` kwarg) | Source-level (v7 `:1259/1290/1312`, 6E.20.2 `:374`) | **KNOWN** for current; **INFERRED** for historical |
| Loaded parameter dtype at runtime | Not verified (no model loaded + printed dtype in historical run) | **INFERRED** |
| Forward compute dtype | Depends on param dtype | **INFERRED** |
| LoRA param dtype | Depends on `get_peft_model` + PEFT version + `autocast_adapter_dtype` default | **INFERRED** |
| Gradient dtype (autograd) | Follows param dtype (PyTorch convention) | **INFERRED** |
| CUDA kernel dtype | Follows input dtype | **INFERRED** |

### Key subtlety
PEFT `get_peft_model` has parameter `autocast_adapter_dtype: bool = True`
(PEFT 0.20.0). This casts LoRA adapter weights to match the base model dtype.
**Older PEFT versions** may not have this parameter, or may default differently.
If the historical run used a PEFT version where LoRA weights were initialized in
float32 (before autocast_adapter_dtype was added), the LoRA forward pass could
compute in float32 rather than bf16, producing different numerical results.

The checkpoint loading (`strict=False`) should overwrite initialized weights, but
the **dtype of the LoRA layer itself** (bf16 vs float32 parameter) depends on PEFT
initialization behavior, which depends on version. **This is historically unknown.**

### Honest conclusion
The claim "both runs are bf16" is well-supported for the **model base parameters**
(config + from_pretrained behavior), but the **LoRA computation precision** depends
on PEFT version, which is historically unknown. The bf16 chain is **inferred, not
runtime-verified** for the historical run.

---

## 4. Adversarial search for runtime-visible semantic differences

### PROVEN IDENTICAL (source-level, hash/line verified)
- `per_sample_decision_gradient` body (`6BCAF131…`)
- `compute_raw_gradient_cosine` body (`6BCAF131…`)
- `compute_cosine_similarity` body (`F.cosine_similarity` + zero-norm guard)
- LoRA config: `r=16, alpha=32, dropout=0.05, targets=[7 modules], bias="none, task_type="CAUSAL_LM"` — identical in both runners
- `from_pretrained(MODEL_ID, revision=MODEL_REVISION)` — no `torch_dtype`, no `attn_implementation`, no `use_cache` override — identical call
- `model.eval()` before `cos_H1` (v7 `:754`, 6E.20.2 `:491`)
- `torch.no_grad()` during inference (v7 `:433/:756`, 6E.20.2 `:493`)
- No autocast/AMP in either runner
- Tokenization path identical
- d_pos = `v_idx[4]`, last layer hidden state — identical

### PROVEN DIFFERENT
- **Library versions**: current = transformers 5.8.0, PEFT 0.20.0, torch 2.13.0+cu126, CUDA 12.6. Historical: **ALL UNKNOWN**.
- **GPU architecture**: current = GTX 1650 (Turing sm_75). Historical: **UNKNOWN**.
- **Attention backend**: current = SDPA (no flash_attn, no xformers, Qwen2 default in transformers 5.8.0). Historical: **UNKNOWN** — depends on historical transformers version + GPU + available libraries. Different attention backends (eager vs SDPA vs Flash) produce different bf16 numerical results in the forward pass, which propagate to different hidden states → different gradients → different cos_H1.
- **PEFT internals**: `get_peft_model` behavior (especially `autocast_adapter_dtype` default and LoRA layer dtype) depends on PEFT version. Historical PEFT version **UNKNOWN**.

### HISTORICALLY UNKNOWN (no provenance, no runtime verification)
- All library versions (transformers, PEFT, PyTorch, CUDA, cuBLAS, tokenizers)
- GPU architecture, CUDA kernel dispatch, cuBLAS kernel selection
- Attention backend (SDPA vs eager vs Flash)
- Flash Attention / xformers availability
- `torch.use_deterministic_algorithms` setting
- `cudnn.deterministic` / `cudnn.benchmark` settings
- Device (CPU vs GPU) — runs.json has zero device metadata
- Seed / RNG state
- Default dtype context (torch.set_default_dtype)
- Model parallelism / device_map
- Gradient checkpointing
- All environment metadata (absent from all 6E.20.1 provenance files)

### Assessment
The source code is identical, but the **runtime behavior** depends on library versions
and hardware that are entirely unknown for the historical run. The attention backend
is a particularly significant factor: different SDPA/eager/Flash implementations
produce meaningfully different bf16 numerical results in the forward pass. This is a
**legitimate unresolved alternative explanation** that cannot be ruled out with available
evidence.

---

## 5. Separate two conclusions that must not be conflated

### Question A — Is the 1e-4 cross-environment gate adequately specified?

**NO.**

The gate requires the B0 anchor (`0.428355`) to be reproduced within `1e-4` in a
subsequent execution. However:

1. The historical run's environment is **entirely unrecorded** — no torch/CUDA/GPU/
   seed/dtype/version metadata in `runs.json`, `sha256_manifest.json`,
   `pre_registration_snapshot.json`, or `p5_loo_report.json`.
2. Without environment provenance, **environment parity cannot be verified or ensured**
   for any subsequent run.
3. The gate therefore tests *something* — but it cannot be determined whether it tests
   evaluator stability (same environment), evaluator correctness (any environment), or
   simply environment coincidence.

**The provenance gap alone is sufficient to conclude the gate is inadequately specified.**
This does not require identifying the cause of the divergence.

### Question B — Has the 0.009654 divergence been proven to result from runtime drift?

**NO.**

What has been proven:
- Source code is identical (no computational mismatch at the code level).
- Both runs infer bf16 from config + standard library behavior.
- Historical environment is unverifiable.

What has NOT been proven:
- That the historical run used a different GPU, CUDA, torch, transformers, PEFT,
  attention backend, or any other runtime factor.
- That any specific runtime factor *caused* the specific 9.654e-03 divergence.
- That the divergence mechanism is bf16 cross-GPU cuBLAS drift (as opposed to, e.g.,
  attention backend difference, PEFT version behavior, or an unidentified factor).

**Absence of proven computational mismatch ≠ proof of runtime causation.** The absence
means runtime drift is *compatible with* the evidence, but the evidence is also compatible
with an unidentified computational difference that was not caught by source-level audit.

### Different answers
- Question A: Gate inadequately specified (proven by provenance absence).
- Question B: Divergence cause unresolved (plausible runtime drift, not proven).

These are **independent conclusions**. The gate's inadequacy does not prove the
divergence's cause. The divergence's unresolved cause does not excuse the gate's
inadequacy.

---

## 6. Final classification

**B — Runtime/environment drift plausible, not proven.**

Rationale:
- No computational mismatch found at the source level (rejects A as a *code-level*
  explanation).
- Historical provenance is entirely absent, preventing verification of environment
  parity.
- Multiple historically unknown runtime factors exist (attention backend, library
  versions, GPU) that are PLAUSIBLE sources of bf16 numerical divergence.
- However, no direct evidence causally links the specific 9.654e-03 divergence to any
  specific runtime factor.
- The synthetic sensitivity test showed that independent bf16 rounding is insufficient
  (4e-5), meaning the mechanism — if runtime drift — is more complex than simple
  endpoint rounding. The specific mechanism has not been identified or demonstrated.
- The "~3 ULP" observation is a weak heuristic without independent probative value.

**Classification C (insufficient evidence) is also defensible**, but B is more precise:
the evidence is *consistent with* runtime drift and *incompatible with* code-level
mismatch, but does not *demonstrate* runtime causation. "Plausible, not proven" is the
most honest characterization.

---

## Governance conclusions

### Stage 5 halt
**Remains valid.** The halt was triggered by a factual observation: the recomputed
cos_H1 (0.418701) differs from the historical anchor (0.428355) by 9.654e-03, exceeding
the tolerance (1e-4). This observation is not invalidated by the inability to identify
the exact cause. The halt is a safety gate, not a causal claim.

### Tolerance relaxation
**NOT JUSTIFIED NOW.** Without understanding the actual cause of the divergence,
relaxing the tolerance is a policy decision, not a scientifically justified remediation.
If the divergence is due to a code-level issue that was missed, relaxing tolerance would
mask a real problem. If it is due to environment drift, relaxing to ~1e-2 would be
appropriate — but this cannot be concluded from available evidence.

### Specification/provenance remediation proposal
**MAY PROCEED (independently).** The provenance gap (no environment metadata in any
6E.20.1 artifact) is a real, demonstrable deficiency that should be addressed regardless
of the root cause. A proposal to require environment provenance in future executions
is justified by the provenance audit alone, without requiring causal attribution of
the divergence.

### Active authorization
**REMAINS BOUND** to the halted runner `16BEF31E56D781170021092052CF708A92875EE68697FEB81FB77764E15E691C` and test `55A71B4115BA050FA22C3E83290A6B5D4C1FB45977D6448EF24000E03E5530A1` (auth SHA `9CC01618E3CD1CDC2364A07DF53E7BE3A803DEAA2D284D2EAA0761717A6B4B2A`). It **cannot authorize a changed runner**. Any remediation that alters the runner SHA (tolerance change, determinism pinning, env-pinning) requires a **new L5 authorization** binding the new runner hash.

---

## Appendix: Evidence inventory

### Directly verified
- Code identity: psdg, crgc, cosine, readout, LoRA config, from_pretrained, model.eval, no_grad — all verified via hash inspection and line-by-line comparison.
- Current environment: python 3.14.3, torch 2.13.0+cu126, CUDA 12.6, GTX 1650 (sm_75), transformers 5.8.0, PEFT 0.20.0, model config torch_dtype=bfloat16, no flash_attn, no xformers, SDPA backend, use_deterministic_algorithms=False, cudnn.deterministic=False.
- Provenance gap: all four 6E.20.1 output files contain zero environment metadata (verified via regex scan).
- Synthetic sensitivity: independent bf16 rounding of final vectors → cos delta median 4e-05 (insufficient to explain 9.654e-03).

### Inferred (not runtime-verified)
- Both runs compute in bf16 (from config + from_pretrained behavior + PyTorch conventions).
- cos_H1 is computed on bf16 tensors (from code structure: bf16 gradient tensors → bf16 F.cosine_similarity).

### Unknown (no evidence available)
- Historical GPU, CUDA, torch, transformers, PEFT, cuBLAS, tokenizers versions.
- Historical attention backend (SDPA vs eager vs Flash).
- Historical PEFT `autocast_adapter_dtype` behavior.
- Historical device (CPU vs GPU), seed, deterministic settings.
- Whether the specific divergence was caused by runtime drift (vs. an unidentified factor).

---

ROOT-CAUSE CAUSATION STATUS: B — TOLERANCE CHANGE: NOT JUSTIFIED — NO REMEDIATION APPLIED.
