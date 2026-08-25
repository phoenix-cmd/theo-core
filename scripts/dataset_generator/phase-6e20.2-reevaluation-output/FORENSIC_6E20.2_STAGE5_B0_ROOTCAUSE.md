# Forensic Root-Cause Investigation — Phase 6E.20.2 Stage 5 B0/S8 Reproducibility Failure

**Date:** 2026-08-19
**Mode:** READ-ONLY forensic investigation. No `--execute` rerun, no Stage 5 retry, no code/artifact/tolerance/authorization modification.
**Observed failure:** `Stage 5 HALT: B0 reproducibility failure at step 4! Recomputed=0.418701, Historical=0.428355, Diff=9.654000e-03 > 0.0001`
**Delta:** `abs(0.418701 − 0.428355) = 0.009654 > 0.0001`

---

## 1. Compared Paths

- **Historical (frozen value source):** `run_phase_6e20_execution_v7.py` (6E.20.1 runner, `C8E298C8…`) → F7 `compute_cos_H1_from_live_model` → frozen 6E.18 `PER_SAMPLE_DECISION_GRADIENT` / `COMPUTE_RAW_GRADIENT_COSINE`. Value `0.428355` recorded in `phase-6e20.1-execution-output/runs.json` (event `evaluation`, `checkpoint_step=4`, `cos_H1=0.428355`).
- **Current:** `run_phase_6e20_reevaluation.py` (`16BEF31E…`) → `run_stage5_b0_historical_reproducibility_anchor` → `compute_live_cos_H1` → frozen 6E.18 `psdg_fn` / `crgc_fn` (same file, hash `6BCAF131…`).

---

## 2. Dimension-by-Dimension Classification

| # | Dimension | Classification | Evidence |
|---|-----------|----------------|----------|
| 1 | Model init & revision resolution | **IDENTICAL** | Both `AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)`; Stage 2 verified `model.config.vocab_size=151936`, pinned revision present. |
| 2 | Tokenizer loading/config | **IDENTICAL** | Both `AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)`; no extra tokenizer args. |
| 3 | `train()` vs `eval()` state | **IDENTICAL** | v7 `evaluate_checkpoint_at_state` calls `model.eval()` (line 754) **before** cos_H1 (line 806) → eval. 6E.20.2 model stays eval post-`from_pretrained`. **Rejected earlier train/eval hypothesis.** |
| 4 | `no_grad` / autograd / graph | **IDENTICAL** | Both compute real autograd gradients via frozen `psdg`; no `no_grad` wrap around the gradient path. |
| 5 | Device / dtype / precision / autocast / TF32 / deterministic / backend | **DIFFERENT — CANDIDATE ROOT CAUSE** | Neither runner pins `allow_tf32`, `use_deterministic_algorithms`, or a seed. Default float32 TF32 tensor-core math applies. Historical runtime (torch/cuda/cuBLAS/GPU) is unrecorded and differs from the current environment. |
| 6 | LoRA restore / load order / param selection | **IDENTICAL** | Same `LoraConfig` (r=16, α=32, dropout=0.05, targets=q/k/v/o/gate/up/down, bias=none, CAUSAL_LM). `load_state_dict(strict=False)`; `lora_params = requires_grad` params; 336 keys verified. |
| 7 | Checkpoint state-dict / strict=False / key filtering | **IDENTICAL** | Both `load_state_dict(strict=False)` on the same SHA-verified checkpoint files; same 336-key set. |
| 8 | Readout patching / class replacement | **IDENTICAL** | Both monkeypatch `Readout.forward_logits` to `detach().cpu().float()` before `np.asarray` (6E.20.2 `ensure_readout_class_patch` ≡ v7 patch). |
| 9 | B0 checkpoint selection & step mapping | **IDENTICAL** | Both load `historical_out_dir/B0/checkpoint_{t}/model.pt` for `t ∈ {4,6,14,15}`. |
| 10 | S8 readout selection & dtype/device | **IDENTICAL** | Both use S8 readout; `w` = `w_S8.npy` (SHA-verified); bias = `−median(w·H16[:8]ᵀ)` (frozen `build_readouts` formula ≡ 6E.20.2 inline). float64 `w·h + b`. |
| 11 | Prompt reconstruction / corpus / dedup / panel order / labels | **IDENTICAL** | Both `np.random.seed(42)` + same SHA-verified corpus + same family-group/dedup/shuffle → identical 16 panel prompts; labels `[-100]*len(p_ids)+t_tokens`. |
| 12 | Tokenization args (pad/trunc/maxlen/special/attn) | **IDENTICAL** | Both `tokenizer(prompt, return_tensors="pt")` — no padding/truncation/max_length/special-token args; `attention_mask=[1]*len`. |
| 13 | Forward-margin calculation | **IDENTICAL** | `forward... margin = float64(w·h) + b`; patch moves `h` to CPU float first. |
| 14 | PSDG / CRGC identity & gradient target | **IDENTICAL** | **Same frozen file** `run_phase_6e18_causal_intervention.py` hash `6BCAF131…` (current) = `frozen_6e18_gtotal_machinery_sha256` in 6E.20.1 manifest. Gradient target = LoRA params of decision loss. |
| 15 | Gradient aggregation & cosine | **IDENTICAL** | Both aggregate 8 POS + 8 ABS gradients; `crgc` returns `cos_H1_aggregate` via identical formula. |
| 16 | Randomness / nondeterminism | **IDENTICAL at source** | No torch seed set in either; in eval mode dropout is OFF → deterministic *given a fixed environment*. Only nondeterminism is runtime float32 (TF32/cuBLAS). |
| 17 | Library / runtime version drift | **DIFFERENT — CANDIDATE ROOT CAUSE** | Historical 6E.20.1 environment unrecorded (runs.json has **zero** env keys). Current: transformers 5.8.0, Python 3.14, CUDA/cuBLAS unknown. Cross-version float32 gradient-cosine drift of ~1e-2 is expected. |

**Summary of classifications:** 15 × IDENTICAL, 2 × DIFFERENT–CANDIDATE (both runtime/precision, #5 and #17), 0 × DIFFERENT–EXPLAINED, 0 × UNRESOLVED.

---

## 3. First Point Where Equivalence Can No Longer Be Demonstrated

There is **no source-level divergence**: equivalence holds through all 17 dimensions at the source/logical level. Equivalence is lost only at **runtime numerical execution** — the float32 autograd gradient and its cosine are evaluated on the physical GPU/CPU using TF32 tensor-core math and cuBLAS kernels whose results vary by torch/cuda/cuBLAS version. This is not pinnable or observable at any source line; it manifests solely in the numeric result. The historical value `0.428355` is reproducible **only** in the (unrecorded) historical environment.

---

## 4. Provenance Trace of `0.428355`

- Produced by 6E.20.1 v7 F7 `compute_cos_H1_from_live_model` (S8 readout, B0 `checkpoint_4`, eval mode, `cuda:0`) and stored in `runs.json` as `cos_H1 = 0.428355` (companion `rho_c = 1.0` is the separate analytic readout-only value, confirming `cos_H1` here is the live F7 value).
- Frozen 6E.18 machinery hash recorded in 6E.20.1 `sha256_manifest.json` = `6BCAF131…` — **matches the current frozen file exactly** → same gradient-cosine code.
- `runs.json` contains **no environment provenance** (no torch/cuda/device/seed/TF32/platform/python-version keys). The frozen reproducibility record therefore **lacks sufficient information to support a 1e-4 reproducibility requirement** — the anchor value is coupled to an unrecorded runtime.

---

## 5. Final Classification

**C — Numerical/runtime nondeterminism or environment drift.**

The two computations are the *same intended computation* (no code regression, no algorithm or data difference, identical readout/tokenization/LoRA/psdg/mode/device-at-logical-level). The `0.009654` cos_H1 divergence is attributable to float32 numerical execution differences (TF32 tensor-core rounding, cuBLAS algorithm selection, per-version kernel differences) that neither runner pins and that the frozen provenance does not record. The `1e-4` tolerance is far tighter than cross-environment float32 reproducibility of a gradient cosine permits.

*(Contributing B-aspect: the frozen record lacks environment provenance, so the historical anchor cannot be reproduced to 1e-4 from the specification alone. This reinforces C rather than changing it.)*

---

## 6. Minimum Justified Remediation Path

1. **The 6E.20.2 runner's scientific computation is correct** and source-equivalent to the historical path. The defect is in the **B0/S8 reproducibility specification** (absolute `1e-4` cos_H1 gate across environments), not in hypothesis evaluation.
2. **Re-specify, do not re-science.** Either:
   - (a) Relax/replace the anchor with an environment-robust compatibility check — e.g., an absolute cos_H1 tolerance reflecting float32 cross-environment reproducibility (~1e-2), or a relative/structural check — keeping its explicit "environment compatibility only, not evidence for H1" role; **or**
   - (b) Pin the runtime so `1e-4` becomes achievable: set `torch.backends.cuda.matmul.allow_tf32 = False`, disable cuBLAS nondeterminism, pin torch/transformers/cuda versions, record environment provenance + seed, and re-derive the anchor value from that pinned environment.
3. **No change to H1–H5 logic, checkpoints, readouts, corpus, or protocol is required.**

**Must the current L5 authorization be superseded?**
**YES — if the remediation alters the runner.** Both (a) (tolerance constant `B0_REPRODUCIBILITY_TOLERANCE`) and (b) (environment-pinning code) modify `run_phase_6e20_reevaluation.py`, which changes its SHA-256. The current L5 authorization binds runner `16BEF31E…`; it would no longer match and **must be superseded** by a new authorization binding the corrected runner hash. The runner's scientific logic does not require change — only the reproducibility specification does.

STOP.
