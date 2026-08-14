# THEO SLM v0 — Phase 6E.2-R Independent Integrity Audit (v1)

**Phase:** 6E.2-R (integrity audit of the Phase 6E.2 real LoRA training experiment)
**Date of audit:** 2026-08-11 (UTC)
**Audit mode:** Anti-simulation, read-only. No training, no evaluation, no tuning, no deployment, no data modification.
**Machine-readable artifacts:** `theo-data/datasets/theo_slm_v0_artifacts/phase-6e2r/`
**Verdict:** **HOLD**

---

## 1. Executive Summary

The Phase 6E.2 adapter checkpoint located at
`theo-data/datasets/theo_slm_v0_artifacts/phase-6e2/adapter_checkpoint/` is **genuine and real**.
Every functional property that can be independently verified was verified, including:

- File hashes (`adapter_model.safetensors` SHA-256 `d4a32b87…` matches all recorded manifests).
- Weight-tensor structure (336 LoRA tensors; measured trainable parameters **8,798,208**, exactly matching the manifest).
- Dataset split reconstruction (212/52, 198 families, zero family leakage, hash-identical split).
- Training log completeness (135 monotonic steps across 5 epochs; validation step alignment).
- Input/target projection (no forbidden metadata; all concept ids grounded; all targets valid JSON).
- Environment reverification (live environment matches the recorded training environment, including `peft 0.20.0`).
- **Token-exact reload reproduction**: two independent fresh processes reproduced the recorded 33-token greedy smoke-test sequence (hash `c944d54c…`) from the on-disk base + adapter.
- Isolation: frozen corpus hash `a7b4e845…` unchanged; benchmark/probe instruments untouched by 6E.2.

The verdict is **HOLD** (not PASS) because the Phase 6E.2 **report document** contains four
documentation-level discrepancies (D1–D4 below) and the training script carries a
docstring process-overclaim (it states it verifies benchmark/probe immutability but the code
does not). None of these discrepancies invalidate the trained artifact itself; they must be
corrected and re-committed before the phase is declared clean.

---

## 2. Audit Scope, Authority, and Constraints

- **Scope:** Everything the Phase 6E.2 run claimed to produce: the adapter checkpoint, all eleven
  JSON manifests, the training log, validation logs, smoke/reload test artifacts, the executed
  training script, the git history, and the immutability of the frozen corpus and evaluation instruments.
- **Authority:** This audit is authorized to read, hash, recompute, and reload — but **not** to train,
  evaluate on the 51-case benchmark or 15-case probe, tune thresholds, optimize hyperparameters,
  modify the corpus, change labels, generate synthetic data, deploy, or begin 6E.3.
- **Evidence rule:** Phase 6C is treated as **INVALIDATED / non-evidentiary**. No claim in this
  report rests on 6C metrics, 6C checkpoints, or simulated claims.
- **Classification scheme:** every finding is classified as
  `VERIFIED BY INDEPENDENT EXECUTION`, `VERIFIED STATICALLY`, `NOT VERIFIED`, `DISCREPANCY`, or `NOT APPLICABLE`.

---

## 3. Methodology and Anti-Simulation Measures

1. **Independence:** every number was recomputed by scripts written for this audit
   (`phase-6e2r/scripts/*.py`), not copied from 6E.2 manifests.
2. **Live code reuse:** the split was re-derived by executing the same algorithm as
   `split_dataset_generated` / `split_dataset_grouped` used in `run_phase_6e2_real_training.py`
   (grouped by seed-family, 80/20, `torch.randperm(seed=42)`), ensuring exact algorithm parity.
3. **Content hashing:** all artifacts hashed byte-for-byte (SHA-256) from disk.
4. **Fresh-process reload proof:** two independent OS processes each cold-loaded the base model and
   the on-disk adapter and greedily generated on the dev-record prompt; outputs compared token-for-token
   against the recorded smoke test.
5. **Immutability evidence:** git history (recent commits for the whole repo and for the
   `evaluation/` subtree), file mtimes, and content hashes were used to prove the corpus and
   evaluation instruments were never modified by 6E.2.
6. **Discrepancy handling:** discrepancies are reported exactly as found and are **not silently repaired**.

---

## 4. Evidence Chain: Phase 6E.1 Base-Model Provenance

- Repo `Qwen/Qwen2.5-0.5B-Instruct`, revision **`7ae557604adf67be50417f59c2c2f167def9a775`**.
- `model.safetensors` SHA-256 **`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`**
  (988,097,824 bytes; matches the Hub LFS OID).
- Measured parameters **494,032,768**; tokenizer `Qwen2Tokenizer`, vocab 151,665, round-trips 4/4 exact.
- Snapshot: `C:\Users\bs162\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae55760…`.
- Status: Phase 6E.1 PASS (9/9). This chain is the authorized base for 6E.2.

**Classification:** VERIFIED BY INDEPENDENT EXECUTION (Phase 6E.1 acquisition audit).

---

## 5. Adapter Checkpoint Integrity

Recomputed SHA-256 and sizes from disk (`adapter-integrity.json`):

| File | Size (bytes) | SHA-256 | Reported | Match |
|---|---|---|---|---|
| `adapter_config.json` | 1,314 | `355dd497…70dcd55` | same | ✅ |
| `adapter_model.safetensors` | 35,237,104 | `d4a32b87…5af325517` | same | ✅ |
| `README.md` | 5,402 | `90bf2f3f…441afb` | same | ✅ |

`adapter_model.safetensors` holds **336 tensors** (7 modules × 24 layers × A/B).
Independent tensor-shape summation yields **8,798,208** trainable parameters, exactly matching the
experiment manifest. Shapes are consistent with the real base model:
`q/o_proj` 896, `k/v_proj` output 128 (GQA with `num_key_value_heads=2`, head_dim 64), and
`gate/up/down_proj` 4864. Breakdown: attention 2,162,688 (A 16×896 ×96, B 896×16 ×48, B 128×16 ×48),
MLP 6,635,520 (A 16×4864 / B 4864×16 / down B 896×16). Sum **8,798,208**.

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 6. Adapter Configuration Contract

`adapter_config.json` (PEFT 0.20.0 format): `peft_type=LORA`, `r=16`, `lora_alpha=32`,
`lora_dropout=0.05`, `bias=none`, `task_type=CAUSAL_LM`, `use_dora=false`, `use_rslora=false`,
`inference_mode=true`, and target modules exactly the seven {`q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`}. `base_model_name_or_path` pins the 6E.1 revision
`7ae55760…`. All six contract checks pass.

**Classification:** VERIFIED STATICALLY (config semantics read from the artifact).

---

## 7. Dataset Split Reconstruction

Re-executed the split algorithm as coded in the training script (`split-reconstruction.json`):

- Total **264** records, **198** seed-families (176 single-record families + 22 four-record `_A–_D` families).
- Train **212** (158 families), dev **52** (40 families) — matches every claimed count.
- Family overlap train/dev = **0** (zero leakage); every record assigned exactly once; 264 unique case ids.
- Split identity hash recomputed = `d186b3718c2b32beaceb328e903f8be700a94c995af76ac79561406f6a3a0e80`
  **matches** `dataset-split-manifest.json`.

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 8. Training Input Projection Audit

For all **264** records, the prompt was reconstructed with the training script's prompt builder and
scanned for forbidden metadata tokens (benchmark/probe labels `GOLD_*`, reviewer metadata, generator
metadata, gold-target leakage):

- Forbidden hits across all prompts: **0** (`training-input-audit.json`).
- Every prompt concept id present in the record's `grounding_snapshot.concept_ids`: **True** (264/264).
- Corpus label distribution: SHOULD_PROPOSE 67, SHOULD_ABSTAIN 197 (unchanged, corpus-intrinsic).

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 9. Training Target Audit

All 264 targets parsed as valid JSON with contract fields `decision` + `reasoning`
(`training-target-audit.json`): **100% valid**. The target prompt structure matches the recorded
`expected_completion` format used during training (`decision`/`reasoning`, matching corpus labels).

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 10. Training Log Verification

`training.log` contains **135** JSONL lines (one per global step): `global_step` runs monotonically
1→135, 5 epochs (27 steps/epoch), per-step losses present, timestamps strictly increasing
(`training-log-verification.json`). Timestamp span **1,600.4 s**, consistent with a real GPU run on
the GTX 1650 (4 GB) and with the manifest's claimed 1,656.21 s execution window
(14:25:55Z environment capture → 14:53:31Z checkpoint save).

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 11. Validation Log Cross-Consistency

`validation-logs.json` records 5 validation checkpoints at global steps **27 / 54 / 81 / 108 / 135**
— exactly the final step of each epoch in `training.log`. Train loss: 0.4739 → 0.0513 → 0.0338 →
0.0296 → 0.0259. Dev loss: 0.1064 → 0.0516 → 0.0375 → 0.0386 → 0.0333. Per-epoch durations
327.33–331.60 s, coherent with the log span. Loss curves are plausible monotone-decreasing
(dev loss shows mild, non-divergent fluctuation).

**Classification:** VERIFIED STATICALLY (cross-referenced against the training log).

---

## 12. Environment Reverification

Live environment (fresh process, same interpreter): Python 3.14.3, torch 2.13.0+cu126,
transformers 5.15.0, peft 0.20.0, accelerate 1.14.0, huggingface_hub 1.27.0 — **all match** the
6E.2 `environment-manifest.json`. `adapter_config.json` `peft_version=0.20.0` equals the live peft
version (`environment-reverification.json`: `all_matched=True`).

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 13. Fresh-Process Reload Reproducibility

Two independent OS processes cold-loaded the pinned base model + on-disk adapter, then greedily
generated on the dev-record prompt (case `td://v0/household/case_004_A`):

- Round 1 tokens **exactly** equal the recorded 33-token sequence; token hash `c944d54c…` matches.
- Round 2 tokens **exactly** equal Round 1 (deterministic across independent processes).
- Decoded output: `{"decision": "SHOULD_ABSTAIN", "reasoning": "Epistemic thresholding triggered:
  insufficient evidence or distractor pattern detected."}` — identical to the recorded smoke test.
- The recorded `expected_completion` references `concept://house/sink`, which is among the grounded
  concepts of the actual smoke-test record — further confirming which record was used.

Note: in PEFT inference mode all LoRA weights have `requires_grad=False`, so
`num_parameters(only_trainable=True)` reports 0 in the reloaded model; the authoritative trainable
count (8,798,208) is measured from the safetensors metadata (§5). This is an introspection-mode
artifact, not a weight discrepancy.

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 14. Isolation & Instrument Immutability Audit

- Corpus `candidate_records.json`: recomputed SHA-256 **`a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0`** — **unchanged**.
- 51-case benchmark instrument (`theo_core/evaluation/benchmarks/`, `ALL_CASES`): last touching
  commit **`06d253a`** (predates 6E.2). 6E.2 commits (`0b0e0e4`, `38a5bd1`) touched only docs and the
  training script — never `theo_core/evaluation/`.
- 15-case probe cases + 6A.1/probe result JSONs present with pre-6E.2 mtimes; content hashes recorded.
- No phase-6e2 artifact references the benchmark or probe (zero hits), and the training script
  contains no code that reads them.

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 15. Git Provenance Audit

- `theo-core` working tree: **clean**; HEAD `38a5bd1` (2026-08-11 20:26:02 +0530) adds exactly
  `run_phase_6e2_real_training.py` + `theo-slm-phase-6e2-real-training-v1.md`; prior commit `0b0e0e4`.
- `theo-providers` working tree: **clean**; `a738e6f` (20:26:08 +0530) adds `peft>=0.14.0`.
- The two commits postdate the training window (checkpoint saved 14:53:31Z ≈ 20:23 IST) — i.e. the
  script and docs were committed after execution, which is consistent with a real (non-simulated) run.

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 16. Unexpected Artifact Scan

Workspace-wide scan found only the expected artifacts: the 6E.2 adapter safetensors, the pinned base
model cache (`models--Qwen--Qwen2.5-0.5B-Instruct`), and the earlier-phase `models--Qwen--Qwen3-0.6B`
cache. No stray `.pt`/`.pth`/`.bin`/`.gguf` checkpoints, no extra `adapter_checkpoint/` directories,
no cached datasets.

**Classification:** VERIFIED BY INDEPENDENT EXECUTION.

---

## 17. Discrepancies and Non-Evidentiary Findings

### Discrepancies (documentation-level; do not invalidate the artifact)

| ID | Finding | Impact |
|---|---|---|
| **D1** | Report text states **130 steps / 26 per epoch**; the manifest and `training.log` record **135 steps / 27 per epoch**. | Documentation only. Artifact evidence is self-consistent at 135/27. |
| **D2** | Report text states **Total Training Execution Time 1,645.85 s**; `experiment-manifest.json` records **1,656.21 s**. | Documentation only. Independent log span (1,600.4 s) bounds both values. |
| **D3** | Report §7 smoke-test prompt displays a **medical train-split record** (`case_001_A`); the recorded artifact used the **smoke-detector dev-split record** (`case_004_A`). | Reporting error in the report text; the recorded artifact is genuine (reproduced token-for-token, §13). |
| **D4** | Training script **docstring** claims it verifies the 51-case benchmark & 15-case probe exist unchanged and that inputs contain no `GOLD_*`/reviewer/generator metadata; the **code implements no such checks** (only corpus/base/adapter/token hashing). | Documentation overclaim. Instrument immutability is independently established by this audit (§14). |

### Non-evidentiary findings

- **Phase 6C (simulated 51-case benchmark + 15-case probe, "pass" claims): INVALIDATED / non-evidentiary.**
  No 6C metric, checkpoint, or claim is used as evidence in this audit. The Phase 6E.2 artifacts and
  this audit stand entirely on real on-disk evidence.
- The reported benchmark/probe "SHA-256 verified unchanged" in the 6E.2 report is not backed by
  training-script execution (D4); it is, however, independently confirmed by this audit.

---

## 18. Verdict and Recommendation

**VERDICT: HOLD**

The Phase 6E.2 LoRA adapter is **real**: 17/19 objectives verified by independent execution, 2/19
verified statically, with **17 PASS / 2 PASS-static / 0 functional failures**. The artifact chain is
coherent end-to-end (base model → split → inputs → targets → training log → validation → environment
→ checkpoint → token-exact reload). No functional or data-integrity defect was found.

HOLD is returned because the phase is not yet *clean*: the 6E.2 report contains four documentation
discrepancies (D1–D4) and the training script overclaims a check it does not perform. These are
repair-and-recommit items, not retraining items.

**Recommendations (require human authorization before execution):**
1. Correct the 6E.2 report text (steps 130→135, per-epoch 26→27, execution time, smoke-test prompt to
   the actual `case_004_A` record) and recommit.
2. Align the training-script docstring with actual behavior, or implement the claimed
   benchmark/probe immutability check, and recommit.
3. After the corrective commit(s), re-run only the static re-read of the affected JSONs; no re-training
   is warranted by this audit.
4. Do **not** begin 6E.3 or any evaluation until the corrected phase is re-authorized.

**End of audit.**
