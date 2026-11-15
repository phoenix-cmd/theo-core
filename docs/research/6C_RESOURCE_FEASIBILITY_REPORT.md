# 6C_RESOURCE_FEASIBILITY_REPORT

**Document ID:** `docs/research/6C_RESOURCE_FEASIBILITY_REPORT.md`  
**Date:** 2026-08-25  
**Depends on:** `theo-slm-phase-6c-execution-readiness-package.md` (APPROVED, readiness stage)  
**Purpose:** Final read-only resource-feasibility check on the current machine, prior to model acquisition.  
**Execution performed:** read-only inspection only. **No model weights downloaded, no training, no model
selected, no 6D begun.** Frozen 300-GOLD dataset (`97bc6bc6…`) and repo-resident 6E.20.x source
artifacts preserved.

---

## 1. Measured machine state (read-only)

| Resource | Measured value |
|---|---|
| **Execution volume (C:) free** | **109.64 GB** (total 486.89 GB) |
| **RAM** | 16.2 GB total / **~6.0 GB free** |
| **GPU** | NVIDIA GeForce GTX 1650 — 4.29 GB total / **3.45 GB free**, driver/CUDA 12.6 |
| **CUDA / PyTorch** | torch 2.13.0+cu126, `torch.cuda.is_available() == True`, 1 device |
| **Key libraries** | `transformers`, `peft`, `accelerate`, `psutil` INSTALLED |
| **Not yet installed** | `bitsandbytes` (required for 4-bit QLoRA), `datasets` (optional) |
| **Model cache location** | `C:\Users\bs162\.cache\huggingface` (on C:, within the 109.64 GB) |
| **6E20.1 checkpoint bundle** | REMOVED (reclaimed; ~103.9 GB freed — per user, archived to cloud) |
| **300-GOLD dataset** | `adjudication.json` SHA-256 `97BC6BC6A0ADDBD35FAA77A9437F63458240C45AE7BB1668BF4C2DA01573EDE4` (intact) |

---

## 2. Comparison vs approved package §4 requirements

| Requirement (package §4) | Threshold | Measured | Verdict |
|---|---|---|---|
| **Disk free** (candidates ~5.5 GB + §8 ≥15 GB buffer) | ≥ ~20.5 GB | **109.64 GB** | **PASS** |
| **Temp/cache headroom** (transient HF download ~2 GB) | ≥ ~2 GB | 109.64 GB | **PASS** |
| **RAM** (peak QLoRA training: M1 ~1.5–2.5 GB, M2 ~1.2–2.0 GB) | ≥ 4 GB free | ~6.0 GB free | **PASS** (modest margin) |
| **VRAM** (model budget ≤2 GB; QLoRA peak M1 ~0.6–0.8 GB) | ≤2 GB model / free ≥1 GB | 3.45 GB free (4.29 total) | **PASS** |
| **M1 footprint** (QLoRA ~3 GB) | fits | fits in 109.64 GB | **PASS** |
| **M2 footprint** (QLoRA ~2.5 GB) | fits | fits in 109.64 GB | **PASS** |
| **Eval-output headroom** (<0.5 GB/candidate) | ≥1 GB | 109.64 GB | **PASS** |
| **CUDA / PyTorch** (GPU training stack) | torch+CUDA present | 2.13+cu126, CUDA avail | **PASS** |
| **QLoRA working footprint** (VRAM ~0.6–0.8 GB, RAM ~2.5 GB) | within free | VRAM 3.45 / RAM 6.0 free | **PASS** |

**Full fine-tune contingency (package §4.2: ~7.8 GB VRAM for M1) is NOT feasible on this 4.29 GB GPU.**
The QLoRA path is therefore the **mandatory** training method on this hardware; a full-FT fallback would
require CPU training (RAM ~8–9 GB — tight against ~6 GB free) or a larger GPU. This does not block the
gate, because QLoRA is the package's primary/recommended method and clears all behavioral gates.

---

## 3. Gate verdict

> **ALL RESOURCE GATES PASS.**

The execution volume now has **109.64 GB free** (cleared from the prior 12.55 GB), comfortably exceeding
the §8 ≥20.5 GB requirement. RAM (~6 GB free) and VRAM (3.45 GB free) both satisfy the QLoRA working
footprint. The CUDA/PyTorch stack is present.

---

## 4. Open environment-provisioning notes (non-blocking)

1. **Install `bitsandbytes`** before 6C.3 QLoRA 4-bit training (the 4-bit base-load backend). `transformers`
   + `peft` + `accelerate` are already present.
2. **Pin model revisions + bind hashes** at acquisition (package §3): `Qwen/Qwen2.5-0.5B-Instruct`
   (M1) and `HuggingFaceTB/SmolLM2-360M-Instruct` (M2), recorded into `6c-artifact-manifest.json`.
3. **Confirm HF cache path** is on C: (within the 109.64 GB) — default `~/.cache/huggingface`.
4. **Full-FT fallback is hardware-infeasible here** (4.29 GB GPU); do not plan full-FT on this machine.

---

## 5. Preservation confirmation

- **300-GOLD dataset:** `adjudication.json` hash `97BC6BC6…` verified **unchanged** (300 GOLD: 84 POS /
  66 ABSTAIN / 150 NEG; 44 REJECT excluded). Immutable per package constraint 1.
- **6E.20.x artifacts:** repo-resident research conclusion + scripts (`theo-core/scripts/dataset_generator/
  6E.20_FINAL_RESEARCH_CONCLUSION.md` and related) are **untouched**. The only 6E.20.x item removed was
  the bulk **checkpoint bundle** (`6E20.1_CHECKPOINTS_FOR_CLOUD_UPLOAD`, ~103.9 GB), which per the user has
  been cloud-archived; its local reclamation is what produced the freed 109.64 GB.
- **No deletion** of any dataset, source, or artifact occurred during this check.

---

## 6. Stop — await separate authorization

This is a **read-only prerequisite pass**. It confers **NO** execution authority. The following remain
**NOT performed and NOT authorized** by this document:

- ☐ Downloading model weights / tokenizer / config (M1, M2, or M3).
- ☐ Starting 6C.3 build / QLoRA training.
- ☐ Selecting a final model.
- ☐ Beginning 6D comparative evaluation.

**All resource gates pass. Present this report for the separate L5 acquisition/build authorization**
(model acquisition + 6C.3 build). Do not download weights until that authorization is granted.
