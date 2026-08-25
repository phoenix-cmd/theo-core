# THEO SLM v0 — Phase 6C Resource Feasibility Report

**Document ID:** `docs/research/theo-slm-phase-6c-resource-feasibility-report.md`  
**Date:** 2026-08-25  
**Depends on:** `theo-slm-phase-6c-execution-readiness-package.md` (APPROVED, readiness stage)  
**Purpose:** Verify the §8 resource-feasibility precondition **before** any model acquisition.  
**Execution performed:** read-only disk/manifest inspection only. **No weights downloaded, no training,
no model selected, no 6D begun. The local 6E20.1 bundle was NOT deleted.**

---

## 1. Precondition verification results

The §8 precondition chain is a strict AND. Each step is reported below.

### 1.1 — "Confirm the 6E.20.1 checkpoint archive/upload is complete" → **UNVERIFIED (FAIL)**

- The bundle at `C:\Users\bs162\Desktop\6E20.1_CHECKPOINTS_FOR_CLOUD_UPLOAD\` is, per its own `README.md`,
  a **local staging bundle "for off-machine (cloud) archival"** that "may be deleted … once the cloud
  copy is confirmed complete and hash-verified."
- **No cloud-upload-completion artifact exists** in the bundle or elsewhere reachable from this
  environment (no upload log, no remote-confirmation marker, no cloud tooling/credentials present).
- **Conclusion:** upload completeness cannot be confirmed. Step 1 of the precondition fails.

### 1.2 — "Verify the archived checkpoint data against the existing SHA-256 manifest" → **PARTIAL / INSUFFICIENT**

A recursive scan found **exactly one** `sha256_manifest.json` (in `phase-6e20.1-execution-output`).
Verification of its enumerated file entries (read-only SHA-256):

| Manifest scope | Files enumerated | Matched | Mismatched | Missing |
|---|---|---|---|---|
| `phase-6e20.1-execution-output/sha256_manifest.json` | 6 (small `step0` `.npy`) | 6 | 0 | 0 |

**Coverage gaps (critical):**
- The single manifest's `checkpoints`, `code`, and `corpus` dicts are **all empty**. It therefore does
  **not** cover the 56 `model.pt` checkpoint files (~53 GB of the 57.3 GB subdir).
- The other three subdirs — `phase-6e20.1-v6-FAILED-diskfull` (30.19 GB),
  `phase-6e20.1-v6-HALTED-scope-deviation` (16.38 GB), and `phase-6e20.1-v3-ABANDONED-output` (0 GB) —
  have **no `sha256_manifest.json` at all**. ~46.6 GB of the bundle has zero integrity record.
- **Conclusion:** only 6 of 158 files are integrity-covered; the bulk archive (incl. all model
  checkpoints) is **not independently verifiable** from the existing manifests.

### 1.3 — "Reclaim the local ~96.756 GB only after archive integrity is independently confirmed" → **NOT PERFORMED (BLOCKED)**

- Because §1.1 (upload complete) is unconfirmed **and** §1.2 (full archive integrity) is unverifiable,
  the user's own release condition is **not met**.
- **No deletion occurred.** The 103.891 GB bundle remains on `C:`. Reclaim is deferred pending a
  verified cloud copy + complete per-file manifests.

### 1.4 — "Measure actual free disk space on the execution volume after cleanup" → **12.55 GB (pre-cleanup)**

Cleanup did not occur (blocked), so the measured value is the **current** free space on `C:`:

| Volume | Total | Used | **Free (measured)** |
|---|---|---|---|
| `C:` (execution volume) | 486.89 GB | 474.33 GB | **12.55 GB** |

> The `6E20.1_CHECKPOINTS_FOR_CLOUD_UPLOAD` bundle occupies **103.891 GB / 158 files** on this same
> volume (README claims 96.756 GB / 157 files — a **discrepancy of +7.13 GB / +1 file** worth flagging;
> the README total is stale/inaccurate).

---

## 2. Candidate-specific storage requirements (from approved package §4)

All figures are the package's planning estimates, re-stated for the gate decision. Training method
default = **QLoRA** (keeps VRAM/RAM within §F budgets).

| Component | M1 (Qwen2.5-0.5B) | M2 (SmolLM2-360M) | Notes |
|---|---|---|---|
| Base model download (safetensors) | ~1.0 GB | ~0.75 GB | pinned revision + SHA-256 |
| QLoRA adapters (≤3 ckpts) | <0.2 GB | <0.15 GB | |
| Evaluation artifacts | <0.5 GB | <0.5 GB | predictions/logs/reports |
| **Per-candidate total (QLoRA)** | **~3.0 GB** | **~2.5 GB** | package §4.3 |
| Peak training VRAM | ~0.6–0.8 GB | ~0.5–0.7 GB | ≤2 GB §F target — **OK** |
| Peak training RAM | ~1.5–2.5 GB | ~1.2–2.0 GB | ≤4 GB §F target — **OK** |

**Working-space components beyond the two candidates:**

| Item | Estimate |
|---|---|
| M1 + M2 base-model + HF caches (tokenizer/config/safetensors) | ~1.5 GB |
| Transient download/extraction peak (HF temp before cache move) | ~2.0 GB |
| Combined eval outputs (both candidates) | <0.5 GB |
| **Subtotal working need (both, sequential)** | **~10–12 GB** |
| **Safety margin (package §8: ≥15 GB buffer)** | **≥15 GB** |
| **Required free space to PASS §8** | **≈ 20.5 GB** |

VRAM, RAM, and inference-latency criteria are **met** (≤2 GB VRAM, ≤4 GB RAM, ≤0.5 s/case GPU). The only
binding constraint is **disk free space**.

---

## 3. Resource gate determination

| Scenario | Free space | Meets §8 (≥ ~20.5 GB)? |
|---|---|---|
| **Current (no reclaim — actual measured)** | **12.55 GB** | **NO — FAIL** |
| Projected post-reclaim (12.55 + 103.891) | 116.44 GB | YES (would pass) |

**Gate verdict: NOT PASSED (BLOCKED).**

- The execution volume has **12.55 GB free**, below the §8 requirement of ≈20.5 GB (candidates
  ~5.5 GB + ≥15 GB buffer). Running both M1 and M2 acquisition + QLoRA training + evaluation within
  12.55 GB is **not safely feasible** (transient download peak + dual caching would overrun).
- The reclaim that would resolve this is itself **blocked** (§1.1/§1.2 unmet). Therefore the gate cannot
  pass in its current state.

---

## 4. Remediation path to pass the gate

To reach a passing resource gate, one of the following must occur **before** any model acquisition:

1. **Complete the cloud upload** of the 6E20.1 bundle (external action, out of scope for this step).
2. **Generate complete per-file SHA-256 manifests** covering **all 158 files across all four subdirs**
   (current manifests omit `model.pt` checkpoints and are absent for 3 subdirs).
3. **Verify the cloud copy** against those complete manifests (the README's stated post-upload step).
4. **Only then reclaim** the local ~103.9 GB → frees the volume to ~116 GB, comfortably clearing §8.
5. **Re-run this feasibility check**; the gate should then pass and a separate human execution
   authorization for model acquisition can be issued.

*Alternative if upload is deprioritized:* explicitly accept single-copy risk, **or** reduce 6C scope to
fit 12.55 GB — evaluate **M1 only** (drop M2), use INT4/GGUF (not FP16) to cut the base download ~4×,
and avoid dual caching (sequential, single-model at a time). This still requires a human decision and a
re-run of this check; it does not by itself satisfy the "archive integrity independently confirmed"
precondition for deletion.

---

## 5. Execution authorization boundary — explicit non-authorization

This report is a **read-only prerequisite check**. It confers **NO** execution authority. The following
remain **NOT performed and NOT authorized** by this document:

- ☐ Downloading any model weights / tokenizer / config (M1, M2, or M3).
- ☐ Starting 6C.3 build / training / fine-tuning (including QLoRA).
- ☐ Selecting a final model.
- ☐ Beginning 6D comparative evaluation.
- ☐ Deleting / reclaiming the 6E20.1 local bundle (blocked per §1.3).

**STOP — resource gate NOT PASSED.** A separate human execution authorization for model acquisition
should be issued **only after** the gate passes (reclaim completed under verified upload, or an
explicit scoped-risk decision), at which point this report is re-run to confirm the measured free space.
