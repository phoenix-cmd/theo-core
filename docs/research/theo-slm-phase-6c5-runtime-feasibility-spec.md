# 6C.5 Cloud Runtime Feasibility — Specification

**Date:** 2026-08-27
**Purpose (sole):** determine whether the required execution environment is *technically obtainable*
on the existing approved Kaggle GPU/provider path. This is a read-only probe — it performs **no training**
and **no dataset mutation**. The scientific experiment, training specification, and all frozen artifacts remain
untouched. The v9 smoke-gate evidence is preserved.
**Deliverables:** `feasibility_6c5_kaggle.py` (test) + this spec + independent audit + execution authorization.
**No live execution yet.**

---

## 1. Required runtime (bound; identical to `cloud_6c5.py` PINNED)
| Component | Required |
|---|---|
| Python | **CPython 3.14.3** |
| torch | **2.13.0+cu126** |
| CUDA runtime | **12.6** |
| transformers | **5.8.0** |
| peft | **0.20.0** |
| bitsandbytes | **0.50.1** |

The pins are duplicated deliberately in `feasibility_6c5_kaggle.REQUIRED` and asserted equal to
`cloud_6c5.PINNED` by an offline test, so the feasibility gate and the training gate cannot drift.

## 2. What the test does (in order)
1. Runs on the Kaggle GPU image (base interpreter is **Python 3.12**) **without replacing it**.
2. Builds/installs an **isolated CPython 3.14.3** environment: conda → pyenv → official source build.
3. Installs the EXACT pinned dependencies into that isolated environment.
4. Re-executes a capture payload under 3.14.3 to record the realized stack:
   Python, torch, CUDA runtime, NVIDIA driver, Transformers, PEFT, bitsandbytes, the **actual**
   bitsandbytes package SHA-256, Linux/glibc.
5. Verifies GPU accessibility and QLoRA prerequisites (bitsandbytes 4-bit importable on CUDA).
6. Performs a **minimal import/runtime smoke only** — ZERO model training, ZERO dataset mutation.
7. Emits a **machine-readable compatibility manifest** (`6c5_runtime_feasibility_manifest.json`).
8. **FAILS CLOSED** (non-zero exit) if any required version cannot be reproduced exactly.

## 3. Verdict semantics (must distinguish, never silently substitute)
- `BASE_IMAGE_IS_3_12` — informational: the Kaggle base itself uses Python 3.12. EXPECTED; does **not**
  by itself make the run infeasible. The isolated 3.14.3 environment decides feasibility.
- `FEASIBLE` — isolated Python 3.14.3 + all pins reproduced exactly (base 3.12 is fine).
- `INFEASIBLE` — required stack could not be reproduced; exact reason recorded.

The manifest records `base_image_python` and `base_image_is_3_12` separately from the realized
`python` so reviewers can see the distinction. **3.12 is never written into `REQUIRED`.**

## 4. Hard stopping rule (set now)
```
Kaggle Python 3.14.3 feasibility test
        ↓
  PASS  → environment considered for a fresh L5 training authorization
  FAIL  → ABANDON the Kaggle path for this experiment
```
No ninth / tenth / eleventh provider workaround. If `INFEASIBLE`, document the exact incompatibility
and stop — do **not** relax the pins.

## 5. Notes
- The feasibility kernel sets `enable_internet = True` (required to build the isolated interpreter and
  pull wheels). This is a one-time read-only probe, NOT the training run (training stays internet-off).
- The kernel uses a dedicated slug `theo-6c5-bc-feasibility`, distinct from the smoke and training kernels;
  prior evidence (failed training kernel, v9 smoke PASS) is preserved.
- `cloud_6c5.py` is NOT modified; this script is an independent, orchestration-only artifact.
