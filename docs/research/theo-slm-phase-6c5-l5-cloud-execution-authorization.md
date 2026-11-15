# 6C.5 L5 Cloud Execution Authorization — v13 (post Remediation #15)

**Document:** `theo-slm-phase-6c5-l5-cloud-execution-authorization.md`  
**Date:** 2026-08-28  
**Supersedes:** L5 v12.  
**Status:** **AUTHORIZED FOR LIVE PATH VALIDATION KERNEL FIRST (`validate_cloud_paths_6c5_kaggle.py --execute`). TRAINING KERNEL AUTHORIZATION CONDITIONAL ON LIVE PATH VALIDATION PASS.**

---

## 1. Execution Protocol & Prerequisites

1. **Pre-Training Gate (Mandatory):**
   - Execute `py -3.14 validate_cloud_paths_6c5_kaggle.py --execute` to push the dedicated path-validation kernel.
   - The path validator kernel must run on Kaggle GPU, execute pre-resolution diagnostics, locate `/kaggle/input/datasets/anakinistic/theo-6c5-inputs` via the bounded tree resolver (depth <= 4), verify bundle root (depth <= 3), verify `bundle_manifest.json` identity (`454cfc42...`), verify file existence, verify frozen scientific SHA-256 bindings, verify `cloud_6c5.py` SHA (`c03f5d0e...`), execute `cloud_6c5` import smoke test, and emit `[PATH_VALIDATION_PASS]`.
   - If the path validation kernel fails -> **HALT**. No training kernel may be submitted.
2. **Training Execution (After Path Validation PASS):**
   - Execute `py -3.14 launch_6c5_kaggle.py --bundle C:\Users\bs162\Desktop\THEO\6c5_cloud_bundle --execute`.
   - Exactly one Architecture B run and one Architecture C run (**B×1, C×1**).
   - No auto-retry, no sweeps, no architecture mutation, no dataset mutation.
3. **Retrieval & Local Evaluation:**
   - Execute `py -3.14 retrieve_6c5_kaggle_results.py --execute` followed by `py -3.14 verify_6c5_cloud_results.py`.
   - Evaluation performed strictly locally on GTX 1650.

---

## 2. Cloud Pre-Execution Flow (Remediation #15 Bounded Resolver & Smoke Gate)

1. Unconditionally print `/kaggle/input` recursive tree diagnostics up to depth 4 with type metadata (`print_mount_diagnostics`).
2. Identify the single matching dataset mount under `/kaggle/input` matching owner `Anakinistic` and slug `theo-6c5-inputs` (`locate_dataset_mount`, depth <= 4).
3. Search ONLY within the matched dataset mount (depth <= 3) for directories containing `theo-core/`, `theo-data/`, and `bundle_manifest.json` (`resolve_cloud_bundle_root`).
4. Verify `bundle_manifest.json` contains authorized `bundle_sha256` (`454cfc42f70b1457cf8837f8bd55c392e443f572640d6d26114d736e14b6ed31`).
5. Output diagnostic path manifest (`DATASET_MOUNT_ROOT`, `RESOLVED_THEO_ROOT`, `THEO_CORE`, `THEO_DATA`, `experiment_spec.py`, `binding_manifest`, `cloud_6c5.py`).
6. Pre-import verify required files (`experiment_spec.py`, `6c5_binding_manifest.json`, `cloud_6c5.py`) and verify `cloud_6c5.py` SHA-256 (`c03f5d0e...`).
7. Set `THEO_ROOT = RESOLVED_THEO_ROOT` and derive all paths strictly from the resolved root.
8. Import `cloud_6c5` and execute training pipeline.

---

## 3. Pinned Runtime Environment (Proven Feasible)

| Component | Pinned Baseline | Verification Gate |
|---|---|---|
| Python | `3.14.3` | `sys.version.split()[0] == '3.14.3'` |
| PyTorch | `2.13.0+cu126` | `torch.__version__ == '2.13.0+cu126'` |
| CUDA Runtime | `12.6` | `torch.version.cuda == '12.6'` |
| Transformers | `5.8.0` | `transformers.__version__ == '5.8.0'` |
| PEFT | `0.20.0` | `peft.__version__ == '0.20.0'` |
| bitsandbytes | `0.50.1` | `bitsandbytes.__version__ == '0.50.1'` |
| Linux bnb Package SHA | `fdddb68ee4f6906045d1c211a01ede71e23582f650274247fbdc94af35d88eaa` | Exact dir_hash match |
| Host Environment | Linux / glibc 2.35, Tesla P100-PCIE-16GB | Proven in L5 feasibility probe |

---

## 4. Frozen Scientific Bindings & Input Hashes

| Variable / File | Frozen Value / SHA-256 |
|---|---|
| Adjudication / Dataset SHA | `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4` |
| Fixture SHA (66-case corpus) | `f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457` |
| Split Manifest SHA | `eda4cb6695b6651a7cef8bdee0a7a990f4f799a6bf9148269ee758d0704ee232` |
| Canonical Prompt Hash | `a29e12a2839b106d3a3632e1f353961d74182245a41602b445d06f71abb097ba` |
| M1 Model Revision | `7ae55760` |
| M1 `model.safetensors` | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| M1 `config.json` | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| Training Random Seed | `20260826` |
| AUX_LOSS_WEIGHT | `1.0` (locked; auxiliary evidence head for Architecture C) |
| Cloud Bundle SHA-256 | `454cfc42f70b1457cf8837f8bd55c392e443f572640d6d26114d736e14b6ed31` |

---

## 5. Orchestrator & Science Source Code Hashes (Post-Remediation #15)

| File | SHA-256 |
|---|---|
| `kaggle_6c5_common.py` | `c726ad1fcc9c68cda2887977fdaeee0076b9fbb131f821cb032ba43976ff6ec4` |
| `launch_6c5_kaggle.py` | `2c7fed53b859a591b1289d6c290e1231744d0bc0a7ae4ce30e01e923d6466fd0` |
| `validate_cloud_paths_6c5_kaggle.py` | `3ad7f6ce5034c744dc47fc28dc579b7b4afbfb9821ffe471ea664f53493d7b38` |
| `cloud_6c5.py` | `c03f5d0eef95187225c5126621ceea414e719b1f1a58cc4f50d1751792bb4071` |
| `prepare_6c5_cloud_bundle.py` | `d8e4375488b042c0eb527545d898b3b3bfaa1543016c549807e38054b96b6a1e` |
| `smoke_test_6c5_kaggle.py` | `ccfd8d78ba180e875809b518904278b16d7c6ceddea07e35b446725450827671` |
| `feasibility_6c5_kaggle.py` | `1013d1174e2c060a42b763f824be94bed25b566d4f3f0361c05cb5aa11b103b8` |
| `test_6c5_orchestrator.py` | `2bf66e0b44843753b70fb4683a6556fb2c8f2a9ebc6fd6235a8857584e3f064c` |
| `experiment_spec.py` | `dfe8ed5df1ca16fe7da883c77573e69e810dce67e111025e03d83da05409b8b3` |
| `train_6c5.py` | `b1b5e9ca4b4917316d7d347addec67f8a7643c251437b54eae240c2b65a6c05d` |
| `phase2_audit.py` | `8bab7bd12162162dc87160ec63ee6f5d482807777474e93459cf3462ee40144f` |
| `amendment_audit.py` | `ab750ddde5d4520448552f66a3e5d6ff677577cd70a80bb5d75c4dad6db81ee2` |
| `bind_6c5.py` | `fc2c69efdf052b1893417a990997fb67922610e2f774f2fca6cab440252a185a` |
| `eval_6c5.py` | `f5c028b159f8c1584157f042a9b80820edc57c2d7703d5b62f45ae1487d0ee37` |
| `verify_6c5_cloud_results.py` | `cfa620413ee1fbf14c183d5fc51ac77c7f02b0e401546869da731a07d6bdca30` |
| `retrieve_6c5_kaggle_results.py` | `cc95e0abd9e280fbdc8fd9d560c0fa37b2af1df68a4c7c42d1ff3044e65b83b9` |
| `requirements.txt` | `ddc24d6ba1219765f3cd6ec5a3d737dee54c5c4b2632c67e9dbbe620c1f13360` |
| `arch_b/b_architecture.py` | `b2b432ff63e4c70be3a2d7504d03215ddf159a432af1c1de0e135f8d702914e8` |
| `arch_c/c_architecture.py` | `cbabdd916ec43eff11af89cf44a9ead2104629420189ccfdf256e1d50250b90d` |

---

## 6. Verification Status

- Offline Orchestration Test Suite: **173/173 PASSED** (`test_6c5_orchestrator.py`)
- Scientific Gate Audit: **16/16 PASSED** (`amendment_audit.py`)
- Provider Smoke Test: **PASSED**
- Runtime Feasibility Probe: **FEASIBLE**
