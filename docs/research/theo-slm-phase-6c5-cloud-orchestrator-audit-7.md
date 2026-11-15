# 6C.5 Cloud Orchestrator — Source Audit #7 (Provider-Failure #7 Remediation)

**Date:** 2026-08-27
**Scope:** Remediation of provider-failure #7 — `AttributeError: 'KaggleApi' object has no attribute 'kernel_metadata_get'`
during live smoke metadata verification. This is an SDK/API integration defect, not a scientific or training failure.
**Author:** opencode (autonomous remediation, supervised)
**Status:** AUDIT PASSES — pending fresh L5 v8 authorization before any provider execution.

---

## 1. Root Cause (Provider-Failure #7)

The live smoke test progressed to provider metadata verification but crashed locally with:
`AttributeError: 'KaggleApi' object has no attribute 'kernel_metadata_get'`.

The installed Kaggle CLI/SDK is **2.2.4**. It exposes kernel metadata **only** through
`kernels_pull(kernel, path, metadata=True)`, which downloads `kernel-metadata.json` into `path`
and returns `path`. That emitted metadata includes `id`, `title`, `language`, `kernel_type`,
`is_private`, `enable_gpu`, `enable_internet`, `dataset_sources`, `kernel_sources`,
`competition_sources`, etc. There is **no** `kernel_metadata_get` method (nor any single-object
metadata getter). The smoke verifier wrongly assumed `api.kernel_metadata_get(full)` existed.

## 2. Remediation Applied

### 2.1 `smoke_test_6c5_kaggle.py` — `query_kernel_metadata`
- Removed the unsupported `api.kernel_metadata_get(full)` call.
- Now uses the provider-supported operation: `api.kernels_pull(full, path=tempfile.mkdtemp(), metadata=True)`
  then reads the downloaded `kernel-metadata.json`.
- Added `_norm_slug(s)` normalization: Kaggle normalizes owner/dataset slugs to **lowercase** in
  kernel metadata, so the dataset-source and title/slug comparisons are slug-normalized (case-insensitive)
  to avoid false failures from provider casing.
- Added a **clean HALT path**: any exception during retrieval (HTTP/permission/serialization) calls
  `C.halt(...)` so verification never silently proceeds to training.
- New hash: `542961411672d71330e39de71b625a427ff08ba2b1592e80043df4be1011bb8e`

### 2.2 `test_6c5_orchestrator.py`
- `test_smoke_uses_supported_sdk_metadata_call`:
  - Hard guard: fails if the string `kernel_metadata_get` appears anywhere in `smoke_test_6c5_kaggle.py`
    (prevents regression to the non-existent method).
  - Faithful mock of the SDK contract (`kernels_pull(path, metadata=True)` writes
    `kernel-metadata.json` with the same shape the real SDK emits, including lowercased dataset slugs),
    asserting the verifier (a) calls `kernels_pull` with `metadata=True`, (b) queries the SAME canonical
    slug, and (c) `verify_provider_metadata` PASSES against SDK-shaped metadata.
  - Made deterministic by explicitly setting `C.KAGGLE_DATASET_OWNER = "Anakinistic"` (restored after).
- `test_smoke_metadata_retrieval_failure_halts`: asserts a retrieval failure HALTS (no training proceeds).
- Suite: **55 → 60 checks, all PASS** (fully offline, no Kaggle contact).
- New hash: `8109b82e18a6491309b9aa497be26fbcc29226a4036c5b765cbf65ef1a0ab7f9`

## 3. Source Hashes — Post-Remediation #7

### 3.1 Orchestrator files (changed in remediation #7 marked *)
| File | SHA-256 (prefix) | Status |
|---|---|---|
| `kaggle_6c5_common.py` | `1648f332e80528a62504ffbcfa8a4cb535a7f4637ff83075a183ea23e798c050` | unchanged (#5) |
| `prepare_6c5_cloud_bundle.py` | `d8e4375488b042c0eb527545d898b3b3bfaa1543016c549807e38054b96b6a1e` | unchanged |
| `launch_6c5_kaggle.py` | `57fe359fec61b2e589d0af09312699dbe50c2b1d1c0749fc5d815a95c400bd45` | unchanged (#6) |
| `retrieve_6c5_kaggle_results.py` | `cc95e0abd9e280fbdc8fd9d560c0fa37b2af1df68a4c7c42d1ff3044e65b83b9` | unchanged |
| `verify_6c5_cloud_results.py` | `cfa620413ee1fbf14c183d5fc51ac77c7f02b0e401546869da731a07d6bdca30` | unchanged |
| `amendment_audit.py` | `ab750ddde5d4520448552f66a3e5d6ff677577cd70a80bb5d75c4dad6db81ee2` | unchanged |
| `test_6c5_orchestrator.py` | `8109b82e18a6491309b9aa497be26fbcc29226a4036c5b765cbf65ef1a0ab7f9` | **CHANGED (#7)** |
| `smoke_test_6c5_kaggle.py` | `542961411672d71330e39de71b625a427ff08ba2b1592e80043df4be1011bb8e` | **CHANGED (#7)** |

### 3.2 Scientific / architecture files (UNMODIFIED across remediations #3–#7)
Recorded as the frozen scientific binding (verified unchanged from audit‑6).
| File | SHA-256 (prefix) |
|---|---|
| `arch_b/b_architecture.py` | `b2b432ff63e4c70be3a2d7504d03215ddf159a432af1c1de0e135f8d702914e8` |
| `arch_c/c_architecture.py` | `cbabdd916ec43eff11af89cf44a9ead2104629420189ccfdf256e1d50250b90d` |
| `phase6c5/experiment_spec.py` | `dfe8ed5df1ca16fe7da883c77573e69e810dce67e111025e03d83da05409b8b3` |
| `phase6c5/cloud_6c5.py` | `56fd6f6aaa68ef9ae238efccb3735cf8e275fa68b6b7816a0c54e358cd48198f` |
| `phase6c5/train_6c5.py` | `b1b5e9ca4b4917316d7d347addec67f8a7643c251437b54eae240c2b65a6c05d` |
| `phase6c5/eval_6c5.py` | `f5c028b159f8c1584157f042a9b80820edc57c2d7703d5b62f45ae1487d0ee37` |
| `phase6c5/phase2_audit.py` | `8bab7bd12162162dc87160ec63ee6f5d482807777474e93459cf3462ee40144f` |

### 3.3 Frozen inputs (unchanged)
- Adjudication `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4`
- Fixture `f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457`
- Split `eda4cb6695b6651a7cef8bdee0a7a990f4f799a6bf9148269ee758d0704ee232`
- Prompt `a29e12a2839b106d3a3632e1f353961d74182245a41602b445d06f71abb097ba`
- Dataset `6ec947711773e6488dd5221620b62a0c2dfe1ea46e8208b192ec4777275b3c7b`
- M1 `model.safetensors` `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`, config `18e18afc…`
- **Cloud bundle SHA `ace20fc0f9d5c8a1ac4fb8e718614986f0cbef2122c8524501a7b721e4163db6`** (unchanged)

## 4. Invariants Re-Verified
- [x] No scientific / architecture / dataset / supervision / gate / threshold / seed logic changed.
- [x] Canonical deterministic smoke slug generation preserved (`theo-6c5-bc-smoke-<bundle-hash-prefix>`).
- [x] Exact dataset binding preserved: `Anakinistic/theo-6c5-inputs`.
- [x] `language == "python"`, `kernel_type == "script"`, `enable_gpu == True`, `enable_internet == False` preserved.
- [x] No-training `run.sh` preserved (no `cloud_6c5.py` invocation).
- [x] Smoke kernel distinct from training kernel slug (failed attempts preserved as evidence).
- [x] No `kernel_metadata_get` reference remains in the smoke verifier.
- [x] Clean HALT path on metadata retrieval failure.
- [x] All L5 / hash / bundle / scientific gates preserved.
- [x] 60/60 offline orchestrator checks PASS.

## 5. Conclusion
**AUDIT PASSES.** Remediation #7 is a provider-SDK integration fix only (uses the supported
`kernels_pull(..., metadata=True)` and normalizes provider slug casing), with no scientific change.
All invariants preserved. Training remains gated behind a passing provider smoke test under a fresh
L5 authorization. No provider smoke test or training was executed on this host (no Kaggle credentials).

**Requires:** fresh **L5 v8** authorization binding the post-#7 hashes above before any
`smoke_test_6c5_kaggle.py --execute`. The historical failed smoke/training kernel attempts are
preserved as evidence (not re-pushed, not deleted).
