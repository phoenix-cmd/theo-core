# 6C.5 Cloud Orchestrator — Source Audit #6 (Provider-Failure #6 Remediation)

**Date:** 2026-08-27
**Scope:** Remediation of provider-failure #6 — Kaggle `409 Client Error: Conflict ... SaveKernel` with warning `Your kernel title does not resolve to the specified id.`
**Author:** opencode (autonomous remediation, supervised)
**Status:** AUDIT PASSES — pending fresh L5 v7 authorization before any provider execution.

---

## 1. Root Cause (Provider-Failure #6)

The training-kernel metadata used a free-text `title` (`"THEO 6C.5 B/C cloud training"`)
that, after Kaggle's `slugify`, resolves to a **different slug** than the `id` slug
(`Anakinistic/theo-6c5-bc-training`). Kaggle rejects this with `409 SaveKernel` and the
warning `Your kernel title does not resolve to the specified id.` This is a
kernel-identity consistency failure, not a scientific defect.

## 2. Remediation Applied

### 2.1 `launch_6c5_kaggle.py` (training kernel)
- Removed the hard-coded, free-text `title`.
- `title` is now **derived from the canonical slug** (`KAGGLE_USERNAME/KAGGLE_KERNEL_SLUG`),
  so Kaggle's title→slug normalization always matches the `id` slug.
- No change to `language`, `kernel_type`, `enable_gpu`, `enable_internet`, or the bound
  `dataset_sources`.
- New hash: `57fe359fec61b2e589d0af09312699dbe50c2b1d1c0749fc5d815a95c400bd45`

### 2.2 `smoke_test_6c5_kaggle.py` (dedicated provider smoke kernel)
Rewritten so the smoke kernel identity is **single-source, deterministic, and provider-consistent**:
- `SMOKE_KERNEL_SLUG = theo-6c5-bc-smoke-<bundle-hash-prefix>` derived from `bundle_manifest.json`.
- `title` == `slug` (the #6 guard: title→slug == id slug, preventing 409).
- `id` == `KAGGLE_USERNAME/SMOKE_KERNEL_SLUG`; the API query uses the same slug.
- `dataset_sources` == `[KAGGLE_DATASET_OWNER/KAGGLE_DATASET_SLUG]` (provider-failure #5 guard preserved).
- `run.sh` contains **no invocation of `cloud_6c5.py`** (no training; smoke only).
- After push, live kernel metadata is queried and asserted: dataset source present,
  `language=="python"`, `kernel_type=="script"`, `enable_gpu==True`, `enable_internet==False`,
  `title == slug`. Any failure HALTS before any training kernel is started.
- New hash: `d8c3de0e0427e4b41548d733d44fc6c8d9e3aff704ccdb6993597e8554c716be`

### 2.3 `test_6c5_orchestrator.py`
- Updated `test_smoke_builder_binds_dataset_source` to provide a bundle manifest
  (required by deterministic slug derivation) and to assert the new deterministic slug
  (`theo-6c5-bc-smoke-<prefix>`) and `title == slug`.
- Added `test_smoke_slug_is_deterministic_and_distinct` (determinism, hash-sensitivity,
  distinct from training slug, slug-derived title).
- Added `test_smoke_metadata_provider_consistency` (single canonical identity for build + query).
- Suite: **47 → 55 checks, all PASS** (no execution, fully offline).
- New hash: `08d73e5ad52595a95e547bcd8a609ae4dfb80ef08702d91c04b3e28d68b62ce9`

## 3. Source Hashes — Post-Remediation #6

### 3.1 Orchestrator files (changed in remediation #6 marked *)
| File | SHA-256 (prefix) | Status |
|---|---|---|
| `kaggle_6c5_common.py` | `1648f332e80528a62504ffbcfa8a4cb535a7f4637ff83075a183ea23e798c050` | unchanged (#5) |
| `prepare_6c5_cloud_bundle.py` | `d8e4375488b042c0eb527545d898b3b3bfaa1543016c549807e38054b96b6a1e` | unchanged |
| `launch_6c5_kaggle.py` | `57fe359fec61b2e589d0af09312699dbe50c2b1d1c0749fc5d815a95c400bd45` | **CHANGED (#6)** |
| `retrieve_6c5_kaggle_results.py` | `cc95e0abd9e280fbdc8fd9d560c0fa37b2af1df68a4c7c42d1ff3044e65b83b9` | unchanged |
| `verify_6c5_cloud_results.py` | `cfa620413ee1fbf14c183d5fc51ac77c7f02b0e401546869da731a07d6bdca30` | unchanged |
| `amendment_audit.py` | `ab750ddde5d4520448552f66a3e5d6ff677577cd70a80bb5d75c4dad6db81ee2` | unchanged |
| `test_6c5_orchestrator.py` | `08d73e5ad52595a95e547bcd8a609ae4dfb80ef08702d91c04b3e28d68b62ce9` | **CHANGED (#6)** |
| `smoke_test_6c5_kaggle.py` | `d8c3de0e0427e4b41548d733d44fc6c8d9e3aff704ccdb6993597e8554c716be` | **CHANGED (#6)** |

### 3.2 Scientific / architecture files (UNMODIFIED across remediations #3–#6)
These were not touched by any orchestrator remediation; current hashes recorded as the
frozen scientific binding for v7.
| File | SHA-256 (prefix) |
|---|---|
| `arch_b/b_architecture.py` | `b2b432ff63e4c70be3a2d7504d03215ddf159a432af1c1de0e135f8d702914e8` |
| `arch_c/c_architecture.py` | `cbabdd916ec43eff11af89cf44a9ead2104629420189ccfdf256e1d50250b90d` |
| `phase6c5/experiment_spec.py` | `dfe8ed5df1ca16fe7da883c77573e69e810dce67e111025e03d83da05409b8b3` |
| `phase6c5/cloud_6c5.py` | `56fd6f6aaa68ef9ae238efccb3735cf8e275fa68b6b7816a0c54e358cd48198f` |
| `phase6c5/train_6c5.py` | `b1b5e9ca4b4917316d7d347addec67f8a7643c251437b54eae240c2b65a6c05d` |
| `phase6c5/eval_6c5.py` | `f5c028b159f8c1584157f042a9b80820edc57c2d7703d5b62f45ae1487d0ee37` |
| `phase6c5/phase2_audit.py` | `8bab7bd12162162dc87160ec63ee6f5d482807777474e93459cf3462ee40144f` |
| `phase6c5/amendment_audit.py` | `ab750ddde5d4520448552f66a3e5d6ff677577cd70a80bb5d75c4dad6db81ee2` |

### 3.3 Frozen inputs (unchanged from prior binding)
- Adjudication `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4`
- Fixture `f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457`
- Split `eda4cb6695b6651a7cef8bdee0a7a990f4f799a6bf9148269ee758d0704ee232`
- Prompt `a29e12a2839b106d3a3632e1f353961d74182245a41602b445d06f71abb097ba`
- Dataset `6ec947711773e6488dd5221620b62a0c2dfe1ea46e8208b192ec4777275b3c7b`
- M1 model.safetensors `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`, config `18e18afc…`
- **Cloud bundle SHA `ace20fc0f9d5c8a1ac4fb8e718614986f0cbef2122c8524501a7b721e4163db6`** (unchanged)

## 4. Invariants Re-Verified
- [x] No scientific / architecture / dataset / supervision / gate / threshold / seed logic changed.
- [x] 16-check phase-2 + amendment audit logic intact (offline, not re-run here; gates unchanged).
- [x] Orchestrator dry-run default; `--execute` required; no auto-retry; HALT on failure.
- [x] Provider smoke test uses a dedicated slug distinct from training slug (failed training kernel preserved).
- [x] Smoke `run.sh` never invokes `cloud_6c5.py` (no training in smoke).
- [x] Fail-closed: any provider metadata mismatch HALTS before training.
- [x] `language=="python"`, `kernel_type=="script"`, `enable_gpu==True`, `enable_internet==False` retained.
- [x] `dataset_sources` bound to exact owner/slug (provider-failure #5 guard retained).
- [x] Kernel `title` derived from `slug` (provider-failure #6 guard).
- [x] 55/55 offline orchestrator checks PASS.

## 5. Conclusion
**AUDIT PASSES.** Remediation #6 is orchestration/provider-identity only, with no scientific
change. All invariants preserved. Training remains gated behind a passing provider smoke test
under fresh L5 v7 authorization. No provider smoke test or training was executed on this host
(no Kaggle credentials).

**Requires:** fresh L5 v7 authorization binding the post-#6 hashes above before any
`smoke_test_6c5_kaggle.py --execute` or training kernel submission.
