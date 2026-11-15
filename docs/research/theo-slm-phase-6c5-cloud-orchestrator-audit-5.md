# 6C.5 Cloud Orchestrator — Independent Source Audit (post Remediation #5 / Provider failure #5)

**Date:** 2026-08-27
**Scope:** Provider failure #5 — the pushed kernel had `"dataset_sources": []` even though the
dataset `Anakinistic/theo-6c5-inputs` was `ready`. The defect is in the kernel-source
attachment/metadata generation or push path, not dataset creation.
**Files touched:** `kaggle_6c5_common.py` (added `KAGGLE_DATASET_OWNER`), `launch_6c5_kaggle.py`
(bind `dataset_sources` to the exact `owner/slug`; added `write_kernel_metadata` + kernel-slug
override), new `smoke_test_6c5_kaggle.py` (dedicated provider smoke test), `test_6c5_orchestrator.py`
(added 4 regression tests). **No scientific logic, bindings, or controls changed.**
**Method:** Direct source read + full offline suite (**47/47 PASS**).
**Verdict:** Orchestration-only fix; every invariant preserved. **PASS — no defect.** Fresh L5 (v6) required.

---

## 1. Fixes applied
- **`kaggle_6c5_common.py`** — added `KAGGLE_DATASET_OWNER = os.environ.get("KAGGLE_DATASET_OWNER", KAGGLE_USERNAME)`
  so the input-dataset owner is a first-class, overridable binding (defaults to the pushing username).
- **`launch_6c5_kaggle.py`**
  - `_kernel_metadata(user, kernel_slug=None)` now sets
    `"dataset_sources": [f"{C.KAGGLE_DATASET_OWNER}/{C.KAGGLE_DATASET_SLUG}"]` — the exact authorized
    slug, independent of the kernel-owner username. Also supports a kernel-slug override.
  - New `write_kernel_metadata(user, kernel_dir, kernel_slug=None)` serializes that dict to disk.
  - `language:"python"`, `kernel_type:"script"`, `enable_gpu:true`, `enable_internet:false` unchanged.
- **New `smoke_test_6c5_kaggle.py`** — a SEPARATE provider gate (not `launch --execute`, which can
  start training). It pushes a dedicated no-op smoke kernel (`theo-6c5-bc-smoke`, so the existing
  failed training kernel is preserved as evidence), then **queries the LIVE kernel metadata via the
  Kaggle API** and asserts `dataset_sources == ["Anakinistic/theo-6c5-inputs"]`, `language=="python"`,
  `kernel_type=="script"`, `enable_gpu==True`, `enable_internet==False`. It HALTs if any check fails
  and never invokes `cloud_6c5.py`.

## 2. Invariant verification

| # | Invariant | Evidence | Result |
|---|-----------|----------|--------|
| 1 | No scientific logic changed | Only kernel-source binding + smoke gate; `EXPECTED`, `ALLOWLIST`, `validate_binding_hashes`, `cloud_6c5.py`, `phase2_audit.py` untouched. | PASS |
| 2 | Scientific bindings identical | `EXPECTED`/`validate_binding_hashes` unchanged; dataset `97bc6bc6…`, fixture `f99ca025…`, split `eda4cb66…`, prompt `a29e12a2…`, M1 `7ae55760…` intact. | PASS |
| 3 | Allowlist unchanged | `ALLOWLIST` unmodified; orchestrator files not in bundle (bundle SHA still `ace20fc0…`). | PASS |
| 4 | No checkpoint enters bundle | `EXCLUDE_PATTERNS` + guards intact. | PASS |
| 5 | `--execute` L5-gated | `check_l5_authorization` unchanged; launch still HALTs `--execute` when L5 inactive. | PASS |
| 6 | No-retry preserved | `run_kaggle_cmd` `halt_on_fail` unchanged; launch single poll, no loop. | PASS |
| 7 | B×1 / C×1 preserved | `cloud_6c5.py` unchanged: trains B once + C once. | PASS |
| 8 | 16 Phase-2 checks enforced | `phase2_audit.py` unchanged; re-run by `cloud_6c5.py`. | PASS |
| 9 | Linux runtime hard-gated | `cloud_6c5.py` dep gate + bnb wheel SHA capture intact. | PASS |
| 10 | Local latency separate | `eval_6c5.py` (local) still owns ≤0.5 s criterion. | PASS |

## 3. Defect-fix regression coverage (offline; the part dry-runs couldn't see)
- `test_kernel_metadata_contains_bound_dataset_source` — with `KAGGLE_DATASET_OWNER=Anakinistic`, the
  generated metadata's `dataset_sources == ["Anakinistic/theo-6c5-inputs"]`.
- `test_kernel_metadata_owner_override` — owner settable independently of kernel-owner username.
- `test_written_kernel_metadata_contains_dataset_source` — the *serialized* file binds the dataset source.
- `test_smoke_builder_binds_dataset_source` — the dedicated smoke kernel binds the exact slug, uses the
  dedicated `theo-6c5-bc-smoke` slug (preserves failed training kernel), and its `run.sh` does NOT
  invoke `cloud_6c5.py`.

## 4. Provider-level verification (added, requires authenticated account to run live)
`smoke_test_6c5_kaggle.py` pushes the smoke kernel and queries live metadata, asserting the attachment
and settings. This closes the gap that local dry-runs could not see (failures #4/#5 were provider/API
contract issues). It must run on the authenticated `Anakinistic` account; this host has no credentials,
so the live push/query is not executed here — the offline generation is verified instead.

## 5. Test results
- Full suite: **47/47 PASS** (added 4 checks for dataset-source binding + smoke builder).
- Dry-run launch still resolves to `python -m kaggle datasets create -p <bundle> --dir-mode zip` and
  `python -m kaggle kernels push …`; no Kaggle contact made.

## 6. Conclusion
Remediation #5 corrects the kernel dataset-attachment defect and adds a provider-level gate that must
pass before any training kernel is started. Four orchestrator source files changed (common, launch,
test, and the new smoke script), so a fresh L5 authorization (v6) binding the new hashes is required.
**No upload, install, or training occurred.** Per the new architecture, B/C training proceeds only
after the provider smoke test passes and a fresh L5 training authorization is approved.
