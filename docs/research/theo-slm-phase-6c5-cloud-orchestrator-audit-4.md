# 6C.5 Cloud Orchestrator — Independent Source Audit (post Remediation #4)

**Date:** 2026-08-27
**Scope:** The two Kaggle *submission* defects discovered during the v4 real-run attempt.
- **Defect A:** `kaggle datasets create` did not use `--dir-mode`, so Kaggle skipped the nested
  `theo-core/` and `theo-data/` folders and the dataset would be empty of model/data files.
- **Defect B:** kernel metadata set `"language": "script"`, which Kaggle rejects
  (`valid options are ['python','r','rmarkdown']`); the kernel push failed.
**Files touched:** `launch_6c5_kaggle.py` (added `_datasets_create_args` / `_kernel_metadata`,
fixed the create args and the metadata language), `test_6c5_orchestrator.py` (added 2 regression
tests). **No scientific logic, bindings, or controls changed.**
**Method:** Direct source read + full offline suite (**37/37 PASS**) + dry-run launch.
**Verdict:** Orchestration-only fix; every invariant preserved. **PASS — no defect.** Fresh L5 (v5) required.

---

## 1. Fixes applied
- **Defect A** — `launch_6c5_kaggle.py`:
  - New `_datasets_create_args(bundle_dir)` returns
    `["kaggle","datasets","create","-p",bundle_dir,"--dir-mode","zip"]`.
  - `launch()` now calls `C.run_kaggle_cmd(_datasets_create_args(bundle_dir), dry_run)`.
  - After resolution this becomes `python -m kaggle datasets create -p <bundle> --dir-mode zip`,
    so the full nested tree (incl. `theo-core/`, `theo-data/`) is uploaded as a zip.
- **Defect B** — `launch_6c5_kaggle.py`:
  - New `_kernel_metadata(user)` returns `language="python"`, `kernel_type="script"` (unchanged).
  - `_write_kernel` now serializes `_kernel_metadata(user)` to `kernel-metadata.json`.

## 2. Invariant verification

| # | Invariant | Evidence | Result |
|---|-----------|----------|--------|
| 1 | No scientific logic changed | Only submission-packaging glue; `EXPECTED`, `ALLOWLIST`, `validate_binding_hashes`, `cloud_6c5.py`, `phase2_audit.py` untouched. | PASS |
| 2 | Scientific bindings identical | `EXPECTED`/`validate_binding_hashes` unchanged; dataset `97bc6bc6…`, fixture `f99ca025…`, split `eda4cb66…`, prompt `a29e12a2…`, M1 `7ae55760…` intact. | PASS |
| 3 | Allowlist unchanged | `ALLOWLIST` unmodified; orchestrator files are NOT in the bundle (bundle SHA still `ace20fc0…`). | PASS |
| 4 | No checkpoint enters bundle | `EXCLUDE_PATTERNS` + `scan_exclusions` + `build_allowlist_paths` guard intact. | PASS |
| 5 | `--execute` L5-gated | `check_l5_authorization` unchanged; launch still HALTs `--execute` when L5 inactive (test confirms). | PASS |
| 6 | No-retry preserved | `run_kaggle_cmd` `halt_on_fail` unchanged; launch still single poll, no loop. | PASS |
| 7 | B×1 / C×1 preserved | `cloud_6c5.py` unchanged: trains B once + C once. | PASS |
| 8 | 16 Phase-2 checks enforced | `phase2_audit.py` unchanged; re-run by `cloud_6c5.py` as final gate. | PASS |
| 9 | Linux runtime hard-gated | `cloud_6c5.py` dep gate + actual bnb wheel SHA capture intact. | PASS |
| 10 | Local latency separate | `eval_6c5.py` (local) still owns ≤0.5 s criterion. | PASS |

## 3. Defect-fix regression coverage (the part dry-runs couldn't see)
- `test_datasets_create_command_shape`: pins both resolution branches and asserts the **resolved**
  command is exactly `python -m kaggle datasets create -p <bundle> --dir-mode zip` (module branch)
  and `… datasets create -p <bundle> --dir-mode zip` (exe-fallback branch); `--dir-mode` value is
  `"zip"` (not merely present); bundle path preserved; never a bare `kaggle` on PATH.
- `test_kernel_metadata_serialization`: asserts the serialized metadata is
  `{"language":"python","kernel_type":"script", ...}` and GPU on / internet off preserved.

## 4. Test results
- Full suite: **37/37 PASS** (added: command-shape structural equivalence + metadata serialization).
- Dry-run launch resolves to `python -m kaggle datasets … --dir-mode zip` and `python -m kaggle
  kernels push …`; no Kaggle contact made.

## 5. Conclusion
Remediation #4 corrects both provider/API contract failures, is minimal, and preserves every control.
Two orchestrator source files changed (`launch_6c5_kaggle.py`, `test_6c5_orchestrator.py`), so a fresh
L5 authorization (v5) binding the new hashes is required. **No upload, install, or training occurred.**
The user's recommended next step — a real provider-level smoke test with the authenticated account
(dataset + kernel metadata accepted, files present) — remains valuable and is explicitly *not* a
training run; it should follow v5 approval.
