# 6C.5 Orchestrator — Path-Resolution Remediation / Audit Report

**Date:** 2026-08-27
**Defect:** Windows/Git-Bash path normalization defect in the orchestration layer.
**Severity:** Engineering-only. No scientific artifact, dataset, model, split, prompt, gate,
architecture, or training parameter was modified or invalidated.

---

## 1. Symptom
The dry-run of `prepare_6c5_cloud_bundle.py` HALTED with:

```
spec_from_file_location() receives:
/c/Users/bs162/Desktop/THEO/C:\Users\bs162\Desktop\THEO/.../experiment_spec.py
```

The path layer concatenated a Git-Bash style `/c/Users/...` prefix with an already-absolute
Windows `C:\Users\...` path.

## 2. Root cause
In `kaggle_6c5_common.py`, `ROOT` was a **raw string** defaulting to `C:\Users\bs162\Desktop\THEO`.
Under Git Bash, `THEO_ROOT` (or the working path) can be `/c/Users/bs162/Desktop/THEO`.
Subpaths were built with `os.path.join(root, rel)` where a mixed/absolute second component
produced the `/c/.../C:\...` concatenation. The root was never normalized to a single native
form, and `normalize_root` did not exist.

## 3. Fix (orchestration layer only)
File: `theo-core/scripts/model_selection/phase6c5/kaggle_6c5_common.py`

- Added `normalize_root(raw)` — normalizes a repository root **exactly once** into a native
  `pathlib.Path`:
  - Windows native `C:\Users\...` → native `Path` (idempotent);
  - Git Bash / MSYS `/c/Users/...` and `/cygdrive/c/...` → `C:\Users\...`;
  - relative paths resolved against `cwd`;
  - a single `Path.resolve()`; no `/c/...` prefix can coexist with a `C:\` absolute.
- `ROOT = normalize_root(os.environ.get("THEO_ROOT", r"C:\Users\bs162\Desktop\THEO"))` — computed
  once at import, stored as a `Path`.
- All subpath construction now uses `Path` division (`ROOT / "..."`) instead of `os.path.join`
  with a possibly-absolute string. `Path / absolute` **replaces** rather than concatenates, so a
  `/c/...` + `C:\...` mix can never recur.
- `validate_binding_hashes`, `scan_exclusions`, `build_allowlist_paths`, `check_l5_authorization`
  now normalize their incoming `root` and use `Path` arithmetic. `validate_binding_hashes` also
  HALTs cleanly (SystemExit) if `experiment_spec.py` cannot load from the given root.

File: `theo-core/scripts/model_selection/phase6c5/prepare_6c5_cloud_bundle.py`

- `--root` / `--out` normalized via `C.normalize_root`; `root`/`out` stored as `Path`.
- Exclusion guard stringifies `src` before `.lower()`.

**Scientific invariants preserved:** allowlist, frozen hashes, `EXCLUDE_PATTERNS`, L5 gate, HALT
behavior, dry-run default, `--execute` requirement, and no-auto-retry are all unchanged. No
training, upload, or install was performed.

## 4. Verification

### 4.1 Offline test suite — `test_6c5_orchestrator.py` → **14/14 PASS**
New path-layer regression tests added:
- `test_normalize_root_windows_abs` — Windows absolute → native Path; hashes validate.
- `test_normalize_root_gitbash` — `/c/Users/...` → `C:\...`; hashes validate (regression of the bug).
- `test_normalize_root_already_normalized` — idempotent.
- `test_normalize_root_invalid_rejected` — nonexistent root HALTS (SystemExit).
(Pre-existing tests unchanged: binding integrity, exclusion safety, verify pass/mismatch,
dry-run plan, `--execute` gating.)

### 4.2 Dry-run bundle build (the exact Git-Bash invocation that exposed the bug)
```
python theo-core/scripts/model_selection/phase6c5/prepare_6c5_cloud_bundle.py --force
```
Run twice:
- Normal Windows `THEO_ROOT` → `BUNDLE SHA-256 = 037ae1af0ba8d700bf4461b25e3a4760d39407f75adc47778213e21052ce1c7e`
- `THEO_ROOT=/c/Users/bs162/Desktop/THEO` (Git Bash style) → **identical** `BUNDLE SHA-256`.
Both now succeed (previously the Git-Bash form HALTED). Identical SHA proves the normalization
is consistent and deterministic.

### 4.3 Bundle manifest content check
- 17 allowlisted files present (no more, no less).
- Zero `6E.20.1` / `checkpoint` files.
- On-disk scan of `6c5_cloud_bundle/`: 18 files (17 + manifest); zero `6e20`/`checkpoint`/`.git`/`__pycache__` leaks.

## 5. Orchestrator source hashes (post-fix)

| SHA-256 | File |
|---------|------|
| `b2ed83694b46433315f57a3f7955fdc8cf95339ffff390f5e7d0b2270dfdcc89` | `kaggle_6c5_common.py` |
| `d8e4375488b042c0eb527545d898b3b3bfaa1543016c549807e38054b96b6a1e` | `prepare_6c5_cloud_bundle.py` |
| `fb80834651da97ae22e82de3dd183847a193bb3a4ce4906bd612e8a3f19c83b0` | `launch_6c5_kaggle.py` |
| `cc95e0abd9e280fbdc8fd9d560c0fa37b2af1df68a4c7c42d1ff3044e65b83b9` | `retrieve_6c5_kaggle_results.py` |
| `cfa620413ee1fbf14c183d5fc51ac77c7f02b0e401546869da731a07d6bdca30` | `verify_6c5_cloud_results.py` |
| `d31d02d233df3d94be9b424dbbc0f05bcc64c0f7ab4c00e64682cc718b93e85f` | `test_6c5_orchestrator.py` |

## 6. Status & next step
- The defect is fixed and verified at dry-run (the intended place to catch it).
- **No `launch_6c5_kaggle.py --execute` was run.** No cloud upload or training occurred.
- Orchestrator source changed → **fresh execution authorization required** before any cloud
  action. Sequence to follow:

  fix path layer → tests → dry-run → source audit → **fresh execution authorization** →
  ONLY THEN cloud upload/training.

**Recommendation:** Re-authorize per the established L5 process, then execute the orchestrator
on Kaggle per `theo-slm-phase-6c5-kaggle-run-sheet.md`.
