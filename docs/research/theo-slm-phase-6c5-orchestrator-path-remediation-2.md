# 6C.5 Orchestrator — Path-Resolution Remediation #2 (Audit + Report)

**Date:** 2026-08-27
**Status:** Remediation COMPLETE at dry-run. Orchestrator source changed → **existing L5 authorization
(theo-slm-phase-6c5-l5-cloud-execution-authorization.md v2) is now STALE** and requires a fresh
source audit + fresh authorization before any cloud execution. No Kaggle upload, install, training,
or 6E.20.x work was performed.

---

## 1. Root cause (this time)

The first remediation normalized `ROOT` to a `pathlib.Path` but still rendered Git-Bash `/c/...`
paths as Windows `C:\...` **strings** and then passed them to `pathlib.Path`. Under a
**Git-Bash / MSYS Python, `pathlib.Path` is `PosixPath`**, so:

- `Path("C:\\Users\\...")` on POSIX becomes a *non-absolute* `PosixPath('C:Users...')`, which then
  gets re-joined with `cwd` (`Path.cwd() / p`).
- MSYS `os.path.join` *concatenates* a git-form dir with a Windows-form dir instead of replacing it
  (a Windows-absolute second arg only replaces on `nt` `os.path`).

The combined effect produced the reported `/c/Users/.../phase6c5/C:\Users\.../phase6c5/C:\Users\...`
path when the supplied `--root` was combined with a script-directory/`__file__`-derived absolute.

## 2. Fix (orchestration/path layer only)

File: `kaggle_6c5_common.py`
- `normalize_root(raw)` now:
  - reduces `/c/...`, `/cygdrive/c/...`, and `C:\...` to a `(drive, rest)` pair;
  - **renders to the CURRENT interpreter's native form** (`C:\...` when `os.name == "nt"`,
    `/c/...` under MSYS/Git-Bash Python) — never the wrong form for the running Python;
  - resolves exactly once via `Path.resolve()`;
  - **rejects** inputs with multiple drive roots or mixed `/c/` + `C:` forms (HALT);
  - **verifies the result is the actual THEO repository root** (must contain `theo-core`/`theo-data`)
    before returning — a concatenated/stale path cannot satisfy this, so it HALTs closed.
- `validate_binding_hashes(root)`: builds the spec path **exclusively** as
  `canonical / "theo-core" / "scripts" / "model_selection" / "phase6c5" / "experiment_spec.py"`,
  with a hard invariant asserting the spec path is under the canonical root and contains no mixed
  `/c/ + C:` form, and logs both `canonical_root` and `spec_path` before loading.
- `scan_exclusions` / `build_allowlist_paths` already use `Path` division; no `os.path.join` of
  mixed absolute/relative forms remains.

File: `amendment_audit.py` (the L5 gate's read-only subprocess)
- Replaced `__file__`-based `HERE` with `C.normalize_root(...)` → `HERE = ROOT / "theo-core" /
  "scripts" / "model_selection" / "phase6c5"`, and sets `THEO_ROOT` to the canonical form so
  `experiment_spec` resolves portably.

File: `prepare_6c5_cloud_bundle.py`
- `--root` / `--out` normalized via `C.normalize_root`; exclusion guard stringifies before `.lower()`.

**Scientific invariants preserved:** allowlist, frozen hashes, `EXCLUDE_PATTERNS`, L5 gate, HALT
behavior, dry-run default, and `--execute` requirement are unchanged. No dataset, model revision,
training, B/C architecture, prompt, split, gate, threshold, experiment specification, or
authorization semantics was modified.

## 3. Verification

### 3.1 Offline test suite — `test_6c5_orchestrator.py` → **21/21 PASS**
New path-equivalence regression tests added:
- `test_root_forms_resolve_to_same_native_path` — Windows `C:\...` and Git-Bash `/c/...` roots yield
  the identical native `Path` (and are absolute).
- `test_relative_dot_resolves_to_theo` — relative `.` from the THEO dir resolves to the THEO root.
- `test_spec_and_bundle_paths_consistent` — `experiment_spec.py` is under the canonical root, no
  mixed `/c/ + C:` form.
- `test_root_out_combos_resolve_identically` — (Win root+rel out) == (Win root+abs Win out) ==
  (Git-Bash root+abs Git out); absolute-out == relative-out when cwd is the repo.
(Pre-existing tests unchanged; all still pass.)

### 3.2 Dry-runs (no Kaggle contact)
Run from `phase6c5/`, exactly as supplied:
```
python prepare_6c5_cloud_bundle.py --root C:\Users\bs162\Desktop\THEO --out 6c5_cloud_bundle --force
python prepare_6c5_cloud_bundle.py --root /c/Users/bs162/Desktop/THEO --out 6c5_cloud_bundle --force
```
- **Both succeeded.**
- **Both produced the SAME bundle SHA-256:** `ace20fc0f9d5c8a1ac4fb8e718614986f0cbef2122c8524501a7b721e4163db6`
- **Both produced the same 17 allowlisted files.**
- **Zero** `6E.20.1` / `checkpoint` files in the bundle.
- (Bundle SHA differs from the prior `037ae1af…` only because `amendment_audit.py` — which is in the
  allowlist — was legitimately changed by this fix. This is exactly why a fresh authorization is required.)

### 3.3 Bundle content check
- 17 allowlisted files; `6E.20.1`/`checkpoint` count = 0.
- Manifest `root` is the canonical THEO path.

## 4. Orchestrator source hashes (post-remediation #2)

| SHA-256 | File | vs v2 auth |
|---------|------|-----------|
| `1757f11a078d76b22955bebd52164b58e2543ddc6d867e62a504934fcc5e2ba0` | `kaggle_6c5_common.py` | **CHANGED** (b2ed836…) |
| `d8e4375488b042c0eb527545d898b3b3bfaa1543016c549807e38054b96b6a1e` | `prepare_6c5_cloud_bundle.py` | unchanged |
| `fb80834651da97ae22e82de3dd183847a193bb3a4ce4906bd612e8a3f19c83b0` | `launch_6c5_kaggle.py` | unchanged |
| `cc95e0abd9e280fbdc8fd9d560c0fa37b2af1df68a4c7c42d1ff3044e65b83b9` | `retrieve_6c5_kaggle_results.py` | unchanged |
| `cfa620413ee1fbf14c183d5fc51ac77c7f02b0e401546869da731a07d6bdca30` | `verify_6c5_cloud_results.py` | unchanged |
| `ab750ddde5d4520448552f66a3e5d6ff677577cd70a80bb5d75c4dad6db81ee2` | `amendment_audit.py` | **CHANGED** (not previously bound; now hardened) |
| `57cdb510575f8288074b7fd0fc7b815af2132e8254e55740db9b0ada6e9fc80e` | `test_6c5_orchestrator.py` | **CHANGED** (d31d02d…) |

Bundle SHA (post-fix): `ace20fc0f9d5c8a1ac4fb8e718614986f0cbef2122c8524501a7b721e4163db6`

## 5. Status & next step
- Defect fixed and verified at dry-run (the intended place to catch it). No cloud execution occurred.
- **L5 authorization v2 is STALE** (orchestrator source hashes and bundle SHA changed). Required
  before any cloud run:
  1. fresh source audit of the changed orchestrator files (this report is the audit);
  2. fresh L5 cloud execution authorization binding the hashes in §4 and bundle SHA `ace20fc0…`.
- Only then: `launch_6c5_kaggle.py --execute`, then `retrieve_…` + `verify_6c5_cloud_results.py`,
  then local `eval_6c5.py` on GTX 1650.

**Stopping here for human review / fresh authorization. No `--execute` was run.**
