# 6C.5 Cloud Orchestrator — Independent Source Audit (post Remediation #3)

**Date:** 2026-08-27
**Scope:** Changes made for Kaggle CLI executable resolution (Remediation #3) only.
**Files touched:** `kaggle_6c5_common.py` (added `_kaggle_cmd_prefix` / `resolve_kaggle_command`,
modified `run_kaggle_cmd`), `launch_6c5_kaggle.py` (status poll routed through resolution),
`test_6c5_orchestrator.py` (added 4 regression tests).
**Method:** Direct source read + re-run of full offline test suite (26/26 PASS) + dry-run launch.
**Verdict:** Orchestration-only change. No scientific logic or controls altered. **PASS — no defect.**

---

## 1. Root cause confirmed
`run_kaggle_cmd` executed `subprocess.run(["kaggle", ...])`. On the Windows host the `kaggle`
module is importable (`py -3.14 -m kaggle --version` works) but the `kaggle.exe` launcher is not
on `PATH`, so `CreateProcess` raised `FileNotFoundError: [WinError 2]`. `launch.py:94` compounded
this with a direct `subprocess.run(["kaggle", ...])`.

## 2. Fix (command-resolution layer only)
- `_kaggle_cmd_prefix()`: prefers `[sys.executable, "-m", "kaggle"]` when the `kaggle` module is
  importable (same interpreter that runs the orchestrator); falls back to a *directly resolved*
  `kaggle.exe` under `sys.prefix/Scripts` or via `shutil.which` — never requires it on `PATH`;
  returns `None` if nothing resolves.
- `resolve_kaggle_command(sub_args)` returns the full command or `None`.
- `run_kaggle_cmd` now strips the logical `"kaggle"` token and rebuilds the command from the
  resolved prefix; if nothing resolves and not dry-run, it **HALTs cleanly** instead of raising
  `FileNotFoundError`.
- `launch.py:94` now routes through `C.resolve_kaggle_command(...)` (same fallback + clean HALT).

## 3. Invariant verification

| # | Invariant | Evidence | Result |
|---|-----------|----------|--------|
| 1 | No scientific logic changed | Only CLI-resolution glue added; `EXPECTED`, `ALLOWLIST`, `validate_binding_hashes`, `cloud_6c5.py`, `phase2_audit.py` untouched. | PASS |
| 2 | L5 authorization preserved | `check_l5_authorization` unchanged; `launch` still HALTs `--execute` when L5 inactive (new test confirms). | PASS |
| 3 | Hash controls preserved | `EXPECTED` unchanged; bundle SHA still `ace20fc0…` (orchestrator files are NOT in the allowlist). | PASS |
| 4 | Allowlist unchanged | `ALLOWLIST` unmodified; no checkpoint/6E.20.1 path can enter bundle. | PASS |
| 5 | No-retry preserved | `run_kaggle_cmd` `halt_on_fail` unchanged; launch still single status poll, no loop. | PASS |
| 6 | B×1 / C×1 preserved | `cloud_6c5.py` unchanged: trains B once + C once. | PASS |
| 7 | CLI resolution correct | Prefer `python -m kaggle`; fallback to resolved exe; clean HALT when neither (test covers all 3 branches). | PASS |
| 8 | 16 Phase-2 checks enforced | `phase2_audit.py` unchanged; re-run by `cloud_6c5.py` as final gate. | PASS |
| 9 | Linux runtime hard-gated | `cloud_6c5.py` dep gate + actual bnb wheel SHA capture intact. | PASS |
| 10 | Local latency separate | `eval_6c5.py` (local) still owns ≤0.5 s criterion; `cloud_6c5.py` records only wallclock + notes local measurement. | PASS |

## 4. Test results
- Full suite: **26/26 PASS** (added: module-preferred, direct-exe fallback, none→clean HALT,
  `--execute` still requires active L5).
- Dry-run launch resolves to `C:\Python314\python.exe -m kaggle datasets status …` (no `kaggle`
  token on PATH required). No Kaggle contact made.

## 5. Conclusion
Remediation #3 is correct, minimal, and preserves every control. Three orchestrator source files
changed (`kaggle_6c5_common.py`, `launch_6c5_kaggle.py`, `test_6c5_orchestrator.py`), so a fresh L5
authorization (v4) binding the new hashes is required. **No upload, install, or training occurred.**
