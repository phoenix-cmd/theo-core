# 6C.5 Cloud Orchestrator — Independent Source Audit (post Remediation #2)

**Date:** 2026-08-27
**Scope:** All orchestration sources after the path-resolution remediation #2.
**Method:** Direct source read of `kaggle_6c5_common.py`, `prepare_6c5_cloud_bundle.py`,
`launch_6c5_kaggle.py`, `retrieve_6c5_kaggle_results.py`, `verify_6c5_cloud_results.py`,
`amendment_audit.py`, `test_6c5_orchestrator.py`, `cloud_6c5.py` (executor), `phase2_audit.py`.
**Verdict:** No scientific logic changed; all invariants below hold. **No defect requiring code change.**
Orchestrator sources changed (path layer only) → the prior L5 authorization is superseded by the
fresh authorization produced alongside this audit.

---

## 1. Invariant verification

| # | Invariant | Evidence | Result |
|---|-----------|----------|--------|
| 1 | No scientific logic changed | All orchestrator modules only perform path resolution, allowlist copy, hash validation, L5 gate, and Kaggle-CLI shaping. No architecture/dataset/supervision/gate code present. | PASS |
| 2 | Dataset / fixture / split / prompt / model bindings identical | `kaggle_6c5_common.EXPECTED` = adjudication `97bc6bc6…`, fixture `f99ca025…`, split `eda4cb66…`, prompt `a29e12a2…`, M1 rev `7ae55760`, aux `1.0`, seed `20260826`. `validate_binding_hashes` and `cloud_6c5.py` re-verify every one at runtime. | PASS |
| 3 | Allowlist unchanged | `ALLOWLIST` = the authorized 17 files (M1 weights, adjudication, mapping, 66-case fixture, split, binding manifest, `experiment_spec.py`, `train_6c5.py`, `cloud_6c5.py`, `phase2_audit.py`, `amendment_audit.py`, `bind_6c5.py`, `requirements.txt`). None is a checkpoint/6E.20.1 artifact. | PASS |
| 4 | No checkpoint material can enter bundle | `EXCLUDE_PATTERNS` = `["6e20","6E20","6e.20","checkpoint",".git","__pycache__","node_modules",".ipynb_checkpoints","6c5_cloud_execution_manifest.json"]`; `scan_exclusions` runs first and the bundle manifest reports `excluded_count`; `build_allowlist_paths` re-guards each path; dry-run confirms 0 such files. | PASS |
| 5 | `--execute` remains authorization-gated | `launch_6c5_kaggle.py`: `dry_run = not a.execute` (default dry-run); `if execute and not l5_ok: halt`. `retrieve_…` identical pattern. `check_l5_authorization()` runs `amendment_audit.py` (exit 0) before allowing execution. | PASS |
| 6 | No automatic retry | `run_kaggle_cmd` uses `halt_on_fail=True`; dataset push and the single status poll do not loop on failure; poll explicitly notes "no auto-retry". | PASS |
| 7 | B and C exactly one run each | `cloud_6c5.py` trains in `for arch in ("B","C")` — one pass, no loop counter, no re-submission. | PASS |
| 8 | All 16 Phase-2 checks remain enforced | `phase2_audit.py` implements checks 1–16 (incl. 13b). `cloud_6c5.py` re-runs it as the final gate via `subprocess.run([…,"phase2_audit.py"])` and HALTs on non-zero exit. | PASS |
| 9 | Actual Linux cloud runtime remains hard-gated | `cloud_6c5.py` `PINNED` baseline (py 3.14.3 / torch 2.13.0+cu126 / cuda 12.6 / transformers 5.8.0 / peft 0.20.0 / bnb 0.50.1); `dep_ok` → `sys.exit("HALT…")` on mismatch; actual Linux `bitsandbytes` wheel SHA bound into `provenance.bitsandbytes_wheel_sha256` at runtime (never the Windows hash). | PASS |
| 10 | Local latency measurement remains separate | `cloud_6c5.py` records only `wallclock_s` as provenance and states "Latency gates are measured locally on GTX 1650 after retrieval"; it does not gate on latency. `eval_6c5.py` (run locally) owns the ≤0.5 s criterion. | PASS |

## 2. Path-resolution correctness (the fixed defect)

- `normalize_root` reduces `/c/`, `/cygdrive/c/`, and `C:\` forms to a `(drive, rest)` pair and
  renders to the **current interpreter's native form** (`C:\` on Windows Python, `/c/` under
  MSYS/Git-Bash Python), resolves once, rejects multiple-drive / mixed forms, and **verifies the
  result is the actual THEO repository root** (must contain `theo-core`/`theo-data`).
- All repo paths are built exclusively as `ROOT / "theo-core" / …` / `ROOT / "theo-data" / …`
  (no `os.path.join` of mixed absolute/relative, no `__file__`-based script-dir prepend).
- `validate_binding_hashes` asserts `canonical_root == normalize_root(<canonical THEO>)` and that
  `spec_path` is under the canonical root with no mixed `/c/+C:` form, logging both before load.
- `amendment_audit.py` (L5 gate subprocess) now derives `HERE` from `normalize_root` instead of
  `os.path.abspath(__file__)`.
- Regression suite `test_6c5_orchestrator.py` = **21/21 PASS**, covering Windows / Git-Bash / `.`
  roots, relative/absolute output, and identical bundle SHA across `--root` forms.

## 3. Audit conclusion

The orchestrator is **orchestration-only** and enforces every invariant through positive gates and
fail-closed HALTs. Scientific artifacts are untouched; the only changes are the path-resolution
fix and its tests/audit plumbing. **No defect requiring further code change was identified.**

**Status: PASS — ready for a fresh L5 cloud execution authorization binding the new source hashes
and bundle SHA `ace20fc0…`. No `--execute`, upload, install, or training occurred during this audit.**
