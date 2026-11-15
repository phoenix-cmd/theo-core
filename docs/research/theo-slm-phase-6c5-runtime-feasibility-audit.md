# 6C.5 Cloud Runtime Feasibility — Independent Source Audit

**Date:** 2026-08-27
**Scope:** Audit of `feasibility_6c5_kaggle.py` (read-only Kaggle runtime feasibility test) before any
execution authorization. **Status:** AUDIT PASSES — ready for a feasibility-only execution authorization.
Training is out of scope and remains blocked until a separate L5 training authorization after a PASS.

---

## 1. Read-only guarantees verified (by inspection + offline tests)
- [x] Does **not** import `cloud_6c5.py` training logic; cannot trigger training.
- [x] No call to `train_6c5`, no model instantiation, no gradient step, no dataset write.
- [x] `capture_under` only imports torch/transformers/peft/bitsandbytes and records versions + a dir SHA;
      performs no training and no filesystem writes outside `/kaggle/working`.
- [x] `REQUIRED` equals `cloud_6c5.PINNED` exactly (offline test `test_feasibility_required_pins_match_binding`);
      `python` is `"3.14.3"`, never `"3.12"`.
- [x] `evaluate_compatibility` is fail-closed: any mismatch → `INFEASIBLE`; offline test confirms a realized
      `3.12.7` is rejected (not silently accepted) and `base_is_3_12` is flagged distinctly from infeasible.
- [x] Kernel packaging (`build_feasibility_kernel`) emits a valid-Python `script.py` (offline `ast.parse` passes),
      no shell assignment in `script.py`, `code_file == "script.py"`, `enable_gpu == True`.
- [x] `enable_internet == True` only for the feasibility probe (needed to build the isolated interpreter);
      this is a distinct, read-only kernel — not the training run.

## 1b. Remediation #9 — local/cloud entry-point separation (verified)
- [x] Explicit CLI contract: `--build-kernel` (local packaging only) / `--probe-cloud` (Kaggle evaluator).
- [x] `python feasibility_6c5_kaggle.py` with **no args** does NOT perform a cloud verdict locally
      (offline test `default invocation does NOT emit FEASIBLE/INFEASIBLE`; exits non-zero).
- [x] `--probe-cloud` is **refused outside the Kaggle runtime** (guard on `KAGGLE_KERNEL_RUN_TYPE` /
      `KAGGLE_RUNTIME=1`, set by `run.sh`); no false local `INFEASIBLE` can be emitted against Kaggle.
- [x] `--build-kernel` produces exactly `script.py`, `run.sh`, `kernel-metadata.json`
      (offline test asserts all three; `run.sh` is shell-only and invokes `--probe-cloud`).
- [x] Generated `script.py` contains the cloud evaluator and is valid Python; `run.sh` remains shell-only
      (offline `ast.parse` on `run.sh` raises `SyntaxError`).
- [x] Cloud `main()` still hard-fails (exit 2) on a dependency mismatch; Python 3.12 base is NOT itself
      classified infeasible when 3.14.3 is realized.
- [x] The `kaggle_6c5_common` import was removed from module top-level so the copied `script.py` runs
      standalone on Kaggle without the orchestrator present.

## 1c. Remediation #10 — Kaggle entrypoint must explicitly select `--probe-cloud` (superseded by #11)
- The live run with `code_file: "run.sh"` still failed: the generated `script.py` (a copy of this module)
  contained the shell line `export KAGGLE_RUNTIME=1` and Kaggle executed `script.py` as Python →
  `SyntaxError: invalid syntax`. This confirmed that mixing shell syntax into the Python entrypoint is unsafe;
  the shell wrapper was eliminated entirely in Remediation #11.

## 1d. Remediation #11 — single pure-Python `script.py` entrypoint, no shell wrapper (verified)
- [x] **Eliminated the shell wrapper.** The package now contains only `script.py` + `kernel-metadata.json`.
      `code_file == "script.py"`. No `run.sh` is generated or executed.
- [x] `script.py` is a single, valid-Python entrypoint (compiles via `py_compile`). It contains **no**
      `export`, **no** `NAME=value` shell assignments, **no** shell syntax of any kind.
- [x] Kaggle invokes `python script.py` with no arguments. The `__main__` block detects the Kaggle runtime via
      the native `KAGGLE_KERNEL_RUN_TYPE` env var (NOT `/kaggle/working`, which false-positives on the dev
      host), explicitly selects probe-cloud mode, and sets `os.environ["KAGGLE_RUNTIME"] = "1"` in **Python**
      (never via shell). It then runs the cloud evaluator (`main()`).
- [x] A bare local `python script.py` (no `KAGGLE_KERNEL_RUN_TYPE`) exits non-zero with NO verdict — so the
      dev host cannot emit a false `INFEASIBLE`. `--probe-cloud` is likewise refused off-Kaggle.
- [x] Offline regression tests inspect the ACTUAL generated files: `script.py` compiles; it rejects shell
      `export`/assignment; it contains the explicit `--probe-cloud` selector; the runtime marker is set in
      Python; `metadata code_file == "script.py"`; and no bare `python script.py` line is present.
- [x] No B/C training; no `cloud_6c5.py`; no dataset mutation; Python 3.14.3 remains mandatory for the
      isolated env; base 3.12 remains informational. This is the final permitted Kaggle remediation.

## 2. Source hash (binding for authorization)
- `feasibility_6c5_kaggle.py` = `1013d1174e2c060a42b763f824be94bed25b566d4f3f0361c05cb5aa11b103b8`

## 3. Frozen artifacts — confirmed NOT modified
- `cloud_6c5.py` unchanged (hash `56fd6f6aaa68ef9ae238efccb3735cf8e275fa68b6b7816a0c54e358cd48198f`); the
  Python 3.14.3 gate is intact.
- 300-GOLD, 66-case fixture, split, prompt, B/C architecture, training hyperparameters, gates, M1 revision,
  adjudication, dataset — untouched.
- v9 smoke-gate evidence preserved; no re-submission of the training kernel.

## 4. Offline test coverage (orchestrator suite now 114 checks, all PASS)
Feasibility-specific tests:
- `test_feasibility_required_pins_match_binding` — pins equal binding; Python not relaxed to 3.12.
- `test_feasibility_evaluate_fail_closed` — base 3.12 + realized 3.14.3 stack → FEASIBLE; realized 3.12 → INFEASIBLE.
- `test_feasibility_kernel_packaging` — `script.py` compiles; no shell `export`/assignment; `code_file == "script.py"`; GPU/internet on.
- `test_feasibility_cli_build_kernel_and_default_safe` — `--build-kernel` creates only `script.py` + metadata
  (no `run.sh`); default invocation emits no verdict; `--probe-cloud` refused off-Kaggle; entry selects `--probe-cloud`.
- `test_feasibility_kernel_entrypoint_invokes_probe_cloud` — `code_file == "script.py"`; entrypoint explicitly
  selects `--probe-cloud`; runtime marker set in Python; no shell syntax.
- `test_feasibility_generated_script_is_python_only` — inspects the ACTUAL generated `script.py`: compiles;
  no `export`/shell assignment; contains the `--probe-cloud` selector and a Python runtime-marker assignment.
- `test_feasibility_cloud_main_hard_fails_on_mismatch` — cloud `main()` exits 2 on dependency mismatch.
- `test_feasibility_base_312_not_classified_infeasible` — Python 3.12 base is informational, not INFEASIBLE.
- (Smoke tests `test_smoke_uses_supported_sdk_metadata_call` / `test_smoke_metadata_retrieval_failure_halts`
  were made hermetic — function-level mock, no live Kaggle call — to keep the offline suite deterministic.)

## 5. Behavior on Kaggle (designed, not yet executed)
- The kernel entry (`script.py`, running under base 3.12) builds an isolated 3.14.3 env, installs pins, then
  re-executes the capture payload under 3.14.3. It writes `6c5_runtime_feasibility_manifest.json` and exits
  non-zero on INFEASIBLE (fail-closed). It performs no training and mutates no dataset.
- If `torch==2.13.0+cu126` has no wheel for CPython 3.14.3, or the build fails, the result is `INFEASIBLE`
  with the exact reason — and per the hard stopping rule, the Kaggle path is abandoned (no further workaround).

## 6. Conclusion
**AUDIT PASSES.** The feasibility script is a clean, fail-closed, read-only probe that enforces the identical
pins as training and makes the base-3.12 vs isolated-3.14.3 distinction explicit. No scientific logic is touched.
Authorization for the feasibility run (only) may now be issued. Training remains blocked behind a later,
separate L5 authorization contingent on a PASS.
