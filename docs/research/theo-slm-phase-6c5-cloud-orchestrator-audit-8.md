# 6C.5 Cloud Orchestrator — Source Audit #8 (Remediation #8: Kernel Packaging / Runtime Integration)

**Date:** 2026-08-27
**Scope:** Remediation of a kernel packaging defect that caused the submitted Kaggle kernel to fail
at 0.8s with `SyntaxError: invalid decimal literal` on line 3 of `/kaggle/src/script.py`
(`SLUG=theo-6c5-inputs`), plus confirmation that the Python 3.14.3 hard gate remains enforced
(and is NOT silently relaxed to 3.12).
**Author:** opencode (autonomous remediation, supervised)
**Status:** AUDIT PASSES — pending fresh L5 v9 authorization before any provider execution.

---

## 1. Root Cause (Kernel Packaging Defect)

The kernel metadata set `code_file: "run.sh"` (shell wrapper) for a `language: "python"`,
`kernel_type: "script"` kernel. Kaggle executes the script-kernel entry as `script.py`; because the
declared `code_file` (run.sh) contained **shell syntax** (`SLUG=theo-6c5-inputs`, `export ...`),
that shell content was placed into the executed `script.py`, which is invalid Python →
`SyntaxError: invalid decimal literal`.

The fix: the kernel's `code_file` (its Python entry) must be a file containing **valid Python only**,
while the shell wrapper `run.sh` is kept as an auxiliary file (shell syntax only). No shell assignment
may ever appear in `script.py`.

Secondary finding (runtime): Kaggle's base image resolved Python **3.12**, but the cloud runner hard-gates
`Python == 3.14.3`. That gate is **correct and must remain**; the runner must HALT on a 3.12 runtime,
not substitute 3.12 for 3.14.3. (No training occurred — the kernel failed at 0.8s, before model load.)

## 2. Remediation Applied

### 2.1 `launch_6c5_kaggle.py` (training kernel)
- `_kernel_metadata`: `code_file` changed from `"run.sh"` → `"script.py"`.
- Added `_kernel_script_py()`: valid-Python entry that sets `THEO_ROOT/THEO_OUT_B/THEO_OUT_C` via
  `os.environ`, makes output dirs, `chdir`s into the bundled `phase6c5` directory, then `import cloud_6c5`
  (which runs the gated pipeline). No shell assignments; `DATASET_SLUG` is embedded as a Python constant.
- `_write_kernel`: now writes BOTH `script.py` (Python entry) and `run.sh` (auxiliary shell wrapper,
  which now invokes `python script.py`).
- New hash: `84bc95e87b9755c63667681c5661ec52fe7d7a8b195be5378f30d72efe2c94c9`

### 2.2 `smoke_test_6c5_kaggle.py` (provider smoke kernel)
- Added `_write_smoke_script_py()`: valid-Python no-training entry (verifies dataset attachment).
- `build_smoke_kernel`: now writes `script.py` (Python entry) + `run.sh` (auxiliary) and sets
  `code_file: "script.py"`. Slug/title/dataset binding and no-training guarantee preserved.
- New hash: `ccfd8d78ba180e875809b518904278b16d7c6ceddea07e35b446725450827671`

### 2.3 `test_6c5_orchestrator.py`
- `test_launch_kernel_packaging_valid_python_entry`: asserts generated `script.py` parses as Python,
  contains NO shell-style assignment (`NAME=value` with no space), `kernel-metadata.json code_file == "script.py"`,
  and `run.sh` still contains the shell assignment `SLUG=`.
- `test_dataset_slug_passed_via_metadata_and_env`: dataset_sources bound to `Anakinistic/theo-6c5-inputs`,
  `script.py` references the slug, and no `SLUG=` shell form in `script.py`.
- `test_cloud_runner_pins_python_3_14_3`: asserts `cloud_6c5.py` pins `"python": "3.14.3"` (and NOT 3.12),
  checks `py == PINNED["python"]` equality, and that a HALT is wired on mismatch.
- `test_python_312_causes_fail_closed`: replicates the runner's gate with a faked 3.12 runtime and asserts
  the gate FAILS (would HALT, not train); 3.14.3 passes.
- Suite: **60 → 75 checks, all PASS** (fully offline).
- New hash: `8eda6e178be07471874a98a6cb8cfc8731939bd2dd1da714260699a0b0c2292d`

### 2.4 `cloud_6c5.py`
- **UNCHANGED** (hash `56fd6f6aaa68ef9ae238efccb3735cf8e275fa68b6b7816a0c54e358cd48198f`). The Python 3.14.3
  hard gate (line 28 `PINNED["python"]="3.14.3"`, line 85-95 equality + HALT) is intact and was NOT relaxed.

## 3. Source Hashes — Post-Remediation #8

### 3.1 Orchestrator files (changed in remediation #8 marked *)
| File | SHA-256 (prefix) | Status |
|---|---|---|
| `kaggle_6c5_common.py` | `1648f332e80528a62504ffbcfa8a4cb535a7f4637ff83075a183ea23e798c050` | unchanged (#5) |
| `prepare_6c5_cloud_bundle.py` | `d8e4375488b042c0eb527545d898b3b3bfaa1543016c549807e38054b96b6a1e` | unchanged |
| `launch_6c5_kaggle.py` | `84bc95e87b9755c63667681c5661ec52fe7d7a8b195be5378f30d72efe2c94c9` | **CHANGED (#8)** |
| `retrieve_6c5_kaggle_results.py` | `cc95e0abd9e280fbdc8fd9d560c0fa37b2af1df68a4c7c42d1ff3044e65b83b9` | unchanged |
| `verify_6c5_cloud_results.py` | `cfa620413ee1fbf14c183d5fc51ac77c7f02b0e401546869da731a07d6bdca30` | unchanged |
| `amendment_audit.py` | `ab750ddde5d4520448552f66a3e5d6ff677577cd70a80bb5d75c4dad6db81ee2` | unchanged |
| `test_6c5_orchestrator.py` | `8eda6e178be07471874a98a6cb8cfc8731939bd2dd1da714260699a0b0c2292d` | **CHANGED (#8)** |
| `smoke_test_6c5_kaggle.py` | `ccfd8d78ba180e875809b518904278b16d7c6ceddea07e35b446725450827671` | **CHANGED (#8)** |

### 3.2 Scientific / architecture / runner files (UNMODIFIED across remediations #3–#8)
| File | SHA-256 (prefix) |
|---|---|
| `cloud_6c5.py` | `56fd6f6aaa68ef9ae238efccb3735cf8e275fa68b6b7816a0c54e358cd48198f` (unchanged #8) |
| `arch_b/b_architecture.py` | `b2b432ff63e4c70be3a2d7504d03215ddf159a432af1c1de0e135f8d702914e8` |
| `arch_c/c_architecture.py` | `cbabdd916ec43eff11af89cf44a9ead2104629420189ccfdf256e1d50250b90d` |
| `phase6c5/experiment_spec.py` | `dfe8ed5df1ca16fe7da883c77573e69e810dce67e111025e03d83da05409b8b3` |
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
- [x] No scientific / architecture / dataset / supervision / gate / threshold / seed / hyperparameter logic changed.
- [x] `cloud_6c5.py` Python gate remains `3.14.3`; NOT relaxed to 3.12 (hash unchanged).
- [x] Generated `script.py` parses as valid Python (offline `ast.parse`).
- [x] No shell assignment (`NAME=value`) appears in `script.py`; `run.sh` remains shell-only.
- [x] Kernel `code_file == "script.py"` (Python entry); `run.sh` auxiliary.
- [x] Dataset slug bound in `dataset_sources` (`Anakinistic/theo-6c5-inputs`) and referenced by `script.py`.
- [x] Canonical deterministic smoke slug, title==slug, no-training run.sh preserved.
- [x] Failed Kaggle kernel (and prior attempts) preserved as historical evidence; no re-submission during remediation.
- [x] 75/75 offline orchestrator checks PASS.

## 5. Conclusion
**AUDIT PASSES.** Remediation #8 is a kernel-packaging/runtime-integration fix only (Python entry vs shell
wrapper; Python gate untouched). All invariants preserved. Training remains gated behind a passing provider
smoke test under a fresh L5 authorization. No provider smoke test or training was executed on this host (no
Kaggle credentials). The 0.8s fail-closed rejection confirms the safety design is working.

**Requires:** fresh **L5 v9** authorization binding the post-#8 hashes above before any
`smoke_test_6c5_kaggle.py --execute`. Training kernel submission requires a further, separate authorization
after the smoke test passes AND after the runtime Python-version question is resolved (the 3.12 runtime must
NOT cause relaxation of the 3.14.3 pin; instead the execution environment must be redesigned under authorization).
