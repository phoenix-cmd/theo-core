# 6C.5 Cloud Runtime Feasibility — Execution Authorization (FEASIBILITY-ONLY, FINAL REFRESH)

**Date:** 2026-08-27
**Supersedes:** the prior feasibility-only authorization (binding hash `dd847f90…`, before Remediation #11).
This is a narrower authorization for the read-only feasibility probe; it does NOT authorize training. The current
active training-gate authorization remains **L5 v9** (smoke-only); **L5 v10 is NOT issued** by this document.
**Status:** AUTHORIZED to run `feasibility_6c5_kaggle.py` on the approved Kaggle GPU/provider path as a
**read-only environment probe**. Training remains unauthorized.

---

## 1. Authorization Scope
- **Authorized:** running `feasibility_6c5_kaggle.py --build-kernel` locally to produce the kernel package, then
  submitting the dedicated kernel `theo-6c5-bc-feasibility` on the authenticated `Anakinistic` account. Kaggle
  executes `script.py` (the `code_file`), which is a single, valid-Python entrypoint that detects the Kaggle
  runtime (native `KAGGLE_KERNEL_RUN_TYPE`), sets `os.environ["KAGGLE_RUNTIME"]="1"` in Python, and runs the
  probe to determine whether CPython 3.14.3 + the pinned stack is technically obtainable. The kernel may use
  `enable_internet = True` (required to build the isolated interpreter and pull wheels).
- **NOT authorized:** any model training, any `cloud_6c5.py` execution, any dataset mutation, any re-submission
  of the training kernel. Training requires a separate **L5 v10** authorization, which is issued ONLY after a
  `FEASIBLE` result and a fresh review.
- **Prohibited on this host:** any execution (no Kaggle credentials on the Windows/GTX 1650 host).
- **Evidence preserved:** v9 smoke-gate PASS, the failed training kernel (0.8 s SyntaxError), and all prior
  attempts remain as historical evidence.

## 2. Binding — Feasibility Script Hash
| File | SHA-256 |
|---|---|
| `feasibility_6c5_kaggle.py` | `1013d1174e2c060a42b763f824be94bed25b566d4f3f0361c05cb5aa11b103b8` |

### Remediation #9 (entry-point separation) — incorporated
- Explicit CLI: `python feasibility_6c5_kaggle.py --build-kernel` (local packaging, no evaluation) and
  `python feasibility_6c5_kaggle.py --probe-cloud` (Kaggle-only evaluator, refused outside the runtime).
- Default (no-arg) invocation performs **no** cloud verdict locally — cannot emit a false `INFEASIBLE`.
- Remediation #10/11: the shell wrapper was eliminated; `code_file == "script.py"` (pure Python). Kaggle runs
  `python script.py` (no args); the entrypoint explicitly selects probe-cloud mode and sets the runtime marker in
  Python. Verified by offline regression tests; audit PASS (114/114).

### Remediation #11 (final permitted Kaggle remediation) — incorporated
- Root cause of the prior live run: the generated `script.py` contained shell syntax (`export KAGGLE_RUNTIME=1`)
  → `SyntaxError` when Kaggle executed it as Python. Fixed by removing all shell syntax; `script.py` is now pure
  Python that establishes the runtime marker via `os.environ[...]` and runs the probe. No `run.sh` is generated.

## 3. Required runtime (must be reproduced EXACTLY; identical to `cloud_6c5.py` PINNED)
- Python **3.14.3**, torch **2.13.0+cu126**, CUDA runtime **12.6**, transformers **5.8.0**, peft **0.20.0**,
  bitsandbytes **0.50.1**.

## 4. Hard stopping rule (enforced by this authorization)
```
Kaggle Python 3.14.3 feasibility test
        ↓
  PASS (FEASIBLE)  → environment considered for a fresh L5 v10 training authorization
  FAIL (INFEASIBLE)→ ABANDON the Kaggle path for this experiment; do NOT relax pins; no further workarounds
```
This is the **last** authorized Kaggle workaround (#3–#8 + this feasibility probe). No ninth / tenth workaround.

## 5. Gates on the feasibility result
1. The kernel must exit with a machine-readable manifest `6c5_runtime_feasibility_manifest.json` carrying a
   verdict (`FEASIBLE` / `INFEASIBLE`), `base_image_python`, `base_image_is_3_12`, `realized` stack, and
   `mismatches` if any.
2. If `INFEASIBLE`: record the exact incompatibility in
   `theo-slm-phase-6c5-cloud-environment-redesign.md` (FAIL branch) and STOP the Kaggle path.
3. If `FEASIBLE`: the realized `bitsandbytes_package_sha256` and full stack are recorded and may be referenced
   in a subsequent L5 v10 training authorization (which still requires its own explicit issuance).

## 6. Audit Reference
- `theo-slm-phase-6c5-runtime-feasibility-spec.md`
- `theo-slm-phase-6c5-runtime-feasibility-audit.md` — AUDIT PASSES (109/109)
- `theo-slm-phase-6c5-cloud-runtime-compatibility-check.md` — prior check (FAILED at Python gate; this probe
  resolves the open question of whether 3.14.3 is buildable on Kaggle at all)
- `theo-slm-phase-6c5-cloud-environment-redesign.md` — documents the PASS/FAIL branch outcomes
