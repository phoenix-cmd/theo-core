# PHASE 6E.20.2 — SIGMA_c PROTOCOL REMEDIATION REPORT

**Report ID:** 6E20.2-SIGMA-REMEDIATION
**Date:** 2026-08-19
**Mode:** IMPLEMENTATION REMEDIATION ONLY (single source-level protocol mismatch)
**Execution Status:** NOT AUTHORIZED
**L5 Authorization:** DOES NOT EXIST
**Parent audit:** `FORENSIC_6E20.2_POST_REMEDIATION_FINAL_PRE_AUTHORIZATION_SOURCE_AUDIT.md` (verdict A — DO NOT AUTHORIZE; sole failure: sigma_c).
**Remediated finding:** `6E20.2-SIGMA-PROTOCOL-MISMATCH-01`

---

## 1. Finding Remediated

**Finding ID:** `6E20.2-SIGMA-PROTOCOL-MISMATCH-01`

The independent post-remediation pre-authorization source audit found exactly one remaining scientific source-level failure:

- **OLD (incorrect) implementation:**
  ```
  sigma = std(rho) = sqrt( mean( (rho - mean(rho))^2 ) )
  ```
  i.e. `sigma_c = float(np.std(per_sample_rho))` at `run_phase_6e20_reevaluation.py:699`.

- **REQUIRED (authoritative, frozen Protocol v3.1 §5.7 and V3-03):**
  ```
  sigma_c = (1/N) * sum_i (rho_i^2) = mean_i (rho_i^2)
  ```
  the **mean variance share** of the per-sample readout-total alignment.

`std(rho)` is mathematically distinct from `mean(rho^2)` (they coincide only when `mean(rho) == 0`). The protocol explicitly warns that `(mean rho)^2 != mean(rho^2)`. The previous `std` definition was therefore a definitive protocol mismatch.

---

## 2. Exact Source Changes

### 2.1 `run_phase_6e20_reevaluation.py` — `compute_live_rho_c` (line 699)

**OLD:**
```python
    sigma_c = float(np.std(per_sample_rho))
```

**NEW:**
```python
    sigma_c = float(np.mean(np.asarray(per_sample_rho, dtype=np.float64) ** 2))
```

The `rho_c` aggregation (`rho_c = float(np.mean(per_sample_rho))`, line 698) and the entire `rho_c` computation (g_total via frozen `psdg_fn`, `g_R = alpha_c * grad_{lora_params}(s)`, common LoRA space, `compute_readout_total_alignment`) were **NOT** changed.

### 2.2 `run_phase_6e20_reevaluation.py` — docstring (lines 615–616)

Corrected the documentation that previously described `sigma_c` as `std_i rho_i` (a "reporting diagnostic"):

**OLD:**
```python
        rho_c   = mean_i rho_i
        sigma_c = std_i rho_i          # reporting diagnostic (see remediation report §3/§4 re Protocol §5.7)
```

**NEW:**
```python
        rho_c   = mean_i rho_i
        sigma_c = mean_i (rho_i ** 2)  # mean variance share, frozen Protocol v3.1 §5.7 / V3-03
                                      # (NOT std(rho); NOT (mean rho)**2)
```

### 2.3 `test_phase_6e20_reevaluation.py` — import

Added `compute_live_rho_c` to the existing import from `run_phase_6e20_reevaluation` so the real evaluator function can be exercised synthetically.

### 2.4 `test_phase_6e20_reevaluation.py` — TEST D replacement

The previous TEST D asserted `sigma_c == std(rho)` (tautological, and conformant to the *wrong* definition). It was **replaced** by `sigma_c_is_mean_variance_share`, which:

1. Calls the **real** `compute_live_rho_c` with purely synthetic fakes (no checkpoint, no real model, no tokenizer load, no autograd on a real model).
2. Computes the three candidate quantities from the returned `per_sample_rho`:
   - `mean(rho^2)` — CORRECT (Protocol),
   - `std(rho)` — WRONG ALT 1,
   - `(mean rho)^2` — WRONG ALT 2.
3. Asserts the three are demonstrably distinct, and that the implementation's `sigma_c` equals `mean(rho^2)` while being **not** `std(rho)` and **not** `(mean rho)^2`.

The test therefore fails if the implementation accidentally reverts to `std(rho)` or to `(mean rho)^2`.

### 2.5 `execution_authorization_6e20_2_DRAFT.json`

Updated `runner_sha256` and `test_suite_sha256` only; the file remains a `_DRAFT` (non-authorizing) artifact.

**Functions/tests affected:** `compute_live_rho_c` (sigma_c line only); test module import; TEST D. No other function, control, or behavior was modified.

---

## 3. Protocol Conformance

The implemented metric is now:

```
sigma_c = mean_i (rho_i^2)
```

This is exactly the frozen Protocol v3.1 §5.7 / V3-03 definition of `σ_c` (mean variance share). It is **NOT** `std(rho)` and **NOT** `(mean rho)^2`. The distinction is preserved explicitly in both code and documentation.

---

## 4. Synthetic Test Evidence

- **Total tests:** 16
- **Passed:** 16
- **Failed:** 0
- **Test count change:** NONE — TEST D was *replaced* (same number of records), so the suite remains 16/16. No unrelated test was added or removed.

**Exact synthetic sigma test behavior (`sigma_c_is_mean_variance_share`):**

The synthetic panel produces `per_sample_rho` with 8 values of `-1/sqrt(2) ≈ -0.7071` and 8 values of `+1/sqrt(2) ≈ +0.7071`. The three quantities are therefore demonstrably distinct:

| Quantity | Value |
|---|---|
| `mean(rho^2)` (CORRECT, Protocol) | `0.500000` |
| `std(rho)` (WRONG ALT 1) | `0.707107` |
| `(mean rho)^2` (WRONG ALT 2) | `0.000000` |

Returned `sigma_c = 0.500000` → equals `mean(rho^2)`; differs from `std(rho)` (0.707107) and from `(mean rho)^2` (0.000000). Test passes, and would fail on either wrong alternative.

---

## 5. Safety Verification

Explicitly confirmed for this remediation:

- **NO REAL CHECKPOINT LOADED** — no `torch.load` of any historical checkpoint; the test uses synthetic fakes only.
- **NO REAL MODEL LOADED** — no Qwen weights, no `.from_pretrained`, no downloads.
- **NO LIVE INFERENCE** — no evaluator execution; `compute_live_rho_c` invoked only with mock tensors.
- **NO REAL-MODEL AUTOGRAD** — autograd operates only on the synthetic `FakeModel` parameter.
- **NO EVALUATOR EXECUTION** — `run_phase_6e20_reevaluation.py` was not executed end-to-end (no `--execute`, no Stage 1–7 run).
- **NO L5 AUTHORIZATION CREATED** — no `execution_authorization_6e20_2.json` was created or activated.
- **NO FROZEN 6E.20.1 ARTIFACT MODIFIED** — `run_phase_6e20_execution_v*.py`, conformance harnesses, `runs.json`, historical manifests, frozen runner, Protocol v3.1, Plan v4.0, frozen 6E.18 machinery, `execution_authorization_v6.json`, and `execution_authorization_v7.json` were all left untouched. The v6/v7 identity provenance was not revisited.

---

## 6. New SHA-256 Values

| File | SHA-256 |
|---|---|
| `run_phase_6e20_reevaluation.py` | `58d6cb9331e5e092678046ffc756aa24578335f6f63f131e89707678a20a2a9f` |
| `test_phase_6e20_reevaluation.py` | `0b4a9a518a644fe8391a639173ad8f6a4ee257d3576dfc6482e20e90aadc7c3c` |

These values were written into `execution_authorization_6e20_2_DRAFT.json` (`runner_sha256` and `test_suite_sha256`) and **verified to match** the modified files' recomputed hashes. The DRAFT remains suffixed `_DRAFT` and therefore non-authorizing.

Other DRAFT anchor hashes unchanged (Protocol `E16CA4F1…`, Plan `87C8E486…`, 6E.18 `6BCAF131…`).

---

## 7. Scope Verification

This remediation was strictly limited to the single finding `6E20.2-SIGMA-PROTOCOL-MISMATCH-01`:

- Modified: only the `sigma_c` aggregation line + its docstring in `run_phase_6e20_reevaluation.py`, the TEST D test in `test_phase_6e20_reevaluation.py`, and the DRAFT hash fields.
- **Not** modified: `rho_c` computation, `g_total`/`g_R`/`alpha_c`/`psdg_fn` integration, `compute_readout_total_alignment`, LoRA space representation, gradient guards, Stage 6 condition→readout binding, B0/S8 anchor isolation, LoRA restoration validation, authorization ordering, output isolation, immutability protections, 56-vs-28 scope, H1/H4/H5/H3 decision logic, or any frozen 6E.20.1 artifact.

No unrelated remediation was performed.

---

## 8. Final Status

IMPLEMENTATION REMEDIATED — SIGMA_c NOW MATCHES FROZEN PROTOCOL v3.1 — NOT AUTHORIZED FOR EXECUTION — FRESH INDEPENDENT PRE-AUTHORIZATION SOURCE AUDIT REQUIRED BEFORE L5 REVIEW.
