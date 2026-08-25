# PHASE 6E.20.2 — TARGETED REMEDIATION REPORT

## 1. Execution Boundary

| Activity | Performed? |
|---|---|
| Source inspection (Protocol v3.1 §5.7, frozen 6E.18 `per_sample_decision_gradient`) | YES (READ-ONLY) |
| Static analysis / syntax check (`py_compile` of both files) | YES |
| File hashing (`Get-FileHash` SHA-256) | YES |
| Synthetic unit tests (`test_phase_6e20_reevaluation.py`, 16/16 pass) | YES |
| Live 28-checkpoint evaluation (`--execute`) | **NO** |
| Real Qwen2.5-0.5B model loading / `from_pretrained` | **NO** |
| `torch.load` on any historical checkpoint | **NO** |
| L5 authorization file created (`execution_authorization_6e20_2.json`) | **NO** |
| Any frozen 6E.20.1 artifact modified | **NO** |

Explicitly confirmed: **NO LIVE CHECKPOINT EVALUATION**, **NO REAL MODEL LOADING**, **NO L5 AUTHORIZATION**, **NO FROZEN ARTIFACT MODIFICATION**.

## 2. Files Modified

**File:** `run_phase_6e20_reevaluation.py`
**Why modified:** Remediate Audit Finding 1 (degenerate `rho_c`) and document Finding 4 (provenance).
**Functions changed:**
- Added `compute_readout_total_alignment(g_R_flat, g_total_flat)` — pure, guarded per-sample cosine core (common LoRA-parameter space).
- Rewrote `compute_live_rho_c(...)` — now computes genuine `g_total` (frozen `psdg_fn`) and `g_R = alpha_c·∇_θ s` in LoRA-parameter space; signature changed to `(model, readout, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str)`.
- Updated Stage-6 caller (line ~855) to pass `psdg_fn, crgc_fn, device_str` instead of `margin_metrics["H16_eval"]`.
- Added explicit comment documenting the intentional v6→v7 auth SHA mapping (line ~292). No logic change.

**File:** `test_phase_6e20_reevaluation.py`
**Why modified:** Remediate Audit Finding 3 (test coverage gap for critical compute paths).
**Functions changed:** Added imports (`torch`, `compute_readout_total_alignment`); added Tests 10 (rho core A–F) and 11 (cos(H1) core G).

**File:** `execution_authorization_6e20_2_DRAFT.json`
**Why modified:** §8 — keep DRAFT package consistent with remediated artifacts.
**Functions changed:** Updated `runner_sha256` and `test_suite_sha256` to post-fix values. Decision remains `APPROVE`; file name remains `_DRAFT` (non-authorizing). The active `execution_authorization_6e20_2.json` was NOT created.

## 3. rho_c Remediation

**Old defective data flow (hidden-state space, self-projection):**
```
w_vec   = readout.w
u_w     = w_vec / ||w_vec||
for i:  s_i = forward_margin(h_i); alpha_i = (p-1) if POS else p
        g_R_hidden = alpha_i * w_vec
        alignment   = dot(g_R_hidden, u_w) / ||g_R_hidden||   # = sign(alpha_i) ∈ {+1,-1}
rho_c   = mean(alignment) ≈ 0.0 ;  sigma_c = std(alignment)
```
This measured `cos(g_R, readout_direction)` — a constant — never touching `g_total`.

**New g_R data flow (LoRA-parameter space, Protocol §5.7):**
```
for each sample i (8 POS + 8 ABS):
  s_i        = readout scalar z_PRO at decision position (from forward)
  p_PRO,p_ABS= softmax over PATCHED full-vocab decision logits
  alpha_c    = (p_PRO - y_PRO) - (p_ABS - y_ABS)      # y=(1,0) POS, (0,1) ABS
  g_R_i      = alpha_c * autograd.grad(s_i, lora_params)   # readout-only decision gradient
```

**New g_total data flow (LoRA-parameter space, frozen 6E.18 machinery):**
```
  g_total_i  = psdg_fn(model, readout, item_i, lora_params, tokenizer)
             = autograd.grad(dec_loss, lora_params)          # total decision gradient
```

**Common gradient space:** Both `g_R_i` and `g_total_i` are flat 1-D vectors over the **SAME ordered `lora_params`** list (identical ordering, verified by construction). No unrelated tensor is silently flattened; the cosine is mathematically well-defined.

**Final rho_i formula:** `rho_i = <g_R_i, g_total_i> / (||g_R_i|| · ||g_total_i||)`, with `rho_i = 0` if `||g_R_i|| == 0` (Protocol §5.7).

**rho_c formula:** `rho_c = mean_i rho_i` over the 16 panel samples (aggregation set {4,6,14,15} × conditions handled by the caller loop).

**sigma_c formula:** `sigma_c = std_i rho_i` (per explicit task §2 / §6 TEST D). *(See §4 discrepancy note.)*

**Can rho_c still be mathematically forced to 1.0, 0.0, or ±1 solely by implementation structure?**
**NO.** `g_R` is `alpha_c·∇_θ s`, a rich gradient in LoRA space that is generally *not* parallel to `g_total` (which also contains the free LM-head background term per Protocol §5.7). The value 1.0/−1.0 can occur only as a *genuine* scientific outcome (total gradient parallel to readout direction); 0.0 can occur only if `g_R` genuinely vanishes. The algebra no longer structurally forces any of these. Guards reject NaN/Inf, zero-norm, shape mismatch, and non-tensor inputs (they HALT via `PreflightError`, never silently substituting 0/1/NaN or `g_R` for `g_total`).

## 4. Frozen 6E.18 / Protocol Conformance

- **Protocol v3.1 §5.7** (frozen, SHA `E16CA4F1…`) is the authoritative definition: `rho_c^{(x)} = <g_R^{(x)}, g_total^{(x)}> / (||g_R||·||g_total||)`, `g_R = alpha_c·∇_θ s`, `g_total = g_bg + g_R` (total `autograd.grad(dec_loss, lora_param_list)`), `alpha_c = (p_PRO−y_PRO)−(p_ABS−y_ABS)`. The new code implements exactly this data flow.
- **Frozen 6E.18 machinery** (`per_sample_decision_gradient`, snapshot SHA `6BCAF131…`) supplies `g_total` verbatim — the same function `compute_live_cos_H1` already used. `g_R` is computed by an additional `autograd.grad(s, lora_params)` call, matching §5.7's `∇_θ s = autograd.grad(s, lora_param_list)`.
- **Space selection:** Protocol §5.7 defines both objects in **LoRA-parameter gradient space** (Object 1 explicitly states `g_R^θ = alpha_c·∇_θ s`). The old code erroneously used hidden-state space (`alpha·w` vs `w`), which is mathematically degenerate — the parameter-space formulation is the correct, non-degenerate one and is what was implemented.
- **sigma_c discrepancy (flagged):** Task §2 and TEST D specify `sigma_c = std(rho_i)`. Protocol §5.7 defines `σ_c` as the *mean variance share* `mean(rho_i²)`. The implemented aggregation follows the explicit task instruction (`std`). H3 verdicts use only the `rho_c` **mean** (not `sigma_c`), so no scientific verdict is affected. **This wording discrepancy is flagged for L5 reconciliation** — recommend adopting Protocol §5.7's `mean(rho²)` before sign-off if the protocol definition is to be the single source of truth (per the task's FINAL RULE).

## 5. Test Coverage Added

New tests exercise the **pure** `compute_readout_total_alignment` core (synthetic vectors; no model/checkpoint) and the frozen `compute_raw_gradient_cosine` core (lazy import of the frozen snapshot; no real model).

| Test | What it verifies | Historical failure prevented | Synthetic expected outcome |
|---|---|---|---|
| A `rho_genuine_alignment` | `cos(g_R,g_total)` with `g_R≠g_total` matches `dot/||·||` | wrong cosine arithmetic | `0.707107` |
| B `rho_rejects_self_projection` | new formula yields graded value, not old `sign(alpha)`∈{±1} | degenerate self-projection (the Audit-B bug) | `0.707107` (≠ +1/−1) |
| C `rho_non_constant` | `rho_c` = genuine mean cosine, not forced to 0 | constant `≈0.0` collapse | `0.569036` |
| D `rho_sigma_std` | `sigma_c` = std of genuine per-sample rhos | invalid `sigma_c` from invalid inputs | `0.419760` |
| E `rho_zero_norm_protection` | zero-norm operands → `0.0`, not NaN/Inf | silent NaN/Inf scientific value | `0.0, 0.0` |
| F `rho_shape_and_type_guards` | shape mismatch & non-tensor → `PreflightError` | silent invalid substitution | both halt |
| G `cos_H1_core_sign_gate` | raw cosine, no label-sign inversion (core of `compute_live_cos_H1`) | sign-inversion / abs distortion | `opp=-1.0, align=+1.0` |

All 16 tests pass (11 prior + 5 above; TEST G loads the frozen 6E.18 cosine core via the snapshot, no model).

## 6. Provenance Remediation

**Finding:** Stage 1 checks `execution_authorization_v6.json` against `AUTH_6E20_1_V7_SHA256`.
**Investigation (READ-ONLY) of the frozen record:** The v6 auth file's own `note` field states verbatim: *"the runner reads execution_authorization_v6.json (identical content, v7-bound)"* and *"Supersedes v6 auth 4AA21378… (must not be reused)."* Both `execution_authorization_v6.json` and `execution_authorization_v7.json` hash to `5214DA99…` (byte-identical), confirming the v6 file is **intentionally v7-bound** by the frozen 6E.20.1 design.

**Resolution: NONE of Option A/B/C was required** — the mapping is **correct**, not a bug. It was therefore **NOT changed**; instead it is now **explicitly documented** in code (comment at line ~292) so it is no longer "silent." The invariant verified is: *the v6 auth artifact is v7-bound; its expected SHA equals the v7 SHA; altering it will HALT Stage 1 (fail-closed).* The prior audit's "anomaly" flag was a false positive resolved by reading the frozen record. No frozen artifact was modified.

## 7. Regression Preservation

| Control | Status |
|---|---|
| Condition → checkpoint → readout binding (B0→B0, …, R1-Rand→R1-Rand) | **UNCHANGED** (Stage 6 loop intact) |
| Historical B0/S8 anchor isolation | **UNCHANGED** — anchor still hard-codes `readout_s8` only inside `run_stage5_b0_historical_reproducibility_anchor`; does not leak into Stage 6 |
| Statement "PASSING THE B0/S8 HISTORICAL ANCHOR DOES NOT SUPPORT OR REFUTE H1" | **UNCHANGED** (docstring preserved) |
| 56-checkpoint frozen universe vs 28-checkpoint 6E.20.2 input set | **UNCHANGED** |
| LoRA safety (adapter-key validation, missing-key detection, condition/step metadata, non-zero norm, `strict=False` protected by required-key checks) | **UNCHANGED** (Stage 3&4 intact) |
| Immutability paths (all outputs → `phase-6e20.2-reevaluation-output/`, Stage 7 re-manifest) | **UNCHANGED** |
| Authorization ordering (auth → runner-SHA → integrity → model → checkpoint → anchor → evaluation) | **UNCHANGED** |
| Decision thresholds (P1–P5, H4_DELTA_MIN) | **UNCHANGED** |

## 8. Hashes

| Artifact | SHA-256 (uppercase) |
|---|---|
| `run_phase_6e20_reevaluation.py` | `EFC692F03E3A5E1C0A678E146F84FAA8FED0EE16DDA407220F9551441406DF3F` |
| `test_phase_6e20_reevaluation.py` | `369BC80630B5411CE6B1E159BE72F1A19601BF8014A65FDD0236C5CE7556676F` |

DRAFT authorization package (`execution_authorization_6e20_2_DRAFT.json`): **updated** — `runner_sha256` and `test_suite_sha256` set to the values above. The active `execution_authorization_6e20_2.json` was **NOT created**; the `_DRAFT` file remains non-authorizing.

## 9. Validation Performed

- **Syntax / static:** `python -m py_compile run_phase_6e20_reevaluation.py test_phase_6e20_reevaluation.py` → `COMPILE_OK`.
- **Synthetic tests:** `python test_phase_6e20_reevaluation.py` → `all_pass: true, passed: 16, total: 16`.
- **Hash checks:** `Get-FileHash` SHA-256 (read-only) on both files; DRAFT auth updated to match.
- **No live scientific metrics** were produced or reported (no `--execute`, no model, no `torch.load` of checkpoints).

## 10. Remaining Status

IMPLEMENTATION REMEDIATED — NOT AUTHORIZED FOR EXECUTION.
A NEW INDEPENDENT PRE-AUTHORIZATION SOURCE AUDIT IS REQUIRED BEFORE L5 REVIEW.
