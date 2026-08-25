# FORENSIC 6E.20.2 — POST-SIGMA-REMEDIATION FINAL PRE-AUTHORIZATION SOURCE AUDIT

**Audit ID:** 6E20.2-POST-SIGMA-FINAL-AUDIT
**Date:** 2026-08-19
**Phase:** 6E.20.2 — Read-Only Checkpoint Re-Evaluation
**Mode:** FINAL SOURCE AUDIT ONLY (fresh, independent, post-remediation)
**Execution Status:** NOT AUTHORIZED
**Active L5 Authorization:** MUST NOT EXIST (confirmed absent)
**Protocol baseline:** frozen Protocol v3.1 (`E16CA4F1…`), §5.7 (line 182) and V3-03 (line 392): `σ_c = mean_i(ρ_i²)`.

---

## 1. Audit Scope

**Inspected (source-only):**
- `run_phase_6e20_reevaluation.py` (current, hash `58d6cb93…`) — `compute_live_rho_c` (lines 615–701), Stage 6 loop (948–998), decision engine (760–794), readout reconstruction (910–921), Stage 3/4 LoRA validation (380–445), authorization gating (843–859, 1061), immutability/Stage 7 (1000–1006), output isolation (44, 875–1011).
- `test_phase_6e20_reevaluation.py` (current, hash `0b4a9a51…`) — Test 10 (A–F) and Test 11 (G) including the new TEST D (318–376).
- `execution_authorization_6e20_2_DRAFT.json` — hash fields and non-authorizing status.
- Prior reports (initial audit, rho_c remediation, post-rho audit, sigma_c remediation) for traceability only.

**Validation run:**
- Static safety proof of the test suite (no checkpoint/model/tokenizer loads, no file writes) — see Audit E.
- Execution of the existing synthetic-only test suite: **16/16 PASS**.

**Explicitly NOT executed:**
- `--execute` / Stage 1–7 pipeline.
- Any `torch.load` of historical checkpoints; any real base model / Qwen weights / downloads.
- Any real inference or autograd on a real model.
- Creation, renaming, or activation of L5 authorization.
- Modification of any code, test, DRAFT, report, or frozen 6E.20.1 / 6E.18 / Protocol artifact.

---

## 2. Executive Verdict Table

| ID | Audit | Verdict | Severity if FAIL |
|----|-------|---------|------------------|
| A | Condition → Checkpoint → Readout binding | **PASS** | — |
| B | Genuine `rho_c` semantics | **PASS** | — |
| C | Exact `sigma_c` protocol semantics | **PASS** | — |
| D | Test-suite coverage of critical semantics | **PASS** | — |
| E | Self-test safety | **PASS** | — |
| F | Regression audit (critical controls) | **PASS** | — |
| G | Frozen protocol conformance | **PASS** | — |
| H | Provenance / frozen records | **PASS** | — |
| I | Hash and DRAFT authorization consistency | **PASS** | — |

---

## 3. Condition Binding Audit

Normal scientific evaluation path (Stage 6, lines 955–994):
- `readout_c = readouts[c]` (line 957) — each condition uses **its own** reconstructed readout.
- `ckpt_path = paths["historical_out_dir"] / c / f"checkpoint_{t}" / "model.pt"` (line 961) — each condition evaluated against **its own** checkpoint.
- Readouts are reconstructed per-condition from `step0/w_{c}.npy` (lines 915–921); no global fallback.
- The only `readouts["S8"]` reference (line 938) is inside **Stage 5 B0 Historical Reproducibility Anchor**, an isolated validation gate (invariant: "anchor validation only, not evidence for H1"). It is NOT a fallback in the normal scientific evaluation. No `w_all["S8"]`, no `B0`/`S8` global substitution in Stage 6.

| Condition | Checkpoint source | Readout source | Binding | Verdict |
|-----------|------------------|----------------|---------|---------|
| B0 | `historical_out_dir/B0/checkpoint_t` | `readouts["B0"]` (step0/w_B0.npy) | own | PASS |
| F1 | `historical_out_dir/F1/checkpoint_t` | `readouts["F1"]` | own | PASS |
| S2 | `historical_out_dir/S2/checkpoint_t` | `readouts["S2"]` | own | PASS |
| S4 | `historical_out_dir/S4/checkpoint_t` | `readouts["S4"]` | own | PASS |
| S8 | `historical_out_dir/S8/checkpoint_t` | `readouts["S8"]` | own | PASS |
| C4-Rand | `historical_out_dir/C4-Rand/checkpoint_t` | `readouts["C4-Rand"]` | own | PASS |
| R1-Rand | `historical_out_dir/R1-Rand/checkpoint_t` | `readouts["R1-Rand"]` | own | PASS |

No cross-condition contamination. **AUDIT A: PASS.**

---

## 4. rho_c Scientific Semantics Audit

**g_total source (line 681):**
```python
g_total = psdg_fn(model, readout, item, lora_params, tokenizer)
```
Genuine total decision gradient via the **frozen 6E.18 F7 machinery** (`per_sample_decision_gradient`), recomputed from the restored live model in LoRA-parameter space. Correct.

**g_R source (lines 684–691):**
```python
grads_s = torch.autograd.grad(s, lora_params, allow_unused=True, retain_graph=False)
...
g_R = alpha_c * torch.cat(flat_s)   # flat_s = flattened grads of s w.r.t. lora_params
```
`g_R = α_c ∇_{θ} s` where `s = z_PRO` (readout scalar at decision position). Correct.

**alpha_c (lines 674–678):**
```python
delta = p_pro - p_abs
alpha_c = delta - 1.0   # POS panels, y=(1,0)
alpha_c = delta + 1.0   # ABS panels, y=(0,1)
```
Correct decision-loss chain rule `(p_PRO − y_PRO) − (p_ABS − y_ABS)`.

**Common parameter space:** `g_total` (from `psdg_fn`) and `g_R` (from `∇_{lora_params}(s)`) are both expressed over the **same ordered `lora_params` set**; `g_R` is built by concatenating the per-parameter flattened gradients in that exact order. Cosine is well-defined.

**Alignment (line 694):** `rho_i = compute_readout_total_alignment(g_R, g_total)` = `⟨g_R, g_total⟩ / (‖g_R‖·‖g_total‖)`.

**Aggregation (line 699):** `rho_c = float(np.mean(per_sample_rho))` = `mean_i(ρ_i)`.

**Rejected constructs (none present):** no readout self-projection, no `cos(g_R, g_R)`, no `g_total = g_R`, no alignment with normalized `w`, no readout-only geometry, no analytic constants, no structural forcing to 1/0/±1 (the shared-space cosine with guards at lines 547–590 prevents it).

**Structural self-projection possible?** **NO.** `rho_c` requires genuine restored-model gradient information from both `psdg_fn` (g_total) and autograd through the live readout (g_R); neither is degenerate or replaced by a constant.

**AUDIT B: PASS.**

---

## 5. sigma_c Protocol Semantics Audit

**Exact current source (line 699):**
```python
sigma_c = float(np.mean(np.asarray(per_sample_rho, dtype=np.float64) ** 2))
```
This computes exactly `σ_c = mean_i(ρ_i²)` (mean variance share), matching frozen Protocol v3.1 §5.7 / V3-03.

**Comparison:**
- `mean(ρ²)` (CORRECT) — implemented. For the synthetic audit panel: `mean(ρ²) = 0.500000`.
- `std(ρ)` (FAIL alt) — `0.707107`; **not** implemented.
- `(mean ρ)²` (FAIL alt) — `0.000000`; **not** implemented.

The docstring (lines 615–617) now explicitly states `σ_c = mean_i(ρ_i²)` and "NOT std(rho); NOT (mean rho)**2".

**TEST D verification (lines 318–376):** invokes the **real** `compute_live_rho_c` with purely synthetic fakes, computes all three quantities, and asserts the returned `sigma_c` equals `mean(ρ²)` while being **not** `std(ρ)` and **not** `(mean ρ)²`. Observed: `sigma_c=0.500000`, distinct from `std(rho)=0.707107` and `(mean rho)^2=0.000000`; test PASS.

**AUDIT C: PASS** (mean(rho_i²), not std, not (mean)²; TEST D distinguishes all three).

---

## 6. Synthetic Test Coverage Audit

| Critical property | Covered by | Status |
|---|---|---|
| Genuine readout-total alignment | TEST A | ✔ |
| Rejection of old self-projection degeneracy | TEST B | ✔ |
| Non-constant alignment behavior | TEST C | ✔ |
| `rho_c = mean(rho_i)` | TEST C | ✔ |
| `sigma_c = mean(rho_i²)` | TEST D | ✔ |
| Sigma distinct from `std(rho)` | TEST D | ✔ |
| Sigma distinct from `(mean rho)²` | TEST D | ✔ |
| Zero-norm handling | TEST E | ✔ |
| Shape mismatch handling | TEST F | ✔ |
| Non-tensor / invalid input handling | TEST F | ✔ |
| Core `cos(H1)` path | Test 11 (G) | ✔ |

**Would the two previously found defects now be caught?**
- *Degenerate `rho_c` (old self-projection / constant):* YES — TEST B asserts a graded non-constant value and rejects the `{+1,−1}` self-projection regime.
- *`sigma_c = std(rho)` mismatch:* YES — TEST D asserts `mean(ρ²)` and fails if the implementation returns `std(ρ)` or `(mean ρ)²`.

**AUDIT D: PASS.**

---

## 7. Regression and Safety Audit

All previously-passing critical controls re-verified against current source:

- **Condition/checkpoint/readout binding** — Stage 6 `readout_c = readouts[c]`, `ckpt = historical_out_dir/c/checkpoint_t` (Audit A). ✔
- **S4/S8 own-readout semantics** — same binding; no substitution. ✔
- **B0/S8 historical anchor isolation** — Stage 5 uses `readouts["S8"]` as a validation gate only; not injected into scientific verdicts. ✔
- **LoRA missing-key protection despite `strict=False`** — Stage 4 explicitly computes `missing_lora = expected_lora_keys − ckpt_lora_keys` and HALTs on any (lines 408–418), before the `strict=False` load (421). ✔
- **Condition/step metadata validation** — `ckpt["step"]`, `ckpt["condition"]` checked against `t`/`c` (403–406). ✔
- **Nonzero LoRA validation** — adapter tensor norm check (435–436). ✔
- **Authorization before checkpoint/model access** — `check_authorization(this_file)` (line 1061) raises `EXECUTION_NOT_AUTHORIZED` if the active auth file is missing / `decision != APPROVE` / `phase` mismatch; currently no active file exists → execution blocked. ✔
- **Runner SHA validation** — Stage 1 pre-execution and Stage 7 post-execution SHA manifests (885, 1000–1005). ✔
- **DRAFT vs active separation** — active `execution_authorization_6e20_2.json` absent; only `_DRAFT` exists (non-authorizing). ✔
- **Output isolation** — all outputs written under `phase-6e20.2-reevaluation-output/` (DEFAULT_OUTPUT_DIR, line 44). ✔
- **Pre/post immutability manifests** — Stage 7 confirms zero modification of frozen 6E.20.1 files (1006). ✔
- **No frozen writes** — evaluator writes only to its own output dir. ✔
- **56-checkpoint frozen universe vs 28-input evaluation scope** — `AGGREGATION_SET = [4,6,14,15]` (line 78) × 7 conditions = 28 evaluations; only those checkpoints are read. ✔
- **H1/P2 aggregation over {4,6,14,15}** — means computed over `AGGREGATION_SET` (767–770); `p2_s4/p2_s8 = means_cos[S4/S8] >= 0.80` (774–777). ✔
- **H4 logic** — `delta_s4_c4 >= H4_DELTA_MIN AND C4-Rand fails P3` (779–783). ✔
- **H5 logic** — `R1-Rand fails P3 AND F1 passes P3` (785–789). ✔
- **H3 boundary logic** — `rho_S8 >= 0.50 AND rho_S8 >= rho_S4 >= rho_S2` (792–794). ✔
- **NaN/Inf/zero-norm/missing-value guards** — `compute_readout_total_alignment` guards (547–590): non-tensor, shape mismatch, non-1-D, non-finite norm, zero-norm → 0.0, non-finite alignment. ✔

**AUDIT F: PASS.**

---

## 8. Hash and Authorization Audit

**Independently computed SHA-256:**
| File | Computed | DRAFT field |
|------|----------|-------------|
| `run_phase_6e20_reevaluation.py` | `58d6cb9331e5e092678046ffc756aa24578335f6f63f131e89707678a20a2a9f` | `runner_sha256` — MATCH |
| `test_phase_6e20_reevaluation.py` | `0b4a9a518a644fe8391a639173ad8f6a4ee257d3576dfc6482e20e90aadc7c3c` | `test_suite_sha256` — MATCH |

- DRAFT filename retains `_DRAFT` suffix → non-authorizing by construction. ✔
- No active `execution_authorization_6e20_2.json` exists (`Test-Path` = False). ✔
- Old auth hash `4AA21378…` is **not** referenced in the DRAFT. ✔
- Other DRAFT anchors unchanged (Protocol `E16CA4F1…`; Plan `87C8E486…`; 6E.18 `6BCAF131…`).

**AUDIT I: PASS.**

---

## 9. Findings Requiring Code Changes

**NO SOURCE-LEVEL CORRECTIONS REQUIRED BEFORE FREEZE.**

All nine audits PASS. The two previously identified defects (degenerate `rho_c`; `sigma_c = std(rho)`) are resolved in source, covered by tests, and verified by this independent audit. No UNKNOWN critical properties remain.

---

## 10. Final Authorization Recommendation

**C. READY TO FREEZE FOR L5 REVIEW**

No critical FAIL or unresolved critical UNKNOWN remains. The current implementation conforms to frozen Protocol v3.1 §5.7 for both `ρ_c = mean(ρ_i)` and `σ_c = mean(ρ_i²)`, preserves all L5 controls, and is safe to present for explicit human L5 sign-off.

**Do not create L5 authorization. Do not execute the evaluator.** This audit grants no authorization; only explicit human L5 sign-off (activating `execution_authorization_6e20_2.json` from the verified DRAFT) may proceed to execution.

*End of Post-Sigma-Remediation Final Pre-Authorization Source Audit — 6E.20.2.*
