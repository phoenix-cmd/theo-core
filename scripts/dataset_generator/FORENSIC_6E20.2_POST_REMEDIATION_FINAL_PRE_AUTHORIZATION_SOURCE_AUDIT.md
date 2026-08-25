# Forensic Pre-Authorization Source Audit — Phase 6E.20.2 Stage 2 Vocabulary Remediation

**Audit mode:** SOURCE AUDIT ONLY (no execution, no checkpoint load, no model inference/autograd, no auth creation/modification).
**Scope:** Remediated 6E.20.2 runner + synthetic test suite + DRAFT hash update only.
**Date:** 2026-08-19
**Objective:** Determine whether the Stage 2 vocabulary remediation resolves the halted-execution defect (Classification B) without regression or invariant weakening.

All findings below are derived from direct on-disk source reads, recomputed SHA-256 hashes, and a fresh re-run of the synthetic suite — not from prior claims.

---

## A. Stage 2 Semantic Correctness

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| A1 | `VOCAB = 151936` unchanged | **PASS** | `run_phase_6e20_reevaluation.py:67` `VOCAB = 151936` |
| A2 | Authoritative invariant is `model.config.vocab_size == 151936` | **PASS** | helper `validate_stage2_vocab_invariant` first check `model_config_vocab_size != VOCAB` (lines 354–357); called with `model.config.vocab_size` at line 376 |
| A3 | `tokenizer.vocab_size == 151643` and `len(tokenizer) == 151665` treated as SEPARATE tokenizer metrics | **PASS** | new constants `TOKENIZER_BASE_VOCAB = 151643`, `TOKENIZER_FULL_VOCAB = 151665` (lines 68–69); checked independently against 151643/151665 (lines 358–365); passed as `tokenizer.vocab_size`, `len(tokenizer)` (line 376) |
| A4 | None of the tokenizer metrics are compared against 151936 | **PASS** | helper compares only `model_config_vocab_size` to `VOCAB`; tokenizer metrics compared only to `TOKENIZER_BASE_VOCAB`/`TOKENIZER_FULL_VOCAB`; no `tokenizer.* != VOCAB` construct present |
| A5 | Checks fail-closed with `PreflightError` | **PASS** | all three invariant checks `raise PreflightError` (lines 355, 359, 363) |

## B. Root-Cause Remediation

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| B1 | New implementation directly fixes Classification B (measurement/API semantic defect) | **PASS**; the comparison target was changed from a tokenizer metric (`len(tokenizer)`) to the model embedding vocabulary (`model.config.vocab_size`) — exactly the defect mechanism |
| B2 | Known-valid pinned config passes: model 151936, base 151643, full 151665 | **PASS**; synthetic test `stage2_valid_config_passes` → PASS (`validate_stage2_vocab_invariant(151936,151643,151665)` raises nothing) |
| B3 | Old invalid comparison cannot remain on any active Stage 2 path | **PASS**; AST/static check confirms `len(tokenizer) != VOCAB` is absent; `run_stage2_model_and_tokenizer_preflight` now routes exclusively through `validate_stage2_vocab_invariant` (line 376) |

## C. Test Adequacy

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| C1 | Test 12 (all 5 sub-cases) genuinely exercises the real `validate_stage2_vocab_invariant` path | **PASS**; source read lines 437–487: 12a–12d call the real function directly with scalar metrics; 12e encodes the old semantics |
| C2 | Tests would fail under old defective semantics | **PASS**; under old code there was no `validate_stage2_vocab_invariant` (import would fail), and 12a's valid config would have been rejected by the old `len(tokenizer) != VOCAB` check; 12e explicitly demonstrates the old comparison rejects the valid config |
| C3 | Complete suite result consistent with claimed 21/21 PASS | **PASS**; independent re-run: `ALL_PASS True | 21 / 21`, FAILURES: none |

## D. Regression Audit (scientifically critical controls)

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| D1 | Genuine `rho_c = mean(cos(g_R, g_total))` in common LoRA parameter space | **PASS**; `compute_live_rho_c` (617–725): `g_total = psdg_fn(...)` (frozen 6E.18), `g_R = alpha_c * grad(s)`; both flattened over identical `lora_params` ordering (705, 708–715); `rho_c = mean(per_sample_rho)` (723) |
| D2 | `sigma_c = mean(rho_i²)` | **PASS**; line 724 `sigma_c = float(np.mean(np.asarray(per_sample_rho, dtype=np.float64) ** 2))`; synthetic test `sigma_c_is_mean_variance_share` → PASS (distinct from `std(rho)` and `(mean rho)²`) |
| D3 | Condition-resolved own-checkpoint + own-readout binding | **PASS**; Stage 6 (979–987): `readout_c = readouts[c]`; `ckpt_path = historical_out_dir / c / f"checkpoint_{t}" / "model.pt"`; no fallback path |
| D4 | Historical B0/S8 anchor isolation and gate ordering | **PASS**; Stage 5 (728–776): B0+S8-only recompute vs `HISTORICAL_B0_S8_ANCHOR`; explicit "PASSING THIS GATE DOES NOT SUPPORT H1" notice; fail-closed tolerance gate (`B0_REPRODUCIBILITY_TOLERANCE`) |
| D5 | Non-silent LoRA restoration | **PASS**; Stage 3/4 (428–460): `missing_lora` raises Stage-4 HALT; all-zero adapter-norm check raises; `loaded_count` tracked |
| D6 | SHA-256 frozen-artifact manifest | **PASS**; Stage 1 (279–342) verifies protocol/plan/6E.18/runner-bases/auth/corpus + step0 + checkpoint existence with expected hashes, fail-closed; recomputed hashes match recorded values (see F1) |
| D7 | Authorization guard ordering | **PASS**; `check_authorization` (867–890): exists → parse → `decision==APPROVE` → `phase==PHASE_ID` → `runner_sha256` match; all branches `PreflightError` |
| D8 | Output isolation and post-execution immutability check | **PASS**; Stage 7 (1024–1030) re-runs the Stage-1 SHA manifest post-execution; all outputs written under `out_dir_path` (phase-6e20.2-reevaluation-output) |
| D9 | H1/H3/H4/H5 decision logic | **PASS**; `evaluate_protocol_hypotheses` (782–861) implements verbatim Protocol v3.1 §7/§8/§9 rules for H1 (P2_S4&P2_S8), H4 (ΔS4−C4≥0.30 & C4 fails P3), H5 (R1 fails P3 & F1 passes P3), H3 (rho_S8≥0.50 & monotonic) |
| D10 | 28-evaluation aggregation scope from frozen universe | **PASS**; `AGGREGATION_SET = [4, 6, 14, 15]` (line 80); 7 CONDITIONS × 4 = 28; no scope change |

## E. Authorization Binding

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| E1 | Remediated runner SHA-256 on disk | **PASS**; `16BEF31E56D781170021092052CF708A92875EE68697FEB81FB77764E15E691C` (recomputed) |
| E2 | Remediated test SHA-256 on disk | **PASS**; `55A71B4115BA050FA22C3E83290A6B5D4C1FB45977D6448EF24000E03E5530A1` (recomputed) |
| E3 | Existing active L5 authorization still binds old runner `58D6CB93…` | **PASS**; `execution_authorization_6e20_2.json` → `runner_sha256 = 58d6cb9331e5e092678046ffc756aa24578335f6f63f131e89707678a20a2a9f`; decision APPROVE, phase 6E.20.2 |
| E4 | Therefore it CANNOT authorize the remediated runner | **PASS**; `check_authorization` would raise `Runner SHA mismatch` because current runner (16BEF31E…) ≠ active auth runner (58D6CB93…); the active auth was not modified |
| E5 | No active authorization matching the new runner exists | **PASS**; only `execution_authorization_6e20_2_DRAFT.json` references `16BEF31E…`, and it is a DRAFT (non-authorizing). All other auth files are phase 6E.20.1 (runners E2628BDB…, 8EE1C8F4…, 65E6FDA8…, C8E298C8…, 1F5F7EF2…) — none match the new runner |

## F. Frozen Integrity

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| F1 | No 6E.20.1 frozen artifact, protocol, plan, checkpoint, provenance, or prior forensic/halt record modified | **PASS**; recomputed: protocol `E16CA4F1…` = expected, plan `87C8E486…` = expected, frozen 6E.18 machinery `6BCAF131…` = expected. Only the 6E.20.2 runner, 6E.20.2 test suite, the DRAFT (hash update only), and a NEW remediation report were touched. Prior halt record (`pre_execution_sha256_manifest.json`) and `FORENSIC_6E20.2_STAGE2_TOKENIZER_ROOTCAUSE.md` were not modified. Old auth `4AA21378…` not reused. |

---

## Summary Verdict

| Group | Result |
|-------|--------|
| A (Stage 2 semantics) | 5/5 PASS |
| B (Root-cause fix) | 3/3 PASS |
| C (Test adequacy) | 3/3 PASS |
| D (Regression controls) | 10/10 PASS |
| E (Authorization binding) | 5/5 PASS |
| F (Frozen integrity) | 1/1 PASS |

**No FAIL or UNKNOWN items.**

---

## Recommendation

**C — READY TO FREEZE FOR L5 REVIEW**

The Stage 2 fix is semantically correct (A), directly resolves Classification B (B), is exercised by adequate synthetic tests that would fail under the old defective semantics (C), leaves all prior scientifically critical controls intact and regression-free (D), is hash-consistent between on-disk artifacts and the DRAFT (E1/E2), and the existing active L5 authorization remains bound to the OLD runner hash and therefore cannot authorize the remediated runner — while no active authorization matching the new runner exists (E3–E5). Frozen 6E.20.1 / protocol / plan / checkpoint / provenance / prior halt records are unmodified (F).

**STOP.** No new L5 authorization was created or activated, and no `--execute` was performed. The remediated artifacts are submitted for explicit human L5 sign-off; a new authorization binding the post-remediation runner hash (`16BEF31E…`) must be issued only after this C verdict is accepted by the L5 authority, before any execution.
