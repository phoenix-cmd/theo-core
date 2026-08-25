# Runtime Drift Root-Cause Localization Report

- **Phase:** Post-6E.20.3 closure forensic investigation (authorized: `diagnostic_authorization_runtime_drift.json`)
- **Subject:** Divergence of recomputed B0 step-4 `cos(H1)` from the historical anchor `0.428355` under a `1e-4` tolerance
- **Disposition:** ROOT CAUSE LOCATED. The failure is a **training/inference regime mismatch (LoRA dropout active in the reproducibility evaluation)**, not a corruption, not GPU nondeterminism, and not a defect in the frozen 6E.18 scientific machinery.
- **Repair status:** NONE. This is a diagnostic-only phase. No frozen artifact, NEPR, tolerance, anchor, threshold, `evaluate_protocol_hypotheses`, or joint gate was modified.

---

## 1. Executive Summary

| Observation | Value |
|---|---|
| Historical anchor `B0` step-4 `cos(H1)` | `0.428355` |
| 6E.20.2 recomputation (Stage 5 halt) | `0.418701` (diff `9.65e-3`) |
| 6E.20.3 recomputation (Stage 5 halt) | `0.423468` (diff `4.89e-3`) |
| Tolerance `B0_REPRODUCIBILITY_TOLERANCE` | `1e-4` |
| Within-process repeat (train mode), step-4 | `0.420–0.428` band; diff up to `4.4e-3` |
| Within-process repeat (**eval mode**), step-4 | `0.428355` **exactly**, diff `0.0` (bit-identical across 3 independent processes) |
| First divergence boundary | LoRA dropout layer applied during the forward pass inside `per_sample_decision_gradient` (frozen 6E.18 machinery), **because the PEFT model is in `train()` mode** when `compute_live_cos_H1` is invoked at Stage 5 |
| Responsible factor | LoRA dropout (`LORA_DROPOUT = 0.05`) active during gradient computation — **Factor #3: inference mode** |
| GPU nondeterminism? | **Ruled out** (eval mode is bit-identical on the same CUDA device) |

**Conclusion:** The historical anchor `0.428355` was produced with **dropout disabled** (inference/eval regime). The 6E.20.x re-evaluation harness computes `cos(H1)` in **train mode** (dropout active), injecting stochasticity of ~±`4e-3` variance. A value drawn from a stochastic distribution with σ≈`2e-3`–`4e-3` cannot be reproduced to `1e-4`; the historical anchor is simply one (deterministic, dropout-off) draw. The `1e-4` gate is therefore being applied to a quantity whose legitimate sampling variance is ~40× the tolerance.

---

## 2. Scope & Authorization

- Authorized by `phase-6e20.3-reevaluation-output/diagnostic_authorization_runtime_drift.json` (L5 authority) for Phases A (within-environment repeatability), B (first-divergence instrumentation), and C (controlled-factor isolation).
- Binding hashes preserved: runner `f424ecf6…`, test `43889916…`, collect_nepr `e2ee8afe…`, frozen 6E.18 machinery `6BCAF131…`.
- Prohibitions honored: no modification of NEPR, the `1e-4` tolerance, the `0.428355` anchor, P1–P6 thresholds, `evaluate_protocol_hypotheses`, the joint gate, or any frozen 6E.20.1/6E.20.2 artifact. No averaging, best-of, anchor replacement, or correction factor.

---

## 3. Methodology

### Phase A — Within-environment repeatability matrix
`diagnose_runtime_drift.py` replicates the Stage-5 B0 loop **exactly** (same model load, PEFT wrap with `LORA_R=16, ALPHA=32, DROPOUT=0.05`, readout reconstruction, `compute_live_cos_H1` invocation) and records B0 step-1..4 (`HISTORICAL_B0_S8_ANCHOR` keys `{4,6,14,15}`) for:
- **Within-process repeat:** two full B0 passes in the same process (checkpoints reloaded, grads zeroed between passes).
- **Across-process repeat:** four independent process invocations (`run_A1..A4`, `run_probe`).

### Phase B — First-divergence boundary
Instrumented the evaluation at the model-forward / per-sample-gradient / aggregate-cosine boundaries. The decisive localization came from the train-vs-eval contrast in Phase C (see below); the boundary is at the forward pass, before any gradient aggregation.

### Phase C — Controlled-factor isolation (decisive)
`diagnose_drift_phaseC.py` invokes the **unmodified** `compute_live_cos_H1` (which internally applies `ensure_readout_class_patch()`) under two regimes, each with a within-process repeat:
- **TRAIN mode** (faithful to the real runner — `compute_live_cos_H1` never calls `model.eval()`): LoRA dropout `0.05` is **active**.
- **EVAL mode** (`model.eval()` before evaluation): LoRA dropout **disabled**.

Ran across three independent processes (`phaseC_evalmode_C1/C2/C3`). Interpretation: if TRAIN varies but EVAL is bit-identical, the factor is train-mode dropout; if EVAL also varies, the factor is GPU nondeterminism.

---

## 4. Evidence

### Phase A — the divergence is intrinsic to a single evaluation
Across 8 evaluations (4 processes × 2 within-process), **step-4** spans `0.420582` → `0.428513` (spread ≈ `8e-3`). The historical anchor `0.428355` lies **inside** this band. `within_process_same = False` for every run.

| Process | run1 step-4 | run2 step-4 | within-process diff |
|---|---|---|---|
| probe | 0.422461 | 0.426586 | 4.1e-3 |
| A1 | 0.424020 | 0.426454 | 2.4e-3 |
| A2 | 0.425641 | 0.424134 | 1.5e-3 |
| A3 | 0.422904 | 0.428513 | 5.6e-3 |
| A4 | 0.420582 | 0.423726 | 3.1e-3 |

The 6E.20.2 (`0.418701`) and 6E.20.3 (`0.423468`) recomputations are just two more draws from this same stochastic distribution. They differ from each other **and** from the anchor — exactly the signature of run-to-run stochasticity, not a deterministic code defect.

### Phase C — dropout is the responsible factor (decisive)
| Process | TRAIN step-4 (a) | TRAIN step-4 (b) | TRAIN within-diff | EVAL step-4 (a) | EVAL step-4 (b) | EVAL within-diff |
|---|---|---|---|---|---|---|
| C1 | 0.427624 | 0.423261 | **4.36e-3** | 0.428355 | 0.428355 | **0.0** |
| C2 | 0.422207 | 0.422000 | **2.07e-4** | 0.428355 | 0.428355 | **0.0** |
| C3 | 0.426668 | 0.423594 | **3.07e-3** | 0.428355 | 0.428355 | **0.0** |

- **TRAIN mode** (dropout active, as the real runner uses): step-4 varies run-to-run, every within-process diff exceeds `1e-4` (range `2.07e-4`–`4.36e-3`). The magnitude is itself stochastic (depends on the dropout mask draw).
- **EVAL mode** (dropout disabled): step-4 = `0.428355` **to the bit**, identical across all three independent processes (`within_diff = 0.0`).
- `0.428355` (eval-mode recomputation) **equals the historical anchor exactly**. The probability that a train-mode (dropout-on) draw lands on `0.428355` to 6 decimals by chance is negligible, so the anchor was demonstrably computed with dropout **off**.

`verdict_dropout_is_cause = True`, `verdict_gpu_nondeterminism = False` for all three processes.

---

## 5. Root-Cause Determination — First Divergence Boundary

**Boundary:** the LoRA dropout layer, applied during the **forward pass** inside `per_sample_decision_gradient` (frozen 6E.18 machinery, `run_phase_6e18_causal_intervention.py` lines 1008–1021), which feeds `hL` into `readout.forward_logits`.

**Why it is active:** `compute_live_cos_H1` (`run_phase_6e20_reevaluation.py:544`) is invoked by `run_stage5_b0_historical_reproducibility_anchor` (`run_phase_6e20_reevaluation.py:768`) **without ever calling `model.eval()`**. The PEFT model is in `train()` mode (default after `get_peft_model`, and never switched before Stage 5). `LORA_DROPOUT = 0.05` (`run_phase_6e20_reevaluation.py:970`) is therefore active during every per-sample gradient, so each `cos(H1)` is a different stochastic draw.

Note: a `model.eval()` **does** exist in the runner, but only inside `compute_decision_margin_and_metrics` (`run_phase_6e20_reevaluation.py:507`) — the Stage 6/7 margin/accuracy path, which is **not** the B0 reproducibility anchor. The B0 anchor path never sets eval.

This is the earliest point at which the recomputed value diverges from the historical anchor, and it is a **regime mismatch**, not a logic error in `psdg_fn`/`crgc_fn` (which are deterministic given fixed inputs).

---

## 6. What Is NOT the Cause (exclusions verified)

- **GPU nondeterminism** — ruled out: eval mode on the same CUDA device yields bit-identical `0.428355` across 3 processes.
- **Random seed / RNG initialization** — ruled out: eval mode is deterministic regardless of process; train mode varies despite identical seeds/checkpoints.
- **Model init / checkpoint loading** — ruled out: same checkpoint is loaded for every evaluation; train mode still varies, eval mode is stable.
- **Data / prompt order instability** — ruled out: identical `panel_prompts` (16), identical order, identical readout `S8`.
- **Corrupted or wrong checkpoint / wrong model** — ruled out: eval mode reproduces the anchor exactly, proving weights and machinery are correct.
- **Defect in frozen 6E.18 `psdg_fn` / `crgc_fn`** — ruled out: they are deterministic; the variation enters upstream via the train-mode dropout mask.
- **`ensure_readout_class_patch()` / CUDA↔numpy path** — not the cause: it only makes the CUDA tensor consumable; with dropout off the result is deterministic. (Diagnostic note: `psdg_fn` must be called through `compute_live_cos_H1`, which applies the patch; calling `psdg_fn` directly without it raises a CUDA→numpy error — an instrumentation detail, not a scientific defect.)

---

## 7. Disposition & Recommended Remediation (DEFERRED)

The runtime drift is **fully localized** and is a **regime mismatch**, not data/code corruption. 6E.20.3's Stage-5 halt is explained: the evaluator samples a stochastic quantity under train mode while the anchor was fixed under eval (dropout-off) mode.

Recommended remediation (must be its own authorized phase; not performed here):

1. **Align the inference regime.** Either:
   - (a) call `model.eval()` (and/or set LoRA dropout to `0.0`) before `compute_live_cos_H1` in the B0 reproducibility path, so the recomputation is deterministic and reproduces `0.428355`; **or**
   - (b) if train-mode stochasticity is intended, recompute the anchor under the *same* (train-mode) regime and re-derive a tolerance that reflects the observed sampling variance (~`4e-3`), rather than `1e-4`.
2. **Make the anchor and the evaluator agree on the dropout/inference regime before the `1e-4` gate has any meaning.** The current gate conflates "reproducibility failure" with "regime mismatch."

Until then, the `1e-4` tolerance on B0 step-4 `cos(H1)` is not satisfiable by a train-mode evaluator and should not be read as evidence of result corruption.

---

## 8. Reproducibility

```powershell
# Phase A — within-environment repeatability (4 independent processes)
cd theo-core/scripts/dataset_generator
python diagnose_runtime_drift.py A1
python diagnose_runtime_drift.py A2
python diagnose_runtime_drift.py A3
python diagnose_runtime_drift.py A4

# Phase C — controlled-factor isolation (eval vs train mode), 3 processes
python diagnose_drift_phaseC.py C1
python diagnose_drift_phaseC.py C2
python diagnose_drift_phaseC.py C3
```

Outputs land in `phase-6e20.3-reevaluation-output/diagnostic/`.

---

## 9. Artifact Manifest (SHA-256)

### Diagnostic scripts (unchanged scientific logic; instrumentation only)
| File | SHA-256 |
|---|---|
| `diagnose_runtime_drift.py` | `905f9c3fc284c93af924941c6545aa70b95befc9ed1372dfd5809023f1076642` |
| `diagnose_drift_phaseBC.py` | `c05588c40058962d7dcbe184da973c11041e0c7b0927b06d593d78259c97dc96` |
| `diagnose_drift_phaseC.py` | `dffcbe2f0b894ca1ea87ae8e37b3542569cdb92e97b0961782a34a82db871420` |
| `diagnostic_authorization_runtime_drift.json` | `3cde155c9d3f31c88fd40a8249f7ae5b02ccdad40dbb329cb2c36d6c9daeaf41` |

### Diagnostic outputs
| File | SHA-256 |
|---|---|
| `diagnostic/run_probe.json` | `b21349060bab27a91e88be6b6a6aed3c92e5965489d4e9263d3e00436f50818d` |
| `diagnostic/run_A1.json` | `aca62c034d3c7d8d3dde9c4aebf0042e6e6714e9b4f6b6f23b63242366ce1f77` |
| `diagnostic/run_A2.json` | `92b3240919753fb0f12898a470b39293c4b8d3abc8f767886b53d3ff9fe0ee1b` |
| `diagnostic/run_A3.json` | `bed6e39cbbeea1ea6e6dd5994fd4636cd73b9a8d3f1082f9d3158a423ce8040e` |
| `diagnostic/run_A4.json` | `ec02da726f47ab3c6efde9796d8eb7cd4a974674178b54dbf3d8bcc608ef276e` |
| `diagnostic/phaseC_evalmode_C1.json` | `7b8bb605f233cf8ed3ac14ed3baa476fa0791da6b9de3f4a77a97c2cd730509e` |
| `diagnostic/phaseC_evalmode_C2.json` | `86cf48c2eecb1429c424d82d55b42c850d91671bd84440fdaa38e223c5ab95f2` |
| `diagnostic/phaseC_evalmode_C3.json` | `203be1b18b5e810a5b52bbe4f3a132a28a9864609666b05a929b9f85f03a03ba` |

### Frozen references (unchanged)
| Artifact | SHA-256 |
|---|---|
| runner `run_phase_6e20_reevaluation.py` | `f424ecf64ae16eb669428d11120174cf575ad4d498462dc88cf8259df1fc9b38` |
| test `test_phase_6e20_reevaluation.py` | `43889916354f7edb72c5be72fb9a672d6907537b42464eeda4637ccf8ee96dba` |
| `collect_nepr.py` | `e2ee8afe6ee05bf9ad22d753de1c63a12108687eb5f2b86e44d493e4a123095a` |
| frozen 6E.18 machinery | `6bcaf1319ac551848ffdb55aea121d14d808408a192e05f2aa6171161decef96` |

---

*Generated by the Runtime Drift Root-Cause Localization diagnostic phase. No scientific artifacts were modified.*
