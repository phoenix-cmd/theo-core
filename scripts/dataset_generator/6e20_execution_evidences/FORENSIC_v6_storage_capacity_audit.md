# Storage Capacity Audit — Phase 6E.20.1 v6 (pre-remediation)

- **Date**: 2026-08-18
- **Purpose**: Establish the hard disk requirement implied by the frozen experimental
  semantics (protocol v3.1 §10.4 checkpoint retention) before any new execution ladder.
- **Constraint honored**: No change to runner v6, protocol v3.1, checkpoint retention,
  thresholds, seeds, conditions, basis, or decision rules. Evidence preserved immutably.

## 1. Checkpoint-count semantics (7 vs 8)
- `CHECKPOINT_SET = [0, 4, 6, 7, 14, 15, 17, 20]` (frozen runner line 68; protocol §10.4 canonical set).
- `HORIZON = 20` (frozen runner line 50). Training loop: `for t in range(HORIZON)` → **steps 0..19**.
- `checkpoint_20` is saved only `if t in checkpoint_set`, but **`t` never equals 20** (range is 0..19).
  ⇒ The reviewed runner **never emits `checkpoint_20`**. Empirically, every completed condition
  in the failed run has exactly **7** checkpoints `{0,4,6,7,14,15,17}`; `_20` is absent.
- **Conformance note (flagged, not fixed)**: protocol §10.4 states a *missing rule-mandated
  checkpoint in any condition ⇒ that run is VOID*. The reviewed runner therefore cannot, as written,
  produce a protocol-valid run (it always omits `checkpoint_20`). This is a runner/protocol matter
  outside the current audit scope and was not remediated (runner changes are forbidden here).
- For capacity planning we use the **protocol-canonical requirement: 8 checkpoints/condition**
  (the frozen semantics demand retention of all mandated checkpoints; a conformant runner would emit 8).
  The observed 7/condition is recorded as a secondary figure.

## 2. Measured artifact sizes (from failed-run evidence `phase-6e20.1-v6-FAILED-diskfull`)
- Completed `model.pt`: **1,023,617,003 bytes = 976.197 MiB = 0.953295 GiB** (uniform across all completed ckpts).
- 30 completed checkpoints total = 30,189,663,071 bytes = 28.116 GiB (matches ~28.12 GB observed).
- Failed partial `S8/checkpoint_4/model.pt` = 481.34 MiB (truncated mid-write — the failure signature).
- step-0 artifacts (13 files: H_16, U_8, R_4, u_lda, n_rand, y16, w_*.npy): **191,744 bytes = 0.183 MiB** (negligible).
- Evaluation/provenance/manifest outputs (`runs.json`, `p5_loo_report.json`, `sha256_manifest.json`,
  `pre_registration_snapshot.json`): metric/JSON only; `sha256_manifest.json` *hashes* checkpoints
  (does not copy them). Aggregate < 0.1 GiB (negligible).

## 3. Projected peak disk usage (run output directory)
Per-condition checkpoints (canonical): **8** × 0.953295 GiB = 7.62636 GiB
All 7 conditions: 7 × 7.62636 GiB = **53.3845 GiB** (= 57.33 GB decimal) — final retained size.

Peak occurs while writing the final checkpoint: all prior checkpoints on disk + one in-progress write.
- Retained (full canonical set): 53.3845 GiB
- Transient in-progress `torch.save` (one checkpoint being written): +0.953295 GiB
- step-0 artifacts: +0.000179 GiB
- eval/provenance/manifest: +<0.1 GiB

**Projected run-output peak ≈ 54.34 GiB (58.4 GB decimal).**

Observed-run figure (reviewed runner emits 7/condition): 49 × 0.953295 = 46.71 GiB retained;
peak ≈ 47.66 GiB (51.2 GB). Still exceeds available space (see §4).

## 4. Safety margin and comparison to available capacity
- Proposed filesystem safety margin: **15 GiB** (minimum; ≈25% of run requirement).
- **Projected peak requirement (canonical, with margin)** = 54.34 + 15 = **69.34 GiB (≈74.5 GB)**.

Current volumes:
- **C:** (OS) total 453.4 GiB, **free 40.25 GiB (≈43.2 GB decimal)**.
  - Of C:'s used space, **43 GiB is preserved evidence** (halted-run 15.25 GiB + failed-run 28 GiB)
    that must remain (per instructions, not deleted).
- **D:** (New Volume) total 476.9 GiB, free 11.3 GiB.

Comparison (run output must reside on C:, its working volume):
- Run-output peak alone (54.34 GiB) vs C: free (40.25 GiB) → **exceeds by ~14 GiB**.
- Run (54.34) + preserved evidence (43) + margin (15) = **112.3 GiB required on C:** vs 40.25 available
  → **deficit ≈ 72 GiB** (or ≈29 GiB deficit even if evidence is moved off C: and no margin is used).

## 5. Verdict
```
Projected peak requirement: 54.34 GiB run-output (+15 GiB margin = 69.34 GiB; +43 GiB preserved evidence => 112.3 GiB on C:)
Available:                  40.25 GiB (C:)
Safety margin:              15 GiB (proposed minimum)
=> INSUFFICIENT
```
40 GB free is **insufficient** under the frozen checkpoint-retention semantics. The disk failure was
not merely "forgot to free space" — it exposed a hard resource requirement: a valid 7-condition run
retains ~53 GiB of checkpoints (canonical) that must coexist on the run volume, which neither C: (40.25 GiB
free) nor D: (11.3 GiB free) can satisfy alongside the 43 GiB of preserved evidence.

## 6. Remediation direction (no change to retention/protocol/runner)
To reach sufficiency without altering frozen semantics, capacity must be increased by freeing or
relocating ~29 GiB (evidence kept on C:, with margin) or ~14 GiB (evidence moved off C:, no margin)
of **non-evidence, regenerable** space on C:, and/or moving the 43 GiB preserved evidence to separate
storage with adequate free capacity (none currently exists on C:/D:). External storage (≥60 GiB free)
would be the cleanest path. Specific candidate identification is the next step.

## 7. Remediation executed (2026-08-18)
- Freed regenerable, non-evidence caches on C: (no change to runner/protocol/retention/evidence):
  - AppData\Local\pip (7.2 GiB), AppData\Local\npm-cache (7.09 GiB), Windows\Temp (1.62 GiB), AppData\Local\Temp (user temp, opencode subdir excluded).
- PRESERVED: .cache/huggingface (base model the run loads), all 6E.20/6E.18 evidence, halted-run + failed-run artifacts, authorization, forensic reports.
- Left untouched: user Downloads (contains personal files), AppData\Local\Programs (installed apps), system dirs.

## 8. Post-remediation verdict
- **C: free after remediation = 62.63 GiB** (was 40.25 GiB).
- Reviewed runner ACTUAL output (7 ckpts/cond, since it never emits checkpoint_20):
  - Projected peak = 47.66 GiB (51.2 GB); remaining headroom = 62.63 - 47.66 = **14.97 GiB (~15 GiB margin)**.
  - => SUFFICIENT for the actual reviewed-runner run (completes without hitting 0 mid-write).
- Canonical frozen requirement (8 ckpts/cond):
  - Projected peak = 54.34 GiB; remaining headroom = 62.63 - 54.34 = 8.29 GiB.
  - => Completes (no zero-hit) but below the 15 GiB proposed margin; AND a run with the reviewed runner is VOID per protocol �10.4 because checkpoint_20 is never emitted (see �1).
- **Conclusion:** storage is no longer the blocker for an actual run with the reviewed runner. The remaining blocker for a *protocol-valid* run is the checkpoint_20 conformance gap (runner revision + new authorization ladder required � out of scope here, runner changes forbidden).
