# Forensic Record — Phase 6E.20.1 v6 Execution Failure (Disk Capacity)

- **Date**: 2026-08-18
- **Classification**: GENUINE EXECUTION FAILURE (environmental: insufficient disk capacity). Not a runner-logic defect.
- **Disposition**: Halted. Failure state preserved. Runner v6, authorization, protocol, thresholds, seeds, conditions, basis, and decision rules NOT modified. Per authorization instructions, continuation would require a new review/authorization ladder (and disk-capacity remediation).

## Authorization & Runner (unchanged)
- Runner: `run_phase_6e20_execution_v6.py` — SHA `1F5F7EF2894CFC5B6B76D99E0EB57C4DBE9A00700E80B8197784FDF4C45FB15C`
- Authorization: `execution_authorization_v6.json` — SHA `4AA213782144DECF02D6CD062F5102B3E37393A2C3FF7403574DE4FF2CE86779`
- Checker `check_execution_authorization_v6`: ACCEPTED; F1 binding valid; 0 v5 references.
- L1-L4 (fresh, for 1F5F7EF2…): PASS.

## Execution Timeline (passive monitor, PID 30716)
- Step-0 fresh registration: completed 00:36:41 (13 artifacts in `step0/`).
- B0: 7/7 checkpoints completed by 00:55:04.
- F1: 7/7 checkpoints completed by 01:18:56.
- S2: 7/7 checkpoints completed by 01:43:27.  ← correctly trained (dynamic-VOID determined NON-VOID)
- S4: 7/7 checkpoints completed by 02:02:28.  ← correctly trained (dynamic-VOID determined NON-VOID)
- S8: training started 02:05:59; reached 2/7 checkpoints; **crashed during `checkpoint_4` write at ~02:09**.
- C4-Rand, R1-Rand: never started.
- Process 30716: found dead at 02:09:29 (monitor transition `alive=False`).

## Runtime VOID Determination (fresh §6 G3)
- The fresh run was proceeding to train **all 7 conditions** (B0, F1, S2, S4, S8, C4-Rand, R1-Rand). S2 and S4 were NOT excluded — confirming the corrected dynamic-VOID authorization correctly determined them NON-VOID via the fresh §6 G3 gate, consistent with prior G3 forensic evidence (κ₂=0.056847, κ₄=0.086011, κ₈=0.125257, all in [0.05,0.995]).
- Runtime VOID set was **empty** (no condition voided). This validates the corrected authorization scope.

## Failure Point & Error
- Crash location: `run_training` → `train_condition` → `torch.save(...)` at `run_phase_6e20_execution_v6.py:544` (S8 `checkpoint_4` write).
- Traceback root:
  - `RuntimeError: ios_base::badbit set: iostream stream error` (in `torch.serialization.save` / `_save` / `zip_file.write_record`)
  - `RuntimeError: [enforce fail at inline_container.cc:672] . unexpected pos 504720704 vs 504720656`
- `S8/checkpoint_4/model.pt` left partial at **504,721,408 bytes** (vs the normal ~1,023,619,435 bytes) — a truncated checkpoint from the failed write.

## Root Cause
- Disk capacity exhaustion. The run requires ~7 conditions × 7 checkpoints × ~1.02 GB ≈ 50 GB of checkpoint storage.
- Free space at run start was ~19.7 GB (post-cleanup), with ~15.25 GB already tied in preserved halted-run evidence.
- By S8 `checkpoint_4`, ~28 GB of checkpoints had been written, exhausting available space mid-write → `torch.save` failed.
- Post-mortem free space: 8.54 GB (recovered partially after the aborted write released its handle).

## Artifacts
- Completed checkpoints preserved: B0(7), F1(7), S2(7), S4(7) — 28 checkpoint `model.pt` files.
- Partial: S8(2) — `checkpoint_0` complete (1.02 GB), `checkpoint_4` truncated (504 MB).
- Step-0 registration artifacts: 13 files (`step0/`).
- NO `pre_registration_snapshot.json`, `runs.json`, `p5_loo_report.json`, or `sha256_manifest.json` were emitted — the run died before `emit_provenance`.
- NO final `execution_complete` / VOID line in `exec_stdout.log` (stdout empty; stderr holds the traceback).

## Preservation Actions Taken
- Did NOT patch, restart, or otherwise modify the runner / authorization / protocol.
- Relocated partial run (45 items, 28.12 GB) from `phase-6e20.1-execution-output/` to `phase-6e20.1-v6-FAILED-diskfull/` (failure state preserved; canonical clean path left empty).
- Monitor evidence: `6e20_execution_evidences/FORENSIC_v6_execution_monitor.md`.
- IDE Jedi language-server (PID 26112) untouched.

## Required to Re-attempt (decision for human)
1. Remediate disk capacity (free sufficient space for the full ~50 GB checkpoint set; note the 15.25 GB halted-run evidence is preserved by policy and should not be deleted without explicit instruction).
2. A new authorization/review ladder is required before any re-run under a (possibly re-validated) runner, since this run failed and the prior authorization is bound to a run that did not complete. (The runner binary itself is unchanged and its SHA still validates; however, per instructions, continuation under the current authorization is not permitted.)
