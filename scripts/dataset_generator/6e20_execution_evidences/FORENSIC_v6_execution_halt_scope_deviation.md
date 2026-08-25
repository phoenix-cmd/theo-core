# FORENSIC — v6 Execution HALTED (Scope Deviation)

**Date:** 2026-08-18
**Runner:** run_phase_6e20_execution_v6.py
**Runner SHA-256:** 1F5F7EF2894CFC5B6B76D99E0EB57C4DBE9A00700E80B8197784FDF4C45FB15C
**Authorization:** execution_authorization_v6.json (final SHA F8682884F55475573B8E05E380F806A22F6539C029E25612CCEE95D328F6B33B)
**Authorization status:** EXPLICIT HUMAN L5 SIGNOFF (granted)

## What happened
1. Fresh clean 6E.20.1 execution under v6 was started after all authorization checks passed
   (binding to runner SHA 1F5F7EF2..., check_execution_authorization_v6 ACCEPTED, no v5 fallback).
2. Step-0 `collect_H16` completed successfully — the device-placement repair (v6 vs failed v5) worked;
   the model ran on CUDA (GPU 100% confirmed) and did NOT crash as v5 did.
3. Training progressed: B0 (7 checkpoints, complete), F1 (7 checkpoints, complete), then S2 began training.
4. S2 is a VOID condition EXPLICITLY excluded by the signed authorization
   (`void_conditions_excluded: ["S2","S4"]`; scope: "VOID conditions are not trained or evaluated").
   The runner created S2/checkpoint_0 and S2/checkpoint_4 (full model.pt files), i.e. it TRAINED S2.

## Root cause of the deviation
- The runner derives VOID status DYNAMICALLY from `G3_void_map` (compute_void_conditions, v6:251).
  S2/S4 are void ONLY if their G3 (kappa) gate FAILS (self-test F3, v6:994: "s2_s4_void_on_kappa_violation").
- S2 being trained proves this run's runtime G3 gate for S2 PASSED, so the protocol's dynamic logic
  classified S2 as NON-VOID and trained it.
- The signed authorization's `void_conditions_excluded: ["S2","S4"]` was written on the ASSUMPTION that the
  runtime G3 gates would fail for S2/S4 (matching "registered G3 results"). That assumption was contradicted
  by this run's deterministic step-0: S2's G3 gate passed, so S2 was trained.

## Why execution was halted (not allowed to continue)
- The signed authorization binds EXCLUSIVELY to its stated scope. Training a VOID-excluded condition (S2)
  is outside that scope — a material breach of the binding instrument.
- Disk budget: 7 conditions (B0,F1,S2,S4,S8,C4-Rand,R1-Rand) × 7 checkpoints × ~1 GB ≈ 49 GB vs
  only 19.9 GB free. The auto-prune monitor only covered F1/S8/C4-Rand (it assumed S2/S4 would be skipped),
  so the run would have hit disk-full and failed unpredictably (mirroring the v5 device failure).
- Halting preserves authorization integrity and captures clean evidence of the deviation.

## Action taken
- Main execution process (PID 26032) and auto-prune monitor (PID 24668) stopped.
- Partial outputs (B0, F1, S2 checkpoints) relocated to: phase-6e20.1-v6-HALTED-scope-deviation/
  (preserved as forensic evidence of the deviation; NOT deleted; separate from v3/v5 evidence).
- No runner bytes were modified after authorization (per the constraint).
- The halted run is NON-RESUMABLE (single-pass pipeline); any continuation requires a fresh run.

## Open decision (requires human authorization)
The discrepancy is between the authorization document (S2/S4 void-excluded) and the protocol's dynamic
VOID logic (S2 non-void because G3 passed this run). Two paths:
  (A) If S2/S4 training is acceptable whenever the protocol's G3 gate passes, the run was protocol-compliant
      but the auth doc was mislabeled — issue a corrected/re-scoped authorization and re-run v6 fresh.
  (B) If S2/S4 MUST remain excluded regardless, the runtime G3 result is unexpected and must be
      investigated (e.g., compare registered G3 results vs this run's step-0 gates) before any re-run;
      this may require a new runner revision + new authorization.

## Evidence references
- phase-6e20.1-v6-HALTED-scope-deviation/  (B0/, F1/, S2/, step0/)
- execution_authorization_v6.json          (final SHA F8682884...)
- 6e20_execution_evidences/FORENSIC_v5_execution_failure.md  (prior v5 step-0 failure)
- phase-6e20.1-v3-ABANDONED-output/        (prior v3 abandoned)
