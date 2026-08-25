# FORENSIC — v6 G3 Investigation (post-halt)

**Date:** 2026-08-18
**Trigger:** v6 execution halted due to apparent scope deviation (S2 trained vs auth `void_conditions_excluded: ["S2","S4"]`).
**Method:** Ran `python run_phase_6e20_execution_v6.py step0-only` (no training; step-0 registration only) to recover the ACTUAL runtime gates/G3_void_map.

## Actual runtime step-0 gates (v6, this run)
- G3(k=2): kappa_2 = 0.056847  in [0.05, 0.995]  -> PASS
- G3(k=4): kappa_4 = 0.086011  in [0.05, 0.995]  -> PASS
- G3(k=8): kappa_8 = 0.125257  in [0.05, 0.995]  -> PASS
- G1,G2,G4,G5,G6,G7,G8: all PASS
- **runtime `void_conditions: []`** (empty — NO condition is void this run)

## Protocol VOID semantics (v3.1 §6 G3, E-4)
- VOID is CONDITIONAL, not intrinsic:
  - kappa_2 NOT in [0.05,0.995] => S2 VOID
  - kappa_4 NOT in [0.05,0.995] => S4 VOID
  - kappa_8 > 0.995            => S8 VOID
- Since kappa_2 and kappa_4 are INSIDE the band, S2 and S4 are NON-VOID this run and are
  correctly trained/evaluated by the runner.

## Conclusion
- The authorization document's `void_conditions_excluded: ["S2","S4"]` was an INCORRECT
  characterization. It conflated "S2/S4 carry a VOID contingency" with "S2/S4 are void."
- The v6 runner is PROTOCOL-COMPLIANT. Training S2/S4 was correct behavior.
- The halt was caused by an error in the authorization document, NOT by any runner defect.
- No runner change is required. The frozen 6E.18 g_total machinery, F7 repair, and the
  collect_H16 device fix are all correct and validated (L1-L4 PASS; step-0 executed on CUDA).

## Implication for re-run
- To complete 6E.20.1, v6 must be re-run FRESH (halted run is non-resumable).
- The authorization must be CORRECTED to reflect the actual VOID status
  (void_conditions = [] / S2,S4 non-void when kappa in band), then re-signed by human.
- L1-L4 for v6 remain valid (no runner change); only the authorization scope text changes.
