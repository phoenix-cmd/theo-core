# FORENSIC — v6 Runner Identity Reconciliation (3DA208B3 vs 1F5F7EF2)

**Date:** 2026-08-18
**Trigger:** User caught that the corrected L5 draft stated runner SHA `1F5F7EF2…` while the runner
reviewed at L1–L4 was `3DA208B3…` — "these cannot both be the unchanged reviewed runner."

## Investigation (independently computed)
- On-disk `run_phase_6e20_execution_v6.py` SHA-256 = `1F5F7EF2894CFC5B6B76D99E0EB57C4DBE9A00700E80B8197784FDF4C45FB15C`.
- Reverse rename of the checker (`check_execution_authorization_v6` -> `check_execution_authorization_v5`,
  length-preserving: "_v6"/"_v5" both 2 chars) reproduces SHA-256 = `3DA208B3C26B62F78210514F042CF4EC5625F8C61CEAA785DB973F96A6886FFB` **exactly**.
- Occurrences in current file: checker `_v6` name = 7, `_v5` name = 0; auth file `v6.json` = 2, `v5.json` = 0.
- Byte length identical (54260). Reconstruction SHA match proves the **ONLY** difference between
  `3DA208B3…` and `1F5F7EF2…` is the checker function rename.

## Root cause
After L1–L4 (incl. independent L4 review) completed at `3DA208B3…`, the operator performed the checker
rename `check_execution_authorization_v5` -> `check_execution_authorization_v6` (requested in the
"correct final-hash procedure" turn). That edit changed runner bytes, producing `1F5F7EF2…`. This is a
**POST-L4 runner modification**.

## Consequence (operator error acknowledged)
The prior statements "runner unchanged" / "L1–L4 still bind" were **INCORRECT**. L1–L4 validity belongs to
`3DA208B3…`; it does NOT transfer to `1F5F7EF2…`. The modified runner requires fresh L1–L4.

## Evidence preserved (unmodified)
- Reviewed runner bytes at `3DA208B3…`:
  `6e20_execution_evidences/run_phase_6e20_execution_v6_REVIEWED_3DA208B3.py`
  (SHA verified = `3DA208B3…`).
- L4 review record `L4_20260817-v6-collectH16-repair.md` is **internally contradictory**:
  lines 5/25/31 assert `1F5F7EF2…`; lines 53–55 assert `3DA208B3…` PASS. Preserved unmodified.
  The genuine L4 review was performed at `3DA208B3…`.
- Superseded authorization (wrong VOID field) preserved at
  `6e20_execution_evidences/execution_authorization_v6_SUPERSEDED_halted.json` (prior SHA `F8682884…`).

## Resolution plan
- Treat `1F5F7EF2…` as a modified runner.
- Run FRESH L1–L4 validation gates for the exact on-disk hash `1F5F7EF2…`:
  L1 (compile+import+frozen snapshot hash), L2 (self-test 33/33), L3 (harness v4 35/35),
  L4 (independent review documenting `1F5F7EF2…` and the rename diff).
- Only after fresh L1–L4 PASS, present an L5 draft binding to `1F5F7EF2…`.
- Do NOT create/sign the authorization and do NOT execute training until fresh L1–L4 pass.
