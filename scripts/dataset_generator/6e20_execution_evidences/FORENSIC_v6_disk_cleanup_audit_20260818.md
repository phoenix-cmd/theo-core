# Cleanup Audit Record — Pre-Execution Disk Audit (Phase 6E.20.1 v6)

- **Date**: 2026-08-18
- **Task**: Conservative pre-execution disk cleanup audit (no execution started).
- **Scope constraints honored**: Did NOT start execution; did NOT modify runner v6, execution_authorization_v6.json, protocol, Plan v4.0, harness, corpus, or scientific artifacts. Only clearly disposable regenerable bytecode removed; all evidence preserved.
- **Active runner SHA (unchanged)**: `1F5F7EF2894CFC5B6B76D99E0EB57C4DBE9A00700E80B8197784FDF4C45FB15C`

## Disk State
- Free space before: ~19.50 GB
- Free space after:  ~19.69 GB (Δ ≈ +0.19 GB)
- Primary reclaim: 173.9 MB of `__pycache__`/`.pyc` bytecode (regenerable).

## Classification
| Class | Items | Size | Action |
|-------|-------|------|--------|
| SAFE_DELETE | 8,246 `.pyc` files across 46 `__pycache__` dirs (theo-core 4620 / theo-providers 3625 / theo-data 1) | 173.9 MB | DELETED (bytecode regenerates on next import) |
| KEEP (evidence) | `phase-6e20.1-v6-HALTED-scope-deviation/` (B0/F1/S2/step0) | 15.25 GB | PRESERVED — forensic record of the halted run |
| KEEP (evidence) | `6e20_execution_evidences/` (forensics, L4 reviews, REVIEWED_3DA208B3.py, superseded auth, G3 raw step-0) | — | PRESERVED |
| KEEP (evidence) | `phase-6e20.1-v3-ABANDONED-output/` | — | PRESERVED (isolated, not deleted) |
| KEEP (scientific) | 4 historical run logs (run.log, run2.log, training.log, eval log) | 0.08 MB | PRESERVED — research artifacts, not temp debug |
| KEEP (tool-owned) | `C:\Users\bs162\AppData\Local\Temp\opencode` (31,400 files, 4.66 GB) | 4.66 GB | NOT TOUCHED — opencode scratch, outside project |
| KEEP + RELOCATE | `phase-6e20.1-execution-output/step0/` (G3-investigation raw step-0 data: H_16, R_4, U_8, u_lda, w_*.npy, y16.npy) | ~0.2 MB | MOVED to `6e20_execution_evidences/step0_raw_preexec_v6/` (13 files); clean run path emptied |

## Actions Performed
1. Relocated stale `step0/` from the clean execution-output path into the evidence archive (`step0_raw_preexec_v6/`) — preserves G3 forensic raw data AND clears the clean path so the fresh run regenerates its own step-0 (required for dynamic §6 G3 VOID determination).
2. Deleted 8,246 `.pyc` files (regenerable bytecode); removed resulting empty `__pycache__` directories.
3. Verified 0 residual `.pyc` remain under the workspace.

## Post-Audit Verification
- Runner `run_phase_6e20_execution_v6.py` SHA256 = `1F5F7EF2894CFC5B6B76D99E0EB57C4DBE9A00700E80B8197784FDF4C45FB15C` (matches expected, unchanged).
- Clean execution-output path `phase-6e20.1-execution-output/` now contains 0 items (pristine for fresh run).
- Evidence step-0 preserved: 13 files in `step0_raw_preexec_v6/`.
- Halted-run evidence (15.25 GB) left intact.

## Note for Future Decision
The dominant disk consumer (15.25 GB) is the halted v6 output preserved as evidence. It is intentionally KEEP per audit constraints. If disk pressure becomes critical, pruning non-B0 checkpoints (never read by `run_evaluation`, which only restores B0/checkpoint_{0,4,6,7,14,15,17}) could be considered under a separate, explicit authorization — not performed here.
