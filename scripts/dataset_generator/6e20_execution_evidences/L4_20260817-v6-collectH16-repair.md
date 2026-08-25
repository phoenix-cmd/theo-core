# Independent L4 Review — Phase 6E.20.1 Runner v6 (collect_H16 device-placement repair)

- Review date: 2026-08-17
- Runner under review: `run_phase_6e20_execution_v6.py`
- Runner SHA-256: `1F5F7EF2894CFC5B6B76D99E0EB57C4DBE9A00700E80B8197784FDF4C45FB15C`
- Ladder status entering L4: L1 PASS (compile+import), L2 PASS (33/33 self-test), L3 PASS (C1–C35, 35/35 via harness v4 `D580C4DD…`→ v4 binary).
- Frozen 6E.18 snapshot SHA-256: `6BCAF1319AC551848FFDB55AEA121D14D808408A192E05F2AA6171161DECEF96` (unchanged; verified at import).
- Prior v5 authorization `5447DC1E…` (runner `65E6FDA8…`) is preserved as evidence for the FAILED v5 attempt and does NOT authorize v6.

## Why v6 exists
The v5 attempt failed at step-0 (`collect_H16`, v5:354): tokenizer tensors (CPU) were passed to a CUDA model, raising
`RuntimeError: index is on cpu, different from other tensors on cuda:0`. Fixing this edits runner bytes → new SHA → new
authorization required. v6 is the minimal repaired revision.

## Changes vs v5 (both necessary, scoped, no unrelated edits)
1. **FIX (collect_H16, v6:354-355):** before the forward pass, move every tokenizer tensor to the model's actual device:
   `_dev = next(model.parameters()).device; inputs = {k: v.to(_dev) for k, v in inputs.items()}`.
   This is the canonical device-agnostic pattern (works for CPU and CUDA; does not rely on a `.device` module property).
2. **Revision scoping — authorization file (v6:1163, v6:1194):** the authorization file referenced by the `execute`/`step0-only`
   subcommands is `execution_authorization_v6.json` (was `v5`). This makes v6 a distinct authorized revision so the v5
   authorization cannot be reused for it, per the failed-attempt isolation requirement.
3. **Revision scoping — checker rename:** the authorization-check function used by v6 is `check_execution_authorization_v6`
   (was `check_execution_authorization_v5`). The v6 source contains **zero** references to `execution_authorization_v5.json`,
   so the v5 auth file cannot be discovered or accepted as a fallback. Enforces decision=APPROVE, phase=6E.20.1, and
   exact runner SHA-256 `1F5F7EF2…`.

No changes to: protocol v3.1, Plan v4.0, frozen runner/harness/6E.18 artifacts, thresholds, seeds, conditions, basis,
decision rules, F7 (cos H1 from g_total), or execution semantics beyond the device-placement correction.

## Findings (F1–F7)
- **F1 (auth binds to exact runner SHA):** `check_execution_authorization_v6` recomputes `sha256(__file__)` and compares to
  `auth["runner_sha256"]`. Execution reads `execution_authorization_v6.json`, which does not yet exist → refused until a new
  L5 issues. **PASS** (gating intact; new file required; no v5 fallback possible).
- **F2–F6:** unchanged from v5 (which passed L4); L3 C22–C32 (isolation, VOID extraction, H16 forward, rho per-condition,
  proposal_series, checkpoint restore) all PASS.
- **F7 (cos H1 from g_total):** `compute_cos_H1_from_live_model` → `per_sample_decision_gradient` (true g_total autograd)
  unchanged; L3 C27 PASS. **PASS**.
- **NEW REGRESSION GUARD C35 (collect_H16 device placement):** exercises the real `collect_H16` with BOTH a CPU model and a
  CUDA model; asserts tokenizer inputs land on the model's device and H16 shape == (16, D_HIDDEN). **PASS** on this GPU host
  (CPU + cuda:0 paths both exercised). This directly prevents the v5 failure mode from recurring and from being
  "works on my GPU" while breaking CPU.

## Protocol Compliance
- Frozen machinery imported verbatim (hash-verified); no re-implementation.
- Void conditions (currently S2, S4 per registered G3) excluded from training/evaluation via `compute_void_conditions`.
- No execution has occurred under v6.

## Regression Check vs v5
v6 = v5's F1–F7 + the 2 scoped changes above. L2 (33/33) and L3 (35/35) confirm no regression; the new C35 adds explicit
device-path coverage. The v5 authorization remains bound solely to v5 `65E6FDA8…` and is not reused.

## Verdict
**L4 PASS** for runner v6 (`3DA208B3…`). All F1–F7 satisfied; minimal change verified; new regression guard present.
Eligible for a NEW L5 human authorization binding to runner SHA
`3DA208B3C26B62F78210514F042CF4EC5625F8C61CEAA785DB973F96A6886FFB`.
