# Forensic Failure Record — Phase 6E.20.1 Runner v5 Execution Attempt

- Record date: 2026-08-17
- Status: **FAILED ATTEMPT — NOT RESUMED. Evidence preserved immutably.**
- Runner under test: `run_phase_6e20_execution_v5.py`
- Runner SHA-256: `65E6FDA818C7AE3F8D74398C28C8A77F3144BF4BBCBE972EE224BA711FE627E8`
- Authorization artifact: `execution_authorization_v5.json` (signed, immutable)
- Authorization SHA-256: `5447DC1E7F284E2F8FA3518B87832546C5E9ED5A6A08C342B04E772254C6D2E7`
- Authorization decision/phase: APPROVE / 6E.20.1 (binds exclusively to runner `65E6FDA8…`)
- Outcome: **execution halted at step-0; authorization NOT reused for any modified runner**

## Failure Stage
- Pipeline stage: step-0 registration → `run_step0_registration` → `collect_H16`
- Exact location: `run_phase_6e20_execution_v5.py:354` (`out = model(**inputs, output_hidden_states=True)`),
  called from `:369` (`H16 = collect_H16(model, tokenizer)`), called from `:1172` (`reg_result = run_step0_registration(...)`).
- Progress: model + tokenizer loaded; model moved to `cuda:0` (`:1170`); `collect_H16` raised before any `H_16` save
  (save is at `:426`). **No training, no checkpoints, no evaluation results were produced by the v5 attempt.**

## Full Exception
```
Traceback (most recent call last):
  File "run_phase_6e20_execution_v5.py", line 1226, in <module>
    raise SystemExit(main())
  File "run_phase_6e20_execution_v5.py", line 1172, in main
    reg_result = run_step0_registration(model, tokenizer, out_dir / "step0")
  File "run_phase_6e20_execution_v5.py", line 369, in run_step0_registration
    H16 = collect_H16(model, tokenizer)
  File "run_phase_6e20_execution_v5.py", line 354, in collect_H16
    out = model(**inputs, output_hidden_states=True)
  ... (transformers qwen2 forward -> embed_tokens) ...
  File "torch/nn/functional.py", line 2615, in embedding
    return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA__index_select)
```

## Root Cause
`collect_H16` (v5:353-354) calls `tokenizer(formatted, return_tensors="pt")`, which returns **CPU** tensors, and passes
them directly to the model **without** moving them to `model.device`. The runner had already moved the model to `cuda:0`
(v5:1170, `if torch.cuda.is_available(): model = model.cuda()`). The CPU input index tensor therefore reached a CUDA
`nn.Embedding`, triggering the index_select device-mismatch.

Scope of defect: **only `collect_H16`** lacks device placement. The training loop (v5:478-481), `evaluate_checkpoint_at_state`,
and `compute_cos_H1_from_live_model` already move tensors to the target device; the frozen 6E.18 `per_sample_decision_gradient`
moves tensors to its own `DEVICE`. So the failure is isolated to step-0 H16 collection on a GPU host.

## Disposition
- The v5 runner is abandoned for execution (defective in the real-GPU path). No resume.
- The signed L5 authorization (`5447DC1E…`) remains **evidence for the failed v5 attempt only** and MUST NOT authorize any
  modified runner.
- Repair path: minimal device-placement fix in `collect_H16` only → new runner revision (v6) with a NEW SHA-256 → fresh
  L1–L4 → new L5 authorization binding to the v6 SHA. CPU workaround explicitly rejected.

## Evidence Artifacts (preserved)
- `execution_authorization_v5.json` (signed, SHA `5447DC1E…`) — kept immutable.
- `6e20_execution_evidences/L5_draft_v1_unsigned_BC2F1E1C.json`, `L5_draft_v2_unsigned_E70455E8.json`,
  `L5_drafts_manifest.md` — unsigned review drafts (superseded).
- `6e20_execution_evidences/v5_exec_stdout.log`, `v5_exec_stderr.log` — captured execution logs.
- Stale v3 abandoned output relocated to `phase-6e20.1-v3-ABANDONED-output/` (isolated, not deleted).
- L4 review of v5: `6e20_execution_evidences/L4_20260817-v5-F7repair.md`.
