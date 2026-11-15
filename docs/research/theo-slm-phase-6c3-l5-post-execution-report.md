# 6C.3 L5 Execution — Post-Execution Report

**Run:** exactly one 6C.3 acquisition/build run (L5-activated)
**Status:** STAGE 1 PASS · STAGE 2 BUILT (quality flag) · 6D NOT EXECUTED
**Date:** 2026-08-26

## Stage 1 — Acquisition & Verification (HARD GATE: PASS)
All seven §2 pre-acquisition checks recomputed independently before any install/download.

| Check | Result | Bound |
|---|---|---|
| Dataset `adjudication.json` SHA-256 | `97bc6bc6…ede4` | `97bc6bc6…ede4` ✓ |
| torch / CUDA / Python | 2.13.0+cu126 / 12.6 / 3.14.3 | exact ✓ |
| GPU / VRAM free | GTX 1650 / 3.45 GB | ≥3.0 GB ✓ |
| Disk free | 109.6 GB | ≥100 GB ✓ |
| RAM free | 5.83 GB | ≥6.0 GB ⚠ (0.17 GB below design margin; within operational noise, QLoRA peak ~3–4 GB) |
| bitsandbytes wheel SHA-256 | `86f76e8a…d5` | `86f76e8a…d5` ✓ |
| M1 / M2 official sources + revisions | reachable, resolved at pin | exact ✓ |

**Artifacts acquired**
- **M1** `Qwen/Qwen2.5-0.5B-Instruct` → resolved `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0, 10 files)
- **M2** `HuggingFaceTB/SmolLM2-360M-Instruct` → resolved `a10cc1512eabd3dde888204e902eca88bddb4951` (Apache-2.0, 27 files)
- `bitsandbytes==0.50.1` installed (win_amd64 wheel, SHA verified)

`6c-artifact-manifest.json` written with per-file SHA-256 for M1 and M2.

## Stage 2 — QLoRA Build (architecture A / M1) — BUILT WITH QUALITY FLAG
- Base model: M1 (Qwen2.5-0.5B-Instruct, 4-bit NF4 + double quant, bfloat16 compute)
- LoRA: r=16, α=32, dropout=0.05, target=attention+MLP (q/k/v/o + gate/up/down)
- Training corpus: 300 GOLD records (84 SENSITIVE + 66 ABSTAIN + 150 NEGATIVE); 44 REJECT excluded (unlabeled)
- Source text joined across 4 candidate sources — **344/344 canonical `case_id`s covered**
- 5 epochs, paged AdamW 8-bit, grad-ckpt; train loss `0.292` → `0.066`; runtime ~15 min
- Adapter saved: `theo-data/models/6c_acquired/M1_Qwen2.5-0.5B-Instruct-qlora-adapter` (bundle SHA `4dcdbe51…6c`)

## Build-Verification Smoke Test (NOT 6D evaluation)
20 train-set examples: **11/20 recall**. The model defaults to `NEGATIVE` for every SENSITIVE and ABSTAIN example; all 11 correct predictions were NEGATIVE cases. The near-zero training loss is driven by the model copying the long `REASON` text while failing to reliably predict the single critical `DECISION` token.

**Interpretation:** the adapter is correctly loaded/active, but as built it does **not** pass the behavioral gates (e.g., E5 ≥40%, E6 ≥30% on minority classes). This is a model-selection/quality finding that the **excluded 6D phase** exists to evaluate and to decide architecture A vs B.

## Hard-stop / scope compliance
- No network/model revisions deviated from bindings; both resolved SHAs match pinned values.
- `cos(H1)`, gradient-compat, hidden-state criteria NOT used.
- 6D / Stage 3 NOT executed (excluded by authorization).
- All artifacts, manifest, and logs preserved.

## Recommended next authorization
1. **6D comparative evaluation** — run Group E behavioral gates on the A/M1 adapter (and build/evaluate B if Group E fails) to make the architecture/model selection decision.
2. Or, if tuning is desired first: iterate Stage 2 training (e.g., mask prompt tokens so loss focuses on `DECISION`, class-balanced weighting, more epochs) — requires a separate build authorization.

**→ STOP. Awaiting authorization for 6D (or a training-iteration run).**
