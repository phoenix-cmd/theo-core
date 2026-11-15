# 6C.3 Stage-2 Training Remediation — Post-Training Report

**Run:** bounded remediation iteration (authorized, 6D remains BLOCKED)
**Status:** REMEDIATION FAILED (collapse to ABSTAIN) · 6D NOT EXECUTED
**Date:** 2026-08-26

## Changes made (minimum, per authorization)
1. **Loss masking** — prompt/system+user tokens set to `-100`; assistant target restricted to `DECISION: <LABEL>` only. REASON omitted, so the objective trains the *decision* rather than copying REASON text.
2. **Class balancing** — inverse-frequency class-weighted cross-entropy (the approved mechanism, since the 300-GOLD dataset is frozen and cannot be re-balanced).

**Not changed:** 300-GOLD dataset, SemanticInterpretation labels, v0.5/ADR-0028 constraints, cos(H1)/gradient/hidden-state criteria, model architecture, M1/M2 identity/revision, QLoRA-only requirement. Historical v1 adapter and all artifacts preserved.

## Exact training configuration (recorded)
- Seed: `20260826` (Python/NumPy/Torch + `TrainingArguments.seed` + `data_seed`; `use_deterministic_algorithms=True`)
- Data ordering: canonical adjacency order of GOLD records (no shuffle beyond Trainer seed)
- Loss masking: prompt tokens → `-100`; target = `DECISION: <LABEL>` (REASON excluded)
- Class weights (inverse-freq, N=300, K=3): SENSITIVE `1.1905`, ABSTAIN `1.5152`, NEGATIVE `0.6667`
- Class counts: SENSITIVE 84 / ABSTAIN 66 / NEGATIVE 150 (300 GOLD; 44 REJECT excluded)
- Hyperparameters (unchanged from v1 except objective): LoRA r=16, α=32, dropout=0.05, target=attention+MLP; 4-bit NF4 + double quant, bfloat16; epochs=5; lr=2e-4; paged AdamW 8-bit; grad-ckpt; per-device BS=2, grad-accum=4
- New adapter dir: `M1_Qwen2.5-0.5B-Instruct-qlora-adapter-v2` (v1 preserved as evidence)

## Output hashes
- v2 adapter bundle SHA-256: `3462bd30cc1ebf0f9fcd6bad2547cf446e7a319d2393b41f31c7ce405bb9fbe0`
- v1 adapter (historical, unchanged): `4dcdbe51…6c`

## Smoke test — same 20-example evaluation set (identical ordering to v1)
| | v1 (baseline) | v2 (remediation) |
|---|---|---|
| Overall recall | 11/20 (55%) | **5/20 (25%)** |
| SENSITIVE | 0/4 | 0/4 |
| ABSTAIN | 0/5 | 5/5 |
| NEGATIVE | 11/11 | 0/11 |
| Collapse class | NEGATIVE | **ABSTAIN** |

Adapter confirmed **active** (`active_adapters: ['default']`).

## Diagnosis
The v2 run merely **shifted the collapse**: v1 defaulted to the majority class (NEGATIVE); v2 (with ABSTAIN weighted highest) defaults to ABSTAIN. The discriminative GOLD mapping was not robustly learned.
- Confirmatory probe: on **clear out-of-distribution** prompts the adapter emits correct `SENSITIVE`/`NEGATIVE`, showing base instruction-following is intact and the LoRA applies *some* shift.
- On the **real GOLD evaluation examples** it collapses to ABSTAIN (highest class weight), meaning the trained signal does not override base behavior on the actual (often ambiguous) corpus.

## Decision (per authorization)
This is a **collapse** → STOP. No further unbounded training loop was added. 6D is **not** executed automatically.

## Recommended next step (requires separate authorization)
- The bounded, approved mechanism (loss-masking + inverse-freq weighting) was insufficient on this 500M QLoRA at the given capacity/epoch budget. Options to authorize separately: (a) **6D comparative evaluation** to apply Group E gates and decide A vs B; (b) a *new, separately-authorized* training iteration with different levers (e.g., higher LoRA rank / more epochs / prompt-only decision token with stronger signal / explicit balance sampling) — explicitly outside this bounded run; or (c) accept that single-base QLoRA cannot learn this mapping and proceed to architecture B under 6D.

**→ STOP. Awaiting authorization (6D, or a distinct training iteration).**
