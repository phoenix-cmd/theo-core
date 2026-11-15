# THEO SLM v0 — Phase 6C.4 Supervised Evaluation Report

**Authorized scope:** One bounded deterministic QLoRA run per architecture (A/B/C) on the 300‑GOLD 1:1 mapping, then evaluation on the frozen 66‑case instrument (phase‑6e3 populated‑evidence corpus). All prior bindings, gates, the 300‑GOLD dataset, and the 66‑case fixture were held immutable.

## 1. Training (bounded, reproducible)

| Arch | Adapter SHA‑256 | Train/Dev | Final‑epoch loss | Dev acc¹ | Secs |
|------|-----------------|-----------|------------------|----------|------|
| A (decoder‑only) | `d5a92067…887692` | 269 / 31 | 0.7377² | 0.742 | 2489 |
| B (decoder + evidence reranker) | `24333a75…12ad14` | 269 / 31 | 0.5435² | 0.742 | 2489 |
| C (shared encoder / two heads) | `73828c59…5e95d` | 539 / 61 | 0.3218² | 0.443³ | 5414 |

¹ Dev accuracy over the 300‑GOLD 90/10 stratified split (seed 20260826).
² Epoch‑average training loss; per‑step losses converged to ~0.05–0.13.
³ C's dev split mixes evidence‑head rows (no `decision` field) into the decision‑only metric, so 0.443 is diluted. The fixture result below is the correct measure.

Shared config: M1 `Qwen2.5‑0.5B‑Instruct @7ae55760` (frozen), nf4 4‑bit + double quant (fp16 compute), LoRA r16/α32/drop0.05 all‑linear, epochs 3, lr 2e‑4 constant, batch 4 × grad‑acc 4 (eff. 16), seed 20260826, max_len 384, prompt/instruction tokens masked to −100, target = `SemanticInterpretation` envelope only (**`DECISION:` string format explicitly excluded** — the root cause of the earlier v1/v2 collapses).

## 2. Frozen 66‑case fixture evaluation

Fixture = `phase‑6e3/populated‑evidence‑corpus.json` (66 cases: 15 probe + 51 benchmark, with `group` labels; Group E = `sp1://e/001..003`). Deterministic generation (`do_sample=False`).

| Arch | Acc | E0 (decision rate) | Abstain rate | **Group E** | Grounding | Latency (s) | VRAM (GB) |
|------|-----|--------------------|--------------|-------------|-----------|-------------|-----------|
| A | **0.727** | 0.9545 (FAIL, 4.5% UNKNOWN) | 0.015 | **0.0 (FAIL)** | n/a | 3.29 | 0.51 |
| B | 0.470 | 1.0 | 0.712 | **1.0 (PASS)** | 1.0 | 4.37 | 0.52 |
| C | 0.349 | 1.0 | 0.864 | **1.0 (PASS)** | 1.0 | 6.35 | 0.52 |

## 3. Gate assessment (pre‑registered rules)

| Gate | A | B | C |
|------|---|---|---|
| E0 ≤ 2% | FAIL (4.5%) | PASS | PASS |
| Grounding 100% | n/a | PASS | PASS |
| **Group E ≥ 80% (deciding)** | **FAIL 0/3** | **PASS 3/3** | **PASS 3/3** |
| Abstention ≥ 90% | FAIL (1.5%) | FAIL (71%) | FAIL (86%) |
| Latency ≤ 0.5s | FAIL (3.3s) | FAIL (4.4s) | FAIL (6.4s) |
| VRAM ≤ 2GB | PASS | PASS | PASS |

## 4. Interpretation

- **Supervised fine‑tuning transferred decision behavior** to all three architectures from the 300‑GOLD mapping.
- On the **deciding gate (Group E** — the evidence‑selection / distractor‑rejection probe), **only B and C pass (3/3)**; **A fails completely (0/3)**, proposing where it should abstain. A's high raw accuracy (0.727) is misleading: it is trigger‑happy and unsafe on exactly the cases that matter.
- **B and C over‑abstain** on the broader fixture (71% / 86% vs the ≥90% gate), which is what drags their overall accuracy down (0.47 / 0.35). The **≥90% abstention gate is mis‑calibrated** for a fixture that contains a substantial `SHOULD_PROPOSE` share — a gate value this high cannot be met by any model that proposes correctly on those cases.
- **All three fail the ≤0.5s latency gate** (3–6s), which is inherent to LLM generation and was also unmet by every prior 6D realization.
- No architecture clears **all** pre‑registered gates.

## 5. Verdict

**No architecture is confirmed.** B and C satisfy the deciding Group‑E gate but fail the abstention and latency gates; A fails the deciding gate. Selection remains **DEFERRED to human review** (consistent with the authorization).

Recommended human actions before re‑judging:
1. Re‑examine the **abstention gate (≥90%)** — it is inconsistent with the fixture's label distribution; a calibrated value (e.g., measured against the actual PROPOSE/ABSTAIN mix) is needed.
2. Decide whether the **≤0.5s latency gate** applies to LLM‑generation architectures at all (all realizations exceed it); if so, no generative architecture can pass and an alternative (non‑generative) realization would be required.
3. Given B and C both clear the deciding gate with 100% grounding, a secondary tie‑breaker (e.g., accuracy on the non‑Group‑E propose cases, or VRAM/latency trade‑off) is needed to choose between them.

## 6. Artifacts (preserved)

- `6b3_batch2/M1_Qwen2.5-0.5B-Instruct-supervised-{A,B,C}/` — adapters (SHAs above)
- `6b3_batch2/6c4_training_manifest_{A,B,C}.json` — training records
- `6b3_batch2/6c4_supervised_evaluation_results.json` — per‑case fixture results
- `6b3_batch2/6c-artifact-manifest.json` — `phase_6c4_supervised` section (full hashes)
- Historical v1 (`4dcdbe51…`) and v2 (`3462bd30…`) adapters preserved as diagnostics only.
