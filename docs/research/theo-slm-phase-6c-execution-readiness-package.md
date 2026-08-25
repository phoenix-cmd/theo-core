# THEO SLM v0 — Phase 6C Execution / Readiness Package

**Document ID:** `docs/research/theo-slm-phase-6c-execution-readiness-package.md`  
**Date:** 2026-08-25  
**Depends on:** `theo-slm-phase-6c-model-selection-plan.md` (APPROVED) + its audit (APPROVED)  
**Status:** READINESS PACKAGE — **does NOT authorize execution.** Awaiting separate human execution authorization.

---

## 0. Approval lineage & preserved constraints

This package is built on the **APPROVED** 6C model-selection plan and its **APPROVED** independent audit
(2026-08-25). The following constraints are binding on everything below and on any future 6C execution:

1. **Canonical dataset immutable** at SHA-256
   `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4` (300 GOLD: 84 POS / 66 ABSTAIN /
   150 NEG; 44 REJECT excluded).
2. **Batch 3 deferred** — generated only if a later 6C/6D evaluation shows a concrete *blocking* deficiency.
3. **No `cos(H1)` / gradient-compatibility / hidden-state-subspace** criteria in selection or evaluation.
4. **No Qwen-generated labels** as training positives (Qwen outputs = negatives / format-regressions only).
5. **Candidate architectures limited to the approved §F set**: A (decoder-only), B (decoder+reranker),
   C (shared-encoder).
6. **Projected scores are provisional** — final architecture selection uses *measured* 6C.3/6D results,
   never the plan's §4 projections.

---

## 1. Exact candidate models / architectures to be evaluated

The selection axis is **architecture** (A/B/C); the base-model family is a secondary, model-agnostic
choice (v0.5 Constraint 4). Evaluation enumerates base-model × architecture combinations, prioritized
below.

### 1.1 Candidate base models (≤1B open-weight tier, permitted sources only)

| ID | Model (Hugging Face repo) | Params | License | In §F budget? |
|---|---|---|---|---|
| **M1** | `Qwen/Qwen2.5-0.5B-Instruct` | ~0.49B | Apache 2.0 | **Yes** (primary) |
| **M2** | `HuggingFaceTB/SmolLM2-360M-Instruct` | ~0.36B | Apache 2.0 | **Yes** (lightweight CPU) |
| M3* | `meta-llama/Llama-3.2-1B-Instruct` | ~1.24B | Llama 3.2 Community | **No** — over ≤1B budget; evaluate **only** if a budget waiver is separately authorized |

> Clarification on "no Qwen labels": prohibition (4) concerns **training labels**, not base-model
> **weights**. Using `Qwen/Qwen2.5-0.5B-Instruct` as a fine-tune *base* is permitted and was the prior
> 6C.2 recommendation; it is distinct from using Qwen *outputs* as supervision.

### 1.2 Architecture × base-model evaluation matrix (priority order)

| Eval ID | Architecture | Base model | Priority | Notes |
|---|---|---|---|---|
| **E1** | A — single decoder-only | M1 (Qwen2.5-0.5B) | **Primary** | §F default; smallest footprint |
| **E2** | A — single decoder-only | M2 (SmolLM2-360M) | Secondary | CPU-feasibility baseline |
| **E3** | B — decoder + reranker | M1 | **Conditional** | Run only if E1 fails Group E (C7) distractor-rejection gate |
| **E4** | B — decoder + reranker | M2 | Conditional | CPU-feasible reranker variant |
| **E5** | C — shared encoder / two-head | M1 | Lower | Only if E1–E4 fail a v0.5-isolation or budget gate |
| E6* | A or B | M3 (Llama-1B) | Waiver-only | Blocked unless budget waiver authorized |

---

## 2. Permitted model sources

- **Source:** official Hugging Face Hub repositories **only**, from the verified organization namespaces
  (`Qwen/`, `HuggingFaceTB/`, `meta-llama/`). No mirrors, no unverified forks, no locally cached
  third-party weights.
- **License gate:** Apache 2.0 (M1, M2) preferred. M3 (Llama 3.2 Community) requires a separate license
  waiver authorization before any download.
- **Integrity:** download via pinned `revision` (commit SHA); verify blob SHA-256 at fetch time; reject
  on mismatch. No model is fetched from anywhere other than the official repo at the pinned revision.
- **Prohibited content:** no dataset or label file sourced from Qwen *outputs*; the 300-record GOLD
  corpus (`adjudication.json`, hash `97bc6bc6…`) is the sole supervision source.

---

## 3. Exact artifact hashes bound after acquisition

> These hashes are **defined here as the binding set**; the actual values are recorded at acquisition
> time into `theo-data/datasets/theo_slm_v0_review/6b3_batch2/6c-artifact-manifest.json` (created during
> execution, not now). Until then, the bindings are *placeholders* and execution is blocked.

| Artifact | Hash to bind | When bound |
|---|---|---|
| Canonical dataset `adjudication.json` | `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4` | already bound (frozen) |
| Base model repo `resolved_commit` (git revision) | SHA-1 of pinned commit | at download |
| Base model `config.json` | SHA-256 | at download |
| Base model `tokenizer.json` + `tokenizer_config.json` | SHA-256 (each) | at download |
| Base model `model.safetensors` (each shard) | SHA-256 (each) | at download |
| Reranker (B) adapter + config + weights | SHA-256 (each) | post-6C.3 training |
| Encoder (C) adapter + config + weights | SHA-256 (each) | post-6C.3 training |
| 6C.3 training adapter checkpoint(s) | SHA-256 | post-6C.3 training |
| Evaluation output manifest (predictions/logs) | SHA-256 | post-6C/6D |

**Binding rule:** no artifact may be used in training or evaluation unless its hash is recorded and
matches the pinned expectation. A hash mismatch aborts execution (rollback to last green).

---

## 4. Resource / storage requirements per candidate (pre-download feasibility)

> All figures are **planning estimates** (params × precision) to drive the readiness gate; they are
> re-measured at execution. Precision assumptions: FP16 = 2 bytes/param; INT4 = 0.5 bytes/param.
> Primary training method = **QLoRA (4-bit base + LoRA adapters)** to satisfy the §F ≤2 GB VRAM and
> storage-constrained environment.

### 4.1 Static model footprint

| Model | FP16 weights | INT4 weights | Notes |
|---|---|---|---|
| M1 Qwen2.5-0.5B | ~0.98 GB | ~0.25 GB | 32k context, 151k vocab |
| M2 SmolLM2-360M | ~0.72 GB | ~0.18 GB | 8k context |

### 4.2 Training footprint (peak, QLoRA primary)

| Component | M1 (QLoRA) | M2 (QLoRA) | Full-FT alternative (M1) |
|---|---|---|---|
| Base in 4-bit | ~0.25 GB | ~0.18 GB | — (FP16 0.98) |
| LoRA adapters (trainable) | ~0.05 GB | ~0.04 GB | — |
| Optimizer (AdamW on adapters) | ~0.1–0.3 GB | ~0.1–0.3 GB | ~5.9 GB (FP32 m,v + fp32 master) |
| Gradients / activations | ~0.2 GB | ~0.15 GB | ~0.98 GB |
| **Peak training VRAM** | **~0.6–0.8 GB** | **~0.5–0.7 GB** | **~7.8 GB** (exceeds ≤2 GB §F target) |
| **Peak training RAM** | **~1.5–2.5 GB** | **~1.2–2.0 GB** | **~8–9 GB** |

**Feasibility conclusion:** QLoRA keeps both M1 and M2 within the §F ≤2 GB VRAM / ≤4 GB RAM budget and is
the **recommended** training method. Full fine-tune exceeds the VRAM target and is selected only if
QLoRA gates fail *and* storage/RAM headroom is separately confirmed.

### 4.3 Storage budget (disk) per candidate

| Item | M1 (QLoRA) | M2 (QLoRA) | Full-FT delta |
|---|---|---|---|
| Base model download (safetensors) | ~1.0 GB | ~0.75 GB | — |
| Training checkpoints (adapters, ≤3) | ~0.2 GB | ~0.15 GB | +~3 GB (FP16 ckpts) |
| Reranker/encoder component (B/C) | +<1 GB | +<1 GB | — |
| Evaluation artifacts | <0.5 GB | <0.5 GB | <0.5 GB |
| **Per-candidate total (QLoRA)** | **~3 GB** | **~2.5 GB** | **+~3 GB** |

### 4.4 Inference latency (indicative, reconfirm at 6D)

| Model | GPU (s/case) | CPU (s/case) | Meets ≤0.5 s GPU / few s CPU? |
|---|---|---|---|
| M1 Qwen2.5-0.5B | ~0.15 | few | Yes |
| M2 SmolLM2-360M | ~0.10 | few | Yes |

---

## 5. Behavioral / structured-output evaluation matrix

Applied at 6C.3 (build) and 6D (comparative eval) against **behavioral gates only** — no prohibited
signal (constraint 3). Reference instruments are **evaluation-only, byte-locked**: frozen 51-case
benchmark (v0.4.1 + v0.5 Phase-2 expansion) and frozen 15-case semantic probe (6A.2). Training uses the
300-record GOLD via a **grouped-by-seed-family** split that preserves the intentional `/005`∼`/050`
near-duplicate group (split-leakage = 0).

| Gate | Definition | GO threshold | Instrument |
|---|---|---|---|
| **E0** format reliability | outputs failing `SemanticInterpretation` schema/adapter | ≤ 2.0% | probe + benchmark |
| **Grounding** | referenced concept/evidence ids valid in snapshot | 100% | probe + benchmark |
| **E5** semantic novelty | non-derivable, evidence-supported novel interpretations | ≥ 40% | probe (per group, human-verified) |
| **E6** decision relevance | novel interpretation directly answers the task | ≥ 30% | probe |
| **Group E** distractor rejection | distractors ignored on Group E cases | ≥ 80% | probe (deciding gate for A vs B) |
| **Abstention** accuracy | incomplete/premature → model abstains | ≥ 90% | `GOLD_ABSTAIN` split |
| **Shortcut resistance** | balanced-acc classifier on predictions | ≤ 40% (chance) | probe/benchmark |
| **Latency** | s/case | ≤0.5 GPU / few s CPU | timer |
| **Footprint** | VRAM/RAM | ≤2 GB VRAM / ≤4 GB RAM | profiler |
| **v0.5 compat** | baseline invariance, determinism, replay, firewall | all green | suite + benchmark 35/35 |

---

## 6. Exact decision rule — decoder-only primary vs reranker fallback

1. **Primary = A (single decoder-only).** Evaluate **E1 (M1)** first.
2. If **E1** clears **all** gates **and** Group E distractor-rejection ≥ 80% (no regression on E0 /
   grounding / E5 / E6), **select A (E1)**.
3. If **E1** fails Group E (C7) while **E3 (M1 + reranker, B)** clears it **and** meets all other gates,
   **select B (E3)**. The reranker is a **separate, snapshot-only** provider component — no symbolic
   state writes, no violation of v0.5 Constraint 3.
4. **M2 (E2/E4)** is the CPU-feasibility alternative; it is selected over M1 only if M1 fails a
   footprint/latency gate that M2 passes.
5. **C (E5)** is selected only if A and B both fail a v0.5-isolation or budget gate (not projected).
6. **M3 (E6)** is blocked unless a budget waiver is separately authorized.
7. **Final selection is made on measured 6C.3/6D gates, never on the plan's §4 projections** (constraint 6).

---

## 7. Frozen v0.5 / ADR-0028 interface constraints (confirmation)

- **C1 Baseline invariance:** with no provider configured, runtime is byte-identical to v0.4.1 (35/35
  PASS, identical fingerprints/state hashes). The SLM influences cognition only through the five ADR-0028
  hooks.
- **C3 Determinism / replay / grounding:** greedy fixed-seed decoding; byte-stable replay on the probe;
  the 6A.2 rationale-ordering defect is a regression test; every output grounded against the snapshot
  boundary before influencing cognition.
- **C4 Model independence:** the architecture selection (A/B/C) and base-model family are recorded in
  the design doc / research notes, **never** written into ADR-0028. THEO "supports neural providers,"
  it does not "uses \<model\>."
- **Architectural firewall (permanent):** `theo_core` MUST NOT import `theo_providers`; providers import
  `theo_core.models.ports` only. This test gates any 6C build.
- **Dataset immutability (constraint 1):** `adjudication.json` hash `97bc6bc6…` is never modified by 6C.

---

## 8. Strategic recommendation — resource feasibility as a first-class 6C criterion

Because the current environment is **storage-constrained** (the 6E20.1 checkpoints, ~96.8 GB, are still
pending cloud upload and local purge), resource feasibility must be treated as a **first-class selection
criterion**, on par with E0/grounding/novelty.

**Pre-download feasibility checklist — computed for EVERY candidate before any fetch:**

```
for each candidate (model × architecture):
    model_size          = params × precision bytes        # §4.1
    optimizer_footprint = QLoRA (tiny) or Full-FT (~12×P) # §4.2
    vram_required       = base + adapters + optim + act.  # must be ≤ 2 GB (§F)
    ram_required        = peak training RAM               # must be ≤ 4 GB (§F)
    storage_required    = download + checkpoints + eval   # §4.3, vs FREE disk
    inference_latency   = s/case (GPU/CPU)               # must meet §4.4
    gate = (vram_required ≤ 2GB) AND (ram_required ≤ 4GB)
           AND (storage_required ≤ FREE_DISK − BUFFER)
           AND (latency meets target)
    if not gate: candidate is INFEASIBLE -> drop or defer, do NOT attempt build
```

**Rationale:** there is little value in selecting a theoretically excellent architecture the environment
cannot build and evaluate reproducibly. The **next real gate is 6C readiness + resource feasibility**,
not another dataset-generation cycle. Concretely:

- **Free-disk precondition:** verify `FREE_DISK ≥ Σ(candidate storage) + BUFFER (≥15 GB)` *before* any
  download. If not met, the 6E20.1 checkpoint cloud-upload/purge (deferred task) must complete first, or
  candidates must be reduced.
- **Training method default = QLoRA** (§4.2) to stay within VRAM/RAM/storage limits; full fine-tune is a
  registered fallback requiring confirmed headroom.
- Feasibility is recomputed at execution time with measured numbers; the §4 estimates are planning inputs
  only.

---

## 9. Execution authorization boundary — explicit non-authorization

> This package is a **readiness** artifact. It confers **NO** execution authority.

The following remain **NOT authorized** by this document and require separate, explicit human sign-off:

- ☐ Downloading any model weights / tokenizer / config (§2–§3 bindings are placeholders only).
- ☐ Starting 6C.3 build / training / fine-tuning (including QLoRA adapter training).
- ☐ Executing 6D comparative evaluation.
- ☐ Generating Batch 3 (still deferred per constraint 2).

**STOP — awaiting separate human execution authorization** for 6C.3 (and, later, 6D). The readiness gate
of §8 (resource feasibility) must be cleared as part of that authorization.
