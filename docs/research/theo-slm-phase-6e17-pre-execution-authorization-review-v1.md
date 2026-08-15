# Phase 6E.17 Pre-Execution Authorization Review (v1)

- **Status:** FOR HUMAN FINAL AUTHORIZATION. **NOT AUTHORIZED FOR EXECUTION.**
- **Date (UTC):** 2026-08-16
- **Human decision so far:** AMEND AND RESUBMIT — protocol amended to v2.
- **Authoritative documents:** `theo-slm-causal-intervention-authorization-protocol-draft-v2.md` (the protocol, complete); this review is the concise pre-execution summary required by the amendment decision.
- **Framing (binding):** This is a **causal falsification test of H-Causal**, NOT a repair of a proven root cause. `readout misalignment ≺ gradient conflict` (temporal precedence, INFERRED) is not `⇒` (causation, NOT ESTABLISHED).

This review contains: exact intervention mathematics; exact matched baseline definition; capacity-confound control; frozen P1–P5 criteria; primary and secondary endpoints; interpretation matrix; complete hash/code/trajectory provenance plan. No execution is authorized by this document.

---

## 1. Exact intervention mathematics

Target: Qwen2ForCausalLM (Qwen/Qwen2.5-0.5B-Instruct), hidden_size **d=896**, vocab 151936, tied embeddings. Baseline decision margin at the decision position:

```
Δz_b = w_deltaᵀ h_L + b_delta        w_delta = W_emb[PROPOSE] − W_emb[ABSTAIN]   (frozen)
propose iff Δz_b > 0
```

**I1 (aligned trainable readout)** — replaces ONLY the two decision logits at the decision
position; the frozen LM vocabulary head produces all other 151,934 logits and all other
positions, byte-identical; the loss stays full-vocabulary cross-entropy (no reweighting):

```
s = aᵀ h_L + b          a ∈ R^896, b ∈ R        (897 new params)
ẑ_PRO =  s
ẑ_ABS = −s
propose iff s > 0
```

- **Vocabulary head:** fully frozen (no training, no adaptation).
- **Bias:** exactly one scalar `b`. No per-class biases. **Normalization:** none during training (scale-matched init only).
- **Initialization (deterministic, frozen):** `û_LDA` = step-0 Fisher LDA axis of the N=16 POS/ABS panel (must reproduce cos(LDA, w_delta) = 0.0055 ± 0.001); `a(0) = s_scale·û_LDA` with `s_scale = σ(Δz_b)/σ(û_LDAᵀh_L)` over the step-0 panel; `b(0) = −median_POS(a(0)ᵀh_L)`. Step-0 proposal rate must match baseline within ±2 pp (verified, not tuned).
- **Optimizer treatment:** `a,b` in the same AdamW (lr=1e-4, weight_decay=0.01, default betas), constant LR, gradients from the decision-position CE only.
- **LoRA byte-for-byte identical to matched baseline** in all conditions: targets `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`, r=16, alpha=32, dropout=0.05, identical init seed.

**C2 (capacity-matched control):** identical form and 897 params; `c(0) = s_scale·n`, `n` = unit random vector (seed 20260816, rejection-sampled to `|cos(n, û_LDA)| ≤ 0.05`); `b''(0) = −median_POS(c(0)ᵀh_L)`. I1 vs C2 differ ONLY in initialization orientation — identical capacity, trainability, optimizer. This is the capacity-confound control.

## 2. Exact matched baseline definition

Three runs of the SAME program — **B** (frozen w_delta readout), **I1** (aligned readout), **C2** (random readout) — differing ONLY in the §1 intervention variable. Historical runs (6E.13–6E.16) are NOT controls (D1). Frozen and recorded before execution:

1. Base model sha256 `fdf756fa…fb7fe`; revision `7ae557604adf67be50417f59c2c2f167def9a775`.
2. Corpus sha256 `a7b4e845…17eb0`.
3. **Code snapshot hash:** SHA-256 manifest of all executed source + config files, frozen before execution (no git repo currently; commit hash recorded if git is initialized).
4. Tokenizer: transformers 4.43.1, tokenizer file hashes, PROPOSE/ABSTAIN single-token verification.
5. LoRA targets/rank/alpha/dropout/init seed.
6. Optimizer + hyperparameters (AdamW 1e-4, wd 0.01).
7. LR schedule: constant (none) — matches historical loops.
8. Batch ordering: identical seeded shuffle per epoch (Generator(seed=42)) in all three runs.
9. Seeds: 42 (data/LoRA/dropout); 20260816 (C2 readout init only).
10. Panels: N=16, N=24; Euclidean; k∈{1,3,5}; resub + LOO-CV; SVM 3-fold.
11. Checkpoints: t ∈ {0,4,6,7,14,15,17,20} + full-step telemetry.
12. Stopping: fixed horizon 20 steps; online collapse guard (K=3×0.0%) recorded as a stopping event.

**Initial-state verification:** step-0 base weights re-hashed (must equal base hash) and initialized LoRA adapters hashed IDENTICAL across B/I1/C2; step-0 readout params hashed and recorded. Any mismatch voids the run.

## 3. Capacity-confound control

- I1 adds 897 trainable params. C2 adds the same 897 with random orientation.
- I1 vs C2 isolates **alignment** (only orientation differs); C2 vs B isolates **added capacity**.
- If `I1 ≈ C2`: capacity explanation strengthened; alignment claim weakened; H-Causal NOT supported.
- If `I1 ≫ C2` and `I1 ≫ B`: alignment-specific claim supported.

## 4. Primary and secondary endpoints

**Primary causal question:** Does the LDA-aligned trainable readout materially prevent/reduce POS/ABS decision-gradient conflict relative to a matched fresh baseline, and is the effect attributable to alignment (I1 vs C2) rather than capacity (C2 vs B)?

**Primary outcome:** change in POS/ABS decision-gradient alignment, `mean_{t∈{4,6,14,15}} cos(H1)(t)`.
**Co-primary mechanistic endpoints:** `ρ_CD(t) = max(0, −cos(H1)(t))` (common-descent infeasibility); `F(t)` = fraction of (POS,ABS) pairs with label-signed `g_POS·g_ABS > 0`.

**Secondary mechanistic outcomes (do NOT determine H-Causal):** policy performance, margins, collapse timing (t*), readout alignment after training, representation geometry, k-NN/SVM separability, proposal-rate dynamics.

## 5. Frozen P1–P5 criteria

| ID | Metric | Threshold |
|----|--------|-----------|
| P1 | mean cos(H1) over {4,6,14,15}, I1 | `≥ −0.30` |
| C2x | mean cos(H1)(I1) − mean cos(H1)(C2) | `≥ +0.25` |
| P2 | proposal rate > 0 through horizon, I1 | all t ≤ 20 |
| P3 | margin Cohen's d ≥ +0.2; r_eff ≤ 1.6, I1, t=15 | both |
| P4 | resub k-NN1 ∈ [0.65,1.0]; LOO within ±0.1 of B | both |
| P5 | \|cos(trained readout, û_LDA)\| ≥ 0.3 at t ≥ 4 | I1 |

Decision: **Supported** = P1 ∧ C2x ∧ P4. Partial = P1 ∧ ¬P2 ∧ P4. H0 survives = ¬P1. Capacity strengthened = ¬C2x. Any protocol violation = run void.

## 6. Interpretation matrix (pre-registered)

| Result | Interpretation |
|--------|----------------|
| Alignment ↑ + conflict ↓ + feasibility ↑ | Supports H-Causal |
| Alignment ↑ but conflict persists | H-Causal weakened / not sufficient |
| No alignment ↑ + no conflict ↓ | Intervention ineffective; H-Causal not tested strongly enough or unsupported |
| Conflict ↓ without alignment ↑ | Alternative mechanism likely |
| Performance ↑ but conflict persists | Symptom improvement, not causal confirmation |
| One class ↑ while other ↓ | NOT accepted as conflict resolution |
| I1 ≈ C2 (both improve) | Capacity explanation strengthened; alignment claim weakened |
| Protocol violation | Run void |

## 7. Hash / code / trajectory provenance plan

- **Inputs:** base model sha256 + revision; corpus sha256; tokenizer + transformers version; code snapshot SHA-256 manifest (frozen pre-execution).
- **Runs:** per-run manifest = config snapshot, seeds, environment, code hashes, GPU/dtype.
- **Initial state:** step-0 base re-hash; LoRA adapter init hash equal across B/I1/C2; readout init hashes recorded.
- **Checkpoints:** saved adapter + readout params hashed at t ∈ {0,4,6,7,14,15,17,20}.
- **Outputs:** artifacts dir `phase-6e17-causal-intervention/` with SHA-256 manifests; anti-fabrication discipline identical to 6E.16-R.
- **Trajectory identity:** comparisons ONLY within the B/I1/C2 matched pair; never against historical phases; no nominal-step cross-trajectory inference.

---

## Decision required from human reviewer

Approve, amend, or reject the protocol v2 (§3–§12 of `theo-slm-causal-intervention-authorization-protocol-draft-v2.md`). Until approval: **no model execution, no training, no optimizer step, no fine-tuning, no checkpoint creation.**

Sign-off: Reviewer: ____________  Decision: APPROVE / AMEND / REJECT  Date: ____________
