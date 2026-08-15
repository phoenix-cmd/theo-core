# THEO SLM Causal Intervention Authorization Protocol — DRAFT v2 (AMENDED)

- **Status:** DRAFT — DESIGN ONLY. **NOT AUTHORIZED FOR EXECUTION.**
- **Human review decision:** AMEND AND RESUBMIT FOR FINAL AUTHORIZATION (2026-08-16).
- **Supersedes:** `theo-slm-causal-intervention-authorization-protocol-draft-v1.md` (v1).
- **Companion document (required reading):** `theo-slm-phase-6e17-pre-execution-authorization-review-v1.md`.
- **Authority:** Requires explicit human authorization. Phase 6E.17 must not begin until this protocol is approved. No model execution, training, optimizer step, fine-tuning, or checkpoint creation is authorized by this document.
- **Basis:** Phase 6E.16-R `theo-slm-phase-6e16r-forensic-reconciliation-v1.md` (§8, §10 Canonical Interpretation Record).

This v2 incorporates every amendment requested in the human review decision. The
amendment log is at §11.

---

## 1. Causal framing (Amendment 1)

The following remains a **hypothesis**, not an established causal conclusion:

> Readout misalignment `≺` gradient conflict   (temporal precedence — INFERRED)

The evidence establishes temporal precedence and robust geometric association. It does
**not** establish:

> Readout misalignment `⇒` gradient conflict   (causation — NOT ESTABLISHED)

All Phase 6E.17 documentation MUST frame the experiment as a **causal falsification test
of H-Causal**, not as a repair of a proven root cause. The Canonical Interpretation Record
(6E.16-R §10.2) tiers apply unchanged. Any report language that states or implies
"the root cause is confirmed" is prohibited.

**H-Causal (falsifiable):** The failure arises substantially because the fixed
PROPOSE/ABSTAIN vocabulary contrast is poorly aligned with the task-discriminative
representation geometry; modifying the decision formulation to provide an appropriately
aligned trainable output geometry should reduce the measured scalar conflict and improve
simultaneous POS/ABS optimization.

**H0 (null to be falsified):** The scalar decision conflict and subsequent gate collapse
are caused by the training dynamics / objective themselves, independently of the readout
geometry; an aligned trainable readout produces no measurable reduction in scalar conflict
and no change in collapse onset. H-Causal is supported only if I1 beats both the matched
baseline AND the capacity-matched control (C2) on the primary endpoint (Amendment 5).

## 2. One minimal intervention family — unchanged and un-broadened (Amendments 6)

The authorized conceptual intervention family remains exactly:

**I1 — Trainable decision readout initialized from the step-0 discriminative axis.**

The following are DEFERRED, not part of I1, and explicitly PROHIBITED during the
evaluation of I1: verbalizer changes; multi-token labels; classifier-head alternatives;
PCGrad; MGDA; rank increases; LoRA target changes; optimizer changes; loss reweighting;
and any combination of multiple simultaneous fixes. If I1 fails, that is evidence; it must
not trigger an automatic sequence of additional interventions.

## 3. Exact intervention mathematics (Amendment 2)

### 3.1 Target model and baseline decision contrast

- Model: Qwen2ForCausalLM = Qwen/Qwen2.5-0.5B-Instruct; hidden_size **d = 896**;
  vocab_size 151936; `tie_word_embeddings = true`; revision
  `7ae557604adf67be50417f59c2c2f167def9a775` (verified in `model-config.json`).
- Training objective: standard next-token cross-entropy over the full vocabulary
  (labels with −100 masking), as in `run_phase_6e6_corrective_training.py`.
- The decision position predicts the decision token PROPOSE or ABSTAIN. With
  `z_tok = W_emb[token]ᵀ h_L + b[token]` (tied unembedding = embedding matrix), the
  **baseline decision margin** is:
  `Δz_b = z_PRO − z_ABS = w_deltaᵀ h_L + b_delta`,
  where `w_delta = W_emb[PROPOSE] − W_emb[ABSTAIN]` and
  `b_delta = b[PROPOSE] − b[ABSTAIN]` are both FROZEN. Decision: propose iff Δz_b > 0
  (matches the historical margin-sign ↔ proposal-rate relationship; collapse = margin
  locks negative).

### 3.2 I1 readout (single unambiguous implementation)

At the decision position ONLY, the two decision logits are replaced:

```
s(t) = a(t)ᵀ h_L(t) + b(t)
ẑ_PRO =  s
ẑ_ABS = −s
margin: Δz_I1 = ẑ_PRO − ẑ_ABS = 2 s        (propose iff s > 0)
```

- `a(t) ∈ R^896`, `b(t) ∈ R` are the ONLY new parameters (897 total).
- The other 151,934 vocabulary logits at the decision position and ALL logits at all
  other positions are computed by the **frozen original LM vocabulary head**, byte-identical
  to baseline. The loss at the decision position remains full-vocabulary cross-entropy
  (the two replaced logits simply take different values); the loss form, position weights,
  and mask are unchanged. **The intervention affects only the two decision logits at the
  decision position — nothing else. It does NOT reweight the loss and does NOT change the
  autoregressive target loss elsewhere.**
- **Original vocabulary head:** fully frozen (tied embedding weights and biases are not
  trained and not adapted).
- **Bias terms:** exactly one scalar bias `b` on the readout. No per-class biases.
- **Normalization:** `a` is NOT normalized and has no norm constraint during training.
  Initial scale is matched (below). No norm-clamp, no temperature parameter.
- **Parameter count added:** d + 1 = **897**.

### 3.3 Initialization (deterministic, frozen — no tuning)

- `û_LDA` = unit Fisher LDA axis of the step-0 hidden-state POS/ABS panel, exactly as
  recorded in `phase-6e16/diagnostic_basis_alignment_trajectory.json` (step 0, LDA;
  cos(LDA, w_delta) = 0.0055). The vector is re-derived by the same construction
  (`LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` on the N=16 panel) and
  must reproduce cos = 0.0055 ± 0.001 before execution proceeds.
- `s_scale = σ(Δz_b^(0)) / σ(û_LDAᵀ h_L^(0))`, both σ computed over the frozen step-0
  N=16 panel. Single deterministic scalar.
- `a(0) = s_scale · û_LDA`
- `b(0) = −median_{POS examples}(a(0)ᵀ h_L^(0))`  (centers the POS margin at zero-median so
  the step-0 decision-rate profile matches the baseline; no search).
- The step-0 decision-rate profile of I1 must match the baseline profile to within the
  pre-registered tolerance (±2 percentage points proposal rate on the N=16 panel),
  verified and recorded, NOT tuned.

### 3.4 C2 capacity-matched control (random orientation) — Amendment 5

Same mathematical form and parameter count as I1, differing ONLY in initialization
orientation:

```
ẑ_PRO'' = cᵀ h_L + b'' ;  ẑ_ABS'' = −ẑ_PRO''      (c ∈ R^896, b'' ∈ R; 897 params)
c(0) = s_scale · n ;  b''(0) = −median_POS(c(0)ᵀ h_L)
n = unit vector, drawn from torch.Generator(seed=20260816), standard normal, normalized,
    rejection-sampled until |cos(n, û_LDA)| ≤ 0.05  (deterministic given the seed;
    the realized cos is logged in the run manifest)
```

C2 is trained identically to I1. Its ONLY difference from I1 is the orientation of the
initialized readout (aligned vs random). I1 and C2 therefore carry IDENTICAL added
trainable capacity and identical optimizer treatment; only orientation differs. This is
the pre-registered control for the **capacity confound** (see §6).

### 3.5 Optimizer treatment of the new parameters

- `a,b` (and `c,b''`) are trained by the SAME optimizer instance as the LoRA parameters:
  AdamW, lr=1e-4, weight_decay=0.01, default betas (0.9, 0.999), eps=1e-8. No separate
  learning rate, no separate warmup, no separate schedule.
- They receive gradient from the decision-position cross-entropy only.
- **LoRA configuration remains byte-for-byte identical to the matched baseline** across
  all three conditions: targets `["q_proj","k_proj","v_proj","o_proj","gate_proj",
  "up_proj","down_proj"]`, r=16, lora_alpha=32, lora_dropout=0.05, identical init seed,
  identical optimizer, identical scheduler (constant LR; no scheduler in the historical
  training loops).

## 4. Matched baseline definition (Amendment 3)

A fresh control run (condition **B**) is mandatory. Historical runs (6E.13–6E.16) MUST
NOT be used as controls (D1: trajectory identity). Conditions B, I1, C2 are three runs of
the SAME program differing ONLY in the pre-registered intervention:

| Program module | B (baseline) | I1 (aligned readout) | C2 (random readout) |
|----------------|--------------|----------------------|---------------------|
| decision logits at decision position | frozen `w_delta, b_delta` | `s = aᵀh_L+b`; ẑ_PRO=s, ẑ_ABS=−s | `s''=cᵀh_L+b''`; ẑ_PRO=s'', ẑ_ABS=−s'' |
| new trainable params | none | a,b (897) | c,b'' (897) |
| readout init orientation | n/a | `û_LDA` (cos 0.0055→aligned) | random (|cos(n,û_LDA)|≤0.05) |
| everything else | — | IDENTICAL | IDENTICAL |

**Frozen and recorded in the pre-registration snapshot (before execution):**
1. Base model hash: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`; model revision `7ae557604adf67be50417f59c2c2f167def9a775`.
2. Corpus hash: `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0`.
3. Code snapshot hash: SHA-256 manifest of every executed source file and config at freeze time (the repo is not currently a git repo; a code snapshot directory with a SHA-256 manifest is created and frozen BEFORE execution; if git becomes available, the commit hash is also recorded).
4. Tokenizer: `transformers` version 4.43.1, tokenizer file hashes, decision-token verification (`PROPOSE` and `ABSTAIN` must be single tokens; reproduced from the 6E.15 script).
5. LoRA targets, r, alpha, dropout (as §3.5); LoRA init seed.
6. Optimizer + hyperparameters (AdamW lr=1e-4, weight_decay=0.01, betas, eps).
7. Learning-rate schedule: constant (no scheduler) — matches the historical training loops.
8. Batch ordering policy: identical seeded shuffle per epoch for all three conditions (`torch.Generator().manual_seed(42)`), documented in the manifest.
9. Seed policy: seed=42 for data ordering, LoRA init, and dropout; seed=20260816 for C2 readout init only (documented).
10. Diagnostic panels: N=16 (8 POS + 8 ABS) and N=24 (+8 NEG); Euclidean; k∈{1,3,5}; resubstitution + LOO-CV; linear SVM 3-fold.
11. Evaluation checkpoints: t ∈ {0, 4, 6, 7, 14, 15, 17, 20} plus full-step telemetry.
12. Stopping criteria: fixed horizon of 20 optimizer steps, with the online collapse-detector guard (halt after K=3 consecutive steps at 0.0% proposal) recorded as a stopping event, not as a silent termination.

**Initial-state verification (mandatory):** each of B, I1, C2 begins from an
independently loaded base model whose weights are re-hashed at step 0 (must equal the
base-model hash), and whose initialized LoRA adapter parameters are hashed and asserted
IDENTICAL across B, I1, C2. The step-0 readout parameters (a(0), c(0)) are hashed and
recorded. Any mismatch voids the run.

## 5. Pre-registered primary causal test (Amendment 4)

**Primary causal question (single):**

> Does replacing the fixed misaligned decision readout with the pre-registered
> LDA-aligned trainable readout materially prevent or reduce the emergence of POS/ABS
> decision-gradient conflict relative to a matched fresh baseline, and is that effect
> attributable to alignment (I1 vs C2) rather than added trainable capacity (C2 vs B)?

**Primary endpoint (decision metric):** the change in POS/ABS decision-gradient alignment,
measured as the mean over the conflict window steps {4, 6, 14, 15} of:

1. **POS/ABS decision-gradient alignment** `cos(H1)(t)` — mean pairwise label-signed
   cross-cosine between POS and ABS decision gradients (computed as in 6E.13/6E.14;
   negative = conflict; the historical lock-in value is ≈ −0.93 to −0.997).
2. **Minimum-norm common-descent residual** `ρ_CD(t) = max(0, −cos(H1)(t))` —
   pre-registered operationalization of common-descent (in)feasibility: 0 = feasible,
   →1 = infeasible. The implementation MUST use this definition, frozen; it does not
   depend on any historical artifact's formula.
3. **Per-example simultaneous improvement feasibility** `F(t)` — fraction of (POS,ABS)
   pairs on the N=16 panel with `g_POS · g_ABS > 0` (label-signed), at each checkpoint.

`cos(H1)` is THE pre-registered primary outcome; ρ_CD and F are co-primary mechanistic
endpoints. All other metrics (policy performance, margins, collapse timing, readout
alignment, representation geometry, k-NN/SVM separability) are **secondary mechanistic
outcomes** and do NOT determine H-Causal acceptance.

## 6. Capacity-confound control and failure conditions (Amendment 5)

**Capacity confound.** I1 adds 897 trainable parameters. Without C2, a positive I1 result
cannot distinguish "aligned readout" from "extra trainable degrees of freedom". C2 (§3.4)
is the mandatory control: I1 and C2 are identical in capacity, trainability, and
optimizer treatment; they differ only in initialization orientation. Therefore:

- If `I1 ≈ C2` on the primary endpoint → capacity explanation strengthened; the
  alignment-specific claim is weakened. H-Causal NOT supported.
- If `I1 ≫ C2` and `I1 ≫ B` → alignment-specific claim supported.

**H-Causal is NOT supported if the intervention:**
1. Successfully changes readout alignment but gradient conflict still emerges
   substantially unchanged (cos(H1) stays ≲ −0.6).
2. Improves the readout geometry but does not improve common-descent feasibility (ρ_CD
   unchanged).
3. Prevents apparent collapse only by degrading one class (e.g., balanced accuracy on
   POS or ABS collapses to ≤ chance).
4. Improves aggregate metrics while preserving the same per-example POS/ABS trade-off
   (per-class decision confusion worsens while the mean improves).
5. Produces an advantage explainable only by unmatched additional trainable capacity
   rather than by alignment (i.e., I1 does not separate from C2).

**Protocol violations void the run:** any deviation from §3 (exact math), §4 (frozen
controls), or §7 (trajectory identity) — including any un-pre-registered variant,
optimizer difference, seed difference, or checkpoint divergence between the three
conditions.

## 7. Frozen success/failure criteria (P1–P5) and decision rules

Thresholds are FROZEN here; no post-hoc adjustment.

| ID | Criterion | Definition | Threshold |
|----|-----------|-----------|-----------|
| P1 | Conflict reduction (primary) | mean `cos(H1)` over {4,6,14,15}, I1 | `≥ −0.30` |
| C2x | Alignment-exclusion (primary, mandatory) | mean `cos(H1)`(I1) − mean `cos(H1)`(C2) | `≥ +0.25` |
| P2 | Collapse prevention (secondary) | proposal rate strictly > 0 through horizon, I1 | `> 0.0%` at all t ≤ 20 |
| P3 | Margin maintenance (secondary) | scalar-margin Cohen's d, I1, t=15 | `≥ +0.2`; margin r_eff ≤ 1.6 |
| P4 | No separability degradation (guard) | resub k-NN1 (both panels), I1 | `∈ [0.65, 1.0]`; LOO-CV within ±0.1 of B |
| P5 | Readout engagement (guard) | max \|cos(w_delta', û_LDA)\| for trained readout, I1 | `≥ 0.3` at t ≥ 4 |

**Decision rules (frozen):**
- **H-Causal supported:** P1 AND C2x AND P4 hold.
- **H-Causal partially supported:** P1 holds, P2 fails, P4 holds → write up, return to
  human review for a second, separately approved hypothesis about the residual mechanism.
- **H0 survives:** P1 fails → M4 demoted to correlate, not cause; NO automatic follow-up
  interventions; write up and return to human review.
- **Capacity explanation strengthened:** C2x fails (I1 ≈ C2) → alignment-specific claim
  weakened; write up and return to human review.
- **Any protocol violation:** run void regardless of outcome.

## 8. Interpretation matrix (pre-registered — Amendment 8)

| Result | Interpretation |
|--------|----------------|
| Alignment improves + conflict decreases + simultaneous feasibility improves | Supports H-Causal |
| Alignment improves but conflict persists | H-Causal weakened / not sufficient |
| No alignment improvement + no conflict improvement | Intervention ineffective; H-Causal not tested strongly enough or unsupported |
| Conflict improves without alignment improvement | Alternative mechanism likely |
| Performance improves but conflict persists | Symptom improvement, not causal confirmation |
| One class improves while the other worsens | NOT accepted as conflict resolution |
| Both I1 and C2 improve similarly (C2x fails) | Capacity explanation strengthened; alignment-specific claim weakened |
| Protocol violation (any §3/§4/§7 deviation) | Run void |

## 9. Trajectory identity discipline (Amendment 7)

- Comparisons are valid ONLY within the matched experimental pair
  (B↔I1↔C2), never against historical phase numbers except as reproduction checks (§4.12
  and §11 of v1). "Nominally identical checkpoint numbers across unrelated trajectories"
  must not be compared and inferred as mechanistic progression.
- Every trajectory has its own: initialization identity (step-0 hashes), per-checkpoint
  parameter hashes, run manifest (config snapshot + seeds + environment), and immutable
  provenance record (SHA-256 manifest of inputs, code, and outputs), following the
  6E.16-R anti-fabrication discipline.
- Checkpoint hashes: saved adapters + readout params hashed at t ∈ {0,4,6,7,14,15,17,20}.

## 10. Outputs and forensic packaging

- Artifact directory `phase-6e17-causal-intervention/` (pending human approval of the
  phase): pre-registration snapshot (this document + hash), run configs (B, I1, C2), full
  telemetry, spectra, alignment trajectories, chronology JSON, checkpoint hashes,
  SHA-256 manifests.
- Report: `theo-slm-phase-6e17-causal-intervention-v1.md` with an explicit update to the
  Canonical Interpretation Record tier assignments and an explicit verdict row selected
  from the §8 matrix.

## 11. Amendment log (v1 → v2)

1. Causal framing: experiment re-labeled as falsification test of H-Causal; `≺` vs `⇒`
   distinction made binding in all documentation.
2. Intervention tightened to an unambiguous implementation (exact math §3; frozen head;
   decision production; decision-position-only scope; initialization; bias; no
   normalization; param count 897; optimizer treatment; LoRA byte-for-byte identity).
3. Matched baseline definition strengthened with the 12-item frozen/recorded list; code
   snapshot hash; step-0 identity verification for every run.
4. Primary causal test pre-registered (single primary question + `cos(H1)` primary
   outcome + ρ_CD and F co-primary endpoints; others demoted to secondary).
5. Failure conditions defined as rigorously as success conditions; mandatory C2
   capacity-matched random-orientation control added; capacity-confound decision rules.
6. No-broaden list made explicit (no PCGrad/MGDA/rank/verbalizer/multi-token/optimizer/
   loss changes during I1 evaluation).
7. Trajectory identity discipline made binding (within-pair comparisons only; per-run
   provenance).
8. Interpretation matrix pre-registered (8 rows).

## 12. Authorization gate (this document authorizes nothing)

Before any execution, a human reviewer must approve ALL of: the single intervention
variable (I1) and the deferred/prohibited list (§2); the exact mathematics (§3); the
matched baseline definition and frozen list (§4); the primary test and P1/C2x as primary
(§5, §7); the C2 capacity control and failure conditions (§6); the interpretation matrix
(§8); and the trajectory/provenance plan (§9–§10).

Sign-off fields:

- Reviewer name / role: ____________
- Decision: APPROVE / AMEND / REJECT
- Amendments (if any): ____________
- Date: ____________

Until completed by a human, this protocol is a draft and NO training, fine-tuning,
optimizer step, or checkpoint creation may occur.
