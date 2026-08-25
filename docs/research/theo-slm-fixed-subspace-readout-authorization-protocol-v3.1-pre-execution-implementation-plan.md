# THEO SLM Fixed Subspace Readout Authorization Protocol — v3.1 Pre-Execution Implementation & Conformance Plan

- **Status:** PLANNING ONLY — DESIGN APPROVED FOR IMPLEMENTATION PLANNING ONLY. **NO EXECUTION.**
- **Governing artifact (frozen):** `theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md`, SHA-256 `E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85`, 400 lines. Frozen by human decision 2026-08-16 after the targeted re-audit PASS (design-review scope). This plan does **not** amend v3.1; any future amendment requires a documented amendment and re-adoption (§12, §13 of v3.1).
- **Phase:** 6E.20 (Fixed Low-Dimensional & Subspace Readout Architectures).
- **Superseded governing draft:** v3.0, SHA-256 `F6F1DC23D2D4678F4C6F9A7FE09F59C37B21227DF7EA50E053BC53123A8D00B1` (audited, read-only, unmodified).
- **Authority status:** No model execution, training, fine-tuning, optimizer steps, checkpoint generation, or scientific verdicts are authorized by this plan. Execution remains conditional on: (1) implementation conformance to this plan, (2) formal pre-execution audit, and (3) explicit signed human authorization (`execution_authorization.json`, decision=APPROVE, phase=6E.20) — §11 of v3.1.

---

## 1. Purpose and Scope

This plan converts the frozen v3.1 design into an ordered implementation and conformance program. It defines, per v3.1 section, the concrete objects, routines, assertions, and artifacts an implementation must produce, and how conformance will be verified. It is a **planning document only**: it specifies work to be done later, it executes nothing, and it changes no frozen file.

In scope: everything required to implement and verify Phase 6E.20 per frozen v3.1 (§3–§10), including the preflight mode (§10.6), the metric layer (§5.7/§10.5), the gates (§6), the evaluation and verdict logic (§7–§9), and the provenance/authorization artifacts (§10.10).

Out of scope (prohibited at this stage): any runner execution, training, optimizer steps, checkpoint generation, scientific verdicts, and any modification of v3.0/v3.1 or of Phase 6E.18 execution logs and the Phase 6E.19 adjudication record (v3.1 §10.8).

---

## 2. Governing Inputs (frozen, read-only)

| Input | Role | Source |
|---|---|---|
| v3.1 protocol (frozen SHA above) | normative design | `docs/research/theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md` |
| v3.0 protocol (SHA above) | superseded, audited, unmodified | `docs/research/theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.md` |
| Phase 6E.18 runner | source of inherited operational rules and of the legacy hash / metric computations | `scripts/dataset_generator/run_phase_6e18_causal_intervention.py` |
| Phase 6E.18 v2.1.3 protocol | source of inherited §6.2/§7.1/§10 rules cited by v3.1 | `theo-causal-intervention-authorization-protocol-draft-v2.1.3.md` |
| Phase 6E.18 artifacts | registered historical hashes/values (read-only) | `readout_initialization.json`, `t2_basis.json`, `pre_registration_snapshot.json` |
| Phase 6E.19 forensic record | informational context only; no value enters any threshold | `theo-slm-phase-6e18-post-execution-forensic-analysis-v1.md` |

---

## 3. Conformance Model (v3.1 § → Implementation Requirement → Verification)

Each conformance row names the v3.1 section, the required implementation deliverable, and the verification method. Verification is by code review, static analysis, deterministic unit/reconstruction tests, and `--preflight` self-checks — all of which are themselves part of implementation and are authorized to be *written and run as dry self-tests on synthetic/registered data* only; none runs the model, a training step, or an optimizer step.

| # | v3.1 § | Implementation requirement | Conformance verification |
|---|---|---|---|
| C1 | §3.1 | Step-0 panel collection `H_16 ∈ R^16×896` (layer-24 last-token hidden states, rows 0:8 POS / 8:16 ABS) and pinned label vector `y16 = [+1]^8 ⊕ [-1]^8`; panel hash (canonical convention) recorded in manifest | reconstruction determinism test; label-vector encoding test |
| C2 | §3.2 | `û_LDA` computation (sklearn LDA, `solver="lsqr"`, `shrinkage="auto"`), unit norm; legacy hash must equal registered `u_hat_LDA_hash == ff3cab49…`; canonical hash `u_lda_tensor_hash`; artifact `u_lda.npy` | G1 test: legacy identity + canonical artifact hash + reconstruction (cosine ≥ 0.9999, max-abs ≤ 1e-6) |
| C3 | §3.3 | `s_scale` per formula with ε = 1e-12; reference `0.06681545` matched at preflight; degenerate guard (non-finite or ≤ 0 ⇒ registration fails ⇒ run VOID, G8) | preflight assertion test |
| C4 | §3.4 | Covariance + SVD exactly as 6E.18 §8; canonical sign convention (multiply column by −1 if largest-magnitude element < 0, ties by lowest row index); two hash conventions implemented and kept distinct; legacy `basis_sha256` of sign-unpinned `U_8^raw` must equal `b8dfbd6a…`; nested `U_2⊂U_4⊂U_8`; artifact `U_8.npy` + `U_8_tensor_hash`; cross-convention equality never asserted | G2 test; nesting test; sign-convention unit tests; negative test that legacy ≠ canonical hashes |
| C5 | §3.5 | `κ_k = ‖U_k U_kᵀ û_LDA‖₂` (k ∈ {2,4,8}) and `κ_R = ‖R_4 R_4ᵀ û_LDA‖₂`; finite and in (0,1) | G3/G5/G8 assertions |
| C6 | §3.6 | B0 vector `w_δ = W_head[PROPOSE] − W_head[ABSTAIN]` from `model.get_output_embeddings().weight` (`w_head[PRO_ID] − w_head[AB_ID]`), PRO_ID=9117, AB_ID=1867, `b_δ = 0`; `w_delta_norm = 0.564183` matched; unembedding source only (no `W_emb`) | identity test vs registered value; source-assertion test |
| C7 | §3.7 | `R_4` (QR of 896×4 standard-normal, `torch.Generator(seed=20260820)`) and `n_rand` (unit, `|cos(n_rand, û_LDA)| ≤ 0.05`, same seed, rejection sampling); artifacts `R_4.npy`, `n_rand.npy` + hashes | determinism test (re-draw identical); cosine-bound test |
| C8 | §4 | Seven conditions (B0, F1, S2, S4, S8, C4-Rand, R1-Rand) with EXACTLY ZERO trainable readout parameters; common gain `‖w‖₂ = s_scale` for the six scalar readouts; B0 excluded; bias `b_c = −median(w_c @ H_16[:8])` for all scalar conditions (B0 `b_δ = 0`) | parameter-count assertions at registration and (later, at execution) every checkpoint (§10.1) |
| C9 | §5.1–§5.5 | Forward `s = wᵀh_L + b`, logits `z_PRO = s`, `z_ABS = −s` (runner lines 331–345, 375–379); weight forms per §5.3/§5.4/§5.5; readout-only backward `∂L/∂h_L\|_readout = α w` | synthetic forward/backward identity tests |
| C10 | §5.6 | Pairwise-distinctness checks (B0 vs all; F1 vs Sk; S2/S4/S8 mutual; C4-Rand vs all; R1-Rand vs all) verified numerically at preflight | preflight distinctness assertions |
| C11 | §5.7 | Dual gradient objects per the runner's `per_sample_decision_gradient` (lines 1006–1033): `g_R^θ = α_c ∇_θ s` and `g_total = autograd.grad(dec_loss, lora_param_list)`; `α_POS = p_PRO − p_ABS − 1`, `α_ABS = p_PRO − p_ABS + 1` (post-patch softmax values); decomposition `g_total = g_bg + g_R`; `ρ_c^(x)` per-sample cosine, `ρ_c` mean over {4,6,14,15}×{POS,ABS}, `σ_c = mean((ρ_c^(x))²)`; readout-only cosine note (sign ∈ {−1,0}) | synthetic-vector arithmetic tests: α derivation, ρ ∈ [−1,+1], `(mean ρ)² ≠ mean(ρ²)`, decomposition residual, readout-only cosine sign set |
| C12 | §6 | Gates G1–G8 implemented as registration/preflight assertions with exact thresholds; VOID semantics per condition (§6 scope-of-VOID, fixes m-5) and run-level VOID for G6 | gate unit tests: each gate on passing and failing synthetic inputs; VOID-scope logic tests |
| C13 | §7 | Hypothesis evaluations H1–H7 using §5.7 metrics and §8 thresholds; H3 requires `ρ_S8 ≥ 0.50`, `ρ_S8 ≥ ρ_S4 ≥ ρ_S2`; aggregation set {4,6,14,15} | metric-level tests; boundary tests at each threshold |
| C14 | §8 | P1–P6 absolute thresholds exactly as frozen; JOINT SUCCESS = P2∧P4∧P5∧P6; PARTIAL = P1∧P3; P5 LOO-CV comparator (16 splits, mean ± SD, non-gating) per the pinned refit scope | threshold-constant tests (no observed-value leakage); LOO-CV unit test on the frozen panel structure |
| C15 | §9 | Outcome matrix (§9.1) and secondary rows (§9.2) implemented; SUCCESS cell requires P6; H4/H5/H6/H3 rows per §9.2 | verdict-logic tests for every matrix cell |
| C16 | §10.1–§10.4 | Zero-readout-params assertion at every checkpoint; protocol hash binding to the frozen v3.1 SHA before any checkpoint evaluation; verbatim data/training config (§10.3) including seed=42 split, model revision, LoRA r=16/α=32/dropout=0.05/7 target modules, AdamW lr=1e-4/wd=0.01/clip 1.0, grad_accum=2/batch=4, λ=10, T=20, decision position `v_idx[4]`, PRO_ID/AB_ID, batch-RNG rule; checkpoint set {0,4,6,7,14,15,17,20} | static conformance review of inherited config against runner lines 92–93, 73–74, 265, 982, 1006–1027, 1149–1150; assertion tests |
| C17 | §10.5 | `cos(H1)` on the mean total gradients without label-sign inversion; interpretation binding (background-dominated; not a confinement statistic); inherited sign-assertion gate (Phase 6E.18 v2.1.3 §6.2, ε = 0.001) must pass before any verdict | G6 test + synthetic cosine tests |
| C18 | §10.6 | Non-training `--preflight` mode verifying all §3.4 conventions, sign convention, κ gates, degeneracy guards, confinement residual (G7), decomposition identity, parameter counts, distinctness, sign-assertion, and §5.7/§10.5 metric arithmetic on synthetic vectors | preflight self-check battery |
| C19 | §10.7 | B0 collapse guard: K = 3 consecutive zero-proposal steps ⇒ B0 truncates; no other condition truncates; missing matched comparator ⇒ frozen step-0 anchor + `UNAVAILABLE` recorded; no back-filling | guard logic tests |
| C20 | §10.8–§10.9 | Void/deviation rules; verdict double gate + refusal triplet `PRE_EXECUTION`/`NO_SCIENTIFIC_VERDICT`/`EXECUTION_NOT_AUTHORIZED` | refusal-triplet tests (no verdict without technical evaluation AND human authorization) |
| C21 | §10.10 | Provenance artifacts: `sha256_manifest.json`, reference tensors (7 `w_<cond>`, `u_lda`, `U_8`, `R_4`, `n_rand`, `H_16`, `y16`) each with SHA-256; `runs.json`; `pre_registration_snapshot.json` (doc hash + adopted-decision log); `p5_loo_report.json`; `execution_authorization.json` schema check (decision=APPROVE, phase=6E.20; the 6E.18 file does not authorize 6E.20) | artifact-schema and manifest-consistency tests |

---

## 4. Implementation Workstreams

- **W1 — Hash & artifact layer (C2, C4, C7, C21).** Implement the legacy `tensor_hash` (float32 torch-save bytes) and the canonical v3 hash (float64 C-contiguous little-endian raw bytes); `.npy` writers; manifest binding. Reproduce the two registered legacy hashes on the frozen-registered objects as a bootstrap test.
- **W2 — Registration objects (C1–C7).** Panel/labels, û_LDA, s_scale, SVD + sign pinning, κ values, w_δ, random objects; determinism and reconstruction tests.
- **W3 — Condition construction (C8–C10).** Seven readouts, biases, parameter-count guards, distinctness preflight checks.
- **W4 — Gradient & metric layer (C11, C17).** Dual gradient objects, α coefficients, ρ_c/σ_c, cos(H1), readout-only cosine, decomposition identity; synthetic-vector test battery.
- **W5 — Gate and preflight engine (C12, C18).** G1–G8 assertions, VOID semantics, `--preflight` orchestration.
- **W6 — Evaluation and verdict layer (C13–C15, C19–C20).** P1–P6, outcome matrix, H1–H7, LOO-CV comparator, collapse guard, refusal triplet.
- **W7 — Provenance & authorization (C21).** Manifest, snapshot, runs schema, authorization-file schema validation.
- **W8 — Conformance harness.** Automated execution of the C1–C21 verification battery on synthetic/registered data; report mapping each conformance row to PASS/FAIL with evidence.

---

## 5. Pre-Execution Conformance Gates (prerequisites to any execution authorization request)

1. **G-PLAN-1:** Every C1–C21 row implemented and verified by the W8 harness with a PASS disposition or an explicit, documented non-blocking deviation.
2. **G-PLAN-2:** `--preflight` dry-run passes on synthetic/registered data (no model forward/backward on the real panel, no training).
3. **G-PLAN-3:** Frozen v3.1 SHA-256 rebinding still matches `E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85`; no frozen file modified.
4. **G-PLAN-4:** Formal pre-execution audit of the implementation against this plan and v3.1, and explicit signed human authorization (`execution_authorization.json`, decision=APPROVE, phase=6E.20) — v3.1 §11.
5. **G-PLAN-5:** Refusal triplet is the only verdict emitted until all of the above hold.

---

## 6. Explicit Non-Goals and Prohibited Actions

This plan authorizes no model execution, training, fine-tuning, optimizer steps, checkpoint generation, or scientific verdicts. It also prohibits modification of v3.0, frozen v3.1, Phase 6E.18 execution logs, the Phase 6E.19 adjudication record, and the Phase 6E.18 artifacts. All verification in this plan is limited to deterministic self-tests on synthetic vectors and registered/historical objects (hashing, reconstruction, arithmetic), which involve no model forward/backward on the real panel and no training state.

---

## 7. Deferred Non-Blocking Items (recorded; carried from v3.1 §13)

These were explicitly deferred in v3.1 §13 and are re-recorded here for implementation awareness. None blocks the design review or affects the SUCCESS gate or metric validity; all are candidate MINOR cleanups for a future revision:

- **V3-07:** multi-condition gate VOID mapping for G4/G7 (condition-level vs run-level VOID semantics for gates applied to multiple conditions).
- **V3-08:** no named gate for the `n_rand` cosine bound (currently a construction constraint in §3.7 rather than a numbered gate).
- **V3-13:** `y16` numeric encoding note — the ±1 encoding is coef-direction-equivalent to the 6E.18 `[1]^8 ⊕ [0]^8` encoding under unit normalization; noted in v3.1 §8 P5.
- **V3-14:** provenance phrase in §3.4/§10.6 ("sign-unpinned SVD output") to be tightened.
- **V3-15:** "fully enumerated" overstatement in §10.3 header to be revisited.
- **V3-16:** reference-tensor hashes not listed in `pre_registration_snapshot.json` (schema note).
- **V3-17/V3-18 (INFO):** informational notes carried forward.

---

## 8. Deliverables and Acceptance for Implementation Completion

1. Implementation source (new Phase 6E.20 runner), a separate new artifact from the 6E.18 runner, with no modification to 6E.18.
2. The W8 conformance report mapping C1–C21 to PASS/FAIL with evidence.
3. `--preflight` implementation matching §10.6.
4. Updated provenance templates (`sha256_manifest.json`, `pre_registration_snapshot.json`, `execution_authorization.json` schema) conforming to §10.10.
5. An implementation-completion note referencing this plan and the frozen v3.1 SHA.

Completion of the plan does **not** authorize execution. Execution requires G-PLAN-4 (audit + signed human authorization) and remains HARD-STOPPED until then.

---

## 9. Hard Stop

```
===============================================================================
HARD STOP — DESIGN APPROVED FOR IMPLEMENTATION PLANNING ONLY — NO EXECUTION
===============================================================================
This plan is a PLANNING artifact. It does NOT authorize any model execution,
training, fine-tuning, optimizer steps, checkpoint generation, or scientific
verdict for Phase 6E.20.

Execution remains conditional on:
  (1) conformance of the implementation to this plan and to frozen v3.1
      (SHA-256 E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85),
  (2) formal pre-execution audit, and
  (3) explicit signed human authorization
      (execution_authorization.json, decision=APPROVE, phase=6E.20).
===============================================================================
```
