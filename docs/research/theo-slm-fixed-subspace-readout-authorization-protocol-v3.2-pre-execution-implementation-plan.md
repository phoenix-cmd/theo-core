# THEO SLM Fixed Subspace Readout Authorization Protocol — v3.1 Pre-Execution Implementation & Conformance Plan (Plan v3.2)

- **Status:** PLANNING ONLY — DESIGN APPROVED FOR IMPLEMENTATION PLANNING ONLY. **NO EXECUTION.**
- **Supersedes:** `theo-slm-fixed-subspace-readout-authorization-protocol-v3.1-pre-execution-implementation-plan.md` (Plan v3.1, SHA-256 `2A45DB113710BD102A2FBAB6E98885A3BDC9CDD0A7301896298154C67F19F0EA`, 136 lines, preserved unmodified). Plan v3.1 was audited with verdict **AMEND AND RESUBMIT**: findings F-01 (BLOCKING), F-02…F-05 (MAJOR), F-06…F-11 (MINOR). Plan v3.2 materializes all amendments; the full mapping is in §9.
- **Governing protocol (frozen, unchanged):** `theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md`, SHA-256 `E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85`, 400 lines. This plan does **not** amend the frozen protocol; any future protocol amendment requires a documented amendment and re-adoption (§12, §13 of v3.1). The plan's version number (v3.2) is independent of the governing protocol version (v3.1).
- **Phase:** 6E.20 (Fixed Low-Dimensional & Subspace Readout Architectures).
- **Superseded governing draft:** v3.0, SHA-256 `F6F1DC23D2D4678F4C6F9A7FE09F59C37B21227DF7EA50E053BC53123A8D00B1` (audited, read-only, unmodified).
- **Authority status:** No model execution, training, fine-tuning, optimizer steps, checkpoint generation, or scientific verdicts are authorized by this plan. Execution remains conditional on: (1) implementation conformance to this plan, (2) formal pre-execution audit, and (3) explicit signed human authorization (`execution_authorization.json`, decision=APPROVE, phase=6E.20) — §11 of v3.1.

---

## 1. Purpose and Scope

This plan converts the frozen v3.1 design into an ordered implementation and conformance program. It defines, per v3.1 section, the concrete objects, routines, assertions, and artifacts an implementation must produce, how conformance will be verified, and the non-skippable acceptance ladder that separates implementation completion from execution authorization. It is a **planning document only**: it specifies work to be done later, it executes nothing, and it changes no frozen file.

In scope: everything required to implement and verify Phase 6E.20 per frozen v3.1 (§3–§10), including the two-mode preflight (§10.6 as split in C18), the metric layer (§5.7/§10.5), the gates (§6), the evaluation and verdict logic (§7–§9), and the provenance/authorization artifacts (§10.10).

Out of scope (prohibited at this stage): any runner execution, training, optimizer steps, checkpoint generation, scientific verdicts, collection of the real step-0 panel, and any modification of v3.0/v3.1 or of Phase 6E.18 execution logs and the Phase 6E.19 adjudication record (v3.1 §10.8).

---

## 2. Governing Inputs (frozen, read-only)

| Input | Role | Source |
|---|---|---|
| v3.1 protocol (frozen SHA above) | normative design | `docs/research/theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md` |
| v3.0 protocol (SHA above) | superseded, audited, unmodified | `docs/research/theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.md` |
| Phase 6E.18 runner | source of inherited operational rules and of the legacy hash / metric computations | `scripts/dataset_generator/run_phase_6e18_causal_intervention.py` |
| Phase 6E.18 v2.1.3 protocol (filename corrected, fixes F-05) | source of inherited §6.2/§7.1/§10 rules cited by v3.1 | `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.3.md` |
| Phase 6E.18 artifacts | registered historical hashes/values (read-only) | `readout_initialization.json`, `t2_basis.json`, `pre_registration_snapshot.json` (paths resolved relative to the 6E.18 `ARTIFACT_DIR`; see F-13) |
| Phase 6E.19 forensic record | informational context only; no value enters any threshold | `theo-slm-phase-6e18-post-execution-forensic-analysis-v1.md` |

Note (provenance fact, supports F-04): Phase 6E.18 computed the step-0 panel `H16` **in memory only** (runner lines 262–272) and persisted **only derived hashes/values** (`readout_initialization.json`). There is no persistent 6E.18 `H16` tensor artifact to reuse.

---

## 3. Conformance Model (v3.1 § → Implementation Requirement → Verification)

Each conformance row names the v3.1 section, the required implementation deliverable, and the verification method. Verification is by code review, static analysis, deterministic unit/reconstruction tests, and `--preflight` self-checks, per the acceptance ladder (§5.1). Per F-02/F-06, the W8 harness archives **raw evidence** and conformance is confirmed by an **independent reviewer** via independent rerun or independently re-run subset verification; the implementation does not self-attest its own conformance. All L1–L4 verification is limited to dry self-tests on synthetic vectors and registered/historical objects; none runs the model, a training step, or an optimizer step, and none performs a real-panel forward pass.

| # | v3.1 § | Implementation requirement | Conformance verification |
|---|---|---|---|
| C1 | §3.1 | Frozen step-0 panel `H_16 ∈ R^16×896` (layer-24 last-token hidden states, rows 0:8 POS / 8:16 ABS), **collected ONLY at authorized execution (L5, step 0)** from the frozen base model at the pinned revision (§10.3, see C16). NOT treated as a pre-existing persistent 6E.18 tensor artifact (6E.18 never persisted it; runner lines 262–272). Stored as `H_16.npy` with canonical-convention panel hash in the manifest; pinned label vector `y16 = [+1]^8 ⊕ [-1]^8`; **margin decision rule `s > 0 ⇒ POS, s ≤ 0 ⇒ ABS` (§3.1; fixes F-11)**. At L1–L4, all H16-dependent checks use only the registered legacy hashes and deterministic reconstruction/hashing on synthetic/registered data — no real-panel forward pass (fixes F-04) | L2/L3 determinism and hashing-function tests on synthetic data; L5 step-0 registration preflight verifies G1/G2 dual-convention reconstruction against stored `H_16.npy`/`u_lda.npy`/`U_8.npy` and registered hashes before any training step (fixes F-04, F-09) |
| C2 | §3.2 | `û_LDA` computation (sklearn LDA, `solver="lsqr"`, `shrinkage="auto"`), unit norm; legacy hash must equal registered `u_hat_LDA_hash == ff3cab49…`; canonical hash `u_lda_tensor_hash`; artifact `u_lda.npy` | G1 test: legacy identity + canonical artifact hash + reconstruction (cosine ≥ 0.9999, max-abs ≤ 1e-6); legacy identity verified at L5 step-0 registration from the collected H16 (F-04) |
| C3 | §3.3 | `s_scale` per formula with ε = 1e-12; reference `0.06681545` matched at step-0 registration; degenerate guard (non-finite or ≤ 0 ⇒ registration fails ⇒ run VOID, G8) | preflight assertion test |
| C4 | §3.4 | Covariance + SVD exactly as 6E.18 §8; canonical sign convention (multiply column by −1 if largest-magnitude element < 0, ties by lowest row index); two hash conventions implemented and kept distinct; legacy `basis_sha256` of sign-unpinned `U_8^raw` must equal `b8dfbd6a…`; nested `U_2⊂U_4⊂U_8`; artifact `U_8.npy` + `U_8_tensor_hash`; cross-convention equality never asserted | G2 test; nesting test; sign-convention unit tests; **diagnostic (not a gate; fixes F-10):** legacy vs canonical hashes recorded and reported as distinct serializations per §3.4 — v3.1 asserts only that equality is never required or asserted, so no inequality gate is imposed |
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
| C16 | §10.1–§10.4 | Zero-readout-params assertion at every checkpoint; protocol hash binding to the frozen v3.1 SHA before any checkpoint evaluation; verbatim data/training config (§10.3) including seed=42 split, **model `Qwen/Qwen2.5-0.5B-Instruct`, `d = 896`, revision `7ae557604adf67be50417f59c2c2f167def9a775` (fixes F-08)**, LoRA r=16/α=32/dropout=0.05/7 target modules, AdamW lr=1e-4/wd=0.01/clip 1.0, grad_accum=2/batch=4, λ=10, T=20, **decision-prompt convention `format_prompt(...) + '{"decision": "'` with decision logits at `v_idx[4]` (fixes F-08)**, PRO_ID 9117 / AB_ID 1867, batch-RNG rule; checkpoint set {0,4,6,7,14,15,17,20} | static conformance review of inherited config against runner lines 92–93, 73–74, 265, 982, 1006–1027, 1149–1150; assertion tests |
| C17 | §10.5 | `cos(H1)` on the mean total gradients without label-sign inversion; interpretation binding (background-dominated; not a confinement statistic); inherited sign-assertion gate (Phase 6E.18 v2.1.3 §6.2, ε = 0.001) must pass before any verdict | G6 test + synthetic cosine tests |
| C18 | §10.6 | Non-training `--preflight` in **two modes (fixes F-04, F-09):** (a) **L4 pre-execution preflight** on synthetic/registered data only — no real-panel forward pass — verifying sign convention, κ gates, degeneracy guards, confinement residual (G7) and decomposition identity on synthetic vectors, parameter-count logic, distinctness, sign-assertion (G6), metric arithmetic (§5.7/§10.5), hash functions, and config assertions; (b) **L5 step-0 registration preflight** at authorized-execution start, immediately after H16 collection and before any training/optimizer step, verifying **G1/G2 dual-convention reconstruction (legacy identity + canonical artifact hashes + stored-tensor reconstruction)** per §10.6 | L4 preflight self-check battery (recorded outputs); L5 registration-preflight assertions |
| C19 | §10.7 | B0 collapse guard: K = 3 consecutive zero-proposal steps ⇒ B0 truncates; no other condition truncates; missing matched comparator ⇒ frozen step-0 anchor + `UNAVAILABLE` recorded; no back-filling | guard logic tests |
| C20 | §10.8–§10.9 | Void/deviation rules; verdict double gate + refusal triplet `PRE_EXECUTION`/`NO_SCIENTIFIC_VERDICT`/`EXECUTION_NOT_AUTHORIZED` | refusal-triplet tests (no verdict without technical evaluation AND human authorization) |
| C21 | §10.10 | Provenance artifacts: `sha256_manifest.json` (code, base model, corpus, checkpoints, all §3 objects; canonical v3 hashes for new artifacts, legacy hashes for historical identity); reference tensors (`u_lda.npy`, `U_8.npy`, `R_4.npy`, `n_rand.npy`, `w_<cond>.npy` ×7, `H_16.npy`, `y16`) each with SHA-256; **`runs.json` (step-by-step telemetry per §10.10: per-checkpoint metrics, gate dispositions, telemetry; fixes F-11)**; `pre_registration_snapshot.json` (doc hash + adopted-decision log); `p5_loo_report.json`; `execution_authorization.json` schema check (decision=APPROVE, phase=6E.20; the 6E.18 file does not authorize 6E.20) | artifact-schema and manifest-consistency tests |

---

## 4. Implementation Workstreams

- **W1 — Hash & artifact layer (C2, C4, C7, C21).** Implement the legacy `tensor_hash` (float32 torch-save bytes) and the canonical v3 hash (float64 C-contiguous little-endian raw bytes); `.npy` writers; manifest binding. Verify both hashing functions against known-good synthetic vectors at L2/L3. The **registered legacy-hash value reproduction** (G1/G2 legacy identity: `u_hat_LDA_hash == ff3cab49…`, `basis_sha256 == b8dfbd6a…`) is verified at **L5 step-0 registration from the collected H16** — not at implementation time, because H16 is not a pre-existing artifact (fixes F-04).
- **W2 — Registration objects (C1–C7).** Panel/labels, û_LDA, s_scale, SVD + sign pinning, κ values, w_δ, random objects; determinism and reconstruction tests. Panel collection is L5-scoped (C1).
- **W3 — Condition construction (C8–C10).** Seven readouts, biases, parameter-count guards, distinctness preflight checks.
- **W4 — Gradient & metric layer (C11, C17).** Dual gradient objects, α coefficients, ρ_c/σ_c, cos(H1), readout-only cosine, decomposition identity; synthetic-vector test battery.
- **W5 — Gate and preflight engine (C12, C18).** G1–G8 assertions, VOID semantics, two-mode `--preflight` orchestration (L4 pre-execution and L5 registration).
- **W6 — Evaluation and verdict layer (C13–C15, C19–C20).** P1–P6, outcome matrix, H1–H7, LOO-CV comparator, collapse guard, refusal triplet.
- **W7 — Provenance & authorization (C21).** Manifest, snapshot, runs telemetry schema, authorization-file schema validation.
- **W8 — Conformance harness (fixes F-02).** Automated execution of the C1–C21 verification battery on synthetic/registered data at L3, emitting a per-row PASS/FAIL report **with raw archived evidence** (logs, tensor comparisons, assertion outputs) to a conformance-evidence directory. Results are subject to **independent review and independently reproducible rerun/subset verification** (§5.1 L4).

---

## 5. Pre-Execution Conformance Gates and Acceptance Ladder

### 5.1 Five-stage non-skippable acceptance ladder (fixes F-03)

| Stage | Requirement | Evidence artifact | Unreachability rule |
|---|---|---|---|
| **L1 — Compile/import** | New Phase 6E.20 runner package imports and initializes without error; static conformance review of inherited config (C16) against the frozen runner | build/import log; config review record | entry stage |
| **L2 — Unit tests** | Deterministic unit/reconstruction tests on synthetic vectors and registered/historical objects (hashing functions, metric arithmetic, gate thresholds, sign convention, decomposition identity) | unit-test report | unreachable unless L1 PASS |
| **L3 — Protocol conformance** | C1–C21 battery run by W8; every **required** row PASS; raw evidence archived; no waiver (G-PLAN-1) | conformance report + evidence archive | unreachable unless L1 ∧ L2 PASS |
| **L4 — Independent preflight** | Independent reviewer re-runs or independently re-verifies a defined subset of the W8 battery **and** the L4 pre-execution `--preflight` dry-run on synthetic/registered data; outputs recorded; no real-panel forward pass (G-PLAN-2) | independent review record; preflight log | unreachable unless L1 ∧ L2 ∧ L3 PASS |
| **L5 — Human authorization** | Formal pre-execution audit (auditor independent of the implementer) + explicit signed human authorization (`execution_authorization.json`, decision=APPROVE, phase=6E.20) | audit report; authorization file | unreachable unless L1 ∧ L2 ∧ L3 ∧ L4 PASS |

Failure of any stage blocks all later stages; no skipping, no retroactive waivers, no stage reordered. Stage transitions are recorded in the conformance log with the evidence artifact attached. Execution eligibility begins only upon L5 PASS. The refusal triplet (`PRE_EXECUTION`/`NO_SCIENTIFIC_VERDICT`/`EXECUTION_NOT_AUTHORIZED`) is the only verdict emitted until L5 PASS.

### 5.2 Conformance gates

1. **G-PLAN-1 (L3) — Protocol conformance, no waiver (fixes F-01):** Every C1–C21 row implemented and verified by the W8 harness with a **PASS** disposition and raw archived evidence. **A failed required conformance check does NOT satisfy the gate.** Exceptions are limited **exclusively** to the seven pre-registered deferred items (§7), each exception requiring: (a) documented human review, (b) an explicit recorded disposition, and (c) a **formal v3.1 protocol amendment (§13) and re-adoption BEFORE execution eligibility**. No deferred item may relax, weaken, or remove a C row that implements a gate (G1–G8), a SUCCESS/PARTIAL criterion (P1–P6, §8), a hypothesis evaluation (H1–H7, §7), a metric definition (§5.7/§10.5), or a VOID/refusal rule (§10.8–§10.9).
2. **G-PLAN-2 (L4) — Independent preflight (fixes F-04, F-06):** L4 pre-execution `--preflight` dry-run passes on synthetic/registered data only (no model forward/backward on the real panel, no training). Outputs are recorded; PASS requires **independent observation or an independent rerun** in addition to the implementer's self-run.
3. **G-PLAN-3 — Frozen-artifact integrity:** Frozen v3.1 SHA-256 rebinding still matches `E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85`; no frozen file modified. Re-verified at every stage boundary.
4. **G-PLAN-4 (L5) — Independent audit + human authorization (fixes F-02, F-06):** Formal pre-execution audit performed by a reviewer **independent of the implementer**, with access to the evidence archive and authority to re-run the W8 battery or an independently re-run subset, plus explicit signed human authorization (`execution_authorization.json`, decision=APPROVE, phase=6E.20) — v3.1 §11.
5. **G-PLAN-5 — Refusal:** The refusal triplet is the only verdict emitted until all of the above hold.

---

## 6. Explicit Non-Goals and Prohibited Actions

This plan authorizes no model execution, training, fine-tuning, optimizer steps, checkpoint generation, or scientific verdicts. It also prohibits modification of v3.0, frozen v3.1, Phase 6E.18 execution logs, the Phase 6E.19 adjudication record, and the Phase 6E.18 artifacts.

**H16 provenance boundary (fixes F-04):** the step-0 panel `H_16` is **not** a pre-existing persistent 6E.18 tensor artifact (verified: 6E.18 runner computes it in memory at lines 262–272 and persists only derived hashes). H16 collection is an **authorized-execution-stage (L5) activity** using the frozen base model at the pinned revision (C16); it is never performed at L1–L4. Preflight claims involving H16 use only the registered legacy hashes and deterministic reconstruction/hashing on synthetic/registered data, and must not perform an unauthorized execution-stage forward pass.

All L1–L4 verification is limited to deterministic self-tests on synthetic vectors and registered/historical objects (hashing, reconstruction, arithmetic, gate and verdict logic), which involve no model forward/backward on the real panel and no training state. Collection of the real panel, and any step-0 registration preflight against it, occur only at L5 under signed authorization.

---

## 7. Deferred Non-Blocking Items (recorded; carried from v3.1 §13)

These were explicitly deferred in v3.1 §13 and are re-recorded here. **Binding non-assumption clause (fixes F-07):** no deferred item may be relied upon, assumed, or silently implemented at execution time; any reliance requires a documented v3.1 amendment and re-adoption (§13). Deferred items never relax or override any C1–C21 requirement, and each may be raised as an exception under G-PLAN-1 only subject to (a)–(c) of that gate.

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
2. The W8 conformance report mapping C1–C21 to PASS/FAIL **with raw archived evidence** (fixes F-02) and the conformance log recording stage transitions (fixes F-03).
3. Two-mode `--preflight` implementation matching §10.6 as split in C18.
4. The independent review record for L4 and the L5 audit report (fixes F-02, F-06).
5. Updated provenance templates (`sha256_manifest.json`, `pre_registration_snapshot.json`, `runs.json` telemetry schema, `execution_authorization.json` schema) conforming to §10.10 (fixes F-11).
6. An implementation-completion note referencing this plan (v3.2) and the frozen v3.1 SHA.

Completion of the plan does **not** authorize execution. Execution requires the L5 gate (G-PLAN-4: independent audit + signed human authorization) and remains HARD-STOPPED until then.

---

## 9. Amendment Log (Plan v3.1 → Plan v3.2)

| Finding | Severity | Amendment | Location |
|---|---|---|---|
| F-01 | BLOCKING | Open-ended "non-blocking deviation" waiver removed; failed required conformance check never satisfies G-PLAN-1; exceptions limited exclusively to pre-registered deferred items with documented human review + formal v3.1 amendment before execution eligibility | §5.2 G-PLAN-1 |
| F-02 | MAJOR | Raw evidence archive; independent reviewer; independently reproducible rerun/subset verification of W8 | §3 intro, W8, §5.1 L4, G-PLAN-4, §8 |
| F-03 | MAJOR | Strict five-stage non-skippable acceptance ladder L1–L5 with unreachability rules | §5.1, G-PLAN-5, §8 |
| F-04 | MAJOR | H16 provenance pinned: L5-only collection from frozen revision; 6E.18 non-persistence fact stated; preflight uses only reconstruction/hashes, no real-panel forward pass; two-mode preflight | C1, C2, W1, W2, C18, §6, G-PLAN-2 |
| F-05 | MAJOR | Governing v2.1.3 protocol filename corrected to `theo-slm-causal-intervention-authorization-protocol-draft-v2.1.3.md` | §2 |
| F-06 | MINOR | Independent observation/rerun required for G-PLAN-2; G-PLAN-4 auditor independence mandated | G-PLAN-2, G-PLAN-4, §5.1 L4, §8 |
| F-07 | MINOR | Binding non-assumption clause added for all deferred items | §7 |
| F-08 | MINOR | C16 expanded: exact model revision `7ae557604adf67be50417f59c2c2f167def9a775`, `d = 896`, decision-prompt convention | C16 |
| F-09 | MINOR | C18 explicitly references G1/G2 dual-convention reconstruction; L5 registration preflight | C18, C1 |
| F-10 | MINOR | C4 legacy-vs-canonical inequality check downgraded to a diagnostic, not a gate | C4 |
| F-11 | MINOR | C1 margin decision rule; C21 `runs.json` step-by-step telemetry content | C1, C21 |

---

## 10. Hard Stop

```
===============================================================================
HARD STOP — DESIGN APPROVED FOR IMPLEMENTATION PLANNING ONLY — NO EXECUTION
===============================================================================
This plan (v3.2) is a PLANNING artifact. It does NOT authorize any model
execution, training, fine-tuning, optimizer steps, checkpoint generation,
real-panel collection, or scientific verdict for Phase 6E.20.

Execution remains conditional on:
  (1) conformance of the implementation to this plan and to frozen v3.1
      (SHA-256 E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85),
      established by the five-stage acceptance ladder (L1–L5, §5.1), and
  (2) explicit signed human authorization
      (execution_authorization.json, decision=APPROVE, phase=6E.20).
===============================================================================
```
