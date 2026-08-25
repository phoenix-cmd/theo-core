# THEO SLM Fixed Subspace Readout Authorization Protocol — v4.0 Execution Extension Plan

- **Status:** IMPLEMENTATION AUTHORIZED (L1–L3). **EXECUTION NOT AUTHORIZED.**
- **Supersedes:** Plan v3.2.1 (`theo-slm-fixed-subspace-readout-authorization-protocol-v3.2.1-pre-execution-implementation-plan.md`, SHA-256 `F6A145DDA276621421A13E22215CBF5F9DE3052C712CD8CD4055076C3D3679F1`, 189 lines, preserved unmodified).
- **Governing protocol (frozen, unchanged):** `theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md`, SHA-256 `E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85`, 400 lines.
- **Phase:** 6E.20 (Fixed Low-Dimensional & Subspace Readout Architectures) — Execution Extension (6E.20.1).
- **Authority status:** Implementation of the new execution runner (`run_phase_6e20_execution.py`) and its conformance harness is authorized, limited to L1–L3. Model execution, training, fine-tuning, optimizer steps, checkpoint generation, real-panel (H16) collection, and scientific verdicts are **not** authorized by this plan. Execution requires a NEW L5 human authorization (`execution_authorization.json`, decision=APPROVE, phase=6E.20.1). The existing 6E.20 authorization file MUST NOT be reused.

---

## 1. Purpose and Scope

Plan v4.0 extends the pre-execution implementation program to cover the execution stages that are currently stubs in the frozen Phase 6E.20 runner. It defines: (a) the new execution runner that imports W1–W7 machinery from the frozen runner and implements real step-0 registration, training, evaluation, and provenance; (b) a new conformance harness; (c) the acceptance ladder for the new runner. It does not amend the frozen protocol v3.1 or the frozen runner.

In scope: real H16 collection from the frozen base model, real step-0 registration (LDA, SVD, kappas, w_delta, s_scale, R4, n_rand), 7-condition training with LoRA (T=20), evaluation (H1–H7, P1–P6, LOO-CV P5), provenance emission, authorization gating.

Out of scope: modification of v3.1, frozen runner, frozen harness, frozen plan v3.2.1, frozen 6E.20 authorization file, or Phase 6E.18 artifacts.

---

## 2. Governing Inputs (frozen, read-only)

| Input | Role |
|---|---|
| v3.1 protocol (SHA above) | Normative design |
| Frozen Plan v3.2.1 (SHA above) | Pre-execution plan (preserved) |
| Frozen Phase 6E.20 runner (SHA `550FAA34…`) | W1–W7 machinery source (read-only import) |
| Frozen Phase 6E.20 harness (SHA `8239C2A1…`) | W8 conformance reference |
| Frozen 6E.20 authorization file (SHA `4CFABC2F…`) | Historical record ONLY; MUST NOT be reused |
| Phase 6E.18 runner | Inherited constants (read-only) |

---

## 3. What's New (Execution Stages)

### 3.1 Step-0 Registration (v3.1 §10.6 mode b, real)

Load the frozen base model (`Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`) via HuggingFace `transformers`. Collect the real H16 panel (layer-24 last-token hidden states, rows 0:8 POS / 8:16 ABS). Compute û_LDA, s_scale, U_8 (SVD + sign pinning), kappas, w_delta, R_4, n_rand. Verify G1/G2 dual-convention reconstruction against registered legacy hashes. Persist artifacts (H_16.npy, u_lda.npy, U_8.npy, R_4.npy, n_rand.npy, w_*.npy ×7, y16).

### 3.2 Training (v3.1 §10.3–§10.4)

7 conditions (B0, F1, S2, S4, S8, C4-Rand, R1-Rand). LoRA r=16, α=32, dropout=0.05, 7 target modules. AdamW lr=1e-4, wd=0.01, grad_accum=2, batch=4, clip=1.0. λ=10 at decision position v_idx[4]. T=20. Checkpoints at {0, 4, 6, 7, 14, 15, 17, 20}. B0 collapse guard (K=3). Zero readout parameters per condition.

### 3.3 Evaluation (v3.1 §10.5, §7–§9)

For each checkpoint: balanced accuracy, cos(H1), ρ_c, σ_c. Criteria P1–P6 evaluation. Verdicts per condition (SUCCESS/PARTIAL/FAIL/etc). Hypotheses H1–H7. LOO-CV P5 (16 splits). Outcome matrix.

### 3.4 Provenance (v3.1 §10.10)

`sha256_manifest.json`, `runs.json` (step-by-step telemetry), `pre_registration_snapshot.json`, `p5_loo_report.json`, reference tensors, code hash.

---

## 4. Conformance Model (New Rows)

| # | Requirement | Verification |
|---|---|---|
| C22 | Step-0 registration: real model load, H16 collection, G1/G2 reconstruction, artifacts persisted | L5 registration preflight assertions |
| C23 | Training: 7 conditions × T=20, LoRA config, checkpoint set, zero readout params, collapse guard | L5 checkpoint assertions; synthetic-path training logic test |
| C24 | Evaluation: H1–H7, P1–P6, verdicts, LOO-CV P5, outcome matrix | L5 evaluation audit; synthetic-path metric tests |
| C25 | Provenance: manifest, runs.json, snapshot, p5_loo_report, reference tensors | Schema and consistency tests |
| C26 | Authorization: refuse without new L5 authorization (phase=6E.20.1); 6E.20 file not reusable | Authorization gating test |

---

## 5. Acceptance Ladder

| Stage | Requirement |
|---|---|
| **L1** | New runner + harness compile and import; config constants match frozen protocol §10.3 |
| **L2** | Unit tests on synthetic vectors (training logic, evaluation logic, provenance schemas, authorization gating) |
| **L3** | Full conformance battery (C1–C26) on synthetic/registered data; raw evidence archived |
| **L4** | Independent review; independently reproducible rerun/subset |
| **L5** | NEW human authorization (`execution_authorization.json`, decision=APPROVE, phase=6E.20.1) |

---

## 6. Authorization Non-Reuse

The frozen 6E.20 authorization file (`execution_authorization.json` in `phase-6e20-fixed-subspace-readout/`, SHA `4CFABC2F…`, phase=6E.20) is a historical record of the pre-execution L5 gate. It MUST NOT be used as authorization for the new execution runner. The new runner requires a distinct authorization file with `phase=6E.20.1` in a new artifact directory.

---

## 7. Hard Stop

```
===============================================================================
HARD STOP — IMPLEMENTATION AUTHORIZED, EXECUTION NOT AUTHORIZED
===============================================================================
Plan v4.0 authorizes IMPLEMENTATION of the execution extension runner
(run_phase_6e20_execution.py) and its conformance harness, limited to
acceptance-ladder stages L1-L3. It does NOT authorize any model execution,
training, fine-tuning, optimizer steps, checkpoint generation, real-panel
(H16) collection, or scientific verdicts.

Execution remains conditional on:
  (1) L1-L3 conformance of the implementation to this plan and to frozen v3.1
      (SHA-256 E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85),
  (2) the L4 independent-review gate, and
  (3) explicit signed human authorization
      (execution_authorization.json, decision=APPROVE, phase=6E.20.1).

The existing 6E.20 authorization file (phase=6E.20) MUST NOT be reused.
===============================================================================
```
