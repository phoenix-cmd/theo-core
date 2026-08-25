# FORENSIC 6E.20.1 FINAL EXECUTION REPORT

**Phase:** 6E.20.1 — Fixed-Subspace Readout Execution Extension (v7)
**Date of execution / closure:** 2026-08-18
**Status:** CLOSED — frozen and immutable
**Report type:** Final forensic completion record (no further experiment authorized)

---

## 1. Executive Verdict

| Dimension | Result |
|---|---|
| Procedural execution | **COMPLETE / VALID** |
| Forensic preservation | **INTACT** (all artifacts, evidence, and auth chain hash-bound) |
| VOID determination | Dynamic per §6 G3 performed by the run → `void_conditions = []` (**not void**) |
| Scientific result | **NEGATIVE** (cos(H1) ≈ 0.39–0.46; required ≥ 0.80) |
| Overall decision | **No automatic follow-up.** Phase 6E.20.1 is closed. Any causal investigation requires a new phase and new authorization. |

**One-line summary:** The experiment was executed correctly and completely under the frozen protocol; the intervention did **not** produce the required effect. This is a valid negative result, not an execution failure.

---

## 2. Immutable Identity Table

All hashes are SHA-256, uppercase, verified at closure (2026-08-18).

| Artifact | SHA-256 |
|---|---|
| Protocol v3.1 (frozen) | `E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85` |
| Plan v4.0 (frozen) | `87C8E486572F0B9D42EE49E21B2E93A56AD1C7CB755DD9EFAE5A0F25F463304A` |
| 6E.18 causal-intervention snapshot (frozen, F7 g_total machinery) | `6BCAF1319AC551848FFDB55AEA121D14D808408A192E05F2AA6171161DECEF96` |
| Frozen runner `run_phase_6e20_fixed_subspace_readout.py` | `550FAA345721649225F32748C39053077B08BE165426EF3FE38C2F6E276A1BD9` |
| v6 execution runner (frozen evidence of disk-full failure) | `1F5F7EF2894CFC5B6B76D99E0EB57C4DBE9A00700E80B8197784FDF4C45FB15C` |
| **Final v7 execution runner** | **`C8E298C85294B8CC5737D5A8EE44679C7A778EEF09187894F1FE1CF435115A8B`** |
| **Final authorization (`execution_authorization_v7.json`)** | **`5214DA997764F62C9370238E055CD5394925FBFE8F0C61478B8D192B198C4F91`** |
| Authorization bound copy (`execution_authorization_v6.json`) | `5214DA997764F62C9370238E055CD5394925FBFE8F0C61478B8D192B198C4F91` (identical content) |
| Regression harness `regression_checkpoint20_v7.py` | `321F4DECD44ABE7F8D82A9F5E1CA34CCF874EA1BA3145778A4214BB2CA28E4BB` |
| Conformance harness v5 `run_phase_6e20_execution_conformance_harness_v5.py` | `A3F8211D14E0ACF3F67B393749370E3B3E6E6147DFB6C2A196F985E9435E92A5` |

The final runner and final authorization are separately identified and hash-bound. The runner reads only `execution_authorization_v6.json` (identical content to the immutable `execution_authorization_v7.json`).

---

## 3. Authorization and Revision Chain

The original scope was a *minimal* runner revision (the `range(HORIZON)` → `range(HORIZON+1)` fix for `checkpoint_20`). Execution then surfaced several latent defects during evaluation; **each was halted, reviewed, revised, re-validated (L2 self-test 33/33 + conformance C1–C35 35/35), and explicitly re-authorized** before proceeding. This is why the runner SHA and authorization SHA changed across the recovery.

| Step | Runner SHA | Authorization SHA | Notes |
|---|---|---|---|
| v6 original (frozen evidence) | `1F5F7EF2…` | `4AA21378…` (v6 auth, **superseded — MUST NOT be reused**) | Disk-full failure; preserved as evidence |
| v7 initial range-fix (L5 sign-off) | `F535C789B0C7EECA1C1D0B1B130485D8DF7A70ED5A1E201C955CE3C7605BD312` | `AEE4CF0A68F5F509D95735D25F3459ECABD7DD08DCF2C504E3EF5A269CB26CD7` | Training executed; 56 checkpoints written; evaluation crashed |
| v7 final (CHANGE 2–5 recovery, L5 re-sign-off) | `C8E298C8…` | `5214DA99…` | Final valid run; see §6 |

**Prior (superseded) authorizations preserved as evidence:**
- `6e20_execution_evidences/execution_authorization_v6_FINAL_4AA213782144DECF02D6CD062F5102B3E37393A2C3FF7403574DE4FF2CE86779_EVIDENCE.json` (v6 auth, must not be reused)
- `6e20_execution_evidences/execution_authorization_v7_PRE_EVALFIX_F535C789_AEE4CF0A_EVIDENCE.json` (initial v7 range-fix auth, before eval-path recovery)

**v7 runner changes (all within the L5 re-signed recovery):**
- **CHANGE 1** (original): training loop `range(HORIZON+1)` so `checkpoint_20` is emitted as the final trained state (GRAD_ACCUM=2 → identical optimizer-step count and checkpoint positions).
- **CHANGE 2**: class-level runtime patch of `readout.forward_logits` → `.detach().cpu().float()` before the frozen `np.asarray(h)` in `forward_margin` (frozen readout file left unmodified; fixes CUDA/BFloat16 numpy crash on the live-model cos(H1) eval path).
- **CHANGE 3**: `evaluate-only` applies the **same** `peft.get_peft_model` LoRA wrapper as `train_condition`, so F7 cos(H1) uses LoRA-only gradients (fits the 4 GiB GPU) and `restore_checkpoint` loads the trained LoRA weights.
- **CHANGE 4**: fix `compute_d15_bal15_from_checkpoint`'s call to `evaluate_checkpoint_at_state` (was passing `H16,y16` instead of `panel_prompts`; crashed every non-step-15 checkpoint).
- **CHANGE 5**: `emit_provenance`'s `sha256_manifest` resolves registration artifacts from `out_dir/step0` (where `run_step0_registration` writes them) instead of recording them `MISSING`.

---

## 4. Execution Completeness

| Check | Result |
|---|---|
| Conditions executed | **7** — B0, F1, S2, S4, S8, C4-Rand, R1-Rand |
| Dynamic §6 G3 VOID determination | Performed by the run; `void_conditions = []` → **no condition VOID** |
| Checkpoints per condition | **8 / 8** for every condition |
| Mandated checkpoint set `{0,4,6,7,14,15,17,20}` | **Present in all 7 conditions** (including `checkpoint_20`) |
| Total checkpoints | **56 / 56** |
| Fresh step-0 registration | Performed (real H16 panel, LDA axis, basis, registration gates) |
| Provenance emitted | `runs.json`, `sha256_manifest.json`, `pre_registration_snapshot.json`, `p5_loo_report.json` — all present |
| Final evaluate-only run | PID 27560 — completed cleanly, no traceback |

Because no condition is VOID, the full frozen checkpoint set is retained for all 7 conditions, satisfying Protocol §10.4.

---

## 5. Failure / Recovery Chronology

This section distinguishes **superseded failed runs** from the **final valid run**. Earlier failures are part of the forensic history; they do **not** invalidate the final completed execution because each was halted, reviewed, revised, re-validated, and explicitly re-authorized.

| # | Run | Outcome | Disposition |
|---|---|---|---|
| 1 | v6 `execute` (PID 30716) | Disk-full (28.12 GiB evidence) | Frozen evidence `phase-6e20.1-v6-FAILED-diskfull/` |
| 2 | v6 scope-deviation halt (15.6 GiB evidence) | Halted for scope deviation | Frozen evidence `phase-6e20.1-v6-HALTED-scope-deviation/` |
| 3 | v7 `execute` (PID 7712) | **Training COMPLETE** (56 ckpts); evaluation crashed: `TypeError: can't convert cuda:0 BFloat16 tensor to numpy` in frozen `readout.forward_margin` | Superseded; training output preserved |
| 4 | v7 `evaluate-only` (PID 6840, CHANGE 4) | Evaluation + provenance completed, but manifest recorded registration artifacts `MISSING` (pre-CHANGE 5) | Superseded by final run |
| 5 | **v7 `evaluate-only` (PID 27560, CHANGE 5 — FINAL)** | **COMPLETE, clean, valid**; `evaluation_complete`, `void_conditions=[]`, all provenance with real hashes | **Final valid run — frozen** |

The training output from run #3 (56 checkpoints) was reused by runs #4 and #5 via the `evaluate-only` subcommand (no retraining), so no compute was wasted and the checkpoints are the same validated artifacts.

---

## 6. Final Scientific Results

Evaluation records (from `runs.json`): per-checkpoint cos(H1) using the protocol-mandated F7 total-decision-gradient object (imported frozen 6E.18 `per_sample_decision_gradient`, S8 readout reference condition), computed on each mandated checkpoint step after restoring the B0 checkpoint and applying the trained LoRA.

| Checkpoint step | cos(H1) | rho_c | Verdict |
|---|---|---|---|
| 0  | 0.461261 | 1.0 | PARTIAL |
| 4  | 0.428355 | 1.0 | PARTIAL |
| 6  | 0.409713 | 1.0 | PARTIAL |
| 7  | 0.402869 | 1.0 | PARTIAL |
| 14 | 0.427470 | 1.0 | PARTIAL |
| 15 | 0.417450 | 1.0 | PARTIAL |
| 17 | 0.393592 | 1.0 | PARTIAL |
| 20 | 0.391386 | 1.0 | PARTIAL |

- **Observed cos(H1) range:** ≈ **0.391 – 0.461** (mean ≈ 0.417).
- **Required threshold (Protocol P2):** **cos(H1) ≥ +0.80**.
- **Result:** **NEGATIVE** — the required separation/alignment criterion is **not met** at any checkpoint; all verdicts are `PARTIAL` (never `JOINT_SUCCESS`).
- **rho_c = 1.0** for all checkpoints (readout-only path; expected and not the F7 metric).
- **Gates:** G1 ✓, G2 ✓, G3 kappa_2/k4/k8 ✓ (within [0.05, 0.995]), G3_void_map all `False`, G4 ✓, G5 ✓ → **no VOID**, run not aborted.

---

## 7. Provenance Integrity

- `sha256_manifest.json` records real SHA-256 hashes for all registration artifacts; **`MISSING entries: NONE`**.
- Registration artifact hashes (first 24 hex):
  - `u_lda.npy`   : `b72c6c215d40a76e33e1755f…`
  - `U_8.npy`     : `7c99c57787078ae6b3aa82ac…`
  - `R_4.npy`     : `87eafb205d6ca713be314bb3…`
  - `n_rand.npy`  : `7ea2b434ce81c66ab18d153e…`
  - `H_16.npy`    : `a4593e4aad2bfab0cbbb4a22…`
  - `y16.npy`     : `d8e9b87393a6ee9c24d56c7e…`
- `manifest["protocol_sha256"]`, `manifest["plan_v40_sha256"]`, `manifest["frozen_6e18_gtotal_machinery_sha256"]` all match the immutable identity table (§2).
- `pre_registration_snapshot.json` and `p5_loo_report.json` present and schema-consistent.

---

## 8. Scientific Interpretation (no overclaiming)

**What is established:**
- Under the exact Phase 6E.20.1 protocol, frozen basis, defined conditions, LoRA setup, and tested intervention, the executed run is **procedurally valid and complete**, and it **fails to satisfy the predefined success criterion** (cos(H1) ≥ 0.80).
- The negative result is **reproducible from the frozen artifacts**: the 56 checkpoints, the final runner, and the final authorization are all hash-bound and preserved.

**What is NOT established (open questions, not conclusions):**
- This result does **not** prove the intervention *definitively cannot work*. It proves only that, under this specific configuration, the required magnitude was not observed.
- Possible (untested) reasons for the ~0.4 ceiling: the intervention may be too weak; the chosen readout geometry may not capture the separation mechanism; the effect may exist but saturate near ~0.4 rather than ~0.8; the conditions may not induce sufficiently different decision-gradient directions; LoRA's accessible subspace may constrain the intended movement; or the threshold may be intentionally stringent (which is legitimate, but means `PARTIAL` is still informative.
- The evaluation harness computes F7 cos(H1) on the B0-restored checkpoints using the S8 readout as the reference condition (per Protocol §10.5). Per-condition cos(H1) beyond this reference configuration is not separately recorded in `runs.json`; this is a scope limitation of the current runner, not a defect in the final result.

**Bottom line:** A clean experiment returned a negative result. That is scientifically valuable and is preferable to a positive result from a procedurally broken run.

---

## 9. Final Closure Statement

**Phase 6E.20.1 is CLOSED.**

- The final v7 runner (`C8E298C8…`), final authorization (`5214DA99…`), 56 checkpoints, 4 provenance files, manifests, authorization chain, evaluation logs, the v6 failures, and prior v7 authorizations are **frozen and immutable**.
- **No further experiment is authorized.** Any causal investigation of the negative cos(H1) result requires a **new phase** and **new explicit authorization**; it must not be launched by modifying or re-running the frozen 6E.20.1 artifacts.
- Do not modify: the final runner, the final output directory, the 56 checkpoints, the provenance files, the manifests, the authorization chain, the evaluation logs, the v6 failures, or the prior v7 authorizations.

---

## 10. Closure Decision (Human Authorization, 2026-08-18)

**Status label (authoritative):**

> **6E.20.1 = PROCEDURALLY VALID · SCIENTIFICALLY NEGATIVE · CLOSED**

**Human decision:**
- **No new execution or intervention is authorized.** Do not launch a follow-up experiment by modifying or re-running the frozen 6E.20.1 artifacts.
- The next phase of work is **causal analysis only** — reasoning about *why* the executed run produced cos(H1) ≈ 0.39–0.46 (rather than ≥ 0.80). Causal analysis is permitted to read and reason over the frozen artifacts; it must not mutate them and must not spawn new training/evaluation runs.
- The recorded result is **trustworthy enough to investigate.** The historical failure modes that previously blocked interpretation are all resolved and preserved as evidence:
  - broken/modified runner → replaced by hash-bound final runner `C8E298C8…`
  - missing/late checkpoint → 56/56 present, incl. `checkpoint_20` in all conditions
  - invalid or reused authorization → single immutable L5 auth `5214DA99…`; old `4AA21378…` marked MUST-NOT-REUSE
  - disk failure → v6 disk-full frozen as evidence, not repeated
  - incomplete provenance → final manifest `MISSING entries: NONE`, real registration hashes present

**Implication:** Any future execution that revisits the negative finding must be a **new phase** with a **new explicit authorization**. Until then, 6E.20.1 remains a clean negative result suitable for causal investigation from first principles.

*End of forensic record.*
