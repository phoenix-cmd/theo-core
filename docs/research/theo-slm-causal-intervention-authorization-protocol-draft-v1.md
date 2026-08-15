# THEO SLM Causal Intervention Authorization Protocol — DRAFT v1

- **Status:** DRAFT — DESIGN ONLY. **NOT AUTHORIZED FOR EXECUTION.**
- **Date (UTC):** 2026-08-16
- **Authority:** Requires explicit human authorization before any execution. Phase 6E.17 must not begin until this protocol (or an approved revision) is accepted.
- **Basis:** Phase 6E.16-R `theo-slm-phase-6e16r-forensic-reconciliation-v1.md` (§8, §10 Canonical Interpretation Record).
- **Scope of this document:** define exactly what one causal intervention experiment would test, and the conditions under which it may be authorized. This document performs NO experiment.

---

## 1. Why an intervention is now the right step (and why it is still gated)

Phases 6E.13–6E.16 produced only observational geometry forensics. Their limit:

- Observational evidence establishes the ordering `readout misalignment ≺ gradient conflict` (temporal precedence, INFERRED), NOT `readout misalignment ⇒ gradient conflict` (causation).
- Every candidate explanation that remains is classified INFERRED or NOT ESTABLISHED in the Canonical Interpretation Record (§10.2).

Purely observational analysis of this kind has been exhausted. One **single-variable causal experiment** is the only way to move the top-line interpretation from INFERRED to ACTUALLY MEASURED. However, execution is gated on human authorization because (a) it is the first non-read-only step in this research track, and (b) the hard-stop discipline that produced the corrected record must not be abandoned by starting Phase 6E.17 prematurely.

## 2. Pre-registered causal hypothesis (single)

> **H-Causal:** The failure arises substantially because the fixed PROPOSE/ABSTAIN
> vocabulary contrast is poorly aligned with the task-discriminative representation
> geometry; modifying the decision formulation to provide an appropriately aligned
> trainable or alternative output geometry should reduce the measured scalar conflict and
> improve simultaneous POS/ABS optimization.

Companion null hypothesis (the outcome the experiment must be able to distinguish from):

> **H0:** The scalar decision conflict and subsequent gate collapse are caused by the
> training dynamics / objective themselves, independently of the readout geometry;
> an aligned trainable readout produces no measurable reduction in scalar conflict and no
> change in collapse onset.

The experiment is a falsification test of H0. If H0 survives, M4 (scalar decision-contrast
bottleneck) is demoted to a correlate, not a cause.

## 3. One minimal intervention family

Only ONE intervention variable may be manipulated in the first experiment. The selected
minimal family is:

**I1 — Trainable decision readout aligned to the task-discriminative geometry.** Replace
the frozen vocabulary contrast `w_delta = w_PRO - w_ABS` (applied to the fixed output
embedding layer) with a trainable linear readout head over the hidden state that is
initialized to align with the measured discriminative direction at step 0 (the Fisher LDA
axis; `u_LDA` in `phase-6e16/diagnostic_basis_alignment_trajectory.json`), then trained
jointly with the LoRA. Everything else — architecture, optimizer, LoRA, data, horizon —
is unchanged.

Rationale for choosing I1 over the other candidates (verbatim from H-Causal and the
NOT-ESTABLISHED list):

| Candidate | Decision | Reason |
|-----------|----------|--------|
| Trainable aligned readout (I1) | **SELECTED** | Directly tests the alignment component of H-Causal; minimal surface area; single variable. |
| Different verbalizer (e.g., PROPOSE via a different token) | DEFERRED | Changes the token-level prior; confounds "alignment" with "token priors"; save for a follow-up. |
| Multi-token decision formulation | DEFERRED | Larger architectural change; violates minimality for the first test. |
| Learned classifier replacing the readout entirely | DEFERRED | Overlaps I1 but changes the decision boundary form; only after I1 result. |
| LoRA rank / optimizer / data changes | PROHIBITED | Outside H-Causal; would break the single-variable discipline. |

**Explicit prohibition (immutable):** no combination of fixes may be tried until H-Causal
vs H0 is evaluated with I1 alone. "Trying multiple fixes until one works" is prohibited.

## 4. Immutable controls

These are fixed for BOTH the baseline and intervention conditions and are non-negotiable:

1. **Base model:** Qwen/Qwen2.5-0.5B-Instruct, sha256
   `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
2. **Corpus:** THEO SLM v0 corpus, sha256
   `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0`. Corpus read-only.
3. **Optimizer / LoRA:** AdamW lr=1e-4, weight_decay=0.01, LoRA r=16, lora_alpha=32,
   lora_dropout=0.05, batch_size=4, grad_accum_steps=2, torch.bfloat16 (the 6E.14/6E.15
   configuration family — see §6 trajectory identity).
4. **Seed policy:** identical seeds for all stochastic components in both conditions
   (identical data ordering, identical init). Only the intervention variable differs.
5. **Diagnostic panels:** N=16 (8 POS + 8 ABS) and N=24 (+8 NEG), Euclidean distance,
   k in {1,3,5}, resubstitution + LOO-CV, linear SVM 3-fold.
6. **Checkpoint set:** t in {0,4,6,7,14,15} and the full-step telemetry used for collapse
   onset and conflict timing.
7. **Metrics and their definitions:** frozen exactly as defined in phases 6E.12–6E.16
   (cos(H1) decision-gradient conflict, scalar-margin r_eff, row-normalized joint r_eff,
   max |cos(w_delta, u_disc)|, Cohen's d margin separation, k-NN/SVM separability,
   proposal-rate collapse onset t*). No metric may be redefined post hoc.
8. **Forensic protocol:** all artifacts saved with SHA-256 provenance; same
   anti-fabrication discipline as 6E.14–6E.16.

## 5. Matched baseline (required; historical phases are NOT the baseline)

Due to D1 (trajectory identity), the historical 6E.14/6E.15/6E.16 checkpoints are
different parameter states and CANNOT serve as the control. A **fresh control run** must
be executed under the identical configuration family as the intervention (including the
new code paths and any I1 changes compiled out), producing its own full trajectory.

The control run must demonstrate that the observational fingerprints reproduce under the
matched config before the intervention is evaluated:

- cos(H1) at step 6 near the historical lock-in value (≤ -0.8) and near-perfect opposition (-0.99) by t=6–14;
- collapse onset t* = 15 (or within the matched horizon);
- scalar-margin r_eff in [1.0, 1.6] throughout;
- row-normalized joint r_eff >= 3.4 at step 0 and >= 3.0 at t=15;
- max |cos(w_delta, u_disc)| <= 0.05 at all checkpoints.

If the control does NOT reproduce these fingerprints, the experiment is halted and the
protocol must be revisited (the fingerprint set itself becomes suspect).

## 6. Trajectory identity controls

- One trajectory per condition (control, I1). No checkpoint reuse across conditions.
- Both conditions use the SAME configuration family and SAME seed policy; the ONLY
  difference is the decision readout variable (frozen w_delta contrast vs aligned
  trainable readout I1).
- Step-0 state (base model before training) must be hash-identical in both conditions
  (base model sha256, §4.1).
- All comparisons are WITHIN trajectory-pair (control vs I1), never against historical
  phase numbers except as reproduction checks in §5.
- Any config difference discovered post hoc between conditions invalidates the pair.

## 7. Pre-registered success / failure criteria

These are fixed BEFORE execution. No threshold adjustments after results are seen.

**Primary outcome (H-Causal's core prediction):**
- P1 — Decision-gradient conflict reduction: mean cos(H1) over steps {4,6,14,15} in the I1
  trajectory is >= -0.30 (control expected ~-0.93 to -0.997). If P1 holds with
  non-overlapping CI, the alignment component of H-Causal is supported.

**Secondary outcomes (strength and safety):**
- P2 — Collapse prevention: proposal rate stays strictly > 0 through the matched horizon
  in the I1 trajectory (control locks to 0% at t*=15).
- P3 — Margin maintenance: scalar-margin Cohen's d stays >= +0.2 through t=15 in the I1
  trajectory (no inversion), and margin r_eff remains near-rank-1 (<= 1.6) — i.e., the
  scalar contrast itself is preserved while its alignment is fixed.
- P4 — No separability degradation: resub k-NN1 in {0.65, 1.0} (does not drop materially
  below the 0.75 baseline), LOO-CV within +/- 0.1 of control. Guards against a trivially
  collapsed model that "avoids conflict" by losing task information.
- P5 — No readout-geometry drift: max |cos(w_delta', u_disc)| for the trained readout
  w_delta' stays >= 0.3 at t >= 4 (i.e., the trained readout actually engaged the
  discriminative direction, not merely de-coupled from it).

**Decision rules:**
- H-Causal supported: P1 holds AND P2 holds AND P4 holds.
- H-Causal partially supported: P1 holds, P2 fails, P4 holds (conflict reduced but
  collapse not prevented) — proceed to a second, separately approved hypothesis about the
  residual mechanism.
- H0 survives: P1 fails (no conflict reduction) — M4 demoted to correlate; do NOT proceed
  to alternative fixes; write up and return to human review.
- Protocol violation (any §4/§6 deviation): the run is void, regardless of outcome.

## 8. Outputs and forensic packaging

- Machine-readable artifact directory `phase-6e17-causal-intervention/` (naming pending
  human approval of the phase) with: run configs (both conditions), full trajectory
  telemetry, spectra, alignment trajectories, chronology JSON, SHA-256 manifests,
  pre-registration snapshot (this document, hash-frozen before execution).
- Report: `theo-slm-phase-6e17-causal-intervention-v1.md` (draft naming), including an
  explicit update to the Canonical Interpretation Record tier assignments.
- Everything subject to the same anti-fabrication and provenance discipline as 6E.16-R.

## 9. Authorization gate (this document does not authorize anything)

Before any execution, a human reviewer must approve ALL of the following explicitly:

1. The single intervention variable is I1 and the deferred list (§3) is respected.
2. The immutable controls (§4) and trajectory identity controls (§6) are accepted.
3. The matched control requirement (§5) is accepted as mandatory.
4. The success/failure thresholds (§7) are accepted as written.
5. The run is authorized as one experiment with one hypothesis, with no "try multiple
   fixes" fallback.

Sign-off fields (to be completed by human reviewers):

- Reviewer name / role: ____________
- Decision: APPROVE / AMEND / REJECT
- Amendments (if any): ____________
- Date: ____________

Until these fields are completed by a human, this protocol is a draft and NO training,
fine-tuning, or model execution may occur.
