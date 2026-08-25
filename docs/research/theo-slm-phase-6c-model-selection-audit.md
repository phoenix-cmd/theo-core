# THEO SLM v0 — Phase 6C Model-Selection Methodology: Independent Audit

**Document ID:** `docs/research/theo-slm-phase-6c-model-selection-audit.md`  
**Date:** 2026-08-25  
**Role:** Independent review of `theo-slm-phase-6c-model-selection-plan.md`  
**Mode:** READ-ONLY. No weights downloaded, no training, no dataset mutation, no 6C/6D execution.

---

## 1. Audit scope & independence statement

This audit reviews the **methodology** of the companion plan, not the eventual model. It verifies that:

1. the selection corpus is authoritative and immutable;
2. no prohibited selection signal is used (`cos(H1)`, gradient-compatibility, hidden-state subspace
   alignment, Qwen labels);
3. only approved behavioral / structured-output criteria are used;
4. the candidate set is within the approved 6B §F set;
5. the selection is compatible with ADR-0028 / v0.5 constraints;
6. the evaluation discipline (frozen instruments, grouped split, behavioral gates) is sound.

I was not involved in drafting the plan's conclusions and assess it against the binding authorities:
`theo-slm-design-v0.md` (§A–§H), `theo-slm-phase-6b-review-packet.md` (Gates 1–6), `v0.5-neural-interface-plan.md`
(Constraints 1–4), `v0.4.1-reference.md`, and the post-Batch-2 readiness review.

---

## 2. Verification — corpus authority & immutability

| Check | Result |
|---|---|
| Canonical hash `97bc6bc6…` cited and matched | **PASS** |
| GOLD = 300 (84 POS / 66 ABSTAIN / 150 NEG); 44 REJECT excluded | **PASS** |
| No Batch-3 generation performed | **PASS** (deferred per authorization) |
| No mutation of `adjudication.json` | **PASS** |
| Source of GOLD labels = human/symbolic review (6B.2/6B.3), not Qwen | **PASS** |

**Finding:** The corpus used for selection is the correct, frozen, post-Batch-2 canonical state. No
label-contamination path is present.

---

## 3. Verification — prohibited signals explicitly excluded

| Prohibited signal | Present in plan? | Evidence in plan | Verdict |
|---|---|---|---|
| `cos(H1)` | **No** | §6 table lists it as excluded; §2 criteria contain no hidden-state metric; cites `cos_H1(SENS)=0.322966<0.95` INVALID | **PASS** |
| Gradient-compatibility metrics | **No** | §6 cites design §B/§G exclusion; absent from §2/§4 | **PASS** |
| Hidden-state subspace alignment | **No** | §6 cites review-packet Gate 1 ("no aligned readout subspace"); capability list is observable behavior | **PASS** |
| Qwen-generated labels as positives | **No** | §6 cites design §D/D1; Qwen = negatives/format-regressions only | **PASS** |

**Finding:** All four prohibited signals are demonstrably absent from the criteria set, the scoring
matrix (§4), and the decision rule (§5). The plan correctly anchors the exclusion in the binding
authorities rather than asserting it informally.

---

## 4. Verification — only approved behavioral / structured-output criteria used

The §2 criteria (C1–C12) are all measurable from model **outputs** or **deployment properties**:
structured-output failure (E0), grounding, semantic novelty (E5), decision relevance (E6), Group E
distractor rejection, abstention accuracy, latency, footprint, and v0.5 interface compatibility. None
requires probing internal model state.

**Finding:** Criteria conform to design §G ("behavioral / structured-output gates only") and
review-packet Gate 4. **PASS.**

---

## 5. Verification — candidate set within approved §F set

The plan scores exactly the three §F-approved architecture comparisons: single decoder-only (A),
decoder + reranker (B), shared-encoder/two-head (C). The base-model family (≤1B open-weight) is
correctly treated as an implementation detail owned by the design doc, not ADR-0028 — satisfying v0.5
Constraint 4 (model independence).

**Finding:** Candidate set is within scope; no out-of-set architecture is introduced. **PASS.**

---

## 6. Verification — v0.5 / ADR-0028 compatibility

| Constraint | Addressed in plan? | Verdict |
|---|---|---|
| C1 baseline invariance (no-provider = v0.4.1) | §8 | **PASS** |
| C3 determinism / replay / grounding | §7, §8 | **PASS** |
| C4 model independence (not frozen into ADR-0028) | §3, §5, §8 | **PASS** |
| Architectural firewall (`theo_core` ∌ `theo_providers`) | §8 | **PASS** |

**Finding:** The plan keeps the SLM strictly behind the ADR-0028 hooks and preserves the frozen
interface. **PASS.**

---

## 7. Verification — evaluation discipline

- Reference instruments (frozen 51-case benchmark + 15-case probe) are declared evaluation-only and
  byte-locked; never trained on. **PASS.**
- Grouped-by-seed-family split preserving the intentional `/005`∼`/050` near-duplicate group is
  specified, preventing split-leakage. **PASS.**
- Behavioral GO thresholds (E0 ≤2%, grounding 100%, E5 ≥40%, E6 ≥30%, Group E ≥80%, abstain ≥90%,
  shortcut ≤40%) are carried from design §F/§G. **PASS.**
- No `cos(H1)`-style metric can silently enter the 6C eval (review-packet Gate 4 note). **PASS.**

---

## 8. Residual-risk findings & recommendations

| # | Finding | Severity | Recommendation |
|---|---|---|---|
| R1 | §4 architecture scores are **projections**, not measurements (6A.2 only measured the reference SLM, not THEO SLM v0). | Informational (inherent to a plan) | Acceptable for planning; **mandate** that A/B/C selection is finalized only on measured 6C.3/6D gates, never on §4 projections. |
| R2 | Positive fraction = 28% vs §13 ~40% guideline. | Recommended (non-blocking, per §12) | Monitor at 6C/6D; authorize Batch-3 **only** if a concrete *blocking* deficiency (e.g., zero-positive capability) appears. Do not up-weight synthetically. |
| R3 | 220/300 GOLD have empty `capability_family`/`domain`; capability-stratified coverage measurable only on 80 records. | Informational | Behavioral gates rely on the fully-tagged frozen probe/benchmark, so selection is unaffected; document this coverage limit in the 6C/6D report. |
| R4 | Within-batch structural correlations (percept r=0.61, proposition r=0.50) exist on features that are **never model inputs**. | Informational | Correctly excluded from selection; keep documented so a future run does not mistake them for usable signals. |
| R5 | Stopping/rescue discipline (design §H.4): if a measurement instrument fails its validation control, stop — do not increase training dose. | Process | Ensure this gate is wired into the 6C.3 build plan as a hard exit before any training increase. |

---

## 9. Audit verdict

- **Methodology is SOUND for a planning / read-only selection analysis.**
- All four prohibited signals are verifiably excluded; all selection criteria are approved behavioral /
  structured-output criteria; the candidate set is within the §F-approved set; v0.5 / ADR-0028
  compatibility is preserved; the evaluation discipline is correct.
- The only inherent limitation (R1) — that architecture scores are pre-build projections — is
  appropriate for a plan and is mitigated by the explicit rule that final selection occurs on measured
  gates.
- **No authorization to build, download weights, train, or evaluate is conferred by this audit.** 6C.3
  build and 6D evaluation each require separate explicit human sign-off.

**Auditor sign-off:** methodology APPROVED for planning purposes; proceed to human review of the plan
+ audit before any 6C execution.

---

## 10. Human Approval Record (2026-08-25)

- **Status:** The plan **and** this audit are **APPROVED** for planning purposes.
- **Preserved constraints confirmed by this audit:**
  1. Canonical dataset immutable at `97bc6bc6…` (no mutation; 44 REJECT excluded).
  2. Batch 3 deferred (no generation performed).
  3. No `cos(H1)` / gradient-compatibility / hidden-state-subspace criteria present (verified §3).
  4. No Qwen-generated labels as positives (verified §2/§3).
  5. Candidate architectures within approved §F set (verified §5).
  6. §4 projected scores are provisional and must be superseded by measured 6C.3/6D results (finding R1).
- **This approval confers no execution authority:** weight download, training, and 6D evaluation each
  require separate explicit human authorization (see the 6C execution/readiness package).
