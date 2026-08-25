# THEO SLM v0 — Fixed-Subspace-Readout Experiment (6E.20.x) Outcome

**Status:** CLOSED (inconclusive). No new authorization. Frozen.
**Source of truth:** `theo-core/scripts/dataset_generator/6E.20_FINAL_RESEARCH_CONCLUSION.md`
**Date recorded:** 2026-08-25

---

## 1. What 6E.20.x was

The 6E.20.x line (6E.20.1 → 6E.20.7) was a training-methodology sub-experiment within the
broader THEO SLM v0 program. It tested whether a **fixed low-dimensional aligned readout
subspace** induces **hidden-state gradient compatibility** measurable by the `cos(H1)`
gradient-cosine observable.

## 2. What was established

- The execution pipeline was repaired to the point of reaching the actual scientific
  evaluation: condition-resolved 28-evaluation run completed (6E.20.5); Stage 7 integrity
  passed; the historical B0 anchor was reproduced exactly after an eval-mode fix (6E.20.4).
- **H5 (behavioral specificity) is the one robust positive:** structured/LDA-derived
  readouts drive strong PRO/ABS behavioral separation (Cohen's *d*) versus random readouts.
- A complete, auditable forensic trail exists from 6E.20.1 through 6E.20.7.

## 3. What failed

- The primary geometric observable **failed its own pre-registered sensitivity control**:
  `cos_H1(SENS) = 0.322966 < 0.95`.
- Consequently `cos(H1)` is declared **INVALID for measuring H1**; the broader premise
  (fixed subspaces can induce gradient compatibility) is **UNRESOLVED**, not refuted.
- Formal protocol verdicts — **H1 REFUTED, H3 REFUTED, H4 REFUTED, H5 SUPPORTED** — are
  preserved as historical records, but the geometric REFUTED verdicts are **not valid
  scientific rejections of the premise** (the instrument that produced them is invalid).

## 4. Durable lessons for THEO SLM v0

> **`cos(H1)` is not a validated training/evaluation signal for THEO SLM v0.**

1. **Do not use hidden-state gradient-cosine compatibility (`cos(H1)`) as a training
   signal, regularization term, or evaluation metric** for THEO SLM v0 — it is not
   validated.
2. **Evaluate candidate models on behavioral / structured-output gates** (E0 /
   structured-output failure rate, grounding rate, semantic-novelty rate,
   decision-relevance) — exactly the gates specified in `theo-slm-design-v0.md` §G. Keep
   that discipline.
3. **Any future gradient-based probe must pass a pre-registered sensitivity control**
   before its output informs any scientific conclusion (the 6E.20.7 lesson).
4. **Stopping rule:** when a measurement instrument fails its validation control, stop
   and redesign the measurement; do **not** increase training dose or iterate on the same
   metric (rescue loop).

## 5. Branch status

- 6E.20.x is **frozen**. No new authorization, no implementation, no training.
- **6E.20.8** (measurement-instrument redesign) is a **separate future branch**, gated,
  requiring a **materially different measurement strategy or scientific question**. It is
  **not** an automatic continuation of 6E.20.x.
- The roadmap now points back to the actual product path:
  **6B review → 6C build → 6D comparative evaluation → v0.5 Neural Runtime**.

## 6. References

- `theo-core/scripts/dataset_generator/6E.20_FINAL_RESEARCH_CONCLUSION.md` (authoritative closure)
- `theo-core/docs/research/theo-slm-design-v0.md` (6B; see §H for the folded lesson)
- `theo-core/docs/implementation/v0.5-neural-interface-plan.md` (roadmap of record)
- `theo-core/adr/ADR-0028-neural-symbolic-interface.md` (frozen interface contract)
