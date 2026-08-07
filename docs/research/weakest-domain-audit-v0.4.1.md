# Weakest-Domain Audit — v0.4.1 Baseline

**Experiment**: Weakest-domain audit (corpus sweep, engagement + failure-mode analysis)
**Baseline**: THEO v0.4.1 (frozen, tag `v0.4.1`, commit `d02a062`)
**Date**: 2026-08-08
**Status**: Results (research note — not a release artifact)
**Method**: every case in the 35-case / 6-domain corpus executed via
`BenchmarkHarness.run`; per-case and per-domain engagement diagnostics extracted
(hypotheses, fired rules, derived beliefs, conflicts, thought-DAG nodes,
confidence, decision type, intent). Script: temp audit harness, not committed.

---

## Headline Result

**All 35 cases pass.** Pass rate is 1.0 across all six domains, so the corpus
contains no *failing* failure modes. The audit therefore measures *how much of
the cognitive machinery each domain actually exercises*, and where the pipeline
produces degenerate or compressed signals.

## Per-Domain Engagement

| Domain | Cases | Pass | Hypoth | Rules fired | Derived | Conflicts | DAG nodes | Mean conf | Intent |
|---|---|---|---|---|---|---|---|---|---|
| taxonomy | 5 | 5/5 | 1.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | maintain_conversation |
| commonsense | 5 | 5/5 | 2.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 | maintain_conversation |
| contradiction | 7 | 7/7 | 2.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 | maintain_conversation |
| ambiguity | 5 | 5/5 | 3.2 | 0.0 | 1.0 | 2.2 | 0.0 | 1.0 | maintain_conversation |
| causal_reasoning | 8 | 8/8 | 2.8 | 0.9 | 1.9 | 1.8 | 0.9 | 1.0 | maintain_conversation |
| uncertainty | 5 | 5/5 | 1.8 | 1.0 | 2.0 | 0.8 | 1.0 | 0.78 | maintain_conversation |

*Hypoth* = mean generated hypotheses; *Rules fired* = mean fired rules;
*Derived* = mean derived beliefs; *Conflicts* = mean resolved conflicts;
*DAG nodes* = mean thought-DAG node count.

## Findings

### F1. The corpus is uniformly green — it captures no failing failure modes
All 35 cases pass with exactly the pinned expectations. The corpus validates
the *intended* behavior but cannot yet probe *degradation*: no case is designed
to be failed by the current runtime. Weakest-domain signal must come from
engagement metrics, not failures.

### F2. Decision confidence is compressed — 31/35 cases report 1.0
Only the uncertainty domain differentiates confidence (0.65–0.80). taxonomy,
commonsense, contradiction, ambiguity, and causal_reasoning all report
confidence 1.0 — including ambiguity cases that hold **4 competing hypotheses**
and contradiction cases resolved via ties. Confidence does not currently encode
remaining uncertainty outside the uncertainty domain; it behaves like a
pass/fail flag, not a calibrated posterior.

### F3. Rule-based inference is sparsely exercised — 24/35 cases fire zero rules
Only causal_reasoning (mean 0.9) and uncertainty (mean 1.0) fire rules and grow
a thought DAG. taxonomy, commonsense, contradiction, and ambiguity never invoke
the inference engine. Most of the corpus exercises perception → retrieval →
hypothesis → decision, with the Inference and Thought-Graph stages idle.

### F4. taxonomy is the least-exercised domain — near-degenerate
taxonomy runs **1.0 hypotheses, 0.0 rules, 0.0 conflicts, 0.0 DAG nodes, and 0
retrieved memories**. Its decision reduces to a single percept-self-match with
concept activation. It is the weakest domain by every engagement axis: it
exercises essentially only PERCEPTION and HYPOTHESIS self-matching.

If taxonomy is intended to exercise inheritance (e.g., Dog → Mammal → Animal,
requiring traversal, inheritance, and property propagation), this domain is
currently testing **recognition rather than inference** — no rules fire, so it
probably is not stressing the subsystem it is named after.

### F5. Intent is uniformly `maintain_conversation`
All 35 decisions carry the same intent. The GoalManager's intent vocabulary
(acknowledge_greeting, remember_fact, provide_recommendation, answer_question,
maintain_conversation) is never differentiated by the corpus, so intent
selection is untested as a failure mode.

### F6. Law 6 (ambiguity) is exercised correctly — a positive result
ambiguity produces the widest hypothesis/conflict breadth (3.2 hyp, 2.2
conflicts) and correctly avoids premature collapse (5/5 pass): multiple
hypotheses are generated and conflicts retained. This is a structurally correct
outcome. The limitation is that ambiguity is not reflected *numerically* —
**zero inference**, and **confidence 1.0** — so the runtime keeps ambiguity in
the hypothesis set without letting the decision layer see it.

## Weakest-Domain Ranking

By engagement + signal quality, weakest to strongest:

1. **taxonomy** — degenerate (single self-match, no inference, no memory).
2. **commonsense** — retrieval-only (no inference, no DAG).
3. **contradiction** — revision/conflict-only (no inference, no DAG).
4. **ambiguity** — wide hypothesis set, but no inference and compressed confidence.
5. **causal_reasoning** — engages rules + DAG, but confidence still 1.0.
6. **uncertainty** — the only domain exercising inference, DAG, *and*
   differentiated confidence.

## What the Measurements Do and Don't Show

- **The symbolic runtime never fails on this corpus** — so the corpus currently
  measures *intended behavior*, not degradation. The first research target is
  therefore degradation, not correctness.
- **Confidence compression limits the decision policy's ability to distinguish
  equally valid and uncertain reasoning paths** (F2). This is a *signal gap*:
  the data shows 31/35 confidence estimates are 1.0, and that hypothesis
  ranking has little observable influence today. It does **not** show that a
  different confidence model would change those 31 final decisions — whether it
  does depends on how decision policies actually consume confidence.
- **The seam for learned components, if any, is estimation, not reasoning**:
  the symbolic runtime already generates admissible hypotheses, enforces
  constraints, maintains provenance, and applies deterministic rules. A
  neuro-symbolic component would contribute by estimating what the symbolic
  layer treats coarsely — plausibility, confidence, relevance, salience,
  ranking — while the symbolic runtime remains the arbiter.
- **Corpus gaps observed**: (a) intent-differentiated cases are absent (F5),
  (b) no case has a provably *wrong* expected outcome, so degradation is
  untested, (c) no case combines inference chains with genuine uncertainty, so
  confidence and DAG depth never co-vary.
- **Baseline freeze holds**: these are findings *about* the frozen v0.4.1
  baseline; they motivate experiments, not changes to the frozen architecture.

## Research Questions

The audit identifies the following hypotheses for future investigation:

1. Can calibrated confidence improve ambiguity handling — i.e., does exposing
   remaining uncertainty to the decision policy change decision quality, not
   merely confidence values?
2. Can richer taxonomy rules (inheritance traversal, property propagation)
   increase inference engagement in the taxonomy domain?
3. Should decision policies consume confidence differently — and what is the
   empirical effect of each consumption policy on the corpus?
4. What benchmark characteristics best distinguish symbolic from
   neuro-symbolic performance (e.g., confidence-graded ambiguity vs. ranked
   hypothesis quality)?

## Reproducibility

Audit run under the frozen baseline (tag `v0.4.1`). Per-case JSON is available
in the session temp output (`audit_report.json`); the harness itself is the
committed `theo benchmark run` path (`BenchmarkHarness.run_all`).
