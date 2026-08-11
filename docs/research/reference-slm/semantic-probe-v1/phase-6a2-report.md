# Phase 6A.2 Report — Qwen3-0.6B Semantic Capability Probe

**Status:** APPROVED — **NEGATIVE / DIAGNOSTIC** (human-reviewed, 2026-08-11)
**Date:** 2026-08-11
**Phase:** 6A.2 (semantic capability probe, frozen 15-case instrument)
**Instrument:** `semantic-probe-v1` — frozen spec + manifest (theo-core, commit `5e42654`), read-only
**Prompt:** `qwen3-hypothesis-v1` via the frozen evaluator (unchanged; no tuning, no parser changes)
**Model:** `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` (unchanged from 6A.1)
**Run policy:** one-shot — one greedy generation per case, seed 0, `temperature=0.0`, `do_sample=false`, `max_new_tokens=512`, CUDA fp16 on GTX 1650

---

## 1. Objective

Measure whether the reference SLM, operating as an ADR-0028
`HypothesisProposalProvider`, can produce a grounded *semantic interpretation*
that the current symbolic runtime **cannot already derive** — the property the
6A.1 corpus could not expose (0/20 decision-relevant proposals, all premise or
taxonomy echoes).

The probe is deliberately structured so the target content is **not present
verbatim** in the input: 15 cases across five capability groups (A–E ×3):
paraphrase, inference, contradiction-inference, belief-free, and
distractor-sensitive evidence selection.

## 2. Frozen methodology

- Frozen inputs: `semantic-probe-v1-spec.md` (spec, freeze rules, E0–E6 rubric,
  evaluation funnel) and `semantic-probe-v1-cases.json` (15 cases). Neither was
  modified.
- Frozen evaluator (`theo-providers/src/theo_providers/semantic_probe/evaluator.py`):
  deterministic measurement (grounding validation, exact-repeat detection, token
  containment, taxonomy-echo and trap checks, evidence-relevance scoring) that
  pre-labels E0–E6 **without judging semantic novelty or decision relevance**.
  Those are human review slots (`human_e_class`, `human_decision_relevance`,
  `human_note`) applied after the run via `apply_human_labels`.
- Frozen parser: unmodified `qwen3/parse_completion`. Rejected proposals never
  reach assessment (there is no `grounded=False` path).
- One generation per case, raw output preserved verbatim next to its parsed
  interpretation. Determinism verified by replay.
- Provenance/hash machinery identical to 6A.1: `model_hash a80410e0…49cca8`,
  `tokenizer_hash 92ba2f61…1a743` (Hub manifest, never by double-loading weights).

## 3. Model provenance

| field | value |
|---|---|
| model_name | Qwen/Qwen3-0.6B |
| model_revision | c1899de289a04d12100db370d81485cdf75e47ca |
| model_hash | a80410e02451ae27828f89ccc87eb55362a14b561fec1937b24a599c9849cca8 |
| tokenizer_hash | 92ba2f610797f9ac063d2fcd678beb23eced9256c0144ab80c593b7ece91a743 |
| device | cuda (torch 2.13.0+cu126, restored from the CPU-only build that initially blocked the run) |
| torch_dtype | torch.float16 |
| seed / temperature / do_sample / max_new_tokens | 0 / 0.0 / false / 512 |
| transformers_version | 5.14.1 |

## 4. Results — aggregate (deterministic pre-labels)

| metric | value |
|---|---|
| Total cases | 15 |
| Structured-output failures (E0) | **8/15 (53.3%)** |
| Parser-accepted proposals | 7/15 |
| Accepted proposals grounded | **7/7 (100%)** — zero E1 |
| E0–E6 pre-label distribution | E0 8 · E1 0 · E2 1 · E3 2 · E4 0 · E5 1 · E6 3 |
| Grounded % | 46.7% (7/15 cases) |
| Semantic novelty pre-label (E5+E6) | 26.7% (4/15) |
| Repeat/paraphrase (E2+E3) | 20% (3/15) |
| Rule-echo (E4) | 0% |
| Unsupported (E0+E1) | 53.3% (8/15) |
| Evidence-relevance (Group-E-scoped, spec §4) | 0.0 |
| Latency (wall, 15 cases) | 773.7 s ≈ **51.6 s/case** |

### A–E groups (pre-labels)

| group | capability | E0 | E2 | E3 | E5 | E6 | grounded | novelty |
|---|---|---|---|---|---|---|---|---|
| A | paraphrase | 2 | 0 | 0 | 0 | 1 | 33% | 33% |
| B | inference | 2 | 0 | 0 | 0 | 1 | 33% | 33% |
| C | contradiction-inference | 1 | 0 | 0 | 1 | 1 | 67% | 67% |
| D | belief-free | 1 | 1 | 1 | 0 | 0 | 67% | 0% |
| E | evidence + distractors | 2 | 0 | 1 | 0 | 0 | 33% | 0% |

## 5. Human-reviewed classifications

Review applied the deciding rule from the probe spec: *could the symbolic
runtime already derive this proposition from what it currently possesses?*
Yes → repeat/echo. No, but supported by the evidence → E5. Plus a meaningful
connection to the task → E6.

| case | proposal (accepted) | pre | **human** | rationale for review decision |
|---|---|---|---|---|
| a/001 | "The container was broken by the impact on the floor." | E5 | **E3** | Paraphrase of the percept ("shattered after hitting the floor"); runtime already possesses it. Not decision-relevant. |
| b/002 | "The outage disrupted electricity, causing the lights to go out and the microwave clock to blink, while the fridge hummed to life." | E5 | **E6** | Genuine abduction to an unobserved cause; not derivable (no beliefs/rules; nothing asserts an outage); directly answers "what explains the observations?". Retained as the isolated signal. |
| c/001 | "The door is locked from the outside and accessible to the inside." | E5 | **E3** | Echo of stored belief ("The door was locked from the outside." — the proposal cites the belief id); added clause contradicts the percept. Not decision-relevant. |
| c/002 | "The room is dark and the light is on." | E5 | **E3** | Conjunction of two stored beliefs ("room light is on" + "room is dark"); no new content. Not decision-relevant. |

**Reviewed distribution:** E0 8 · E1 0 · E2 1 · E3 5 · E4 0 · E5 0 · E6 1
(**semantic novelty 1/15 = 6.7%**, decision-relevance 1/15 = 6.7%, repeat/
paraphrase 6/15 = 40%).

**Reviewed per-group:** A `{E0:2,E3:1}` · B `{E0:2,E6:1}` · C `{E0:1,E3:2}` ·
D `{E0:1,E2:1,E3:1}` · E `{E0:2,E3:1}`. Novelty per group: only B > 0 (33.3%).

## 6. b/002 — the isolated successful case

`sp1://b/002` (group B, inference). Percept: "The lights went out. The
microwave clock was blinking. The fridge hummed to life." Concepts: power,
electricity, light, outage. **No beliefs, no rules.** Ground-truth target:
"There is a power outage" (decision target: "what explains the observations?").

The accepted proposal posits an **unobserved cause** (outage) that explains the
observed symptoms. The symbolic runtime cannot derive this: nothing in its
possessions asserts an outage and no rule maps the symptoms to it. It therefore
clears the E5 test, and because it answers the decision target, it clears E6.

**Governance:** b/002 is retained as a capability probe / gold-standard case and
a positive training example for 6B (see `theo-slm-design-v0.md`, §D). It is an
isolated signal, **not** evidence of provider readiness.

## 7. Group E — evidence-selection failure

Group E is the distractor-sensitive group: each case supplies the concepts an
answer needs plus designated distractors, and the ground-truth candidate must
cite the relevant evidence. Results:

- e/001 — accepted proposal "The sky is blue and the road is wet." cited
  `sky` + `wet`, evidence-relevance **0.0** (did not cite the rain evidence; the
  intended interpretation "recent rain" was not produced).
- e/002, e/003 — structured-output failures (E0).

Reviewed Group E: no proposal cleared E5/E6, and the spec metric (mean
evidence-relevance over Group E proposals) is **0.0**. The distractor design
demonstrates the model does not reliably select the right evidence.

## 8. Structured-output failure rate (8/15, 53.3%)

The 8 E0 cases are **format failures, not non-engagement**. Qwen produced
hypothesis-shaped JSON in all 15 cases, but in 8 cases the output was malformed:

- literal `...` elision inside the JSON structure, e.g.
  `{"hypotheses": [{"proposition": "...", "referenced_ids": ["concept://…"], "relation": "both"}], ...}` —
  the `...` makes the object unparseable;
- one case echoed the prompt's `{ ... }` template placeholder verbatim.

The frozen parser correctly rejected all malformed output ("no JSON").
Rejected count per case = 1; no partial proposals were recovered. This is a
*fidelity* failure of the model to the constrained output schema, and it is the
single largest source of the negative result.

## 9. Latency

- Total wall: 773.7 s for 15 cases (one generation each) ≈ **51.6 s/case**.
- The model generates to near the 512-token cap on most cases (verbose,
  self-repetitive continuation, e.g. c/001 repeats the same proposition ~40×),
  so effective throughput is ~10 tok/s on the GTX 1650 (CUDA fp16).
- For reproducibility of the *environment*: the venv had silently resolved to a
  CPU-only `torch 2.13.0+cpu` build (~1.5 tok/s); it was restored to the 6A.1
  build `torch 2.13.0+cu126` (`[tool.uv.index]` pytorch-cu126) before the run.
  This is infrastructure provenance, not an experiment change.

## 10. Determinism

- Replay (fresh process, same frozen run) produced **byte-identical raw outputs**
  for all 15 cases; classifications and metrics identical.
- One cosmetic issue: the pre-label `rationale` string rendered referenced ids
  via `str(frozenset)`, whose iteration order depends on `PYTHONHASHSEED`.
  Fixed in the evaluator to render the already-sorted tuple; verified byte-stable
  across hash seeds. Raw outputs, classifications, and metrics are unchanged; the
  stored raw artifacts were not regenerated.
- transformers 5.14.1 logs that `temperature`/`top_k` flags "may be ignored";
  because decoding is greedy this does not affect determinism (same note as
  6A.1, §8).

## 11. Architecture boundary verification

- `theo_core` imports no `torch`/`transformers`/`qwen`/`huggingface_hub`;
  direction remains `theo-core ← theo-providers ← qwen3`.
- All accepted proposals passed grounding (7/7); unknown ids = 0. The provider
  proposes only; the runtime retains authority. No symbolic state was modified.
- The evaluator is a pure measurement layer over the unchanged parser; it cannot
  create Thoughts, Decisions, Beliefs, or Rules.

## 12. Scientific interpretation

Qwen3-0.6B did **not** demonstrate reliable semantic capability under the frozen
15-case probe:

- 8/15 outputs failed the constrained schema;
- 7/7 accepted proposals were grounded, but only 1/15 survived strict human
  review as a defensible novel, decision-relevant interpretation (b/002);
- the remainder were percept paraphrases (a/001) or echoes of stored beliefs
  (c/001, c/002);
- the distractor group (E) showed no reliable evidence selection (0.0).

The isolated b/002 success shows the *pattern* is learnable — observations →
unobserved-cause interpretation where the conclusion is not in the symbolic
knowledge — which is exactly the capability 6B should target. It is not,
however, evidence that the reference provider is usable.

## 13. Verdict — NEGATIVE / DIAGNOSTIC

> Qwen3-0.6B did not demonstrate reliable semantic capability under the frozen
> 15-case probe. One case (b/002) produced a defensible novel, decision-relevant
> abductive interpretation, but the overall experiment was dominated by malformed
> structured outputs, paraphrase/belief echoes, and failure on distractor-sensitive
> evidence selection. The isolated successful case is retained as a capability
> signal, not evidence of provider readiness.

**Architecture boundary: PASS. Semantic capability: NOT DEMONSTRATED.**

## 14. Why 6A.3 and 6A.4 are skipped (deferred, not deleted)

- **6A.3 (calibration)** ranks provider outputs to inform decision confidence.
  Ranking is meaningless when the provider does not reliably produce
  interpretable, evidence-anchored hypotheses: the reference SLM generated
  proposals in 7/15 cases, all grounded, but only 1/15 was defensibly
  decision-relevant. Calibration on this signal would not be a meaningful
  experiment.
- **6A.4 (combined)** layers the reference SLM's proposals and calibration into
  the runtime. Its stated prerequisite is 6A.1+6A.2+6A.3 individually passing;
  6A.2 and 6A.3 do not pass.
- The plans remain in `docs/implementation/v0.5-neural-interface-plan.md`,
  marked **SKIPPED / NOT JUSTIFIED**; they may be revisited if the THEO SLM
  (6B) demonstrates the missing capability.

## 15. Artifacts

- `theo-providers/semantic_probe_results.json` — raw per-case records, raw
  output verbatim, parse + assessment + pre-labels (frozen raw evidence)
- `theo-providers/semantic_probe_summary.json` — auto-preliminary summary
- `theo-providers/semantic_probe_labels.json` — human-reviewed labels (separate
  from raw results; E0–E6 schema validated by `apply_human_labels`)
- `theo-providers/semantic_probe_reviewed_summary.json` — human-reviewed summary
- `theo-providers/src/theo_providers/semantic_probe/evaluator.py` + `eval_semantic_probe.py`
  — frozen evaluator + runner
- `theo-core/docs/research/reference-slm/semantic-probe-v1/{semantic-probe-v1-spec.md,
  semantic-probe-v1-cases.json}` — frozen instrument (commit `5e42654`)
- `theo-core/docs/research/theo-slm-design-v0.md` — the 6B design this result feeds
- This report.
