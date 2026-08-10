# Phase 6A.1 Report — Qwen3-0.6B as Reference Hypothesis-Proposal Provider

**Status:** DRAFT FOR REVIEW (unapproved; 6A.2 gated on human decision)
**Date:** 2026-08-10
**Phase:** 6A.1 (controlled reference-SLM experiment, ADR-0028)
**Corpus:** frozen 51-case blind benchmark (unchanged; no edits, removals, or weak assertions)
**Prompt:** `qwen3-hypothesis-v1` (single documented prompt; no tuning this phase)
**Baseline:** v0.4.1 runtime with no provider (frozen, unmodified)

---

## 1. Objective

Measure, under blind conditions, whether an external 0.6B SLM (`Qwen/Qwen3-0.6B`,
pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`) operating strictly as a
`HypothesisProposalProvider` (ADR-0028) adds any decision or reasoning value to the
symbolic runtime, or whether it can be rolled back without observable change.

The SLM proposes only; it never decides. Authority stays with the runtime.

## 2. Method

- Baseline pass over the 51-case corpus with `BenchmarkHarness.run_all` (no provider).
- Per-case diagnostic loop running the Qwen proposal config through
  `execute_cycle` (single model call per case) capturing: decision, golden match,
  grounded proposals, fired rules, confidence delta.
- Replay pass with a fresh `ProviderCoordinator` to verify decision determinism.
- All proposals must pass `verify_grounding` (≥1 valid `GroundingSnapshot` id) or be
  rejected; no `grounded=False` path exists.
- Determinism: greedy decoding, `do_sample=False`, `temperature=0.0`, seed 0.
- Strict boundary: Qwen code lives only in `theo-providers/`; no `theo_core` module
  imports `torch`/`transformers`; direction is `theo-core <- theo-providers <- qwen3`.
- Driver: `theo-providers/eval_phase6a1_qwen.py` (ruff+mypy clean). Raw log
  `eval_phase6a1_qwen.log`; machine-readable `eval_phase6a1_qwen.json`.

## 3. Results — aggregate

| Metric | Value |
|---|---|
| Total cases | 51 |
| Baseline pass rate | 51/51 |
| Qwen-config pass rate | 51/51 |
| **Decision changes (baseline → Qwen)** | **0** |
| Replay deterministic (fresh coordinator) | True |
| Cases with ≥1 proposal | 20/51 |
| Cases with ≥1 grounded proposal | 20/51 |
| Proposals generated | 20 |
| Proposals grounded | 20 |
| Proposals rejected (ungrounded/malformed) | 0 |
| Proposals duplicate of existing hypothesis | 0 |
| Proposals novel (not already a symbolic hypothesis) | 20 |
| Rule-conclusion proposals (corrected) | 2 |
| Rule-conclusion proposals, fired rule | 2 |
| Mean symbolic hypotheses/case | 2.25 |
| Mean fired rules/case | 0.47 |
| Mean confidence delta (baseline → Qwen) | +0.0000 |

**Headline:** the reference SLM changed **zero** decisions, produced **zero** confidence
deltas, and was replay-deterministic. All 51 cases passed both with and without it.
By the decision-impact criterion the provider is behaviorally invisible.

### Per-domain proposal summary

| Domain | cases | gen | grounded | rejected | novel | dup | rule-concl |
|---|---|---|---|---|---|---|---|
| ambiguity | 8 | 5 | 5 | 0 | 5 | 0 | 0 |
| causal_reasoning | 10 | 4 | 4 | 0 | 4 | 0 | 1 |
| commonsense | 7 | 2 | 2 | 0 | 2 | 0 | 0 |
| contradiction | 9 | 1 | 1 | 0 | 1 | 0 | 0 |
| taxonomy | 9 | 5 | 5 | 0 | 5 | 0 | 0 |
| uncertainty | 8 | 3 | 3 | 0 | 3 | 0 | 1 |

## 4. Results — 51-case blind matrix

Columns: `sym` = symbolic hypotheses fired, `fr` = fired rules, `gen/gr/rej/nov/dup` =
generated/grounded/rejected/novel/duplicate proposals, `rcf` = rule-conclusion fired,
`dc` = decision changed, `bc/qc` = baseline/Qwen confidence.

| case | domain | sym | fr | gen | gr | rej | nov | dup | rcf | dc | bc | qc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bm://ambiguity/001 | ambiguity | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://ambiguity/002 | ambiguity | 3 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://ambiguity/003 | ambiguity | 4 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://ambiguity/004 | ambiguity | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://ambiguity/005 | ambiguity | 3 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://ambiguity/006 | ambiguity | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://ambiguity/007 | ambiguity | 3 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://ambiguity/008 | ambiguity | 3 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/001 | causal_reasoning | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/002 | causal_reasoning | 3 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/003 | causal_reasoning | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/004 | causal_reasoning | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/005 | causal_reasoning | 4 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/006 | causal_reasoning | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/007 | causal_reasoning | 2 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/008 | causal_reasoning | 2 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/009 | causal_reasoning | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://causal_reasoning/010 | causal_reasoning | 2 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | False | 1.0 | 1.0 |
| bm://commonsense/001 | commonsense | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://commonsense/002 | commonsense | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://commonsense/003 | commonsense | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://commonsense/004 | commonsense | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://commonsense/005 | commonsense | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://commonsense/006 | commonsense | 3 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://commonsense/007 | commonsense | 2 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://contradiction/001 | contradiction | 2 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://contradiction/002 | contradiction | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://contradiction/003 | contradiction | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://contradiction/004 | contradiction | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://contradiction/005 | contradiction | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://contradiction/006 | contradiction | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://contradiction/007 | contradiction | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://contradiction/008 | contradiction | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://contradiction/009 | contradiction | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://taxonomy/001 | taxonomy | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://taxonomy/002 | taxonomy | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://taxonomy/003 | taxonomy | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://taxonomy/004 | taxonomy | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://taxonomy/005 | taxonomy | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://taxonomy/006 | taxonomy | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://taxonomy/007 | taxonomy | 3 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | False | 1.0 | 1.0 |
| bm://taxonomy/008 | taxonomy | 5 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://taxonomy/009 | taxonomy | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | False | 1.0 | 1.0 |
| bm://uncertainty/001 | uncertainty | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | False | 0.8 | 0.8 |
| bm://uncertainty/002 | uncertainty | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | False | 0.7 | 0.7 |
| bm://uncertainty/003 | uncertainty | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | False | 0.8 | 0.8 |
| bm://uncertainty/004 | uncertainty | 2 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | False | 0.7 | 0.7 |
| bm://uncertainty/005 | uncertainty | 2 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | False | 1.0 | 1.0 |
| bm://uncertainty/006 | uncertainty | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | False | 0.9 | 0.9 |
| bm://uncertainty/007 | uncertainty | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | False | 0.7 | 0.7 |
| bm://uncertainty/008 | uncertainty | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | False | 0.6 | 0.6 |

## 5. Proposal inventory and per-proposal valuation

All 20 proposals, with referenced ids and an honest content valuation. Valuation
classes (defined in §6): **A** premise/percept restatement, **B** taxonomy
subsumption restatement, **C** mild inferential attribution, **D** fired-rule
conclusion echo, **E** degenerate.

| case | proposition | refs | relation (stored as rationale) | class |
|---|---|---|---|---|
| bm://ambiguity/002 | "the door is closed" | belief://b_door_closed | "one short phrase" | A |
| bm://ambiguity/003 | "it will rain today" | belief://b_weather_rain | "one short phrase" | A |
| bm://ambiguity/005 | "the bank closes today" | percept://c1b1ae17 | "one short phrase" | A |
| bm://ambiguity/007 | "the date is set for the meeting" | belief://b_date_meeting | "one short phrase" | A |
| bm://ambiguity/008 | "the light is bright and easy to carry" | belief://b_light_bright, belief://b_light_weight | "both" | A |
| bm://causal_reasoning/002 | "Smoke is rising from the chimney" | belief://percept/6dc4a7c6 | "one short phrase" | A |
| bm://causal_reasoning/007 | "The steam is rising from the kettle due to fire being nearby." | belief://c_steam, percept://c8e3bf97 | "causal" | C |
| bm://causal_reasoning/008 | "The road is icy and hazardous." | belief://c_road, percept://87d4ef0c | "one short phrase" | C |
| bm://causal_reasoning/010 | "snow is falling and the roads are slippery" | belief://c_snow, belief://inf/causal/snow_slippery/1 | "and" | D |
| bm://commonsense/006 | "is an apple healthy" | belief://percept/db2c9754 | "one short phrase" | E |
| bm://commonsense/007 | "bats eat insects" | belief://c_bats_mammals, percept/f614a16f | "one short phrase" | A |
| bm://contradiction/001 | "The sky is blue" | belief://b_sky_blue | "one short phrase" | A |
| bm://taxonomy/001 | "A mammal is an animal that has four legs and a fur coat." | concept://animal, concept://mammal | "one short phrase" | B |
| bm://taxonomy/002 | "A car is a vehicle." | concept://car, concept://vehicle | "one short phrase" | B |
| bm://taxonomy/003 | "A single plant is a single organism." | concept://organism, concept://plant | "one short phrase" | B |
| bm://taxonomy/004 | "fruit is a type of food" | concept://food, concept://fruit | "one short phrase" | B |
| bm://taxonomy/007 | "a canine is a dog" | belief://c_canine, belief://c_dog | "one short phrase" | B |
| bm://uncertainty/004 | "signal detected" | belief://percept/91c169e9 | "one short phrase" | A |
| bm://uncertainty/005 | "Diversify your portfolio" | belief://c_market, belief://inf/uncertainty/volatile_diversify/1 | "one short phrase" | D |
| bm://uncertainty/007 | "the fog is visible" | belief://percept/a7529822 | "one short phrase" | A |

Notes:

- **commonsense/007** references `percept/f614a16f` — a non-URI-form id without `://`.
  It was accepted because `build_grounding` places legacy evidence ids (from belief
  support traces) in the grounding id sets and the parser accepts any grounding
  member. Honest implication: grounding membership, not URI shape, is the validity
  check. Flag for tightening in any future prompt/parser revision.
- All 20 proposals carry the literal prompt-example placeholder `"one short phrase"`
  as their relation (2 use `"both"`/`"causal"`, 1 uses `"and"`). The parser stores it
  as `rationale`. This is **prompt mimicry, not semantic rationale**, and is scored as
  such — no proposal provides a meaningful relation/explanation.

## 6. Usefulness classification

Blind funnel per the exit gate (`generated → grounded → accepted → novel →
decision-relevant → decision-improving`):

```
generated              20/20
grounded               20/20   (verify_grounding pass; 0 rejected)
accepted                n/a     (proposal-only design; no auto-acceptance path this phase)
novel                  20/20   (not an existing symbolic hypothesis — set membership only)
decision-relevant       0/20   (no case showed a proposal changing or informing the decision)
decision-improving      0/20   (confidence delta +0.0000 everywhere; no acceptance path)
```

**Verdict:** "novel" here is a weak formal property (the proposition string was not
already a committed hypothesis). It does **not** mean decision-useful. Under the
strong criteria, **0/20 proposals were decision-relevant and 0/20 decision-improving.**

### Honest premise-echo analysis

Of the 20 grounded proposals, an honest content valuation yields:

| class | meaning | count | cases |
|---|---|---|---|
| A | premise/percept restatement — repeats an input belief or percept with no new propositional content | **10** | ambiguity/002,003,005,007,008; causal/002; commonsense/007; contradiction/001; uncertainty/004,007 |
| B | taxonomy subsumption restatement — re-asserts an is-a relationship already encoded in the concept definitions | **5** | taxonomy/001,002,003,004,007 |
| C | mild inferential attribution — adds a near-transparent inference (icy→hazardous, steam→fire) | **2** | causal/007,008 |
| D | fired-rule conclusion echo — restates a conclusion already committed via the fired rule's derived belief | **2** | causal/010, uncertainty/005 |
| E | degenerate — question form instead of an assertable proposition | **1** | commonsense/006 |

**Bottom line: no proposal demonstrated non-derivable, decision-relevant semantic
content under the current corpus and proposal interface.** 18/20 proposals (A + B + D)
contain no propositional content beyond what the symbolic runtime already commits; the
remaining 2 (class C) are weak near-tautological inferences. None proposes a genuinely
novel interpretation — exactly the property a useful hypothesis proposer should
provide, and none was observed under the frozen corpus and prompt.

## 7. Rule engagement

Counting both direct `rule://…` references and the *derived* `belief://inf/…` ids
through which Qwen reaches fired-rule conclusions, exactly **2/20** proposals show rule
engagement, both on fired rules:

- `bm://causal_reasoning/010` → `belief://inf/causal/snow_slippery/1` (rule
  `rule://causal/snow_slippery`, fired)
- `bm://uncertainty/005` → `belief://inf/uncertainty/volatile_diversify/1` (rule
  `rule://uncertainty/volatile_diversify`, fired)

These are the only cases where the SLM referenced a derived inference rather than raw
input. Both are **confirmation echoes**: the SLM cites a derived belief the runtime
already committed. This is the sole evidence of rule engagement in the phase, and it
adds no new content.

**Correction note:** the raw run log (`eval_phase6a1_qwen.log`) reports
"rule-conclusion 0" because the original eval metric counted `rule://…` references
only. The JSON dump was corrected (2 → both fired) to count the derived
`belief://inf/…` provenance. This report supersedes the log's aggregate line.

## 8. Latency and determinism (as data)

Measured on the real model (GTX 1650, CUDA fp16) via `scripts/qwen3_timing_probe.py`
(two identical calls):

| call | wall time | new tokens | tok/s | hit 512 cap |
|---|---|---|---|---|
| 1 | 51.1 s | 512 | 10.02 | True |
| 2 | 50.3 s | 512 | 10.18 | True |

- Both completions were byte-identical (deterministic; repeated-` ```json `-fence
  degeneration observed on the probe prompt, and a non-JSON completion → 0 proposals
  for one real corpus case).
- Full 51-case diagnostic run ≈ 90 min wall (≈ 1 generate call per case at ~50 s).
- **Strict-parameter honesty:** transformers 5.14.1 logs that the `temperature` and
  `top_k` generation flags "are not valid and may be ignored" under the unified
  generation API. Because decoding is greedy (`do_sample=False`), determinism does
  not depend on them and was verified empirically (identical outputs across calls and
  across fresh-coordinator replay). The strict parameter record (seed 0,
  temperature 0.0, `do_sample=False`, `max_new_tokens=512`) is documented in
  `model-info.json`; the ignored-flag warning is noted here so it is not mistaken for
  evidence of nondeterminism.

## 9. Boundary and isolation verification

- No `proposal://` provenance appeared in any belief/decision in isolation tests or on
  the real model (no belief cites an SLM-generated id).
- `theo_core` imports no `torch`/`transformers`/`qwen`/`huggingface_hub`; the qwen3
  modules import only `theo_core.models.ports`. Direction is
  `theo-core ← theo-providers ← qwen3`.
- Provider execution DTOs carry `provider_name="qwen3_hypothesis"`,
  `provider_version=0.1.0`, model hash, and tokenizer hash.
- Model/tokenizer hashes are computed from the Hub manifest
  (`lfs.sha256`/`blob_id` at the pinned revision), never by loading weights twice:
  `model_hash a80410e02451ae27828f89ccc87eb55362a14b561fec1937b24a599c9849cca8`,
  `tokenizer_hash 92ba2f610797f9ac063d2fcd678beb23eced9256c0144ab80c593b7ece91a743`.

## 10. Rollback hypothesis

**Hypothesis (null form):** removing the Qwen provider leaves runtime behavior
unchanged.

- **Decision changes:** 0/51.
- **Replay:** fresh coordinator reproduces identical decisions (True).
- **Pass rates:** baseline 51/51 == Qwen 51/51.
- **Confidence deltas:** +0.0000 for every case.

**Status: strongly supported.** Under the decision-impact criterion the provider is
behaviorally invisible and can be rolled back without regression.

**Converse (the honest negative result):** this phase produced **no evidence** that the
reference SLM adds decision value. All proposals were premise echoes (10),
taxonomy restatements (5), weak inferences (2), rule-confirmation echoes (2), or
degenerate (1). The only path that could justify continued SLM investment — proposals
that change or meaningfully inform decisions — had zero occurrences.

**Recommendation for the human gate (6A.2 Semantic Capability Probe):** do not proceed
to calibration (6A.3) on this evidence. If the SLM path continues, the prompt (banned
this phase) and the acceptance gating design are the two levers; both need a hypothesis
that a reference SLM can produce content the runtime cannot derive itself, with a probe
case that exhibits it.

## 11. Artifacts

- `theo-providers/eval_phase6a1_qwen.py` — driver (ruff+mypy clean)
- `theo-providers/eval_phase6a1_qwen.log` — incremental run log
- `theo-providers/eval_phase6a1_qwen.json` — 51-case per-case diagnostics (corrected)
- `theo-providers/qwen_proposals_sample.json` — 20 proposals with refs/relations
- `theo-providers/model-info.json` — model/revision/hash/generation provenance
- `theo-providers/scripts/qwen3_smoke_test.py`, `scripts/qwen3_timing_probe.py`
- `theo-core/docs/research/reference-slm/qwen3-0.6b/prompt-v1.txt` — archived prompt
  (sync-tested against `qwen3-hypothesis-v1`)
- `theo-core/docs/research/reference-slm/qwen3-0.6b/phase-6a1-report.md` — this report

## 12. Open items for review

1. Approve the corrected rule-conclusion metric semantics (`rule://` + derived
   `belief://inf/`).
2. Decide whether the non-URI `percept/f614a16f` grounding member warrants a parser
   tightening (boundary note, §5).
3. Gate 6A.2 (calibration) pending the recommendation in §10.
