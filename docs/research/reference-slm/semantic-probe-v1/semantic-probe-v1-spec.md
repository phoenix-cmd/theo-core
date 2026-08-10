# Semantic Probe v1 — Phase 6A.2 Specification

Status: **FROZEN** (v1)
Date: 2026-08-10
Owner: human + assistant (probe cases designed by human; this document is the frozen artifact)

This document defines the Phase 6A.2 Semantic Capability Probe. It is the frozen specification
for steps 1-3 of the 6A.2 implementation order:

1. Design the 15 probe cases — **DONE (this document)**
2. Freeze `semantic-probe-v1` — **DONE (this directory, committed)**
3. Define the E0-E6 evaluation rubric and Semantic Novelty definition — **DONE (this document)**

No model code, probe evaluator, or model runs happen until this spec is committed.

---

## 1. Purpose and question

Phase 6A.1 established that Qwen3-0.6B, wired as the ADR-0028 HypothesisProposalProvider,
produced **zero decision-relevant proposals** on the frozen 51-case corpus (architecture gate
PASS, cognitive gate NULL). The corpus is designed to stress the symbolic runtime, not the SLM:
almost every case's expected content is already derivable, and the SLM interface (propose a
candidate hypothesis) is a poor match for probing semantic capability.

6A.2 asks a different, narrower question:

> **Can a small SLM produce grounded semantic interpretation that the symbolic runtime
> cannot derive via its existing lexical and concept machinery?**

This is a **capability** question, not a decision-quality question. The probe is deliberately
engineered to give the SLM a fair chance to show semantic interpretation *if it has the
capability*. A negative result under this design is strong evidence the reference SLM has no
decision-relevant semantic capability; a positive result identifies a capability profile that
THEO SLM v0 could be designed against.

## 2. REPEAT vs DERIVE (the central distinction)

Every probe case is constructed so the target proposition is **not already present** in the
percept, beliefs, concept labels, or rules. The target may only be reached by combining or
interpreting available evidence. We classify each generated proposal as:

- **REPEAT** — the proposal restates the percept or an existing belief/concept/rule conclusion
  (possibly paraphrased). The symbolic runtime could produce it by token matching alone.
- **DERIVE / INTERPRET** — the proposal goes beyond any single stored item: it combines
  evidence, maps surface text to concepts not present in the text, or draws a semantic
  consequence that no rule fires. This is the capability being probed.

The probe evaluator labels each proposal E0-E6 (Section 6). E2/E3/E4 are REPEAT; E5/E6 are
DERIVE. **E5/E6 are the targets.**

## 3. Scope and design constraints

1. **Separate corpus.** The 15 probe cases live under `docs/research/reference-slm/semantic-probe-v1/`.
   The frozen 51-case corpus and its evidence are **not** touched (6A.1 freeze stands).
2. **No derivation by the runtime.** For every case: (a) no inference rule can fire to derive
   the target; (b) the target's key token is absent from the percept text and from every belief
   proposition; (c) the target cannot be reached by token matching to concept labels alone.
   Each case carries a `non_derivability_note` asserting this. A validator checks the structural
   invariants (Section 8).
3. **Input never contains the answer.** The percept text never contains the target concept's
   surface form. Concepts are supplied as *knowledge* (available vocabulary), so the SLM must
   *select* the right concept for an interpretation — selection is the capability.
4. **Grounding requirements are pre-specified.** Every case declares the evidence/concepts a
   valid E5/E6 proposal must reference, plus any designated distractor items that must NOT be
   referenced (Group E).
5. **Human ground truth.** Each case has a human-authored target candidate. Qwen 0.6B and any
   future THEO SLM are both scored against the same ground truth.
6. **Qwen is a measuring instrument, not a teacher.** Its outputs are never used as training
   targets. If its output is garbage/echoes, that is a *measurement*, not training material.
7. **Grouping mirrors capability areas, not corpus domains.** A=paraphrase, B=multi-fact
   interpretation, C=semantic contradiction, D=taxonomic interpretation, E=distractor
   resistance. 3 cases per group, 15 total.

## 4. The 15 cases

Legend: `K` = knowledge the case must supply (concepts/beliefs/edges); the percept is what the
SLM observes. `grounding` = the ids an E5/E6 proposal must reference. `distractors` = ids
available in knowledge that must NOT be referenced (Group E). Capability targets:
A paraphrase, B inference, C contradiction-driven inference, D taxonomy interpretation,
E evidence selection.

### Group A — Semantic paraphrase (meaning survives surface change)

| id | name | percept | concepts (K) | target | grounding | non-derivable because |
|----|------|---------|--------------|--------|-----------|------------------------|
| sp1://a/001 | container shattered | "The container shattered after hitting the floor." | container, floor, impact, break | "The container broke." | container, break | "shattered" and "hitting" match no label; runtime activates container+floor only |
| sp1://a/002 | towel damp | "The towel still feels damp after hanging all night." | towel, damp, moisture, dry | "The towel contains moisture." | towel, moisture | "moisture" absent; "damp" present only as an adjective echo trap |
| sp1://a/003 | offer turned down | "She turned down the job offer." | person, job, offer, refuse | "She refused the job offer." | refuse, offer, job | "turned down" matches no label; "refuse" requires the idiom mapping |

Trap propositions (must NOT count as E5): "The container shattered." (E3, restates percept),
"The towel is damp." (E3, verbatim-belief-style restatement), "She said no to the offer."
(E3 unless grounded in refuse).

### Group B — Multi-fact interpretation (combination implies; target concept absent)

| id | name | percept | concepts (K) | target | grounding | non-derivable because |
|----|------|---------|--------------|--------|-----------|------------------------|
| sp1://b/001 | rain | "Ravi picked up an umbrella before leaving. Dark clouds covered the sky. The street was already wet." | umbrella, rain, cloud, street, wet | "It is raining or has recently rained." | umbrella, cloud, wet | "rain" absent from percept; no rule maps umbrella+cloud+wet to rain |
| sp1://b/002 | power outage | "The lights went out. The microwave clock was blinking. The fridge hummed to life." | power, electricity, light, outage | "There is a power outage." | power, light, electricity | "outage" absent; no rule combines the three observations |
| sp1://b/003 | birthday | "A cake with candles was on the table. Balloons filled the room. Friends shouted 'Surprise!'" | cake, candle, birthday, party | "It is a birthday celebration." | cake, candle, party | "birthday" absent; no rule maps cake+candles+balloons to birthday |

Trap: "The street is wet." / "The lights went out." / "A cake with candles was on the table."
are all E3 (percept restatement), even though they name supplied concepts.

### Group C — Semantic contradiction (tension must be interpreted, not copied)

| id | name | percept | beliefs (K) | concepts (K) | target | grounding | non-derivable because |
|----|------|---------|-------------|--------------|--------|-----------|------------------------|
| sp1://c/001 | door blocked | "Nobody can enter through the door." | door_open 0.9, door_locked_outside 0.8 | door, enter, lock, accessible | "The door is inaccessible." | door_open, door_locked_outside, accessible, door | "inaccessible" absent; conflict machinery fires on neither belief alone |
| sp1://c/002 | light broken | "Someone just left the room." | light_on 0.9, room_dark 0.7 (conflict edge) | light, dark, broken, room | "The light is broken." | light_on, room_dark, broken | "broken" absent; percept is a distractor, not the cause |
| sp1://c/003 | fish in danger | "The tank has no water." | fish_alive 0.9, water_gone 0.8 | fish, water, danger, tank | "The fish is in danger." | fish_alive, water_gone, danger, fish | "danger" absent; combining alive+no-water requires interpretation |

Trap hypotheses: C1 "The door is closed." is an **E5-but-unsupported** (contradicts the
door_open belief; must fail a semantic-consistency check against beliefs). C2 "The room is dark
because someone left." cites the percept as cause — **E3** (restates the percept relation, no
new interpretation). C3 "The water is gone." — **E3** (copies the water_gone belief).

### Group D — Taxonomic interpretation (weak taxonomy; no belief to copy)

| id | name | percept | concepts + edges (K) | target | grounding | non-derivable because |
|----|------|---------|----------------------|--------|-----------|------------------------|
| sp1://d/001 | mammal | "The animal has fur, nurses its young, and has three toes." | animal, mammal, bird, reptile; mammal/bird/reptile is_a animal | "The animal is likely a mammal." | animal, mammal | "mammal" absent; no edge fur->mammal; is_a edges only link to animal |
| sp1://d/002 | bicycle | "The vehicle has two wheels, pedals, a chain, and a bell." | vehicle, bicycle, car, motorcycle; each is_a vehicle | "The vehicle is likely a bicycle." | vehicle, bicycle | "bicycle" absent; no feature->bicycle edges |
| sp1://d/003 | bird | "The creature has feathers, lays eggs, and builds nests in trees." | animal, bird, mammal, reptile; each is_a animal | "The creature is likely a bird." | animal, bird | "bird" absent; no feature->bird edges |

Trap: "A mammal is an animal." — **E4** (echoes the supplied is_a edge). Grounded but an echo.
Group D has **no initial beliefs**, so the only copyable content is concept labels and edges —
this isolates the taxonomy-interpretation capability.

### Group E — Distractor resistance (evidence selection; relevant vs irrelevant)

These cases deliberately expose extra, irrelevant concepts that appear in the percept. A valid
E5/E6 proposal must reference only the relevant evidence and must NOT reference any distractor
id. This is the strongest test of *evidence relevance*: it separates "can name concepts" from
"can select the evidence that actually supports the interpretation".

| id | name | percept | relevant concepts | distractor concepts | target | grounding (must) | must NOT cite |
|----|------|---------|-------------------|---------------------|--------|------------------|---------------|
| sp1://e/001 | recent rain | "The sky is blue. A red car is parked outside. The dog is sleeping in the sun. The road is wet. Pedestrians carry umbrellas." | weather, rain, wet, umbrella | sky, car, dog | "Recent rainfall is likely." | wet, umbrella | sky, car, dog |
| sp1://e/002 | cat in home | "The house has three bedrooms. The kitchen has a food bowl and a litter box. The windows are large." | home, cat, pet, food | bedroom, window | "A cat is likely kept in the home." | cat, food, home | bedroom, window |
| sp1://e/003 | soccer | "The building has blue walls. A group of people wear shin guards. A round black-and-white ball sits in a net. The carpet is grey." | sport, soccer, ball, goal | building, carpet | "They are likely playing soccer." | soccer, ball, goal | building, carpet |

Trap: citing any distractor id, or citing relevant ids plus distractor ids, downgrades the
evidence-relevance score even if the proposition is otherwise E5.

## 5. Semantic Novelty (SN)

A proposal has **Semantic Novelty** if and only if **all** of the following hold:

1. **Not a textual duplicate** — the proposal is not verbatim (normalized: lowercase, whitespace
   collapsed) equal to any stored percept sentence, belief proposition, concept label, or rule
   conclusion.
2. **Not a paraphrase** — the proposal is not a surface rewording of any single stored item
   (i.e., its semantic content is not already present under different surface form).
3. **Not a taxonomy restatement** — the proposal does not re-assert an existing is_a/related
   edge or a concept definition (e.g., "a mammal is an animal").
4. **Not a rule echo** — the proposal is not the conclusion of a fired rule (for probe cases,
   rules are absent, so this checks the general form).
5. **Requires combining or interpreting evidence** — the proposal integrates at least two
   distinct grounding items (or maps surface text to a non-surface concept) to reach content
   that no single item contains.
6. **All supporting claims are grounded** — every referent in the proposal resolves to a
   supplied knowledge id (belief id, concept id, or percept-derived evidence id) present in the
   case's grounding universe.

SN is a necessary condition for E5/E6. It is scored per proposal and aggregated per case.

## 6. Proposal classification (E0-E6)

Classify in the listed order (first match wins):

| class | name | definition |
|-------|------|------------|
| E0 | malformed | parser rejected: non-JSON, truncated, repeated fences, wrong schema, or no candidate text |
| E1 | ungrounded | well-formed but referencing >=1 unknown id, or empty/absent grounding; rejected by the grounding validator |
| E2 | exact repeat | passes grounding; normalized text equals a stored percept/belief/label |
| E3 | paraphrase | passes grounding; restates a single stored item or the percept without combining (SN criterion 5 fails) |
| E4 | taxonomy/rule echo | passes grounding; re-asserts an existing edge/definition (SN criterion 3 fails) |
| E5 | semantic interpretation | passes grounding; SN holds (combines/interprets evidence into content not already present) |
| E6 | decision-relevant semantic interpretation | E5 AND the interpretation bears on the case's `decision_target` (Section 7) |

Notes:

- E0/E1 are **unsupported**: the proposal never reaches semantic evaluation.
- E2/E3/E4 are **REPEAT**: content the symbolic runtime could produce.
- E5/E6 are **DERIVE**: the capability being probed. **The probe's success criterion is a
  non-zero rate of E5/E6 proposals with grounded references**, because the runtime cannot
  derive these.
- For C1, a proposal like "The door is closed." that passes SN but contradicts a stored belief
  is classified **E5-unsupported** (a sub-flag) and must not count toward the capability gate.
  The evaluator records `consistency_vs_beliefs` as a per-proposal field.

## 7. Evaluation funnel

For each case, per generated proposal:

```
SLM generation (Qwen3-0.6B, frozen eval config)
   -> HypothesisProposal (schema validation)          [E0 on failure]
   -> grounding validation vs case grounding universe [E1 on failure]
   -> SN check (Section 5)                            [E2/E3/E4 on failure]
   -> semantic-consistency check vs beliefs           [E5-unsupported flag]
   -> decision relevance vs decision_target           [E6 iff relevant]
   -> human review of every proposal
```

Per-case `decision_target` (human-authored): the question the interpretation must bear on.

- sp1://a/* — "did the event happen / what happened?" (E6 = answers the event question, e.g.,
  "The container broke.")
- sp1://b/002 — "what explains the observations?" (E6 = the outage explanation)
- sp1://c/001 — "can someone enter?" (E6 = no / door inaccessible)
- sp1://e/001 — "what weather is present?" (E6 = rain)

Every proposal is **manually inspected and classified** by a human reviewer. No auto-E6.
The pipeline may pre-label, but the human is authoritative.

## 8. Capability gate (6A.2 exit criteria)

A **capability profile** is produced: per group, per case, and overall.

Metrics (all rates over generated proposals that parse):

- grounded proposal rate = (E2+E3+E4+E5+E6) / total
- unsupported rate = (E0+E1) / total
- paraphrase rate = E3 / total
- rule/taxonomy echo rate = E4 / total
- **semantic novelty rate = (E5+E6) / total** — primary metric
- **decision relevance rate = E6 / total** — secondary metric
- evidence relevance score (Group E) = (relevant refs - distractor refs) / total refs per
  proposal, averaged per case

The **gate** for a positive 6A.2 result: **semantic novelty rate > 0** with at least one E5
proposal whose grounding, consistency, and relevance survive human review — per group.
A zero in every group means the reference SLM has no demonstrable semantic capability on this
probe, and 6A.3 calibration must not start (stop the reference-SLM track).

Decision-change counts are **not** a gate here. Decision relevance is measured for the
capability profile, not required for success. (That is 6A.4's job.)

## 9. Capability map -> THEO SLM v0

The per-group results become a **teacher/reference capability map**:

- Group A results describe the model's lexical/paraphrase generalization (can it map surface
  variation to stored concepts at all).
- Group B describes combination inference (can it integrate multiple observations into one
  non-stored conclusion).
- Group C describes contradiction-driven inference (can it read a semantic tension and produce
  an explanation).
- Group D describes taxonomic generalization (can it classify from features using weak
  taxonomy edges).
- Group E describes evidence selection (can it ignore irrelevant observables while citing the
  deciding evidence).

THEO SLM v0 targets only the capabilities where Qwen demonstrates a grounded, human-verified
E5/E6 result; everything else is out of scope for v0. If no group demonstrates E5/E6, the
reference-SLM track stops and there is no v0 capability target.

## 10. Ground-truth labeling protocol

1. Each case's `ground_truth.candidate` and `decision_target` are human-authored and frozen.
2. A proposal is compared to ground truth **semantically, not textually**: the human reviewer
   decides whether the proposal conveys the target meaning (E6) or a subset (E5).
3. Review is done without knowledge of which config produced the proposal (blind labeling).
4. Every proposal is labeled exactly once; disagreements are resolved by the human.
5. Ground truth is the standard for both Qwen 0.6B and any future THEO SLM run — no per-model
   truth.

## 11. Freeze rules

1. After commit, `semantic-probe-v1` (this spec + `semantic-probe-v1-cases.json`) is frozen:
   no case edits, no re-interpretation of the rubric, no metric changes, no re-runs.
2. Step 4 (probe evaluator) and step 5 (Qwen run) may only begin after this commit.
3. If the probe reveals a design flaw (e.g., a target derivable by the runtime), the flaw is
   reported and a new probe version (v2) is designed — never an in-place edit of v1.
4. Any findings update the capability map (Section 9) before any THEO SLM v0 design begins.
