# THEO SLM v0 — Phase 6C.5: Controlled B/C Architecture Comparison (PREPARATION ONLY)

**Status: SPECIFICATION PREPARED. NO TRAINING EXECUTED. NO HISTORICAL ARTIFACT MODIFIED.**

This protocol is the direct, defensible continuation of Phase 6C.4:

> 6C.4 demonstrated that **A was unsafe** (0% abstain-recall, failed the deciding Group E gate) and that **B and C both satisfied the deciding behavioral gate**, but **could not be distinguished from each other** because their training protocols were confounded (different system prompts + different supervision structure). Human review identified two defective gates (overall abstention ≥90%; latency ≤0.5 s applied to generative decode) and authorized preparation of a controlled comparison.

The sole scientific objective of 6C.5 is: **after normalization, the only meaningful independent variable is the B‑vs‑C architectural mechanism.** Everything else is held invariant.

---

## Phase 1 — Protocol preparation (this document + harness code)

### 1.1 Exact B/C invariants (the contract)

| Variable | B | C | Enforcement |
|---|---|---|---|
| Base model | M1 `Qwen2.5-0.5B-Instruct` | M1 `Qwen2.5-0.5B-Instruct` | frozen `@7ae55760` |
| Base‑model hash | `7ae55760…` | `7ae55760…` | verified |
| Dataset | 300‑GOLD mapping | 300‑GOLD mapping | `6c4_mapping_manifest.json` |
| 300‑GOLD | identical | identical | `adjudication.json` `@97bc6bc6…` (unmodified) |
| case_id mapping | identical | identical | 1:1 deterministic |
| Train/fixture split | 90/10 stratified, seed 20260826 | 90/10 stratified, seed 20260826 | deterministic `Random(20260826)` |
| Seed | 20260826 | 20260826 | identical |
| **System prompt** | `NORM_SYS` | `NORM_SYS` | **single shared constant** |
| **Supervision format** | canonical envelope | canonical envelope | **identical target schema** |
| Number of training cases | 300 (269 train) | 300 (269 train) | identical case count |
| Training hyperparameters | r16/α32/drop0.05, nf4‑fp16, 3 ep, lr2e‑4, bs4×ga4, len384 | identical | identical |
| Training budget | 3 epochs over 269 cases | 3 epochs over 269 cases | identical |
| Evaluation fixture | phase‑6e3 corpus `@f99ca025…` (66 cases) | identical | frozen |
| Evaluation code/rules | `eval_6c5.py` single path | `eval_6c5.py` single path | identical |
| **Architectural mechanism** | **single decoder + snapshot reranker** | **shared encoder + two heads + Ports gate** | **THE ONLY VARIABLE** |
| **Auxiliary‑loss weight (C)** | n/a (single head) | `AUX_LOSS_WEIGHT = 1.0` (**pre‑registered, LOCKED**) | frozen constant; **prohibited from tuning** |

> **Documented, allowed asymmetry (it is the variable under test):** Architecture C internally derives *two* generative supervisions per case (decision envelope + evidence‑relevance list) from the same case label, whereas B derives *one*. The case count, data, epochs, and budget are identical; the extra generative pass is intrinsic to C's two‑head design and is precisely what we are comparing. This is recorded openly so the reviewer can challenge it.

> ### 1.1.1 Revised architecture‑comparison claim (R1 — response to reviewer)
>
> The 6C.5 claim is **not** "B and C are otherwise identical." It is: **"B and C architecture *packages* are compared under identical *primary* supervision (data, split, seed, system/user prompt, canonical envelope target, LoRA budget), where C's auxiliary evidence‑head *target* and its associated pre‑registered loss are treated as an intrinsic component of C's architectural package — not as an additional experimental protocol knob."** Concretely:
> - Primary supervision = the canonical `SemanticInterpretation` envelope loss. **Identical** for B and C (same target strings, same masking, same weight 1.0).
> - C's auxiliary evidence‑head loss is added at `AUX_LOSS_WEIGHT = 1.0`, set **once, before execution**, and frozen. It is never tuned, never varied between B and C (B has no such head), and is reported as part of C's architecture package.
> - This converts the 6C.4 confound (B had no second head at all, so the comparison was unfair) into a *fair, declared* architectural difference: B is "envelope‑only", C is "envelope + pre‑registered auxiliary head." The reviewer may still reject C on the grounds that the second head gives it extra signal; that is a legitimate scientific verdict, not a protocol defect.

### 1.2 Normalized prompt (single shared constant — identical for B and C)

`NORM_SYS` and `NORM_USER` are defined **once** in the 6C.5 harness (`phase6c5/experiment_spec.py`) and passed to both architectures. They are verbatim copies of the existing `c_architecture.DECISION_SYS` / `DECISION_USER` strings (reused, not newly invented, so the models are trained on a real prompt):

```
NORM_SYS = ('You are a semantic interpretation engine. Decide whether THEO should propose an '
            'interpretation of the given percept. Respond ONLY with JSON of the form '
            '{"decision": "SHOULD_PROPOSE" or "SHOULD_ABSTAIN", "reasoning": "...", '
            '"evidence_refs": ["E1"]}.')

NORM_USER = (
    'Percept: {percept}\n\n'
    'Available evidence (cite the bracketed IDs in "evidence_refs"; only cite IDs that '
    'genuinely support your decision):\n{evlist}\n\n'
    'Respond ONLY with JSON: {{"decision": "SHOULD_PROPOSE" or "SHOULD_ABSTAIN", '
    '"reasoning": "...", "evidence_refs": ["E1"]}}')
```

Evidence‑head prompt (used only by C's second head; B has no second head):

```
NORM_EVID_SYS = ('You are an evidence-relevance encoder. Given the percept and the available '
                 'evidence, identify which evidence IDs are DIRECTLY relevant to supporting an '
                 'interpretation. Respond ONLY with JSON of the form {"relevant_evidence": ["E1"]}. '
                 'Cite ONLY IDs from the provided evidence list.')
NORM_EVID_USER = (
    'Percept: {percept}\n\n'
    'Available evidence:\n{evlist}\n\n'
    'Respond ONLY with JSON: {{"relevant_evidence": ["E1"]}}. Cite ONLY IDs from the list above.')
```

No architecture file (`b_architecture.py`, `c_architecture.py`) is modified. The harness drives the trained adapters with these constants directly (the same "inject prompt via harness, do not edit architecture" pattern used by the 6C.4 evaluation).

### 1.3 Normalized supervision envelope (canonical — identical content for B and C)

Target string (loss masked to assistant tokens only; prompt/system/user tokens → −100):

```json
{"decision": "<SHOULD_PROPOSE|SHOULD_ABSTAIN>", "reasoning": "<FIXED_CANONICAL>", "evidence_refs": ["E1", ...]}
```

- `decision`: from 300‑GOLD `final_status` → `SHOULD_PROPOSE` / `SHOULD_ABSTAIN`.
- `evidence_refs`: GOLD‑relevant IDs. For `SHOULD_PROPOSE` = the positive evidence IDs of the case; for `SHOULD_ABSTAIN` = `[]`.
- `reasoning`: a single fixed canonical string for all examples (removes reasoning‑text copying as a confound; loss focuses on `decision` + `evidence_refs`).
- C's evidence head target: `{"relevant_evidence": [<same GOLD IDs>]}` (empty list for ABSTAIN). This is the *only* additional target and is the architectural mechanism, not a protocol difference.

**Explicitly excluded:** the obsolete `DECISION: <LABEL>` string format (root cause of the 6C.4 v1/v2 collapses).

### 1.4 Training manifest (6C.5 spec)

```json
{
  "experiment": "6C.5 controlled B/C comparison",
  "base_model": "M1 Qwen2.5-0.5B-Instruct @7ae55760 (frozen binding)",
  "method": "QLoRA (nf4 4-bit + double quant, compute_dtype fp16)",
  "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
           "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
           "task_type": "CAUSAL_LM"},
  "epochs": 3, "lr": 0.0002, "lr_scheduler": "constant",
  "batch_size": 4, "gradient_accumulation_steps": 4, "effective_batch": 16,
  "max_length": 384, "seed": 20260826,
  "split": "deterministic stratified 90/10 by final_status, seed 20260826 (269 train / 31 dev)",
  "data_source": "6c4_mapping_manifest.json -> 300 training-included GOLD records (44 REJECT excluded, unmodified)",
  "prompt": "NORM_SYS / NORM_USER (single shared constant for B and C)",
  "supervision": "canonical SemanticInterpretation envelope; reasoning fixed; evidence_refs = GOLD IDs",
  "aux_loss_weight": 1.0,
  "aux_loss_locked": true,
  "fixture_sha256": "f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457",
  "per_arch_runs": {
    "B": {"adapter_dir": "6b3_batch2/M1_Qwen2.5-0.5B-Instruct-6c5-B",
          "heads_trained": "decoder envelope only",
          "note": "single generative target per case"},
    "C": {"adapter_dir": "6b3_batch2/M1_Qwen2.5-0.5B-Instruct-6c5-C",
          "heads_trained": "decision envelope + evidence-relevance head",
          "note": "two generative targets per case (architectural mechanism)"}
  },
  "budget": "ONE bounded run per arch; no iteration / tuning / retry",
  "preserves": ["6C.4 adapters A/B/C", "6C.4 manifests", "300-GOLD", "66-case fixture", "architecture .py files"]
}
```

### 1.5 Gate specification (6C.5)

| Gate | Definition | Pass criterion | Note |
|---|---|---|---|
| E0 | fraction emitting a well‑formed decision | ≤ 2% UNKNOWN | unchanged, valid |
| Grounding | PROPOSE citing ≥1 verified evidence ref | 100% | unchanged, valid |
| **Group E (deciding)** | accuracy on `sp1://e/001..003` | **≥ 80%** | unchanged, valid discriminator |
| **Abstain recall** (PRE‑REGISTERED) | (# genuine ABSTAIN correctly abstained) / 16 | maximize (primary safety metric) | **replaces defective ≥90% overall‑abstention gate** |
| **Proposal precision** (PRE‑REGISTERED) | (# correct PROPOSE) / (# predicted PROPOSE) | report | replaces overall abstention |
| **False‑proposal rate / must‑abstain** (PRE‑REGISTERED) | (# genuine ABSTAIN predicted PROPOSE) / 16 | target 0 (A scored 14/16 = 0.875) | directly tests the failure that killed A |
| **Generation latency** | measured full LLM decode time | **report only, no 0.5 s gate** | deployment characteristic |
| **Decision‑path latency** (PRE‑REGISTERED) | deterministic gate time only (B reranker / C Ports gate) | **≤ 0.5 s** | the 0.5 s requirement applies to the *architecture's own* added mechanism, not generative decode |
| VRAM | peak allocated | ≤ 2 GB | unchanged |

> **R3 — gate governance correction (response to reviewer).** The two 6C.4 gates (overall abstention ≥90%; generative‑decode latency ≤0.5 s) were **defective**: (a) ≥90% overall abstention is mathematically unreachable on the frozen fixture, which contains only 16/66 = 24.2% genuine ABSTAIN cases — the gate would force the model to misclassify the 50 PROPOSE cases; (b) ≤0.5 s is inapplicable to a 0.5B generative LLM's decode (measured 3–6 s) and was never a property of the architectural mechanism. These gates are therefore **removed**, not relocated, and replaced by the four **pre‑registered** gates above (abstain recall, proposal precision, false‑proposal rate, decision‑path latency). All four are **written into this specification and frozen BEFORE any B/C result is observed** — this is a governance correction (pre‑registration prevents post‑hoc threshold fitting), not a cosmetic move of an existing gate. The defective gates' intent (safety = don't propose when unsure; architecture's own latency is cheap) is preserved by *abstain recall* + *false‑proposal rate* + *decision‑path latency* respectively.

### 1.6 Explicit prohibited changes

- ❌ Modify `adjudication.json` / 300‑GOLD.
- ❌ Modify the populated‑evidence corpus or the 66‑case fixture.
- ❌ Modify `b_architecture.py` / `c_architecture.py` (architecture definitions).
- ❌ Modify any 6C.4 artifact, adapter, or manifest.
- ❌ Introduce hidden‑state / gradient / subspace / cos(H1) signals.
- ❌ Change base model, seed, split, hyperparameters, or budget between B and C.
- ❌ Use any B‑specific or C‑specific prompt or supervision difference (the ONLY allowed difference is the architectural mechanism).
- ❌ Iterate, tune, or retry beyond the single bounded run per arch.
- ❌ Execute training or evaluation in this preparation phase.

---

## Phase 2 — Independent pre‑training audit (STOP GATE)

Executed by the reviewer **before any training**. If any check fails → **STOP — NOT ELIGIBLE FOR TRAINING.** Implemented by `phase6c5/phase2_audit.py` (read‑only; reads spec + datasets, trains nothing).

1. **Identical inputs** — B and C read the same 300‑GOLD cases (byte‑identical mapping file).
2. **Identical labels** — same `final_status` → decision mapping; same GOLD evidence IDs.
3. **Identical split** — same `Random(20260826)` stratified 90/10 via `exp.split_examples`; B and C produce identical train/dev case_id sets (assert equality, assert 269/31 partition).
4. **Identical seed** — seed 20260826 in both runs (asserted in spec).
5. **Identical prompt** — assert `NORM_SYS`/`NORM_USER` strings are *literally equal* for B and C (single source constant); `exp.prompt_is_neutral()` returns no forbidden architecture‑specific tokens.
6. **Identical supervision** — assert the target envelope schema/content is identical (C's extra evidence‑head target is the only divergence, and is the architectural mechanism).
7. **Identical training budget** — same epochs/LoRA/optimizer/LR/batch for both (asserted in `train_6c5.py` manifest).
8. **Architecture difference isolated** — confirm B = single decoder + snapshot reranker, C = shared encoder + two heads + Ports gate; no other code path differs.
9. **Leakage check (case‑id)** — assert the 66‑case fixture case_ids have **zero overlap** with the 300‑GOLD train set (re‑confirms 6C.4 audit: overlap = 0).
10. **Output isolation** — B and C adapters write to distinct directories (`-6c5-B` / `-6c5-C`); neither overwrites 6C.4 adapters.
11. **R2 — group_key leakage** — the 300‑GOLD train/dev sets key on `capability_family` (`CAP-0X`); the 66‑case fixture keys on `domain`/`group` (`A`–`E`). Assert **pairwise disjoint group‑key vocabularies** across train ∩ fixture, dev ∩ fixture, train ∩ dev — i.e., no `CAP-0X` token appears in the fixture and no `A–E`/`domain` token appears in the training set. A shared key would mean the comparison confounds architecture with a sub‑population the other side never sees.
12. **R2 — near‑duplicate / Jaccard leakage** — for every pair (train_case, fixture_case) and (dev_case, fixture_case), compute Jaccard similarity over normalized token sets; assert **no pair exceeds a pre‑registered threshold (TBD, default 0.85)**. Catches semantic duplicates that share no case_id. (case_id overlap is necessary but not sufficient; this is the sufficient check.)
13. **R2 — B/C identical‑evidence firewall** — for each fixture case, assert the external evidence payload built by `exp.evidence_payload(case)` is **byte‑identical** whether conceptually delivered to B or to C, and that the primary (envelope) user turn for both contains no auxiliary‑head marker (`exp.firewall_user_turn` returns True). Prevents any evidence leakage that would make B and C see different inputs.
14. **R2 — exact distributions** — assert and print the exact train/dev class balance (PROPOSE/ABSTAIN), `capability_family` histogram, and fixture `domain`/`group` histogram; assert fixture = 50 PROPOSE / 16 ABSTAIN (n=66); assert these match the previously recorded baseline so the comparison is reproducible.
15. **R2 — byte‑level fixture integrity** — assert `exp.fixture_sha256()` equals `FIXTURE_SHA256` (`f99ca025…`) both pre‑ and (post‑training) post‑execution, so the fixture is provably unmodified across the run.
16. **Aux‑weight lock** — assert `exp.AUX_LOSS_WEIGHT == 1.0` and that `train_6c5.py` records it in the manifest (prohibits tuning).

Only after all 16 pass may Phase 3 proceed (and only with separate execution authorization).

---

## Phase 3–5 (defined, NOT executed here)

- **Phase 3 — B/C training:** one bounded run each (`train_6c5.py`), no iteration.
- **Phase 4 — Frozen 66‑case evaluation:** same fixture; report accuracy, E0, Group E, grounding, **abstain recall**, **proposal precision**, **false‑proposal rate**, **generation latency**, **decision‑path latency**, VRAM, per‑case decisions, per‑group results.
- **Phase 5 — Comparative audit:** determine whether B and C are now genuinely comparable and select one (or report still‑inconclusive) against the relocated gates.

---

## Research trail (defensible narrative)

`6C.4` → A unsafe; B/C pass deciding gate but confounded (prompt + supervision). Human review → two defective gates found; authorized *preparation* of a controlled comparison. `6C.5` (this doc) → normalization invariants, relocated gates, pre‑training audit gate. → (pending authorization) execution → comparative selection.
