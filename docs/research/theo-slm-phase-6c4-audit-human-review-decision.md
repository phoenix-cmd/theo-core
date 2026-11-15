# THEO SLM v0 — Phase 6C.4 Supervised Results: Numerical Audit & Human‑Review Decision

**Status: FROZEN.** All artifacts below are immutable (SHA‑256 pinned). No further training or modification was performed in producing this document.

## 1. Frozen-state inventory (pinned hashes)

| Artifact | SHA‑256 (or reference) |
|---|---|
| `adjudication.json` (300‑GOLD + 44 REJECT) | `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4` |
| populated‑evidence corpus (66‑case fixture) | `f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457` |
| Adapter A | `d5a920676d67ca395ac35bd7249622b4cd58a11db8350f6e114525bbf8887692` |
| Adapter B | `24333a7569b34270ef927650e9926f10fe82059a1ef35f19a4b7b967a112ad14` |
| Adapter C | `73828c5903aeb05c5457976303e39fa6679580f2c9f55b2b46f69de74b65e95d` |
| `6c4_supervised_evaluation_results.json` | `bb2ea818e87beb97c52bcc596d31cbe28940e075064a382b30460a20900d29ec` |
| Training manifests A/B/C | `6c4_training_manifest_{A,B,C}.json` |
| `6c-artifact-manifest.json` (incl. `phase_6c4_supervised`) | JSON validated |

## 2. Numerical audit (recomputed independently from raw rows)

Fixture expected distribution: **50 SHOULD_PROPOSE / 16 SHOULD_ABSTAIN** (n=66). A perfect model therefore abstains at most 24.2% of cases.

| Arch | Acc | E0 (valid decision) | Abstain rate (pred) | **Group E** | Grounding | Proposal precision | Proposal recall | Abstain recall | Latency |
|------|-----|--------------------|--------------------|-------------|-----------|-------------------|-----------------|----------------|---------|
| A | 0.727 | 0.955 (3 UNKNOWN) | 0.015 | **0/3 = 0.0** | n/a | 48/62 = 0.774 | 48/50 = 0.96 | **0/16 = 0.0** | 3.29 s |
| B | 0.470 | 1.000 | 0.712 | **3/3 = 1.0** | 1.000 | 17/19 = 0.895 | 17/50 = 0.34 | 14/16 = 0.875 | 4.37 s |
| C | 0.349 | 1.000 | 0.864 | **3/3 = 1.0** | 1.000 | 8/9 = 0.889 | 8/50 = 0.16 | 15/16 = 0.9375 | 6.35 s |

Group E cases (`sp1://e/001..003`, all expected ABSTAIN): A predicted PROPOSE on all three (0/3); B and C predicted ABSTAIN on all three (3/3).

**Audit conclusion:** reported metrics reproduce exactly from the raw per‑case rows; no arithmetic or aggregation error found.

## 3. B vs C comparability — CONFOUND IDENTIFIED

B and C are **not** a clean architecture comparison. Two protocol differences conflate the architectural mechanism with the training protocol:

1. **Different system prompts.** B's decision stage uses `JSON_SYS` (describes `{"decision","reasoning"}`); C's uses `DECISION_SYS` (explicitly includes `evidence_refs`). The user templates are near‑identical, but the system prompt differs.
2. **Different supervision structure.** B trains a *single* generative envelope (`decision`+`reasoning`+`evidence_refs`). C trains *two* separate heads (decision envelope **and** an independent `relevant_evidence` list), i.e. ~2× training targets (C n_train=539 vs B n_train=269 from the same 300 GOLD).

The only genuine architectural difference — the post‑hoc verification gate (B: snapshot reranker; C: `Ports` gate) — is *nearly identical logic* (override unsupported PROPOSE→ABSTAIN). The observed behavioral gap (B abstains 71%, C abstains 86%) is therefore attributable to **prompt template + two‑head supervision**, not to the architectural gate. **B and C cannot be ranked against each other from this experiment alone.**

A vs B **is** comparable (both use the `b_arch` prompt family; B only adds evidence). The A→B delta is a valid measurement of the evidence‑reranker effect.

## 4. Gate‑semantics review

| Gate | Verdict on definition | Effect on A/B/C |
|------|----------------------|-----------------|
| **E0 ≤ 2% UNKNOWN** | Valid (well‑formed‑output requirement). | A FAILS (4.5% UNKNOWN). B/C PASS. |
| **Grounding 100%** | Valid (PROPOSE must cite verified evidence). | A n/a (decoder‑only, no evidence); B/C PASS. |
| **Group E ≥ 80% (deciding)** | Valid and the legitimate discriminator — measures abstention on explicit distractor/heavy‑evidence cases. | A FAILS 0/3; B/C PASS 3/3. |
| **Abstention ≥ 90%** | **MIS‑CALIBRATED / INCOHERENT.** The fixture contains only 16/66 (24.2%) genuine ABSTAIN cases; no correct model can reach 90% abstention. As literally applied it penalises *correct* proposals and cannot be satisfied. Intended semantics (epistemic caution) should be re‑expressed as a meaningful metric — e.g. **proposal precision** (when it proposes, is it right?) or **abstain‑recall on genuinely‑ambiguous cases**. | A 1.5%, B 71%, C 86% — all "FAIL" under the literal gate, but the gate itself is defective. |
| **Latency ≤ 0.5 s** | **Inapplicable to the chosen paradigm.** All realizations are generative 0.5B‑LLM decodes (3–6 s). The bound appears written for a deterministic classifier, not the generative SLM used throughout. Either declared out‑of‑scope for generative architectures or it definitively disqualifies all of them. | All three FAIL (3.3 / 4.4 / 6.4 s). |
| **VRAM ≤ 2 GB** | Valid. | All PASS (~0.5 GB). |

## 5. Written human‑review decision

**Decision: NO architecture is confirmed.** Selection remains **DEFERRED**. Reasoning:

1. **Architecture A is unsafe and rejected.** It fails the deciding Group E gate (0/3) and, more seriously, has **0% abstain‑recall** — it proposes on all 16 cases that must be abstained (14 false‑proposes + 2 unparseable). High raw accuracy (0.727) is an artefact of the 50/16 PROPOSE‑heavy fixture. A must not be selected.
2. **Architectures B and C pass the deciding gate** (Group E 3/3) and grounding (100%), and show safe behaviour (abstain‑recall 0.875 / 0.9375, proposal precision 0.895 / 0.889). They are the only viable candidates.
3. **B vs C cannot be decided from this experiment** because of the prompt + supervision confound (§3). Declaring a B/C winner now would be unsound.
4. **Two gates are defective** (abstention ≥90%; latency ≤0.5 s) and must be re‑specified before any pass/fail conclusion is meaningful.

## 6. Recommended next experiment (requires separate authorization — NOT executed)

If the goal is to choose between B and C, a **controlled** run is required to remove the confound:

- Train B and C with **identical prompts** (pick one system prompt family) and **identical supervision structure** (both single‑envelope, or both two‑head), so the *only* difference is the architectural gate.
- Re‑specify the abstention gate as **proposal precision ≥ 0.90** (and/or abstain‑recall on ambiguous cases), and either exempt generative architectures from the ≤0.5 s latency gate or restate it as a relative budget.
- Re‑evaluate both on the same frozen 66‑case fixture; only then is a B/C ranking defensible.

**No further experiment was authorized or executed in producing this decision.** Proceeding to a controlled B/C comparison requires explicit user authorization.
