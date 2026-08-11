# THEO SLM Training Experiment Protocol Specification v0 — Phase 6C.3

**Document ID:** `docs/research/theo-slm-training-experiment-v0.md`  
**Date:** 2026-08-11  
**Status:** FROZEN EXPERIMENTAL PROTOCOL SPECIFICATION  
**Authoritative Training Corpus:** [`theo-data/datasets/theo_slm_v0_deduplicated/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_deduplicated/) (`ds-v0.3-deduplicated`, SHA-256: `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0`)  
**Selected Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (490M Parameters)

---

## 1. Complete Hyperparameter & Configuration Specification

| Hyperparameter / Parameter | Exact Specification Value |
|---|---|
| **Base Model Revision** | `Qwen/Qwen2.5-0.5B-Instruct` (Commit: `main`) |
| **Tokenizer Revision** | `Qwen/Qwen2.5-0.5B-Instruct` (Fast Tokenizer, 151,936 Vocab) |
| **Dataset Revision** | `ds-v0.3-deduplicated` (264 candidate records) |
| **Train / Dev Split** | Grouped-by-Seed 80/20 (208 Train records / 56 Dev records, Seed `20260811`) |
| **Maximum Sequence Length** | 2,048 tokens |
| **Optimization Method** | LoRA PEFT ($r=16, \alpha=32$, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) |
| **Precision Mode** | FP16 / BF16 mixed precision |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) |
| **Base Learning Rate** | $2.0 \times 10^{-4}$ |
| **Learning Rate Scheduler** | Cosine Annealing with Warmup |
| **Warmup Ratio** | $0.05$ (5% of total training steps) |
| **Weight Decay** | $0.01$ |
| **Per-Device Batch Size** | 4 records |
| **Gradient Accumulation** | 2 steps (Effective Batch Size = 8) |
| **Total Epochs** | 5 Epochs (130 Optimization Steps) |
| **Random Seeds** | `42` (Data & LoRA initialization) |
| **Checkpoint Policy** | Checkpoint saved every 26 steps (1 epoch) |
| **Evaluation Frequency** | Evaluated after every epoch on Train, Dev, Probe, and Benchmark |

---

## 2. Input Projection Schema & No-Leakage Verification (Step 2)

Input payload projection strictly excludes generator metadata:

$$\text{Input Payload} = \{\text{percept, task, concepts, beliefs, rules, candidate\_proposition, grounding\_snapshot}\}$$

$$\text{Excluded Metadata} = \{\text{human\_review\_status, GOLD\_POSITIVE/ABSTAIN/HARD\_NEGATIVE, capability\_id, tier, generator\_id, seed\_id, provenance}\}$$

$$\text{Input Payload} \cap \text{Excluded Metadata} = \emptyset \quad (\text{SHA-256 Input Projection Hash: } \texttt{e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855})$$

---

## 3. Supervision Targets & Structured Output Adapter (Steps 3 & 4)

- **Target JSON Schema:** Emits `SemanticInterpretation` JSON matching design specification:
  ```json
  {
    "proposition": "Indicates strep throat condition state.",
    "supporting_evidence_ids": ["ev://med/001"],
    "referenced_concept_ids": ["conc://med/strep_throat"],
    "semantic_relation": "causal",
    "confidence": 0.88
  }
  ```
- **Adapter Validation:** Deterministic adapter parses schema into `HypothesisProposal`. Structural formatting errors produce an immutable `E0` failure. Malformed JSON is never silently repaired.

---

## 4. Experimental Plan (Steps 5–8)

1. **Step 5 — Infrastructure Sanity Experiment:** 1-epoch validation run to verify gradients, loss reduction, checkpoint saving, and adapter parsing.
2. **Step 6 — Controlled Experiment A (Semantic Supervision Only):** Trained on 208 records using primary positive/abstention target propositions.
3. **Step 7 — Controlled Overfitting Analysis:** Compare Train vs Grouped Dev vs Frozen 15-case Semantic Probe across all checkpoints.
4. **Step 8 — Controlled Experiment B (Ablation with Explicit Negative Supervision):** Trained on 208 records incorporating explicit negative trap rejection targets.

---

## 5. Numerical GO / HOLD / FAIL Gate Criteria (Step 11)

| Metric Gate | Definition | Minimum GO Threshold |
|---|---|---|
| **E0 Format Error Rate** | Proportion of malformed JSON / adapter failures | $\le 2.0\%$ |
| **Grounding Validity** | Proportion of referenced IDs resolving against snapshot | $100.0\%$ |
| **Semantic Novelty Rate (E5)**| Proportion of non-derivable, evidence-supported novel interpretations | $\ge 40.0\%$ |
| **Decision Relevance Rate (E6)**| Proportion of novel proposals answering decision task | $\ge 30.0\%$ |
| **Distractor Rejection** | Group E cases ignoring distractor IDs | $\ge 80.0\%$ |
| **Abstention Precision** | Correct abstention on incomplete/premature cases | $\ge 90.0\%$ |
| **Shortcut Resistance** | Adversarial TF-IDF classifier Balanced Accuracy | $\le 40.0\%$ (Random Chance) |
| **Useful Proposal Rate** | Composite headline (Novel $\land$ Non-derivable $\land$ Grounded $\land$ Relevant) | $\ge 35.0\%$ |

---

## Governance & Immutability Confirmation

- **Frozen Corpus `ds-v0.3-deduplicated`:** Immutable (SHA-256: `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0`).
- **Frozen Benchmark & Semantic Probe:** Untouched.
- **ADR-0028 & Provider Contracts:** Untouched.
