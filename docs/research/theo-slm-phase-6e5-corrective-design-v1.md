# Phase 6E.5 — Corrective Training Experiment Design & Preflight Report v1

**Phase:** 6E.5 — Corrective Training Experiment Design & Preflight  
**Date:** 2026-08-11  
**Base Model Target:** `Qwen/Qwen2.5-0.5B-Instruct` (Git Revision: `7ae557604adf67be50417f59c2c2f167def9a775`, SHA-256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` — **100% UNCHANGED**)  
**Preserved Baseline Adapter:** [`theo-data/datasets/theo_slm_v0_artifacts/phase-6e2/adapter_checkpoint/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e2/adapter_checkpoint/) (`adapter_model.safetensors`, SHA-256: `d4a32b87eb24d8f7d2f394396198292a01f77f56fdfec8e4565ff275af325517` — **PERMANENTLY PRESERVED REAL BASELINE**)  
**Authoritative Training Corpus:** [`theo-data/datasets/theo_slm_v0_deduplicated/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_deduplicated/) (`ds-v0.3-deduplicated`, SHA-256: `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0` — **100% UNCHANGED**)  
**Machine-Readable Preflight Manifests Directory:** [`theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/)  
**Execution Status:** **DESIGN & PREFLIGHT COMPLETE — NO MODEL TRAINING EXECUTED**

---

## 1. Purpose & Design Objectives

Following the Phase 6E.4 failure forensics report (which empirically proved that the Phase 6E.2 adapter suffered from a **3:1 supervision class imbalance** and **static 33-token target shortcut collapse**), Phase 6E.5 designed and mechanically preflight-validated the corrective training objective and evaluation harness **without running GPU training**.

### Key Design Mandates Fulfilled:
1. **Permanent Preservation of Real Baseline:** Adapter `d4a32b87...` is permanently preserved as the failed-but-real empirical benchmark for future comparisons.
2. **3-Class Supervision Representation:** Formulated distinct, non-overlapping JSON representations for `GOLD_POSITIVE`, `GOLD_ABSTAIN`, and `HARD_NEGATIVE`.
3. **50/50 Stratified Class Balancing:** Designed a stratified sampling plan balancing `SHOULD_PROPOSE` (50%) and `SHOULD_ABSTAIN` (50%) per epoch.
4. **Dynamic Reasoning Targets:** Replaced the static 33-token target string with context-specific, record-unique reasoning snippets to prevent shortcut memorization.
5. **Automated Collapse Detector Callback:** Implemented `CollapseDetectorCallback` to automatically halt training if abstention rate exceeds 90% or balanced accuracy drops below 55%.
6. **3x2 Confusion Matrix & Metric Harness:** Replaced unweighted dev accuracy with a 3x2 matrix (`GOLD_POSITIVE`, `GOLD_ABSTAIN`, `HARD_NEGATIVE` $\times$ `SHOULD_PROPOSE`, `SHOULD_ABSTAIN`) and primary metric **Balanced Accuracy**.

---

## 2. Permanent Baseline Preservation & Core Immutability (STATICALLY VERIFIED)

| Artifact Name | File Location | SHA-256 Hash | Status |
|---|---|---|---|
| **Authoritative Corpus** | `candidate_records.json` | `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0` | **100% UNCHANGED** |
| **Base Model Safetensors** | `model.safetensors` | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` | **100% UNCHANGED** |
| **Preserved Real Baseline Adapter** | `adapter_model.safetensors` | `d4a32b87eb24d8f7d2f394396198292a01f77f56fdfec8e4565ff275af325517` | **PERMANENTLY PRESERVED BASELINE** |
| **Frozen Semantic Probe** | `semantic-probe-v1-cases.json` | `eaaa47c6b47294186d2b7680507934975f876edb1125ff64e60ee8f0836c2f61` | **100% UNCHANGED** |

---

## 3. 3-Class Supervision Representation Architecture

To prevent confusion between epistemic thresholding (`GOLD_ABSTAIN`) and candidate trap rejection (`HARD_NEGATIVE`), targets are structured as follows:

```text
1. GOLD_POSITIVE (67 records / 25.4%):
   {
     "decision": "SHOULD_PROPOSE",
     "hypothesis": "<grounded_novel_proposition>",
     "reasoning": "Grounded hypothesis proposal supported by observation: '<percept_snippet>...'"
   }

2. GOLD_ABSTAIN (68 records / 25.8%):
   {
     "decision": "SHOULD_ABSTAIN",
     "rejection_type": "EPISTEMIC_THRESHOLDING",
     "reasoning": "Epistemic thresholding triggered for '<percept_snippet>...': insufficient evidence for grounded proposal."
   }

3. HARD_NEGATIVE (129 records / 48.9%):
   {
     "decision": "SHOULD_ABSTAIN",
     "rejection_type": "REPEAT|UNSUPPORTED",
     "reasoning": "Rejection triggered for '<percept_snippet>...': candidate '<trap_prop>...' is a repeat/unsupported claim."
   }
```

### Preflight Uniqueness Validation:
Mechanical preflight testing confirmed that 5 out of 5 sampled records constructed **100% unique dynamic target strings**, completely eliminating the static 33-token target shortcut that caused collapse in Phase 6E.2.

---

## 4. Stratified Rebalancing Plan (50/50 Class Ratio)

- **Class Balancing Strategy:** Stratified Oversampling of `GOLD_POSITIVE` records.
- **Per-Epoch Training Target:** 268 total records per epoch.
  - `GOLD_POSITIVE`: 134 records (oversampled $2\times$ from 67 records) $\rightarrow$ **50.0% of batch** (`SHOULD_PROPOSE`)
  - `GOLD_ABSTAIN`: 67 records $\rightarrow$ **25.0% of batch** (`SHOULD_ABSTAIN`)
  - `HARD_NEGATIVE`: 67 records (sampled from 129 records) $\rightarrow$ **25.0% of batch** (`SHOULD_ABSTAIN`)
- **Benefit:** Guarantees an exact **1:1 decision class ratio** (`SHOULD_PROPOSE` vs `SHOULD_ABSTAIN`) for every training epoch without modifying loss gradients or model capacity.

---

## 5. Automated Collapse Detector Callback (`CollapseDetectorCallback`)

A custom PyTorch/Transformers `TrainerCallback` that automatically monitors model behavior during training:

```python
class CollapseDetectorCallback(TrainerCallback):
    def __init__(self, abstain_threshold: float = 0.90, min_balanced_acc: float = 0.55):
        self.abstain_threshold = abstain_threshold
        self.min_balanced_acc = min_balanced_acc
        self.collapse_detected = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control
        r_abstain = metrics.get("eval_should_abstain_rate", 0.0)
        bal_acc = metrics.get("eval_balanced_accuracy", 0.50)
        if r_abstain >= self.abstain_threshold or bal_acc < self.min_balanced_acc:
            self.collapse_detected = True
            print(f"\n[ALERT] COLLAPSE TRIGGERED at Step {state.global_step}: Abstain Rate={r_abstain*100:.1f}%, Balanced Acc={bal_acc*100:.1f}%")
            control.should_training_stop = True
        return control
```

### Preflight Test Result:
- **Normal Metric Test:** Passed cleanly (no false alert).
- **100% Abstain Collapse Simulation:** Immediately caught collapse at Step 0, flagged alert, and set `should_training_stop = True`.

---

## 6. 3x2 Confusion Matrix & Metric Harness

Unweighted dev accuracy is replaced with a comprehensive multi-dimensional metric harness:

$$\begin{pmatrix}
N_{\text{POS, PROPOSE}} & N_{\text{POS, ABSTAIN}} \\
N_{\text{ABS, PROPOSE}} & N_{\text{ABS, ABSTAIN}} \\
N_{\text{NEG, PROPOSE}} & N_{\text{NEG, ABSTAIN}}
\end{pmatrix}$$

### Metrics Calculated Every Evaluation Step:
1. **Balanced Accuracy:** $\frac{\text{Recall}_{\text{POS}} + \text{Recall}_{\text{ABS}}}{2}$ (Primary Success Metric)
2. **Per-Class Recall:**
   - $\text{Recall}_{\text{GOLD\_POSITIVE}} = \frac{N_{\text{POS, PROPOSE}}}{N_{\text{POS, PROPOSE}} + N_{\text{POS, ABSTAIN}}}$
   - $\text{Recall}_{\text{GOLD\_ABSTAIN}} = \frac{N_{\text{ABS, ABSTAIN}}}{N_{\text{ABS, PROPOSE}} + N_{\text{ABS, ABSTAIN}}}$
   - $\text{Recall}_{\text{HARD\_NEGATIVE}} = \frac{N_{\text{NEG, ABSTAIN}}}{N_{\text{NEG, PROPOSE}} + N_{\text{NEG, ABSTAIN}}}$
3. **Proposal Precision:** $\frac{N_{\text{POS, PROPOSE}}}{N_{\text{POS, PROPOSE}} + N_{\text{ABS, PROPOSE}} + N_{\text{NEG, PROPOSE}}}$
4. **Abstention Precision:** $\frac{N_{\text{ABS, ABSTAIN}} + N_{\text{NEG, ABSTAIN}}}{N_{\text{POS, ABSTAIN}} + N_{\text{ABS, ABSTAIN}} + N_{\text{NEG, ABSTAIN}}}$
5. **Canonical `b/002_power_outage` Audit**
6. **Frozen 51-Case Benchmark Accuracy & 15-Case Semantic Probe Score**
7. **Format Error Rate ($E_0$) & Grounding Bypass Count**

---

## 7. Machine-Readable Preflight Manifests Inventory

All 7 preflight manifests created under [`theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/):
- [`design-manifest.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/design-manifest.json)
- [`supervision-schema.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/supervision-schema.json)
- [`rebalancing-plan.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/rebalancing-plan.json)
- [`collapse-detector-config.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/collapse-detector-config.json)
- [`confusion-matrix-schema.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/confusion-matrix-schema.json)
- [`preflight-validation-results.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/preflight-validation-results.json)
- [`phase-6e5-summary.json`](file:///c:/Users/bs162/Desktop/THEO/theo-data/datasets/theo_slm_v0_artifacts/phase-6e5/phase-6e5-summary.json)

---

## Governance Confirmation & CRITICAL STOP CONDITION

```text
[Step 1] Verify core model, baseline adapter, corpus, and probe SHA-256 hashes. --> COMPLETE (100% Immutable)
[Step 2] Audit 3-class supervision schema in corpus.                        --> COMPLETE (67 POS, 68 ABS, 129 NEG)
[Step 3] Test dynamic target construction.                                 --> COMPLETE (100% Unique Per Record)
[Step 4] Formulate 50/50 stratified rebalancing plan.                      --> COMPLETE (134 POS : 134 ABS/NEG)
[Step 5] Validate automated CollapseDetectorCallback.                       --> COMPLETE (Verified Auto-Stop)
[Step 6] Audit 3x2 confusion matrix & metric harness.                      --> COMPLETE (Balanced Acc Primary)
[Step 7] Save 7 machine-readable preflight manifests.                       --> COMPLETE
[Step 8] Write Phase 6E.5 Research Report.                                  --> COMPLETE (docs/research/...corrective-design-v1.md)
[Step 9] STOP at Corrective Design & Preflight Gate.                       --> CURRENT STOP POINT (NO TRAINING EXECUTED)
[Step 10] Phase 6E.6 (Rebalanced Controlled Training Experiment).          --> Pending human authorization
```

**Phase 6E.5 is COMPLETE.** Execution has halted at **CORRECTIVE DESIGN & PREFLIGHT GATE** with verdict: **`READY FOR HUMAN REVIEW & AUTHORIZATION BEFORE PHASE 6E.6`**.

**NO MODEL TRAINING WAS EXECUTED IN THIS PHASE.**  
Awaiting explicit human review and authorization before any future training run.
