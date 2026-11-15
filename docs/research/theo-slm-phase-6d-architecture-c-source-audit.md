# Architecture C — Independent Source Audit (Pre-Execution)

**Audited module:** `theo-core/scripts/model_selection/arch_c/c_architecture.py`
**Contract tests:** `theo-core/scripts/model_selection/arch_c/test_c_contract.py` (C1–C7 ALL PASS)
**Audit date:** 2026-08-26
**Author of audit:** independent review of the C implementation against the 6C plan §F/§5 and the authorization constraints.

---

## 1. Authorization constraints vs implementation

| Required constraint | Status | Evidence |
|---|---|---|
| Shared encoder / two-head (§F/§5 candidate C) | MET | `SharedEncoder` (M1 4-bit) serves `DecisionHead` + `EvidenceHead`; `ArchitectureC` wires both to one encoder (C7) |
| Within ≤1B budget | MET | M1 = Qwen2.5-0.5B-Instruct (0.5B) @ `7ae55760…`; no extra large params |
| v0.5 isolation (no symbolic-state mutation) | MET | `Ports` is read-only; heads call `.get/.ids` only (C1/C2); `ArchitectureC.run` creates a fresh `Ports` rather than mutating any existing one |
| Two heads cannot bypass frozen neural-symbolic interface | MET | Heads receive only `(percept, evidence)` text + the shared encoder; verification is exclusively via `Ports` (C2/C6) |
| No `cos(H1)` / hidden-state subspace / gradient signals | MET | Source scan negative for `hidden_states`, `subspace`, `cos(H1)`, `gradient`, `peft`, `LoraConfig`, `backward(`, `embedding`, `logits` (C3). Encoder used ONLY in generative mode (no hidden-state projection, no readout tuning) |
| No Qwen-generated label signals | MET | C is evaluated ZERO-SHOT in this build; no labels consumed at inference; source free of `Qwen-generated`/`label_type`/`adjudication.json`/`GOLD_`/`REJECT` (C4) |
| Architectural firewall (no `theo_core`/`theo_providers` import) | MET | Import set = `__future__, json, os, re, time, torch, transformers` (C5) |
| Same §7 gates / thresholds / decision rules unchanged | MET | Eval reuses the identical 66-case fixture and gate definitions as A/B |
| 300-GOLD / instrument / populated corpus / A,B artifacts unchanged | MET | C reads only; no writes to any prior artifact |
| Latency gate (≤0.5s) unchanged | MET | Not altered; reported honestly (expected to fail on this hardware) |

## 2. Deliberate design decision (flagged)
The plan §F describes C's two heads as "encoder for evidence/relation encoding + decoder head," with supervision possible. In this **bounded build**:
- Both heads are implemented as **generative prompts over the shared M1 encoder** (Head 1 → decision JSON; Head 2 → relevant-evidence JSON). This is a faithful "shared encoder / two-head" realization that avoids any hidden-state extraction (and therefore any subspace/cos(H1)/gradient concern).
- **No supervised head training is performed in this step.** Rationale: (a) per-case GOLD input texts reside in `candidate_records_batch*` under a different `case_id` namespace than `adjudication.json`, and reliably reconstructing the exact prior join is fragile and risks misusing the frozen dataset; (b) the frozen instrument carries no evidence-relevance corpus to supervise an evidence head. C is therefore evaluated **zero-shot**, exactly as A (M1_base) and B (decoder+reranker) were evaluated. A supervised C (trained decision + evidence heads) remains a separate authorization requiring a verified GOLD-input join / evidence-relevance corpus.

This does NOT change the architecture's isolation or the gates; it only scopes what is measured now.

## 3. Behavioral verification (contract tests C1–C7)
- **C1** Ports strictly read-only (no `set/add/update/put/delete/mutate/write/insert`; `.get/.ids` do not mutate).
- **C2** Heads only READ via Ports; no symbolic-state mutation (injected `SpyPorts` entries unchanged; no write calls).
- **C3/C4** Source free of prohibited/hidden-state/Qwen-label/privileged signals.
- **C5** No `theo_core`/`theo_providers` import.
- **C6** Gate logic: verified PROPOSE upheld; unsupported PROPOSE overridden→ABSTAIN; ABSTAIN preserved.
- **C7** Two distinct head objects share one encoder instance; populated-mode run issues exactly 2 `generate` calls (proving two-head-over-one-encoder).

## 4. Audit verdict
Architecture C conforms to the authorization and the §F/§5 definition within the bounded scope. It is provably: shared-encoder/two-head, snapshot-only via the read-only `Ports` interface, read-only w.r.t. symbolic state, free of `cos(H1)`/hidden-state-subspace/gradient/Qwen-label signals, and firewall-clean. **Cleared for execution** of the 6D evaluation on Architecture C (populated-evidence mode; empty-snapshot mode is degenerate-by-design and documented as such).

**Residual caveat (carried into the 6D report):** like A and B, C runs on the 0.5B decoder path on this GPU/offload hardware, so the ≤0.5s latency gate is expected to fail independent of architecture quality. A supervised-C variant requires a verified GOLD-input join and is out of scope for this step.
