# THEO SLM v0 — Phase 6C.3 L5 Execution Authorization (Acquisition / Build)

**Document ID:** `docs/research/theo-slm-phase-6c3-l5-execution-authorization.md`  
**Date:** 2026-08-26  
**Status:** FORMAL L5 EXECUTION AUTHORIZATION — binding the exact acquisition/build scope.  
**Readiness:** `theo-slm-phase-6c3-acquisition-build-authorization-package.md` APPROVED; `6C_RESOURCE_FEASIBILITY_REPORT.md` = PASS.  
**Execution state:** **NOT YET ACTIVE.** Nothing is downloaded or installed until this authorization is
expressly activated. This document only binds scope.

---

## 0. Scope statement

This authorization binds the **exact** acquisition/build scope for THEO SLM v0.5 candidate construction
under the frozen symbolic architecture. It covers:

- **Stage 1 — Acquisition:** install pinned dependency + download M1/M2 at bound revisions from official
  sources + verify hashes/licenses/revisions + write `6c-artifact-manifest.json`.
- **Stage 2 — 6C.3 Build:** QLoRA training of candidates (gated on Stage 1 pass).
- **Stage 3 / 6D:** **explicitly OUT OF SCOPE** of this authorization (see §6).

It does **not** authorize 6D comparative evaluation, Batch 3 generation, dataset mutation, or any change to
v0.4 / v0.4.1 contracts, ADR-0028, the 300-GOLD dataset, or the 6E.20.x closure.

---

## 1. Bound acquisition/build scope (exact)

| Item | Bound value |
|---|---|
| **M1 model ID** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **M1 pinned revision** | `7ae557604adf67be50417f59c2c2f167def9a775` (initial pin; acquisition records actual resolved SHA; mismatch = hard stop) |
| **M2 model ID** | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| **M2 revision** | **Resolved only at acquisition**, then frozen/bound in `6c-artifact-manifest.json` |
| **Allowed sources** | `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct` (org `Qwen/`); `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct` (org `HuggingFaceTB/`) — official orgs only; no mirrors/forks |
| **`bitsandbytes` version** | `==0.50.1` (pinned; `win_amd64` wheel SHA-256 `86f76e8a3278fbbfc3fa0d79d1c4e706ebc214babd57f0ea30e2da509bbdaad5`) |
| **torch / CUDA / Python** | torch `2.13.0+cu126` (DO NOT upgrade), CUDA 12.6, Python 3.14.3 |
| **Other libs** | `transformers`, `peft`, `accelerate`, `psutil` — current installed versions (unchanged) |
| **Dataset** | `adjudication.json` SHA-256 `97BC6BC6A0ADDBD35FAA77A9437F63458240C45AE7BB1668BF4C2DA01573EDE4` (immutable, 300 GOLD) |
| **Training method** | **QLoRA only** (4-bit base + LoRA); full fine-tune prohibited on 4.29 GB GPU |
| **Eval signals** | behavioral/structured-output only; **no** `cos(H1)` / gradient-compat / hidden-state-subspace / Qwen-label signals |

---

## 2. Pre-acquisition re-verification gate (mandatory, read-only, independent)

Before ANY install/download, the execution MUST independently re-run these checks. Each must match the
**expected** baseline below (captured 2026-08-26, read-only). **Any mismatch → hard stop** (do not proceed;
escalate to human).

| # | Check | Expected (pre-binding snapshot) | Hard stop on mismatch |
|---|---|---|---|
| 1 | Dataset SHA-256 `97bc6bc6…` | `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4` | dataset altered |
| 2 | torch / CUDA / Python | torch `2.13.0+cu126`, CUDA `12.6`, Python `3.14.3` | stack drift |
| 3 | GPU present + VRAM | GTX 1650, total 4.29 GB, free ≥ 3.0 GB | insufficient VRAM |
| 4 | Free disk (C:) | ≥ ~100 GB (measured 109.64 GB) | insufficient disk |
| 5 | Free RAM | ≥ ~6 GB (measured 6.7 GB) | insufficient RAM |
| 6 | `bitsandbytes==0.50.1` wheel + SHA | `win_amd64` wheel SHA-256 `86f76e8a…` (from PyPI 0.50.1) | wrong/unofficial wheel |
| 7 | Official model sources reachable | `Qwen/Qwen2.5-0.5B-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct` via pinned revision | source/rev mismatch |

> The snapshot above was produced by a read-only probe on 2026-08-26 and matches the approved package. It is
> the **baseline**; the execution must reproduce it independently before acting.

---

## 3. Stage 1 — Acquisition (install / acquire / verify ONLY)

Authorized actions (after the §2 gate passes and this authorization is active):
1. `pip install bitsandbytes==0.50.1` (optionally `--hash` with the SHA-256 in §1).
2. Download M1 @ `7ae55760…` and M2 @ resolved revision from official sources (pinned `revision`).
3. Compute + record SHA-256 of `model.safetensors` (each shard), `config.json`, `tokenizer.json`,
   `tokenizer_config.json` for both models into `6c-artifact-manifest.json`.
4. Record the resolved repo commit SHA per model; confirm M1 == `7ae55760…` (else hard stop).
5. Verify license == Apache 2.0 for both (confirm `LICENSE` file at pinned revision).
6. Re-confirm dataset hash `97bc6bc6…` unchanged.
7. Mark Stage 1 complete only when all hashes/revisions/licenses are recorded and consistent.

**Stage 1 does NOT train and does NOT run 6D.** Progression to Stage 2 requires Stage 1 fully verified.

---

## 4. Hard-stop rule

Any of the following **prevents progression to training (Stage 2)**:
- dataset hash ≠ `97bc6bc6…`;
- torch/CUDA/Python drift from §1;
- `bitsandbytes` version ≠ `0.50.1` or wheel SHA mismatch;
- M1 resolved revision ≠ `7ae55760…`;
- M2 revision not resolved/recorded;
- any model artifact SHA mismatch vs recorded expectation;
- license ≠ Apache 2.0;
- free disk < ~20 GB or free VRAM < ~1 GB at build time.

On any hard stop: abort, roll back to last green, and escalate — do **not** attempt training.

---

## 5. Stage 2 — 6C.3 QLoRA build (gated)

Executed only after Stage 1 passes and this authorization is active:
- Method: QLoRA (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`,
  double quant; LoRA on attention + MLP; rank/alpha recorded).
- Corpus: 300-GOLD (`97bc6bc6…`) via grouped-by-seed-family split preserving the `/005`∼`/050` group
  (44 `REJECT` excluded).
- Candidate order: A (decoder-only) primary; B (decoder+reranker) only if A fails Group E evidence-relevance.
- Behavioral gates (E0, grounding, E5, E6, Group E ≥80%, abstention ≥90%, shortcut ≤40%, latency, footprint).
- v0.5/ADR-0028 constraints enforced (baseline invariance, determinism/replay/grounding, model
  independence, architectural firewall).

---

## 6. Stage 3 / 6D — explicitly excluded here

6D comparative evaluation is **NOT authorized by this document**. It requires a **separate** authorization.
This acquisition/build authorization must not execute 6D unless a subsequent authorization explicitly
includes it. Preserve the frozen evaluation-only instruments (51-case benchmark + 15-case probe) untouched.

---

## 7. Preservation clause (binding)

The following remain strictly preserved throughout:
- **THEO v0.4 / v0.4.1 contracts** and all canonical runtime behavior.
- **ADR-0028** (provider interface, determinism, dependency direction, capability discovery) — unchanged.
- **300-GOLD dataset** (`adjudication.json`, `97bc6bc6…`) — immutable; used read-only for training.
- **6E.20.x closure** — the research branch stays closed; no re-opening of fixed-subspace-readout work; the
  reclaimed 6E20.1 checkpoint bundle remains archived off-machine (not restored locally).

---

## 8. Activation / stop

This document is the **binding scope**. Execution (§3, §5) may occur **only after this L5 authorization is
expressly activated**. Until then: no download, no `pip install`, no training, no 6D.

**STOP — present for L5 activation sign-off.**
