# THEO SLM v0 — Phase 6C.3 Acquisition / Build Authorization Package

**Document ID:** `docs/research/theo-slm-phase-6c3-acquisition-build-authorization-package.md`  
**Date:** 2026-08-26  
**Depends on:** `theo-slm-phase-6c-execution-readiness-package.md` (APPROVED), `6C_RESOURCE_FEASIBILITY_REPORT.md` (PASS)  
**Status:** AUTHORIZATION REQUEST — **STOP. Awaiting separate L5 (acquisition/build) authorization.**  
**This document binds exact versions / revisions / sources. It performs NO execution.**

---

## 0. Milestone note — leaving the repair/forensic phase

All 6E.20.x remediation gates are closed. The next authorized work is **not** about fixing 6E.20.x; it is
about **building the actual THEO SLM v0.5 neural-runtime candidate under the frozen symbolic architecture**
(ADR-0028 / v0.5 plan). Per the readiness review, the current 300-GOLD corpus has **no hard blocker**, so
**Batch 3 is NOT generated now** — the project moves forward rather than chasing the recommended dataset
ratio indefinitely.

---

## 1. Authorization boundary

| Out-of-scope (NOT performed by this package) | Status |
|---|---|
| Downloading any model weights | **not performed** (pending L5) |
| Installing `bitsandbytes` / modifying the venv | **not performed** (pending L5) |
| 6C.3 training / fine-tuning | **not performed** (separate stage, pending L5) |
| 6D comparative evaluation | **not performed** (excluded from acquisition step) |
| Modifying `adjudication.json` (hash `97bc6bc6…`) | **not performed** (immutable) |

This package **binds** the exact acquisition/build parameters so that execution, once authorized, is
reproducible and cannot drift into an unpinned `pip install` or an unverified download.

---

## 2. Bound model candidates (exact IDs + pinned revisions)

| ID | Model ID (exact) | Pinned revision | License | Role |
|---|---|---|---|---|
| **M1** | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` *(proposed initial pin — see note)* | Apache 2.0 | primary base (E1/E3) |
| **M2** | `HuggingFaceTB/SmolLM2-360M-Instruct` | **RESOLVED_AT_ACQUISITION** (recorded SHA) | Apache 2.0 | CPU-feasibility base (E2/E4) |

**Revision-binding rule:**
- M1's proposed pin `7ae55760…` was observed in the 6E20.1 evidence manifest for
  `Qwen/Qwen2.5-0.5B-Instruct` and is the initial pin. At acquisition, the download MUST be executed at an
  explicit commit SHA; the actually-resolved SHA (which must equal the pin unless the repo history changed)
  is recorded in `6c-artifact-manifest.json`. A mismatch from the expected pin is a **hard stop** (re-confirm
  with human before proceeding).
- M2 has no pre-asserted commit in this package (none was observable). Acquisition MUST resolve the current
  `main` commit SHA of `HuggingFaceTB/SmolLM2-360M-Instruct` and **record it**; the recorded SHA becomes the
  bound revision for all future reproducibility.
- **M3** (`meta-llama/Llama-3.2-1B-Instruct`) remains **excluded** (exceeds the ≤1B §F budget; would
  require a separate license-waiver authorization).

---

## 3. Allowed official sources

| Model | Allowed source (official org only) |
|---|---|
| M1 | `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct` (org `Qwen/`) |
| M2 | `https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct` (org `HuggingFaceTB/`) |

- **No mirrors, no forks, no unverified third-party re-hosts.** Download only via pinned `revision` from
  the official repo.
- Retrieval method: `huggingface_hub.snapshot_download(repo_id, revision=<pinned>, allow_patterns=[...])`
  or `transformers.AutoModel.from_pretrained(repo_id, revision=<pinned>)` — both record the resolved commit.

---

## 4. Bound dependency stack (exact versions)

| Component | Bound version | Verification status |
|---|---|---|
| Python | 3.14.3 | verified present |
| `torch` | 2.13.0+cu126 | verified present — **DO NOT upgrade** |
| CUDA (driver/runtime) | 12.6 | verified (`nvidia-smi`) |
| `transformers` | installed (current) | verified present |
| `peft` | installed (current) | verified present |
| `accelerate` | installed (current) | verified present |
| `psutil` | installed (current) | verified present |
| **`bitsandbytes`** | **`==0.50.1`** | **TO INSTALL** (pinned) |
| `bitsandbytes` wheel (win_amd64) SHA-256 | `86f76e8a3278fbbfc3fa0d79d1c4e706ebc214babd57f0ea30e2da509bbdaad5` | for install hardening |

**Compatibility assumptions (already verified in `6C_RESOURCE_FEASIBILITY_REPORT.md`):**
- `bitsandbytes` 0.50.1 supports **Python 3.10–3.14**, **CUDA 11.8/12/13**, **PyTorch ≥2.3**, NVIDIA
  **SM75+**. The local stack is torch 2.13.0+cu126 / CUDA 12.6 / Python 3.14.3 / **GTX 1650 (Turing, SM75)**
  — fully within support. The `win_amd64` wheel exists, so Windows install is supported.
- **Installation command (pinned, no drift):** `pip install bitsandbytes==0.50.1` (optionally with
  `--hash` using the SHA-256 above). This must NOT pull an unpinned/newer version.

---

## 5. SHA-256 manifests for downloaded artifacts

At acquisition, every downloaded file is hashed and recorded in
`theo-data/datasets/theo_slm_v0_review/6b3_batch2/6c-artifact-manifest.json`. Schema:

| Artifact | Hash bound | When |
|---|---|---|
| Canonical dataset `adjudication.json` | `97BC6BC6A0ADDBD35FAA77A9437F63458240C45AE7BB1668BF4C2DA01573EDE4` | already bound (frozen) |
| M1 `model.safetensors` (each shard) | recorded at acquisition | post-download |
| M1 `config.json` | recorded | post-download |
| M1 `tokenizer.json` + `tokenizer_config.json` | recorded | post-download |
| M2 `model.safetensors` (each shard) | recorded | post-download |
| M2 `config.json` | recorded | post-download |
| M2 `tokenizer.json` + `tokenizer_config.json` | recorded | post-download |
| Resolved repo `revision` (commit SHA) per model | recorded | post-download |
| `bitsandbytes` wheel SHA-256 | recorded (pinned above) | post-install |

**Binding rule:** no artifact enters training/eval unless its hash is recorded and matches the pinned
expectation. A hash/revision/license mismatch **aborts** the step (rollback to last green; human notified).

---

## 6. License verification

| Model | Expected license | Verification action |
|---|---|---|
| M1 `Qwen/Qwen2.5-0.5B-Instruct` | Apache 2.0 | confirm `LICENSE` file in repo at pinned revision == Apache 2.0 |
| M2 `HuggingFaceTB/SmolLM2-360M-Instruct` | Apache 2.0 | confirm `LICENSE` == Apache 2.0 |

- Both are Apache-2.0 → permitted for the v0.5 build.
- M3 (Llama 3.2 Community) is **excluded**; it would require a separate license-waiver authorization.

---

## 7. Frozen v0.5 / ADR-0028 constraints (binding, unchanged)

- **C1 Baseline invariance:** with no provider configured, runtime == v0.4.1 (35/35 PASS).
- **C3 Determinism / replay / grounding:** greedy fixed-seed; byte-stable replay; 6A.2 rationale-ordering
  defect is a regression test; every output grounded against the snapshot boundary.
- **C4 Model independence:** architecture selection + base-model choice recorded in design doc / research
  notes, **never** written into ADR-0028.
- **Architectural firewall (permanent):** `theo_core` MUST NOT import `theo_providers`.

---

## 8. QLoRA-only training path (binding)

- **Training method is QLoRA exclusively** (4-bit base + LoRA adapters). Full fine-tune is **prohibited on
  this hardware**: the GTX 1650 has 4.29 GB VRAM, and a full-FT M1 needs ~7.8 GB VRAM (package §4.2) — it
  cannot fit. QLoRA peak VRAM (~0.6–0.8 GB) fits with ~3.45 GB free.
- Bound QLoRA config: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`,
  `bnb_4bit_compute_dtype=torch.bfloat16` (or `float16`), `bnb_4bit_use_double_quant=True`; LoRA on
  attention + MLP projections; rank/alpha set at 6C.3 build (recorded).
- The 300-GOLD corpus (`97bc6bc6…`) is used via a **grouped-by-seed-family** split preserving the
  intentional `/005`∼`/050` near-duplicate group (split-leakage = 0). 44 `REJECT` records excluded.
- **No `cos(H1)` / gradient-compatibility / hidden-state-subspace / Qwen-label signals** in any training or
  evaluation (design §B/§G/§H; review-packet Gates 2–4).

---

## 9. Execution staging (what L5 would authorize)

| Stage | Authorized action (post-L5) | Training? | 6D? |
|---|---|---|---|
| **Stage 1 — ACQUISITION** | install `bitsandbytes==0.50.1`; download M1+M2 at pinned revisions from official sources; compute + record SHA-256 of all artifacts; verify license; verify dataset hash `97bc6bc6…` unchanged; write `6c-artifact-manifest.json` | **NO** | **NO** |
| Stage 2 — 6C.3 BUILD | QLoRA training of candidates (A primary; B reranker fallback if Group E fails); behavioral gates | YES | NO |
| Stage 3 — 6D EVAL | comparative evaluation vs v0.4.1 / heuristic / reference SLM | — | YES |

**Hard gates inside this package:**
- **No training before Stage-1 acquisition + hash/license/revision verification passes.** Stage 2 may not
  begin until Stage 1's `6c-artifact-manifest.json` is complete and all hashes match.
- **No 6D evaluation in the acquisition step.** Stage 3 is a separate, later authorization.
- **Dataset hash `97bc6bc6…` must be re-confirmed unchanged** immediately before any training use.

---

## 10. Stop — separate L5 authorization required

This package is a **binding authorization request**. It performs no execution. The following require the
separate **L5 acquisition/build authorization** before any command runs:

- ☐ `pip install bitsandbytes==0.50.1`
- ☐ Download M1 (`Qwen/Qwen2.5-0.5B-Instruct` @ `7ae55760…`) and M2 (`HuggingFaceTB/SmolLM2-360M-Instruct` @ resolved SHA)
- ☐ Record `6c-artifact-manifest.json` (hashes + revisions + licenses)
- ☐ (subsequently) 6C.3 QLoRA build, then 6D evaluation

**STOP — present this package for the separate L5 authorization.**
