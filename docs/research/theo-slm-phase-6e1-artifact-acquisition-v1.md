# Phase 6E.1 — Real SLM Artifact Acquisition & Reproducibility Setup v1

**Phase:** 6E.1 — Artifact Acquisition & Reproducibility Setup
**Audit date:** 2026-08-11
**Model target:** `Qwen/Qwen2.5-0.5B-Instruct`
**Verdict:** **PASS** (all 9 acquisition steps executed, 9/9 passed)

---

## 1. Purpose

Establish a real, independently verifiable Qwen2.5-0.5B-Instruct artifact
environment before any training occurs, replacing the previously unverifiable
Phase 6C.2–6C.10 claims (Phase 6D verdict: HOLD). This phase performs **no**
training, no LoRA, no evaluation of the frozen benchmark, no corpus
modification, and no deployment.

## 2. Evidence classification

Every claim below is tagged:

- **ACTUALLY EXECUTED** — the value originates from a process that ran during
  this phase (HF API call, file download, SHA-256 computation, model load, GPU
  generation, tokenizer round-trip).
- **STATICALLY VERIFIED** — the value was read/verified from material on disk
  or from a live authoritative API response during this phase.
- **NOT VERIFIED** — could not be confirmed.

No value in this report is a hard-coded constant used as a measurement.

## 3. Repo survey & existing SLM/reference architecture (STATICALLY VERIFIED)

| Item | Finding |
|---|---|
| `theo-providers` | Only real neural stack on the machine: Qwen3-0.6B reference provider (`theo_providers.qwen3`), documented in `model-info.json`. No Qwen2.5 artifacts exist in this project. |
| `theo-core` | Symbolic cognitive runtime (`theo_core`). No SLM runtime integration, no `ENABLE_THEO_SLM_V0` flag, no LoRA/peft/transformers references (verified by grep during Phase 6D). |
| `theo-data/datasets/theo_slm_v0_deduplicated` | Frozen corpus, SHA-256 `a7b4e845...` — **untouched** this phase. |
| Phase 6C.2–6C.10 | Simulation scripts with hard-coded success values. Treated as **non-evidentiary**; preserved as historical forensic artifacts, not reused. |
| Dependency declarations | `theo-providers/pyproject.toml` declares `providers-slm` extras: torch, transformers, accelerate, huggingface-hub, safetensors. **peft is not declared**; it was installed this phase (see §4). |

## 4. Environment verification (ACTUALLY EXECUTED)

Captured in `environment-manifest.json` (full machine-readable record):

| Component | Value |
|---|---|
| Python | 3.14.3 (CPython, `theo-providers/.venv`) |
| PyTorch | 2.13.0+cu126 |
| Transformers | 5.15.0 |
| PEFT | **0.20.0 — installed this phase** (`uv pip install peft`; 0.20.0 resolved cleanly; not yet declared in pyproject) |
| Accelerate | 1.14.0 |
| huggingface_hub | 1.27.0 |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| numpy | 2.5.2 |
| CUDA (torch) | 12.6, cuDNN 91002, `torch.cuda.is_available()=True` |
| GPU | NVIDIA GeForce GTX 1650, capability (7,5), 4.0 GiB total VRAM (3.46 GiB free at manifest time) |
| Driver | 592.00 (nvidia-smi reports CUDA 13.1 host driver) |
| Disk free | C: 56.3 GiB, D: 16.5 GiB (at acquisition time) |
| Env vars | HF_HOME / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE all unset (default cache used) |

## 5. Authoritative source & revision (ACTUALLY EXECUTED)

Queried live from the Hugging Face Hub API (`HfApi().model_info(..., files_metadata=True)`):

- **Repo:** `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct`
- **Revision (main @ time of acquisition):** `7ae557604adf67be50417f59c2c2f167def9a775`
- Created 2024-09-16, pipeline `text-generation`, public, not gated.
- Single weight file: `model.safetensors`, **988,097,824 bytes**, LFS content
  SHA-256 **`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`**
  (obtained from the Hub API, not computed).

Full per-file metadata (sizes, LFS sha256, git blob ids) is in `model-provenance-manifest.json`.

## 6. Download (ACTUALLY EXECUTED)

- `snapshot_download(Qwen/Qwen2.5-0.5B-Instruct, revision=7ae55760...)` into the
  default HF cache. Elapsed ~61 s. 9 files fetched.
- **Material location on disk:**
  `C:\Users\bs162\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775`
- Recorded in `download-log.json`.

## 7. Artifact hashes (ACTUALLY EXECUTED)

Local SHA-256 computed for all 9 files in `artifact-hashes.json`:

| File | Size (bytes) | SHA-256 (computed locally) |
|---|---|---|
| config.json | 659 | recorded in artifact-hashes.json |
| generation_config.json | 242 | recorded in artifact-hashes.json |
| merges.txt | 1,671,839 | recorded in artifact-hashes.json |
| **model.safetensors** | **988,097,824** | **`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`** |
| tokenizer.json | 7,031,645 | recorded in artifact-hashes.json |
| tokenizer_config.json | 7,305 | recorded in artifact-hashes.json |
| vocab.json | 2,776,833 | recorded in artifact-hashes.json |
| README.md, LICENSE | 4,917 / 11,343 | recorded in artifact-hashes.json |

**Integrity: local `model.safetensors` SHA-256 equals the authoritative Hub LFS
OID → `matches_authoritative_oid = true`.**

## 8. Tokenizer verification (ACTUALLY EXECUTED)

- Class: `Qwen2Tokenizer` (loaded from local snapshot, `trust_remote_code=False`)
- Vocabulary size (from `get_vocab()`): **151,665**
- Special tokens: bos=`<|im_start|>`, eos=`<|im_end|>`, pad=None, unk=None (recorded in JSON)
- Round-trip probes (encode→decode): 4/4 **exact**, including English, a THEO-style
  sentence, Cyrillic text, and the empty string.
- Recorded in `tokenizer-verification.json`.

## 9. Model configuration & parameter count (ACTUALLY EXECUTED)

From `config.json` (verbatim, in `model-config.json`):

- Architecture `Qwen2ForCausalLM`, `model_type=qwen2`
- hidden_size 896, intermediate_size 4864, num_hidden_layers 24,
  num_attention_heads 14, num_key_value_heads 2, vocab_size 151936,
  max_position_embeddings 32768, tie_word_embeddings true,
  rope_theta 1e6, rms_norm_eps 1e-6, torch_dtype bfloat16 (config declares; loaded fp16)

Measured from the **loaded state dict** in a fresh process (dtype fp16, `cuda:0`):

- **Total parameters: 494,032,768 (~0.494B)**
- Trainable parameters: 494,032,768 (no LoRA/adapters — base model)
- Estimated fp16 weights: 988,065,536 bytes (≈ matches safetensors size)
- Load: 7.63 s; GPU free after load: 2.41 GiB (fits GTX 1650 4 GB)

## 10. Minimal real inference smoke test (ACTUALLY EXECUTED)

Fresh process; model loaded fp16 to CUDA; greedy generation (`do_sample=False`,
seed 0) on the fixed prompt `"The sky is blue because"` (5 prompt tokens),
`max_new_tokens=32`.

- Model load: 1.5 s; generation: 3.13 s (32 tokens)
- Full output: `The sky is blue because of the water in it. The sky is not blue because of the water in it.\nThis justifies what answer for what question? Q & A:\n\nQuestion`
- Token ids, token strings, and decoded text are recorded in `inference-smoke-test-result.json`.

## 11. Independent reloadability proof (ACTUALLY EXECUTED)

Second, separate fresh process:

- On-disk `model.safetensors` SHA-256 recomputed: matches the artifact manifest and the Hub OID.
- Parameter count recomputed from reloaded weights: 494,032,768 (identical).
- Same greedy generation re-run: **generated token ids identical** to the smoke test
  (determinism + same physical artifact).
- Recorded in `reload-proof.json`.

## 12. Machine-readable artifacts

Directory: `theo-data/datasets/theo_slm_v0_artifacts/`

| File | Content |
|---|---|
| `environment-manifest.json` | environment + versions + CUDA/GPU/disk |
| `model-provenance-manifest.json` | Hub metadata, revision, per-file LFS metadata, local hashes, integrity cross-checks |
| `artifact-hashes.json` | per-file SHA-256 + authoritative-OID match |
| `tokenizer-verification.json` | tokenizer class, vocab, round-trips |
| `model-config.json` | config.json verbatim + measured parameter count |
| `inference-smoke-test-result.json` | smoke test inputs/outputs/timings |
| `reload-proof.json` | independent reload determinism proof |
| `download-log.json` | download record |
| `acquisition-result.json` | 9-step verdict: **PASS** |
| `scripts/*.py` | the exact scripts used (reproducible audit trail) |

## 13. ACTUALLY EXECUTED vs STATICALLY VERIFIED vs NOT VERIFIED

**ACTUALLY EXECUTED:** environment collection, Hub API queries, download (61 s),
SHA-256 computation of all 9 files, tokenizer round-trips, model load + parameter
count, GPU inference, independent reload determinism check.

**STATICALLY VERIFIED:** repo survey contents, config.json values (read from the
downloaded file), HF cache snapshot layout, file sizes.

**NOT VERIFIED:** nothing required by this phase. Explicitly **non-verifiable by
design** (and preserved for the historical record): the prior Phase 6C.2–6C.10
metrics, which remain non-evidentiary until independently reproduced. Note also:
peft 0.20.0 is installed in the shared venv but is **not yet declared** in
`theo-providers/pyproject.toml` — declare it before Phase 6E.2.

## 14. Anti-simulation compliance

- No measured result is hard-coded; every number in §4–§11 came from a live run.
- No simulated training curves, no fabricated hashes, no PASS from constants.
- The model exists materially on disk and reloads identically in a fresh process
  (§11), satisfying the "prove materially on disk, independently reloadable" test.

## 15. Hard stop

Phase 6E.1 is complete. **No training, no LoRA adapters, no evaluation of the
frozen benchmark, no corpus modification, no deployment, and no Phase 6E.2
work was performed.** Awaiting explicit human authorization before proceeding.

## 16. References

- Phase 6D report: `theo-core/docs/research/theo-slm-phase-6d-full-system-audit-v1.md`
- Phase 6D machine-readable results: `theo-core/docs/research/phase-6d-results/`
- Phase 6E.1 artifacts: `theo-data/datasets/theo_slm_v0_artifacts/`
- Frozen corpus (untouched): `theo-data/datasets/theo_slm_v0_deduplicated/`
