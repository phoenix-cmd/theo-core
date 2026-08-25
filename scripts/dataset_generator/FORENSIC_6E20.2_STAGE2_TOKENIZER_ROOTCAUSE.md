# FORENSIC ROOT-CAUSE REPORT — 6E.20.2 STAGE 2 TOKENIZER VOCABULARY MISMATCH

**Report ID:** 6E20.2-STAGE2-TOKENIZER-ROOTCAUSE
**Date:** 2026-08-19
**Mode:** SOURCE + ENVIRONMENT FORENSIC INVESTIGATION ONLY (no execution, no modification, no download, no repair)
**Halt trigger:** `Stage 2 HALT: Vocab mismatch 151665 != 151936`
**Classification:** **B. Measurement/API semantic defect — correct tokenizer is present, but Stage 2 checks the wrong vocabulary metric.**

---

## 1. Evidence — Which tokenizer/revision was actually resolved

- Local Hugging Face cache: `C:\Users\bs162\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct`
- `refs/main` → `7ae557604adf67be50417f59c2c2f167def9a775`
- `snapshots/` contains exactly: `7ae557604adf67be50417f59c2c2f167def9a775`

This is the **exact pinned `MODEL_REVISION`** in `run_phase_6e20_reevaluation.py:63` (`MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"`). The execution's Stage 2 loaded the tokenizer **from this correct, pinned revision** — not a stale or wrong cached copy. (The model weights also loaded successfully from the same snapshot, further confirming revision identity.)

## 2. Evidence — Pinned artifact vocabulary metrics (from the resolved snapshot)

| Artifact | Field | Value |
|---|---|---|
| `config.json` | `vocab_size` | **151936** |
| `config.json` | `model_type` | `qwen2` |
| `tokenizer.json` | `model.type` | `BPE` |
| `tokenizer.json` | base `vocab` size | **151643** |
| `tokenizer.json` | `added_tokens` count | **22** |
| implied `len(tokenizer)` | base + added | **151643 + 22 = 151665** |
| `tokenizer_config.json` | `tokenizer_class` | `Qwen2Tokenizer` |

`151665` (observed failure) = `151643 (base) + 22 (added special tokens)`. This matches the Stage 2 observed value **exactly and necessarily**.

## 3. Evidence — Library versions

- `transformers` = 5.8.0
- `tokenizers` = 0.22.2
- `huggingface_hub` = 1.14.0

## 4. Evidence — The check that failed

`run_phase_6e20_reevaluation.py`:
- Line 67: `VOCAB = 151936`
- Line 348: `tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)`
- Line 351: `if len(tokenizer) != VOCAB:` → `raise PreflightError(f"Stage 2 HALT: Vocab mismatch {len(tokenizer)} != {VOCAB}")`

The check compares **`len(tokenizer)` (the full tokenizer length, base + added special tokens = 151665)** against **`VOCAB = 151936`**, which is the **model's embedding `vocab_size`** (`config.json`), not the tokenizer length.

## 5. Root-Cause Analysis (why 151665 ≠ 151936)

- `VOCAB = 151936` is the **model's `config.vocab_size`** (the embedding-matrix row count) for the pinned `Qwen/Qwen2.5-0.5B-Instruct` revision. This is the **authoritative identity value** for the pinned model.
- `len(tokenizer)` returns the **tokenizer's full vocabulary length**, which for a Qwen2 tokenizer equals `base_vocab (151643) + added_special_tokens (22) = 151665`.
- These are **two different quantities**. A correct, pinned tokenizer will *always* report `len(tokenizer) = 151665`, never `151936`. The invariant value `151936` can only be satisfied by `model.config.vocab_size` (or, equivalently, the model embedding dimension), **not** by `len(tokenizer)`.

Therefore:
- **Not A (wrong cached revision):** the resolved snapshot is provably the pinned commit `7ae5576…`; `config.json` `vocab_size` is 151936, confirming the correct model.
- **Not C (frozen invariant defect):** `151936` is the correct, authoritative model `vocab_size` for the pinned model; the value itself is right.
- **Not D (insufficient evidence):** the cause is established with file-level, revision-level, and version-level evidence.
- **B (Measurement/API semantic defect):** the correct tokenizer/model is present and matches the pinned revision, but Stage 2 measures the wrong vocabulary metric (`len(tokenizer)` instead of `model.config.vocab_size`).

## 6. Comparison to Frozen Sources (Protocol v3.1, Plan v4.0, 6E.18 machinery)

- The pinned `Qwen/Qwen2.5-0.5B-Instruct` model's `config.json` `vocab_size = 151936` is the authoritative basis for the invariant `VOCAB = 151936`. This is consistent with the frozen Protocol v3.1 / Plan v4.0 pinning of that exact model+revision.
- The 6E.18 / 6E.20.1 lineage established model-vocabulary identity via the **model's** `vocab_size` (embedding dimension). The 6E.20.2 Stage 2 substituted `len(tokenizer)` for that check, introducing a semantic mismatch that the genuine tokenizer cannot satisfy (the real tokenizer length is 151665).
- No frozen artifact, protocol, plan, or 6E.18 source is inconsistent with `151936` as the **model** vocab size; the defect is purely in *which API quantity* Stage 2 compares to it.

## 7. Authorization Status

- The active L5 authorization `execution_authorization_6e20_2.json` is **unchanged** (SHA-256 `6A54E01421C65D2ABAD82B1A622787C9D6DD730A5CE2C11BCE131AFB73D9FBB2` verified post-halt). It remains **cryptographically valid** for the **current, unmodified** runner (`58D6CB93…`) and decision (`APPROVE`).
- The halted attempt stopped at a **preflight invariant before any evaluation**; no scientific computation, checkpoint load, or output occurred. The authorization was therefore **NOT consumed or invalidated** by the halt.
- Execution is, however, **operationally blocked** pending root-cause resolution. Because the required remediation modifies `run_phase_6e20_reevaluation.py` (changing its SHA-256), the **current authorization will no longer bind the corrected runner**. Consequently, after remediation + re-audit, a **new L5 authorization must be issued binding the post-remediation runner hash** before any execution may proceed. The existing authorization is valid-but-superseded-once-the-runner-changes.

## 8. Recommended Remediation Path (NOT implemented here)

- In `run_phase_6e20_reevaluation.py`, `run_stage2_model_and_tokenizer_preflight()` (lines ~343–366): replace the `len(tokenizer) != VOCAB` check with a check against the **model's** vocabulary identity, e.g.:
  - `assert model.config.vocab_size == VOCAB`  (VOCAB = 151936, the model embedding vocab), and/or
  - add consistency assertions: `tokenizer.vocab_size == 151643` and `len(tokenizer) == 151665`, so the tokenizer's base+added structure is also validated — but the *gating* comparison must use the model `vocab_size`, not `len(tokenizer)`.
- Do **not** alter `MODEL_ID`, `MODEL_REVISION`, `VOCAB=151936`, any frozen artifact, protocol, plan, or the scientific `rho_c`/`sigma_c` logic.
- After the change: re-run the synthetic suite (expect 16/16), recompute the runner SHA-256, and re-issue the L5 authorization binding the new runner hash; then re-attempt execution under the fresh authorization.

**Leave implementation for a separate authorized step.**
