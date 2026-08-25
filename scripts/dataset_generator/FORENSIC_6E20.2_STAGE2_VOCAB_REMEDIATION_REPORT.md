# Forensic Remediation Report — Phase 6E.20.2 Stage 2 Tokenizer Vocabulary Check

**Phase:** 6E.20.2 (Read-Only Checkpoint Re-Evaluation)
**Remediation date:** 2026-08-19
**Authorizing constraint:** Narrowly scoped remediation of Classification B defect only.
**Status:** REMEDIATED · NOT EXECUTED · PENDING FRESH INDEPENDENT PRE-AUTHORIZATION SOURCE AUDIT

---

## 1. Root Cause Classification

**Classification B — Measurement / API semantic defect.** (Accepted per human directive.)

The pinned model revision and the frozen invariant `VOCAB = 151936` are **correct**. The
defect was that Stage 2 compared the **wrong vocabulary metric** (`len(tokenizer)`) against the
model embedding/config vocabulary size.

- The correct, pinned tokenizer/model is present in the HF cache (revision
  `7ae557604adf67be50417f59c2c2f167def9a775` — exactly the pinned `MODEL_REVISION`). Not a
  wrong/stale cache → rules out A.
- `model.config.vocab_size = 151936` is the authoritative model embedding vocabulary for
  `Qwen/Qwen2.5-0.5B-Instruct` → the frozen invariant value is **correct** → rules out C.
- `tokenizer.vocab_size = 151643` (base vocabulary) and `len(tokenizer) = 151665`
  (base + 22 added special tokens). The old check `len(tokenizer) != VOCAB` compares
  151665 against 151936 and therefore **always fails on the valid pinned configuration** —
  a measurement/API semantic error, not an artifact or value error.

---

## 2. Exact Semantic Difference Between the Three Vocabulary Metrics

| Metric | Source | Value for pinned `Qwen/Qwen2.5-0.5B-Instruct` | Meaning |
|---|---|---|---|
| `model.config.vocab_size` | Model `config.json` embedding matrix width | **151936** | Authoritative model embedding vocabulary. **This is what must equal the frozen invariant `VOCAB`.** |
| `tokenizer.vocab_size` | Tokenizer base vocabulary file | **151643** | Size of the base (pre-trained) vocabulary, excluding added special tokens. Must NOT be compared to `VOCAB`. |
| `len(tokenizer)` | Base + added special tokens | **151665** | Full tokenizer size (151643 base + 22 added tokens). Must NOT be compared to `VOCAB`. |

Verified live from cache (read-only, no download/execution):
`tokenizer.vocab_size == 151643`, `len(tokenizer) == 151665`.

**Key point:** `len(tokenizer)` (151665) and `tokenizer.vocab_size` (151643) can **never**
equal `model.config.vocab_size` (151936). The invariant must gate on `model.config.vocab_size`,
and the tokenizer metrics may only be checked for **internal consistency** against their own
correct semantics (151643 and 151665 respectively).

---

## 3. Exact Code Changes

### 3.1 Runner — `run_phase_6e20_reevaluation.py` (`58D6CB93…` → `16BEF31E…`)

**(a) New constants** (near `VOCAB`):
```python
VOCAB = 151936
TOKENIZER_BASE_VOCAB = 151643
TOKENIZER_FULL_VOCAB = 151665
LAYER_INDEX = 24
```

**(b) New pure, fail-closed helper** (placed immediately before
`run_stage2_model_and_tokenizer_preflight`):
```python
def validate_stage2_vocab_invariant(model_config_vocab_size, tokenizer_base_vocab, tokenizer_full_len):
    """Fail-closed Stage 2 vocabulary invariant.

    The authoritative model embedding vocabulary is ``model.config.vocab_size``
    (VOCAB = 151936) and must be compared to VOCAB, never to a tokenizer metric.
    ``tokenizer.vocab_size`` (base vocabulary, 151643) and ``len(tokenizer)``
    (base + added special tokens, 151665) are checked only for consistency and
    must NOT be compared against VOCAB. All deviations are fail-closed.
    """
    if model_config_vocab_size != VOCAB:
        raise PreflightError(
            f"Stage 2 HALT: Model config vocab mismatch {model_config_vocab_size} != {VOCAB}"
        )
    if tokenizer_base_vocab != TOKENIZER_BASE_VOCAB:
        raise PreflightError(
            f"Stage 2 HALT: Tokenizer base vocab mismatch {tokenizer_base_vocab} != {TOKENIZER_BASE_VOCAB}"
        )
    if tokenizer_full_len != TOKENIZER_FULL_VOCAB:
        raise PreflightError(
            f"Stage 2 HALT: Tokenizer full vocab mismatch {tokenizer_full_len} != {TOKENIZER_FULL_VOCAB}"
        )
```

**(c) `run_stage2_model_and_tokenizer_preflight()` — replaced the invalid invariant.** The old
code:
```python
    if len(tokenizer) != VOCAB:
        raise PreflightError(f"Stage 2 HALT: Vocab mismatch {len(tokenizer)} != {VOCAB}")
```
is replaced by:
```python
    validate_stage2_vocab_invariant(model.config.vocab_size, tokenizer.vocab_size, len(tokenizer))
```

**Unchanged by design:** `MODEL_ID`, `MODEL_REVISION`, `VOCAB = 151936`, all frozen artifact
paths/hashes, all checkpoint/evaluation logic, all hypothesis logic, and the existing
PRO_ID/AB_ID tokenizer identity checks (which remain in the function, after the helper call).
The check remains fail-closed via `PreflightError`; no warnings, fallback values, silent
substitutions, or automatic downloads/revision changes were introduced.

### 3.2 Test suite — `test_phase_6e20_reevaluation.py` (`0B4A9A51…` → `55A71B41…`)

Added imports: `TOKENIZER_BASE_VOCAB`, `TOKENIZER_FULL_VOCAB`, `VOCAB`,
`validate_stage2_vocab_invariant`.

Added **Test 12** (5 synthetic sub-cases) exercising only the pure helper with scalar
arguments — no real model/tokenizer load, no download, inference, or checkpoint access:

- **12a `stage2_valid_config_passes`** — valid pinned config
  `(VOCAB, TOKENIZER_BASE_VOCAB, TOKENIZER_FULL_VOCAB)` **passes** even though
  `len(tokenizer)=151665 != VOCAB=151936`.
- **12b `stage2_wrong_model_vocab_fails`** — `model.config.vocab_size = VOCAB-1` **fails**.
- **12c `stage2_wrong_tokenizer_base_vocab_fails`** — `tokenizer.vocab_size = 151642` **fails**.
- **12d `stage2_wrong_tokenizer_full_vocab_fails`** — `len(tokenizer) = 151664` **fails**.
- **12e `stage2_old_invalid_comparison_rejects_valid_config`** — demonstrates the **old**
  comparison (`len(tokenizer) != VOCAB`) would **reject** the known-valid synthetic
  configuration (151665 != 151936), reproducing the root-cause-B defect.

---

## 4. Test Results

Full synthetic suite run from the remediated code:

```
ALL_PASS True 21 / 21
```

New Stage 2 sub-cases (all PASS):
- `stage2_valid_config_passes :: raised=None`
- `stage2_wrong_model_vocab_fails :: halted=True`
- `stage2_wrong_tokenizer_base_vocab_fails :: halted=True`
- `stage2_wrong_tokenizer_full_vocab_fails :: halted=True`
- `stage2_old_invalid_comparison_rejects_valid_config :: old_len(tokenizer)=151665 != VOCAB=151936`

Both modified files pass `python -m py_compile` (exit 0). The pre-existing 16 tests
(readout math, decision logic, rho_c genuineness, sigma_c = mean(ρ²), authorization
refusals, frozen 6E.18 cosine core, etc.) continue to pass; total rose from 16 to 21.

---

## 5. Old and New SHA-256

| Artifact | Old SHA-256 | New SHA-256 |
|---|---|---|
| Runner `run_phase_6e20_reevaluation.py` | `58D6CB9331E5E092678046FFC756AA24578335F6F63F131E89707678A20A2A9F` | `16BEF31E56D781170021092052CF708A92875EE68697FEB81FB77764E15E691C` |
| Test suite `test_phase_6e20_reevaluation.py` | `0B4A9A518A644FE8391A639173AD8F6A4EE257D3576DFC6482E20E90AADC7C3C` | `55A71B4115BA050FA22C3E83290A6B5D4C1FB45977D6448EF24000E03E5530A1` |

---

## 6. Authorization Status

- **Active L5 authorization** (`execution_authorization_6e20_2.json`, hash
  `6A54E01421C65D2ABAD82B1A622787C9D6DD730A5CE2C11BCE131AFB73D9FBB2`) **is unchanged** and
  still binds the **OLD** runner hash
  `58D6CB9331E5E092678046FFC756AA24578335F6F63F131E89707678A20A2A9F`.
- Because the runner's SHA-256 changed (`58D6CB93…` → `16BEF31E…`), the active authorization's
  `runner_sha256` no longer matches the remediated runner. The active L5 authorization is
  therefore **cryptographically bound to the old runner hash and CANNOT authorize the
  remediated runner**. It was **not** modified.
- The **DRAFT** (`execution_authorization_6e20_2_DRAFT.json`, non-authorizing) was updated
  **only** with the new runner/test hashes
  (`16BEF31E…` / `55A71B41…`). It remains a DRAFT (no L5 sign-off) and is not an active
  authorization.
- **No new L5 authorization was created or activated.**
- The active authorization was **not consumed or invalidated** by the earlier preflight halt
  (no evaluation occurred); it remains valid only for the old, unmodified runner.

---

## 7. Execution Status

- **No `--execute` run was performed** during this remediation.
- **No execution occurred** at any point. The Stage 2 halt remains the only execution contact,
  and it halted at the preflight invariant before any checkpoint load or evaluation.
- **No frozen 6E.20.1 artifact, protocol, plan, checkpoint, provenance, or prior halt record**
  was modified. The old `4AA21378…` auth was not reused.

---

## 8. Next Step (outside this task's authority)

The remediated artifacts are submitted for a **fresh independent pre-authorization source
audit**. A new L5 authorization that binds the post-remediation runner hash
(`16BEF31E…`) must be issued **only after** that audit approves, before any `--execute`.
