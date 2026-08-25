# FORENSIC 6E.20.2 — FINAL PRE-AUTHORIZATION SOURCE AUDIT

**Mode:** SOURCE AUDIT ONLY — no execution, no model load, no checkpoint `torch.load`, no code modification, no authorization file creation.
**Date:** 2026-08-18
**Auditor:** opencode (read-only forensic)
**Scope:** Pre-authorization review of the Phase 6E.20.2 Read-Only Checkpoint Re-Evaluation artifacts to determine whether they are safe and scientifically valid to authorize.

---

## §1 Executive Summary & Authorization Recommendation

| Item | Finding |
|---|---|
| Condition-resolved evaluation (each ckpt with its OWN readout) | **DESIGN PASS** — correctly implemented in Stage 6 loop |
| Live `rho_c` computation (H3 readout-steering metric) | **CRITICAL FAIL** — degenerate, does not compute readout-total-gradient alignment |
| Historical B0 reproducibility anchor | **PASS** — hard-codes S8 readout, matches frozen runs.json within 1e-4 |
| SHA-256 integrity manifest | **PASS** (with provenance anomaly — see §8) |
| Non-silent LoRA restoration | **PASS** |
| Output isolation / immutability | **PASS** (Stage 7 re-manifest) |
| Authorization guard | **PASS** (checked before `--execute`) |
| Model/tokenizer identity | **PASS** (Stage 2) |
| Decision-logic verbatim match to Protocol v3.1 | **PASS** |
| Test-suite coverage of critical compute paths | **FAIL** — no coverage of `compute_live_rho_c`, `compute_live_cos_H1`, `reconstruct_panel_prompts`, Stage 1 |
| Draft-auth hash consistency (runner + test) | **PASS** |
| Frozen-invariant compliance | **PASS** (design) |

### Authorization Recommendation: **A — DO NOT AUTHORIZE (as-is)**

The runner is **not safe to authorize in its current form**. One defect is execution-blocking for scientific validity:

- **`compute_live_rho_c` is mathematically degenerate.** It returns `≈ 0.0 ± sign(alpha_i)` for every condition because it aligns the readout gradient with the readout direction itself (a constant), never touching the genuine total gradient `g_total`. Consequently **H3 (readout steering) can never be SUPPORTED** — it is decided on garbage input. This is the same defect family as 6E.20.1's `rho_c = 1.0` constant, not a fix.

Secondary findings: the test suite does not exercise any gradient/rho computation (so it cannot catch the above), and a provenance anomaly exists (`execution_authorization_v6.json` is byte-identical to v7's auth). All are remediable; the `cos(H1)` path (H1/H4/H5) is correctly implemented and scientifically valuable once `rho_c` is fixed.

**Required remediations before re-audit:** (1) rewrite `compute_live_rho_c` to compute genuine `cos(g_readout, g_total)` via the frozen 6E.18 machinery; (2) fix or remove the spurious `auth_6e20_1_v6` Stage-1 check (line 293) and investigate the v6/v7 auth identity; (3) extend the unit test suite to cover `compute_live_rho_c` and `compute_live_cos_H1` on synthetic vectors. After remediation this can move to **C — READY TO FREEZE**.

---

## §2 Scope & Method

- Reviewed three audit targets (see §3) by full source read. No file was executed; no `torch` import occurred; no checkpoint was loaded.
- Verified file hashes with `Get-FileHash` (SHA-256) — read-only.
- Compared code against frozen Protocol v3.1 (`E16CA4F1…`), Plan v4.0 (`87C8E486…`), and the 6E.20.1 frozen `runs.json` anchor values.

---

## §3 Audit Target Inventory

| File | Role | SHA-256 (uppercase) | Status |
|---|---|---|---|
| `run_phase_6e20_reevaluation.py` | 6E.20.2 evaluator (959 lines) | `011063DB857425C1F336B61C24754E5FD5F4A8EB7D8CCFEDEAB5A324EF2346A1` | Matches draft-auth `runner_sha256` |
| `test_phase_6e20_reevaluation.py` | Synthetic unit test suite (285 lines) | `9FCB5DD99253FA8D661E7C8B5C91FEC1264B1D5A8D6F94651ADCAECE4C484719` | Matches draft-auth `test_suite_sha256` |
| `execution_authorization_6e20_2_DRAFT.json` | Draft L5 authorization (26 lines) | n/a (not yet signed; file name is `_DRAFT`, not the runner-expected `execution_authorization_6e20_2.json`) | Pending human L5 sign-off |

**Provenance anomaly:** `execution_authorization_v6.json` and `execution_authorization_v7.json` both hash to `5214DA997764F62C9370238E055CD5394925FBFE8F0C61478B8D192B198C4F91`. The "v6" file is therefore NOT a distinct historical v6 authorization — it is content-identical to the v7 auth. This contradicts the frozen-artifact record that v6 auth = `4AA21378…`.

---

## §4 Audit Matrix

| Audit | Subject | Verdict |
|---|---|---|
| A | Condition-resolved evaluation (own ckpt + own readout, 7×{4,6,14,15}) | **PASS** |
| B | Live `rho_c` genuine readout-total alignment | **FAIL** (critical) |
| C | Historical B0 reproducibility anchor | **PASS** |
| D | SHA-256 integrity manifest | **PASS** (anomaly noted, §8) |
| E | Non-silent LoRA adapter restoration | **PASS** |
| F | Output isolation & post-exec immutability | **PASS** |
| G | Authorization guard before execute | **PASS** |
| H | Model/tokenizer identity verification | **PASS** |
| I | Decision-logic verbatim vs Protocol v3.1 | **PASS** |
| J | Test-suite coverage of critical compute | **FAIL** |
| K | Draft-auth runner/test hash consistency | **PASS** |
| L | Frozen-invariant (no 6E.20.1 mutation) compliance | **PASS** |
| M | B0 anchor gates before 28 evaluations | **PASS** |

**Totals:** 11 PASS, 2 FAIL, 0 UNKNOWN.

---

## §5 Audit A — Condition-Resolved Evaluation (PASS)

Stage 6 (lines 830–873) loops over `CONDITIONS` and `AGGREGATION_SET`, loading each condition's own checkpoint and passing `readout_c = readouts[c]` into `compute_live_cos_H1`. This is the correct design that 6E.20.1 lacked (6E.20.1 hard-coded B0-checkpoint + S8-readout for all conditions). The 6E.20.1 causal-analysis finding (H1 "untested due to wrong readout/checkpoint pairing") is structurally addressed here. **PASS.**

---

## §6 Audit B — Live `rho_c` Computation (CRITICAL FAIL)

`compute_live_rho_c` (lines 545–580) claims to compute "readout-total alignment `rho_c` per protocol v3.1 §5.7." It does **not**. Proof:

```
w_vec = readout.w ;  u_w = w_vec / ||w_vec||
For each sample i:
    s_i      = forward_margin(h_i)                 # readout scalar
    y_i      = 1 if i<8 else 0
    p_pro    = sigmoid(s_i)
    alpha_i  = (p_pro - 1.0) if y_i==1 else p_pro
    g_R      = alpha_i * w_vec                     # readout-gradient w.r.t. hidden h
    alignment = dot(g_R, u_w) / ||g_R||
```

`dot(alpha_i·w_vec, w_vec/||w_vec||) = alpha_i·||w_vec||`. `||g_R|| = |alpha_i|·||w_vec||`.
⇒ `alignment = alpha_i / |alpha_i| = sign(alpha_i) ∈ {−1, +1}` (for `alpha_i ≠ 0`).

Per-sample `rho ∈ {−1,+1}`; over 16 panel prompts: 8 positives give `alpha = p−1 < 0 ⇒ −1`, 8 negatives give `alpha = p > 0 ⇒ +1` ⇒ **mean `rho_c = 0.0` for every condition.**

Consequences:
1. `rho_c` never uses `g_total` — the function signature (lines 545–551) does not even receive `psdg_fn`/`crgc_fn`. It measures the readout gradient aligned with *itself*, a constant, not the readout-total gradient alignment the protocol requires.
2. H3 (`evaluate_protocol_hypotheses`, lines 670–674) requires `rho_S8 >= 0.50` → with `rho_c ≈ 0.0`, **H3 is REFUTED for all conditions, unconditionally.** The readout-steering hypothesis is rendered untestable.
3. This is the same defect *class* as 6E.20.1's `rho_c = 1.0` constant (readout-only path, no `g_total`) — i.e., the historical defect is **not fixed**, merely changed from `1.0` to `0.0`.

**Required fix:** compute genuine `rho_c = mean_i cos(g_readout_i, g_total_i)` where `g_total_i` comes from the frozen `psdg_fn`/`crgc_fn` machinery (as `compute_live_cos_H1` already does) and `g_readout_i` is the readout-induced gradient direction. **FAIL.**

---

## §7 Audit C — Historical B0 Reproducibility Anchor (PASS)

`run_stage5_b0_historical_reproducibility_anchor` (lines 583–631) recomputes B0 checkpoints using `readouts["S8"]` (line 817) — correctly mirroring the *historical* B0-with-S8 evaluation path — and asserts `|recomputed − HISTORICAL_B0_S8_ANCHOR| ≤ 1e-4` where the anchor values (`{4:0.428355, 6:0.409713, 14:0.427470, 15:0.417450}`) equal the frozen 6E.20.1 `runs.json` `cos(H1)`. The gate runs *before* Stage 6 (M: PASS) and the docstring explicitly states passing validates environment/prompt parity only, not H1. The genuine-gradient path (`compute_live_cos_H1`) is used, so the anchor is meaningful. **PASS.**

---

## §8 Audit D — SHA-256 Integrity Manifest (PASS, anomaly)

`run_stage1_pre_execution_sha_manifest` (lines 277–338) verifies protocol, plan, 6E.18 snapshot, base/v7 runners, v6/v7 auths, corpus, all `step0/*.npy`, all 7 `w_{c}.npy`, and all 28 checkpoints.

- Corpus expected hash `a7b4e845…` (line 54) matches the verified deduplicated-corpus hash — **consistent**.
- **Latent bug (non-fatal):** line 293 maps `auth_6e20_1_v6` to `AUTH_6E20_1_V7_SHA256` (there is no `AUTH_6E20_1_V6_SHA256` constant). In principle this would HALT Stage 1 if the v6 file differed from v7. It does **not** halt here only because, on disk, `execution_authorization_v6.json` hashes to `5214DA99…` — identical to v7 (see §3). This is a copy-paste correctness defect *and* a provenance red flag (the "v6" auth is not a distinct historical artifact). Recommend defining the correct expected hash or dropping the spurious check. **PASS-with-anomaly.**

---

## §9 Audit E — Non-Silent LoRA Restoration (PASS)

`run_stage3_stage4_checkpoint_smoke_and_lora_validation` (lines 367–452) `torch.load`s all 28 checkpoints (addressing the feasibility-audit P6 gap), checks root keys `state_dict/step/condition/readout_w/readout_b`, enforces `expected_lora_keys − ckpt_lora_keys` non-empty (line 409–416, raises `PreflightError` on silent drop), and verifies non-zero adapter norms. `strict=False` load + key-set diff is the correct anti-silent-drop pattern. Test 8 also validates missing-key detection. **PASS.**

---

## §10 Audit F — Output Isolation & Immutability (PASS)

All telemetry/manifests/reports are written exclusively to `phase-6e20.2-reevaluation-output/` (lines 764, 784, 819, 875, 882, 890). Stage 7 (lines 879–885) re-runs the Stage-1 manifest post-execution to confirm zero mutation of frozen 6E.20.1 files. **PASS.**

---

## §11 Audit G — Authorization Guard (PASS)

`check_authorization` (lines 722–745) is invoked in `main()` *before* `run_evaluation_pipeline` on the `--execute` path (line 940), refusing on missing file, `decision != APPROVE`, `phase != PHASE_ID`, or runner-SHA mismatch. Draft auth name `_DRAFT` differs from the expected `execution_authorization_6e20_2.json`, so it is correctly **not** yet active. **PASS.**

---

## §12 Audit H — Model/Tokenizer Identity (PASS)

Stage 2 (lines 341–364) pins `MODEL_ID = Qwen/Qwen2.5-0.5B-Instruct`, `MODEL_REVISION = 7ae55760…`, and asserts vocab `151936`, token IDs `PRO_ID=9117`, `AB_ID=1867`, `hidden_size=896`, `num_hidden_layers=24`. Matches frozen base config. **PASS.**

---

## §13 Audit I — Decision Logic Verbatim (PASS)

`evaluate_protocol_hypotheses` (lines 637–716) implements:
- H1: `mean_cos(S4)≥0.80 AND mean_cos(S8)≥0.80`
- H4: `mean_cos(S4)−mean_cos(C4-Rand)≥0.30 AND C4-Rand d(15)<0.80`
- H5: `R1-Rand d(15)<0.80 AND F1 d(15)≥0.80`
- H3: `rho_S8≥0.50 AND rho_S8≥rho_S4≥rho_S2`

These match Protocol v3.1 §7/§8/§9.2 verbatim. (Note: H3's *inputs* are broken by Audit B, but the *rule* itself is correct.) **PASS.**

---

## §14 Audit J — Test-Suite Coverage (FAIL) & Final Decision

`test_phase_6e20_reevaluation.py` imports only `{AGGREGATION_SET, CONDITIONS, H4_DELTA_MIN, P1–P5_THRESHOLD, PHASE_ID, PreflightError, Readout, check_authorization, compute_sha256, ensure_readout_class_patch, evaluate_protocol_hypotheses}`. It tests:
- readout forward math & tensor patch,
- decision-engine verdicts (all-pass / H1 / H4 / H3 refuted cases),
- synthetic sign-gate convention,
- LoRA missing-key detection (logic only, no real tensors),
- `check_authorization` refusals.

**It does NOT import or test** `compute_live_rho_c`, `compute_live_cos_H1`, `reconstruct_panel_prompts`, or Stage-1/3/5. Therefore the unit suite **cannot detect the Audit-B degeneracy nor the §8 latent bug**. The most scientifically critical computation is entirely uncovered. **FAIL.**

### Final Decision

**A — DO NOT AUTHORIZE the current runner.** The `rho_c`/H3 path is scientifically invalid (Audit B), the safety net is inadequate (Audit J), and a provenance anomaly (§3/§8) requires explanation. Remediations (§1) are straightforward and the `cos(H1)` path is sound, so the phase remains worth pursuing after a corrected re-audit.

*No code was modified, no authorization created, and no execution performed during this audit.*
