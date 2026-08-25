# FORENSIC 6E.20.1 — READ-ONLY RE-EVALUATION FEASIBILITY AUDIT

**Phase:** 6E.20.1 — Fixed-Subspace Readout Execution Extension (v7)
**Audit date:** 2026-08-18
**Mode:** READ-ONLY feasibility audit of the 56 frozen checkpoints for non-mutating re-evaluation
**Constrained by:** 6E.20.1 is CLOSED. No artifact was modified, moved, deleted, retrained, regenerated, monkeypatched, executed, or inferred. This audit confirms *reconstructability only*; it does not run inference or recomputation. Any re-evaluation described below is a **recommendation only** and requires a new phase + explicit L5 authorization.

---

## 1. Objective

Determine whether the 56 frozen checkpoints can be correctly re-evaluated **without retraining**, such that each of the seven conditions is measured with (a) its own checkpoint and (b) its own protocol-defined readout, thereby directly testing the previously-untested H1/P2 claim (causal report §6, Finding F1).

---

## 2. Verdict

### ✅ FEASIBLE (subject to two external-but-pinned dependencies)

All reconstruction dependencies required to re-evaluate every condition with its own checkpoint and its own readout are present in the frozen artifacts, and the evaluation pipeline (F7 total-gradient machinery, LoRA restore, panel reconstruction, readout reconstruction) is fully determined by pinned constants. **No retraining is required.**

The only dependencies lying *outside* the frozen repository are:
1. **Base model weight + tokenizer** — `Qwen/Qwen2.5-0.5B-Instruct` @ revision `7ae557604adf67be50417f59c2c2f167def9a775` (HuggingFace; external but revision-pinned). Existed at run time.
2. **Corpus file** — `theo_slm_v0_deduplicated/candidate_records.json`, hash-pinned `a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0` (external but hash-pinned). Required only to deterministically reproduce the 16 panel prompts.

Neither is a defect in the frozen 6E.20.1 artifacts; both are standard, pinned inputs. If either is unavailable at re-evaluation time, that is an environment issue, not a reproducibility gap in the experiment record.

---

## 3. Dependency Audit (evidence-backed)

### 3.1 Base model & tokenizer pinning — ✅
- `MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"` (`run_phase_6e20_fixed_subspace_readout.py:65`).
- `MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"` (line 66).
- Tokenizer loaded as `AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)` with **no extra settings** (v7 `main`, line 1258/1290/1311) → default, deterministic formatting/tokenization.
- A built-in self-check (`frozen runner:883-887`) hard-asserts `PRO_ID==9117, AB_ID==1867, D_HIDDEN==896, VOCAB==151936, LORA_R==16, …, MODEL_ID/REVISION` exactly; the runner refuses to run otherwise. This pins the scientific identity of the model+tokenizer+vocab.

### 3.2 LoRA reconstruction — ✅
- `LORA_R=16, LORA_ALPHA=32, LORA_DROPOUT=0.05` (frozen runner:59-61).
- `LORA_TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` (line 62).
- `LORA_BIAS = "none"`, `LORA_TASK_TYPE = "CAUSAL_LM"` (lines 63-64).
- Each checkpoint was written by `train_condition` as `{"state_dict": model_with_lora.state_dict(), "step", "condition", "readout_w", "readout_b"}` (v7:626-630) using this exact `LoraConfig`.
- Restoration uses `get_peft_model(model, lora_config).load_state_dict(ckpt["state_dict"], strict=False)` (v7 CHANGE 3 / `restore_checkpoint:711-727`). This was **proven to work for B0** in the final valid run (PID 27560 produced cos(H1) values, which required restoring B0 checkpoints). All seven conditions were trained by the identical `train_condition` code with the identical `LoraConfig`, so their `model.pt` files share the same adapter key structure → restoration is structurally guaranteed for all.

### 3.3 Checkpoint metadata — ✅ (presence verified; format proven)
- **Presence:** glob of `phase-6e20.1-execution-output/*/checkpoint_*/model.pt` returns **56/56** files — exactly 8 per condition × 7 conditions (B0, F1, S2, S4, S8, C4-Rand, R1-Rand).
- **Identification:** condition and step are encoded both in the directory path (`<COND>/checkpoint_<t>/`) and in the `model.pt` metadata (`step`, `condition` per v7:627-628; read back by `restore_checkpoint:721-727`).
- **Expected set present for every condition:** `{0,4,6,7,14,15,17,20}` — confirmed by the glob listing (each condition shows checkpoint_0,4,6,7,14,15,17,20).
- **Readability:** not exercised here (no `torch.load`, per no-execution constraint). Supported by (i) the explicit, deterministic write schema and (ii) the successful B0 restore in PID 27560. A future authorized re-run should still confirm a `torch.load` on all 56 as a first step; this is a verification step, not a reconstruction gap.

### 3.4 Frozen readout reconstruction — ✅
- `Readout` is a plain `(name, w, b, trainable_params, subspace)` dataclass; `forward_logits(h) = (w·h+b, -(w·h+b))` (`frozen runner:241-254`). **No hidden runtime-only state** beyond `(w,b)`.
- All seven readout weight vectors are saved: `step0/w_{B0,F1,S2,S4,S8,C4-Rand,R1-Rand}.npy` (confirmed by glob).
- `b` reconstructable exactly:
  - B0: `b = 0.0` (constant; `build_readouts:287`).
  - Others: `b = -median(w · H16[:8])` (`build_readouts:289`), where `H16[:8]` = first 8 rows of saved `step0/H_16.npy` (present, hash-bound in `sha256_manifest.json`). Also redundantly stored as `readout_b` inside each `model.pt`.
- `H_16.npy` present and hash-bound; `u_lda.npy`, `U_8.npy` (→ `U_2=U_8[:,:2]`, `U_4=U_8[:,:4]`), `R_4.npy`, `n_rand.npy` all present (so the readout *definitions* are independently re-derivable from `build_readouts`, not just the saved `w`).
- The evaluation uses only `w` and `b` (via `forward_logits`/`forward_margin`); no `U_k` evaluation-time dependency.

### 3.5 Evaluation determinism — ✅
- **Panel prompts:** `collect_panel_prompts` (`v7:244-261`) rebuilds the 16 prompts from the hash-pinned corpus via fixed seed-42 logic (`reconstruct_data:217-241`), producing exactly 8 POS + 8 ABS with labels `y=[1]*8+[-1]*8`. Fully deterministic; no stored artifact required.
- **Prompt template:** `format_prompt` (`v7:203-214`) is a fixed string template, no RNG.
- **Eval mode:** `model.eval()` and gradient-free forward in `evaluate_checkpoint_at_state` (v7:754) and `per_sample_decision_gradient` (6E.18:1011-1032).
- **Device/dtype:** model on `cuda:0`; CHANGE 2 patches `readout.forward_logits` to `.detach().cpu().float()` the hidden tensor *before* the frozen `np.asarray` (v7:402-407). This is a runtime-only patch (no file modification) and preserves scientific semantics: the autograd gradient `∇_{lora} dec_loss` is computed on-GPU in native dtype; only the readout's numpy input is moved to CPU float, which cannot alter the gradient object. Confirmed in the causal report as the correct, protocol-preserving fix.
- **F7 machinery:** `per_sample_decision_gradient` + `COMPUTE_RAW_GRADIENT_COSINE` imported verbatim from the hash-verified frozen 6E.18 snapshot (`6BCAF131…`); sign-gate (G6) satisfied (runs.json).

---

## 4. Coverage Matrix (7 conditions × 8 checkpoints)

Legend: **Ck** = checkpoint file present (glob-verified); **Rd** = readout reconstructable from `w_{name}.npy` + `b`; **Ev** = evaluable with own checkpoint + own readout (Ck ∧ Rd). All cells ✅.

| Condition | Readout source (reconstruction) | Tests | ck0 | ck4 | ck6 | ck7 | ck14 | ck15 | ck17 | ck20 |
|---|---|---|---|---|---|---|---|---|---|---|
| **B0** (baseline) | `w_B0.npy`, b=0 | Baseline (not required to pass P2) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **F1** (fixed LDA) | `w_F1.npy`, b=-med(w·H16[:8]) | Reference fixed-readout (6E.18 F1 analog) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **S2** (k=2) | `w_S2.npy`, b=-med(w·H16[:8]) | Treatment (subspace, k=2) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **S4** (k=4) | `w_S4.npy`, b=-med(w·H16[:8]) | **H1 target** (subspace, k=4) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **S8** (k=8) | `w_S8.npy`, b=-med(w·H16[:8]) | **H1 target** (subspace, k=8) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **C4-Rand** | `w_C4-Rand.npy`, b=-med(w·H16[:8]) | Control (H4: random 4D) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **R1-Rand** | `w_R1-Rand.npy`, b=-med(w·H16[:8]) | Control (H5: random 1D) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Total: 56/56 checkpoints present; 7/7 readouts reconstructable; 56/56 evaluable.** Every cell satisfies both "own checkpoint" and "own protocol-defined readout."

---

## 5. Does this directly test the previously-untested H1/P2?

**Yes.** The causal report (§6, F1) established that `run_evaluation` hard-coded B0 checkpoints + the S8 readout and never computed `cos(H1)` for S4/S8 — leaving H1 ("Both S4 and S8 achieve `mean cos(H1) ≥ +0.80`") untested. The re-evaluation procedure below computes `cos(H1)` for **each condition using that condition's own readout**, so:
- **S4 and S8** `cos(H1)` over the protocol aggregation set `AGGREGATION_SET = {4,6,14,15}` (frozen runner:69) directly evaluates **H1 and P2**.
- **C4-Rand / R1-Rand** evaluate the **H4 / H5** control contrasts.
- **F1** provides the fixed-LDA reference; **B0** the unembedding baseline.

This converts the current "B0-only, S8-probed" negative into a complete, condition-resolved measurement of the actual hypothesis.

---

## 6. Proposed Procedure (RECOMMENDATION ONLY — do not execute without new authorization)

1. Load `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604…` + tokenizer (default settings) onto `cuda:0`.
2. Apply `peft.get_peft_model` with `LoraConfig(r=16, alpha=32, dropout=0.05, target_modules=[q,k,v,o,gate,up,down]_proj, bias="none", task_type="CAUSAL_LM")` — identical to training (CHANGE 3).
3. Reconstruct the 7 `Readout` objects from `step0/w_{name}.npy` + `b` (`0` for B0; `-median(w·H16[:8])` otherwise), using the frozen `Readout` dataclass and hash-bound `H_16.npy`.
4. Reconstruct the 16 panel prompts deterministically via `collect_panel_prompts` (corpus hash `a7b4e845…`, seed 42) → 8 POS + 8 ABS, `y=[1]*8+[-1]*8`.
5. For each condition `c` ∈ {B0,F1,S2,S4,S8,C4-Rand,R1-Rand}, for each step `t` ∈ {0,4,6,7,14,15,17,20}:
   a. Restore `c`'s checkpoint: `get_peft_model(model).load_state_dict(ckpt["state_dict"], strict=False)` (proven for B0; structurally identical for all).
   b. Apply the CHANGE 2 runtime patch (`forward_logits` → `.detach().cpu().float()` before `np.asarray`) — no file modification.
   c. Compute `cos(H1)` via the imported frozen 6E.18 `per_sample_decision_gradient` + `COMPUTE_RAW_GRADIENT_COSINE`, using `c`'s Readout on the 16 panel prompts.
   d. (Recommended fix of causal-report F2/F4) Also compute the **live** `rho_c` (§5.7 readout–total alignment) and the per-condition margins/balanced-accuracy/Cohen's-d, and **persist all of them** in provenance (the original run discarded these — causal report F6).
6. Aggregate S4 and S8 over `{4,6,14,15}`; test H1/P2 (`mean cos(H1) ≥ +0.80`). Compare controls per H4/H5.
7. Emit full provenance (per-condition metrics + live `rho_c` + training telemetry if available from the 56 checkpoints' `readout_w/readout_b`/training is not re-run, so telemetry stays N/A).

No new evaluator code is required beyond correcting `run_evaluation` to loop over `CONDITIONS` and use each condition's own readout (the single change identified in the causal report, §8 rec. 1). The F7 machinery, LoRA config, readouts, panel, and checkpoints are all already reconstructable from frozen artifacts.

---

## 7. Remaining Blockers / Caveats

- **None internal to the frozen 6E.20.1 artifacts.** All 56 checkpoints, all 7 readouts, the F7 machinery, the panel reconstruction, and the LoRA/config constants are present and pinned.
- **External pinned inputs:** base model+tokenizer (HF, revision-pinned) and corpus (hash-pinned). Both existed at run time. If either is unavailable at re-evaluation, that is an environment dependency, not a record defect — and both are standard, verifiable artifacts.
- **Full `torch.load` readability of all 56** was not exercised here (per the no-execution constraint). It is supported by the deterministic write schema and the proven B0 restore, and should be the first verification step of any authorized re-run.

---

## 8. Conclusion

The 56 frozen checkpoints are **sufficient** to perform a new, non-mutating evaluation that correctly measures each condition using its own checkpoint and its own protocol-defined readout. **No retraining is required.** The re-evaluation would directly and for the first time test H1/P2 (S4/S8 `cos(H1) ≥ +0.80`), resolving the central gap identified in the causal analysis. Verdict: **FEASIBLE.**

*End of read-only feasibility audit.*
