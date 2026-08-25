# FORENSIC 6E.20.1 — STRICT READ-ONLY RE-EVALUATION FEASIBILITY AUDIT (PASS/FAIL/UNKNOWN)

**Phase:** 6E.20.1 — Fixed-Subspace Readout Execution Extension (v7)
**Audit date:** 2026-08-18
**Mode:** STRICTLY READ-ONLY. No frozen artifact, checkpoint, provenance, authorization, runner, protocol, or report was modified, deleted, moved, rewritten, or regenerated. No training, no inference, no recomputation, no `torch.load` of any checkpoint was performed. All checks are filesystem-level or code/constant inspection. Any re-evaluation described is a **recommendation only**, requiring a new phase + explicit L5 authorization.

---

## 1. Prerequisite Verdict Table

| # | Prerequisite | Verdict | Basis |
|---|---|---|---|
| P1 | Deterministic reconstruction of the 16 panel prompts + labels | **PASS** | Corpus hash verified equal to pinned expected; fixed seed-42 + fixed POS/ABS construction |
| P2 | Exact base-model identity / revision | **PASS** | `MODEL_ID`/`MODEL_REVISION` pinned; hard self-check |
| P3 | Exact tokenizer identity / revision / configuration | **PASS** | `AutoTokenizer.from_pretrained(MODEL_ID, revision=…)`; no extra settings |
| P4 | Exact LoRA architecture + config to load each checkpoint | **PASS** | `r=16,α=32,dropout=0.05,7 target modules,bias=none,task=CAUSAL_LM`; identical across all conditions |
| P5 | Integrity of all 56 checkpoints (filesystem) | **PASS** | 56/56 present, non-zero (~1.02 GB each), 0 corrupt/empty |
| P6 | Loadability of all 56 checkpoints (`torch.load` + key match) | **UNKNOWN** | Direct load NOT executed (per no-execution constraint); STRONGLY SUPPORTED by frozen save schema + proven B0 restore + identical architecture |
| P7 | Unambiguous condition → protocol-defined readout mapping | **PASS** | 1:1 code-defined map; all 7 `w_{name}.npy` present; `b` reconstructable |
| P8 | Separate read-only evaluator can compute H/P metrics without modifying frozen record | **PASS** | All inputs external to `runs.json`; new evaluator writes to a NEW location |
| P9 | Checkpoints sufficient to test H1/P2 **per condition** (not B0/S8-only) | **PASS** | Each condition's own checkpoint + own readout present/reconstructable; directly yields per-condition `cos(H1)` |

**Summary: 8 PASS, 1 UNKNOWN (P6), 0 FAIL.** The single UNKNOWN is a direct-load test that was intentionally not executed; it is supported to PASS by structural + empirical evidence (see §3.6).

---

## 2. P1 — Panel Prompt & Label Reconstruction — PASS

- `collect_panel_prompts` (`run_phase_6e20_execution_v7.py:244-261`) rebuilds 16 prompts (8 POS + 8 ABS) from `reconstruct_data` (`:217-241`), which uses `np.random.seed(42)` and a fixed family-split/balancing procedure over the corpus.
- **This audit verified the corpus hash**: `Get-FileHash` of `theo-data/.../theo_slm_v0_deduplicated/candidate_records.json` = `A7B4E84509B1C0DC03B81425125F061CEDAFE288CB18F153E8399B19CF717EB0`, exactly equal to `EXPECTED_CORPUS_HASH` (`v7:200`). **MATCH = YES.**
- Labels are fixed by construction: `y16 = [1]*8 + [-1]*8` (`v7:455`); `format_prompt` (`:203-214`) is a constant template with no RNG.
- **Conclusion:** given the corpus (hash-verified present), the exact 16 prompts + labels are reproducible with zero stored-prompt dependency.

## 3. P2/P3 — Base Model & Tokenizer Pinning — PASS

- `MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"`, `MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"` (`run_phase_6e20_fixed_subspace_readout.py:65-66`).
- Tokenizer loaded as `AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)` with **no additional arguments** (`v7 main:1258,1290,1311`) → default, deterministic tokenization/formatting.
- A built-in self-check (`frozen runner:883-887`) hard-asserts `PRO_ID==9117, AB_ID==1867, D_HIDDEN==896, VOCAB==151936, LORA_R==16, MODEL_ID/REVISION` exactly; the runner refuses otherwise. This pins the model+tokenizer+vocab scientific identity.

## 4. P4 — LoRA Architecture & Load Config — PASS

- `LORA_R=16, LORA_ALPHA=32, LORA_DROPOUT=0.05` (`frozen runner:59-61`); `LORA_TARGETS=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` (`:62`); `LORA_BIAS="none"`, `LORA_TASK_TYPE="CAUSAL_LM"` (`:63-64`).
- Checkpoints were written by `train_condition` as `model_with_lora.state_dict()` (v7:626-630) under this exact `LoraConfig`. To **load**, one reapplies `get_peft_model(model, lora_config)` (identical config) then `load_state_dict(ckpt["state_dict"], strict=False)` (v7:711-727; CHANGE 3 rationale v7:32-41, 1315-1325).
- All 7 conditions share the identical `LoraConfig` and save logic → identical adapter key structure and tensor shapes (rank `r=16` fixes `lora_A`/`lora_B` shapes). This audit's filesystem stat shows all 56 checkpoints are near-identical in size (1023617003–1023619435 bytes), consistent with a uniform LoRA architecture.

## 5. P5 — Checkpoint Integrity (Filesystem) — PASS

- Glob of `phase-6e20.1-execution-output/*/checkpoint_*/model.pt` → **56/56** files (8 per condition × 7 conditions: B0, F1, S2, S4, S8, C4-Rand, R1-Rand).
- This audit's stat: `COUNT=56`, `MIN_BYTES=1,023,617,003`, `MAX_BYTES=1,023,619,435`, `ZERO_BYTE_FILES=0`, `TOTAL_BYTES=57,322,652,008`.
- Condition and step are encoded in the path (`<COND>/checkpoint_<t>/`) and in `model.pt` metadata (`step`, `condition`; v7:627-628; read back at v7:721-727). Expected set `{0,4,6,7,14,15,17,20}` present for every condition.
- **Conclusion:** no missing, empty, or truncated checkpoints at the filesystem level.

## 6. P6 — Checkpoint Loadability (Direct `torch.load`) — UNKNOWN

- **Not executed** in this audit (strict no-execution constraint; no `torch.load` of any checkpoint).
- **Supporting evidence for PASS:**
  1. **Frozen save schema** (`v7:626-630`) deterministically writes `model_with_lora.state_dict()` + `step` + `condition`.
  2. **Empirical precedent:** the completed run (PID 27560) successfully restored **B0** checkpoints and produced `cos(H1)` values — proving the load/restore path works for real `model.pt` files of this exact format.
  3. **Architectural uniformity:** all 7 conditions trained by identical code/config; this audit's size stat confirms uniform adapter architecture, so B0's proven loadability generalizes structurally to S2/S4/S8/C4-Rand/R1-Rand.
- **Recommended first step of any authorized re-run:** explicitly `torch.load` all 56 `model.pt` and assert `state_dict` keys match the `LoraConfig`-wrapped model. This converts P6 from UNKNOWN to PASS before metric computation. It is a verification step, not the re-evaluation itself.

## 7. P7 — Condition → Readout Mapping — PASS

- `build_readouts` (`frozen runner:257-292`) defines exactly 7 readouts keyed by condition name, each a `(w, b)` `Readout` (`frozen runner:241-254`; `forward_logits(h)=(w·h+b, -(w·h+b))`). No runtime-only state beyond `(w,b)`.
- Saved weight files present (glob): `step0/w_{B0,F1,S2,S4,S8,C4-Rand,R1-Rand}.npy`.
- `b` reconstructable: B0 → `0.0` (`:287`); others → `-median(w·H16[:8])` (`:289`), where `H16[:8]` = first 8 rows of saved, hash-bound `step0/H_16.npy`. (Also redundantly stored as `readout_b` inside each `model.pt`.)
- Mapping is **1:1 and code-defined** — unambiguous:
  - B0 → `w_B0` (unembedding, b=0)
  - F1 → `w_F1` (s_scale·u_lda)
  - S2 → `w_S2` ((s_scale/k2)·proj(U2,u))
  - S4 → `w_S4` ((s_scale/k4)·proj(U4,u))
  - S8 → `w_S8` ((s_scale/k8)·proj(U8,u))
  - C4-Rand → `w_C4-Rand` ((s_scale/kR)·proj(R4,u))
  - R1-Rand → `w_R1-Rand` (s_scale·n_rand)

## 8. P8 — Separate Read-only Evaluator Without Modifying Frozen Record — PASS

- All required inputs (checkpoints, `w_{name}.npy`, `H_16.npy`, `u_lda.npy`, corpus, F7 machinery, LoRA/config constants) are **external to `runs.json` / `sha256_manifest.json`**. A new evaluator reads these frozen artifacts and writes results to a **new** output location (e.g., a new report/directory), leaving `runs.json` and all provenance untouched.
- The F7 gradient object (`per_sample_decision_gradient`, `COMPUTE_RAW_GRADIENT_COSINE`) is imported verbatim from the hash-verified frozen 6E.18 snapshot (`6BCAF131…`; `6E18:400-431,1006-1033`). The CHANGE 2 cpu/float patch (v7:402-407) is a runtime-only monkeypatch (no file modification) that preserves gradient semantics.
- **Conclusion:** a separate evaluator can compute H1 (per-condition `cos(H1)`) and P-metrics without altering any frozen execution record.

## 9. P9 — Sufficiency to Test H1/P2 **Per Condition** (Not B0/S8-Only) — PASS

- The completed runner's limitation (causal report F1) was a **code-path choice**: `run_evaluation` hard-codes `B0` checkpoints + the `S8` readout (`v7:866,805`), so H1 ("Both S4 and S8 achieve `mean cos(H1) ≥ +0.80`", protocol v3.1:214) was **never evaluated**. This is a data/code-path gap, **not** a checkpoint-data gap.
- The frozen artifacts **do** contain everything needed per condition:
  - Each condition's checkpoint holds that condition's **trained LoRA adapter** (trained with that condition's own readout as the decision objective — `v7:547-550,661`).
  - Each condition's **own readout** is reconstructable (P7).
  - The 16-panel + F7 machinery are shared and deterministic (P1, P8).
- Therefore re-evaluating **S4 with the S4 readout** and **S8 with the S8 readout**, aggregated over `AGGREGATION_SET = {4,6,14,15}` (`frozen runner:69`), directly yields the H1/P2 quantities. C4-Rand/R1-Rand support H4/H5 controls; F1 is the fixed-LDA reference; B0 the unembedding baseline.
- **Conclusion:** the 56 checkpoints are **sufficient to correctly test H1/P2 per condition**, provided the evaluator loops over `CONDITIONS` and uses each condition's own readout — i.e., does **not** repeat the B0/S8 path.

---

## 10. Final Recommendation

**FEASIBILITY: PASS with one UNKNOWN-to-be-resolved (P6 direct load).** All prerequisites required to re-evaluate every condition with its own checkpoint and its own protocol-defined readout are satisfied by the frozen artifacts, with the single exception that a direct `torch.load` of all 56 checkpoints was not performed in this audit (per the no-execution constraint). That direct-load check is strongly supported to PASS by the frozen save schema, the empirically proven B0 restore, and the verified architectural uniformity, and should be the **first verification step** of any authorized re-run.

**The existing 56 frozen checkpoints are sufficient to perform a new, non-mutating evaluation that correctly tests H1/P2 per condition.** No retraining is required. The prior "negative" result was driven by measuring the B0 baseline through the S8 readout (causal report F1); the checkpoint data needed to actually test the hypothesis is present and reconstructable.

**Recommended (authorization required before any execution):**
1. As step one of an authorized re-run, `torch.load` all 56 `model.pt` and assert key/shape match (resolves P6).
2. Write a corrected evaluator that loops `CONDITIONS` and computes `cos(H1)` with each condition's **own** readout (fixes the B0/S8-path gap).
3. Compute the **live** `rho_c` (protocol §5.7) and per-condition margins/BA/Cohen's-d, and **persist all** in provenance (original run discarded these — causal report F6).
4. Aggregate S4/S8 over `{4,6,14,15}` to test H1/P2; compare controls per H4/H5.
5. Write all outputs to a **new** location; do not overwrite the frozen `runs.json`/provenance.

*End of strict read-only feasibility audit.*
