# THEO SLM v0 — Phase 6C.5 Cloud Execution Environment Amendment (Kaggle GPU)

**Status: PREPARATION ONLY. NO CLOUD TRAINING EXECUTED. NO LOCAL TRAINING RESUMED.**

**Revision (conditional-pass correction applied):** the dependency baseline is now platform‑specific
— the local Windows `bitsandbytes==0.50.1` wheel/hash is historical evidence only; Kaggle/Colab binds
its **own Linux** `bitsandbytes==0.50.1` wheel SHA‑256 at execution time and must never reuse the
Windows hash. The independent `amendment_audit.py` re‑ran and **ALL GATES PASS** (local = historical
Windows baseline; cloud must reproduce the pinned stack or HALT).

This amendment moves the two bounded 6C.5 training runs (Architecture B, Architecture C)
to a cloud GPU environment (Kaggle) while **preserving every scientific variable** of the
frozen Phase‑2 specification. It exists because the user paused the local execution
authorization and requested a cloud amendment that is independently reviewed before any
training begins.

> **Authorization gate:** Cloud training MUST NOT begin until (a) this amendment is reviewed
> and (b) a **fresh execution authorization** explicitly binds the cloud runtime assumptions
> recorded here. Until then, `cloud_6c5.py` is a prepared artifact only.

---

## 1. Scientific variables preserved (byte‑identical to the frozen spec)

Every variable below is reproduced from `6c5_binding_manifest.json` (written pre‑execution,
hash‑chained into the cloud manifest). The cloud run verifies each against the binding before
training; any mismatch → HALT.

| Variable | Value / binding | Enforcement |
|---|---|---|
| Base model M1 | `Qwen2.5-0.5B-Instruct` @ `7ae55760` (frozen) | file hashes verified at runtime |
| Dataset | 300‑GOLD mapping → `6c4_mapping_manifest.json` | `dataset_sha256 = 6ec947711773e6488dd5221620b62a0c2dfe1ea46e8208b192ec4777275b3c7b` |
| 300‑GOLD immutability | `adjudication.json` `@97bc6bc6…` | unchanged; not re‑loaded for training |
| Fixture | 66‑case `phase‑6e3/populated‑evidence‑corpus.json` | `fixture_sha256 = f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457` |
| Train/dev split | deterministic stratified 90/10, seed 20260826 | `split_manifest_sha256 = eda4cb6695b6651a7cef8bdee0a7a990f4f799a6bf9148269ee758d0704ee232` (269 train / 31 dev) |
| Seed | 20260826 | asserted |
| System/user prompt | `NORM_SYS` / `NORM_USER` (single shared constant) | `prompt_hash = a29e12a2839b106d3a3632e1f353961d74182245a41602b445d06f71abb097ba` |
| Canonical supervision | `SemanticInterpretation` envelope (identical B/C) | schema asserted |
| Auxiliary‑loss weight (C) | `AUX_LOSS_WEIGHT = 1.0` (locked) | asserted; prohibited from tuning |
| Hyperparameters | LoRA r16/α32/drop0.05, nf4‑fp16 QLoRA, 3 ep, lr2e‑4, constant sched, warmup0, bs4×ga4 (eff 16), len384, adamw_torch, fp16, max_grad_norm1.0 | asserted in `train_6c5.py` |
| Architecture code | `b_architecture.py` / `c_architecture.py` UNMODIFIED | not imported/touched by cloud run |
| Evaluation gates | E0, grounding, E5/E6 (N/A on fixture), Group E ≥80% (deciding), abstain recall, proposal precision, false‑proposal rate, shortcut proxy, generation latency (reported), decision‑path latency ≤0.5 s, VRAM, v0.5/ADR‑0028 — **pre‑registered, unaltered thresholds** | `theo-slm-phase-6c5-gates.md` |

Nothing in this amendment edits architecture definitions, the dataset, labels, gates, or thresholds.

---

## 2. Preferred cloud environment

- **Preferred:** Kaggle **T4 × 2** (if the accelerator option is available to the account).
- **Otherwise:** document the **exact GPU actually assigned** (e.g., T4 ×1, P100, A100) in the
  cloud manifest at runtime via `torch.cuda.get_device_properties`. The scientific variables do
  **not** depend on the GPU; only throughput differs.

The two runs (B, C) execute **sequentially** in the single cloud session to keep adapter
isolation identical to the local plan.

---

## 3. Kaggle inputs & hash verification

Upload as Kaggle datasets / inputs (read‑only) and point `THEO_ROOT` at the mirrored tree:

- `M1_Qwen2.5-0.5B-Instruct/` — exact frozen weights (verify `model.safetensors` + `config.json`
  hashes against `6c5_binding_manifest.json → base_model.file_hashes`).
- `theo_slm_v0_review/6b3_batch2/` containing `6c4_mapping_manifest.json`,
  `candidate_records_batch2_remediated.json`, and `6c5_binding_manifest.json`,
  `6c5_split_manifest.json`.
- `theo_slm_v0_deduplicated/candidate_records.json`.
- `theo_slm_v0_review/6b3_batch1/candidate_records_batch1.json`.
- `theo_slm_v0_artifacts/phase-6e3/populated-evidence-corpus.json`.

`cloud_6c5.py` **refuses to train** unless:
1. M1 weight hashes == binding `base_model.file_hashes`,
2. `build_examples()` `data_sha256` == `6ec94771…`,
3. `fixture_sha256()` == `f99ca025…`,
4. recomputed split manifest SHA == `eda4cb66…`.

---

## 4. Dependency pinning — platform‑specific baseline (CORRECTED)

The adapters are evaluated **locally** (see §7), so the cloud stack must produce PEFT adapters
loadable by the local stack. Pin exactly:

```
python==3.14.3
torch==2.13.0+cu126        # CUDA runtime 12.6
transformers==5.8.0
peft==0.20.0
bitsandbytes==0.50.1
```

**Platform‑specific bitsandbytes rule (this is the corrected item):**
- The **local Windows `bitsandbytes==0.50.1` wheel/hash is HISTORICAL EVIDENCE ONLY.** It must
  never be reused as the cloud artifact hash.
- Kaggle / Colab must install the **official Linux x86‑64 `bitsandbytes==0.50.1` wheel**.
- `cloud_6c5.py` computes and binds the **actual Linux wheel/package SHA‑256 at execution time**
  (`bitsandbytes_wheel_sha256`) and records it in the cloud manifest.
- If the cloud cannot reproduce the required torch/CUDA/Python stack, the runner **HALTs** rather
  than silently substituting versions (`cloud_6c5.py` hard‑gates the pinned baseline).

`amendment_audit.py` re‑runs these gates read‑only; the local run captures the Windows stack as
the historical baseline and explicitly flags its bnb hash as not‑for‑cloud‑use.

---

## 5. Provenance captured by `cloud_6c5.py` (hard‑gated, platform‑specific)

Before training, `cloud_6c5.py` captures and **hard‑gates** every item below. Any deviation from
the frozen scientific variables or the pinned dependency baseline → HALT.

- **Scientific hashes:** `adjudication SHA 97bc6bc6…`, `fixture SHA f99ca025…`,
  `split SHA eda4cb66…`, `prompt hash a29e12a2…`, `model revision 7ae55760` (M1 weight hashes).
- **Dependency baseline (pinned, must match exactly):** Python `3.14.3`, PyTorch `2.13.0+cu126`,
  CUDA runtime `12.6`, Transformers `5.8.0`, PEFT `0.20.0`, bitsandbytes `0.50.1`.
- **Runtime provenance:** Python version, PyTorch version, CUDA runtime, NVIDIA driver,
  GPU model / count / **compute capability**, Transformers, PEFT,
  **bitsandbytes version + actual Linux wheel SHA‑256** (bound at runtime), OS / glibc,
  **dependency lock/hash** (`pip freeze` SHA).
- `binding_manifest_sha256` (chains this run to the exact frozen spec).
- Per‑arch: adapter directory hash (all files), `trainable_params`, `optimizer_steps`, wall‑clock train time.
- Re‑run result of the 16 Phase‑2 checks (must all PASS immediately before training).

The Windows local bitsandbytes hash is recorded only as historical evidence and is explicitly
excluded from the cloud artifact hash.

---

## 6. Execution procedure (after fresh authorization)

1. Attach inputs (§3) and set `THEO_ROOT`, `THEO_OUT_B`, `THEO_OUT_C` env vars.
2. Run the independent read‑only `amendment_audit.py` on the target host to confirm the
   scientific hashes and the pinned dependency baseline; any failure → HALT.
3. `python cloud_6c5.py`:
   - hard‑gates the dependency baseline (Python/torch/CUDA/transformers/peft/bitsandbytes) →
     HALT if the cloud cannot reproduce it (no silent substitution),
   - binds the **actual Linux bitsandbytes wheel SHA‑256** at runtime,
   - verifies all scientific hashes (§3/§5) → HALT on mismatch,
   - re‑runs the 16 Phase‑2 checks as final gate,
   - captures full provenance (§5),
   - trains B once, then C once (identical frozen spec),
   - writes adapter dirs + `6c5_cloud_execution_manifest.json`.
4. Retrieve adapter directories and their hashes; transfer to local `OUT_B` / `OUT_C`.

---

## 7. Latency is measured on LOCAL hardware — not the cloud

> **The cloud training speed must NOT redefine the ≤0.5 s latency criterion.**

- Generation latency and the **decision‑path latency ≤0.5 s** gate are computed by the
  **local** `eval_6c5.py` on the GTX 1650 target after the cloud adapters are retrieved.
- The cloud run reports only training throughput and (optionally) a cloud‑side decode‑time
  figure **for information only**; it carries no authority over the frozen gate.
- All other gates (E0, grounding, Group E, abstain recall, proposal precision,
  false‑proposal rate, shortcut proxy, VRAM on local) are also evaluated locally so the
  comparison stays apples‑to‑apples with 6C.4.

---

## 8. Artifact return & hash chain

```
cloud                         -> local
6c5_cloud_execution_manifest.json  (provenance + adapter hashes)
M1_...-6c5-B/  (adapter)           -> 6b3_batch2/M1_Qwen2.5-0.5B-Instruct-6c5-B/
M1_...-6c5-C/  (adapter)           -> 6b3_batch2/M1_Qwen2.5-0.5B-Instruct-6c5-C/
```

Local `eval_6c5.py` then runs unchanged; its `6c5_evaluation_results.json` plus this
manifest form the complete, reviewable record. No 6C.4 artifact is overwritten.

---

## 9. What is explicitly OUT of scope for the cloud run

- No hyperparameter sweep, retry, early reconfiguration, or architecture modification.
- No Batch 3 generation, dataset modification, or Architecture D.
- No gate/threshold alteration after results are observed.
- No local training until the cloud adapters are returned and locally evaluated.
