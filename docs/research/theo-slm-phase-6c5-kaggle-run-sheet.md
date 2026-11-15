# 6C.5 Kaggle/Colab run sheet (user-executed)

The agent has prepared and validated everything; execution happens on a Linux cloud GPU.

## 1. Upload inputs as a Kaggle dataset (read-only), preserving this layout
```
THEO_ROOT/
  theo-data/models/6c_acquired/M1_Qwen2.5-0.5B-Instruct/   (model.safetensors, config.json)
  theo-data/datasets/theo_slm_v0_review/adjudication.json
  theo-data/datasets/theo_slm_v0_review/6b3_batch2/
        6c4_mapping_manifest.json
        candidate_records_batch2_remediated.json
        6c5_binding_manifest.json
        6c5_split_manifest.json
  theo-data/datasets/theo_slm_v0_deduplicated/candidate_records.json
  theo-data/datasets/theo_slm_v0_review/6b3_batch1/candidate_records_batch1.json
  theo-data/datasets/theo_slm_v0_artifacts/phase-6e3/populated-evidence-corpus.json
  theo-core/scripts/model_selection/phase6c5/
        experiment_spec.py  train_6c5.py  cloud_6c5.py
        phase2_audit.py    amendment_audit.py  requirements.txt
```
The runner verifies every hash (M1, adjudication 97bc6bc6, fixture f99ca025, split eda4cb66,
prompt a29e12a2) against `6c5_binding_manifest.json` and HALTs on any mismatch.

## 2. Environment
```bash
pip install -r theo-core/scripts/model_selection/phase6c5/requirements.txt
# torch must resolve to 2.13.0+cu126; if a different build is installed, the runner HALTs.
```

## 3. Run (writable output dir for adapters)
```bash
export THEO_ROOT=/kaggle/input/<your-dataset>/THEO
export THEO_OUT_B=/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-B
export THEO_OUT_C=/kaggle/working/M1_Qwen2.5-0.5B-Instruct-6c5-C
mkdir -p $THEO_OUT_B $THEO_OUT_C
python /kaggle/input/<your-dataset>/THEO/theo-core/scripts/model_selection/phase6c5/cloud_6c5.py
```
`cloud_6c5.py` will: hard-gate the dependency baseline, re-run the 16 Phase-2 checks, bind the
**Linux** bitsandbytes wheel SHA, train B once then C once, and write
`/kaggle/working/6c5_cloud_execution_manifest.json` plus the two adapter dirs (each hashed).

## 4. Return to local for latency eval
Download `M1_Qwen2.5-0.5B-Instruct-6c5-B/`, `M1_Qwen2.5-0.5B-Instruct-6c5-C/`, and
`6c5_cloud_execution_manifest.json`. Place adapters at the local
`theo-data/datasets/theo_slm_v0_review/6b3_batch2/...-6c5-{B,C}` paths, then the agent runs
`eval_6c5.py` on the GTX 1650 (latency gates measured locally; cloud speed does not redefine ≤0.5 s).
