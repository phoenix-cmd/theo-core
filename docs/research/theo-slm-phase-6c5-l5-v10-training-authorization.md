# 6C.5 Cloud Runtime — L5 v13 Training Authorization (PATH VALIDATION GATE FIRST)

**Document:** `theo-slm-phase-6c5-l5-v10-training-authorization.md`  
**Date:** 2026-08-28  
**Supersedes:** L5 v12.  
**Status:** **AUTHORIZED FOR LIVE PATH VALIDATION GATE FIRST** (`validate_cloud_paths_6c5_kaggle.py --execute`). Full training submission (`launch_6c5_kaggle.py --execute`) is conditionally authorized upon live `[PATH_VALIDATION_PASS]`.

---

## 1. Execution Order & Gates

1. **Gate 1: Live Kaggle Path & Hash Validation:**
   ```bash
   py -3.14 validate_cloud_paths_6c5_kaggle.py --execute
   ```
   Must locate `/kaggle/input/datasets/anakinistic/theo-6c5-inputs`, output recursive tree diagnostics, resolve bounded bundle root, verify `bundle_manifest.json` identity, verify required files, verify expected SHA-256 bindings, verify `cloud_6c5.py` SHA (`c03f5d0e...`), execute `cloud_6c5` import smoke test, and emit `[PATH_VALIDATION_PASS]`.
2. **Gate 2: B×1 / C×1 Training Submission:**
   Only after Gate 1 passes, submit training kernel:
   ```bash
   py -3.14 launch_6c5_kaggle.py --execute
   ```
3. **Gate 3: Retrieval & Local Verification:**
   ```bash
   py -3.14 retrieve_6c5_kaggle_results.py --execute
   py -3.14 verify_6c5_cloud_results.py
   ```

---

## 2. Frozen Scientific Bindings & Runtime Baseline

- Adjudication SHA: `97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4`
- Fixture SHA: `f99ca025eaaa3073e296e7cb734581f0871315553b5c145bb7f2fc27a4ac6457`
- Split SHA: `eda4cb6695b6651a7cef8bdee0a7a990f4f799a6bf9148269ee758d0704ee232`
- Prompt Hash: `a29e12a2839b106d3a3632e1f353961d74182245a41602b445d06f71abb097ba`
- M1 Revision: `7ae55760`
- Seed: `20260826`
- AUX_LOSS_WEIGHT: `1.0`
- Pinned Stack: Python `3.14.3`, PyTorch `2.13.0+cu126`, CUDA `12.6`, Transformers `5.8.0`, PEFT `0.20.0`, bitsandbytes `0.50.1` (Linux package SHA `fdddb68e...`)
- Bundle SHA: `454cfc42f70b1457cf8837f8bd55c392e443f572640d6d26114d736e14b6ed31`

---

## 3. Bound Source Hashes (Remediation #15)

| File | SHA-256 |
|---|---|
| `kaggle_6c5_common.py` | `c726ad1fcc9c68cda2887977fdaeee0076b9fbb131f821cb032ba43976ff6ec4` |
| `launch_6c5_kaggle.py` | `2c7fed53b859a591b1289d6c290e1231744d0bc0a7ae4ce30e01e923d6466fd0` |
| `validate_cloud_paths_6c5_kaggle.py` | `3ad7f6ce5034c744dc47fc28dc579b7b4afbfb9821ffe471ea664f53493d7b38` |
| `cloud_6c5.py` | `c03f5d0eef95187225c5126621ceea414e719b1f1a58cc4f50d1751792bb4071` |
| `prepare_6c5_cloud_bundle.py` | `d8e4375488b042c0eb527545d898b3b3bfaa1543016c549807e38054b96b6a1e` |
| `test_6c5_orchestrator.py` | `2bf66e0b44843753b70fb4683a6556fb2c8f2a9ebc6fd6235a8857584e3f064c` |
| `amendment_audit.py` | `ab750ddde5d4520448552f66a3e5d6ff677577cd70a80bb5d75c4dad6db81ee2` |
| `experiment_spec.py` | `dfe8ed5df1ca16fe7da883c77573e69e810dce67e111025e03d83da05409b8b3` |
| `train_6c5.py` | `b1b5e9ca4b4917316d7d347addec67f8a7643c251437b54eae240c2b65a6c05d` |
