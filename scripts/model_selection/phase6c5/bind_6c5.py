"""
6C.5 — Pre-execution binding manifest (READ-ONLY; captures frozen spec before training).

Writes:
  - 6c5_split_manifest.json  (train/dev case_id lists) + its SHA256
  - 6c5_binding_manifest.json (all pre-registered parameters)
"""
import os, json, hashlib, importlib.util, glob

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp", os.path.join(HERE, "experiment_spec.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

def sha256_file(path, cap=2**30):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()

def sha256_str(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()

examples, data_sha = exp.build_examples()
train_ex, dev_ex = exp.split_examples(examples)
train_ids = [e["case_id"] for e in train_ex]
dev_ids   = [e["case_id"] for e in dev_ex]

split_doc = {"n_train": len(train_ids), "n_dev": len(dev_ids),
             "train": train_ids, "dev": dev_ids, "seed": exp.SEED}
split_path = os.path.join(exp.BATCH2, "6c5_split_manifest.json")
json.dump(split_doc, open(split_path, "w"), indent=2)
split_sha = sha256_file(split_path)

prompt_blob = "||".join([exp.NORM_SYS, exp.NORM_USER, exp.NORM_EVID_SYS, exp.NORM_EVID_USER])
prompt_hash = sha256_str(prompt_blob)

# base model binding: hash config + weight file(s)
weight_files = sorted(glob.glob(os.path.join(exp.M1_DIR, "*.safetensors")) +
                      glob.glob(os.path.join(exp.M1_DIR, "*.bin")))
base_hashes = {}
for wf in weight_files:
    base_hashes[os.path.basename(wf)] = sha256_file(wf)
cfg = os.path.join(exp.M1_DIR, "config.json")
base_hashes["config.json"] = sha256_file(cfg) if os.path.exists(cfg) else "MISSING"

fixture_computed = exp.fixture_sha256()

binding = {
    "experiment": "6C.5 controlled B/C comparison",
    "status": "PRE-EXECUTION BINDING (frozen before any training)",
    "dataset_sha256": data_sha,
    "dataset_source": "6c4_mapping_manifest.json -> 300 training-included GOLD (44 REJECT excluded, unmodified)",
    "fixture_sha256_computed": fixture_computed,
    "fixture_sha256_expected": exp.FIXTURE_SHA256,
    "fixture_match": fixture_computed == exp.FIXTURE_SHA256,
    "adjudication_sha256_expected": "97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4",
    "adjudication_sha256_computed": exp.adjudication_sha256(),
    "adjudication_match": exp.adjudication_sha256() == "97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4",
    "prompt_hash": prompt_hash,
    "prompt_constants": {"NORM_SYS": exp.NORM_SYS, "NORM_USER": exp.NORM_USER,
                          "NORM_EVID_SYS": exp.NORM_EVID_SYS, "NORM_EVID_USER": exp.NORM_EVID_USER},
    "split_manifest_path": split_path,
    "split_manifest_sha256": split_sha,
    "split": {"seed": exp.SEED, "n_train": len(train_ids), "n_dev": len(dev_ids),
              "stratified_by": "final_status (tgt)"},
    "seed": exp.SEED,
    "base_model": {
        "path": exp.M1_DIR,
        "documented_revision": "@7ae55760 (frozen binding)",
        "file_hashes": base_hashes,
    },
    "method": "QLoRA (nf4 4-bit + double quant, compute_dtype fp16)",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
             "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
             "task_type": "CAUSAL_LM", "bias": "none"},
    "aux_loss_weight": exp.AUX_LOSS_WEIGHT,
    "aux_loss_locked": True,
    "training_params": {
        "epochs": 3, "lr": 2e-4, "lr_scheduler": "constant", "warmup_steps": 0,
        "per_device_train_batch_size": 4, "gradient_accumulation_steps": 4, "effective_batch": 16,
        "max_length": 384, "optim": "adamw_torch", "fp16": True, "max_grad_norm": 1.0,
        "primary_supervision": "canonical SemanticInterpretation envelope (identical B/C)",
        "c_auxiliary_head": "envelope + evidence-relevance head at aux_loss_weight (architectural mechanism)",
    },
    "firewall": {"evidence_token": exp.EVIDENCE_TOKEN,
                 "primary_turn_must_not_contain_token": True},
}
out = os.path.join(exp.BATCH2, "6c5_binding_manifest.json")
json.dump(binding, open(out, "w"), indent=2)
print("WROTE", out)
print(f"  dataset_sha={data_sha}")
print(f"  fixture match={binding['fixture_match']} ({fixture_computed})")
print(f"  prompt_hash={prompt_hash}")
print(f"  split_sha={split_sha}  train/dev={len(train_ids)}/{len(dev_ids)}")
print(f"  base weights hashed: {list(base_hashes.keys())}")
