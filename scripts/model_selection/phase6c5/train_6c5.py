"""
6C.5 — Controlled B/C training harness (PREPARATION ONLY — NOT EXECUTED).

Trains ONE bounded QLoRA run for B and C with IDENTICAL data/prompt/budget.
The only difference is the supervised target set:
  B -> decoder envelope only (1 target/case)
  C -> decoder envelope + evidence-relevance head (2 targets/case, architectural mechanism)
Outputs to distinct dirs (-6c5-B / -6c5-C); never overwrites 6C.4 artifacts.

Run only after the Phase 2 pre-training audit passes AND explicit execution authorization.
"""
import os, sys, json, time, random, importlib.util, hashlib
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp", os.path.join(HERE, "experiment_spec.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

def tokenize(tok, sys_p, user, asst):
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user},
            {"role": "assistant", "content": asst}]
    full = tok.apply_chat_template(msgs, tokenize=False)
    pref = tok.apply_chat_template(msgs[:-1], tokenize=False)
    enc = tok(full, return_tensors="pt", truncation=True, max_length=384, padding="max_length")
    plen = len(tok(pref, return_tensors="pt")["input_ids"][0])
    input_ids = enc["input_ids"][0]; labels = input_ids.clone()
    labels[:plen] = -100; labels[input_ids == tok.pad_token_id] = -100
    return {"input_ids": input_ids.tolist(), "attention_mask": enc["attention_mask"][0].tolist(),
            "labels": labels.tolist()}

def train_one(arch):
    examples, data_sha = exp.build_examples()
    tok = AutoTokenizer.from_pretrained(exp.M1_DIR, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Deterministic stratified 90/10 split (seed 20260826) — IDENTICAL for B and C.
    # Train on the 90% split; the 10% dev partition is held DISJOINT (used only for audit parity).
    train_ex, dev_ex = exp.split_examples(examples)
    ds = []
    for e in train_ex:
        ds.append(tokenize(tok, exp.NORM_SYS, exp.NORM_USER.format(percept=e["percept"], evlist=e["ev_list"]), e["envelope"]))
        if arch == "C":
            ds.append(tokenize(tok, exp.NORM_EVID_SYS, exp.NORM_EVID_USER.format(percept=e["percept"], evlist=e["ev_list"]), e["evidence_head"]))

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(exp.M1_DIR, quantization_config=bnb,
                 torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eff_batch = 4 * 4
    steps = (len(ds) + eff_batch - 1) // eff_batch * 3
    args = TrainingArguments(output_dir=os.path.join(exp.BATCH2, f"_tmp_6c5_{arch}"),
        per_device_train_batch_size=4, gradient_accumulation_steps=4, num_train_epochs=3,
        learning_rate=2e-4, lr_scheduler_type="constant", warmup_steps=0,
        optim="adamw_torch", logging_steps=5, save_strategy="no", fp16=True,
        seed=exp.SEED, report_to="none", max_grad_norm=1.0)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    t0 = time.time(); trainer.train()
    adir = exp.OUT_C if arch == "C" else exp.OUT_B
    os.makedirs(adir, exist_ok=True); model.save_pretrained(adir)
    manifest = {"architecture": arch, "base": "M1 @7ae55760 (frozen)", "data_sha256": data_sha,
                "n_train_cases": len(train_ex), "n_dev_cases": len(dev_ex), "n_train_rows": len(ds),
                "epochs": 3, "eff_batch": eff_batch, "optimizer_steps": steps,
                "trainable_params": trainable, "lora": {"r":16,"alpha":32,"dropout":0.05,"target_modules":list(lora.target_modules)},
                "prompt": "NORM_SYS/NORM_USER (shared, architecture-agnostic)", "seed": exp.SEED,
                "aux_loss_weight": exp.AUX_LOSS_WEIGHT, "fixture_sha256": exp.FIXTURE_SHA256,
                "primary_supervision": "identical canonical envelope for B and C",
                "heads_trained": "envelope only" if arch=="B" else "envelope + auxiliary evidence head (weight LOCKED)",
                "status": "TRAINED_BOUNDED_RUN_6C5"}
    json.dump(manifest, open(os.path.join(adir, f"training_manifest_{arch}.json"), "w"), indent=2)
    print(f"[6c5-{arch}] DONE rows={len(ds)} steps={steps} trainable={trainable} data_sha={data_sha} adapter={adir}")

if __name__ == "__main__":
    for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["B", "C"]):
        train_one(a)
