"""
Phase B/C diagnostic for Runtime Drift Root-Cause Localization.

Localizes the first divergence boundary and isolates the responsible factor.
For B0 step t=4 (decisive step), with a single fixed checkpoint:

  Boundary 1 - RAW MODEL OUTPUT: logits for a fixed input (train mode, call1 vs call2)
  Boundary 2 - PER-SAMPLE GRADIENTS: psdg_fn outputs g_pos/g_abs (call1 vs call2)
  Boundary 3 - AGGREGATE cos(H1): crgc_fn output (call1 vs call2)

Then a controlled-factor isolation (Factor #3: inference mode):
  Repeat the evaluation under model.eval() (LoRA dropout disabled). If eval-mode
  evaluations become identical, the responsible factor is train-mode dropout
  (compute_live_cos_H1 never sets eval mode).

Diagnostic instrumentation only; all scientific functions (psdg_fn, crgc_fn,
compute_live_cos_H1) are invoked unchanged.
"""
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import torch

import run_phase_6e20_reevaluation as R
from run_phase_6e20_reevaluation import (
    resolve_project_paths,
    run_stage2_model_and_tokenizer_preflight,
    load_frozen_6e18_gradient_machinery,
    Readout,
    reconstruct_panel_prompts,
    HISTORICAL_B0_S8_ANCHOR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS, LORA_BIAS, LORA_TASK_TYPE,
    PRO_ID, AB_ID,
    CONDITIONS,
)
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).resolve().parent
TARGET_T = 4  # decisive step


def build_item(tokenizer, prompt_str, dec_token_id, device_str):
    enc = tokenizer(prompt_str, return_tensors="pt")
    p_ids = enc["input_ids"][0].tolist()
    align = [dec_token_id] * 4
    t_tokens = align + [dec_token_id]
    input_ids = p_ids + t_tokens
    labels = [-100] * len(p_ids) + t_tokens
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device_str),
        "labels": torch.tensor(labels, dtype=torch.long, device=device_str),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device_str),
    }


def capture_boundary(model, readout, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str):
    """Replicates compute_live_cos_H1 internals but returns raw logits, per-sample
    gradients, and the aggregate cos so each boundary can be fingerprinted."""
    lora_params = [p for p in model.parameters() if p.requires_grad]

    # Boundary 1: raw logits for a fixed input (panel_prompts[0], PRO side)
    item0 = build_item(tokenizer, panel_prompts[0][1], PRO_ID, device_str)
    with torch.no_grad():
        out0 = model(input_ids=item0["input_ids"].unsqueeze(0),
                     attention_mask=item0["attention_mask"].unsqueeze(0))
        raw_logits = out0.logits.detach().clone()

    g_pos, g_abs = [], []
    for i in range(8):
        item = build_item(tokenizer, panel_prompts[i][1], PRO_ID, device_str)
        g_pos.append(psdg_fn(model, readout, item, lora_params, tokenizer))
    for j in range(8):
        item = build_item(tokenizer, panel_prompts[8 + j][1], AB_ID, device_str)
        g_abs.append(psdg_fn(model, readout, item, lora_params, tokenizer))

    cos = crgc_fn(g_pos, g_abs)

    def fp(tensors):
        blobs = []
        for t in tensors:
            blobs.append(t.detach().cpu().numpy().tobytes())
        h = hashlib.sha256(b"".join(blobs)).hexdigest()
        norms = [float(t.detach().norm().item()) for t in tensors]
        return {"hash": h[:32], "norms": norms, "count": len(tensors)}

    return {
        "raw_logits_hash": hashlib.sha256(raw_logits.cpu().numpy().tobytes()).hexdigest()[:32],
        "raw_logits_sample": raw_logits.flatten()[:4].tolist(),
        "g_pos": fp(g_pos),
        "g_abs": fp(g_abs),
        "cos_H1": float(cos["cos_H1_aggregate"]),
    }


def load_ckpt(model, paths, t):
    ckpt_path = paths["historical_out_dir"] / "B0" / f"checkpoint_{t}" / "model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.zero_grad(set_to_none=True)


def main():
    out_dir = HERE / "phase-6e20.3-reevaluation-output" / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_project_paths()
    base_model, tokenizer = run_stage2_model_and_tokenizer_preflight()
    psdg_fn, crgc_fn, device_str = load_frozen_6e18_gradient_machinery(paths["frozen_6e18_snapshot"])
    if torch.cuda.is_available():
        base_model = base_model.to(device_str)

    H16_step0 = np.load(paths["historical_out_dir"] / "step0" / "H_16.npy")
    readouts = {}
    for c in CONDITIONS:
        w_vec = np.load(paths["historical_out_dir"] / "step0" / f"w_{c}.npy")
        bias = 0.0 if c == "B0" else float(-np.median(w_vec @ H16_step0[:8].T))
        readouts[c] = Readout(name=c, w=w_vec.astype(np.float64), b=bias)
    panel_prompts = reconstruct_panel_prompts(paths["corpus"])
    if len(panel_prompts) != 16:
        raise RuntimeError(f"Panel prompt count {len(panel_prompts)} != 16")

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias=LORA_BIAS, task_type=LORA_TASK_TYPE,
    )
    eval_peft_model = get_peft_model(base_model, lora_config)
    readout_s8 = readouts["S8"]

    # ---- Train mode (as the actual runner does; compute_live_cos_H1 sets no eval) ----
    load_ckpt(eval_peft_model, paths, TARGET_T)
    train_call1 = capture_boundary(eval_peft_model, readout_s8, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str)
    load_ckpt(eval_peft_model, paths, TARGET_T)
    train_call2 = capture_boundary(eval_peft_model, readout_s8, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str)

    # ---- Eval mode (Factor #3: inference mode isolation) ----
    eval_peft_model.eval()
    load_ckpt(eval_peft_model, paths, TARGET_T)
    eval_call1 = capture_boundary(eval_peft_model, readout_s8, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str)
    load_ckpt(eval_peft_model, paths, TARGET_T)
    eval_call2 = capture_boundary(eval_peft_model, readout_s8, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str)

    def cmp(a, b):
        return {
            "raw_logits_same": a["raw_logits_hash"] == b["raw_logits_hash"],
            "g_pos_same": a["g_pos"]["hash"] == b["g_pos"]["hash"],
            "g_abs_same": a["g_abs"]["hash"] == b["g_abs"]["hash"],
            "cos_diff": abs(a["cos_H1"] - b["cos_H1"]),
        }

    out = {
        "target_step": TARGET_T,
        "historical_anchor": HISTORICAL_B0_S8_ANCHOR[TARGET_T],
        "tolerance": 1e-4,
        "train_mode_call1_cos": train_call1["cos_H1"],
        "train_mode_call2_cos": train_call2["cos_H1"],
        "train_mode_within_same": cmp(train_call1, train_call2),
        "eval_mode_call1_cos": eval_call1["cos_H1"],
        "eval_mode_call2_cos": eval_call2["cos_H1"],
        "eval_mode_within_same": cmp(eval_call1, eval_call2),
        "train_vs_eval_call1_cos_diff": abs(train_call1["cos_H1"] - eval_call1["cos_H1"]),
        "details": {
            "train_call1": train_call1,
            "train_call2": train_call2,
            "eval_call1": eval_call1,
            "eval_call2": eval_call2,
        },
    }
    out_path = out_dir / "phaseBC_isolation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[B/C] train_mode cos: {train_call1['cos_H1']:.6f} vs {train_call2['cos_H1']:.6f} "
          f"(logits_same={out['train_mode_within_same']['raw_logits_same']}, "
          f"g_pos_same={out['train_mode_within_same']['g_pos_same']}, "
          f"g_abs_same={out['train_mode_within_same']['g_abs_same']})")
    print(f"[B/C] eval_mode  cos: {eval_call1['cos_H1']:.6f} vs {eval_call2['cos_H1']:.6f} "
          f"(logits_same={out['eval_mode_within_same']['raw_logits_same']}, "
          f"g_pos_same={out['eval_mode_within_same']['g_pos_same']}, "
          f"g_abs_same={out['eval_mode_within_same']['g_abs_same']})")
    print(f"[B/C] train_vs_eval cos_diff = {out['train_vs_eval_call1_cos_diff']:.6f}")
    print(f"[B/C] -> {out_path.name}")


if __name__ == "__main__":
    main()
