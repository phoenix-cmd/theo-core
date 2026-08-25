"""
Phase C (decisive) — Controlled-factor isolation: inference mode.

Uses the unmodified compute_live_cos_H1 (which internally applies
ensure_readout_class_patch) to evaluate B0 cos(H1) under two regimes:

  REGIME 1 - TRAIN mode (faithful to the real runner; compute_live_cos_H1 never
            sets eval, so LoRA dropout=0.05 is ACTIVE):
              * load ckpt -> cos_a ; load ckpt -> cos_b  (within-process repeat)
  REGIME 2 - EVAL mode (model.eval() disables LoRA dropout):
              * load ckpt -> cos_c ; load ckpt -> cos_d  (within-process repeat)

Interpretation:
  * If TRAIN regime varies within-process (|cos_a - cos_b| > 1e-4) but EVAL regime
    is identical (|cos_c - cos_d| <= 1e-4), the responsible factor is train-mode
    dropout (compute_live_cos_H1 omits model.eval()).
  * If EVAL regime ALSO varies, the cause is GPU nondeterminism (not dropout).

Invoked multiple times (separate processes) for a cross-process view.
"""
import sys
import json
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
    compute_live_cos_H1,
    HISTORICAL_B0_S8_ANCHOR,
    B0_REPRODUCIBILITY_TOLERANCE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS, LORA_BIAS, LORA_TASK_TYPE,
    CONDITIONS,
)
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).resolve().parent


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else "run"
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

    def eval_once(mode):
        eval_peft_model.train() if mode == "train" else eval_peft_model.eval()
        res = {}
        for t, exp_cos in HISTORICAL_B0_S8_ANCHOR.items():
            ckpt_path = paths["historical_out_dir"] / "B0" / f"checkpoint_{t}" / "model.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            eval_peft_model.load_state_dict(ckpt["state_dict"], strict=False)
            eval_peft_model.zero_grad(set_to_none=True)
            ca = compute_live_cos_H1(eval_peft_model, readout_s8, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str)
            eval_peft_model.load_state_dict(ckpt["state_dict"], strict=False)
            eval_peft_model.zero_grad(set_to_none=True)
            cb = compute_live_cos_H1(eval_peft_model, readout_s8, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str)
            res[str(t)] = {"cos_a": ca, "cos_b": cb, "within_diff": abs(ca - cb),
                           "anchor": exp_cos, "diff_vs_anchor_a": abs(ca - exp_cos)}
        return res

    train_res = eval_once("train")
    eval_res = eval_once("eval")

    def summarize(res):
        step4 = res["4"]
        return {
            "step4_cos_a": step4["cos_a"], "step4_cos_b": step4["cos_b"],
            "step4_within_diff": step4["within_diff"],
            "step4_anchor": step4["anchor"],
            "step4_diff_vs_anchor_a": step4["diff_vs_anchor_a"],
            "all_steps": {t: {"cos_a": res[t]["cos_a"], "cos_b": res[t]["cos_b"],
                              "within_diff": res[t]["within_diff"]} for t in res},
            "max_within_diff": max(res[t]["within_diff"] for t in res),
        }

    out = {
        "run_id": run_id,
        "phase": "C-inference-mode-isolation",
        "device_str": device_str,
        "tolerance": B0_REPRODUCIBILITY_TOLERANCE,
        "lora_dropout": LORA_DROPOUT,
        "train_mode": summarize(train_res),
        "eval_mode": summarize(eval_res),
        "verdict_dropout_is_cause": (
            train_res["4"]["within_diff"] > B0_REPRODUCIBILITY_TOLERANCE
            and eval_res["4"]["within_diff"] <= B0_REPRODUCIBILITY_TOLERANCE
        ),
        "verdict_gpu_nondeterminism": (
            eval_res["4"]["within_diff"] > B0_REPRODUCIBILITY_TOLERANCE
        ),
    }
    out_path = out_dir / f"phaseC_evalmode_{run_id}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[C:{run_id}] TRAIN step4 a={train_res['4']['cos_a']:.6f} b={train_res['4']['cos_b']:.6f} d={train_res['4']['within_diff']:.2e}")
    print(f"[C:{run_id}] EVAL  step4 c={eval_res['4']['cos_a']:.6f} d={eval_res['4']['cos_b']:.6f} d={eval_res['4']['within_diff']:.2e}")
    print(f"[C:{run_id}] dropout_is_cause={out['verdict_dropout_is_cause']} gpu_nondeterminism={out['verdict_gpu_nondeterminism']}")


if __name__ == "__main__":
    main()
