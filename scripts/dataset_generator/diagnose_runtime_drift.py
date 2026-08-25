"""
Phase A diagnostic for Runtime Drift Root-Cause Localization (6E.20.3 post-closure).

NOT a repair. Pure forensic instrumentation. Replicates the exact B0 recomputation
pipeline of run_phase_6e20_reevaluation.py WITHOUT modifying any scientific logic,
then records repeated evaluations to establish within-environment repeatability.

Each invocation = ONE independent process run:
  * load model + tokenizer (identical to runner Stage 2)
  * load frozen 6E.18 machinery (psdg_fn, crgc_fn, device_str)
  * reconstruct readouts + panel prompts (identical to runner)
  * wrap in PEFT LoRA (identical config to runner)
  * capture RNG / seed state BEFORE evaluation
  * b0_run1: full 4-step B0 cos(H1) evaluation (each call cleanly zero-graded)
  * b0_run2: full 4-step B0 cos(H1) evaluation again (same process, reloaded ckpts)
  * capture RNG / seed state AFTER evaluation
  * record NEPR hash (environment provenance)
  * emit JSON artifact

Invoke multiple times (separate processes) to build the repeatability matrix.
"""
import sys
import json
import random
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
    compute_live_cos_H1,
    HISTORICAL_B0_S8_ANCHOR,
    B0_REPRODUCIBILITY_TOLERANCE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS, LORA_BIAS, LORA_TASK_TYPE,
    compute_sha256,
    PRO_ID, AB_ID,
    CONDITIONS,
    collect_nepr,
)
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).resolve().parent


def gather_seed_states():
    try:
        py = random.getstate()
        py_repr = {"state": [int(x) for x in py[1][:4]], "last": int(py[2]), "count": len(py[1])}
    except Exception as e:
        py_repr = {"error": str(e)}
    try:
        npy = np.random.get_state()
        npy_repr = {"alg": str(npy[0]), "pos": int(npy[2]), "has_gauss": int(npy[3]), "size": int(npy[1].shape[0])}
    except Exception as e:
        npy_repr = {"error": str(e)}
    try:
        cpu = torch.get_rng_state()
        cpu_repr = {"hash": hashlib.sha256(cpu.cpu().numpy().tobytes()).hexdigest()[:32], "size": int(cpu.shape[0])}
    except Exception as e:
        cpu_repr = {"error": str(e)}
    try:
        if torch.cuda.is_available():
            cuda = torch.cuda.get_rng_state_all()
            cuda_repr = [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16] for t in cuda]
        else:
            cuda_repr = "cuda-unavailable"
    except Exception as e:
        cuda_repr = {"error": str(e)}
    return {"python": py_repr, "numpy": npy_repr, "torch_cpu": cpu_repr, "torch_cuda": cuda_repr}


def b0_evaluate(eval_peft_model, tokenizer, paths, panel_prompts, readout_s8, psdg_fn, crgc_fn, device_str):
    """Replicate runner's Stage 5 B0 loop exactly (one clean pass over the 4 anchors)."""
    results = {}
    for t, exp_cos in HISTORICAL_B0_S8_ANCHOR.items():
        ckpt_path = paths["historical_out_dir"] / "B0" / f"checkpoint_{t}" / "model.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        eval_peft_model.load_state_dict(ckpt["state_dict"], strict=False)
        eval_peft_model.zero_grad(set_to_none=True)
        cos = compute_live_cos_H1(
            eval_peft_model, readout_s8, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str
        )
        results[str(t)] = cos
    return results


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

    nepr = collect_nepr(
        model=base_model,
        peft_model=eval_peft_model,
        phase_id="6E.20.3-DIAG",
        runner_sha256=compute_sha256(HERE / "run_phase_6e20_reevaluation.py"),
        test_sha256=compute_sha256(HERE / "test_phase_6e20_reevaluation.py"),
    )

    seed_before = gather_seed_states()
    b0_run1 = b0_evaluate(eval_peft_model, tokenizer, paths, panel_prompts, readout_s8, psdg_fn, crgc_fn, device_str)
    b0_run2 = b0_evaluate(eval_peft_model, tokenizer, paths, panel_prompts, readout_s8, psdg_fn, crgc_fn, device_str)
    seed_after = gather_seed_states()

    # Stage-5 decisive metrics (replicate runner halt logic)
    def verdict(run):
        diffs = {t: abs(run[t] - HISTORICAL_B0_S8_ANCHOR[int(t)]) for t in run}
        max_t = max(diffs, key=diffs.get)
        return {
            "per_step_cos": run,
            "per_step_diff_vs_anchor": {t: diffs[t] for t in run},
            "decisive_step": max_t,
            "decisive_recomputed": run[max_t],
            "decisive_anchor": HISTORICAL_B0_S8_ANCHOR[int(max_t)],
            "decisive_diff": diffs[max_t],
            "tolerance": B0_REPRODUCIBILITY_TOLERANCE,
            "halts": diffs[max_t] > B0_REPRODUCIBILITY_TOLERANCE,
        }

    out = {
        "run_id": run_id,
        "phase": "A-repeatability",
        "device_str": device_str,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "b0_run1": verdict(b0_run1),
        "b0_run2": verdict(b0_run2),
        "within_process_same": all(
            abs(b0_run1[t] - b0_run2[t]) <= 1e-9 for t in b0_run1
        ),
        "seed_before": seed_before,
        "seed_after": seed_after,
        "nepr_hash": nepr["category_F_metadata"]["nepr_hash"],
        "nepr_schema": nepr["nepr_schema_version"],
        "runner_sha256": compute_sha256(HERE / "run_phase_6e20_reevaluation.py"),
        "test_sha256": compute_sha256(HERE / "test_phase_6e20_reevaluation.py"),
        "collect_nepr_sha256": compute_sha256(HERE / "collect_nepr.py"),
    }
    out_path = out_dir / f"run_{run_id}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[A] run_{run_id} -> {out_path.name}")
    print(f"    b0_run1 step4={b0_run1['4']:.6f} step6={b0_run1['6']:.6f} step14={b0_run1['14']:.6f} step15={b0_run1['15']:.6f}")
    print(f"    b0_run2 step4={b0_run2['4']:.6f} step6={b0_run2['6']:.6f} step14={b0_run2['14']:.6f} step15={b0_run2['15']:.6f}")
    print(f"    within_process_same={out['within_process_same']}")


if __name__ == "__main__":
    main()
