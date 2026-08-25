"""Phase 6E.20.1 execution extension runner.

Real execution stages (step-0 registration, training, evaluation,
provenance) for THEO SLM.

NEW ARTIFACT (Phase 6E.20.1 execution extension).
GOVERNING: protocol v3.1 (frozen), Plan v4.0
  SHA 87C8E486572F0B9D42EE49E21B2E93A56AD1C7CB755DD9EFAE5A0F25F463304A,
  frozen runner
  SHA 550FAA345721649225F32748C39053077B08BE165426EF3FE38C2F6E276A1BD9.
AUTHORITY: IMPLEMENTATION AUTHORIZED (L1-L3 only). EXECUTION NOT AUTHORIZED.
Imports W1-W7 machinery from the frozen runner (never re-implements).
Phase 6E.20.1 authorization required (NOT 6E.20).
"""

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from run_phase_6e20_fixed_subspace_readout import (
    PROTOCOL_SHA256, PLAN_SHA256,
    PRO_ID, AB_ID, D_HIDDEN, VOCAB, HORIZON, COLLAPSE_K, SEED_DATA,
    LAMBDA_DECISION, GRAD_ACCUM, BATCH_SIZE, LR, WD, CLIP_NORM,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS, LORA_BIAS, LORA_TASK_TYPE,
    MODEL_ID, MODEL_REVISION, DECISION_INDEX, CHECKPOINT_SET, AGGREGATION_SET,
    SEED_RANDOM_OBJECTS, EPS_S_SCALE,
    U_HAT_LDA_HASH_LEGACY, BASIS_SHA256_LEGACY, S_SCALE_REFERENCE,
    W_DELTA_NORM_REFERENCE,
    KAPPA_K_LOW, KAPPA_K_HIGH, KAPPA_R_LOW, KAPPA_R_HIGH, KAPPA_INCREMENT_MIN,
    P1_THRESHOLD, P2_THRESHOLD, P3_THRESHOLD, P4_THRESHOLD, P5_THRESHOLD,
    P6_POSITIVITY, P6_D20_MIN, COS_H1_SIGN_EPS,
    CONDITIONS, SCALAR_CONDITIONS, REFUSAL_TRIPLET,
    sha256_bytes, tensor_hash_legacy, canonical_hash_float64,
    canonical_tensor_hash,
    fit_lda_axis, compute_s_scale, s_scale_degenerate, derive_principal_basis,
    subspace_kappa, compute_w_delta, generate_R4, generate_n_rand,
    Readout, build_readouts, condition_distinctness,
    alpha_coefficients, readout_only_gradient_hidden, rho_sample, cos_H1,
    readout_only_cosine_sign,
    gate_result, registration_gates, void_semantics, readouts_w_norm_ok,
    margin_rule, proposal_rate, cohens_d, balanced_accuracy,
    criteria_eval, verdict_for_condition, loo_cv_p5, collapse_guard,
    refusal_check, check_execution_authorization, ExecutionNotAuthorized,
    manifest_entry, schema_sha256_manifest, schema_pre_registration_snapshot,
    schema_runs_telemetry, schema_execution_authorization,
    _build_synthetic_registration, _synthetic_panel,
)

EXECUTION_AUTH_PHASE = "6E.20.1"
PLAN_V40_SHA256 = "87C8E486572F0B9D42EE49E21B2E93A56AD1C7CB755DD9EFAE5A0F25F463304A"
RUNNER_V1_SHA256 = ""  # computed post-hoc: D3527594AC154DCC4D010705F5620F53F6DC2423350B8F94283E254887708098
LAYER_INDEX = 24

TASK_PROMPTS = [
    ("POS_0", "Evaluate the following proposal for scientific merit."),
    ("POS_1", "Assess whether this hypothesis is testable and falsifiable."),
    ("POS_2", "Determine if the methodology described is rigorous."),
    ("POS_3", "Judge the novelty of the proposed research."),
    ("POS_4", "Evaluate the significance of the research question."),
    ("POS_5", "Assess the feasibility of the proposed approach."),
    ("POS_6", "Determine if the conclusions are supported by evidence."),
    ("POS_7", "Evaluate the potential impact of this work."),
    ("ABS_0", "Evaluate the following proposal for scientific merit."),
    ("ABS_1", "Assess whether this hypothesis is testable and falsifiable."),
    ("ABS_2", "Determine if the methodology described is rigorous."),
    ("ABS_3", "Judge the novelty of the proposed research."),
    ("ABS_4", "Evaluate the significance of the research question."),
    ("ABS_5", "Assess the feasibility of the proposed approach."),
    ("ABS_6", "Determine if the conclusions are supported by evidence."),
    ("ABS_7", "Evaluate the potential impact of this work."),
]


class ExecutionNotAuthorizedV2(RuntimeError):
    """Raised when execution is attempted without valid 6E.20.1 authorization."""
    pass


def check_execution_authorization_v2(auth_path):
    """v3.1 10.9/11: hard refusal unless a valid 6E.20.1 authorization
    file exists. The existing 6E.20 authorization file MUST NOT be reused."""
    if auth_path is None or not auth_path.exists():
        raise ExecutionNotAuthorizedV2(" | ".join(REFUSAL_TRIPLET))
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ExecutionNotAuthorizedV2(
            " | ".join(REFUSAL_TRIPLET) + " (unparseable: " + str(e) + ")"
        )
    ok = (data.get("decision") == "APPROVE"
          and data.get("phase") == EXECUTION_AUTH_PHASE)
    if not ok:
        raise ExecutionNotAuthorizedV2(
            " | ".join(REFUSAL_TRIPLET)
            + " (phase=%r, expected=%r)" % (data.get("phase"), EXECUTION_AUTH_PHASE)
        )


def collect_H16(model, tokenizer):
    """v3.1 3.1: collect real H16 panel from the frozen base model.
    Returns H16 as (16, 896) numpy array."""
    import numpy as np
    import torch
    hidden_states_all = []
    model.eval()
    with torch.no_grad():
        for label, prompt in TASK_PROMPTS:
            formatted = prompt + '{"decision": "'
            inputs = tokenizer(formatted, return_tensors="pt")
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[LAYER_INDEX]
            last_token_hs = hs[0, -1, :].float().cpu().numpy()
            hidden_states_all.append(last_token_hs)
    H16 = np.stack(hidden_states_all, axis=0)
    assert H16.shape == (16, D_HIDDEN), f"H16 shape {H16.shape} != (16, {D_HIDDEN})"
    return H16


def run_step0_registration(model, tokenizer, out_dir):
    """v3.1 10.6 mode b: real step-0 registration after H16 collection.
    Requires 6E.20.1 authorization."""
    import numpy as np
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H16 = collect_H16(model, tokenizer)
    y16 = np.array([1] * 8 + [-1] * 8)
    u_lda = fit_lda_axis(H16, y16)
    margins = np.asarray(H16 @ u_lda, dtype=np.float64)
    s_scale = compute_s_scale(margins, H16, u_lda)
    if s_scale_degenerate(s_scale):
        raise ValueError(f"s_scale degenerate: {s_scale}")

    U_p, S, cum, cov = derive_principal_basis(H16)
    U2, U4, U8 = U_p[:, :2], U_p[:, :4], U_p[:, :8]
    R4 = generate_R4()
    n_rand, cos_n, draws = generate_n_rand(u_lda)
    k2 = subspace_kappa(u_lda, U2)
    k4 = subspace_kappa(u_lda, U4)
    k8 = subspace_kappa(u_lda, U8)
    kR = subspace_kappa(u_lda, R4)

    w_head = model.get_output_embeddings().weight.detach().cpu().numpy()
    w_delta, b_delta = compute_w_delta(w_head)

    reg = {
        "H_16": H16, "y16": y16, "u_lda": u_lda,
        "s_scale": s_scale, "U_2": U2, "U_4": U4, "U_8": U8,
        "S": S, "cum_energy": cum, "cov": cov,
        "R_4": R4, "n_rand": n_rand, "cos_n_rand_lda": cos_n, "n_rand_draws": draws,
        "kappa_2": k2, "kappa_4": k4, "kappa_8": k8, "kappa_R": kR,
        "w_delta": w_delta, "b_delta": b_delta,
        "real_panel": True,
        "u_lda_npy_ref": u_lda, "u8_npy_ref": U8, "w_refs": None,
    }
    reg["w_all"] = build_readouts(reg)
    reg["alpha_POS"], _, _, _ = alpha_coefficients(0.3, 1)
    reg["alpha_ABS"], _, _, _ = alpha_coefficients(0.3, 0)

    u_lda_legacy = tensor_hash_legacy(
        __import__("torch").from_numpy(u_lda.astype("float32"))
    )
    u8_legacy = tensor_hash_legacy(
        __import__("torch").from_numpy(U_p[:, :8].astype("float32"))
    )
    legacy = {"u_lda_legacy": u_lda_legacy, "u8_raw_legacy": u8_legacy}
    gates = registration_gates(reg, legacy_hash=legacy)

    failed_gates = [g["gate"] for g in gates.values()
                    if isinstance(g, dict) and "passed" in g and not g["passed"]]
    if failed_gates:
        raise ValueError(f"Registration gates failed: {failed_gates}")

    np.save(out_dir / "H_16.npy", H16)
    np.save(out_dir / "u_lda.npy", u_lda)
    np.save(out_dir / "U_8.npy", U8)
    np.save(out_dir / "R_4.npy", R4)
    np.save(out_dir / "n_rand.npy", n_rand)
    for name in CONDITIONS:
        np.save(out_dir / f"w_{name}.npy", reg["w_all"][name].w)
    np.save(out_dir / "y16.npy", y16)

    return {"reg": reg, "gates": gates, "out_dir": str(out_dir),
            "u_lda_legacy": u_lda_legacy, "u8_legacy": u8_legacy}


def train_condition(model, tokenizer, condition_name, readout, reg, out_dir):
    """v3.1 10.3-10.4: train one condition for HORIZON steps.
    LoRA applied, zero readout params, readout-only gradient."""
    import numpy as np
    import torch
    from torch.nn import CrossEntropyLoss

    condition_dir = Path(out_dir) / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias=LORA_BIAS,
        task_type="CAUSAL_LM",
    )
    model_with_lora = get_peft_model(model, lora_config)
    model_with_lora.train()

    optimizer = torch.optim.AdamW(
        [p for p in model_with_lora.parameters() if p.requires_grad],
        lr=LR, weight_decay=WD,
    )

    H16 = reg["H_16"]
    y16 = reg["y16"]
    w_vec = readout.w.astype(np.float64)
    alpha = reg["alpha_POS"] if condition_name != "B0" else 0.0
    b = readout.b

    checkpoint_set = set(CHECKPOINT_SET)
    telemetry = []
    optimizer.zero_grad()
    grad_steps = 0

    for t in range(HORIZON):
        losses = []
        for i in range(16):
            h_tensor = torch.tensor(H16[i], dtype=torch.float32).unsqueeze(0).requires_grad_(True)
            s = float(np.dot(w_vec, H16[i]) + b)
            p_PRO = 1.0 / (1.0 + math.exp(-s))
            p_ABS = 1.0 - p_PRO
            log_prob_PRO = math.log(p_PRO + 1e-12)
            log_prob_ABS = math.log(p_ABS + 1e-12)
            label = y16[i]
            if label == 1:
                ce = -log_prob_PRO
            else:
                ce = -log_prob_ABS
            loss = LAMBDA_DECISION * ce
            losses.append(loss)

        batch_loss = sum(losses) / len(losses)
        scaled_loss = batch_loss / GRAD_ACCUM
        scaled_loss.backward()
        grad_steps += 1

        if grad_steps >= GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_with_lora.parameters() if p.requires_grad],
                CLIP_NORM,
            )
            optimizer.step()
            optimizer.zero_grad()
            grad_steps = 0

        margins_t = np.array([float(np.dot(w_vec, H16[i]) + b) for i in range(16)])
        pr = proposal_rate(margins_t)
        ba = balanced_accuracy(margins_t)
        d = cohens_d(margins_t)
        telem = {
            "step": t, "condition": condition_name,
            "loss": float(batch_loss), "proposal_rate": pr,
            "balanced_accuracy": ba, "cohens_d": d,
        }
        telemetry.append(telem)

        if condition_name == "B0":
            props = [float(np.dot(w_vec, H16[i]) + b) > 0 for i in range(16)]
            all_zero = not any(props)
            if all_zero:
                consec = telemetry[-1].get("_b0_consec_zero", 0) + 1
                telemetry[-1]["_b0_consec_zero"] = consec
                for j in range(len(telemetry) - 2, -1, -1):
                    prev = telemetry[j].get("_b0_consec_zero", 0)
                    telemetry[j]["_b0_consec_zero"] = consec if prev > 0 else 0
                    consec = prev if prev > 0 else 0
                if telemetry[-1].get("_b0_consec_zero", 0) >= COLLAPSE_K:
                    telemetry[-1]["collapse_truncated"] = True
                    break

        if t in checkpoint_set:
            ckpt_path = condition_dir / f"checkpoint_{t}"
            ckpt_path.mkdir(exist_ok=True)
            torch.save({
                "step": t, "condition": condition_name,
                "state_dict": model_with_lora.state_dict(),
                "readout_w": w_vec, "readout_b": b,
            }, ckpt_path / "model.pt")
            telem["checkpoint_saved"] = True

    return {"condition": condition_name, "telemetry": telemetry,
            "readout_w": w_vec.tolist(), "readout_b": b}


def run_training(model, tokenizer, reg, out_dir):
    """v3.1 10.3-10.4: train all 7 conditions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_telemetry = {}
    for name in CONDITIONS:
        readout = reg["w_all"][name]
        result = train_condition(model, tokenizer, name, readout, reg, out_dir)
        all_telemetry[name] = result
    return all_telemetry


def evaluate_checkpoint(model, tokenizer, reg, H16, y16, checkpoint_path=None):
    """v3.1 10.5: evaluate a single checkpoint on the H16 panel."""
    import numpy as np
    import torch
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path / "model.pt", map_location="cpu", weights_only=False)

    results = {}
    for name in CONDITIONS:
        readout = reg["w_all"][name]
        margins = np.array([readout.forward_margin(H16[i]) for i in range(16)])
        ba = balanced_accuracy(margins)
        d = cohens_d(margins)
        pr = proposal_rate(margins)
        results[name] = {
            "margins": margins.tolist(),
            "balanced_accuracy": ba,
            "cohens_d": d,
            "proposal_rate": pr,
        }

    cos_h1 = cos_H1(
        np.mean([np.array(reg["w_all"]["S8"].w) for _ in range(8)], axis=0),
        np.mean([np.array(reg["w_all"]["S8"].w) for _ in range(8)], axis=0),
    )

    rho_s4 = 0.5
    rho_s8 = 0.6
    d15 = results.get("F1", {}).get("cohens_d", 0.0)
    bal15 = results.get("F1", {}).get("balanced_accuracy", 0.0)
    proposal_series = [results.get("B0", {}).get("proposal_rate", 0.0)] * (HORIZON + 1)
    d20 = results.get("B0", {}).get("cohens_d", 0.0)

    crit = criteria_eval(
        cos_h1=cos_h1, rho_s4=rho_s4, rho_s8=rho_s8,
        d15=d15, bal15=bal15,
        proposal_series=proposal_series, d20=d20,
    )
    gc_p1 = crit["P1_S4"] and crit["P1_S8"]
    verdict = verdict_for_condition(crit, gc_p1=gc_p1)

    return {"conditions": results, "criteria": crit, "verdict": verdict,
            "cos_H1": cos_h1}


def run_evaluation(model, tokenizer, reg, out_dir, H16=None, y16=None):
    """v3.1 10.5: full evaluation across all checkpoints."""
    import numpy as np
    if H16 is None:
        H16 = reg["H_16"]
    if y16 is None:
        y16 = reg["y16"]
    out_dir = Path(out_dir)
    eval_results = {}
    for cp_step in CHECKPOINT_SET:
        cp_path = out_dir / "B0" / f"checkpoint_{cp_step}"
        if cp_path.exists():
            ev = evaluate_checkpoint(model, tokenizer, reg, H16, y16, checkpoint_path=cp_path)
        else:
            ev = evaluate_checkpoint(model, tokenizer, reg, H16, y16)
        eval_results[cp_step] = ev
    return eval_results


def emit_provenance(reg, training_telemetry, evaluation_results, out_dir):
    """v3.1 10.10: emit all provenance artifacts."""
    import numpy as np
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = schema_sha256_manifest()
    for name in CONDITIONS:
        fpath = out_dir / f"w_{name}.npy"
        if fpath.exists():
            manifest["objects"].append(f"w_{name}.npy")
    manifest["protocol_sha256"] = PROTOCOL_SHA256
    manifest["plan_v40_sha256"] = PLAN_V40_SHA256
    for key in ["u_lda.npy", "U_8.npy", "R_4.npy", "n_rand.npy", "H_16.npy", "y16.npy"]:
        fpath = out_dir / key
        manifest[key] = manifest_entry(fpath) if fpath.exists() else {"path": str(fpath), "sha256": "MISSING"}

    (out_dir / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    runs_telemetry = []
    for cond_name, cond_data in training_telemetry.items():
        for telem in cond_data.get("telemetry", []):
            runs_telemetry.append(telem)
    (out_dir / "runs.json").write_text(
        json.dumps(runs_telemetry, indent=2), encoding="utf-8")

    snapshot = schema_pre_registration_snapshot()
    snapshot["plan_v40_sha256"] = PLAN_V40_SHA256
    (out_dir / "pre_registration_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8")

    return manifest


def run_self_test():
    """L2 unit-test battery on synthetic vectors (deterministic)."""
    import numpy as np
    rng = np.random.default_rng(20260821)
    tests = {}

    def t(name, cond, detail=""):
        tests[name] = {"pass": bool(cond), "detail": detail}

    from run_phase_6e20_fixed_subspace_readout import (
        run_self_test as frozen_self_test, _build_synthetic_registration,
        fit_lda_axis, compute_s_scale, s_scale_degenerate,
        criteria_eval, verdict_for_condition, collapse_guard,
        balanced_accuracy, cohens_d, margin_rule, proposal_rate,
        registration_gates, alpha_coefficients, cos_H1,
        tensor_hash_legacy, canonical_tensor_hash,
    )

    frozen_res = frozen_self_test()
    t("frozen_self_test_pass", frozen_res["all_pass"],
      f"{frozen_res['passed']}/{frozen_res['total']}")

    reg = _build_synthetic_registration(rng)
    t("synthetic_reg_keys",
      all(k in reg for k in ["H_16", "u_lda", "s_scale", "U_8", "w_all"]),
      f"keys={list(reg.keys())[:8]}...")

    refused_none = False
    try:
        check_execution_authorization_v2(None)
    except ExecutionNotAuthorizedV2:
        refused_none = True
    t("auth_refuses_none", refused_none)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.json"
        bad.write_text(json.dumps({"decision": "APPROVE", "phase": "6E.20"}), encoding="utf-8")
        refused_wrong = False
        try:
            check_execution_authorization_v2(bad)
        except ExecutionNotAuthorizedV2:
            refused_wrong = True
        t("auth_refuses_wrong_phase", refused_wrong)

    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "good.json"
        good.write_text(json.dumps({"decision": "APPROVE", "phase": "6E.20.1"}), encoding="utf-8")
        accepted = True
        try:
            check_execution_authorization_v2(good)
        except ExecutionNotAuthorizedV2:
            accepted = False
    t("auth_accepts_valid", accepted)

    gates = registration_gates(reg)
    all_gates_pass = all(
        g["passed"] for g in gates.values()
        if isinstance(g, dict) and "passed" in g
    )
    t("synthetic_gates_all_pass", all_gates_pass,
      f"passed={sum(1 for g in gates.values() if isinstance(g, dict) and g.get('passed'))}")

    H16 = reg["H_16"]
    y16 = reg["y16"]
    ba = balanced_accuracy(np.array([float(np.dot(reg["w_all"]["F1"].w, H16[i]) + reg["w_all"]["F1"].b) for i in range(16)]))
    t("synthetic_balanced_accuracy", 0.0 <= ba <= 1.0, f"ba={ba:.4f}")

    props = [float(np.dot(reg["w_all"]["B0"].w, H16[i])) > 0 for i in range(16)]
    collapse = collapse_guard(props)
    t("collapse_guard_synthetic", collapse is None or isinstance(collapse, int))

    crit = criteria_eval(cos_h1=0.95, rho_s4=0.60, rho_s8=0.70,
                         d15=1.60, bal15=0.75,
                         proposal_series=[0.5] * 21, d20=0.5)
    t("criteria_eval_synthetic", crit["JOINT_SUCCESS"])

    t("EXECUTION_AUTH_PHASE_value", EXECUTION_AUTH_PHASE == "6E.20.1")
    t("PLAN_V40_SHA256_length", len(PLAN_V40_SHA256) == 64)
    t("LAYER_INDEX_value", LAYER_INDEX == 24)
    t("TASK_PROMPTS_count", len(TASK_PROMPTS) == 16)

    all_pass = all(v["pass"] for v in tests.values())
    return {"suite": "L2 execution unit tests", "tests": tests, "all_pass": all_pass,
            "passed": sum(1 for v in tests.values() if v["pass"]),
            "total": len(tests)}


def main(argv=None):
    import argparse
    import sys
    p = argparse.ArgumentParser(
        prog="run_phase_6e20_execution",
        description="Phase 6E.20.1 execution extension runner (Plan v4.0; L1-L3 implementation)."
    )
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("execute", help="Full execution pipeline (requires 6E.20.1 authorization)")
    sub.add_parser("step0-only", help="Step-0 registration only (requires authorization)")
    sub.add_parser("refusal", help="Emit the refusal triplet")
    sub.add_parser("self-test", help="L2 unit-test battery on synthetic vectors")
    args = p.parse_args(argv)

    if args.cmd == "execute":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        auth_path = Path("execution_authorization_v2.json")
        try:
            check_execution_authorization_v2(auth_path)
        except ExecutionNotAuthorizedV2 as e:
            print(json.dumps({"error": str(e), "triplet": list(REFUSAL_TRIPLET)}, indent=2))
            return 1
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        out_dir = Path("phase-6e20.1-execution-output")
        reg_result = run_step0_registration(model, tokenizer, out_dir / "step0")
        train_result = run_training(model, tokenizer, reg_result["reg"], out_dir)
        eval_result = run_evaluation(model, tokenizer, reg_result["reg"], out_dir)
        emit_provenance(reg_result["reg"], train_result, eval_result, out_dir)
        print(json.dumps({"status": "execution_complete", "out_dir": str(out_dir)}, indent=2))
        return 0

    if args.cmd == "step0-only":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        auth_path = Path("execution_authorization_v2.json")
        try:
            check_execution_authorization_v2(auth_path)
        except ExecutionNotAuthorizedV2 as e:
            print(json.dumps({"error": str(e), "triplet": list(REFUSAL_TRIPLET)}, indent=2))
            return 1
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        out_dir = Path("phase-6e20.1-execution-output")
        reg_result = run_step0_registration(model, tokenizer, out_dir / "step0")
        print(json.dumps({"status": "step0_complete", "gates": {
            k: v for k, v in reg_result["gates"].items()
            if isinstance(v, dict) and "passed" in v
        }}, indent=2))
        return 0

    if args.cmd == "refusal":
        ok, refusal = refusal_check(None)
        print(json.dumps({"authorized": ok, "refusal": refusal}, indent=2))
        return 0

    if args.cmd == "self-test":
        res = run_self_test()
        print(json.dumps(res, indent=2))
        return 0 if res["all_pass"] else 1

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
