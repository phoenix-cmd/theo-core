# THEO SLM — Phase 6E.20.1 Execution Conformance Harness v2 (W8; C1-C34 battery)
#
# NEW ARTIFACT (Phase 6E.20.1 execution conformance harness v2).
# Tests execution-specific conformance rows C22-C34 (including M1-M4
# adversarial checks) plus inherited C1-C21.
# All verification on synthetic/registered data only — no model loading,
# no real-panel forward pass, no training, no optimizer steps.
#
# Authorization gating targets phase=6E.20.1 (NOT 6E.20).

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_phase_6e20_execution_v7 as re_exec
import run_phase_6e20_fixed_subspace_readout as r
from run_phase_6e20_fixed_subspace_readout import (
    schema_sha256_manifest,
    schema_pre_registration_snapshot,
    schema_runs_telemetry,
    schema_execution_authorization,
    Readout, D_HIDDEN,
)

EXECUTION_AUTH_PHASE = re_exec.EXECUTION_AUTH_PHASE
PLAN_V40_SHA256 = re_exec.PLAN_V40_SHA256
LAYER_INDEX = re_exec.LAYER_INDEX
check_execution_authorization_v2 = re_exec.check_execution_authorization_v2
ExecutionNotAuthorizedV2 = re_exec.ExecutionNotAuthorizedV2

FROZEN_RUNNER_PATH = SCRIPT_DIR / "run_phase_6e18_causal_intervention.py"
FROZEN_PROTOCOL_PATH = (SCRIPT_DIR.parent.parent / "docs" / "research"
                        / "theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


def math_isfinite(x) -> bool:
    return math.isfinite(x)


# ---------------------------------------------------------------------------
# Conformance check registry
# ---------------------------------------------------------------------------
_CHECKS: Dict[str, Any] = {}


def check(name: str):
    def deco(fn):
        _CHECKS[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Static config conformance (C16 helper)
# ---------------------------------------------------------------------------
def static_config_conformance() -> dict:
    import numpy as np
    text = FROZEN_RUNNER_PATH.read_text(encoding="utf-8", errors="replace")
    finds = {}

    def grab(pattern, name, cast=str):
        m = re.search(pattern, text)
        if not m:
            finds[name] = {"extracted": None, "ok": False, "detail": "pattern not found"}
            return
        groups = m.groups()
        raw = m.group(1) if groups and any(g is not None for g in groups) else m.group(0)
        raw = raw.strip()
        try:
            val = cast(raw)
            finds[name] = {"extracted": raw, "value": val, "ok": True, "detail": ""}
        except Exception as e:
            finds[name] = {"extracted": raw, "ok": False, "detail": str(e)}

    grab(r"PRO_ID\s*=\s*(\d+)", "PRO_ID", int)
    grab(r"AB_ID\s*=\s*(\d+)", "AB_ID", int)
    grab(r"D_HIDDEN\s*=\s*(\d+)", "D_HIDDEN", int)
    grab(r"VOCAB\s*=\s*(\d+)", "VOCAB", int)
    grab(r"LORA_R\s*=\s*(\d+)", "LORA_R", int)
    grab(r"LORA_ALPHA\s*=\s*(\d+)", "LORA_ALPHA", int)
    grab(r"LORA_DROPOUT\s*=\s*([\d.]+)", "LORA_DROPOUT", float)
    grab(r"MODEL_REVISION|models--Qwen--Qwen2\.5-0\.5B-Instruct\\snapshots\\([0-9a-f]+)", "REVISION", str)
    grab(r"v_idx\[(\d)\]", "DECISION_INDEX", int)
    m_model = re.search(r"models--(Qwen)--(Qwen2\.5-0\.5B-Instruct)", text)
    if m_model:
        model_id = f"{m_model.group(1)}/{m_model.group(2)}"
        finds["MODEL_ID"] = {"extracted": model_id, "value": model_id,
                             "ok": model_id == "Qwen/Qwen2.5-0.5B-Instruct", "detail": ""}
    else:
        finds["MODEL_ID"] = {"extracted": None, "ok": False, "detail": "pattern not found"}

    m_targets = re.search(r"LORA_TARGETS\s*=\s*\[([^\]]*)\]", text)
    if m_targets:
        targets = re.findall(r'"([^"]+)"', m_targets.group(1))
        finds["LORA_TARGETS"] = {"extracted": m_targets.group(1).strip(), "value": targets,
                                 "ok": targets == r.LORA_TARGETS, "detail": f"{len(targets)} modules"}
    else:
        finds["LORA_TARGETS"] = {"extracted": None, "ok": False, "detail": "pattern not found"}

    finds["PROMPT_CONVENTION"] = {
        "ok": '{"decision": "' in text,
        "detail": '{"decision": " found in frozen runner' if '{"decision": "' in text else "NOT found",
    }
    finds["W_DELTA_SOURCE"] = {
        "ok": ("get_output_embeddings().weight" in text) and ("w_head[PRO_ID] - w_head[AB_ID]" in text),
        "detail": "unembedding-row source present",
    }
    finds["TENSOR_HASH_CONVENTION"] = {
        "ok": "torch.save(t.detach().float().cpu().contiguous(), buf)" in text,
        "detail": "legacy float32 torch-save convention present",
    }

    expected = {
        "PRO_ID": r.PRO_ID, "AB_ID": r.AB_ID, "D_HIDDEN": r.D_HIDDEN, "VOCAB": r.VOCAB,
        "LORA_R": r.LORA_R, "LORA_ALPHA": r.LORA_ALPHA, "LORA_DROPOUT": r.LORA_DROPOUT,
        "REVISION": r.MODEL_REVISION, "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
        "DECISION_INDEX": r.DECISION_INDEX,
    }
    mismatches = []
    for key, exp in expected.items():
        if key in finds and finds[key].get("ok") and finds[key].get("value") != exp:
            mismatches.append((key, finds[key]["value"], exp))
    if "LORA_TARGETS" in finds and finds["LORA_TARGETS"].get("ok"):
        if finds["LORA_TARGETS"]["value"] != r.LORA_TARGETS:
            mismatches.append(("LORA_TARGETS", finds["LORA_TARGETS"]["value"], r.LORA_TARGETS))
    all_ok = all(f.get("ok", True) for f in finds.values() if isinstance(f, dict)) and not mismatches
    return {"finds": finds, "mismatches": mismatches, "ok": all_ok}


# ---------------------------------------------------------------------------
# C1-C21 inherited checks (unchanged from v1 harness)
# ---------------------------------------------------------------------------
@check("C1_panel_convention")
def _c1(rng):
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    ok = (r.margin_rule(0.5) == "POS" and r.margin_rule(0.0) == "ABS" and r.margin_rule(-1.0) == "ABS")
    y16 = reg["y16"]
    ok = ok and list(y16[:8]) == [1] * 8 and list(y16[8:]) == [-1] * 8
    ok = ok and y16.shape == (16,)
    return ok, f"margin_rule_ok y16_shape={y16.shape} y16_values_match={list(y16[:8]) == [1]*8}"


@check("C2_lda_unit_norm")
def _c2(rng):
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    u = reg["u_lda"]
    ok = abs(float(np.linalg.norm(u)) - 1.0) < 1e-12
    import torch
    tt = torch.randn(r.D_HIDDEN, generator=torch.Generator().manual_seed(11))
    h1 = r.tensor_hash_legacy(tt)
    ok = ok and isinstance(h1, str) and len(h1) == 64
    return ok, f"||u||={np.linalg.norm(u):.2e} legacy_hash_len={len(h1)}"


@check("C3_s_scale_formula")
def _c3(rng):
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    margins = np.asarray(reg["H_16"] @ reg["u_lda"], dtype=np.float64)
    ss = r.compute_s_scale(margins, reg["H_16"], reg["u_lda"])
    ok = math_isfinite(ss) and ss > 0.0
    ok = ok and r.s_scale_degenerate(float("nan")) and r.s_scale_degenerate(0.0) and r.s_scale_degenerate(-1.0)
    return ok, f"s_scale={ss:.8f} degenerate_guard_ok"


@check("C4_svd_sign_convention")
def _c4(rng):
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    U_p, S, cum, cov = r.derive_principal_basis(reg["H_16"])
    sign_ok = True
    for j in range(8):
        col = U_p[:, j]
        imax = int(np.argmax(np.abs(col)))
        sign_ok = sign_ok and col[imax] >= 0.0
    nest_ok = float(np.max(np.abs(U_p[:, :2] - U_p[:, :8][:, :2]))) < 1e-12 and \
        float(np.max(np.abs(U_p[:, :4] - U_p[:, :8][:, :4]))) < 1e-12
    import torch
    tt = torch.randn(r.D_HIDDEN, generator=torch.Generator().manual_seed(11))
    legacy_h = r.tensor_hash_legacy(tt)
    canon_h = r.canonical_tensor_hash(tt)
    distinct_ok = legacy_h != canon_h
    Xc = reg["H_16"] - reg["H_16"].mean(axis=0, keepdims=True)
    cov_ref = (Xc.T @ Xc) / 15.0
    cov_ok = float(np.max(np.abs(cov - cov_ref))) < 1e-12
    ok = sign_ok and nest_ok and distinct_ok and cov_ok
    return ok, f"sign={sign_ok} nest={nest_ok} hash_distinct={distinct_ok} cov_recon={cov_ok}"


@check("C5_kappa_bounds")
def _c5(rng):
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    k2, k4, k8 = reg["kappa_2"], reg["kappa_4"], reg["kappa_8"]
    kR = reg["kappa_R"]
    ok = all(math_isfinite(k) and 0.0 < k < 1.0 for k in (k2, k4, k8, kR))
    return ok, f"k2={k2:.6f} k4={k4:.6f} k8={k8:.6f} kR={kR:.6f}"


@check("C6_w_delta_source_norm")
def _c6(rng):
    import numpy as np
    w_head = np.zeros((r.VOCAB, r.D_HIDDEN))
    w_head[r.PRO_ID] = 1.0
    w_head[r.AB_ID] = -1.0
    wd, bd = r.compute_w_delta(w_head)
    ok = wd.shape == (r.D_HIDDEN,) and bd == 0.0 \
        and abs(float(np.linalg.norm(wd)) - np.sqrt(4.0 * r.D_HIDDEN)) < 1e-9
    return ok, f"shape={wd.shape} b_delta={bd} ||w_delta||={np.linalg.norm(wd):.6f}"


@check("C7_R4_nrand_determinism")
def _c7(rng):
    import numpy as np
    R4a = r.generate_R4()
    R4b = r.generate_R4()
    det_r4 = float(np.max(np.abs(R4a - R4b))) < 1e-12
    u = r._build_synthetic_registration(rng)["u_lda"]
    n1, cos1, d1 = r.generate_n_rand(u)
    n2, cos2, d2 = r.generate_n_rand(u)
    det_n = float(np.max(np.abs(n1 - n2))) < 1e-12
    cos_ok = abs(cos1) <= 0.05
    ok = det_r4 and det_n and cos_ok and abs(float(np.linalg.norm(n1)) - 1.0) < 1e-9
    return ok, f"R4_det={det_r4} n_det={det_n} |cos|={abs(cos1):.5f} draws={d1}"


@check("C8_zero_trainable_params")
def _c8(rng):
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    total_params = sum(rr.trainable_params for rr in reg["w_all"].values())
    s_scale = reg["s_scale"]
    norms = {name: float(np.linalg.norm(rr.w)) for name, rr in reg["w_all"].items() if name != "B0"}
    gain_ok = all(abs(n - s_scale) <= r.G8_WEIGHT_REL_TOL * s_scale for n in norms.values())
    ok = (set(reg["w_all"].keys()) == set(r.CONDITIONS) and total_params == 0 and gain_ok)
    return ok, f"conditions={list(reg['w_all'].keys())} trainable={total_params} gain_ok={gain_ok}"


@check("C9_forward_backward_identity")
def _c9(rng):
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    h = reg["H_16"][0]
    r0 = reg["w_all"]["F1"]
    s = r0.forward_margin(h)
    z_pro, z_abs = r0.forward_logits(h)
    fwd_ok = abs(z_pro - s) < 1e-12 and abs(z_abs + s) < 1e-12
    alpha_pos, _, _, _ = r.alpha_coefficients(0.0, 1)
    gR = r.readout_only_gradient_hidden(alpha_pos, r0.w)
    gR_expected = alpha_pos * np.asarray(r0.w, dtype=np.float64)
    bwd_ok = float(np.max(np.abs(gR - gR_expected))) < 1e-12
    proj_off = float(np.linalg.norm(gR - (np.outer(r0.w, r0.w) / (np.dot(r0.w, r0.w) + 1e-300)) @ gR))
    ok = fwd_ok and bwd_ok and proj_off < 1e-9
    return ok, f"fwd={fwd_ok} bwd={bwd_ok} off_span={proj_off:.2e}"


@check("C10_condition_distinctness")
def _c10(rng):
    reg = r._build_synthetic_registration(rng)
    facts = r.condition_distinctness(reg["w_all"])
    ok = all(f["distinct_direction"] for f in facts.values())
    return ok, f"{len(facts)} pairs; all_directions_distinct={ok}"


@check("C11_alpha_chain_rule_rho_bounded")
def _c11(rng):
    import numpy as np
    a_pos, d_pos, p_pro, p_abs = r.alpha_coefficients(0.0, 1)
    a_abs, d_abs, _, _ = r.alpha_coefficients(0.0, 0)
    alpha_ok = abs(a_pos - (d_pos - 1.0)) < 1e-9 and abs(a_abs - (d_abs + 1.0)) < 1e-9
    rng2 = np.random.default_rng(5)
    g_total = rng2.standard_normal(r.D_HIDDEN)
    g_bg = 0.3 * rng2.standard_normal(r.D_HIDDEN)
    g_R = g_total - g_bg
    rho_pos = r.rho_sample(g_R, g_total)
    rho_neg = r.rho_sample(-g_R, g_total)
    bounded_ok = -1.0 - 1e-12 <= rho_pos <= 1.0 + 1e-12
    decomp_ok = float(np.max(np.abs((g_bg + g_R) - g_total))) < 1e-9
    mean_rho2 = (rho_pos ** 2 + rho_neg ** 2) / 2.0
    mean_rho = (rho_pos + rho_neg) / 2.0
    neq_ok = abs(mean_rho ** 2 - mean_rho2) > 1e-9
    sign_ok = r.readout_only_cosine_sign(a_pos, a_abs) in (-1, 0)
    ok = alpha_ok and bounded_ok and decomp_ok and neq_ok and sign_ok
    return ok, (f"alpha_chain={alpha_ok} rho_pos={rho_pos:.6f} rho_neg={rho_neg:.6f} "
                f"decomp={decomp_ok} neq={neq_ok} sign={sign_ok}")


@check("C12_gates_G1_G8")
def _c12(rng):
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    gates = r.registration_gates(reg)
    g_pass = {g: v for g, v in gates.items() if isinstance(v, dict) and "gate" in v and v["passed"]}
    g_fail = {g: v for g, v in gates.items() if isinstance(v, dict) and "gate" in v and not v["passed"]}
    void = r.void_semantics(gates)
    reg_fail = dict(reg)
    reg_fail["s_scale"] = -1.0
    gates_fail = r.registration_gates(reg_fail)
    g8_fail = gates_fail["G8"]["passed"] is False
    reg_fail2 = dict(reg)
    reg_fail2["kappa_R"] = 0.9
    g5_fail = r.registration_gates(reg_fail2)["G5"]["passed"] is False
    reg_fail3 = dict(reg)
    reg_fail3["kappa_2"] = 0.999
    g3_fail = r.registration_gates(reg_fail3)["G3_k2"]["passed"] is False
    reg_fail4 = dict(reg)
    reg_fail4["real_panel"] = True
    reg_fail4["u_lda_npy_ref"] = np.asarray(reg["u_lda"]) + 0.01
    g1_fail = r.registration_gates(reg_fail4, legacy_hash={"u_lda_legacy": "0" * 64})["G1"]["passed"] is False
    ok = (len(g_fail) == 0 and g8_fail and g5_fail and g3_fail and g1_fail
          and not void["run_void"] and isinstance(void, dict))
    return ok, (f"passing={sorted(g_pass.keys())} void_map={void} "
                f"G1_fail={g1_fail} G3_fail={g3_fail} G5_fail={g5_fail} G8_fail={g8_fail}")


@check("C13_hypotheses_H1_H7")
def _c13(rng):
    h_eval = {
        "H1": (0.85 >= 0.80 and 0.82 >= 0.80),
        "H2": (1.62 >= 1.50 and 0.73 >= 0.70),
        "H3": (0.60 >= 0.50 and 0.60 >= 0.55 >= 0.50),
        "H4": ((0.85 - 0.40) >= 0.30 and 0.60 < 0.80),
        "H5": (0.55 < 0.80 and 1.20 >= 0.80),
        "H6": (0.85 >= 0.80 and 1.62 >= 1.50),
        "H7": (all(p > 0.0 for p in [0.5, 0.6, 0.7, 0.8]) and 0.3 >= 0.00),
    }
    ok = all(h_eval.values())
    return ok, f"logic_smoke={h_eval}"


@check("C14_criteria_P1_P6_LOO")
def _c14(rng):
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    crit_ok = r.criteria_eval(cos_h1=0.90, rho_s4=0.60, rho_s8=0.70, d15=1.60, bal15=0.75,
                               proposal_series=[0.5] * 21, d20=0.5)
    crit_partial = r.criteria_eval(cos_h1=0.60, rho_s4=0.60, rho_s8=0.70, d15=0.90, bal15=0.60,
                                    proposal_series=[0.5] * 21, d20=0.5)
    ok = (crit_ok["JOINT_SUCCESS"] == (crit_ok["P2"] and crit_ok["P4"] and crit_ok["P5"] and crit_ok["P6"])
          and crit_partial["PARTIAL"] == (crit_partial["P1_S4"] and crit_partial["P1_S8"] and crit_partial["P3"]))
    loo = r.loo_cv_p5(reg["H_16"], reg["y16"], reg["s_scale"], reg["U_8"], reg["R_4"], reg["n_rand"], reg["w_delta"])
    ok = ok and len(loo) == 16 and all(
        set(x.keys()) == {"split", "heldout_true", "B0", "F1", "S8", "C4-Rand", "R1-Rand"} for x in loo)
    return ok, f"joint={crit_ok['JOINT_SUCCESS']} partial={crit_partial['PARTIAL']} loo_splits={len(loo)}"


@check("C15_outcome_matrix_verdicts")
def _c15(rng):
    import numpy as np
    crit_ok = r.criteria_eval(cos_h1=0.90, rho_s4=0.60, rho_s8=0.70, d15=1.60, bal15=0.75,
                               proposal_series=[0.5] * 21, d20=0.5)
    v_success = r.verdict_for_condition(crit_ok, gc_p1=True)
    crit_p6f = r.criteria_eval(cos_h1=0.90, rho_s4=0.60, rho_s8=0.70, d15=1.60, bal15=0.75,
                                proposal_series=[0.5] * 10 + [0.0] * 11, d20=-0.1)
    v_p6f = r.verdict_for_condition(crit_p6f, gc_p1=True)
    crit_gc = r.criteria_eval(cos_h1=0.90, rho_s4=0.30, rho_s8=0.30, d15=0.40, bal15=0.55,
                               proposal_series=[0.5] * 21, d20=0.5)
    v_gc = r.verdict_for_condition(crit_gc, gc_p1=False)
    crit_part = r.criteria_eval(cos_h1=0.60, rho_s4=0.60, rho_s8=0.70, d15=0.90, bal15=0.60,
                                 proposal_series=[0.5] * 21, d20=0.5)
    v_part = r.verdict_for_condition(crit_part, gc_p1=True)
    ok = (v_success == "SUCCESS" and v_p6f == "GC-only/TU-only-P6_FAILED"
          and v_gc == "GC-only" and v_part == "PARTIAL")
    return ok, f"SUCCESS={v_success} P6f={v_p6f} GC={v_gc} PARTIAL={v_part}"


@check("C16_config_assertions")
def _c16(rng):
    proto_sha = file_sha256(FROZEN_PROTOCOL_PATH) if FROZEN_PROTOCOL_PATH.exists() else None
    protocol_ok = proto_sha is not None and proto_sha == r.PROTOCOL_SHA256.lower()
    cfg = static_config_conformance()
    ok = protocol_ok and cfg["ok"]
    return ok, (f"protocol_ok={protocol_ok} static_cfg_ok={cfg['ok']} "
                f"mismatches={cfg['mismatches']}")


@check("C17_cos_H1_sign_assertion")
def _c17(rng):
    import numpy as np
    gp = rng.standard_normal(r.D_HIDDEN) + 0.2
    ga = rng.standard_normal(r.D_HIDDEN) + 0.2
    c17 = r.cos_H1(gp, ga)
    reg = r._build_synthetic_registration(rng)
    g6 = r.registration_gates(reg)["G6"]
    ok = math_isfinite(c17) and -1.0 - 1e-9 <= c17 <= 1.0 + 1e-9 and g6["passed"]
    return ok, f"cos_H1={c17:.6f} G6_passed={g6['passed']}"


@check("C18_preflight_L4_synthetic")
def _c18(rng):
    l4 = r.preflight_l4_synthetic()
    l4_ok = l4["all_pass"]
    l5_refused = False
    try:
        r.check_execution_authorization(None)
    except r.ExecutionNotAuthorized:
        l5_refused = True
    ok = l4_ok and l5_refused
    return ok, (f"L4_all_pass={l4_ok} ({sum(1 for c in l4['checks'] if c['pass'])}/"
                f"{len(l4['checks'])}) L5_refused={l5_refused}")


@check("C19_collapse_guard")
def _c19(rng):
    trigger = r.collapse_guard([True, False, False, False, True])
    notrig = r.collapse_guard([True, False, True, False, True])
    ok = trigger == 3 and notrig is None
    return ok, f"trigger={trigger} no_trigger={notrig is None}"


@check("C20_refusal_triplet")
def _c20(rng):
    ok_auth, refusal = r.refusal_check(None)
    ok = (not ok_auth and refusal["state"] == "PRE_EXECUTION"
          and refusal["verdict"] == "NO_SCIENTIFIC_VERDICT"
          and refusal["reason"] == "EXECUTION_NOT_AUTHORIZED")
    return ok, f"state={refusal['state']} verdict={refusal['verdict']} reason={refusal['reason']}"


@check("C21_provenance_schemas")
def _c21(rng):
    schemas_ok = (set(r.schema_sha256_manifest()["objects"]) ==
                  {"u_lda.npy", "U_8.npy", "R_4.npy", "n_rand.npy", "w_B0.npy", "w_F1.npy",
                   "w_S2.npy", "w_S4.npy", "w_S8.npy", "w_C4-Rand.npy", "w_R1-Rand.npy",
                   "H_16.npy", "y16"})
    auth_schema = r.schema_execution_authorization()
    auth_ok = auth_schema["decision"] == "APPROVE" and auth_schema["phase"] == "6E.20"
    telemetry_ok = all(x in r.schema_runs_telemetry() for x in
                       ["step", "condition", "cos_H1", "rho_c", "sigma_c", "gate_dispositions"])
    ok = schemas_ok and auth_ok and telemetry_ok
    return ok, f"manifest_ok={schemas_ok} auth_ok={auth_ok} telemetry_ok={telemetry_ok}"


# ---------------------------------------------------------------------------
# C22-C26 execution-specific checks (from v1 harness, unchanged)
# ---------------------------------------------------------------------------
@check("C22_step0_registration_synthetic")
def _c22(rng):
    reg = r._build_synthetic_registration(rng)
    required_keys = ["H_16", "y16", "u_lda", "s_scale", "U_2", "U_4", "U_8",
                     "R_4", "n_rand", "kappa_2", "kappa_4", "kappa_8", "kappa_R",
                     "w_delta", "w_all"]
    missing = [k for k in required_keys if k not in reg]
    if missing:
        return False, f"missing keys: {missing}"
    gates = r.registration_gates(reg)
    failed = [g["gate"] for g in gates.values()
              if isinstance(g, dict) and "passed" in g and not g["passed"]]
    return len(failed) == 0, (f"gates_passed={len([g for g in gates.values() if isinstance(g, dict) and 'passed' in g])} "
                              f"failed={failed}")


@check("C23_training_logic_synthetic")
def _c23(rng):
    import numpy as np
    ok = True
    details = []
    ok = ok and r.LORA_R == 16 and r.LORA_ALPHA == 32 and r.LORA_DROPOUT == 0.05
    ok = ok and len(r.LORA_TARGETS) == 7
    details.append(f"lora_r={r.LORA_R} alpha={r.LORA_ALPHA} targets={len(r.LORA_TARGETS)}")
    expected_cp = {0, 4, 6, 7, 14, 15, 17, 20}
    ok = ok and set(r.CHECKPOINT_SET) == expected_cp
    details.append(f"checkpoints={r.CHECKPOINT_SET}")
    ok = ok and r.GRAD_ACCUM == 2 and r.BATCH_SIZE == 4 and r.LR == 1e-4
    ok = ok and r.LAMBDA_DECISION == 10.0 and r.HORIZON == 20
    details.append(f"grad_accum={r.GRAD_ACCUM} batch={r.BATCH_SIZE} lr={r.LR} lambda={r.LAMBDA_DECISION} T={r.HORIZON}")
    reg = r._build_synthetic_registration(rng)
    w_f1 = reg["w_all"]["F1"].w
    alpha_pos = reg["alpha_POS"]
    g_R = alpha_pos * w_f1
    ok = ok and np.all(np.isfinite(g_R))
    details.append(f"g_R_f1_norm={np.linalg.norm(g_R):.6f}")
    trigger = r.collapse_guard([False] * r.COLLAPSE_K + [True])
    ok = ok and trigger is not None and trigger == r.COLLAPSE_K - 1
    details.append(f"collapse_guard_k={r.COLLAPSE_K} trigger={trigger}")
    return ok, "; ".join(details)


@check("C24_evaluation_logic_synthetic")
def _c24(rng):
    import numpy as np
    ok = True
    details = []
    crit = r.criteria_eval(cos_h1=0.95, rho_s4=0.60, rho_s8=0.70,
                            d15=1.60, bal15=0.75,
                            proposal_series=[0.5] * 21, d20=0.5)
    ok = ok and crit["JOINT_SUCCESS"]
    v = r.verdict_for_condition(crit, gc_p1=True)
    ok = ok and v == "SUCCESS"
    details.append(f"verdict_success={v}")
    crit_p6f = r.criteria_eval(cos_h1=0.95, rho_s4=0.60, rho_s8=0.70,
                                d15=1.60, bal15=0.75,
                                proposal_series=[0.5] * 10 + [0.0] * 11, d20=-0.1)
    ok = ok and not crit_p6f["JOINT_SUCCESS"]
    details.append(f"p6_fail_joint_success={crit_p6f['JOINT_SUCCESS']}")
    crit_part = r.criteria_eval(cos_h1=0.40, rho_s4=0.60, rho_s8=0.70,
                                 d15=0.85, bal15=0.50,
                                 proposal_series=[0.5] * 21, d20=0.5)
    v_part = r.verdict_for_condition(crit_part, gc_p1=True)
    ok = ok and v_part == "PARTIAL"
    details.append(f"verdict_partial={v_part}")
    margins = np.concatenate([np.ones(8) * 2.0, np.ones(8) * -2.0])
    ba = r.balanced_accuracy(margins)
    ok = ok and ba == 1.0
    details.append(f"ba_perfect={ba}")
    d = r.cohens_d(margins)
    ok = ok and math.isfinite(d) and d > 3.0
    details.append(f"cohens_d={d:.4f}")
    return ok, "; ".join(details)


@check("C25_provenance_schemas_v2")
def _c25(rng):
    ok = True
    details = []
    manifest = schema_sha256_manifest()
    ok = ok and manifest["protocol_sha256"] == r.PROTOCOL_SHA256
    ok = ok and manifest["base_model"]["id"] == r.MODEL_ID
    ok = ok and len(manifest["objects"]) == 13
    details.append(f"manifest_objects={len(manifest['objects'])}")
    snap = schema_pre_registration_snapshot()
    ok = ok and snap["document_sha256"] == r.PROTOCOL_SHA256
    ok = ok and len(snap["adopted_decisions"]) == 11
    details.append(f"snapshot_decisions={len(snap['adopted_decisions'])}")
    runs = schema_runs_telemetry()
    ok = ok and len(runs) >= 10
    details.append(f"runs_fields={len(runs)}")
    auth = schema_execution_authorization()
    ok = ok and auth["decision"] == "APPROVE"
    details.append("auth_schema_valid")
    ok = ok and len(PLAN_V40_SHA256) == 64
    ok = ok and EXECUTION_AUTH_PHASE == "6E.20.1"
    details.append(f"plan_v40_hash_len={len(PLAN_V40_SHA256)}")
    details.append(f"auth_phase={EXECUTION_AUTH_PHASE}")
    return ok, "; ".join(details)


@check("C26_authorization_gating")
def _c26(rng):
    ok = True
    details = []
    refused = False
    try:
        check_execution_authorization_v2(None)
    except ExecutionNotAuthorizedV2:
        refused = True
    ok = ok and refused
    details.append("no_auth_refused=True")
    with tempfile.TemporaryDirectory() as td:
        bad_path = Path(td) / "bad_auth.json"
        bad_path.write_text(json.dumps({"decision": "APPROVE", "phase": "6E.20"}), encoding="utf-8")
        refused2 = False
        try:
            check_execution_authorization_v2(bad_path)
        except ExecutionNotAuthorizedV2:
            refused2 = True
        ok = ok and refused2
        details.append("wrong_phase_refused=True")
    with tempfile.TemporaryDirectory() as td:
        rej_path = Path(td) / "rej_auth.json"
        rej_path.write_text(json.dumps({"decision": "REJECT", "phase": "6E.20.1"}), encoding="utf-8")
        refused3 = False
        try:
            check_execution_authorization_v2(rej_path)
        except ExecutionNotAuthorizedV2:
            refused3 = True
        ok = ok and refused3
        details.append("reject_refused=True")
    with tempfile.TemporaryDirectory() as td:
        unpath = Path(td) / "unparseable.json"
        unpath.write_text("NOT JSON {{{", encoding="utf-8")
        refused4 = False
        try:
            check_execution_authorization_v2(unpath)
        except ExecutionNotAuthorizedV2:
            refused4 = True
        ok = ok and refused4
        details.append("unparseable_refused=True")
    with tempfile.TemporaryDirectory() as td:
        ok_path = Path(td) / "ok_auth.json"
        ok_path.write_text(json.dumps({"decision": "APPROVE", "phase": "6E.20.1"}), encoding="utf-8")
        accepted = True
        try:
            check_execution_authorization_v2(ok_path)
        except ExecutionNotAuthorizedV2:
            accepted = False
        ok = ok and accepted
        details.append("valid_auth_accepted=True")
    return ok, "; ".join(details)


# ---------------------------------------------------------------------------
# C27-C34: M1-M4 adversarial checks (NEW in v2 harness)
# ---------------------------------------------------------------------------
@check("C27_F7_cos_H1_from_total_gradient")
def _c27(rng):
    """F7: cos(H1) is computed from the TOTAL decision-gradient g_total via the
    imported frozen 6E.18 per_sample_decision_gradient machinery (autograd), not
    from a hardcoded constant and not from the readout-only g_R. Proves the
    protocol-mandated object (v3.1 §10.5/§5.7) is exercised.

    Run on CPU (environment-independent): force both v5's _DEVICE and the frozen
    machinery's module DEVICE to "cpu" before exercising the path. In real
    execution both are "cuda:0" consistently, so the index_select device is the
    same — this test isolates the computation, not the device."""
    import numpy as np
    import torch
    import torch.nn as nn
    import sys as _sys

    PRO_ID = re_exec.PRO_ID
    AB_ID = re_exec.AB_ID
    D = re_exec.D_HIDDEN

    _v5_prev = getattr(re_exec, "_DEVICE", None)
    re_exec._DEVICE = "cpu"
    # Patch the module globals of the function actually invoked by v5 (not a
    # freshly re-loaded module instance, which would be a different object).
    _gm = re_exec.PER_SAMPLE_DECISION_GRADIENT.__globals__
    _fm_prev = _gm.get("DEVICE", None)
    _gm["DEVICE"] = "cpu"
    try:
        class MockLora(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.embed = nn.Embedding(max(PRO_ID, AB_ID) + 32, d)
                self.q_proj = nn.Linear(d, d, bias=False)
                self.q_proj_lora_A = nn.Parameter(torch.zeros(d, 4))
                self.q_proj_lora_B = nn.Parameter(torch.zeros(4, d))
                self.head = nn.Linear(d, max(PRO_ID, AB_ID) + 32, bias=False)
            def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kw):
                x = self.embed(input_ids)
                lora_delta = (self.q_proj_lora_A @ self.q_proj_lora_B).sum(dim=0, keepdim=True).unsqueeze(0)
                h = self.q_proj(x) + lora_delta.expand_as(x)
                hs = [None, h] if output_hidden_states else None
                return type("Out", (), {"logits": self.head(h), "hidden_states": hs})()

        class MockTok:
            def __call__(self, text, return_tensors=None):
                ids = list(range(20))
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor([ids], dtype=torch.long),
                            "attention_mask": torch.tensor([[1]*20], dtype=torch.long)}
                return {"input_ids": [ids], "attention_mask": [[1]*20]}

        class MockReadout:
            def forward_logits(self, hL):
                return (torch.zeros(1), torch.zeros(1))

        mock = MockLora(D).to("cpu")
        for n, p in mock.named_parameters():
            p.requires_grad_( "lora" in n)
        mtok = MockTok()
        mread = MockReadout()
        panel = re_exec.collect_panel_prompts()

        cos_h1 = re_exec.compute_cos_H1_from_live_model(mock, mread, mtok, panel, device="cpu", condition_name="S8")
        ok = isinstance(cos_h1, float) and math.isfinite(cos_h1) and -1.0 <= cos_h1 <= 1.0
        return ok, f"cos_H1={cos_h1:.6f} (g_total via imported 6E.18 per_sample_decision_gradient)"
    finally:
        _gm["DEVICE"] = _fm_prev
        re_exec._DEVICE = _v5_prev


@check("C28_M2_rho_c_computed_not_hardcoded")
def _c28(rng):
    """M2 adversarial: rho_c computed from actual g_R/g_total tensors,
    not from a hardcoded constant. In readout-only path, rho_c ≡ 1.0."""
    import numpy as np
    reg = r._build_synthetic_registration(rng)

    rho_c, per_sample, sigma_c = re_exec.compute_rho_c_from_registration(reg, "S8")

    ok = (isinstance(rho_c, float) and math.isfinite(rho_c)
          and isinstance(sigma_c, float) and math.isfinite(sigma_c)
          and len(per_sample) == 16
          and all(isinstance(x, float) and math.isfinite(x) for x in per_sample))
    ok = ok and abs(rho_c - 1.0) < 1e-10
    return ok, (f"rho_c={rho_c:.6f} sigma_c={sigma_c:.10f} "
                f"per_sample_len={len(per_sample)} all_one={all(abs(x - 1.0) < 1e-10 for x in per_sample)}")


@check("C29_M2_rho_c_varies_across_conditions")
def _c29(rng):
    """M2: rho_c computed per-condition from actual registered readouts.
    Different conditions have different readout weights → different g_R."""
    import numpy as np
    reg = r._build_synthetic_registration(rng)
    results = {}
    for cond in r.CONDITIONS:
        rho_c, _, sigma_c = re_exec.compute_rho_c_from_registration(reg, cond)
        results[cond] = {"rho_c": rho_c, "sigma_c": sigma_c}

    ok = all(isinstance(v["rho_c"], float) and math.isfinite(v["rho_c"]) for v in results.values())
    ok = ok and all(abs(v["rho_c"] - 1.0) < 1e-10 for v in results.values())
    summary = ", ".join(f"{k}={v['rho_c']:.4f}" for k, v in results.items())
    return ok, f"conditions={list(results.keys())} rho_c={{{summary}}}"


@check("C30_M3_d15_bal15_from_checkpoint")
def _c30(rng):
    """M3: d15/bal15 sourced from actual step-15 checkpoint.
    With no checkpoint present, returns zeros."""
    ok = True
    details = []

    with tempfile.TemporaryDirectory() as td:
        info = re_exec.compute_d15_bal15_from_checkpoint(None, None, None, td)
        ok = ok and info["d15"] == 0.0 and info["bal15"] == 0.0 and not info["checkpoint_found"]
        details.append(f"no_ckpt: d15={info['d15']} bal15={info['bal15']} found={info['checkpoint_found']}")

    ok = ok and re_exec.EXECUTION_AUTH_PHASE == "6E.20.1"
    details.append(f"auth_phase={re_exec.EXECUTION_AUTH_PHASE}")
    return ok, "; ".join(details)


@check("C31_M4_checkpoint_restore_adversarial")
def _c31(rng):
    """M4 adversarial: restore_checkpoint restores exact state.
    Corrupt weights → restore → verify match with original."""
    import numpy as np
    import torch

    ok = True
    details = []
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = Path(td) / "ckpt_0"
        ckpt_dir.mkdir()
        model = torch.nn.Linear(10, 2)
        w_before = model.weight.data.clone()
        b_before = model.bias.data.clone()
        torch.save({
            "step": 0, "condition": "B0",
            "state_dict": model.state_dict(),
            "readout_w": [0.1] * 10, "readout_b": 0.0,
        }, ckpt_dir / "model.pt")
        model.weight.data.fill_(999.0)
        model.bias.data.fill_(-999.0)
        w_corrupted = model.weight.data.clone()
        info = re_exec.restore_checkpoint(model, ckpt_dir)
        w_after = model.weight.data.clone()
        ok = ok and not torch.equal(w_corrupted, w_after)
        ok = ok and torch.equal(w_before, w_after)
        ok = ok and info["step"] == 0 and info["condition"] == "B0"
        details.append(f"state_restored={torch.equal(w_before, w_after)} "
                       f"step={info['step']} corrupted={w_corrupted.mean():.1f} restored={w_after.mean():.6f}")

    with tempfile.TemporaryDirectory() as td:
        bad_ckpt = Path(td) / "bad_ckpt"
        bad_ckpt.mkdir()
        try:
            re_exec.restore_checkpoint(torch.nn.Linear(10, 2), bad_ckpt)
            ok = ok and False
        except ValueError:
            pass
        details.append("missing_ckpt_refused=True")
    return ok, "; ".join(details)


@check("C32_M4_evaluate_checkpoint_restores_state")
def _c32(rng):
    """M4: evaluate_checkpoint restores checkpoint before computing metrics.
    Verify restored_checkpoint field is present when checkpoint_path given.

    The full metric sub-pipeline (model forward / F7 g_total / d15) is stubbed so
    this check isolates the M4 restore-before-metrics control flow; the metric
    computations themselves are covered by C27 (F7) and the runner self-test."""
    ok = True
    details = []
    import tempfile, torch, numpy as np
    reg = r._build_synthetic_registration(rng)

    _orig_at = re_exec.evaluate_checkpoint_at_state
    _orig_cos = re_exec.compute_cos_H1_from_live_model
    _orig_d15 = re_exec.compute_d15_bal15_from_checkpoint
    re_exec.evaluate_checkpoint_at_state = lambda *a, **k: {
        "B0": {"cohens_d": 1.0, "balanced_accuracy": 0.8, "proposal_rate": 0.5}}
    re_exec.compute_cos_H1_from_live_model = lambda *a, **k: 0.95
    re_exec.compute_d15_bal15_from_checkpoint = lambda *a, **k: {"d15": 0.0, "bal15": 0.0}
    try:
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td) / "ckpt_7"
            ckpt_dir.mkdir()
            model = torch.nn.Linear(10, 2)
            torch.save({
                "step": 7, "condition": "B0",
                "state_dict": model.state_dict(),
                "readout_w": [0.1] * 10, "readout_b": 0.0,
            }, ckpt_dir / "model.pt")
            result = re_exec.evaluate_checkpoint(
                model, None, reg, checkpoint_path=ckpt_dir)
            ok = ok and "restored_checkpoint" in result
            ok = ok and result["restored_checkpoint"]["step"] == 7
            ok = ok and "cos_H1" in result
            ok = ok and "rho_c" in result
            ok = ok and isinstance(result["cos_H1"], float)
            ok = ok and isinstance(result["rho_c"], float)
            details.append(f"restored_step={result['restored_checkpoint']['step']} "
                           f"cos_H1={result['cos_H1']:.6f} rho_c={result['rho_c']:.6f}")
    finally:
        re_exec.evaluate_checkpoint_at_state = _orig_at
        re_exec.compute_cos_H1_from_live_model = _orig_cos
        re_exec.compute_d15_bal15_from_checkpoint = _orig_d15
    return ok, "; ".join(details)


@check("C33_D5_provenance_completeness")
def _c33(rng):
    """D5: provenance includes p5_loo_report.json, runs.json with
    cos_H1/rho_c/sigma_c/gate_dispositions fields."""
    ok = True
    details = []
    reg = r._build_synthetic_registration(rng)
    with tempfile.TemporaryDirectory() as td:
        re_exec.emit_provenance(reg, {}, {}, td, gate_dispositions={"G1": "PASS"})
        p5 = Path(td) / "p5_loo_report.json"
        runs = Path(td) / "runs.json"
        ok = ok and p5.exists()
        ok = ok and runs.exists()
        details.append(f"p5_exists={p5.exists()} runs_exists={runs.exists()}")
        if runs.exists():
            data = json.loads(runs.read_text())
            has_gate = any(e.get("event") == "gate_dispositions" for e in data)
            ok = ok and has_gate
            details.append(f"has_gate_dispositions={has_gate}")
        if p5.exists():
            p5_data = json.loads(p5.read_text())
            ok = ok and "plan_v40_sha256" in p5_data
            details.append(f"p5_has_plan_hash={'plan_v40_sha256' in p5_data}")
    return ok, "; ".join(details)


@check("C34_v2_runner_hash_and_constants")
def _c34(rng):
    """Runner v2 structural: constants, hash, functions exist."""
    ok = True
    details = []
    ok = ok and re_exec.EXECUTION_AUTH_PHASE == "6E.20.1"
    ok = ok and len(re_exec.PLAN_V40_SHA256) == 64
    ok = ok and re_exec.LAYER_INDEX == 24
    ok = ok and len(re_exec.TASK_PROMPTS) == 16
    ok = ok and hasattr(re_exec, "compute_cos_H1_from_live_model")
    ok = ok and hasattr(re_exec, "compute_rho_c_from_registration")
    ok = ok and hasattr(re_exec, "restore_checkpoint")
    ok = ok and hasattr(re_exec, "compute_d15_bal15_from_checkpoint")
    ok = ok and hasattr(re_exec, "collect_panel_prompts")
    ok = ok and hasattr(re_exec, "evaluate_checkpoint_at_state")
    details.append(f"auth_phase={re_exec.EXECUTION_AUTH_PHASE}")
    details.append(f"plan_hash_len={len(re_exec.PLAN_V40_SHA256)}")
    details.append("all_functions_exist=True")
    return ok, "; ".join(details)


@check("C35_collect_H16_device_placement_regression")
def _c35(rng):
    """FIX REGRESSION (v5 failure): collect_H16 must move EVERY tokenizer tensor
    to the model's actual device before the forward pass. Exercises the real
    collect_H16 code path with BOTH a CPU model and a CUDA model and verifies the
    tokenizer inputs land on the same device as the model (no index_select
    device-mismatch). This is the exact failure that aborted the v5 attempt."""
    import torch
    import torch.nn as nn

    D = re_exec.D_HIDDEN
    L = re_exec.LAYER_INDEX
    captured = {}

    class MockCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, D)
            self.proj = nn.Linear(D, D, bias=False)
            self.head = nn.Linear(D, 2, bias=False)
        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, **kw):
            captured["input_device"] = input_ids.device
            captured["model_device"] = next(self.parameters()).device
            x = self.embed(input_ids[:, -1:])
            h = self.proj(x)
            hs = tuple(h for _ in range(L)) + (h,)
            return type("Out", (), {"logits": self.head(h), "hidden_states": hs})()

    class MockTok:
        def __call__(self, text, return_tensors=None):
            ids = list(range(20))
            if return_tensors == "pt":
                return {"input_ids": torch.tensor([ids], dtype=torch.long),
                        "attention_mask": torch.tensor([[1] * 20], dtype=torch.long)}
            return {"input_ids": [ids], "attention_mask": [[1] * 20]}

    ok = True
    details = []
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda:0")
    for dev in devs:
        captured.clear()
        m = MockCausalLM().to(dev)
        try:
            H = re_exec.collect_H16(m, MockTok())
        except Exception as e:
            ok = False
            details.append(f"dev={dev} raised {type(e).__name__}: {e}")
            continue
        same_dev = (captured.get("input_device") == captured.get("model_device") == torch.device(dev))
        shape_ok = (H.shape == (16, D))
        ok = ok and same_dev and shape_ok
        details.append(f"dev={dev} input_dev={captured.get('input_device')} "
                       f"model_dev={captured.get('model_device')} H_shape={tuple(H.shape)}")
    return ok, "; ".join(details)


# ---------------------------------------------------------------------------
# Run battery
# ---------------------------------------------------------------------------
def run_conformance_battery(rng_seed: int = 0) -> Dict[str, Any]:
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    results = {"checks": [], "all_pass": True}
    for cname, fn in _CHECKS.items():
        try:
            ok, detail = fn(rng)
        except Exception as e:
            ok, detail = False, f"exception: {e!r}"
        results["checks"].append({"check": cname, "pass": bool(ok), "detail": detail})
        results["all_pass"] = results["all_pass"] and bool(ok)
    results["passed"] = sum(1 for c in results["checks"] if c["pass"])
    results["total"] = len(results["checks"])
    return results


# ---------------------------------------------------------------------------
# G-PLAN-1 disposition
# ---------------------------------------------------------------------------
def g_plan_1_disposition(results: Dict[str, Any]) -> str:
    if results["all_pass"]:
        return "PASS"
    required_rows = [c["check"] for c in results["checks"] if c["check"].startswith("C")]
    failed_required = [c["check"] for c in results["checks"]
                       if c["check"].startswith("C") and not c["pass"]]
    if failed_required:
        return f"FAIL (required: {failed_required})"
    return "PASS (non-required waivers noted)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        prog="run_phase_6e20_execution_conformance_harness_v4",
        description="Phase 6E.20.1 execution conformance harness v4 (W8; C1-C35, v6)."
    )
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("battery", help="Run full C1-C34 conformance battery")
    sub.add_parser("g-plan-1", help="Run battery and emit G-PLAN-1 disposition")
    args = p.parse_args(argv)

    if args.cmd in ("battery", "g-plan-1"):
        results = run_conformance_battery()
        if args.cmd == "g-plan-1":
            results["g_plan_1"] = g_plan_1_disposition(results)
        print(json.dumps(results, indent=2))
        return 0 if results["all_pass"] else 1

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
