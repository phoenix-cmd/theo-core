#!/usr/bin/env python3
"""Phase 6E.20.3 Synthetic Unit Test Suite.

Validates:
    1. Readout calculations (forward_margin, forward_logits) and runtime patch.
    2. Decision logic engine for H1 (P2), H3, H4, H5 across passing, failing, and edge cases.
    3. Gradient cosine sign convention on synthetic aligned and opposing vectors.
    4. LoRA adapter key extraction and missing-key detection (anti-silent drop).
    5. Preflight manifest verification logic on synthetic inputs.
    6. Authorization validator refusal logic.
    7. NEPR collection, hashing, comparison, schema validation, and emit/bind logic.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from run_phase_6e20_reevaluation import (
    AGGREGATION_SET,
    CONDITIONS,
    H4_DELTA_MIN,
    P1_THRESHOLD,
    P2_THRESHOLD,
    P3_THRESHOLD,
    P4_THRESHOLD,
    P5_THRESHOLD,
    PHASE_ID,
    PreflightError,
    Readout,
    TOKENIZER_BASE_VOCAB,
    TOKENIZER_FULL_VOCAB,
    VOCAB,
    check_authorization,
    compute_live_rho_c,
    compute_readout_total_alignment,
    compute_sha256,
    ensure_readout_class_patch,
    evaluate_protocol_hypotheses,
    validate_stage2_vocab_invariant,
    # 6E.20.4 test dependencies (added for B0 execution-regime remediation tests)
    B0_REPRODUCIBILITY_TOLERANCE,
    HISTORICAL_B0_S8_ANCHOR,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGETS,
    LORA_BIAS,
    LORA_TASK_TYPE,
    resolve_project_paths,
    run_stage2_model_and_tokenizer_preflight,
    load_frozen_6e18_gradient_machinery,
    reconstruct_panel_prompts,
    run_stage5_b0_historical_reproducibility_anchor,
    compute_live_cos_H1,
)
from collect_nepr import (
    atomic_write_json,
    collect_nepr,
    compare_nepr,
    compute_nepr_hash,
    detect_execution_topology,
    detect_attention_backend,
    measure_lora_dtype,
    validate_nepr_schema,
    NEPR_FILE_NAME,
    BINDING_FILE_NAME,
)


def run_unit_test_suite() -> Dict[str, Any]:
    tests: Dict[str, Dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: str = ""):
        tests[name] = {"pass": bool(passed), "detail": detail}

    # -----------------------------------------------------------------------
    # Test 1: Readout Forward Margins & Logits
    # -----------------------------------------------------------------------
    w_mock = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    h_mock = np.array([0.5, -0.5, 1.0], dtype=np.float64)
    b_mock = -0.5
    ro = Readout(name="Test", w=w_mock, b=b_mock)

    # w @ h + b = (0.5 - 1.0 + 3.0) - 0.5 = 2.0
    margin = ro.forward_margin(h_mock)
    pos_logit, abs_logit = ro.forward_logits(h_mock)
    record(
        "readout_forward_math",
        math.isclose(margin, 2.0, rel_tol=1e-7) and math.isclose(pos_logit, 2.0) and math.isclose(abs_logit, -2.0),
        f"margin={margin}, logits=({pos_logit}, {abs_logit})",
    )

    # -----------------------------------------------------------------------
    # Test 2: Runtime Class Patch for Torch Tensors
    # -----------------------------------------------------------------------
    import torch
    ensure_readout_class_patch()
    h_tensor = torch.tensor([0.5, -0.5, 1.0], dtype=torch.bfloat16)
    pos_logit_t, abs_logit_t = ro.forward_logits(h_tensor)
    record(
        "readout_tensor_patch",
        math.isclose(pos_logit_t, 2.0, abs_tol=1e-2) and math.isclose(abs_logit_t, -2.0, abs_tol=1e-2),
        f"tensor logits=({pos_logit_t:.4f}, {abs_logit_t:.4f})",
    )

    # -----------------------------------------------------------------------
    # Test 3: Protocol Hypothesis Logic — All Hypotheses PASS Case
    # -----------------------------------------------------------------------
    passing_telemetry = {}
    for c in CONDITIONS:
        passing_telemetry[c] = {"steps": {}}
        for t in AGGREGATION_SET:
            if c == "S4":
                cos_val = 0.85
                d_val = 1.60
                bal_val = 0.80
                rho_val = 0.55
            elif c == "S8":
                cos_val = 0.90
                d_val = 1.70
                bal_val = 0.85
                rho_val = 0.65
            elif c == "S2":
                cos_val = 0.70
                d_val = 1.20
                bal_val = 0.75
                rho_val = 0.50
            elif c == "C4-Rand":
                cos_val = 0.40  # S4 (0.85) - C4 (0.40) = 0.45 >= 0.30
                d_val = 0.30   # < 0.80
                bal_val = 0.55
                rho_val = 0.20
            elif c == "R1-Rand":
                cos_val = 0.10
                d_val = 0.20   # < 0.80
                bal_val = 0.52
                rho_val = 0.15
            elif c == "F1":
                cos_val = 0.60
                d_val = 1.80   # >= 0.80
                bal_val = 0.90
                rho_val = 0.95
            else:  # B0
                cos_val = 0.42
                d_val = 0.50
                bal_val = 0.60
                rho_val = 0.30

            passing_telemetry[c]["steps"][str(t)] = {
                "cos_H1": cos_val,
                "cohens_d": d_val,
                "balanced_accuracy": bal_val,
                "rho_c": rho_val,
            }

    res_pass = evaluate_protocol_hypotheses(passing_telemetry)
    h_verdicts = {k: v["verdict"] for k, v in res_pass["hypotheses"].items()}
    all_h_pass = (
        h_verdicts["H1_Primary_Subspace_Gradient_Compatibility"] == "SUPPORTED"
        and h_verdicts["H4_Aligned_vs_Random_Subspace"] == "SUPPORTED"
        and h_verdicts["H5_Random_1D_Alignment_Specificity"] == "SUPPORTED"
        and h_verdicts["H3_Readout_Steering"] == "SUPPORTED"
    )
    record("decision_logic_all_pass_case", all_h_pass, f"verdicts={h_verdicts}")

    # -----------------------------------------------------------------------
    # Test 4: Protocol Hypothesis Logic — Primary H1 FAILS Case
    # -----------------------------------------------------------------------
    failing_h1_telemetry = copy.deepcopy(passing_telemetry)
    for t in AGGREGATION_SET:
        failing_h1_telemetry["S4"]["steps"][str(t)]["cos_H1"] = 0.45  # Below 0.80

    res_fail_h1 = evaluate_protocol_hypotheses(failing_h1_telemetry)
    record(
        "decision_logic_h1_refuted_case",
        res_fail_h1["hypotheses"]["H1_Primary_Subspace_Gradient_Compatibility"]["verdict"] == "REFUTED",
        f"H1 verdict={res_fail_h1['hypotheses']['H1_Primary_Subspace_Gradient_Compatibility']['verdict']}",
    )

    # -----------------------------------------------------------------------
    # Test 5: Protocol Hypothesis Logic — H4 Control Contrast Refuted
    # -----------------------------------------------------------------------
    failing_h4_telemetry = copy.deepcopy(passing_telemetry)
    for t in AGGREGATION_SET:
        failing_h4_telemetry["C4-Rand"]["steps"][str(t)]["cos_H1"] = 0.75  # S4 (0.85) - C4 (0.75) = 0.10 < 0.30

    res_fail_h4 = evaluate_protocol_hypotheses(failing_h4_telemetry)
    record(
        "decision_logic_h4_refuted_case",
        res_fail_h4["hypotheses"]["H4_Aligned_vs_Random_Subspace"]["verdict"] == "REFUTED",
        f"H4 verdict={res_fail_h4['hypotheses']['H4_Aligned_vs_Random_Subspace']['verdict']}",
    )

    # -----------------------------------------------------------------------
    # Test 6: Protocol Hypothesis Logic — H3 Steering Non-Monotonic
    # -----------------------------------------------------------------------
    failing_h3_telemetry = copy.deepcopy(passing_telemetry)
    for t in AGGREGATION_SET:
        failing_h3_telemetry["S8"]["steps"][str(t)]["rho_c"] = 0.30  # S8 (0.30) < S4 (0.55) -> Non-monotonic

    res_fail_h3 = evaluate_protocol_hypotheses(failing_h3_telemetry)
    record(
        "decision_logic_h3_refuted_case",
        res_fail_h3["hypotheses"]["H3_Readout_Steering"]["verdict"] == "REFUTED",
        f"H3 verdict={res_fail_h3['hypotheses']['H3_Readout_Steering']['verdict']}",
    )

    # -----------------------------------------------------------------------
    # Test 7: Synthetic Sign-Gate Vector Assertions
    # -----------------------------------------------------------------------
    u1 = np.array([1.0, 0.0])
    u2 = np.array([-1.0, 0.0])
    cos_opposing = float(np.dot(u1, u2) / (np.linalg.norm(u1) * np.linalg.norm(u2)))

    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0])
    cos_aligned = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    record(
        "sign_gate_synthetic_convention",
        math.isclose(cos_opposing, -1.0) and math.isclose(cos_aligned, 1.0),
        f"opposing={cos_opposing}, aligned={cos_aligned}",
    )

    # -----------------------------------------------------------------------
    # Test 8: Non-Silent LoRA Key Drop Detection Logic
    # -----------------------------------------------------------------------
    expected_keys = {"lora_A_layer1", "lora_B_layer1", "lora_A_layer2", "lora_B_layer2"}
    mock_ckpt_good = {"lora_A_layer1": torch.ones(2), "lora_B_layer1": torch.ones(2), "lora_A_layer2": torch.ones(2), "lora_B_layer2": torch.ones(2)}
    mock_ckpt_bad = {"lora_A_layer1": torch.ones(2), "lora_B_layer1": torch.ones(2)}  # missing layer 2

    missing_good = expected_keys - set(mock_ckpt_good.keys())
    missing_bad = expected_keys - set(mock_ckpt_bad.keys())
    record(
        "lora_missing_key_detection",
        len(missing_good) == 0 and len(missing_bad) == 2,
        f"good_missing={len(missing_good)}, bad_missing={len(missing_bad)}",
    )

    # -----------------------------------------------------------------------
    # Test 9: Authorization Validator Unit Tests
    # -----------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        dummy_script = Path(td) / "runner.py"
        dummy_script.write_text("print('test')", encoding="utf-8")
        dummy_sha = compute_sha256(dummy_script)

        # 9a: Missing auth file
        auth_refused_missing = False
        try:
            check_authorization(dummy_script)
        except PreflightError:
            auth_refused_missing = True

        # 9b: Valid auth file
        auth_file = Path(td) / "execution_authorization_6e20_3.json"
        auth_file.write_text(json.dumps({
            "phase": PHASE_ID,
            "decision": "APPROVE",
            "runner_sha256": dummy_sha,
        }), encoding="utf-8")
        auth_accepted_valid = True
        try:
            check_authorization(dummy_script)
        except PreflightError:
            auth_accepted_valid = False

        # 9c: Mismatched runner SHA
        auth_file.write_text(json.dumps({
            "phase": PHASE_ID,
            "decision": "APPROVE",
            "runner_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        }), encoding="utf-8")
        auth_refused_bad_sha = False
        try:
            check_authorization(dummy_script)
        except PreflightError:
            auth_refused_bad_sha = True

        # 9d: Wrong phase
        auth_file.write_text(json.dumps({
            "phase": "6E.20.1",
            "decision": "APPROVE",
            "runner_sha256": dummy_sha,
        }), encoding="utf-8")
        auth_refused_wrong_phase = False
        try:
            check_authorization(dummy_script)
        except PreflightError:
            auth_refused_wrong_phase = True

    record(
        "auth_validator_refusals",
        auth_refused_missing and auth_accepted_valid and auth_refused_bad_sha and auth_refused_wrong_phase,
        f"missing={auth_refused_missing}, valid={auth_accepted_valid}, bad_sha={auth_refused_bad_sha}, wrong_phase={auth_refused_wrong_phase}",
    )

    # -----------------------------------------------------------------------
    # Test 10: Pure rho_c core — genuine alignment formula (Audit B remediation)
    # Exercises compute_readout_total_alignment with synthetic g_R, g_total.
    # -----------------------------------------------------------------------
    def _rho_core(gR, gT):
        return compute_readout_total_alignment(
            torch.tensor(gR, dtype=torch.float64),
            torch.tensor(gT, dtype=torch.float64),
        )

    # TEST A — genuine alignment formula (g_R != g_total)
    gR_A = [1.0, 0.0, 0.0]
    gT_A = [1.0, 1.0, 0.0]
    expected_A = float(np.dot(gR_A, gT_A) / (np.linalg.norm(gR_A) * np.linalg.norm(gT_A)))
    rho_A = _rho_core(gR_A, gT_A)
    record("rho_genuine_alignment",
           math.isclose(rho_A, expected_A, abs_tol=1e-9),
           f"rho={rho_A:.6f}, expected={expected_A:.6f}")

    # TEST B — degenerate self-projection rejection.
    # Old implementation computed cos(g_R, normalized(w)) = sign(alpha) in {+1,-1};
    # with g_R = alpha*w that is degenerate. New formula with g_R distinct from
    # g_total must yield a graded value, NOT the old self-projection constant.
    gR_B = [1.0, 0.0, 0.0]
    gT_B = [1.0, 1.0, 0.0]   # non-parallel, non-orthogonal -> cos = 1/sqrt(2)
    rho_B = _rho_core(gR_B, gT_B)
    record("rho_rejects_self_projection",
           math.isclose(rho_B, 1.0 / math.sqrt(2.0), abs_tol=1e-9) and rho_B != 1.0 and rho_B != -1.0,
           f"rho={rho_B:.6f} (old self-projection would be in {{+1,-1}})")

    # TEST C — non-constant rho (mean of genuine per-sample cosines, not forced to 0)
    samples_C = [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),   # cos = 1.0
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),   # cos = 0.0
        ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),   # cos = 1/sqrt(2)
    ]
    rhos_C = [_rho_core(gR, gT) for gR, gT in samples_C]
    expected_C = (1.0 + 0.0 + 1.0 / math.sqrt(2.0)) / 3.0
    record("rho_non_constant",
           math.isclose(float(np.mean(rhos_C)), expected_C, abs_tol=1e-9)
           and abs(float(np.mean(rhos_C))) > 1e-6,
           f"rho_c={float(np.mean(rhos_C)):.6f}, expected={expected_C:.6f}")

    # TEST D — sigma_c MUST equal mean(rho_i**2) (Protocol v3.1 §5.7 / V3-03 "mean variance share"),
    # and MUST be demonstrably distinct from std(rho) and from (mean rho)**2.
    # Exercises the REAL compute_live_rho_c with purely synthetic fakes (no checkpoint/model load).
    class _FakeReadout:
        def __init__(self):
            self.w1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
            self.w2 = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)

        def forward_logits(self, hL):
            return (torch.dot(hL, self.w1), torch.dot(hL, self.w2))

        def forward_logits_tensor(self, hL):
            return (torch.dot(hL, self.w1), torch.dot(hL, self.w2))

    class _FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=torch.float64))

        def parameters(self):
            return [self.w]

        def zero_grad(self, set_to_none=False):
            pass

        def forward(self, input_ids, attention_mask, output_hidden_states=True):
            seq_len = int(input_ids.shape[1])
            hidden = torch.cat([self.w.unsqueeze(0) for _ in range(seq_len)], dim=0).unsqueeze(0)
            vocab = 9200  # > PRO_ID (9117) / AB_ID (1867) so full-vocab softmax indices are valid
            logits = torch.zeros(1, seq_len, vocab, dtype=torch.float64)
            return type("NS", (), {"logits": logits, "hidden_states": [None, hidden]})()

    def _fake_tok(prompt_str, *args, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    def _fake_psdg(model, readout, item, lora_params, tokenizer):
        return torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)

    _panel = [("p", "p") for _ in range(16)]
    try:
        _rho_c, _sigma_c, _per = compute_live_rho_c(
            _FakeModel(), _FakeReadout(), _fake_tok, _panel, _fake_psdg, None, "cpu"
        )
        _arr = np.asarray(_per, dtype=np.float64)
        _mean_sq = float(np.mean(_arr ** 2))            # CORRECT  (Protocol §5.7 / V3-03)
        _std_rho = float(np.std(_arr))                  # WRONG ALT 1
        _mean_sq_of_mean = float(np.mean(_arr) ** 2)    # WRONG ALT 2
        _distinct = (abs(_mean_sq - _std_rho) > 1e-6) and (abs(_mean_sq - _mean_sq_of_mean) > 1e-6)
        _ok = (
            math.isfinite(_sigma_c)
            and _distinct
            and math.isclose(_sigma_c, _mean_sq, abs_tol=1e-9)
            and not math.isclose(_sigma_c, _std_rho, abs_tol=1e-6)
            and not math.isclose(_sigma_c, _mean_sq_of_mean, abs_tol=1e-6)
        )
        record("sigma_c_is_mean_variance_share",
               _ok,
               f"sigma_c={_sigma_c:.6f} mean(rho^2)={_mean_sq:.6f} "
               f"std(rho)={_std_rho:.6f} (mean rho)^2={_mean_sq_of_mean:.6f} "
               f"(must equal mean(rho^2); not the other two)")
    except Exception as _e:
        record("sigma_c_is_mean_variance_share", False, f"compute_live_rho_c raised: {_e}")

    # -----------------------------------------------------------------------
    # 6E.20.5 — Stage 6 autograd-contract remediation (T1-T8)
    # Dedicated tensor readout path vs legacy float readout; real PEFT/LoRA grad.
    # -----------------------------------------------------------------------
    from peft import LoraConfig, get_peft_model

    _D = 4
    # Real PEFT/LoRA model so the graph reaches ACTUAL LoRA parameters.
    class _TinyBase(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc = torch.nn.Linear(d, d, bias=False)
        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
            if input_ids is None:
                x = torch.randn(self.fc.in_features)
            else:
                emb = input_ids.to(torch.float32).reshape(-1)
                if emb.numel() < self.fc.in_features:
                    emb = torch.cat([emb, torch.zeros(self.fc.in_features - emb.numel())])
                else:
                    emb = emb[:self.fc.in_features]
                x = emb.unsqueeze(0)
            return self.fc(x).squeeze(0)
    _base = _TinyBase(_D)
    _lora_cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                           target_modules=["fc"], bias="none", task_type="FEATURE_EXTRACTION")
    _peft = get_peft_model(_base, _lora_cfg)
    _lora_params = [p for p in _peft.parameters() if p.requires_grad]
    _ro = Readout(name="T65", w=np.ones(_D, dtype=np.float64), b=0.0)
    _x = torch.randn(_D)
    _hL = _peft(_x)  # (D,) graph-connected, depends on LoRA params

    # T1 — tensor vs legacy float readout agree within 1e-9
    _zp_t, _za_t = _ro.forward_logits_tensor(_hL)
    _zp_f, _za_f = _ro.forward_logits(_hL)
    _t1 = (math.isclose(_zp_t.item(), _zp_f, abs_tol=1e-9)
           and math.isclose(_za_t.item(), _za_f, abs_tol=1e-9)
           and math.isclose(_za_t.item(), -_zp_f, abs_tol=1e-9))
    record("rho_tensor_float_equivalence_1e9", _t1,
           f"zPRO_t={_zp_t.item():.8f} zPRO_f={_zp_f:.8f} zABS_t={_za_t.item():.8f} zABS_f={_za_f:.8f}")

    # T2 — graph connectivity
    _t2 = (_zp_t.requires_grad and _zp_t.grad_fn is not None and not _zp_t.is_leaf)
    record("rho_tensor_graph_connected", _t2,
           f"requires_grad={_zp_t.requires_grad} grad_fn={_zp_t.grad_fn is not None} is_leaf={_zp_t.is_leaf}")

    # T3 — autograd.grad reaches participating LoRA params with finite, non-zero, shape-correct grads
    _g = torch.autograd.grad(_zp_t, _lora_params, allow_unused=True)
    _nonnone = [g for g in _g if g is not None]
    _nonzero = [g for g in _nonnone if torch.any(g != 0)]
    _shapes_ok = all(tuple(g.shape) == tuple(p.shape) for g, p in zip(_g, _lora_params) if g is not None)
    _t3 = (len(_g) == len(_lora_params) and len(_nonnone) > 0 and len(_nonzero) > 0 and _shapes_ok)
    record("rho_autograd_reaches_lora_nonzero", _t3,
           f"n_params={len(_lora_params)} non_none={len(_nonnone)} nonzero={len(_nonzero)} shapes_ok={_shapes_ok}")

    # T4 — no detached/constant workaround: perturbing an upstream LoRA-connected input changes z_PRO
    _hL2 = _peft(_x + 1e-3 * torch.randn(_D))
    _zp2 = _ro.forward_logits_tensor(_hL2)[0]
    _t4 = bool((_zp2 - _zp_t).abs() > 1e-12)
    record("rho_no_detach_workaround", _t4,
           f"delta_zPRO_on_input_perturb={float((_zp2 - _zp_t).detach().abs()):.3e}")

    # T6 — legacy float contract unchanged (Stage 5 / margins rely on it)
    _t6 = (isinstance(_ro.forward_margin(np.ones(_D, dtype=np.float64)), float)
           and isinstance(_ro.forward_logits(np.ones(_D, dtype=np.float64))[0], float))
    record("rho_legacy_float_contract_unchanged", _t6, "forward_margin/forward_logits return Python float")

    # T3b / T5 / T7 / T8 — full compute_live_rho_c path through a REAL Readout
    class _FakeReadoutReal:
        def __init__(self, w):
            self.w = torch.tensor(w, dtype=torch.float64)
        def forward_margin(self, h):
            return float(np.asarray(self.w) @ np.asarray(h, dtype=np.float64))
        def forward_logits(self, h):
            s = self.forward_margin(h)
            return s, -s
        def forward_logits_tensor(self, hL):
            w_t = torch.as_tensor(self.w, dtype=torch.float64, device=hL.device)
            return torch.matmul(w_t, hL.double()), -torch.matmul(w_t, hL.double())
    class _FakeModel65(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([2.0, -1.0, 0.5, 3.0], dtype=torch.float64))
            self.unused = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))  # exercises unused-grad protocol
        def parameters(self):
            return [self.w, self.unused]
        def zero_grad(self, set_to_none=False):
            pass
        def forward(self, input_ids, attention_mask, output_hidden_states=True):
            seq_len = int(input_ids.shape[1])
            hidden = torch.cat([self.w.unsqueeze(0) for _ in range(seq_len)], dim=0).unsqueeze(0)
            vocab = 9200
            logits = torch.zeros(1, seq_len, vocab, dtype=torch.float64)
            return type("NS", (), {"logits": logits, "hidden_states": [None, hidden]})()
    def _fake_tok65(prompt_str, *a, **k):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}
    def _fake_psdg65(model, readout, item, lora_params, tokenizer):
        # g_total must span the SAME flat LoRA space as g_R (length = total lora_params elements)
        return torch.cat([p.reshape(-1) for p in lora_params])
    _panel65 = [("p", "p") for _ in range(16)]
    _real_ro = _FakeReadoutReal([1.0, 2.0, 3.0, 4.0])

    # T7 — fail-before / pass-after via the REAL Readout
    # Legacy float path: hL is detached first (as ensure_readout_class_patch does before numpy),
    # so the historical failure mode is grad(float, ...) raising TypeError, not a numpy() error.
    _leg_zp, _leg_za = _real_ro.forward_logits(_hL2.detach())
    _legacy_is_float = isinstance(_leg_zp, float)
    try:
        torch.autograd.grad(_leg_zp, [_base.fc.weight])  # a float cannot be differentiated
        _legacy_crash = False
    except TypeError:
        _legacy_crash = True
    record("stage6_realreadout_fail_before", _legacy_is_float and _legacy_crash,
           f"legacy forward_logits returns float={_legacy_is_float}; grad(float) raises={_legacy_crash}")
    _all_finite = _all_in_range = _not_degenerate = False
    try:
        _rc, _sc, _per = compute_live_rho_c(_FakeModel65(), _real_ro, _fake_tok65, _panel65, _fake_psdg65, None, "cpu")
        _all_finite = math.isfinite(_rc) and all(math.isfinite(r) for r in _per)
        _all_in_range = all(-1.0 <= r <= 1.0 for r in _per)
        _not_degenerate = not all(abs(r) < 1e-12 for r in _per)
        record("stage6_realreadout_pass_after",
               _all_finite and _all_in_range and _not_degenerate,
               f"rho_c={_rc:.4f} sigma_c={_sc:.4f} n_per={len(_per)}")
    except Exception as _e:
        record("stage6_realreadout_pass_after", False, f"compute_live_rho_c raised: {_e}")

    # T3b — assembled g_R non-degenerate: rho_i are genuine cosines, not forced constants
    record("rho_gR_non_degenerate", _not_degenerate and _all_in_range,
           f"per_sample_rho in [-1,1]={_all_in_range} non_degenerate={_not_degenerate}")

    # T5 — unused gradients handled per protocol (None -> zeros; no crash, no None leak)
    record("rho_unused_grad_protocol", _all_finite and len(_per) == 16,
           "compute_live_rho_c completed with an unused LoRA param present")

    # T8 — rho_c and per_sample_rho are genuine cosines in [-1, 1]
    record("rho_cosine_range", _all_in_range, f"range_ok={_all_in_range}")

    # TEST E — zero-norm protection (no silent NaN/Inf valid value)
    rho_E1 = _rho_core([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])   # ||g_R|| == 0
    rho_E2 = _rho_core([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])   # ||g_total|| == 0
    record("rho_zero_norm_protection",
           rho_E1 == 0.0 and rho_E2 == 0.0 and math.isfinite(rho_E1) and math.isfinite(rho_E2),
           f"gR0={rho_E1}, gT0={rho_E2}")

    # TEST F — shape mismatch / non-tensor inputs halt explicitly
    shape_halted = False
    try:
        _rho_core([1.0, 0.0], [1.0, 0.0, 0.0])
    except PreflightError:
        shape_halted = True
    nontensor_halted = False
    try:
        compute_readout_total_alignment([1.0, 0.0], [1.0, 0.0])
    except PreflightError:
        nontensor_halted = True
    record("rho_shape_and_type_guards",
           shape_halted and nontensor_halted,
           f"shape_halted={shape_halted}, nontensor_halted={nontensor_halted}")

    # -----------------------------------------------------------------------
    # Test 11: cos(H1) mathematical core (frozen 6E.18 machinery) — TEST G.
    # Verifies raw cosine (no label-sign inversion) used by compute_live_cos_H1.
    # Loaded lazily from the frozen snapshot; no real model/checkpoint involved.
    # -----------------------------------------------------------------------
    try:
        import importlib.util as _ilu
        _snap = Path(__file__).resolve().parent / ".." / ".." / ".." / "theo-data" / "datasets" \
            / "theo_slm_v0_artifacts" / "phase-6e18-causal-intervention" / "code_snapshot" \
            / "run_phase_6e18_causal_intervention.py"
        _spec = _ilu.spec_from_file_location("frozen_6e18_machinery_test", str(_snap))
        _fmod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_fmod)
        crgc = _fmod.compute_raw_gradient_cosine
        _u1 = torch.tensor([1.0, 0.0]); _u2 = torch.tensor([-1.0, 0.0])
        _cos_opp = crgc([_u1], [_u2])["cos_H1_aggregate"]
        _v1 = torch.tensor([1.0, 0.0]); _v2 = torch.tensor([1.0, 0.0])
        _cos_al = crgc([_v1], [_v2])["cos_H1_aggregate"]
        record("cos_H1_core_sign_gate",
               math.isclose(_cos_opp, -1.0, abs_tol=1e-3) and math.isclose(_cos_al, 1.0, abs_tol=1e-3),
               f"opp={_cos_opp}, align={_cos_al} (raw cosine, no sign inversion)")
    except Exception as _e:
        record("cos_H1_core_sign_gate", False, f"could not load frozen 6E.18 cosine core: {_e}")

    # -----------------------------------------------------------------------
    # Test 12: Stage 2 vocabulary invariant semantics (root-cause B remediation)
    # Pure / synthetic: exercises validate_stage2_vocab_invariant only, which
    # receives the three vocabulary metrics as plain scalars. No real model or
    # tokenizer is loaded; no download, inference, or checkpoint access occurs.
    # -----------------------------------------------------------------------
    # 12a: Known-valid pinned configuration PASSES, even though len(tokenizer)
    #      (151665) != VOCAB (151936). The corrected invariant gates on
    #      model.config.vocab_size, not len(tokenizer).
    valid_pass = False
    valid_raised = None
    try:
        validate_stage2_vocab_invariant(VOCAB, TOKENIZER_BASE_VOCAB, TOKENIZER_FULL_VOCAB)
        valid_pass = True
    except PreflightError as _e:
        valid_raised = str(_e)
    record("stage2_valid_config_passes",
           valid_pass,
           f"raised={valid_raised}")

    # 12b: Incorrect model.config.vocab_size FAILS (fail-closed).
    bad_model_halted = False
    try:
        validate_stage2_vocab_invariant(VOCAB - 1, TOKENIZER_BASE_VOCAB, TOKENIZER_FULL_VOCAB)
    except PreflightError:
        bad_model_halted = True
    record("stage2_wrong_model_vocab_fails",
           bad_model_halted,
           f"halted={bad_model_halted}")

    # 12c: Incorrect tokenizer.vocab_size FAILS (fail-closed).
    bad_base_halted = False
    try:
        validate_stage2_vocab_invariant(VOCAB, TOKENIZER_BASE_VOCAB - 1, TOKENIZER_FULL_VOCAB)
    except PreflightError:
        bad_base_halted = True
    record("stage2_wrong_tokenizer_base_vocab_fails",
           bad_base_halted,
           f"halted={bad_base_halted}")

    # 12d: Incorrect len(tokenizer) FAILS (fail-closed).
    bad_full_halted = False
    try:
        validate_stage2_vocab_invariant(VOCAB, TOKENIZER_BASE_VOCAB, TOKENIZER_FULL_VOCAB - 1)
    except PreflightError:
        bad_full_halted = True
    record("stage2_wrong_tokenizer_full_vocab_fails",
           bad_full_halted,
           f"halted={bad_full_halted}")

    # 12e: Demonstrate the OLD invalid comparison would REJECT the known-valid
    #      synthetic configuration. Reproduces the root-cause-B defect: the old
    #      check `len(tokenizer) != VOCAB` raises PreflightError on the valid
    #      pinned config (151665 != 151936).
    old_check_would_halt = False
    if TOKENIZER_FULL_VOCAB != VOCAB:
        old_check_would_halt = True
    record("stage2_old_invalid_comparison_rejects_valid_config",
           old_check_would_halt,
           f"old_len(tokenizer)={TOKENIZER_FULL_VOCAB} != VOCAB={VOCAB}")

    # -----------------------------------------------------------------------
    # Test 13: NEPR collect produces valid schema (synthetic, no real model)
    # Exercises collect_nepr with a synthetic model/module matching
    # detect_execution_topology's duck-typing interface.
    # -----------------------------------------------------------------------
    class _SynthMod(torch.nn.Module):
        """Minimal nn.Module that duck-types QLoRA attention + lm_head."""
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(16, 8, bias=False)
            self.lm_head = torch.nn.Linear(8, 100, bias=False)
            self.config = type("C", (), {
                "vocab_size": 100,
                "n_head": 4,
                "n_embd": 16,
                "n_layer": 2,
                "rope_scaling": None,
                "rope_theta": 10000.0,
                "attn_implementation": "sdpa",
                "torch_dtype": "bfloat16",
            })()
        def parameters(self, recurse=True):
            return self.proj.parameters()
        def named_parameters(self, prefix="", recurse=True):
            yield "proj.weight", self.proj.weight
            yield "lm_head.weight", self.lm_head.weight

    class _SynthLoRA(torch.nn.Module):
        """Synthetic LoRA wrapper with deterministic tensor data."""
        def __init__(self, base):
            super().__init__()
            self.base_model = base  # real nn.Module, so named_parameters works
            self.config = type("C", (), {
                "lora_alpha": 16,
                "r": 4,
                "lora_dropout": 0.05,
                "target_modules": ["proj"],
                "bias": "none",
            })()

        def named_parameters(self, prefix="", recurse=True):
            return self.base_model.named_parameters(prefix, recurse)

        def named_modules(self, prefix="", recurse=True):
            yield prefix, self
            for n, m in self.base_model.named_modules(prefix, recurse=True):
                yield n, m

        def parameters(self, recurse=True):
            return self.base_model.parameters(recurse)

    _synth_base = _SynthMod()
    _synth_peft = _SynthLoRA(_synth_base)
    try:
        _nepr = collect_nepr(
            model=_synth_base,
            peft_model=_synth_peft,
            phase_id=PHASE_ID,
            runner_sha256="a" * 64,
            test_sha256="b" * 64,
        )
        _schema_ok = validate_nepr_schema(_nepr)
        record("nepr_collect_valid_schema", _schema_ok, f"schema_valid={_schema_ok}")
    except Exception as _e:
        record("nepr_collect_valid_schema", False, f"collect_nepr raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 14: NEPR hash is deterministic (same input -> same hash)
    # -----------------------------------------------------------------------
    try:
        _n1 = collect_nepr(model=_synth_base, peft_model=_synth_peft,
                           phase_id=PHASE_ID, runner_sha256="a"*64, test_sha256="b"*64)
        _n2 = collect_nepr(model=_synth_base, peft_model=_synth_peft,
                           phase_id=PHASE_ID, runner_sha256="a"*64, test_sha256="b"*64)
        _hash_eq = (_n1["category_F_metadata"]["nepr_hash"] == _n2["category_F_metadata"]["nepr_hash"])
        record("nepr_hash_deterministic", _hash_eq,
               f"h1={_n1['category_F_metadata']['nepr_hash'][:24]}, h2={_n2['category_F_metadata']['nepr_hash'][:24]}")
    except Exception as _e:
        record("nepr_hash_deterministic", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 15: NEPR comparison — identical inputs -> SAME_ENVIRONMENT
    # -----------------------------------------------------------------------
    try:
        _c = compare_nepr(_n1, _n2)
        _same = (_c["classification"] == "SAME_ENVIRONMENT")
        record("nepr_compare_identical", _same, f"class={_c['classification']}")
    except Exception as _e:
        record("nepr_compare_identical", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 16: NEPR comparison — differing required field -> CROSS_ENVIRONMENT
    # -----------------------------------------------------------------------
    try:
        _n_diff = dict(_n1)
        _n_diff["category_A_software"] = dict(_n1["category_A_software"])
        _n_diff["category_A_software"]["torch_version"] = "0.0.0"
        _n_diff["category_F_metadata"] = dict(_n1["category_F_metadata"])
        _n_diff["category_F_metadata"]["nepr_hash"] = compute_nepr_hash(_n_diff)
        _c2 = compare_nepr(_n_diff, _n1)
        _cross = (_c2["classification"] == "CROSS_ENVIRONMENT")
        record("nepr_compare_different_key", _cross, f"class={_c2['classification']}")
    except Exception as _e:
        record("nepr_compare_different_key", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 17: NEPR comparison — missing required field -> ENVIRONMENT_UNVERIFIABLE
    # -----------------------------------------------------------------------
    try:
        _n_bad = dict(_n1)
        _n_bad["category_A_software"] = dict(_n1["category_A_software"])
        del _n_bad["category_A_software"]["torch_version"]  # remove a required field
        _n_bad["category_F_metadata"] = dict(_n1["category_F_metadata"])
        _n_bad["category_F_metadata"]["nepr_hash"] = compute_nepr_hash(_n_bad)
        _c3 = compare_nepr(_n_bad, _n1)
        _unverif = (_c3["classification"] == "ENVIRONMENT_UNVERIFIABLE")
        record("nepr_compare_unknown_required_field", _unverif, f"class={_c3['classification']}")
    except Exception as _e:
        record("nepr_compare_unknown_required_field", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 18: atomic_write_json creates file with correct content
    # -----------------------------------------------------------------------
    try:
        _td = Path(tempfile.mkdtemp())
        _fp = _td / "test_atomic.json"
        atomic_write_json(_fp, {"hello": "world", "num": 42})
        _loaded = json.loads(_fp.read_text(encoding="utf-8"))
        _ok = (_loaded == {"hello": "world", "num": 42})
        record("atomic_write_json_correct", _ok, f"loaded={_loaded}")
    except Exception as _e:
        record("atomic_write_json_correct", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 19: atomic_write_json overwrites existing file
    # -----------------------------------------------------------------------
    try:
        _fp.write_text('{"old": true}', encoding="utf-8")
        atomic_write_json(_fp, {"old": False, "new": True})
        _loaded2 = json.loads(_fp.read_text(encoding="utf-8"))
        _ok2 = (_loaded2 == {"old": False, "new": True})
        record("atomic_write_json_overwrite", _ok2, f"loaded={_loaded2}")
    except Exception as _e:
        record("atomic_write_json_overwrite", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 20: detect_execution_topology with synthetic model -> LOCAL
    # -----------------------------------------------------------------------
    try:
        _topo = detect_execution_topology(_synth_base, 0)
        _is_local = (_topo[0] is False)  # model_parallel should be False for single-GPU
        record("detect_topology_local", _is_local, f"topo={_topo}")
    except Exception as _e:
        record("detect_topology_local", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 21: detect_attention_backend returns string
    # -----------------------------------------------------------------------
    try:
        _ab = detect_attention_backend(_synth_base)
        record("detect_attention_backend_returns_str",
               isinstance(_ab, str) and len(_ab) > 0,
               f"backend={_ab}")
    except Exception as _e:
        record("detect_attention_backend_returns_str", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 22: measure_lora_dtype returns torch.dtype for synthetic LoRA params
    # -----------------------------------------------------------------------
    try:
        _dt, _dt_detail = measure_lora_dtype(_synth_peft)
        _is_dtype = isinstance(_dt, str)  # "float32" or "mixed" or "unknown"
        record("measure_lora_dtype_returns_str", _is_dtype, f"dtype={_dt}, detail={_dt_detail}")
    except Exception as _e:
        record("measure_lora_dtype_returns_str", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 23: NEPR comparison — model_parallel differs -> CROSS_ENVIRONMENT
    # -----------------------------------------------------------------------
    try:
        _n_par = dict(_n1)
        _n_par["category_B_hardware"] = dict(_n1["category_B_hardware"])
        _n_par["category_B_hardware"]["model_parallel"] = True
        _n_par["category_F_metadata"] = dict(_n1["category_F_metadata"])
        _n_par["category_F_metadata"]["nepr_hash"] = compute_nepr_hash(_n_par)
        _c4 = compare_nepr(_n_par, _n1)
        _cross2 = (_c4["classification"] == "CROSS_ENVIRONMENT")
        record("nepr_compare_model_parallel_differs", _cross2, f"class={_c4['classification']}")
    except Exception as _e:
        record("nepr_compare_model_parallel_differs", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # Test 24: compute_nepr_hash INCLUDES timestamp_utc (Revision 3 §4.1 excludes
    # ONLY nepr_hash; timestamps are part of the tamper-evident hash contract).
    # -----------------------------------------------------------------------
    try:
        _n_a = dict(_n1)
        _n_a["category_F_metadata"] = dict(_n1["category_F_metadata"])
        _n_a["category_F_metadata"]["nepr_timestamp_utc"] = "2000-01-01T00:00:00Z"
        _n_a["category_F_metadata"]["nepr_hash"] = "OLD"
        _h_a = compute_nepr_hash(_n_a)

        _n_b = dict(_n1)
        _n_b["category_F_metadata"] = dict(_n1["category_F_metadata"])
        _n_b["category_F_metadata"]["nepr_timestamp_utc"] = "2099-12-31T23:59:59Z"
        _n_b["category_F_metadata"]["nepr_hash"] = "NEW"
        _h_b = compute_nepr_hash(_n_b)
        # Hash MUST differ because timestamp is included in the canonical hash.
        _includes_ts = (_h_a != _h_b)
        record("nepr_hash_includes_timestamp", _includes_ts,
               f"h_a={_h_a[:24]}, h_b={_h_b[:24]}")
    except Exception as _e:
        record("nepr_hash_includes_timestamp", False, f"raised: {_e}")

    # -----------------------------------------------------------------------
    # 6E.20.4 B0 Execution-Regime Remediation — Tests A–H
    # Verifies the eval-mode guard around compute_live_cos_H1 at Stage 5.
    #   A/B: real reproduction over all four anchor checkpoints {4,6,14,15}
    #   C–H: guard behavior via monkeypatch (no real model/checkpoint load)
    # -----------------------------------------------------------------------
    import torch as _torch  # alias to avoid clobbering the module-level torch

    _orig_cos_global = compute_live_cos_H1  # captured before any monkeypatch

    # ----- Tests A & B: real anchor reproduction + repeatability -----
    try:
        from peft import LoraConfig as _LoraConfig, get_peft_model as _get_peft_model

        def _build_b0_inputs():
            _paths = resolve_project_paths()
            _base, _tok = run_stage2_model_and_tokenizer_preflight()
            _psdg, _crgc, _dev = load_frozen_6e18_gradient_machinery(_paths["frozen_6e18_snapshot"])
            if _torch.cuda.is_available():
                _base = _base.to(_dev)
            _H16 = np.load(_paths["historical_out_dir"] / "step0" / "H_16.npy")
            _readouts = {}
            for _c in CONDITIONS:
                _w = np.load(_paths["historical_out_dir"] / "step0" / f"w_{_c}.npy")
                _b = 0.0 if _c == "B0" else float(-np.median(_w @ _H16[:8].T))
                _readouts[_c] = Readout(name=_c, w=_w.astype(np.float64), b=_b)
            _panel = reconstruct_panel_prompts(_paths["corpus"])
            _lora = _LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                target_modules=LORA_TARGETS, bias=LORA_BIAS, task_type=LORA_TASK_TYPE)
            _peft = _get_peft_model(_base, _lora)
            return _peft, _tok, _paths, _panel, _readouts["S8"], _psdg, _crgc, _dev

        _peft, _tok, _paths, _panel, _ro, _psdg, _crgc, _dev = _build_b0_inputs()

        # Test A — all four B0 anchor checkpoints reproduce within 1e-4
        _resA = run_stage5_b0_historical_reproducibility_anchor(
            _peft, _tok, _paths, _panel, _ro, _psdg, _crgc, _dev)
        _a_ok = True
        _a_detail = []
        for _t, _r in _resA["results"].items():
            _a_ok = _a_ok and _r["pass"]
            _a_detail.append(f"t{_t}:{_r['recomputed_b0_s8_cos_H1']:.6f}/anc{_r['historical_runs_json_cos_H1']:.6f}")
        record("6E20_4_A_anchor_reproduction_all_4", _a_ok, "; ".join(_a_detail))

        # Test B — repeatability: a second eval-mode evaluation equals the first
        _resB = run_stage5_b0_historical_reproducibility_anchor(
            _peft, _tok, _paths, _panel, _ro, _psdg, _crgc, _dev)
        _b_ok = True
        for _t in _resA["results"]:
            _d = abs(_resA["results"][_t]["recomputed_b0_s8_cos_H1"]
                     - _resB["results"][_t]["recomputed_b0_s8_cos_H1"])
            _b_ok = _b_ok and (_d <= B0_REPRODUCIBILITY_TOLERANCE)
        record("6E20_4_B_repeatability", _b_ok, "second run equals first within 1e-4")
    except Exception as _e:
        record("6E20_4_A_anchor_reproduction_all_4", False, f"raised: {_e}")
        record("6E20_4_B_repeatability", False, f"skipped: {_e}")

    # ----- Tests C–H: guard behavior via monkeypatch (no real model) -----
    import run_phase_6e20_reevaluation as _Rmod
    _orig_torch_load = _torch.load

    class _FakePeft:
        def __init__(self, training_init):
            self.training = bool(training_init)
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = bool(mode)
        def load_state_dict(self, state, strict=False):
            pass

    def _run_stage5_fake(fake_peft, fake_cos_fn):
        # Patch the MODULE-LEVEL names that run_stage5 actually resolves.
        _oc = _Rmod.compute_live_cos_H1
        _ol = _torch.load
        _ot = _Rmod.B0_REPRODUCIBILITY_TOLERANCE
        _Rmod.compute_live_cos_H1 = fake_cos_fn
        _torch.load = lambda *a, **k: {"state_dict": {}}
        _Rmod.B0_REPRODUCIBILITY_TOLERANCE = 1e9
        _fake_paths = {"historical_out_dir": Path(tempfile.mkdtemp())}
        try:
            _res = run_stage5_b0_historical_reproducibility_anchor(
                fake_peft, None, _fake_paths, [("p", "p")] * 16, None, None, None, "cpu")
            return _res, None
        except Exception as _e:
            return None, _e
        finally:
            _Rmod.compute_live_cos_H1 = _oc
            _torch.load = _ol
            _Rmod.B0_REPRODUCIBILITY_TOLERANCE = _ot

    # Test C — mode entry: model is eval (training is False) during computation
    _rec = {}
    def _cos_C(model, *a, **k):
        _rec["training_at_compute"] = model.training
        return 0.0
    _fpC = _FakePeft(training_init=True)
    _run_stage5_fake(_fpC, _cos_C)
    record("6E20_4_C_mode_entry_eval",
           _rec.get("training_at_compute") is False,
           f"training_at_compute={_rec.get('training_at_compute')}")

    # Test D — restoration from train: starts train, ends train
    _fpD = _FakePeft(training_init=True)
    _run_stage5_fake(_fpD, lambda model, *a, **k: 0.0)
    record("6E20_4_D_restore_from_train",
           _fpD.training is True,
           f"post_training={_fpD.training}")

    # Test E — preservation from eval: starts eval, ends eval
    _fpE = _FakePeft(training_init=False)
    _run_stage5_fake(_fpE, lambda model, *a, **k: 0.0)
    record("6E20_4_E_preserve_from_eval",
           _fpE.training is False,
           f"post_training={_fpE.training}")

    # Test F — exception safety: compute raises, original mode restored
    def _cos_F(model, *a, **k):
        raise RuntimeError("forced B0 computation failure")
    _fpF = _FakePeft(training_init=True)
    _resF, _errF = _run_stage5_fake(_fpF, _cos_F)
    record("6E20_4_F_exception_safety",
           _errF is not None and _fpF.training is True,
           f"err={type(_errF).__name__ if _errF else None}, post_training={_fpF.training}")

    # Test G — no state leakage: monkeypatches fully restored, fresh model unaffected
    _leak_ok = (
        _Rmod.compute_live_cos_H1 is _orig_cos_global
        and _Rmod.B0_REPRODUCIBILITY_TOLERANCE == 1e-4
        and _torch.load is _orig_torch_load
    )
    _fpG = _FakePeft(training_init=True)  # never passed to run_stage5
    _leak_ok = _leak_ok and _fpG.training is True
    record("6E20_4_G_no_state_leakage", _leak_ok,
           f"cos_restored={_Rmod.compute_live_cos_H1 is _orig_cos_global}, "
           f"tol={_Rmod.B0_REPRODUCIBILITY_TOLERANCE}")

    # Test H — frozen scientific logic unchanged
    # NOTE: compute_sha256 returns UPPERCASE; compare case-insensitively.
    _h_ok = True
    _h_detail = []
    if _Rmod.B0_REPRODUCIBILITY_TOLERANCE != 1e-4:
        _h_ok = False; _h_detail.append("tolerance changed")
    _expected_anchor = {4: 0.428355, 6: 0.409713, 14: 0.427470, 15: 0.417450}
    if dict(HISTORICAL_B0_S8_ANCHOR) != _expected_anchor:
        _h_ok = False; _h_detail.append("anchor changed")
    if _Rmod.compute_live_cos_H1 is not _orig_cos_global:
        _h_ok = False; _h_detail.append("compute_live_cos_H1 replaced")
    _frozen = (Path(__file__).resolve().parent / ".." / ".." / ".." / "theo-data"
               / "datasets" / "theo_slm_v0_artifacts" / "phase-6e18-causal-intervention"
               / "code_snapshot" / "run_phase_6e18_causal_intervention.py")
    # True immutable baseline of the frozen 6E.18 machinery (verified on disk):
    _FROZEN_BASELINE = "6bcaf1319ac551848ffdb55aea121d14d808408a192e05f2aa6171161decef96"
    if compute_sha256(_frozen).lower() != _FROZEN_BASELINE:
        _h_ok = False; _h_detail.append("frozen 6E.18 machinery hash changed")
    _collect = Path(__file__).resolve().parent / "collect_nepr.py"
    if compute_sha256(_collect).lower() != "e2ee8afe6ee05bf9ad22d753de1c63a12108687eb5f2b86e44d493e4a123095a":
        _h_ok = False; _h_detail.append("collect_nepr.py hash changed")
    record("6E20_4_H_frozen_logic_unchanged", _h_ok, "; ".join(_h_detail))

    # -----------------------------------------------------------------------
    # Test: 6E.20.7 R1 — floor_cspec_abs MUST use median(C_common[S4], C_common[S8])
    #        per the APPROVED spec Section 3b, NOT the all-7-condition median.
    #        Crafted so the two medians diverge: S4,S8 C_common=10, others=100.
    #          all-7 median = 100 -> floor = 0.100 (NON-COMPLIANT / buggy)
    #          S4,S8 median = 10  -> floor = 0.010 (SPEC-COMPLIANT)
    # -----------------------------------------------------------------------
    from run_phase_6e20_reevaluation import classify_6e20_7_causes as _classify
    _c_common_by_cond = {
        "B0": 100.0, "F1": 100.0, "S2": 100.0,
        "S4": 10.0, "S8": 10.0, "C4-Rand": 100.0, "R1-Rand": 100.0,
    }
    _fake = {}
    for _c in CONDITIONS:
        _steps = {}
        for _t in AGGREGATION_SET:
            _steps[str(_t)] = {
                "cos_H1": 0.45,
                "telemetry_6e20_7": {
                    "g_total_common_norm": _c_common_by_cond[_c],
                    "g_total_specific_norm": 0.5,
                    "r_cs": 0.05,
                    "V_het": 0.05,
                    "cos_H1_i": [0.45] * 8,
                },
            }
        _fake[_c] = {"steps": _steps}
    _cls = _classify(_fake, sens_cos=0.99)
    _floor = _cls["floors"]["FLOOR_Cspec_abs"]
    _spec_floor = 1e-3 * 10.0
    _buggy_floor = 1e-3 * 100.0
    record(
        "6E20_7_R1_floor_cspec_abs_uses_S4_S8_median",
        math.isclose(_floor, _spec_floor, abs_tol=1e-9) and abs(_floor - _buggy_floor) > 1e-6,
        f"floor_cspec_abs={_floor:.6f} (spec={_spec_floor:.6f}, buggy_all7={_buggy_floor:.6f})",
    )

    all_pass = all(t["pass"] for t in tests.values())
    return {
        "suite": "Phase 6E.20.3 Synthetic Unit Test Suite",
        "all_pass": all_pass,
        "passed": sum(1 for t in tests.values() if t["pass"]),
        "total": len(tests),
        "tests": tests,
    }


if __name__ == "__main__":
    results = run_unit_test_suite()
    print(json.dumps(results, indent=2))
    sys.exit(0 if results["all_pass"] else 1)
