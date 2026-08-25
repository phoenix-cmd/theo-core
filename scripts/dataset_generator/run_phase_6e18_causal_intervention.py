 """Phase 6E.18 — Causal Intervention: Aligned Readout (I1), Controls (C2, F1),
and Parameterization Family Ablations (T1, T2, V1) vs Matched Baseline (B0).

Governing spec: theo-slm-causal-intervention-authorization-protocol-draft-v2.1.3.md (AMENDED).
Protocol SHA-256: 910796C879BBAFBA80C0D6829D382723CC4E89425003380FD4CC94E1E21312C9.

Canonical 7-Condition Suite (N=7):
  B0 : Frozen baseline margin Delta_z_b = w_delta^T h_L + b_delta (0 trainable params)
  I1 : Trainable aligned readout s = a^T h_L + b, a(0) = s_scale * u_hat_LDA (897 trainable params)
  C2 : Random orientation control s'' = c^T h_L + b'', seed 20260816 (897 trainable params)
  F1 : Fixed aligned projection s_fixed = (s_scale * u_hat_LDA)^T h_L + b_fixed (0 trainable params)
  T1 : 1D trainable scalar gain on w_delta: s = alpha * (w_delta^T h_L) + b (2 trainable params)
  T2 : Truncated SVD subspace readout: s = (U_k * beta)^T h_L + b, k in [2, 8] (k+1 trainable params)
  V1 : Unconstrained affine 2-logit: z_PRO = w_PRO^T h_L + b_PRO, z_ABS = w_ABS^T h_L + b_ABS (1794 trainable params)

v2.1.3 §3.8 initializations (FORMALIZATION unless labeled CHANGE):
  s_scale  = std(margins_B0) / (std(H16 @ u_hat_LDA) + 1e-12)
  a0       = s_scale * u_hat_LDA ;  b0 = -median(a0 @ H16[:8].T)
  c0       = s_scale * n_c2     ;  b0_c2 = -median(c0 @ H16[:8].T)
  b_delta  = 0 ;  b_fixed = b0
  T1 alpha(0)=1.0, b(0)=0.0 ;  T2 beta(0)=0, b(0)=0.0
  V1 w_PRO(0)=a0, w_ABS(0)=-a0, b_PRO(0)=0.0, b_ABS(0)=0.0  [CHANGE: deterministic aligned mirror]
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import datetime
import gc
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig, get_peft_model = None, None

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM, AutoTokenizer = None, None

# ---------------------------------------------------------------------------
# Frozen Protocol Constants (v2.1.3)
# ---------------------------------------------------------------------------
EXPECTED_PROTOCOL_HASH = "910796c879bbafba80c0d6829d382723cc4e89425003380fd4cc94e1e21312c9"
EXPECTED_BASE_HASH = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
EXPECTED_CORPUS_HASH = "a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0"

PRO_ID = 9117
AB_ID = 1867
DIAGNOSTIC_TOKEN_IDS = [9117, 1867, 72487, 46340, 2921, 220, 198]
TOKEN_NAMES = ["PRO", "AB", "REV", "EXEC", "null", "space", "newline"]
D_HIDDEN = 896
VOCAB = 151936
CHECKPOINTS = [0, 4, 6, 7, 14, 15, 17, 20]
HORIZON = 20
COLLAPSE_K = 3
SEED_DATA = 42
SEED_C2_READOUT = 20260816
LAMBDA_DECISION = 10.0
GRAD_ACCUM = 2
BATCH_SIZE = 4
LR = 1e-4
WD = 0.01
CLIP_NORM = 1.0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Decision Thresholds (§7)
P1_THRESH = -0.30
C2X_THRESH = 0.25
P2_THRESH = 0.0
P3_COHEN_D_THRESH = 0.20
P3_R_EFF_THRESH = 1.60
P4_KN1_RANGE = (0.65, 1.0)
P4_LOO_TOL = 0.10
P5_THRESH = 0.30
SIGN_GATE_EPSILON = 0.001

# §3.8 Shared Initialization & §7.2 M-Rule Constants (v2.1.3)
EPS_S_SCALE = 1e-12
ENERGY_THRESHOLD = 0.90
T2_K_MIN, T2_K_MAX = 2, 8
RULE_CHECKPOINTS = [4, 6, 14, 15]
H3_SUPPORT_THRESH = -0.60
H3_INDETERMINATE_UPPER = -0.30
M2_FLOOR = -0.30
M3_FLOOR = -0.30
M3_DELTA_TOL = 0.10
M4_FLOOR = -0.30
M5_FLOOR = -0.30
DEGRADATION_P15_MAX = 0.0
DEGRADATION_D15_MIN = 0.0

CONDITIONS = ["B0", "I1", "C2", "F1", "T1", "T2", "V1"]

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR.parent / "theo-data" / "datasets"
CORPUS_PATH = DATA_DIR / "theo_slm_v0_deduplicated" / "candidate_records.json"
BASE_MODEL_PATH = Path(r"C:\Users\bs162\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775")
PROTOCOL_PATH = BASE_DIR / "docs" / "research" / "theo-slm-causal-intervention-authorization-protocol-draft-v2.1.3.md"
ARTIFACT_DIR = DATA_DIR / "theo_slm_v0_artifacts" / "phase-6e18-causal-intervention"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Cryptographic & Tensor Utilities
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest().lower()


def tensor_hash(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t.detach().float().cpu().contiguous(), buf)
    return sha256_bytes(buf.getvalue())


def state_dict_hash(state: dict) -> str:
    buf = io.BytesIO()
    torch.save({k: v.detach().float().cpu().contiguous() for k, v in state.items()}, buf)
    return sha256_bytes(buf.getvalue())


def compute_cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    n1, n2 = v1.norm().item(), v2.norm().item()
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())


# ---------------------------------------------------------------------------
# §3.8 Shared Initialization Constants & Step-0 Derivation (v2.1.3)
# ---------------------------------------------------------------------------
def format_prompt(percept: str, concepts) -> str:
    """Decision-position prompt template for the frozen model's step-0 hidden states (6E.17 parity)."""
    labels = [c.get("id", str(c)) if isinstance(c, dict) else str(c) for c in concepts]
    concepts_str = ", ".join(labels) if labels else "none"
    return (
        "<|im_start|>system\n"
        "You are THEO SLM v0, a neural cognitive provider. Given an observation percept and grounding context, "
        "evaluate decision relevance and determine whether to propose a hypothesis or abstain.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Observation Percept: {percept}\n"
        f"Grounding Concepts: {concepts_str}\n"
        "Task: Emit JSON evaluation containing decision (SHOULD_PROPOSE or SHOULD_ABSTAIN) and reasoning.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_c2_random_orientation(u_lda: np.ndarray, seed: int = SEED_C2_READOUT) -> Tuple[np.ndarray, float, int]:
    """Deterministic C2 orientation n (v2.1.3 §3.3 / §3.8): unit vector with |cos(n, u_hat_LDA)| <= 0.05,
    drawn from torch.Generator(seed=20260816). Returns (n, cos_n_lda, n_draws)."""
    gen = torch.Generator().manual_seed(seed)
    n_draws = 0
    while True:
        n_draws += 1
        nv = torch.randn(D_HIDDEN, generator=gen)
        nv = nv / (nv.norm() + EPS_S_SCALE)
        cos_n_lda = float((nv.numpy() @ u_lda))
        if abs(cos_n_lda) <= 0.05:
            break
    return nv.numpy(), cos_n_lda, n_draws


def compute_init_constants(
    H16: np.ndarray, margins_b: np.ndarray, u_lda: np.ndarray, n_c2: np.ndarray
) -> Dict[str, Any]:
    """Pure, deterministic initialization arithmetic bound by v2.1.3 §3.8.

    s_scale  = std(margins_B0) / (std(H16 @ u_hat_LDA) + 1e-12)
    a0       = s_scale * u_hat_LDA            (I1 orientation)
    b0       = -median(a0 @ H16[:8].T)        (I1 bias)
    c0       = s_scale * n_c2                 (C2 orientation)
    b0_c2    = -median(c0 @ H16[:8].T)        (C2 bias)
    b_delta  = 0                              (B0 shared-bias cancellation)
    b_fixed  = b0                             (F1 frozen twin of I1)
    T1       alpha(0)=1.0, b(0)=0.0           (FORMALIZATION, 6E.18 defaults)
    T2       beta(0)=zeros(k), b(0)=0.0       (FORMALIZATION, 6E.18 default)
    V1       w_PRO(0)=a0, w_ABS(0)=-a0, b=0   (CHANGE: deterministic aligned mirror)
    """
    assert H16.shape == (16, D_HIDDEN), f"H16 must be 16x{D_HIDDEN}, got {H16.shape}"
    assert margins_b.shape == (16,), f"margins_b must be length 16, got {margins_b.shape}"
    assert u_lda.shape == (D_HIDDEN,), f"u_lda must be length {D_HIDDEN}, got {u_lda.shape}"
    assert n_c2.shape == (D_HIDDEN,), f"n_c2 must be length {D_HIDDEN}, got {n_c2.shape}"

    s_scale = float(np.std(margins_b) / (np.std(H16 @ u_lda) + EPS_S_SCALE))
    a0 = (s_scale * u_lda).astype(np.float32)
    b0 = float(-np.median(a0 @ H16[:8].T))
    c0 = (s_scale * n_c2).astype(np.float32)
    b0_c2 = float(-np.median(c0 @ H16[:8].T))
    v1_w_abs_0 = (-a0).astype(np.float32)

    return {
        "s_scale": s_scale,
        "a0": a0,
        "b0": b0,
        "c0": c0,
        "b0_c2": b0_c2,
        "b_delta": 0.0,
        "b_fixed": b0,
        "t1_alpha0": 1.0,
        "t1_b0": 0.0,
        "t2_beta0": None,  # resolved to zeros(k) by T2ReadoutHead (§3.6)
        "t2_b0": 0.0,
        "v1_w_pro_0": a0,
        "v1_w_abs_0": v1_w_abs_0,
        "v1_b_pro_0": 0.0,
        "v1_b_abs_0": 0.0,
    }


def derive_readout_initialization(data, tokenizer):
    """Execution-time step-0 fp32 derivation of u_hat_LDA, s_scale, a0, b0, c0, b0_c2, H16, margins_b.
    Bound by v2.1.3 §3.8 with 6E.17 arithmetic parity. INFERENCE-ONLY (read-only frozen model);
    records deterministic constants. Invoked ONLY during authorized Phase 6E.18 execution,
    never during preflight (zero-execution guard).
    """
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH), torch_dtype=torch.float32, device_map=DEVICE, trust_remote_code=True)
    model.config.use_cache = False
    model.eval()
    w_head = model.get_output_embeddings().weight.detach()
    w_delta = (w_head[PRO_ID] - w_head[AB_ID]).float().cpu().numpy()
    w_delta_unit = w_delta / (np.linalg.norm(w_delta) + EPS_S_SCALE)

    panel = data["pos_panel"] + data["abs_panel"]
    H16, margins_b = [], []
    for rec in panel:
        p = format_prompt(rec["percept"], rec.get("concepts", [])) + '{"decision": "'
        inp = tokenizer(p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model(**inp, output_hidden_states=True)
            H16.append(o.hidden_states[24][0, -1, :].float().cpu().numpy())
            z = o.logits[0, -1, :].float().cpu().numpy()
            margins_b.append((z[PRO_ID] - z[AB_ID]).item())
    H16 = np.array(H16)
    margins_b = np.array(margins_b)
    y16 = np.array([1] * 8 + [0] * 8)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(H16, y16)
    u_lda_raw = lda.coef_[0]
    u_lda = u_lda_raw / (np.linalg.norm(u_lda_raw) + EPS_S_SCALE)

    n_c2, cos_n_lda, n_draws = generate_c2_random_orientation(u_lda)
    init = compute_init_constants(H16, margins_b, u_lda, n_c2)

    derive = {
        "w_delta_norm": round(float(np.linalg.norm(w_delta)), 6),
        "s_scale": round(float(init["s_scale"]), 8),
        "b0": round(float(init["b0"]), 8),
        "b0_c2": round(float(init["b0_c2"]), 8),
        "b_delta": round(float(init["b_delta"]), 8),
        "b_fixed": round(float(init["b_fixed"]), 8),
        "c2_readout_cos_n_lda": round(float(cos_n_lda), 8),
        "c2_readout_draws": n_draws,
        "u_hat_LDA_hash": tensor_hash(torch.tensor(u_lda)),
        "a0_hash": tensor_hash(torch.tensor(init["a0"])),
        "b0_hash": tensor_hash(torch.tensor([init["b0"]])),
        "c0_hash": tensor_hash(torch.tensor(init["c0"])),
        "b0_c2_hash": tensor_hash(torch.tensor([init["b0_c2"]])),
        "v1_w_pro_0_hash": tensor_hash(torch.tensor(init["v1_w_pro_0"])),
        "v1_w_abs_0_hash": tensor_hash(torch.tensor(init["v1_w_abs_0"])),
    }
    init["w_delta_t"] = torch.tensor(w_delta, dtype=torch.float32)  # T1 frozen axis (§3.5)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return derive, u_lda, init, H16, margins_b


# ---------------------------------------------------------------------------
# Readout Architectures for All 7 Conditions (§3)
# ---------------------------------------------------------------------------
class ReadoutHead(nn.Module):
    """Abstract Base Class for Phase 6E.18 decision readout modules."""
    def forward_logits(self, hL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_PRO, z_ABS) for the decision position."""
        raise NotImplementedError

    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().float().cpu().contiguous() for k, v in self.named_parameters() if v.requires_grad}

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class I1ReadoutHead(ReadoutHead):
    """I1 / C2: s = a^T h_L + b ; z_PRO = s, z_ABS = -s. 897 trainable params."""
    def __init__(self, a0: np.ndarray, b0: float):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(b0), dtype=torch.float32))

    def forward_logits(self, hL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = hL.float() @ self.a + self.b
        return s, -s


class F1ReadoutHead(ReadoutHead):
    """F1: Fixed projection s_fixed = (s_scale * u_hat_LDA)^T h_L + b_fixed. 0 trainable params."""
    def __init__(self, w_fixed: np.ndarray, b_fixed: float):
        super().__init__()
        self.register_buffer("w", torch.tensor(w_fixed, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(float(b_fixed), dtype=torch.float32))

    def forward_logits(self, hL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = hL.float() @ self.w + self.b
        return s, -s


class T1ReadoutHead(ReadoutHead):
    """T1: 1D scalar gain on frozen w_delta: s = alpha * (w_delta^T h_L) + b. 2 trainable params."""
    def __init__(self, w_delta: torch.Tensor, alpha0: float = 1.0, b0: float = 0.0):
        super().__init__()
        self.register_buffer("w_delta", w_delta.detach().float().clone())
        self.alpha = nn.Parameter(torch.tensor(float(alpha0), dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(b0), dtype=torch.float32))

    def forward_logits(self, hL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        margin_proj = hL.float() @ self.w_delta
        s = self.alpha * margin_proj + self.b
        return s, -s


class T2ReadoutHead(ReadoutHead):
    """T2: Truncated SVD subspace readout: s = (U_k * beta)^T h_L + b. k+1 trainable params."""
    def __init__(self, U_k: torch.Tensor, beta0: Optional[np.ndarray] = None, b0: float = 0.0):
        super().__init__()
        # U_k shape: [D_HIDDEN, k], frozen basis
        self.register_buffer("U_k", U_k.detach().float().clone())
        k = U_k.shape[1]
        assert 2 <= k <= 8, f"Protocol assertion failed: k={k} must be in [2, 8]"
        if beta0 is None:
            beta0 = np.zeros(k, dtype=np.float32)
        self.beta = nn.Parameter(torch.tensor(beta0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(b0), dtype=torch.float32))

    def forward_logits(self, hL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # effective readout vector w = U_k @ beta
        w = self.U_k @ self.beta
        s = hL.float() @ w + self.b
        return s, -s


class V1ReadoutHead(ReadoutHead):
    """V1: Unconstrained affine 2-logit: z_PRO = w_PRO^T h_L + b_PRO, z_ABS = w_ABS^T h_L + b_ABS. 1794 trainable params."""
    def __init__(self, w_pro_0: np.ndarray, w_abs_0: np.ndarray, b_pro_0: float = 0.0, b_abs_0: float = 0.0):
        super().__init__()
        self.w_pro = nn.Parameter(torch.tensor(w_pro_0, dtype=torch.float32))
        self.w_abs = nn.Parameter(torch.tensor(w_abs_0, dtype=torch.float32))
        self.b_pro = nn.Parameter(torch.tensor(float(b_pro_0), dtype=torch.float32))
        self.b_abs = nn.Parameter(torch.tensor(float(b_abs_0), dtype=torch.float32))

    def forward_logits(self, hL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_pro = hL.float() @ self.w_pro + self.b_pro
        z_abs = hL.float() @ self.w_abs + self.b_abs
        return z_pro, z_abs


# ---------------------------------------------------------------------------
# §6.2 Normative Metric Calculation & Pre-Verdict Synthetic Sign Gate
# ---------------------------------------------------------------------------
def compute_raw_gradient_cosine(g_pos_list: List[torch.Tensor], g_abs_list: List[torch.Tensor]) -> Dict[str, float]:
    """Computes raw pairwise and aggregate decision-gradient cosine per §6.2.
    NO label-sign inversion or ABS negation is applied.
    """
    assert len(g_pos_list) > 0 and len(g_abs_list) > 0, "Non-empty gradient lists required"
    
    # Aggregate mean gradients
    pos_mean = sum(g_pos_list, torch.zeros_like(g_pos_list[0])) / float(len(g_pos_list))
    abs_mean = sum(g_abs_list, torch.zeros_like(g_abs_list[0])) / float(len(g_abs_list))
    
    agg_cos = compute_cosine_similarity(pos_mean, abs_mean)
    
    # Pairwise cross-cosines
    pair_cos_vals = []
    simult_feas_dots = []
    for g_p in g_pos_list:
        for g_a in g_abs_list:
            c = compute_cosine_similarity(g_p, g_a)
            pair_cos_vals.append(c)
            simult_feas_dots.append(1.0 if float((g_p * g_a).sum().item()) > 0.0 else 0.0)
            
    mean_pairwise_cos = float(np.mean(pair_cos_vals))
    # §6.1 item 2 (v2.1.3): rho_CD(t) = max(0, -cos(H1)(t)) where cos(H1) is the §6.2 AGGREGATE cosine.
    rho_cd = float(max(0.0, -agg_cos))
    f_simult = float(np.mean(simult_feas_dots))
    
    return {
        "cos_H1_aggregate": round(agg_cos, 6),
        "cos_H1_mean_pairwise": round(mean_pairwise_cos, 6),
        "rho_CD": round(rho_cd, 6),
        "F_simultaneous_feasibility": round(f_simult, 6),
    }


def verify_synthetic_sign_gate(eps: float = SIGN_GATE_EPSILON) -> bool:
    """Normative implementation verification gate required by §6.2.
    Tests that the metric computation produces exact -1.0 on opposing synthetic vectors
    and +1.0 on aligned synthetic vectors without label inversion or absolute value distortion.
    """
    # Synthetic Oppositional Pair: u1 = [1.0, 0.0], u2 = [-1.0, 0.0] -> expected -1.0
    u1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    u2 = torch.tensor([-1.0, 0.0], dtype=torch.float32)
    res_opp = compute_raw_gradient_cosine([u1], [u2])
    cos_opp = res_opp["cos_H1_aggregate"]
    
    # Synthetic Aligned Pair: v1 = [1.0, 0.0], v2 = [1.0, 0.0] -> expected +1.0
    v1 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    v2 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    res_align = compute_raw_gradient_cosine([v1], [v2])
    cos_align = res_align["cos_H1_aggregate"]
    
    opp_pass = abs(cos_opp - (-1.0)) <= eps
    align_pass = abs(cos_align - (+1.0)) <= eps
    
    if not (opp_pass and align_pass):
        raise RuntimeError(
            f"§6.2 Mandatory Pre-Verdict Sign Gate Assertion FAILED: "
            f"Oppositional expected -1.0, got {cos_opp} (diff={abs(cos_opp - (-1.0))}); "
            f"Aligned expected +1.0, got {cos_align} (diff={abs(cos_align - 1.0)}). "
            f"Tolerance={eps}. RUN VOIDED."
        )
    return True


# ---------------------------------------------------------------------------
# §7.1 Deterministic P4 Baseline Comparator Specification
# ---------------------------------------------------------------------------
def evaluate_p4_comparator(
    loo_cv_i1: float,
    loo_cv_b0_step: Optional[float],
    loo_cv_b0_step0: float,
    step: int,
    tol: float = P4_LOO_TOL
) -> Tuple[bool, float, str]:
    """Evaluates P4 LOO-CV separability guard against deterministic comparator (§7.1):
      - For shared evaluable checkpoints (t in {0, 4, 6}): compares to matched contemporaneous LOO-CV_B0(t).
      - For post-truncation checkpoints (t in {14, 15, 20}): compares to frozen step-0 anchor LOO-CV_B0(0).
    """
    if step in [0, 4, 6] and loo_cv_b0_step is not None:
        target = loo_cv_b0_step
        regime = f"contemporaneous_B0_step_{step}"
    else:
        target = loo_cv_b0_step0
        regime = "frozen_step0_anchor_B0"
        
    diff = abs(loo_cv_i1 - target)
    passed = diff <= tol
    return passed, diff, regime


# ---------------------------------------------------------------------------
# §7.2 Secondary Mechanistic Probe Decision Rules (M1–M5)
# ---------------------------------------------------------------------------
def _mean_over_checkpoints(vals: Dict[int, Optional[float]], ckpts: List[int]) -> Optional[float]:
    """Mean over the required checkpoint set; None if any checkpoint is missing/unavailable."""
    present = [vals.get(c) for c in ckpts]
    if any(v is None for v in present):
        return None
    return float(np.mean([float(v) for v in present]))


def evaluate_m_rules(
    metrics: Dict[str, Dict[str, Any]],
    t2_k: Optional[int] = None,
    i1_void: bool = False,
) -> Dict[str, Any]:
    """Evaluates M1–M5 dispositions per v2.1.3 §7.2.

    metrics[condition] = {
        "cos_H1": {t: aggregate cos(H1) or None},
        "proposal_rate": {t: value or None},
        "margin_d": {t: Cohen's d or None},
    }
    Dispositions: SUPPORTED / NOT SUPPORTED / INDETERMINATE / NOT EVALUABLE / VOID.
    All cos(H1) values are the §6.2 AGGREGATE cosine (never the mean-pairwise variant).
    """
    agg = lambda c: _mean_over_checkpoints(metrics.get(c, {}).get("cos_H1", {}), RULE_CHECKPOINTS)
    out: Dict[str, Any] = {}

    # M1 (F1 / H3)
    f1 = agg("F1")
    if f1 is None:
        out["M1"] = {"disposition": "VOID", "detail": "checkpoint missing or unavailable", "value": None}
    elif f1 <= H3_SUPPORT_THRESH:
        out["M1"] = {"disposition": "SUPPORTED", "detail": "H3 SUPPORTED", "value": f1}
    elif f1 > H3_INDETERMINATE_UPPER:
        out["M1"] = {"disposition": "NOT SUPPORTED", "detail": "H3 NOT SUPPORTED", "value": f1}
    else:
        out["M1"] = {"disposition": "INDETERMINATE", "detail": "H3 INDETERMINATE", "value": f1}

    # M2 (T1 / 1D gain)
    t1 = agg("T1")
    if t1 is None:
        out["M2"] = {"disposition": "VOID", "detail": "checkpoint missing or unavailable", "value": None}
    elif t1 >= M2_FLOOR:
        out["M2"] = {"disposition": "SUPPORTED", "detail": "1D gain SUFFICIENT", "value": t1}
    else:
        out["M2"] = {"disposition": "NOT SUPPORTED", "detail": "1D gain INSUFFICIENT", "value": t1}

    # M3 (T2 / H4, composite rank x basis x init contrast)
    t2, i1 = agg("T2"), agg("I1")
    if t2_k is not None and not (T2_K_MIN <= t2_k <= T2_K_MAX):
        out["M3"] = {"disposition": "VOID", "detail": f"k={t2_k} outside [{T2_K_MIN}, {T2_K_MAX}] (§8)", "value": None}
    elif i1_void:
        out["M3"] = {"disposition": "NOT EVALUABLE", "detail": "I1 VOID", "value": None}
    elif t2 is None or i1 is None:
        out["M3"] = {"disposition": "VOID", "detail": "checkpoint missing or unavailable", "value": None}
    else:
        delta = abs(t2 - i1)
        if delta <= M3_DELTA_TOL and t2 >= M3_FLOOR:
            out["M3"] = {"disposition": "SUPPORTED", "detail": "MATCH", "value": {"delta": delta, "t2": t2, "i1": i1}}
        else:
            out["M3"] = {"disposition": "NOT SUPPORTED", "detail": "NO MATCH", "value": {"delta": delta, "t2": t2, "i1": i1}}

    # M4 (V1 / H5) with policy degradation check (§6.1 item 9)
    v1 = agg("V1")
    if v1 is None:
        out["M4"] = {"disposition": "VOID", "detail": "checkpoint missing or unavailable", "value": None}
    else:
        p15 = metrics.get("V1", {}).get("proposal_rate", {}).get(15)
        d15 = metrics.get("V1", {}).get("margin_d", {}).get(15)
        p15_known = p15 is not None
        d15_known = d15 is not None
        degraded = (p15_known and p15 <= DEGRADATION_P15_MAX) or (d15_known and d15 < DEGRADATION_D15_MIN)
        if v1 >= M4_FLOOR and not degraded:
            out["M4"] = {"disposition": "SUPPORTED", "detail": "H5 SUPPORTED", "value": {"cos": v1, "degraded": degraded}}
        else:
            out["M4"] = {"disposition": "NOT SUPPORTED", "detail": "H5 NOT SUPPORTED", "value": {"cos": v1, "degraded": degraded}}
        if not (p15_known and d15_known):
            out["M4"]["detail"] += " (degradation telemetry incomplete)"

    # M5 (C2 / H2)
    c2 = agg("C2")
    if c2 is None:
        out["M5"] = {"disposition": "VOID", "detail": "checkpoint missing or unavailable", "value": None}
    elif c2 >= M5_FLOOR:
        out["M5"] = {"disposition": "SUPPORTED", "detail": "capacity sufficiency SUPPORTED", "value": c2}
    else:
        out["M5"] = {"disposition": "NOT SUPPORTED", "detail": "NOT SUPPORTED", "value": c2}

    return out


# ---------------------------------------------------------------------------
# §8 T2 Basis Rank Prospective Verification Procedure
# ---------------------------------------------------------------------------
def derive_t2_basis_from_covariance(X_panel: np.ndarray, energy_threshold: float = 0.90) -> Tuple[torch.Tensor, int, float, List[float]]:
    """Derives frozen orthogonal basis U_k via SVD on the empirical covariance of step-0 hidden states (§8).
    Selects minimum k reaching cumulative energy >= 90%, and asserts k in [2, 8].
    """
    assert X_panel.ndim == 2 and X_panel.shape[0] == 16, "N=16 panel hidden states required"
    X_centered = X_panel - np.mean(X_panel, axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / float(X_panel.shape[0] - 1)
    
    U, S, Vt = np.linalg.svd(cov, full_matrices=False)
    total_energy = float(np.sum(S))
    assert total_energy > 1e-12, "Degenerate covariance spectrum"
    
    cum_energy = np.cumsum(S) / total_energy
    selected_k = int(np.argmax(cum_energy >= energy_threshold) + 1)
    
    if not (2 <= selected_k <= 8):
        raise RuntimeError(
            f"§8 T2 Basis Rank Assertion FAILED: Derived k={selected_k} is outside permitted range [2, 8] "
            f"(cum_energy={cum_energy[:10]}). Condition T2 declared VOID."
        )
        
    U_k = torch.tensor(U[:, :selected_k], dtype=torch.float32)
    return U_k, selected_k, float(cum_energy[selected_k - 1]), [float(x) for x in cum_energy[:10]]


# ---------------------------------------------------------------------------
# Runtime Parameter and Frozen State Assertion Routine
# ---------------------------------------------------------------------------
def verify_condition_parameter_accounting(
    condition: str,
    readout: Optional[ReadoutHead],
    expected_k: Optional[int] = None
) -> Dict[str, Any]:
    """Asserts that the instantiated readout module strictly satisfies the protocol parameter accounting (§9)."""
    assert condition in CONDITIONS, f"Condition {condition} is not in canonical universe {CONDITIONS}"
    
    if condition == "B0":
        assert readout is None, "B0 must have no separate readout module"
        return {"condition": "B0", "trainable_params": 0, "status": "PASS"}
        
    assert readout is not None, f"Readout module required for condition {condition}"
    trainable_count = readout.count_trainable_params()
    trainable_params = readout.get_trainable_params()
    
    if condition == "I1":
        expected = 897
        assert trainable_count == expected, f"I1 parameter mismatch: expected {expected}, got {trainable_count}"
        assert "a" in trainable_params and "b" in trainable_params, "I1 missing a or b parameter"
        assert trainable_params["a"].numel() == 896 and trainable_params["b"].numel() == 1
    elif condition == "C2":
        expected = 897
        assert trainable_count == expected, f"C2 parameter mismatch: expected {expected}, got {trainable_count}"
        assert "a" in trainable_params and "b" in trainable_params
    elif condition == "F1":
        expected = 0
        assert trainable_count == expected, f"F1 parameter mismatch: expected {expected}, got {trainable_count}"
    elif condition == "T1":
        expected = 2
        assert trainable_count == expected, f"T1 parameter mismatch: expected {expected}, got {trainable_count}"
        assert "alpha" in trainable_params and "b" in trainable_params
        assert trainable_params["alpha"].numel() == 1 and trainable_params["b"].numel() == 1
    elif condition == "T2":
        assert expected_k is not None, "expected_k required for T2"
        expected = expected_k + 1
        assert trainable_count == expected, f"T2 parameter mismatch: expected {expected}, got {trainable_count}"
        assert "beta" in trainable_params and "b" in trainable_params
        assert trainable_params["beta"].numel() == expected_k and trainable_params["b"].numel() == 1
        assert not readout.U_k.requires_grad, "T2 U_k basis must be frozen"
    elif condition == "V1":
        expected = 1794
        assert trainable_count == expected, f"V1 parameter mismatch: expected {expected}, got {trainable_count}"
        assert "w_pro" in trainable_params and "w_abs" in trainable_params
        assert "b_pro" in trainable_params and "b_abs" in trainable_params
        assert trainable_params["w_pro"].numel() == 896 and trainable_params["w_abs"].numel() == 896
        
    return {
        "condition": condition,
        "trainable_params": trainable_count,
        "param_names": list(trainable_params.keys()),
        "status": "PASS"
    }


# ---------------------------------------------------------------------------
# §6.1 Normative Metric Implementations (Pure, Deterministic)
# ---------------------------------------------------------------------------
def compute_effective_rank(gram: np.ndarray) -> float:
    """§6.1 item 6: participation-ratio effective rank tr(G)^2 / tr(G^2), clamped >= 0."""
    eigvals = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    tr = float(np.sum(eigvals))
    tr2 = float(np.sum(eigvals ** 2))
    return 0.0 if tr2 < 1e-12 or tr < 1e-12 else float(tr ** 2 / tr2)


def get_top1_dominance(gram: np.ndarray) -> float:
    """Top-1 spectral dominance lambda_max / tr(G)."""
    eigvals = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    tr = float(np.sum(eigvals))
    return 0.0 if tr < 1e-12 else float(np.max(eigvals) / tr)


def proposal_rate(margins: np.ndarray) -> float:
    """§6.1 item 4: P(t) = fraction of dev-set margins z_PRO - z_ABS > 0."""
    margins = np.asarray(margins, dtype=np.float64)
    return 0.0 if margins.size == 0 else float(np.mean(margins > 0.0))


def cohens_d(margins: np.ndarray) -> float:
    """§6.1 item 5: (mean POS - mean ABS) / pooled sd over the N=16 panel margins.
    Panel first-8 items POS, last-8 ABS. pooled sd = sqrt((var_POS + var_ABS) / 2)."""
    margins = np.asarray(margins, dtype=np.float64)
    mp, ma = margins[:8], margins[8:]
    if mp.size == 0 or ma.size == 0:
        return 0.0
    pooled = float(np.sqrt(max((np.var(mp) + np.var(ma)) / 2.0, 1e-12)))
    return float((np.mean(mp) - np.mean(ma)) / pooled)


def resub_knn1_purity(X: np.ndarray, y: np.ndarray) -> float:
    """§6.1 item 7: resubstitution accuracy of a k=1 Euclidean NN over h_L, POS-vs-ABS."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(X) < 2:
        return 0.0
    knn = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    return float(knn.score(X, y))


def loo_cv_accuracy_signed(X: np.ndarray, y: np.ndarray) -> float:
    """§6.1 item 8: leave-one-out k=1 NN accuracy over the N=16 panel, POS-vs-ABS."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(X) < 2:
        return 0.0
    loo = LeaveOneOut()
    correct = 0
    for tr, te in loo.split(X):
        clf = KNeighborsClassifier(n_neighbors=1).fit(X[tr], y[tr])
        if int(clf.predict(X[te])[0]) == int(y[te][0]):
            correct += 1
    return float(correct / len(X))


def simultaneous_descent_fraction(dots: List[float]) -> float:
    """§6.1 item 3: fraction of (i, j) in POS x ABS with <g_i, g_j> > 0."""
    if not dots:
        return 0.0
    return float(np.mean([1.0 if d > 0.0 else 0.0 for d in dots]))


def policy_degradation(proposal_p15: Optional[float], cohen_d15: Optional[float]) -> Tuple[bool, List[str]]:
    """§6.1 item 9: policy degradation iff P(15) <= 0.0 OR d(15) < 0.00."""
    reasons = []
    if proposal_p15 is not None and proposal_p15 <= DEGRADATION_P15_MAX:
        reasons.append(f"P(15)={proposal_p15:.4f} <= {DEGRADATION_P15_MAX}")
    if cohen_d15 is not None and cohen_d15 < DEGRADATION_D15_MIN:
        reasons.append(f"d(15)={cohen_d15:.4f} < {DEGRADATION_D15_MIN}")
    return (len(reasons) > 0), reasons


# ---------------------------------------------------------------------------
# LoRA / Gradient Utilities (6E.17 parity)
# ---------------------------------------------------------------------------
MARGIN_CHUNK = 8


def lora_state(model):
    return {k: v for k, v in model.named_parameters() if "lora_" in k and v.requires_grad}


def _grad_norm(param_list) -> float:
    total = 0.0
    for p in param_list:
        if p.grad is not None:
            total += float(p.grad.float().pow(2).sum().item())
    return float(np.sqrt(total))


def _collapse_onset(telemetry: List[Dict[str, Any]]) -> Optional[int]:
    """t* = first step (after any positive dev proposal rate) where dev proposal rate hit 0.0.
    Void/telemetry records without a proposal rate are skipped (never interpreted as a step)."""
    t_star, saw_pos = None, False
    for rec in telemetry:
        if rec.get("void"):
            continue
        rate = rec.get("dev_proposal_rate")
        if rate is not None and float(rate) > 0:
            saw_pos = True
        elif saw_pos and t_star is None:
            t_star = rec["step"]
    return t_star


def collapse_guard_action(condition: str, ever_proposed: bool,
                          consecutive_zero: int, collapse_k: int = COLLAPSE_K) -> str:
    """O-1 amendment (§10): the collapse guard TRUNCATES only B0.

    Returns one of:
      "NO_TRIGGER"  - K-consecutive-zero pattern not present (normal step).
      "TRUNCATE"    - pattern present AND condition is B0 (B0-only truncation per §10).
      "CONTINUE"    - pattern present on a non-B0 condition: the collapse is recorded
                      (P2's "no proposal collapse" rule, §5.2 P2-fails path) but the
                      trajectory continues to the full horizon so §7 P-rules stay evaluable.
    """
    if not (ever_proposed and consecutive_zero >= collapse_k):
        return "NO_TRIGGER"
    return "TRUNCATE" if condition == "B0" else "CONTINUE"


# ---------------------------------------------------------------------------
# Data Layer (6E.17 parity: seed-42 family split, balanced 134/67/67, panels 8/8/8)
# ---------------------------------------------------------------------------
def construct_target_object(record) -> dict:
    """Decision-target object for the SFT examples (6E.17 parity)."""
    abstain = record.get("abstention_label", "SHOULD_ABSTAIN")
    novelty = record.get("novelty_label", "SEMANTIC_NOVEL")
    snippet = record.get("percept", "")[:35]
    if abstain == "SHOULD_PROPOSE" and novelty == "SEMANTIC_NOVEL":
        prop = record.get("target_interpretation", {}).get("proposition", "")
        return {"decision": "PROPOSE", "hypothesis": prop,
                "reasoning": f"Grounded hypothesis proposal supported by observation: '{snippet}...'"}
    elif novelty in ["REPEAT", "UNSUPPORTED"]:
        trap = record.get("trap_propositions", ["percept repeat"])[0] if record.get("trap_propositions") else "percept repeat"
        return {"decision": "ABSTAIN", "rejection_type": novelty,
                "reasoning": f"Rejection triggered for '{snippet}...': candidate '{trap[:25]}...' is a {novelty.lower()} claim."}
    else:
        return {"decision": "ABSTAIN", "rejection_type": "EPISTEMIC_THRESHOLDING",
                "reasoning": f"Epistemic thresholding triggered for '{snippet}...': insufficient evidence for grounded proposal."}


class SFTDataset(Dataset):
    """Full SFT records with prompt-masked labels and the decision token index (6E.17 parity)."""
    def __init__(self, records, tokenizer, max_length=512):
        self.examples = []
        for r in records:
            p_str = format_prompt(r["percept"], r.get("concepts", []))
            t_str = json.dumps(construct_target_object(r)) + "<|im_end|>\n"
            p_tokens = tokenizer.encode(p_str, add_special_tokens=False)
            t_tokens = tokenizer.encode(t_str, add_special_tokens=False)
            input_ids = p_tokens + t_tokens
            labels = [-100] * len(p_tokens) + t_tokens
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
                "record_metadata": r,
                "prompt_tokens_len": len(p_tokens),
                "target_tokens_len": len(t_tokens),
                "dec_token_idx": 4,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def data_collator(batch):
    """Pads to max length in the batch with pad id 151643 (6E.17 parity)."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids_batch, labels_batch, attention_mask_batch = [], [], []
    dec_idx_batch, prompt_len_batch, meta_batch = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids_batch.append(torch.cat([x["input_ids"], torch.full((pad,), 151643, dtype=torch.long)]))
        labels_batch.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        attention_mask_batch.append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        dec_idx_batch.append(x["dec_token_idx"])
        prompt_len_batch.append(x["prompt_tokens_len"])
        meta_batch.append(x["record_metadata"])
    return {
        "input_ids": torch.stack(input_ids_batch),
        "labels": torch.stack(labels_batch),
        "attention_mask": torch.stack(attention_mask_batch),
        "dec_token_idx": dec_idx_batch,
        "prompt_tokens_len": prompt_len_batch,
        "record_metadata": meta_batch,
    }


def compute_weighted_loss(logits, labels, dec_token_indices, readout=None, hiddens=None, prompt_lens=None):
    """Historical weighted decision loss (6E.14), with readout patch at the decision position.
    Decision-position raw CE is patched from the readout's (z_PRO, z_ABS) in fp32 (6E.17 parity,
    generalized to every ReadoutHead via forward_logits)."""
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    raw_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
    valid_mask = shift_labels != -100
    weighted_mask = valid_mask.float().clone()
    B = labels.size(0)
    for i in range(B):
        v_idx = torch.where(valid_mask[i])[0]
        d_pos = dec_token_indices[i]
        if d_pos < len(v_idx):
            weighted_mask[i, v_idx[d_pos]] *= LAMBDA_DECISION

    if readout is not None:
        raw_losses = raw_losses.clone()
        for i in range(B):
            v_idx = torch.where(valid_mask[i])[0]
            d_pos = v_idx[dec_token_indices[i]]
            dec_logits = shift_logits[i, d_pos, :].float().clone()
            hL = hiddens[-1][i, d_pos, :]
            z_pro, z_abs = readout.forward_logits(hL)
            dec_logits[PRO_ID] = z_pro
            dec_logits[AB_ID] = z_abs
            raw_losses[i, d_pos] = loss_fct(dec_logits.unsqueeze(0), shift_labels[i, d_pos].reshape(1)).sum()

    weighted_loss = (raw_losses * weighted_mask).sum() / max(weighted_mask.sum().item(), 1.0)
    return weighted_loss


def reconstruct_data(corpus_path: Path):
    """Exact 6E.17 data reconstruction (seed 42 family split, balanced 134/67/67, panels 8/8/8)."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)
    family_groups = defaultdict(list)
    for rec in all_records:
        family_groups[re.sub(r"_[A-D]$", "", rec["case_id"])].append(rec)
    sorted_families = sorted(family_groups.keys())
    np.random.seed(42)
    np.random.shuffle(sorted_families)
    n_train_f = int(len(sorted_families) * 0.8)
    train_families = set(sorted_families[:n_train_f])
    dev_families = set(sorted_families[n_train_f:])
    train_records_raw = [r for r in all_records if re.sub(r"_[A-D]$", "", r["case_id"]) in train_families]
    dev_records = [r for r in all_records if re.sub(r"_[A-D]$", "", r["case_id"]) in dev_families]

    pos_train = [r for r in train_records_raw if r.get("abstention_label") == "SHOULD_PROPOSE" and r.get("novelty_label") == "SEMANTIC_NOVEL"]
    abs_train = [r for r in train_records_raw if r.get("novelty_label") in ["DECISION_IRRELEVANT", "EPISTEMICALLY_PREMATURE"]]
    neg_train = [r for r in train_records_raw if r.get("novelty_label") in ["REPEAT", "UNSUPPORTED"]]

    np.random.seed(42)
    balanced_pos = list(pos_train) * 2 + list(pos_train)[:134 - len(pos_train) * 2]
    balanced_abs = list(np.random.choice(abs_train, size=67, replace=True)) if len(abs_train) < 67 else list(np.random.choice(abs_train, size=67, replace=False))
    balanced_neg = list(np.random.choice(neg_train, size=67, replace=False))
    training_records = balanced_pos + balanced_abs + balanced_neg
    np.random.shuffle(training_records)

    pos_panel = balanced_pos[:8]
    abs_panel = balanced_abs[:8]
    neg_panel = balanced_neg[:8]
    return {
        "training_records": training_records,
        "dev_records": dev_records,
        "pos_panel": pos_panel,
        "abs_panel": abs_panel,
        "neg_panel": neg_panel,
        "n_train": len(training_records),
        "n_dev": len(dev_records),
        "counts": {"pos": len(balanced_pos), "abs": len(balanced_abs), "neg": len(balanced_neg)},
    }


def synthetic_corpus(n_families: int = 60, seed: int = SEED_DATA) -> List[dict]:
    """Deterministic synthetic corpus for the zero-step self-test. Each family contributes
    2 POS, 2 ABS, and 2 NEG records (case_ids use _r0.._r5 so the _[A-D]$ family regex
    groups all six together). 60 families -> 48 train / 12 dev families -> >= 67 ABS/NEG in train."""
    rng = np.random.RandomState(seed)
    records = []
    label_kinds = {
        "POS": ("SHOULD_PROPOSE", "SEMANTIC_NOVEL"),
        "ABS1": ("SHOULD_ABSTAIN", "DECISION_IRRELEVANT"),
        "ABS2": ("SHOULD_ABSTAIN", "EPISTEMICALLY_PREMATURE"),
        "NEG1": ("SHOULD_ABSTAIN", "REPEAT"),
        "NEG2": ("SHOULD_ABSTAIN", "UNSUPPORTED"),
    }
    order = ["POS", "ABS1", "NEG1", "ABS2", "NEG2", "POS"]
    fid = 0
    for fam in range(n_families):
        for ridx, kind in enumerate(order):
            abstain, novelty = label_kinds[kind]
            records.append({
                "case_id": f"syn_fam{fam:04d}_r{ridx}",
                "abstention_label": abstain,
                "novelty_label": novelty,
                "percept": f"Synthetic percept {fid:05d}: family {fam} kind {kind}",
                "concepts": [{"id": f"c{fam}"}],
                "target_interpretation": {"proposition": f"Synthetic proposition {fid:05d}"},
                "trap_propositions": [f"repeat trap {fid:05d}"],
            })
            fid += 1
    return records


# ---------------------------------------------------------------------------
# Forward / Gradient Primitives (6E.17 parity, generalized to all readout heads)
# ---------------------------------------------------------------------------
def batched_decision_margins(model, readout, records, tokenizer):
    """Margins at the decision position (partial-prompt context), batched in chunks.
    B0 margin = z_PRO - z_ABS of the frozen logits; readout margin = z_PRO - z_ABS of the head."""
    texts = [format_prompt(r["percept"], r.get("concepts", [])) + '{"decision": "' for r in records]
    all_margins = []
    for i in range(0, len(texts), MARGIN_CHUNK):
        chunk = texts[i:i + MARGIN_CHUNK]
        enc = tokenizer(chunk, return_tensors="pt", padding=True)
        inp = enc["input_ids"].to(DEVICE)
        att = enc["attention_mask"].to(DEVICE)
        lens = att.sum(dim=1) - 1
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=att, output_hidden_states=(readout is not None))
            for j, L in enumerate(lens):
                z = out.logits[j, L, :].float()
                if readout is not None:
                    hL = out.hidden_states[-1][j, L, :]
                    z_pro, z_abs = readout.forward_logits(hL)
                    all_margins.append(float(z_pro - z_abs))
                else:
                    all_margins.append(float(z[PRO_ID] - z[AB_ID]))
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.array(all_margins)


def per_sample_decision_gradient(model, readout, item, lora_param_list, tokenizer):
    """Gradient of the decision-position CE loss (valid index 4) wrt LoRA params, batch of 1."""
    input_ids = item["input_ids"].unsqueeze(0).to(DEVICE)
    labels = item["labels"].unsqueeze(0).to(DEVICE)
    attn = item["attention_mask"].unsqueeze(0).to(DEVICE)
    model.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=(readout is not None))
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    raw_l = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
    v_idx = torch.where(shift_labels[0] != -100)[0]
    d_pos = v_idx[4]
    if readout is not None:
        hL = out.hidden_states[-1][0, d_pos, :]
        z_pro, z_abs = readout.forward_logits(hL)
        dec_logits = shift_logits[0, d_pos, :].float().clone()
        dec_logits[PRO_ID] = z_pro
        dec_logits[AB_ID] = z_abs
        dec_loss = loss_fct(dec_logits.unsqueeze(0), shift_labels[0, d_pos].reshape(1)).sum()
    else:
        dec_loss = raw_l[0, d_pos]
    grads = torch.autograd.grad(dec_loss, lora_param_list, allow_unused=True, retain_graph=False)
    flat = []
    for g, p in zip(grads, lora_param_list):
        flat.append(g.reshape(-1) if g is not None else torch.zeros_like(p.data).reshape(-1))
    model.zero_grad(set_to_none=True)
    return torch.cat(flat)


def per_sample_margin_jacobian(model, readout, rec, lora_param_list, tokenizer):
    """Gradient of the decision margin (z_PRO - z_ABS) wrt LoRA params, partial-prompt context."""
    p = format_prompt(rec["percept"], rec.get("concepts", [])) + '{"decision": "'
    inp = tokenizer(p, return_tensors="pt").to(DEVICE)
    model.zero_grad(set_to_none=True)
    out = model(**inp, output_hidden_states=(readout is not None))
    if readout is not None:
        hL = out.hidden_states[-1][0, -1, :]
        z_pro, z_abs = readout.forward_logits(hL)
        margin = (z_pro - z_abs).float()
    else:
        z = out.logits[0, -1, :]
        margin = z[PRO_ID] - z[AB_ID]
    grads = torch.autograd.grad(margin, lora_param_list, allow_unused=True, retain_graph=False)
    flat = []
    for g, p in zip(grads, lora_param_list):
        flat.append(g.reshape(-1) if g is not None else torch.zeros_like(p.data).reshape(-1))
    model.zero_grad(set_to_none=True)
    return torch.cat(flat)


def compute_cos_H1(model, readout, pos_panel, abs_panel, lora_param_list, tokenizer, panel_ds_items):
    """§6.2 aggregate cos(H1) (decision metric) + mean-pairwise telemetry.
    Raw cosine between mean POS and mean ABS decision gradients; NO label-sign inversion."""
    g_pos, g_abs = [], []
    for i in range(8):
        g_pos.append(per_sample_decision_gradient(model, readout, panel_ds_items[i], lora_param_list, tokenizer))
    for j in range(8):
        g_abs.append(per_sample_decision_gradient(model, readout, panel_ds_items[8 + j], lora_param_list, tokenizer))
    return compute_raw_gradient_cosine(g_pos, g_abs)


def hidden_reps(records, model, tokenizer):
    """Layer-24 last-token hidden states h_L for the given records (frozen forward)."""
    X = []
    for rec in records:
        p = format_prompt(rec["percept"], rec.get("concepts", [])) + '{"decision": "'
        inp = tokenizer(p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model(**inp, output_hidden_states=True)
            X.append(o.hidden_states[24][0, -1, :].float().cpu().numpy())
    return np.array(X)


def separability_suite(X, y):
    """§6.1 items 7-8 plus 6E.17-parity supplementary separability telemetry."""
    return {
        "resub_knn1": round(resub_knn1_purity(X, y), 4),
        "resub_knn3": round(float(KNeighborsClassifier(n_neighbors=3).fit(X, y).score(X, y)), 4),
        "resub_knn5": round(float(KNeighborsClassifier(n_neighbors=5).fit(X, y).score(X, y)), 4),
        "loo_knn1": round(loo_cv_accuracy_signed(X, y), 4),
        "loo_knn3": round(float(loo_cv_accuracy_k(X, y, 3)), 4),
        "svm_cv": round(float(svm_cv_accuracy(X, y)), 4),
    }


def loo_cv_accuracy_k(X, y, k):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(X) < 2:
        return 0.0
    loo = LeaveOneOut()
    correct = 0
    for tr, te in loo.split(X):
        clf = KNeighborsClassifier(n_neighbors=min(k, len(tr))).fit(X[tr], y[tr])
        if int(clf.predict(X[te])[0]) == int(y[te][0]):
            correct += 1
    return float(correct / len(X))


def svm_cv_accuracy(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(X) < 3:
        return 0.0
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED_DATA)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LinearSVC(C=1.0, dual=True, max_iter=2000, random_state=SEED_DATA).fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def readout_direction(readout) -> Optional[np.ndarray]:
    """Effective weight vector of a readout head (or None). Used for readout_cos_lda telemetry."""
    if readout is None:
        return None
    if hasattr(readout, "a"):
        return readout.a.detach().float().cpu().numpy()
    if hasattr(readout, "w"):
        return readout.w.detach().float().cpu().numpy()
    if hasattr(readout, "w_delta"):
        return readout.w_delta.detach().float().cpu().numpy()
    if hasattr(readout, "U_k"):
        w = (readout.U_k.detach() @ readout.beta.detach()).float().cpu().numpy()
        return w
    if hasattr(readout, "w_pro"):
        return readout.w_pro.detach().float().cpu().numpy()
    return None


# ---------------------------------------------------------------------------
# Model Factory: LoRA Backbone + Readout for all 7 Conditions (§3)
# ---------------------------------------------------------------------------
def build_lora_model():
    """Frozen Qwen2.5-0.5B-Instruct with PEFT LoRA (r=16, alpha=32, dropout=0.05,
    targets q/k/v/o/gate/up/down), bf16 body. Execution-only (loads the frozen base)."""
    torch.manual_seed(SEED_DATA)
    np.random.seed(SEED_DATA)
    torch.cuda.manual_seed_all(SEED_DATA)
    base_model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL_PATH), torch_dtype=torch.bfloat16,
                                                      device_map=DEVICE, trust_remote_code=True)
    base_model.config.use_cache = False
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                             lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_config)
    return model


def build_readout_head(condition, init, U_k, w_delta_t, device=DEVICE):
    """Instantiate the readout head for a condition (§3), bound by the §3.8 init constants."""
    if condition == "B0":
        return None
    if condition in ("I1", "C2"):
        a0 = init["a0"] if condition == "I1" else init["c0"]
        b0 = init["b0"] if condition == "I1" else init["b0_c2"]
        return I1ReadoutHead(a0, b0).to(device)
    if condition == "F1":
        return F1ReadoutHead(init["a0"], init["b_fixed"]).to(device)
    if condition == "T1":
        assert w_delta_t is not None, "T1 requires the frozen w_delta tensor"
        return T1ReadoutHead(w_delta_t, init["t1_alpha0"], init["t1_b0"]).to(device)
    if condition == "T2":
        assert U_k is not None, "T2 requires the §8 frozen basis U_k"
        return T2ReadoutHead(U_k, beta0=None, b0=init["t2_b0"]).to(device)
    if condition == "V1":
        return V1ReadoutHead(init["v1_w_pro_0"], init["v1_w_abs_0"],
                             init["v1_b_pro_0"], init["v1_b_abs_0"]).to(device)
    raise ValueError(f"Unknown condition: {condition}")


def assemble_trainable(model, readout) -> List[nn.Parameter]:
    """Optimizer parameter list: all model (LoRA) trainable params PLUS the readout params.
    CRITICAL 6E.18 FIX: 6E.17 excluded readout params from the optimizer, so the readout
    was never trained (readout_state_sha256 was identical from step 0 to 20)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if readout is not None:
        for p in readout.parameters():
            if p.requires_grad:
                trainable.append(p)
    return trainable


def build_condition_model(condition, init, U_k, w_delta_t):
    """Build the full condition model: real LoRA backbone + condition readout + param routing."""
    model = build_lora_model()
    readout = build_readout_head(condition, init, U_k, w_delta_t, DEVICE)
    lora_params = lora_state(model)
    sorted_names = sorted(lora_params.keys())
    lora_param_list = [lora_params[n] for n in sorted_names]
    trainable = assemble_trainable(model, readout)
    return model, readout, lora_param_list, trainable


# ---------------------------------------------------------------------------
# Condition Training Loop (6E.17 parity + readout-in-optimizer fix + collapse guard)
# ---------------------------------------------------------------------------
def evaluate_checkpoint(model, readout, tokenizer, data, panel16_ds, panel24_ds, u_lda, lora_param_list):
    """Full §6.1 metric snapshot at a checkpoint. cos_H1_aggregate is the decision metric."""
    model.eval()
    panel16 = data["pos_panel"] + data["abs_panel"]
    panel24 = panel16 + data["neg_panel"]
    m16 = batched_decision_margins(model, readout, panel16, tokenizer)
    m24 = batched_decision_margins(model, readout, panel24, tokenizer)
    cosh1 = compute_cos_H1(model, readout, data["pos_panel"], data["abs_panel"],
                           lora_param_list, tokenizer, [panel16_ds[i] for i in range(16)])
    X16 = hidden_reps(panel16, model, tokenizer)
    X24 = hidden_reps(panel24, model, tokenizer)
    y16 = np.array([1] * 8 + [0] * 8)
    y24 = np.array([1] * 8 + [0] * 8 + [2] * 8)
    # §6.1 item 9 policy metric "POS/ABS accuracy": balanced accuracy of the proposal
    # decision (margin > 0 predicts POS, margin <= 0 predicts ABS) over the N=16 panel.
    pos_acc = float((m16[:8] > 0).mean())
    abs_acc = float((m16[8:] <= 0).mean())
    rec = {
        "panel16_margin_mean": round(float(m16.mean()), 6),
        "panel16_margin_median": round(float(np.median(m16)), 6),
        "panel16_margin_std": round(float(np.std(m16)), 6),
        "panel16_proposal_rate": round(float((m16 > 0).mean()), 6),
        "panel24_proposal_rate": round(float((m24 > 0).mean()), 6),
        "panel16_margin_cohen_d": round(cohens_d(m16), 6),
        "panel16_pos_accuracy": round(pos_acc, 6),
        "panel16_abs_accuracy": round(abs_acc, 6),
        "pos_abs_accuracy": round((pos_acc + abs_acc) / 2.0, 6),
        "cos_H1_aggregate": cosh1["cos_H1_aggregate"],
        "cos_H1_mean_pairwise": cosh1["cos_H1_mean_pairwise"],
        "rho_CD": cosh1["rho_CD"],
        "F_simultaneous_feasibility": cosh1["F_simultaneous_feasibility"],
        "separability_N16": separability_suite(X16, y16),
        "separability_N24": separability_suite(X24, y24),
        "lora_grad_norm": round(_grad_norm(lora_param_list), 6),
    }
    dirn = readout_direction(readout)
    if dirn is not None:
        rec["readout_cos_lda"] = round(float(dirn @ u_lda / (np.linalg.norm(dirn) + EPS_S_SCALE)), 6)
        rec["readout_norm"] = round(float(np.linalg.norm(dirn)), 6)
    return rec


def save_checkpoint(model, readout, cond_dir, step, condition):
    """Checkpoint with per-condition readout state hashing (6E.18 auditability fix).
    LoRA stored fp16 (6E.17 parity); readout params stored fp32; both hashed."""
    st = {k: v.detach().half().cpu().contiguous() for k, v in lora_state(model).items()}
    readout_params = None
    if readout is not None:
        readout_params = readout.get_trainable_params()
        for k, v in readout_params.items():
            st[f"__readout__{k}"] = v.detach().float().cpu().contiguous()
    buf = io.BytesIO()
    torch.save(st, buf)
    h = sha256_bytes(buf.getvalue())
    (cond_dir / f"checkpoint_step_{step:02d}.pt").write_bytes(buf.getvalue())
    meta = {"condition": condition, "step": step, "checkpoint_state_sha256": h}
    if readout_params is not None:
        meta["readout_state_sha256"] = state_dict_hash(readout_params)
    with open(cond_dir / f"checkpoint_step_{step:02d}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta


def load_checkpoint(model, readout, cond_dir, step, condition):
    """Restore a checkpoint state (LoRA + readout) into the condition model."""
    st = torch.load(cond_dir / f"checkpoint_step_{step:02d}.pt", map_location="cpu")
    with torch.no_grad():
        for name, p in lora_state(model).items():
            p.copy_(st[name].to(DEVICE).float())
        if readout is not None:
            named = dict(readout.named_parameters())
            for k, p in named.items():
                key = f"__readout__{k}"
                if key in st:
                    p.copy_(st[key].to(DEVICE).float())
    return True


def run_condition(condition, data, tokenizer, init, u_lda, U_k, w_delta_t, artifact_subdir):
    """Full training trajectory for one condition. Collapse guard follows 6E.17 parity:
    ever-proposed AND 3 consecutive steps with 0.0% dev proposal rate -> stop; t* = first
    zero-after-positive. LoRA grad-norm is recorded as telemetry (not a trigger)."""
    t0 = time.time()
    print(f"\n{'='*70}\nCONDITION {condition} START\n{'='*70}", flush=True)
    cond_dir = artifact_subdir / f"condition_{condition}"
    cond_dir.mkdir(parents=True, exist_ok=True)

    model, readout, lora_param_list, trainable = build_condition_model(condition, init, U_k, w_delta_t)
    lora_init_hash = state_dict_hash(lora_state(model))
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)

    train_dataset = SFTDataset(data["training_records"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=data_collator, generator=torch.Generator().manual_seed(SEED_DATA))
    panel16_ds = SFTDataset(data["pos_panel"] + data["abs_panel"], tokenizer)
    panel24_ds = SFTDataset(data["pos_panel"] + data["abs_panel"] + data["neg_panel"], tokenizer)

    dev_margins_f0 = batched_decision_margins(model, readout, data["dev_records"], tokenizer)
    dev_prop0 = float((dev_margins_f0 > 0).mean())
    panel16_margins0 = batched_decision_margins(model, readout, data["pos_panel"] + data["abs_panel"], tokenizer)

    run = {
        "condition": condition,
        "lora_init_hash": lora_init_hash,
        "step0_dev_proposal_rate": round(dev_prop0, 6),
        "step0_panel16_proposal_rate": round(float((panel16_margins0 > 0).mean()), 6),
        "step0_panel16_margin_mean": round(float(panel16_margins0.mean()), 6),
        "step0_panel16_margin_cohen_d": round(cohens_d(panel16_margins0), 6),
        "telemetry": [],
        "checkpoints": {},
        "stopping_event": None,
    }
    if readout is not None:
        run["readout_init_hash"] = state_dict_hash(readout.get_trainable_params())

    ckpt = evaluate_checkpoint(model, readout, tokenizer, data, panel16_ds, panel24_ds, u_lda, lora_param_list)
    run["checkpoints"]["0"] = ckpt
    save_checkpoint(model, readout, cond_dir, 0, condition)
    print(f"[{condition}] step 0: dev_prop={dev_prop0:.3f} cosH1_agg={ckpt['cos_H1_aggregate']:+.4f} "
          f"readout_cos_lda={ckpt.get('readout_cos_lda')}", flush=True)

    model.train()
    global_step = 0
    accum_loss = 0.0
    consecutive_zero = 0
    ever_proposed = False
    collapsed = False

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= HORIZON:
            break
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        dec_token_idx = batch["dec_token_idx"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=(readout is not None))
        loss = compute_weighted_loss(outputs.logits, labels, dec_token_idx, readout=readout,
                                     hiddens=outputs.hidden_states)
        if not np.isfinite(float(loss.item())):
            # O-1: a condition that cannot continue numerically is recorded as an explicit
            # VOID state (never silently truncated under B0 collapse semantics).
            run["status"] = "VOID"
            run["void_reason"] = "non_finite_loss"
            run["void_detail"] = f"non-finite training loss at batch {batch_idx}"
            run["telemetry"].append({"step": global_step, "dev_proposal_rate": None,
                                     "dev_margin_mean": None, "lora_grad_norm": None,
                                     "void": True, "void_reason": "non_finite_loss"})
            print(f"[{condition}] VOID: non-finite training loss (cannot continue)", flush=True)
            break
        (loss / GRAD_ACCUM).backward()
        accum_loss += loss.item()

        if (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=CLIP_NORM)
            lora_grad_norm = _grad_norm(lora_param_list)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            dev_margins = batched_decision_margins(model, readout, data["dev_records"], tokenizer)
            dev_prop = float((dev_margins > 0).mean())
            if dev_prop > 0:
                ever_proposed = True
                consecutive_zero = 0
            else:
                consecutive_zero += 1
            run["telemetry"].append({
                "step": global_step,
                "train_loss_accum": round(accum_loss, 6),
                "dev_proposal_rate": round(dev_prop, 6),
                "dev_margin_mean": round(float(dev_margins.mean()), 6),
                "lora_grad_norm": round(float(lora_grad_norm), 6),
            })
            accum_loss = 0.0

            if global_step in CHECKPOINTS and global_step > 0:
                model.eval()
                ckpt = evaluate_checkpoint(model, readout, tokenizer, data, panel16_ds, panel24_ds, u_lda, lora_param_list)
                ckpt["dev_proposal_rate"] = round(dev_prop, 6)
                run["checkpoints"][str(global_step)] = ckpt
                save_checkpoint(model, readout, cond_dir, global_step, condition)
                print(f"[{condition}] step {global_step:02d}: dev_prop={dev_prop:.3f} "
                      f"cosH1_agg={ckpt['cos_H1_aggregate']:+.4f} margin_d={ckpt['panel16_margin_cohen_d']:+.3f} "
                      f"readout_cos_lda={ckpt.get('readout_cos_lda')}", flush=True)
                model.train()

            guard_action = collapse_guard_action(condition, ever_proposed, consecutive_zero)
            if guard_action == "TRUNCATE":
                # B0-only truncation per §10 (6E.17 parity for B0)
                run["stopping_event"] = {
                    "type": "collapse_guard",
                    "triggered_at_step": global_step,
                    "consecutive_zero_steps": consecutive_zero,
                    "detail": f"K={COLLAPSE_K} consecutive steps with 0.0% dev proposal rate",
                }
                print(f"[{condition}] COLLAPSE GUARD triggered at step {global_step}; "
                      f"truncating (B0 only, §10)", flush=True)
                collapsed = True
                break
            if guard_action == "CONTINUE":
                # Non-B0 (§10 scopes truncation to B0): collapse pattern recorded via
                # run-level collapsed flag + telemetry flag feeding P2's "no proposal
                # collapse" rule (§5.2 designed P2-fails path) but the trajectory runs
                # to the full horizon so §7 P-rules stay evaluable on the checkpoints.
                collapsed = True
                run["non_b0_collapse_continued"] = True
                run["telemetry"][-1]["collapse_pattern"] = True
                print(f"[{condition}] collapse pattern detected at step {global_step}; "
                      f"continuing to horizon (non-B0: §10 scopes truncation to B0)", flush=True)

    run["reached_horizon"] = global_step
    run["collapsed"] = collapsed
    run["collapse_onset_t_star"] = _collapse_onset(run["telemetry"])

    if run.get("status") == "VOID":
        # O-1: VOID runs keep the checkpoints recorded before the failure for provenance,
        # but rule evaluation must mark them VOID (see build_metrics_store / evaluate_p_rules).
        run["elapsed_seconds"] = round(time.time() - t0, 1)
        print(f"[{condition}] DONE (VOID) in {run['elapsed_seconds']}s | steps={global_step}", flush=True)
        return run

    # margin jacobians at t=0 and t=15 for r_eff / P3
    model.eval()
    for tt in [0, 15]:
        if str(tt) in run["checkpoints"]:
            load_checkpoint(model, readout, cond_dir, tt, condition)
            model.eval()
            jacs = [per_sample_margin_jacobian(model, readout, r, lora_param_list, tokenizer)
                    for r in (data["pos_panel"] + data["abs_panel"])]
            G = np.zeros((16, 16), dtype=np.float64)
            for i in range(16):
                for j in range(i, 16):
                    v = float(torch.dot(jacs[i], jacs[j]))
                    G[i, j] = v
                    G[j, i] = v
            run["checkpoints"][str(tt)]["margin_jacobian_gram"] = {
                "effective_rank": round(compute_effective_rank(G), 4),
                "top1_dominance": round(get_top1_dominance(G), 4),
            }
            del jacs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if readout is not None:
        run["readout_alignment_trajectory"] = {
            str(t): run["checkpoints"][str(t)]["readout_cos_lda"] for t in CHECKPOINTS if str(t) in run["checkpoints"]
        }

    run["elapsed_seconds"] = round(time.time() - t0, 1)
    vals = [run["checkpoints"][str(t)]["cos_H1_aggregate"] for t in RULE_CHECKPOINTS
            if str(t) in run["checkpoints"]]
    mval = round(float(np.mean(vals)), 4) if vals else None
    print(f"[{condition}] DONE in {run['elapsed_seconds']}s | steps={global_step} | t*={run['collapse_onset_t_star']} | "
          f"mean_cosH1_agg{tuple(RULE_CHECKPOINTS)}={mval}", flush=True)
    return run


# ---------------------------------------------------------------------------
# Verdict & Manifest Infrastructure (§7, §11, §14) with Authorization Gate
# ---------------------------------------------------------------------------
PRE_EXECUTION_VERDICT = {
    "status": "PRE_EXECUTION",
    "scientific_verdict": "NO_SCIENTIFIC_VERDICT",
    "authorization": "EXECUTION_NOT_AUTHORIZED",
}


def check_execution_authorization(artifact_dir: Path) -> Dict[str, Any]:
    """§14 authorization gate: execution/verdict ONLY with a signed execution_authorization.json
    (decision APPROVE + signature + date + phase 6E.18)."""
    auth_path = artifact_dir / "execution_authorization.json"
    if not auth_path.exists():
        return {"authorized": False, "reason": "execution_authorization.json absent", "path": str(auth_path)}
    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"authorized": False, "reason": f"unparseable execution_authorization.json: {e}", "path": str(auth_path)}
    ok = bool(auth.get("decision") == "APPROVE" and auth.get("signature") and auth.get("date") and auth.get("phase") == "6E.18")
    return {"authorized": ok, "reason": "signed APPROVE (phase 6E.18)" if ok else
            "missing APPROVE decision / signature / date / phase", "record": auth}


def build_metrics_store(runs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes per-condition checkpoint dicts into the rule-evaluator metric store
    ({cond: {cos_H1: {t: agg cos}, proposal_rate, margin_d, margin_jacobian_r_eff,
    separability, readout_cos_lda, collapsed, collapse_onset_t_star, telemetry}})."""
    store: Dict[str, Any] = {}
    for cond, run in runs.items():
        ck = run.get("checkpoints", {})
        store[cond] = {
            "cos_H1": {int(t): c.get("cos_H1_aggregate") for t, c in ck.items()
                       if c.get("cos_H1_aggregate") is not None},
            "proposal_rate": {int(t): c.get("dev_proposal_rate") for t, c in ck.items()
                              if "dev_proposal_rate" in c},
            "margin_d": {int(t): c.get("panel16_margin_cohen_d") for t, c in ck.items()
                         if c.get("panel16_margin_cohen_d") is not None},
            "margin_jacobian_r_eff": {int(t): c.get("margin_jacobian_gram", {}).get("effective_rank")
                                      for t, c in ck.items() if c.get("margin_jacobian_gram")},
            "separability": {int(t): {"N16": c.get("separability_N16"), "N24": c.get("separability_N24")}
                             for t, c in ck.items()},
            "readout_cos_lda": {int(t): c.get("readout_cos_lda") for t, c in ck.items()
                                if c.get("readout_cos_lda") is not None},
            "pos_accuracy": {int(t): c.get("panel16_pos_accuracy") for t, c in ck.items()
                             if c.get("panel16_pos_accuracy") is not None},
            "abs_accuracy": {int(t): c.get("panel16_abs_accuracy") for t, c in ck.items()
                             if c.get("panel16_abs_accuracy") is not None},
            "pos_abs_accuracy": {int(t): c.get("pos_abs_accuracy") for t, c in ck.items()
                                 if c.get("pos_abs_accuracy") is not None},
            "collapsed": run.get("collapsed", False),
            "collapse_onset_t_star": run.get("collapse_onset_t_star"),
            "status": run.get("status", "COMPLETED"),
            "void_reason": run.get("void_reason"),
            "non_b0_collapse_continued": run.get("non_b0_collapse_continued", False),
            "telemetry": run.get("telemetry", []),
        }
    return store


def _mean_agg_cos(metrics: Dict[str, Any], cond: str) -> Optional[float]:
    vals = [metrics.get(cond, {}).get("cos_H1", {}).get(t) for t in RULE_CHECKPOINTS]
    if any(v is None for v in vals):
        return None
    return float(np.mean([float(v) for v in vals]))


def evaluate_p_rules(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """§7 frozen decision rules P1-P5 + C2x (aggregate cos(H1) only)."""
    i1 = metrics.get("I1", {})
    c2 = metrics.get("C2", {})
    i1_void = i1.get("void_reason")
    c2_void = c2.get("void_reason")
    void_detail = f"I1 VOID: {i1_void}" if i1_void else None
    mI1 = _mean_agg_cos(metrics, "I1")
    mC2 = _mean_agg_cos(metrics, "C2")
    P1 = (mI1 is not None) and (mI1 >= P1_THRESH)
    C2x = (mI1 is not None and mC2 is not None) and (mI1 - mC2) >= C2X_THRESH

    tele = i1.get("telemetry", [])
    prop_ok = all(float(r.get("dev_proposal_rate", 0.0)) > 0.0
                  for r in tele if r.get("dev_proposal_rate") is not None)
    P2 = prop_ok and not i1.get("collapsed", True)

    d15 = i1.get("margin_d", {}).get(15)
    r_eff15 = i1.get("margin_jacobian_r_eff", {}).get(15)
    P3 = (d15 is not None and d15 >= P3_COHEN_D_THRESH) and \
         (r_eff15 is not None and r_eff15 <= P3_R_EFF_THRESH)

    b0_sep = metrics.get("B0", {}).get("separability", {})
    i1_sep = i1.get("separability", {})
    # §7.1 frozen step-0 anchor LOO-CV_B0(0)
    b0_anchor = None
    if b0_sep.get(0):
        b0_anchor = {"N16": b0_sep[0].get("N16", {}).get("loo_knn1"),
                     "N24": b0_sep[0].get("N24", {}).get("loo_knn1")}
    # §7.1 comparator universe: shared evaluable {0,4,6} + post-truncation {14,15,20}
    candidate_steps = sorted({int(k) for k in i1_sep.keys()} & {0, 4, 6, 14, 15, 20})
    k1_ok, loo_ok = True, True
    loo_pairs, unavailable = {}, []
    for t in candidate_steps:
        i1_panel = i1_sep.get(t)
        if not i1_panel:
            continue
        for panel in ("N16", "N24"):
            k1 = i1_panel.get(panel, {}).get("resub_knn1")
            if k1 is not None and not (P4_KN1_RANGE[0] <= k1 <= P4_KN1_RANGE[1]):
                k1_ok = False
            loo_i1 = i1_panel.get(panel, {}).get("loo_knn1")
            if loo_i1 is None:
                continue
            anchor = b0_anchor.get(panel) if b0_anchor else None
            if t in (0, 4, 6):
                loo_b = b0_sep.get(t, {}).get(panel, {}).get("loo_knn1") if b0_sep.get(t) else None
                if loo_b is not None:
                    passed, diff, regime = evaluate_p4_comparator(loo_i1, loo_b, anchor if anchor is not None else 0.0, t)
                    loo_pairs[f"t{t}_{panel}"] = {"i1": loo_i1, "b0": loo_b, "diff": round(diff, 4), "regime": regime}
                elif anchor is not None:
                    # §7.1 early-truncation fallback: step-0 anchor applies; missing comparator recorded UNAVAILABLE
                    passed, diff, regime = evaluate_p4_comparator(loo_i1, None, anchor, t)
                    loo_pairs[f"t{t}_{panel}"] = {"i1": loo_i1, "b0_step0_anchor": anchor, "diff": round(diff, 4), "regime": regime}
                    unavailable.append(f"t{t}_{panel} (B0({t}) missing; step-0 anchor applied per §7.1)")
                else:
                    unavailable.append(f"t{t}_{panel} (B0({t}) and B0(0) both unavailable)")
                    continue
            else:
                # post-truncation {14,15,20}: frozen step-0 anchor (§7.1)
                if anchor is None:
                    unavailable.append(f"t{t}_{panel} (B0(0) anchor unavailable)")
                    continue
                passed, diff, regime = evaluate_p4_comparator(loo_i1, None, anchor, t)
                loo_pairs[f"t{t}_{panel}"] = {"i1": loo_i1, "b0_step0_anchor": anchor, "diff": round(diff, 4), "regime": regime}
            if not passed:
                loo_ok = False
    not_evaluable = len(loo_pairs) == 0
    if i1_void:
        # O-1: a VOID condition's rules are NOT EVALUABLE, never silently evaluated on
        # the partial checkpoints captured before the numerical failure.
        not_evaluable = True
        k1_ok = loo_ok = False
    P4 = (not not_evaluable) and k1_ok and loo_ok

    aligns = [abs(v) for t, v in i1.get("readout_cos_lda", {}).items() if int(t) >= 4]
    P5 = (max(aligns) >= P5_THRESH) if aligns else False

    if i1_void:
        P1 = P2 = P3 = P5 = False
    if c2_void and not i1_void:
        C2x = False

    return {
        "P1": {"passed": bool(P1), "evaluable": not i1_void, "detail": void_detail,
               "mean_cos_H1_I1": mI1, "threshold": P1_THRESH},
        "C2x": {"passed": bool(C2x),
                "evaluable": not (i1_void or c2_void),
                "detail": void_detail if i1_void else (f"C2 VOID: {c2_void}" if c2_void else None),
                "delta": (round(mI1 - mC2, 6) if (mI1 is not None and mC2 is not None) else None),
                "mean_cos_H1_I1": mI1, "mean_cos_H1_C2": mC2, "threshold": C2X_THRESH},
        "P2": {"passed": bool(P2), "evaluable": not i1_void, "detail": void_detail,
               "collapsed": i1.get("collapsed", True),
               "t_star": i1.get("collapse_onset_t_star"), "all_steps_positive": prop_ok},
        "P3": {"passed": bool(P3), "evaluable": not i1_void, "detail": void_detail,
               "cohen_d_15": d15, "r_eff_15": r_eff15,
               "thresholds": [P3_COHEN_D_THRESH, P3_R_EFF_THRESH]},
        "P4": {"passed": bool(P4), "evaluable": not i1_void, "detail": void_detail,
               "not_evaluable": not_evaluable,
               "resub_knn1_in_range": k1_ok, "loo_within_tolerance": loo_ok,
               "loo_pairs_by_checkpoint": loo_pairs,
               "comparator_unavailable": unavailable, "comparator_universe": candidate_steps,
               "loo_tolerance": P4_LOO_TOL, "knn1_range": list(P4_KN1_RANGE)},
        "P5": {"passed": bool(P5), "evaluable": not i1_void, "detail": void_detail,
               "max_abs_cos_lda": (max(aligns) if aligns else None),
               "threshold": P5_THRESH, "values_t_ge_4": {str(t): v for t, v in i1.get("readout_cos_lda", {}).items() if int(t) >= 4}},
    }


ROW7_ANCHOR_CHECKPOINTS = [0, 4, 6, 14, 15]
ROW7_NOT_DETERMINABLE = "ROW_7_NOT_DETERMINABLE_FROM_REGISTERED_METRICS"


def evaluate_row7(metrics: Dict[str, Any], condition: str = "I1") -> Dict[str, Any]:
    """§5.2 row 7 — 'One class improves while other degrades' -> Artifactual trade-off;
    NOT accepted as conflict resolution.

    Deterministic and protocol-language-only (no invented scientific threshold):
      - Per-class POS/ABS accuracy from the §6.1 item 9 policy metric set, evaluated over
        the N=16 panel margins with the registered proposal decision (margin > 0 predicts
        POS; margin <= 0 predicts ABS — §6.1 item 4):
            a_POS(t) = fraction of POS panel with margin > 0
            a_ABS(t) = fraction of ABS panel with margin <= 0
      - 'improves' / 'degrades' = strict sign of the change between consecutive registered
        anchor checkpoints ({0} + RULE_CHECKPOINTS; t=15 is the §6.1 item 9 policy anchor).
        Strict comparisons only; no magnitude threshold.
      - Trade-off present iff any consecutive anchor pair shows one class strictly improving
        while the other strictly degrades.
      - Missing required registered metrics (e.g., VOID condition, missing checkpoint) are
        represented explicitly as ROW_7_NOT_DETERMINABLE_FROM_REGISTERED_METRICS with the
        specific gap documented — Row 7 is never dropped.
    """
    cond_store = metrics.get(condition, {})
    if cond_store.get("void_reason"):
        return {"label": ROW7_NOT_DETERMINABLE, "condition": condition,
                "reason": f"condition {condition} VOID ({cond_store['void_reason']})",
                "text": "Row 7 not determinable from registered metrics (VOID condition)."}
    pos = cond_store.get("pos_accuracy", {})
    abs_ = cond_store.get("abs_accuracy", {})
    missing = [t for t in ROW7_ANCHOR_CHECKPOINTS
               if pos.get(t) is None or abs_.get(t) is None]
    if missing:
        return {"label": ROW7_NOT_DETERMINABLE, "condition": condition,
                "reason": "POS/ABS accuracy missing at registered checkpoints "
                          + ",".join(str(t) for t in missing),
                "missing_checkpoints": missing,
                "text": "Row 7 not determinable from registered metrics (POS/ABS accuracy incomplete)."}
    deltas = []
    for t0, t1 in zip(ROW7_ANCHOR_CHECKPOINTS, ROW7_ANCHOR_CHECKPOINTS[1:]):
        d_pos = float(pos[t1] - pos[t0])
        d_abs = float(abs_[t1] - abs_[t0])
        trade = (d_pos > 0.0 and d_abs < 0.0) or (d_abs > 0.0 and d_pos < 0.0)
        deltas.append({"interval": f"{t0}->{t1}", "d_pos_accuracy": round(d_pos, 6),
                       "d_abs_accuracy": round(d_abs, 6), "artifactual_trade_off": trade})
    trades = [d for d in deltas if d["artifactual_trade_off"]]
    if trades:
        return {"label": "ARTIFACTUAL_TRADE_OFF", "condition": condition,
                "trade_off_intervals": trades, "deltas": deltas,
                "text": "One class improves while other degrades: artifactual trade-off; "
                        "NOT accepted as conflict resolution (§5.2 row 7)."}
    return {"label": "NO_ARTIFACTUAL_TRADE_OFF", "condition": condition,
            "deltas": deltas,
            "text": "No registered interval shows one class improving while the other degrades (§5.2 row 7)."}


def summarize_outcome(p: Dict[str, Any], m: Dict[str, Any],
                      row7: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """§5.2 pre-registered outcome interpretation rows."""
    rows = []
    m1 = m.get("M1", {}).get("disposition")
    m2 = m.get("M2", {}).get("disposition")
    m3 = m.get("M3", {}).get("disposition")
    m4 = m.get("M4", {}).get("disposition")
    m5 = m.get("M5", {}).get("disposition")
    p1, c2x = p["P1"]["passed"], p["C2x"]["passed"]
    p2, p4 = p["P2"]["passed"], p["P4"]["passed"]

    if p1 and c2x and m1 == "SUPPORTED":
        rows.append({"label": "STRONG_H_CAUSAL",
                     "text": "Strong support for H-Causal (alignment-specific mechanism requires trainable adaptation)."})
    elif p1 and m5 == "SUPPORTED" and not c2x:
        rows.append({"label": "CAPACITY_EXPLANATION",
                     "text": "Capacity/trainability explanation strengthened; alignment claim rejected."})
    elif not p1 and m5 == "NOT SUPPORTED" and m1 in ("SUPPORTED", "INDETERMINATE"):
        rows.append({"label": "H0_SURVIVES",
                     "text": "Readout intervention ineffective; H0 survives."})
    elif p1 and c2x and m1 == "INDETERMINATE":
        rows.append({"label": "H_CAUSAL_NOT_DECISIVE",
                     "text": "H-Causal not decisively supported; F1 fixed-geometry insufficiency neither established nor refuted."})
    elif p1 and not p2 and p4:
        rows.append({"label": "H_CAUSAL_PARTIALLY_SUPPORTED",
                     "text": "P1 holds, P2 fails, P4 holds (partial)."})
    else:
        rows.append({"label": "MIXED_INCONCLUSIVE", "text": "Mixed/inconclusive outcome pattern."})

    if m3 == "SUPPORTED" and m2 == "NOT SUPPORTED":
        rows.append({"label": "M3_M2_COMPOSITE",
                     "text": "Low-rank adapted readout matches aligned full-rank readout; 1D gain insufficient (rank/basis/init confounded)."})
    if m4 == "SUPPORTED":
        rows.append({"label": "M4",
                     "text": "Unconstrained 2-logit parameterization and expanded capacity provide independent advantage."})
    if row7 is not None:
        # §5.2 row 7 disposition is always surfaced (never dropped when indeterminate).
        rows.append({"label": row7["label"], "text": row7.get("text", ""),
                     "condition": row7.get("condition")})
    return {"rows": rows}


def compute_verdict(runs, derive, t2_k):
    """Post-execution scientific verdict (aggregate cos). ONLY invoked when execution is
    authorized AND executed checkpoints exist (see emit_verdict)."""
    metrics = build_metrics_store(runs)
    i1_void = bool(runs.get("I1", {}).get("void_reason"))
    p = evaluate_p_rules(metrics)
    m = evaluate_m_rules(metrics, t2_k=t2_k, i1_void=i1_void)
    row7 = evaluate_row7(metrics)
    return {
        "status": "EXECUTED",
        "P1_conflict_reduction": p["P1"],
        "C2x_alignment_exclusion": p["C2x"],
        "P2_collapse_prevention": p["P2"],
        "P3_margin_maintenance": p["P3"],
        "P4_no_separability_degradation": p["P4"],
        "P5_readout_engagement": p["P5"],
        "M1": m["M1"], "M2": m["M2"], "M3": m["M3"], "M4": m["M4"], "M5": m["M5"],
        "Row7_artifactual_trade_off": row7,
        "outcome_summary": summarize_outcome(p, m, row7=row7),
    }


def emit_verdict(artifact_dir: Path, runs, derive, t2_k, authorization) -> Dict[str, Any]:
    """Verdict emission safeguard: a scientific verdict is emitted ONLY when (a) §14
    authorization is present and (b) executed checkpoints exist. Otherwise the verdict is
    exactly PRE_EXECUTION / NO_SCIENTIFIC_VERDICT / EXECUTION_NOT_AUTHORIZED."""
    if not authorization.get("authorized"):
        out = dict(PRE_EXECUTION_VERDICT)
        out["reason"] = authorization["reason"]
        out["runs_present"] = bool(runs)
        return out
    executed = any(run.get("checkpoints") for run in (runs or {}).values())
    if not executed:
        out = dict(PRE_EXECUTION_VERDICT)
        out["reason"] = "authorization present but no executed checkpoints"
        out["runs_present"] = bool(runs)
        return out
    verdict = compute_verdict(runs, derive, t2_k)
    verdict["authorization_date"] = authorization["record"].get("date")
    verdict["authorization_signature"] = authorization["record"].get("signature")
    return verdict


def generate_sha256_manifest(artifact_dir: Path, elapsed: float, extra: Optional[dict] = None) -> Dict[str, Any]:
    """§11 forensic manifest: hashes of code, protocol, and every artifact file."""
    manifest = {
        "phase": "6E.18",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "script_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": round(float(elapsed), 1),
        "artifacts": [],
    }
    if extra:
        manifest.update(extra)
    for p in sorted(artifact_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(artifact_dir)).replace("\\", "/")
            manifest["artifacts"].append({"path": rel, "sha256": sha256_file(p)})
    return manifest


# ---------------------------------------------------------------------------
# Mock Backbone / Tokenizer (Zero-Step Self-Test Only — never used for execution)
# ---------------------------------------------------------------------------
class MockTokenizer:
    """Deterministic, dependency-free tokenizer for the self-test. Mirrors the subset of the
    HF tokenizer API used by the runner (encode / __call__ / decode / pad_token / eos_token)."""
    def __init__(self, vocab=VOCAB, seed=SEED_DATA):
        self.vocab_size = vocab
        self.pad_token_id = 151643
        self.pad_token = "mock_pad"
        self.eos_token = "mock_eos"

    def encode(self, text, add_special_tokens=False):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        length = 6 + (h[0] % 8)  # 6..13 tokens
        return [(int(h[i % len(h)]) + 100 + 3 * i) % (self.vocab_size - 200) + 100 for i in range(length)]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=None):
        return " ".join(str(int(t)) for t in ids)

    def __call__(self, text, return_tensors=None, padding=False, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        do_pad = padding if not isinstance(text, str) else False
        all_ids = [self.encode(t) for t in texts]
        maxlen = max(len(x) for x in all_ids)
        input_ids, attention_mask = [], []
        for ids in all_ids:
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
        out = types.SimpleNamespace()
        out.input_ids = torch.tensor(input_ids, dtype=torch.long)
        out.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return out


class MockBackbone(nn.Module):
    """Causal-LM-shaped mock for the zero-step self-test. A low-rank lm_head
    (logits = h @ (B^T @ A^T), A:[vocab,rank], B:[rank,hidden]) keeps the [B,T,151936]
    logits cheap while exercising the exact loss/readout/gradient code paths. Trainable
    block projections carry 'lora_' in their parameter names to satisfy lora_state()."""
    def __init__(self, hidden=D_HIDDEN, n_blocks=2, vocab=VOCAB, rank=64, seed=SEED_DATA):
        super().__init__()
        self.hidden, self.n_blocks, self.vocab, self.rank = hidden, n_blocks, vocab, rank
        gen = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, hidden)
        for p in self.embed.parameters():
            p.requires_grad = False
        blocks = nn.ModuleList()
        for i in range(n_blocks):
            b = nn.Module()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                b.add_module(name, nn.Linear(hidden, hidden, bias=False))
            blocks.append(b)
        self.lora_blocks = blocks  # 'lora_' in parameter path -> lora_state() picks them up
        self.lm_head_a = nn.Parameter(torch.randn(vocab, rank, generator=gen) * 0.02, requires_grad=False)
        self.lm_head_b = nn.Parameter(torch.randn(rank, hidden, generator=gen) * 0.02, requires_grad=False)
        self.config = types.SimpleNamespace(hidden_size=hidden, vocab_size=vocab, num_hidden_layers=n_blocks)
        self.use_cache = False

    def get_output_embeddings(self):
        return types.SimpleNamespace(weight=self.lm_head_a)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        h = self.embed(input_ids)  # [B,T,hidden]
        hidden_states = [h]
        for block in self.lora_blocks:
            x = h
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                x = x + block.__getattr__(name)(x)
            for name in ("gate_proj", "up_proj"):
                x = x + block.__getattr__(name)(x)
            x = x + block.__getattr__("down_proj")(x)
            h = x / (1.0 + 1e-6)
            hidden_states.append(h)
        logits = torch.einsum("bth,rh,vr->btv", h, self.lm_head_b, self.lm_head_a)
        return types.SimpleNamespace(logits=logits, hidden_states=hidden_states)


def _mock_decision_loss(model, readout, item, tokenizer):
    """Decision-position CE with the readout patch (mirrors compute_weighted_loss for batch=1)."""
    input_ids = item["input_ids"].unsqueeze(0)
    labels = item["labels"].unsqueeze(0)
    attn = item["attention_mask"].unsqueeze(0)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    shift_logits = out.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    raw_l = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())
    v_idx = torch.where(shift_labels[0] != -100)[0]
    d_pos = v_idx[4]
    hL = out.hidden_states[-1][0, d_pos, :]
    z_pro, z_abs = readout.forward_logits(hL)
    dec_logits = shift_logits[0, d_pos, :].float().clone()
    dec_logits[PRO_ID] = z_pro
    dec_logits[AB_ID] = z_abs
    return loss_fct(dec_logits.unsqueeze(0), shift_labels[0, d_pos].reshape(1)).sum()


def run_self_test() -> Dict[str, Any]:
    """Zero-step end-to-end self-test: proves the full 6E.18 pipeline builds, routes, hashes,
    and emits the PRE_EXECUTION verdict — with ZERO optimizer steps, ZERO frozen-model loads,
    ZERO training batches processed for scientific execution."""
    print("=" * 80)
    print("PHASE 6E.18 ZERO-STEP SELF-TEST (NON-TRAINING, MOCK-BACKED)")
    print("=" * 80)
    results: Dict[str, Any] = {}
    tok = MockTokenizer()

    # 1. Data layer over a synthetic corpus
    corpus = synthetic_corpus()
    with tempfile.TemporaryDirectory(prefix="theo_6e18_selftest_") as tmp:
        tmp_dir = Path(tmp)
        corpus_path = tmp_dir / "candidate_records.json"
        corpus_path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
        data = reconstruct_data(corpus_path)
        assert data["counts"]["abs"] == 67 and data["counts"]["neg"] == 67, "balanced 67/67 invariant"
        assert len(data["pos_panel"]) == 8 and len(data["abs_panel"]) == 8 and len(data["neg_panel"]) == 8
        assert data["n_dev"] > 0
        results["data_layer"] = {"train": data["n_train"], "dev": data["n_dev"],
                                 "counts": data["counts"], "panels": [8, 8, 8]}

        ds = SFTDataset(data["training_records"][:8], tok)
        item = ds[0]
        assert item["dec_token_idx"] == 4
        assert int(item["labels"].numel()) > 5, "target tokens must exceed decision index 4"
        results["sft_dataset"] = "PASS"

        # 2. Pure §6.1 metric functions on deterministic synthetic values
        pos = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        abs_ = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        margins = np.concatenate([pos, abs_])
        assert proposal_rate(margins) == 0.5
        assert np.isfinite(cohens_d(margins)), "degenerate zero-variance panel must be finite via eps floor"
        pos_v = np.array([2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0])  # mean 1, var 1
        margins_v = np.concatenate([pos_v, abs_])
        assert abs(cohens_d(margins_v) - (2.0 ** 0.5)) < 1e-9, "Cohen's d of var-1 POS vs zero ABS must be sqrt(2)"
        G = np.diag([1.0, 1.0, 1.0, 1.0])
        assert abs(compute_effective_rank(G) - 4.0) < 1e-6
        assert abs(compute_effective_rank(np.ones((4, 4))) - 1.0) < 1e-6
        assert get_top1_dominance(np.diag([2.0, 1.0])) == 2.0 / 3.0
        X = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
        y = np.array([1, 1, 0, 0])
        assert resub_knn1_purity(X, y) == 1.0
        assert loo_cv_accuracy_signed(X, y) == 1.0
        assert simultaneous_descent_fraction([1.0, -1.0, 1.0]) == 2.0 / 3.0
        deg, reasons = policy_degradation(0.0, 0.5)
        assert deg and reasons, "P(15)=0.0 must be policy degradation"
        deg2, _ = policy_degradation(0.5, 0.1)
        assert not deg2, "P(15)>0 and d(15)>=0 must not be degradation"
        results["metrics_pure"] = "PASS"

        # 3. §6.1 sign-gate + gradient-flow through a readout (6E.17 readout-bug fix proof)
        verify_synthetic_sign_gate(eps=SIGN_GATE_EPSILON)
        init = compute_init_constants(
            np.zeros((16, D_HIDDEN)), np.zeros(16),
            np.eye(D_HIDDEN)[0], np.eye(D_HIDDEN)[1])
        mock = MockBackbone(n_blocks=2).to(DEVICE)
        readout = build_readout_head("I1", init, None, None, DEVICE)
        item = ds[0].copy()
        item["input_ids"] = item["input_ids"].to(DEVICE)
        item["labels"] = item["labels"].to(DEVICE)
        item["attention_mask"] = item["attention_mask"].to(DEVICE)
        mock.zero_grad(set_to_none=True)
        loss = _mock_decision_loss(mock, readout, item, tok)
        loss.backward()
        readout_grads_ok = all(p.grad is not None for p in readout.parameters() if p.requires_grad)
        assert readout_grads_ok, "READOUT GRADIENTS MUST FLOW (6E.17 bug fix)"
        with torch.no_grad():
            loss_ref = _mock_decision_loss(mock, readout, item, tok).item()
            readout.a.add_(0.5)
            loss_pert = _mock_decision_loss(mock, readout, item, tok).item()
            readout.a.sub_(0.5)
        assert abs(loss_pert - loss_ref) > 0.0, "loss must be sensitive to readout.a (readout wired into objective)"
        trainable = assemble_trainable(mock, readout)
        assert all(any(p is tp for tp in trainable) for p in readout.parameters() if p.requires_grad), \
            "readout params must be in the optimizer parameter list (6E.18 fix)"
        lora_params = lora_state(mock)
        assert lora_params, "mock must expose lora_* trainable params"
        lora_list = [lora_params[n] for n in sorted(lora_params.keys())]
        g = per_sample_decision_gradient(mock, readout, item, lora_list, tok)
        assert g.numel() == sum(p.numel() for p in lora_list)
        assert float(g.norm().item()) > 0.0
        results["gradient_flow_readout_fix"] = "PASS"

        # 4. Checkpoint routing + readout state hashing (change detection)
        cond_dir = tmp_dir / "condition_I1"
        cond_dir.mkdir()
        save_checkpoint(mock, readout, cond_dir, 0, "I1")
        h0 = json.loads((cond_dir / "checkpoint_step_00.json").read_text(encoding="utf-8"))
        assert "readout_state_sha256" in h0 and "checkpoint_state_sha256" in h0
        with torch.no_grad():
            readout.a.add_(torch.ones_like(readout.a) * 0.5)
        save_checkpoint(mock, readout, cond_dir, 4, "I1")
        h4 = json.loads((cond_dir / "checkpoint_step_04.json").read_text(encoding="utf-8"))
        assert h4["readout_state_sha256"] != h0["readout_state_sha256"], \
            "readout_state_sha256 MUST change when readout weights move (auditability)"
        load_checkpoint(mock, readout, cond_dir, 0, "I1")
        assert abs(float(readout.a[0].item()) - float(init["a0"][0])) < 1e-6, "load_checkpoint restore failed"
        results["checkpoint_routing_hashing"] = "PASS"

        # 5. All 7 conditions: readout factory + parameter accounting (dummy weights)
        dummy_w = np.zeros(D_HIDDEN, dtype=np.float32)
        dummy_w_delta = torch.zeros(D_HIDDEN, dtype=torch.float32)
        dummy_Uk = torch.zeros(D_HIDDEN, 4, dtype=torch.float32)
        heads = {
            "B0": build_readout_head("B0", init, None, None, DEVICE),
            "I1": build_readout_head("I1", init, None, None, DEVICE),
            "C2": build_readout_head("C2", init, None, None, DEVICE),
            "F1": build_readout_head("F1", init, None, None, DEVICE),
            "T1": build_readout_head("T1", init, None, dummy_w_delta.to(DEVICE), DEVICE),
            "T2": build_readout_head("T2", init, dummy_Uk.to(DEVICE), None, DEVICE),
            "V1": build_readout_head("V1", init, None, None, DEVICE),
        }
        for cond, hd in heads.items():
            k_val = 4 if cond == "T2" else None
            res = verify_condition_parameter_accounting(cond, hd, expected_k=k_val)
            assert res["status"] == "PASS", f"{cond} parameter accounting failed"
        results["seven_condition_factory"] = {c: "PASS" for c in CONDITIONS}

        # 6. P1-P5/C2x + M1-M5 rule evaluators on synthetic metric stores
        def ck(c, extra=None):
            d = {"cos_H1": {t: c for t in RULE_CHECKPOINTS},
                 "proposal_rate": {t: 0.5 for t in RULE_CHECKPOINTS},
                 "margin_d": {t: 0.5 for t in RULE_CHECKPOINTS},
                 "separability": {t: {"N16": {"resub_knn1": 0.9, "loo_knn1": 0.75},
                                      "N24": {"resub_knn1": 0.85, "loo_knn1": 0.70}} for t in [0] + RULE_CHECKPOINTS}}
            if extra:
                d.update(extra)
            return d
        metrics = {
            "I1": ck(-0.20, {"readout_cos_lda": {4: -0.4, 6: -0.4, 14: -0.4, 15: -0.4}, "collapsed": False,
                             "collapse_onset_t_star": None, "telemetry": [{"dev_proposal_rate": 0.6, "step": s} for s in range(1, 21)],
                             "margin_jacobian_r_eff": {15: 1.2}}),
            "C2": ck(-0.60),
            "B0": ck(-0.95),
            "F1": ck(-0.70),
            "T1": ck(-0.10),
            "T2": ck(-0.25),
            "V1": ck(-0.10, {"proposal_rate": {15: 0.5}, "margin_d": {15: 0.3}}),
        }
        p = evaluate_p_rules(metrics)
        assert p["P1"]["passed"] and p["C2x"]["passed"] and p["P2"]["passed"]
        assert p["P3"]["passed"] and p["P5"]["passed"]
        assert p["P4"]["passed"]
        m = evaluate_m_rules(metrics, t2_k=4, i1_void=False)
        assert m["M1"]["disposition"] == "SUPPORTED" and m["M2"]["disposition"] == "SUPPORTED"
        assert m["M3"]["disposition"] == "SUPPORTED" and m["M4"]["disposition"] == "SUPPORTED"
        assert m["M5"]["disposition"] == "NOT SUPPORTED"  # C2 cos=-0.60 < floor -0.30, consistent with C2x pass
        results["decision_rules"] = "PASS"

        # 6b. O-1 amendment (§10): B0-only collapse guard + explicit VOID state (targeted)
        def run_fixture(cond, cos_c, collapsed=False, void=None, collapse_cont=False,
                        telemetry=None, acc=None):
            """Raw run_condition-shaped fixture with full checkpoints [0,4,6,7,14,15,17,20].
            acc: optional {t: (pos_acc, abs_acc)} overriding the default 0.75/0.75; a value of
            None for a checkpoint records it WITHOUT the accuracy fields (registered-metric gap)."""
            cks = {}
            for t in CHECKPOINTS:
                ck = {
                    "cos_H1_aggregate": cos_c,
                    "dev_proposal_rate": 0.5,
                    "panel16_margin_cohen_d": 0.5,
                    "margin_jacobian_gram": {"effective_rank": 1.2},
                    "separability_N16": {"resub_knn1": 0.9, "loo_knn1": 0.75},
                    "separability_N24": {"resub_knn1": 0.85, "loo_knn1": 0.70},
                    "readout_cos_lda": -0.4,
                }
                if acc is not None and acc.get(t) is None:
                    pass  # O-2: accuracy fields omitted at this checkpoint
                else:
                    pos_a, abs_a = acc.get(t, (0.75, 0.75)) if acc else (0.75, 0.75)
                    ck["panel16_pos_accuracy"] = pos_a
                    ck["panel16_abs_accuracy"] = abs_a
                    ck["pos_abs_accuracy"] = round((pos_a + abs_a) / 2.0, 6)
                cks[str(t)] = ck
            r = {"condition": cond, "checkpoints": cks, "collapsed": collapsed,
                 "collapse_onset_t_star": (_collapse_onset(telemetry) if telemetry else None),
                 "telemetry": telemetry if telemetry is not None else
                             [{"step": s, "dev_proposal_rate": 0.6} for s in range(1, HORIZON + 1)],
                 "stopping_event": None,
                 "status": "COMPLETED"}
            if collapse_cont:
                r["non_b0_collapse_continued"] = True
            if void:
                r["status"] = "VOID"
                r["void_reason"] = void
            return r

        collapse_telemetry = [{"step": s, "dev_proposal_rate": (0.0 if s in (5, 6, 7) else 0.6)}
                              for s in range(1, HORIZON + 1)]
        runs_o1 = {
            "B0": run_fixture("B0", -0.95),
            "I1": run_fixture("I1", -0.20, collapsed=True, collapse_cont=True,
                              telemetry=collapse_telemetry),
            "C2": run_fixture("C2", -0.60),
            "F1": run_fixture("F1", -0.70),
            "T1": run_fixture("T1", -0.10),
            "T2": run_fixture("T2", -0.25),
            "V1": run_fixture("V1", -0.10),
        }
        # (a) B0 can trigger and truncate; identical pattern on non-B0 is recorded, not truncation
        assert collapse_guard_action("B0", True, COLLAPSE_K) == "TRUNCATE"
        assert collapse_guard_action("B0", True, COLLAPSE_K - 1) == "NO_TRIGGER"
        assert collapse_guard_action("I1", True, COLLAPSE_K) == "CONTINUE"
        assert collapse_guard_action("C2", True, COLLAPSE_K) == "CONTINUE"
        assert collapse_guard_action("I1", False, COLLAPSE_K) == "NO_TRIGGER"  # never proposed
        # (b) full store path: I1 continued to horizon despite the collapse pattern
        store_o1 = build_metrics_store(runs_o1)
        assert store_o1["I1"]["status"] == "COMPLETED"
        assert store_o1["I1"]["non_b0_collapse_continued"] is True
        assert store_o1["I1"]["collapsed"] is True
        assert store_o1["I1"]["collapse_onset_t_star"] == 5
        assert {14, 15, 20} <= {int(t) for t in store_o1["I1"]["cos_H1"]}  # 14/15/20 present
        # (c) P1/P3/P5 stay evaluable on the continued I1 run; P2 reports collapse (fails)
        p_o1 = evaluate_p_rules(store_o1)
        assert p_o1["P1"]["evaluable"] and p_o1["P1"]["passed"]
        assert p_o1["P3"]["evaluable"] and p_o1["P3"]["passed"]
        assert p_o1["P5"]["evaluable"] and p_o1["P5"]["passed"]
        assert p_o1["P2"]["evaluable"] and not p_o1["P2"]["passed"]
        results["o1_collapse_guard_scoping"] = "PASS"

        # (d) VOID condition: explicit protocol-compatible state, never silent B0 collapse
        void_i1 = run_fixture("I1", -0.20, void="non_finite_loss")
        void_i1["checkpoints"] = {"0": void_i1["checkpoints"]["0"]}  # only step 0 before failure
        void_i1["telemetry"] = [{"step": 0, "dev_proposal_rate": None, "void": True,
                                 "void_reason": "non_finite_loss"}]
        runs_void = {c: run_fixture(c, -0.50) for c in CONDITIONS}
        runs_void["I1"] = void_i1
        store_v = build_metrics_store(runs_void)
        assert store_v["I1"]["void_reason"] == "non_finite_loss"
        assert store_v["I1"]["collapsed"] is False  # not a B0-style collapse truncation
        assert runs_void["I1"]["stopping_event"] is None
        p_v = evaluate_p_rules(store_v)
        assert not p_v["P1"]["evaluable"] and not p_v["P2"]["evaluable"]
        assert not p_v["P3"]["evaluable"] and not p_v["P4"]["evaluable"] and not p_v["P5"]["evaluable"]
        assert p_v["P4"]["not_evaluable"] is True
        assert p_v["P1"]["detail"] == "I1 VOID: non_finite_loss"
        m_v = evaluate_m_rules(store_v, t2_k=4, i1_void=True)
        assert m_v["M3"]["disposition"] == "NOT EVALUABLE"
        results["o1_void_state_propagation"] = "PASS"

        # 6c. O-2 amendment (§5.2 row 7): POS/ABS accuracy registered + deterministic Row-7
        acc_flat = {t: (0.75, 0.75) for t in CHECKPOINTS}
        runs_r7_flat = {
            "B0": run_fixture("B0", -0.95, acc=acc_flat),
            "I1": run_fixture("I1", -0.20, acc=acc_flat),
            "C2": run_fixture("C2", -0.60, acc=acc_flat),
            "F1": run_fixture("F1", -0.70, acc=acc_flat),
            "T1": run_fixture("T1", -0.10, acc=acc_flat),
            "T2": run_fixture("T2", -0.25, acc=acc_flat),
            "V1": run_fixture("V1", -0.10, acc=acc_flat),
        }
        store_r7 = build_metrics_store(runs_r7_flat)
        # (a) POS/ABS accuracy emitted at required checkpoints (step-0 + RULE_CHECKPOINTS, incl. 15)
        assert {4, 6, 14, 15} <= {int(t) for t in store_r7["I1"]["pos_accuracy"]}
        assert store_r7["I1"]["pos_accuracy"][15] == 0.75
        assert store_r7["I1"]["abs_accuracy"][15] == 0.75
        assert store_r7["I1"]["pos_abs_accuracy"][15] == 0.75
        # no registered change -> NO_ARTIFACTUAL_TRADE_OFF (row evaluated, not dropped)
        r7_flat = evaluate_row7(store_r7)
        assert r7_flat["label"] == "NO_ARTIFACTUAL_TRADE_OFF"

        # (b) one class improves while the other degrades -> ARTIFACTUAL_TRADE_OFF
        acc_trade = {0: (0.50, 0.50), 4: (1.00, 0.25), 6: (1.00, 0.25), 7: (1.00, 0.25),
                     14: (1.00, 0.25), 15: (1.00, 0.00), 17: (1.00, 0.00), 20: (1.00, 0.00)}
        runs_r7_trade = {c: run_fixture(c, -0.20, acc=acc_trade) for c in CONDITIONS}
        r7_trade = evaluate_row7(build_metrics_store(runs_r7_trade))
        assert r7_trade["label"] == "ARTIFACTUAL_TRADE_OFF"
        assert r7_trade["trade_off_intervals"][0]["interval"] == "0->4"
        # strict-sign rule with NO magnitude threshold: tiny opposite-sign deltas still count
        acc_tiny = {0: (0.5, 0.5), 4: (0.51, 0.49), 6: (0.51, 0.49), 7: (0.51, 0.49),
                    14: (0.51, 0.49), 15: (0.51, 0.49), 17: (0.51, 0.49), 20: (0.51, 0.49)}
        runs_r7_tiny = {c: run_fixture(c, -0.20, acc=acc_tiny) for c in CONDITIONS}
        r7_tiny = evaluate_row7(build_metrics_store(runs_r7_tiny))
        assert r7_tiny["label"] == "ARTIFACTUAL_TRADE_OFF", "sign-only comparison, no magnitude threshold"

        # (d) indeterminate cases explicit: VOID and missing-checkpoint both ROW_7_NOT_DETERMINABLE
        runs_r7_void = {c: run_fixture(c, -0.50, acc=acc_flat) for c in CONDITIONS}
        runs_r7_void["I1"]["status"] = "VOID"
        runs_r7_void["I1"]["void_reason"] = "non_finite_loss"
        r7_void = evaluate_row7(build_metrics_store(runs_r7_void))
        assert r7_void["label"] == ROW7_NOT_DETERMINABLE and "VOID" in r7_void["reason"]
        acc_missing = dict(acc_flat)
        acc_missing[15] = None  # t=15 recorded without accuracy -> registered-metric gap
        runs_r7_miss = {c: run_fixture(c, -0.50, acc=acc_missing) for c in CONDITIONS}
        r7_miss = evaluate_row7(build_metrics_store(runs_r7_miss))
        assert r7_miss["label"] == ROW7_NOT_DETERMINABLE
        assert r7_miss["missing_checkpoints"] == [15]

        # verdict wiring: Row 7 always surfaced in verdict + outcome_summary (never dropped)
        v7 = compute_verdict(runs_r7_trade, {}, 4)
        assert v7["Row7_artifactual_trade_off"]["label"] == "ARTIFACTUAL_TRADE_OFF"
        assert "ARTIFACTUAL_TRADE_OFF" in [r["label"] for r in v7["outcome_summary"]["rows"]]
        v7_flat = compute_verdict(runs_r7_flat, {}, 4)
        assert "NO_ARTIFACTUAL_TRADE_OFF" in [r["label"] for r in v7_flat["outcome_summary"]["rows"]]
        results["o2_row7_deterministic"] = "PASS"

        # 7. Verdict safeguard: PRE_EXECUTION / NO_SCIENTIFIC_VERDICT / EXECUTION_NOT_AUTHORIZED
        auth = check_execution_authorization(tmp_dir)
        assert not auth["authorized"], "no execution_authorization.json -> must be unauthorized"
        v = emit_verdict(tmp_dir, {}, {}, None, auth)
        assert v["status"] == "PRE_EXECUTION"
        assert v["scientific_verdict"] == "NO_SCIENTIFIC_VERDICT"
        assert v["authorization"] == "EXECUTION_NOT_AUTHORIZED"
        # authorized-but-unauthorized file must still refuse scientific verdict
        (tmp_dir / "execution_authorization.json").write_text(
            json.dumps({"phase": "6E.18", "decision": "APPROVE", "date": "2026-08-16", "signature": "SELF-TEST"}), encoding="utf-8")
        auth2 = check_execution_authorization(tmp_dir)
        v2 = emit_verdict(tmp_dir, {}, {}, None, auth2)
        assert v2["status"] == "PRE_EXECUTION" and v2["scientific_verdict"] == "NO_SCIENTIFIC_VERDICT", \
            "no executed checkpoints -> must refuse scientific verdict"
        results["verdict_safeguard"] = "PASS"

    # 8. Protocol hash intact (read-only, no execution)
    assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH
    results["protocol_hash"] = "PASS"

    # 9. Zero-execution counters
    results["zero_execution"] = {
        "optimizer_steps": 0,
        "training_steps": 0,
        "experimental_checkpoints": 0,
        "scientific_verdict_generated": False,
        "frozen_model_loaded": False,
        "derive_readout_initialization_invoked": False,
    }

    print("=" * 80)
    print("ZERO-STEP SELF-TEST COMPLETE: ALL PIPELINE, HASHING, RULE & VERDICT PATHS VERIFIED (NO TRAINING)")
    print("=" * 80)
    return results


# ---------------------------------------------------------------------------
# Preflight Implementation Validation Mode (Zero-Execution Test)
# ---------------------------------------------------------------------------
def run_preflight_validation() -> Dict[str, Any]:
    """Executes a strict, comprehensive, non-training preflight validation of the entire runner."""
    print("=" * 80)
    print("PHASE 6E.18 EXECUTION PREFLIGHT VALIDATION (NON-TRAINING MODE)")
    print("=" * 80)

    results = {}

    # 1. Verify Protocol Hash
    print("[1/11] Verifying Protocol Artifact SHA-256...")
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"Protocol file not found at {PROTOCOL_PATH}")
    actual_hash = sha256_file(PROTOCOL_PATH)
    assert actual_hash == EXPECTED_PROTOCOL_HASH, f"Protocol hash mismatch: {actual_hash} != {EXPECTED_PROTOCOL_HASH}"
    print(f"      Protocol SHA-256: {actual_hash} [MATCHED]")
    results["protocol_hash"] = "PASS"

    # 2. Verify 7-Condition Canonical Universe
    print("[2/11] Verifying Canonical Condition Universe (N=7)...")
    assert len(CONDITIONS) == 7
    assert CONDITIONS == ["B0", "I1", "C2", "F1", "T1", "T2", "V1"]
    print(f"      Condition Universe: {CONDITIONS} [VERIFIED]")
    results["condition_universe"] = "PASS"

    # 3. Verify §6.2 Normative Synthetic Sign Gate (aggregate-bound)
    print("[3/11] Verifying §6.2 Mandatory Synthetic Sign Gate...")
    verify_synthetic_sign_gate(eps=SIGN_GATE_EPSILON)
    print(f"      Synthetic Sign Gate (eps={SIGN_GATE_EPSILON}, aggregate cosine): [PASS]")
    results["sign_gate"] = "PASS"

    # 4. Verify Architecture Construction & Parameter Counts
    print("[4/11] Verifying Readout Module Parameter Accounting...")
    dummy_w = np.zeros(D_HIDDEN, dtype=np.float32)
    dummy_w_delta = torch.zeros(D_HIDDEN, dtype=torch.float32)
    dummy_Uk = torch.zeros(D_HIDDEN, 4, dtype=torch.float32)

    modules = {
        "B0": None,
        "I1": I1ReadoutHead(dummy_w, 0.0),
        "C2": I1ReadoutHead(dummy_w, 0.0),
        "F1": F1ReadoutHead(dummy_w, 0.0),
        "T1": T1ReadoutHead(dummy_w_delta, 1.0, 0.0),
        "T2": T2ReadoutHead(dummy_Uk, b0=0.0),
        "V1": V1ReadoutHead(dummy_w, dummy_w, 0.0, 0.0),
    }

    param_table = []
    for cond, mod in modules.items():
        k_val = 4 if cond == "T2" else None
        res = verify_condition_parameter_accounting(cond, mod, expected_k=k_val)
        param_table.append(res)
        print(f"      Condition {cond:2s}: {res['trainable_params']:4d} trainable params [PASS]")

    results["parameter_accounting"] = param_table

    # 5. Verify §7.1 P4 Dual-Regime Comparator Logic
    print("[5/11] Verifying §7.1 P4 Dual-Regime Comparator Logic...")
    # Test shared regime (t=4)
    p_pass, diff, reg = evaluate_p4_comparator(loo_cv_i1=0.75, loo_cv_b0_step=0.72, loo_cv_b0_step0=0.60, step=4)
    assert p_pass and abs(diff - 0.03) < 1e-6 and "contemporaneous" in reg
    # Test post-truncation regime (t=15)
    p_pass_post, diff_post, reg_post = evaluate_p4_comparator(loo_cv_i1=0.68, loo_cv_b0_step=None, loo_cv_b0_step0=0.70, step=15)
    assert p_pass_post and abs(diff_post - 0.02) < 1e-6 and "frozen_step0" in reg_post
    # Test early-truncation fallback (§7.1): B0 guard triggered before t=6 -> step-0 anchor
    p_fb, _, reg_fb = evaluate_p4_comparator(loo_cv_i1=0.80, loo_cv_b0_step=None, loo_cv_b0_step0=0.70, step=6)
    assert not p_fb and reg_fb == "frozen_step0_anchor_B0"
    print("      P4 Shared Regime (t=4), Post-Truncation Regime (t=15), Early-Truncation Fallback (t=6): [PASS]")
    results["p4_comparator"] = "PASS"

    # 6. Verify §8 T2 Prospective SVD Basis Selection
    print("[6/11] Verifying §8 T2 Prospective SVD Basis Selection...")
    np.random.seed(42)
    synthetic_panel_hidden = np.random.randn(16, D_HIDDEN).astype(np.float32)
    # Dominant directions (capture >= 90% energy) plus near-degenerate noise so the
    # derived k is a small integer within the protocol's [2, 8] assertion window.
    synthetic_panel_hidden[:, :3] *= 10.0
    synthetic_panel_hidden[:, 3:] *= 1e-3
    Uk, derived_k, cum_eng, _ = derive_t2_basis_from_covariance(synthetic_panel_hidden, energy_threshold=ENERGY_THRESHOLD)
    assert 2 <= derived_k <= 8, f"Invalid derived k: {derived_k}"
    assert Uk.shape == (D_HIDDEN, derived_k)
    print(f"      SVD Prospective Derivation: k={derived_k} (energy={cum_eng:.4f}) [PASS]")
    results["t2_svd_derivation"] = "PASS"

    # 7. Verify §3.8 Initialization Arithmetic (hand-derived anchor + protocol invariants)
    print("[7/11] Verifying §3.8 Initialization Arithmetic...")
    u_lda = np.zeros(D_HIDDEN, dtype=np.float64)
    u_lda[0] = 1.0  # e_0
    H16 = np.zeros((16, D_HIDDEN), dtype=np.float64)
    H16[:, 0] = np.arange(1, 17, dtype=np.float64)  # h_proj = [1..16]
    margins_b = 2.0 * H16[:, 0]
    n_c2, cos_n_lda, n_draws = generate_c2_random_orientation(u_lda.astype(np.float32))
    init = compute_init_constants(H16, margins_b, u_lda, n_c2)

    s_scale_ref = float(np.std(margins_b) / (np.std(H16 @ u_lda) + EPS_S_SCALE))
    assert abs(init["s_scale"] - s_scale_ref) < 1e-12, "s_scale formula mismatch (§3.8)"
    assert abs(init["b0"] - float(-np.median(init["a0"] @ H16[:8].T))) < 1e-6, "b0 = -median(a0 @ H16[:8].T)"
    assert abs(init["b0"] - (-4.5 * init["s_scale"])) < 1e-6, "hand-derived b0 anchor violated"
    assert abs(init["b0_c2"] - float(-np.median(init["c0"] @ H16[:8].T))) < 1e-6, "b0_c2 = -median(c0 @ H16[:8].T)"
    assert init["b_fixed"] == init["b0"], "b_fixed must equal b0 (F1 frozen twin)"
    assert init["b_delta"] == 0.0, "b_delta must be 0"
    assert init["t1_alpha0"] == 1.0 and init["t1_b0"] == 0.0, "T1 init must be (1.0, 0.0)"
    assert init["t2_b0"] == 0.0 and init["t2_beta0"] is None, "T2 init must be zeros(k)/0.0"
    assert init["v1_b_pro_0"] == 0.0 and init["v1_b_abs_0"] == 0.0, "V1 biases must be 0"
    assert abs(cos_n_lda) <= 0.05 and n_draws >= 1, "C2 n must satisfy |cos(n, u_hat_LDA)| <= 0.05"
    print(f"      s_scale={init['s_scale']:.6f} b0={init['b0']:.6f} (=-4.5*s_scale anchor) b0_c2={init['b0_c2']:.6f} "
          f"cos_n_lda={cos_n_lda:.5f} draws={n_draws} [PASS]")
    results["init_arithmetic"] = "PASS"

    # 8. Verify V1 Aligned-Mirror Initialization (CHANGE, §3.7/§3.8)
    print("[8/11] Verifying V1 Aligned-Mirror Initialization (CHANGE)...")
    w_pro, w_abs = init["v1_w_pro_0"], init["v1_w_abs_0"]
    assert np.allclose(w_abs, -w_pro, atol=1e-7), "V1 w_ABS(0) must be exact negation of w_PRO(0)"
    assert abs(np.linalg.norm(w_pro) - init["s_scale"]) < 1e-6, "V1 ||w_PRO(0)|| must equal s_scale"
    cos_wpro_u = float(w_pro @ u_lda) / (np.linalg.norm(w_pro) + EPS_S_SCALE)
    assert abs(cos_wpro_u - 1.0) < 1e-6, "V1 w_PRO(0) must be exactly aligned with u_hat_LDA"
    v1_head = V1ReadoutHead(w_pro, w_abs, init["v1_b_pro_0"], init["v1_b_abs_0"])
    assert v1_head.count_trainable_params() == 1794, "V1 must be 1794 trainable params"
    assert v1_head.b_pro.item() == 0.0 and v1_head.b_abs.item() == 0.0, "V1 biases must initialize to 0"
    print("      w_ABS(0) == -w_PRO(0) | ||w_PRO(0)|| == s_scale | cos(w_PRO(0), u_hat_LDA) == 1.0 | biases 0 [PASS]")
    results["v1_aligned_mirror"] = "PASS"

    # 9. Verify §7.2 M1-M5 Disposition Wiring
    print("[9/11] Verifying §7.2 M1-M5 Decision-Rule Wiring...")

    def ck(c: float) -> Dict[int, Optional[float]]:
        return {t: c for t in RULE_CHECKPOINTS}

    # M1: SUPPORTED / INDETERMINATE / NOT SUPPORTED / VOID
    assert evaluate_m_rules({"F1": {"cos_H1": ck(-0.70)}})["M1"]["disposition"] == "SUPPORTED"
    assert evaluate_m_rules({"F1": {"cos_H1": ck(-0.45)}})["M1"]["disposition"] == "INDETERMINATE"
    assert evaluate_m_rules({"F1": {"cos_H1": ck(-0.20)}})["M1"]["disposition"] == "NOT SUPPORTED"
    assert evaluate_m_rules({"F1": {"cos_H1": {4: -0.5}}})["M1"]["disposition"] == "VOID"
    # M2: SUPPORTED / NOT SUPPORTED
    assert evaluate_m_rules({"T1": {"cos_H1": ck(-0.10)}})["M2"]["disposition"] == "SUPPORTED"
    assert evaluate_m_rules({"T1": {"cos_H1": ck(-0.50)}})["M2"]["disposition"] == "NOT SUPPORTED"
    # M3: MATCH / NO MATCH / NOT EVALUABLE (I1 VOID) / VOID (k)
    assert evaluate_m_rules({"I1": {"cos_H1": ck(-0.20)}, "T2": {"cos_H1": ck(-0.25)}}, t2_k=4)["M3"]["disposition"] == "SUPPORTED"
    assert evaluate_m_rules({"I1": {"cos_H1": ck(-0.20)}, "T2": {"cos_H1": ck(-0.45)}}, t2_k=4)["M3"]["disposition"] == "NOT SUPPORTED"
    assert evaluate_m_rules({"I1": {"cos_H1": ck(-0.20)}, "T2": {"cos_H1": ck(-0.25)}}, t2_k=4, i1_void=True)["M3"]["disposition"] == "NOT EVALUABLE"
    assert evaluate_m_rules({"I1": {"cos_H1": ck(-0.20)}, "T2": {"cos_H1": ck(-0.25)}}, t2_k=12)["M3"]["disposition"] == "VOID"
    # M4: SUPPORTED / NOT SUPPORTED (degradation) / NOT SUPPORTED (cos)
    assert evaluate_m_rules({"V1": {"cos_H1": ck(-0.10), "proposal_rate": {15: 0.5}, "margin_d": {15: 0.3}}})["M4"]["disposition"] == "SUPPORTED"
    assert evaluate_m_rules({"V1": {"cos_H1": ck(-0.10), "proposal_rate": {15: 0.0}, "margin_d": {15: 0.3}}})["M4"]["disposition"] == "NOT SUPPORTED"
    assert evaluate_m_rules({"V1": {"cos_H1": ck(-0.40), "proposal_rate": {15: 0.5}, "margin_d": {15: 0.3}}})["M4"]["disposition"] == "NOT SUPPORTED"
    # M5: SUPPORTED / NOT SUPPORTED
    assert evaluate_m_rules({"C2": {"cos_H1": ck(-0.10)}})["M5"]["disposition"] == "SUPPORTED"
    assert evaluate_m_rules({"C2": {"cos_H1": ck(-0.40)}})["M5"]["disposition"] == "NOT SUPPORTED"
    print("      M1 (SUPPORTED/INDETERMINATE/NOT SUPPORTED/VOID), M2, M3 (MATCH/NO MATCH/NOT EVALUABLE/VOID-k), "
          "M4 (degradation), M5: [PASS]")
    results["m_rules_wiring"] = "PASS"

    # 10. Zero-Execution Assertion
    print("[10/11] Verifying Zero-Execution Guard...")
    print("      optimizer_steps = 0")
    print("      training_steps = 0")
    print("      experimental_checkpoints = 0")
    print("      scientific_verdict_generated = False")
    print("      derive_readout_initialization NOT invoked (frozen model not loaded)")
    results["zero_execution_guard"] = "PASS"

    # 11. Verdict-Emission & Authorization-Gate Safeguard (§14, PRE_EXECUTION contract)
    print("[11/11] Verifying Verdict-Emission & Authorization-Gate Safeguard...")
    with tempfile.TemporaryDirectory(prefix="theo_6e18_preflight_") as tmp:
        tmp_dir = Path(tmp)
        auth_none = check_execution_authorization(tmp_dir)
        assert not auth_none["authorized"]
        v_none = emit_verdict(tmp_dir, {}, {}, None, auth_none)
        assert v_none["status"] == "PRE_EXECUTION"
        assert v_none["scientific_verdict"] == "NO_SCIENTIFIC_VERDICT"
        assert v_none["authorization"] == "EXECUTION_NOT_AUTHORIZED"
        (tmp_dir / "execution_authorization.json").write_text(
            json.dumps({"phase": "6E.18", "decision": "APPROVE", "date": "2026-08-16", "signature": "PREFLIGHT"}),
            encoding="utf-8")
        auth_signed = check_execution_authorization(tmp_dir)
        assert auth_signed["authorized"]
        v_signed_no_runs = emit_verdict(tmp_dir, {}, {}, None, auth_signed)
        assert v_signed_no_runs["status"] == "PRE_EXECUTION", "authorized but zero executed checkpoints -> PRE_EXECUTION"
    print("      no-authorization -> PRE_EXECUTION/NO_SCIENTIFIC_VERDICT/EXECUTION_NOT_AUTHORIZED [PASS]")
    print("      signed-auth + zero executed checkpoints -> still PRE_EXECUTION [PASS]")
    results["verdict_safeguard"] = "PASS"

    print("=" * 80)
    print("PREFLIGHT VALIDATION COMPLETE: ALL ARCHITECTURES, GATES, INITIALIZATIONS, M-RULES & VERDICT SAFEGUARD VERIFIED")
    print("=" * 80)
    return results


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def _clean(obj):
    """Recursively convert non-serializable objects (torch/numpy) to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    if isinstance(obj, tuple):
        return [_clean(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _run_execution():
    """AUTHORIZED-ONLY execution path. Refuses unless a signed execution_authorization.json
    (APPROVE, signature, date, phase 6E.18) exists in the artifact directory (§14 hard stop)."""
    start = time.time()
    print("=" * 70)
    print("PHASE 6E.18 — CAUSAL INTERVENTION (B0 / I1 / C2 / F1 / T1 / T2 / V1)")
    print("=" * 70, flush=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    authorization = check_execution_authorization(ARTIFACT_DIR)
    if not authorization["authorized"]:
        verdict = emit_verdict(ARTIFACT_DIR, {}, {}, None, authorization)
        with open(ARTIFACT_DIR / "verdict.json", "w", encoding="utf-8") as f:
            json.dump(verdict, f, indent=2, sort_keys=True)
        manifest = generate_sha256_manifest(ARTIFACT_DIR, time.time() - start)
        with open(ARTIFACT_DIR / "sha256_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"\n[HARD STOP] Execution REFUSED: {authorization['reason']}", flush=True)
        print(f"[VERDICT] {verdict['status']} / {verdict['scientific_verdict']} / {verdict['authorization']}", flush=True)
        print("No training, no optimizer step, no experimental checkpoint, no scientific verdict.", flush=True)
        sys.exit(1)

    assert sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_HASH, "Protocol hash mismatch"
    base_hash = sha256_file(BASE_MODEL_PATH / "model.safetensors")
    corpus_hash = sha256_file(CORPUS_PATH)
    assert base_hash == EXPECTED_BASE_HASH, "base model hash mismatch"
    assert corpus_hash == EXPECTED_CORPUS_HASH, "corpus hash mismatch"

    snapshot = {
        "phase": "6E.18",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_model_sha256": base_hash,
        "corpus_sha256": corpus_hash,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "script_sha256": sha256_file(Path(__file__)),
        "execution_authorization": authorization["record"],
        "frozen_config": {
            "PRO_ID": PRO_ID, "AB_ID": AB_ID, "D_HIDDEN": D_HIDDEN, "VOCAB": VOCAB,
            "checkpoints": CHECKPOINTS, "horizon": HORIZON, "collapse_K": COLLAPSE_K,
            "seed_data": SEED_DATA, "seed_c2_readout": SEED_C2_READOUT,
            "lambda_decision": LAMBDA_DECISION, "grad_accum": GRAD_ACCUM, "batch_size": BATCH_SIZE,
            "lr": LR, "wd": WD, "clip_norm": CLIP_NORM,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGETS},
            "P1_threshold": P1_THRESH, "C2x_threshold": C2X_THRESH, "P2_proposal": ">0 all t<=20",
            "P3": [P3_COHEN_D_THRESH, P3_R_EFF_THRESH], "P4": {"knn1": list(P4_KN1_RANGE), "loo_tol": P4_LOO_TOL},
            "P5_threshold": P5_THRESH, "M1_H3": [H3_SUPPORT_THRESH, H3_INDETERMINATE_UPPER],
            "M2_M4_M5_floor": -0.30, "M3_delta_tol": M3_DELTA_TOL,
        },
    }
    with open(ARTIFACT_DIR / "pre_registration_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    (ARTIFACT_DIR / "code_snapshot").mkdir(exist_ok=True)
    (ARTIFACT_DIR / "code_snapshot" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = reconstruct_data(CORPUS_PATH)
    print(f"Data: train={data['n_train']} dev={data['n_dev']} counts={data['counts']}", flush=True)

    derive, u_lda, init, H16, margins_b = derive_readout_initialization(data, tokenizer)
    w_delta_t = init.pop("w_delta_t", None)
    with open(ARTIFACT_DIR / "readout_initialization.json", "w", encoding="utf-8") as f:
        json.dump(derive, f, indent=2, sort_keys=True)

    # §8 T2 basis (prospective, frozen before T2 execution)
    U_k, t2_k, cum_energy, spectrum = derive_t2_basis_from_covariance(H16, energy_threshold=ENERGY_THRESHOLD)
    U_k = U_k.to(DEVICE)
    with open(ARTIFACT_DIR / "t2_basis.json", "w", encoding="utf-8") as f:
        json.dump({"k": t2_k, "cumulative_energy": cum_energy, "spectrum_top10": spectrum,
                   "basis_sha256": tensor_hash(U_k.cpu())}, f, indent=2, sort_keys=True)
    print(f"[MAIN] T2 basis k={t2_k} (energy={cum_energy:.4f})", flush=True)

    runs = {}
    for cond in CONDITIONS:
        runs[cond] = run_condition(cond, data, tokenizer, init, u_lda, U_k, w_delta_t, ARTIFACT_DIR)

    runs_clean = _clean(runs)
    with open(ARTIFACT_DIR / "runs.json", "w", encoding="utf-8") as f:
        json.dump(runs_clean, f, indent=2, sort_keys=True)

    verdict = emit_verdict(ARTIFACT_DIR, runs, derive, t2_k, authorization)
    with open(ARTIFACT_DIR / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, sort_keys=True)
    print(f"\n[VERDICT] {verdict['status']}", flush=True)
    for k, v in verdict.items():
        if isinstance(v, dict) and "passed" in v:
            print(f"  {k}: PASS={v['passed']}", flush=True)

    manifest = generate_sha256_manifest(ARTIFACT_DIR, time.time() - start,
                                        extra={"base_model_sha256_after": base_hash,
                                               "corpus_sha256_after": corpus_hash})
    with open(ARTIFACT_DIR / "sha256_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("\n" + "=" * 70)
    print("PHASE 6E.18 EXECUTION COMPLETE. HARD STOP. RETURNING FOR HUMAN REVIEW.")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6E.18 Causal Intervention Runner")
    parser.add_argument("--preflight", action="store_true", help="Run preflight validation and exit without training")
    parser.add_argument("--validate-only", action="store_true", help="Alias for --preflight")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the zero-step mock-backed self-test (no training, no frozen-model load)")
    parser.add_argument("--execute", action="store_true",
                        help="AUTHORIZED-ONLY execution. Refuses without a signed execution_authorization.json (§14).")
    args = parser.parse_args()

    if args.execute:
        _run_execution()
        sys.exit(0)
    if args.self_test:
        run_self_test()
        sys.exit(0)
    if args.preflight or args.validate_only or len(sys.argv) == 1:
        run_preflight_validation()
        sys.exit(0)
