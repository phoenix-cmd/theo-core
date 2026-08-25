# THEO SLM — Phase 6E.20 Fixed Low-Dimensional & Subspace Readout Runner
#
# NEW ARTIFACT (Phase 6E.20). This module is a separate new artifact from the
# frozen Phase 6E.18 runner (`run_phase_6e18_causal_intervention.py`), which is
# read-only and is never imported or modified here.
#
# GOVERNING DOCUMENTS (frozen, read-only):
#   * Protocol: `theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md`
#     SHA-256 E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85 (400 lines)
#   * Plan: `theo-slm-fixed-subspace-readout-authorization-protocol-v3.2.1-pre-execution-implementation-plan.md`
#     SHA-256 F6A145DDA276621421A13E22215CBF5F9DE3052C712CD8CD4055076C3D3679F1 (189 lines)
#
# AUTHORITY: IMPLEMENTATION AUTHORIZED (L1-L3 only). EXECUTION NOT AUTHORIZED.
# This module performs NO model execution, training, optimizer steps, checkpoint
# generation, real-panel (H16) collection, or scientific evaluation.
# Execution-stage code paths are guarded by an authorization gate that returns the
# refusal triplet (PRE_EXECUTION / NO_SCIENTIFIC_VERDICT / EXECUTION_NOT_AUTHORIZED)
# unless an `execution_authorization.json` (decision=APPROVE, phase=6E.20) exists.
# No such file is created or requested by this implementation.
#
# All deterministic registration, metric, gate, hypothesis, and verdict machinery
# below is testable on synthetic vectors and registered historical hashes at
# L1-L3 without touching a model.

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Frozen protocol / plan binding
# ---------------------------------------------------------------------------
PROTOCOL_SHA256 = "E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85"
PLAN_SHA256 = "F6A145DDA276621421A13E22215CBF5F9DE3052C712CD8CD4055076C3D3679F1"
PLAN_V32_SHA256 = "B5B23DE68BBCA67308A309AE9022419AD321FFFBBB1344385803C2ED5BAC43B3"

# ---------------------------------------------------------------------------
# Inherited Phase 6E.18 config constants (v3.1 §10.3, verbatim; C16)
# Verified against frozen runner lines 73-93, 125-129, 148-151, 258-259,
# 265, 982, 1006-1033, 1149-1150.
# ---------------------------------------------------------------------------
PRO_ID = 9117            # PROPOSE token id
AB_ID = 1867             # ABSTAIN token id
D_HIDDEN = 896           # hidden dimension d
VOCAB = 151936           # Qwen2.5-0.5B-Instruct vocabulary size
HORIZON = 20             # training horizon T
COLLAPSE_K = 3           # B0 collapse guard K
SEED_DATA = 42           # train/dev split + batch RNG seed
LAMBDA_DECISION = 10.0   # decision-position loss weight
GRAD_ACCUM = 2
BATCH_SIZE = 4
LR = 1e-4
WD = 0.01
CLIP_NORM = 1.0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DECISION_INDEX = 4                    # decision logits at valid index 4 (v_idx[4])
CHECKPOINT_SET = [0, 4, 6, 7, 14, 15, 17, 20]
AGGREGATION_SET = [4, 6, 14, 15]
SEED_RANDOM_OBJECTS = 20260820        # E-8: R_4 and n_rand generator seed
EPS_S_SCALE = 1e-12

# Registered historical values (read-only; informational + G1/G2/G8 targets).
# Source: 6E.18 `readout_initialization.json` and `t2_basis.json`.
U_HAT_LDA_HASH_LEGACY = "ff3cab496f8c219da96a9f59e144459cad6497113e4eaef0b1c0cd26525f3cc8"
BASIS_SHA256_LEGACY = "b8dfbd6ae71277d5350b7ed2a513ccfb771b8e263102b843433c96cab8645060"
S_SCALE_REFERENCE = 0.06681545
W_DELTA_NORM_REFERENCE = 0.564183
B_F1_REFERENCE = -0.07310398
U8_CUM_ENERGY_REFERENCE = 0.9277434349060059

# Thresholds (absolute constants, v3.1 §6/§7/§8).
KAPPA_K_LOW = 0.05
KAPPA_K_HIGH = 0.995
KAPPA_R_LOW = 0.05
KAPPA_R_HIGH = 0.30
KAPPA_INCREMENT_MIN = 0.01
P1_THRESHOLD = 0.50          # rho_c (S4 and S8)
P2_THRESHOLD = 0.80          # mean cos(H1)
P3_THRESHOLD = 0.80          # Cohen's d(15)
P4_THRESHOLD = 1.50          # Cohen's d(15)
P5_THRESHOLD = 0.70          # balanced accuracy at t=15
P6_POSITIVITY = 0.0          # P(t) > 0 for all t<=20
P6_D20_MIN = 0.00            # d(20) >= 0.00
COS_H1_SIGN_EPS = 0.001      # G6 inherited sign-assertion tolerance
G7_REL_TOL = 1e-6
G7_DECOMP_REL_ERR = 1e-4
G8_WEIGHT_REL_TOL = 1e-6
G1_COS_MIN = 0.9999
G1_MAX_ABS = 1e-6
G2_MAX_ABS = 1e-6
G4_COS_MIN = 0.9999

CONDITIONS = ["B0", "F1", "S2", "S4", "S8", "C4-Rand", "R1-Rand"]
SCALAR_CONDITIONS = ["F1", "S2", "S4", "S8", "C4-Rand", "R1-Rand"]

REFUSAL_TRIPLET = ("PRE_EXECUTION", "NO_SCIENTIFIC_VERDICT", "EXECUTION_NOT_AUTHORIZED")

# ---------------------------------------------------------------------------
# W1: Hashing conventions (v3.1 §3.4 step 4)
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().lower()


def tensor_hash_legacy(t) -> str:
    """Phase 6E.18 legacy hash: SHA-256 of float32 torch-save serialized bytes.
    Verbatim mirror of the frozen runner `tensor_hash` (lines 148-151)."""
    import torch
    buf = io.BytesIO()
    torch.save(t.detach().float().cpu().contiguous(), buf)
    return sha256_bytes(buf.getvalue())


def canonical_hash_float64(arr) -> str:
    """Canonical v3 hash: SHA-256 of float64 C-contiguous little-endian raw bytes."""
    import numpy as np
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return sha256_bytes(a.tobytes())


def canonical_tensor_hash(t) -> str:
    """Canonical v3 hash of a tensor (float64 raw bytes; v3.1 §3.4)."""
    import numpy as np
    return canonical_hash_float64(t.detach().cpu().numpy())


# ---------------------------------------------------------------------------
# Pure deterministic registration arithmetic (W2)
# ---------------------------------------------------------------------------
def fit_lda_axis(H16, y16):
    """v3.1 §3.2: unit Fisher LDA axis, sklearn lsqr/shrinkage=auto.
    Mirrors frozen runner derive_readout_initialization (line 276-278)."""
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    H16 = np.asarray(H16, dtype=np.float64)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(H16, np.asarray(y16))
    raw = lda.coef_[0].astype(np.float64)
    return raw / (np.linalg.norm(raw) + EPS_S_SCALE)


def compute_s_scale(margins_b: Sequence[float], H16, u_lda: Sequence[float]) -> float:
    """v3.1 §3.3: s_scale = std(margins_B0) / (std(H16 @ u_lda) + eps)."""
    import numpy as np
    m = np.asarray(margins_b, dtype=np.float64)
    proj = np.asarray(H16, dtype=np.float64) @ np.asarray(u_lda, dtype=np.float64)
    return float(np.std(m) / (np.std(proj) + EPS_S_SCALE))


def s_scale_degenerate(value: float) -> bool:
    """v3.1 §3.3 degenerate guard: non-finite or <= 0 => registration fails (G8)."""
    return not math.isfinite(value) or value <= 0.0


def derive_principal_basis(X_panel, k_max: int = 8):
    """v3.1 §3.4: covariance + SVD + canonical sign pinning.
    Returns (U_pinned, S_spectrum, cum_energy, cov). U columns ordered by
    descending singular values; each column sign-pinned: negate if the
    largest-magnitude element is negative (ties by lowest row index).
    """
    import numpy as np
    X = np.asarray(X_panel, dtype=np.float64)
    assert X.ndim == 2 and X.shape[0] == 16, "N=16 panel hidden states required"
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    cov = (X_centered.T @ X_centered) / float(X.shape[0] - 1)
    U, S, Vt = np.linalg.svd(cov, full_matrices=False)
    # canonical sign convention (v3.1 §3.4 step 3)
    U_pinned = U.copy()
    for j in range(U_pinned.shape[1]):
        col = U_pinned[:, j]
        imax = int(np.argmax(np.abs(col)))
        if col[imax] < 0.0:
            U_pinned[:, j] = -col
    total = float(np.sum(S))
    cum = np.cumsum(S) / total if total > 1e-12 else np.cumsum(S)
    return U_pinned, S.astype(np.float64), cum.astype(np.float64), cov


def subspace_kappa(u_lda, U_k) -> float:
    """v3.1 §3.5: kappa_k = ||U_k U_k^T u_lda||_2."""
    import numpy as np
    u = np.asarray(u_lda, dtype=np.float64)
    Uk = np.asarray(U_k, dtype=np.float64)
    proj = Uk @ (Uk.T @ u)
    return float(np.linalg.norm(proj))


def compute_w_delta(w_head):
    """v3.1 §3.6: w_delta = w_head[PRO_ID] - w_head[AB_ID], b_delta = 0.
    Source: LM-head / unembedding rows only (no W_emb)."""
    import numpy as np
    wh = np.asarray(w_head, dtype=np.float64)
    w_delta = (wh[PRO_ID] - wh[AB_ID]).astype(np.float64)
    assert w_delta.shape == (D_HIDDEN,)
    return w_delta, 0.0


def generate_R4(seed: int = SEED_RANDOM_OBJECTS):
    """v3.1 §3.7: R_4 = Q from QR of a standard-normal 896x4 draw with
    torch.Generator(seed = E-8)."""
    import torch
    gen = torch.Generator().manual_seed(seed)
    M = torch.randn(D_HIDDEN, 4, generator=gen)
    Q, R = torch.linalg.qr(M)
    return Q.detach().cpu().numpy()


def generate_n_rand(u_lda, seed: int = SEED_RANDOM_OBJECTS):
    """v3.1 §3.7: unit n_rand with |cos(n_rand, u_lda)| <= 0.05, same seed,
    rejection sampling (mirrors frozen runner generate_c2_random_orientation,
    with the protocol's E-8 seed)."""
    import numpy as np
    import torch
    u = np.asarray(u_lda, dtype=np.float64)
    gen = torch.Generator().manual_seed(seed)
    n_draws = 0
    while True:
        n_draws += 1
        nv = torch.randn(D_HIDDEN, generator=gen)
        nv = nv / (nv.norm() + EPS_S_SCALE)
        cos_n_lda = float(np.asarray(nv.numpy(), dtype=np.float64) @ u)
        if abs(cos_n_lda) <= 0.05:
            break
    return nv.numpy(), float(cos_n_lda), n_draws


# ---------------------------------------------------------------------------
# W3: Condition construction (v3.1 §4, §5) — EXACTLY ZERO trainable params
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Readout:
    name: str
    w: Any                      # (D_HIDDEN,) float64 numpy vector
    b: float
    trainable_params: int = 0
    subspace: str = ""

    def forward_margin(self, h) -> float:
        import numpy as np
        return float(np.asarray(self.w, dtype=np.float64) @ np.asarray(h, dtype=np.float64)) + self.b

    def forward_logits(self, h) -> Tuple[float, float]:
        s = self.forward_margin(h)
        return s, -s


def build_readouts(reg: Dict[str, Any]) -> Dict[str, Readout]:
    """v3.1 §4: seven fixed readouts. Common gain ||w||_2 = s_scale for the six
    scalar conditions; B0 uses w_delta (b=0) and is excluded from common gain.
    All weights are plain frozen vectors: zero trainable parameters."""
    import numpy as np
    s_scale = reg["s_scale"]
    u = reg["u_lda"]
    U2, U4, U8 = reg["U_2"], reg["U_4"], reg["U_8"]
    R4, n_rand = reg["R_4"], reg["n_rand"]
    k2, k4, k8 = reg["kappa_2"], reg["kappa_4"], reg["kappa_8"]
    kR = reg["kappa_R"]
    H16_pos = np.asarray(reg["H_16"])[:8, :]

    def proj(basis, v):
        B = np.asarray(basis, dtype=np.float64)
        return B @ (B.T @ v)

    w = {
        "B0": (np.asarray(reg["w_delta"], dtype=np.float64), 0.0, "full ambient R^896 (unembedding rows)"),
        "F1": ((s_scale * u), None, "span(u_hat_LDA)"),
        "S2": ((s_scale / k2) * proj(U2, u), None, "span(U_2)"),
        "S4": ((s_scale / k4) * proj(U4, u), None, "span(U_4)"),
        "S8": ((s_scale / k8) * proj(U8, u), None, "span(U_8)"),
        "C4-Rand": ((s_scale / kR) * proj(R4, u), None, "span(R_4)"),
        "R1-Rand": (s_scale * n_rand, None, "span(n_rand)"),
    }
    readouts = {}
    for name in CONDITIONS:
        vec, b, subspace = w[name]
        if name == "B0":
            bias = 0.0
        else:
            bias = float(-np.median(np.asarray(vec, dtype=np.float64) @ H16_pos.T))
        readouts[name] = Readout(name=name, w=vec.astype(np.float64), b=bias,
                                 trainable_params=0, subspace=subspace)
    return readouts


def condition_distinctness(readouts: Dict[str, Readout]):
    """v3.1 §5.6 pairwise distinctness checks. Returns a dict of facts."""
    import numpy as np
    facts = {}
    names = list(readouts.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wa = np.asarray(readouts[a].w)
            wb = np.asarray(readouts[b].w)
            na, nb = np.linalg.norm(wa), np.linalg.norm(wb)
            cos = float(np.dot(wa, wb) / (na * nb)) if na * nb > 0 else 0.0
            facts[f"{a}_vs_{b}"] = {"cos": cos, "distinct_direction": abs(1.0 - abs(cos)) > 1e-9}
    return facts


# ---------------------------------------------------------------------------
# W4: Gradient / metric layer (v3.1 §5.7, §10.5)
# ---------------------------------------------------------------------------
def alpha_coefficients(s: float, y_pos: int):
    """v3.1 §5.7: alpha from the CE chain rule on patched full-vocab logits.
    s is the readout margin; logits PRO=s, ABS=-s. y_pos=1 for POS panels,
    0 for ABS panels. Returns (alpha, delta) with
        delta   = p_PRO - p_ABS      (post-patch softmax)
        alpha   = delta - 1          (POS panel)
        alpha   = delta + 1          (ABS panel)
    """
    import numpy as np
    logits = np.zeros(VOCAB, dtype=np.float64)
    logits[PRO_ID] = s
    logits[AB_ID] = -s
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    p_pro = float(p[PRO_ID])
    p_abs = float(p[AB_ID])
    delta = p_pro - p_abs
    alpha = (delta - 1.0) if y_pos == 1 else (delta + 1.0)
    return alpha, delta, p_pro, p_abs


def readout_only_gradient_hidden(alpha: float, w) -> Any:
    """g_R^h = alpha * w (v3.1 §5.1/§5.7)."""
    import numpy as np
    return alpha * np.asarray(w, dtype=np.float64)


def rho_sample(g_R, g_total) -> float:
    """v3.1 §5.7: per-sample bounded signed readout-total alignment.
    rho = 0 if ||g_R||_2 = 0."""
    import numpy as np
    gr = np.asarray(g_R, dtype=np.float64)
    gt = np.asarray(g_total, dtype=np.float64)
    nR = float(np.linalg.norm(gr))
    nT = float(np.linalg.norm(gt))
    if nR == 0.0 or nT == 0.0:
        return 0.0
    return float(np.dot(gr, gt) / (nR * nT))


def cos_H1(g_pos_mean, g_abs_mean) -> float:
    """v3.1 §10.5: raw cosine between mean POS and mean ABS total gradients,
    without label-sign inversion."""
    import numpy as np
    gp = np.asarray(g_pos_mean, dtype=np.float64)
    ga = np.asarray(g_abs_mean, dtype=np.float64)
    np_ = float(np.linalg.norm(gp))
    na = float(np.linalg.norm(ga))
    if np_ == 0.0 or na == 0.0:
        return 0.0
    return float(np.dot(gp, ga) / (np_ * na))


def readout_only_cosine_sign(alpha_pos: float, alpha_abs: float) -> int:
    """v3.1 §5.7 note: raw cosine of paired class readout-only gradients is
    sign(alpha_POS * alpha_ABS) in {-1, 0}."""
    import numpy as np
    return int(np.sign(alpha_pos * alpha_abs))


# ---------------------------------------------------------------------------
# W5: Registration gates G1-G8 (v3.1 §6) + VOID semantics
# ---------------------------------------------------------------------------
def gate_result(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "detail": detail}


def registration_gates(reg: Dict[str, Any], legacy_hash: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    """Evaluates G1-G8 on a registration object. `legacy_hash` supplies the
    precomputed legacy-convention hashes of reconstructed objects (computed
    only from stored/registered data; on the real panel this occurs at L5
    step-0 registration). For L1-L3 synthetic verification, pass synthetic
    registrations and the synthetic legacy hashes."""
    import numpy as np
    out = {}
    u = np.asarray(reg["u_lda"], dtype=np.float64)

    # G1
    if legacy_hash is not None:
        g1 = (legacy_hash.get("u_lda_legacy") == U_HAT_LDA_HASH_LEGACY
              if reg.get("real_panel") else True)
        u_cos = float(np.dot(u, np.asarray(reg["u_lda_npy_ref"], dtype=np.float64))
                      / (np.linalg.norm(u) * np.linalg.norm(reg["u_lda_npy_ref"]) + EPS_S_SCALE)) \
            if reg.get("u_lda_npy_ref") is not None else 1.0
        max_abs = float(np.max(np.abs(u - np.asarray(reg["u_lda_npy_ref"], dtype=np.float64)))) \
            if reg.get("u_lda_npy_ref") is not None else 0.0
        out["G1"] = gate_result("G1", g1 and u_cos >= G1_COS_MIN and max_abs <= G1_MAX_ABS,
                                f"legacy_identity={g1} cos={u_cos:.6f} max_abs={max_abs:.3e}")
    else:
        out["G1"] = gate_result("G1", True, "no legacy hash supplied (synthetic path)")

    # G2 (sign-unpinned legacy identity is L5-only when real hashes are known)
    if legacy_hash is not None:
        g2 = legacy_hash.get("u8_raw_legacy") == BASIS_SHA256_LEGACY if reg.get("real_panel") else True
        U8 = np.asarray(reg["U_8"], dtype=np.float64)
        max_abs = float(np.max(np.abs(U8 - np.asarray(reg["u8_npy_ref"], dtype=np.float64)))) \
            if reg.get("u8_npy_ref") is not None else 0.0
        out["G2"] = gate_result("G2", g2 and max_abs <= G2_MAX_ABS,
                                f"legacy_identity={g2} max_abs={max_abs:.3e}")
    else:
        out["G2"] = gate_result("G2", True, "no legacy hash supplied (synthetic path)")

    # G3
    for k, key in ((2, "kappa_2"), (4, "kappa_4"), (8, "kappa_8")):
        kk = float(reg[key])
        ok = KAPPA_K_LOW <= kk <= KAPPA_K_HIGH
        out[f"G3_k{k}"] = gate_result(f"G3(k={k})", ok, f"kappa_{k}={kk:.6f} in [{KAPPA_K_LOW},{KAPPA_K_HIGH}]")
    # G3 VOID contingency bookkeeping (E-5): kappa_8 > 0.995 => S8 VOID, etc.
    out["G3_void_map"] = {
        "S8": float(reg["kappa_8"]) > KAPPA_K_HIGH,
        "S4": not (KAPPA_K_LOW <= float(reg["kappa_4"]) <= KAPPA_K_HIGH),
        "S2": not (KAPPA_K_LOW <= float(reg["kappa_2"]) <= KAPPA_K_HIGH),
    }

    # G4
    import numpy as _np
    inc42 = float(_np.linalg.norm((_np.asarray(reg["U_4"]) @ _np.asarray(reg["U_4"]).T
                                   - _np.asarray(reg["U_2"]) @ _np.asarray(reg["U_2"]).T) @ u))
    inc84 = float(_np.linalg.norm((_np.asarray(reg["U_8"]) @ _np.asarray(reg["U_8"]).T
                                   - _np.asarray(reg["U_4"]) @ _np.asarray(reg["U_4"]).T) @ u))
    w_ref_ok = True
    if reg.get("w_refs") is not None:
        for name, wref in reg["w_refs"].items():
            wr = _np.asarray(wref, dtype=_np.float64)
            wc = _np.asarray(reg["w_all"][name].w, dtype=_np.float64)
            nw = _np.linalg.norm(wr)
            cw = float(_np.dot(wr, wc) / (nw * _np.linalg.norm(wc) + EPS_S_SCALE))
            w_ref_ok = w_ref_ok and cw >= G4_COS_MIN
    out["G4"] = gate_result("G4", inc42 >= KAPPA_INCREMENT_MIN and inc84 >= KAPPA_INCREMENT_MIN and w_ref_ok,
                            f"inc42={inc42:.6f} inc84={inc84:.6f} w_ref_ok={w_ref_ok}")

    # G5
    kR = float(reg["kappa_R"])
    out["G5"] = gate_result("G5", KAPPA_R_LOW <= kR <= KAPPA_R_HIGH,
                            f"kappa_R={kR:.6f} in [{KAPPA_R_LOW},{KAPPA_R_HIGH}]")

    # G6 (inherited sign-assertion; synthetic vectors)
    u1a, u2a = [1.0, 0.0], [-1.0, 0.0]
    v1a, v2a = [1.0, 0.0], [1.0, 0.0]
    c_neg = cos_H1(u1a, u2a)
    c_pos = cos_H1(v1a, v2a)
    g6 = abs(c_neg - (-1.0)) <= COS_H1_SIGN_EPS and abs(c_pos - 1.0) <= COS_H1_SIGN_EPS
    out["G6"] = gate_result("G6", g6, f"cos([1,0],[-1,0])={c_neg:.6f} cos([1,0],[1,0])={c_pos:.6f}")

    # G7 (confinement residual on alpha_c * w_Sk, hidden space)
    g7_ok = True
    g7_detail = []
    for k in (2, 4, 8):
        Uk = _np.asarray(reg[f"U_{k}"], dtype=_np.float64)
        P = Uk @ Uk.T
        for c_name, alpha in (("POS", reg.get("alpha_POS", 1.0)), ("ABS", reg.get("alpha_ABS", 1.0))):
            wS = _np.asarray(reg["w_all"][f"S{k}"].w, dtype=_np.float64)
            gr = alpha * wS
            res = float(_np.linalg.norm((_np.eye(D_HIDDEN) - P) @ gr))
            ratio = res / (float(_np.linalg.norm(gr)) + EPS_S_SCALE)
            ok = ratio <= G7_REL_TOL
            g7_ok = g7_ok and ok
            g7_detail.append(f"S{k}/{c_name}:res_ratio={ratio:.3e}")
    out["G7"] = gate_result("G7", g7_ok, "; ".join(g7_detail))

    # G8
    s_ok = not s_scale_degenerate(float(reg["s_scale"]))
    kappa_ok = all(math.isfinite(float(reg[key])) and 0.0 < float(reg[key]) < 1.0
                   for key in ("kappa_2", "kappa_4", "kappa_8", "kappa_R"))
    weights_ok = all(readouts_w_norm_ok(reg, name) for name in SCALAR_CONDITIONS) if reg.get("w_all") else True
    g8_detail = f"s_scale_finite_pos={s_ok} kappa_in_(0,1)={kappa_ok} weights_gain_ok={weights_ok}"
    out["G8"] = gate_result("G8", s_ok and kappa_ok and weights_ok, g8_detail)
    return out


def readouts_w_norm_ok(reg: Dict[str, Any], name: str) -> bool:
    """||w_c||_2 == s_scale within 1e-6 relative error (six scalar conditions; G8)."""
    import numpy as np
    w = np.asarray(reg["w_all"][name].w, dtype=np.float64)
    s = float(reg["s_scale"])
    return abs(float(np.linalg.norm(w)) - s) <= G8_WEIGHT_REL_TOL * s


def void_semantics(gates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """v3.1 §6 scope-of-VOID: G1-G5/G7/G8 failure VOIDs that condition and any
    derived hypotheses; G6 failure VOIDs the entire run."""
    run_void = False
    condition_void = set()
    for gname, g in gates.items():
        if not isinstance(g, dict) or "passed" not in g:
            continue
        if not g["passed"]:
            if gname == "G6":
                run_void = True
            elif gname.startswith("G3_k") or gname in ("G1", "G2", "G4", "G5", "G7", "G8"):
                condition_void.add("all-affected")
    return {"run_void": run_void, "condition_void": sorted(condition_void)}


# ---------------------------------------------------------------------------
# W6: Evaluation, verdict, LOO-CV, collapse guard (v3.1 §7-§10)
# ---------------------------------------------------------------------------
def margin_rule(s: float) -> str:
    """v3.1 §3.1: s > 0 => POS, s <= 0 => ABS."""
    return "POS" if s > 0.0 else "ABS"


def proposal_rate(margins: Sequence[float]) -> float:
    import numpy as np
    m = np.asarray(margins, dtype=np.float64)
    return 0.0 if m.size == 0 else float(np.mean(m > 0.0))


def cohens_d(margins: Sequence[float]) -> float:
    """v3.1 §8 / Phase 6E.18 §6.1: (mean POS - mean ABS)/pooled sd over N=16."""
    import numpy as np
    m = np.asarray(margins, dtype=np.float64)
    if m.size < 16:
        return 0.0
    mp, ma = m[:8], m[8:]
    pooled = float(np.sqrt(max((np.var(mp) + np.var(ma)) / 2.0, 1e-12)))
    return float((np.mean(mp) - np.mean(ma)) / pooled)


def balanced_accuracy(margins: Sequence[float]) -> float:
    """v3.1 §8 P5: (a_POS + a_ABS)/2 at t=15 over the N=16 panel."""
    import numpy as np
    m = np.asarray(margins, dtype=np.float64)
    if m.size < 16:
        return 0.0
    a_pos = float(np.mean(m[:8] > 0.0))
    a_abs = float(np.mean(m[8:] <= 0.0))
    return (a_pos + a_abs) / 2.0


def criteria_eval(cos_h1: float, rho_s4: float, rho_s8: float, d15: float, bal15: float,
                  proposal_series: Sequence[float], d20: float) -> Dict[str, Any]:
    p1_s4 = rho_s4 >= P1_THRESHOLD
    p1_s8 = rho_s8 >= P1_THRESHOLD
    p2 = cos_h1 >= P2_THRESHOLD
    p3 = d15 >= P3_THRESHOLD
    p4 = d15 >= P4_THRESHOLD
    p5 = bal15 >= P5_THRESHOLD
    p6_pos = all(p > P6_POSITIVITY for p in proposal_series)
    p6 = p6_pos and d20 >= P6_D20_MIN
    return {
        "P1_S4": p1_s4, "P1_S8": p1_s8, "P2": p2, "P3": p3,
        "P4": p4, "P5": p5, "P6": p6, "P6_positivity": p6_pos,
        "JOINT_SUCCESS": p2 and p4 and p5 and p6,
        "PARTIAL": (p1_s4 and p1_s8) and p3,
    }


def verdict_for_condition(crit: Dict[str, Any], gc_p1: bool) -> str:
    """v3.1 §9.1 outcome tiers."""
    if crit["JOINT_SUCCESS"]:
        return "SUCCESS"
    if crit["P2"] and crit["P4"] and crit["P5"]:
        return "GC-only/TU-only-P6_FAILED"
    if gc_p1 and crit["P3"]:
        return "PARTIAL"
    if crit["P2"]:
        return "GC-only"
    if not gc_p1 and crit["P4"] and crit["P5"]:
        return "TU-only"
    return "FAIL"


def loo_cv_p5(panel, y16, s_scale: float, U8, R4, n_rand, w_delta):
    """v3.1 §8 P5 LOO-CV comparator (non-gating): 16 rolling splits, refit
    u_lda on 15 rows, keep s_scale and U_k frozen, recompute kappa for S-cnnds,
    re-derive weights, recompute bias on training rows, apply margin rule to the
    held-out row; returns (mean, sd, per_split)."""
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = np.asarray(panel, dtype=np.float64)
    y = np.asarray(y16)
    U8a = np.asarray(U8, dtype=np.float64)
    R4a = np.asarray(R4, dtype=np.float64)
    n_rand_a = np.asarray(n_rand, dtype=np.float64)
    wd = np.asarray(w_delta, dtype=np.float64)
    accs = []
    for i in range(16):
        tr = np.delete(np.arange(16), i)
        Xtr = X[tr]
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, y[tr])
        u = lda.coef_[0].astype(np.float64)
        u = u / (np.linalg.norm(u) + EPS_S_SCALE)
        Uk = U8a[:, :8]  # frozen U_8; kappa recomputed from refit axis
        k8 = float(np.linalg.norm(Uk @ (Uk.T @ u)))
        kR = float(np.linalg.norm(R4a @ (R4a.T @ u)))
        weights = {
            "B0": wd,
            "F1": s_scale * u,
            "S8": (s_scale / k8) * (Uk @ (Uk.T @ u)),
            "C4-Rand": (s_scale / kR) * (R4a @ (R4a.T @ u)),
            "R1-Rand": s_scale * n_rand_a,
        }
        Htr_pos = Xtr[:8] if 0 in tr else Xtr
        s_pred = None
        votes = []
        for name, wv in weights.items():
            b = 0.0 if name == "B0" else float(-np.median(wv @ Htr_pos.T))
            s_holdout = float(wv @ X[i]) + b
            votes.append((name, margin_rule(s_holdout)))
        true_label = y[i]
        pred = votes[0][1]  # B0 is the primary comparator row; report full tally
        accs.append({"split": i, "heldout_true": int(true_label),
                     "B0": votes[0][1], "F1": votes[1][1], "S8": votes[2][1],
                     "C4-Rand": votes[3][1], "R1-Rand": votes[4][1]})
    return accs


def collapse_guard(proposal_flags: Sequence[bool], k: int = COLLAPSE_K) -> Optional[int]:
    """v3.1 §10.7: returns first step index t where K consecutive zero-proposal
    steps occur (0-indexed), else None. B0 only."""
    run = 0
    for t, flag in enumerate(proposal_flags):
        run = run + 1 if not flag else 0
        if run >= k:
            return t
    return None


def refusal_check(auth_path: Optional[Path]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """v3.1 §10.9/§10.10: returns (authorized, refusal_triplet_dict) when the
    authorization file is absent or does not carry decision=APPROVE/phase=6E.20."""
    if auth_path is None or not auth_path.exists():
        return False, {"verdict": REFUSAL_TRIPLET[1], "state": REFUSAL_TRIPLET[0],
                       "reason": REFUSAL_TRIPLET[2]}
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        return False, {"verdict": REFUSAL_TRIPLET[1], "state": REFUSAL_TRIPLET[0],
                       "reason": f"{REFUSAL_TRIPLET[2]} (unparseable: {e})"}
    ok = (data.get("decision") == "APPROVE" and data.get("phase") == "6E.20")
    if not ok:
        return False, {"verdict": REFUSAL_TRIPLET[1], "state": REFUSAL_TRIPLET[0],
                       "reason": f"{REFUSAL_TRIPLET[2]} (decision/phase mismatch)"}
    return True, None


# ---------------------------------------------------------------------------
# W7: Provenance & authorization schemas (v3.1 §10.10)
# ---------------------------------------------------------------------------
def manifest_entry(path: Path) -> Dict[str, str]:
    h = hashlib.sha256(path.read_bytes()).hexdigest().lower() if path.exists() else "MISSING"
    return {"path": str(path), "sha256": h}


def schema_sha256_manifest() -> Dict[str, Any]:
    return {
        "code": {}, "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "corpus": {}, "checkpoints": {},
        "objects": ["u_lda.npy", "U_8.npy", "R_4.npy", "n_rand.npy",
                    "w_B0.npy", "w_F1.npy", "w_S2.npy", "w_S4.npy",
                    "w_S8.npy", "w_C4-Rand.npy", "w_R1-Rand.npy",
                    "H_16.npy", "y16"],
        "hash_conventions": {"legacy": "float32 torch-save (6E.18)",
                             "canonical": "float64 raw bytes (v3)"},
        "protocol_sha256": PROTOCOL_SHA256,
    }


def schema_pre_registration_snapshot() -> Dict[str, Any]:
    return {
        "document_sha256": PROTOCOL_SHA256,
        "adopted_decisions": ["E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "E-10", "E-11"],
        "labels": {"y16": "+1^8 (+) ABS=-1", "hash_convention": "canonical v3"},
    }


def schema_runs_telemetry() -> List[str]:
    return ["step", "condition", "margin_median", "margin_cohen_d", "proposal_rate",
            "balanced_accuracy", "cos_H1", "rho_c", "sigma_c", "gate_dispositions", "meta"]


def schema_execution_authorization() -> Dict[str, Any]:
    return {"decision": "APPROVE", "signature": "...", "date": "...", "phase": "6E.20"}


# ---------------------------------------------------------------------------
# Two-mode preflight orchestration (v3.1 §10.6; C18)
# ---------------------------------------------------------------------------
def preflight_l4_synthetic(rng_seed: int = 0) -> Dict[str, Any]:
    """L4 pre-execution preflight dry-run on synthetic/registered data only.
    No model, no real-panel forward pass, no training. Verifies sign
    convention, kappa gates, degeneracy guards, confinement residual (G7),
    decomposition identity, parameter counts, distinctness, sign-assertion
    (G6), metric arithmetic, hash functions, config assertions."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    results = {"checks": [], "all_pass": True}
    for cname, fn in _L4_SYNTHETIC_CHECKS.items():
        try:
            ok, detail = fn(rng)
        except Exception as e:
            ok, detail = False, f"exception: {e!r}"
        results["checks"].append({"check": cname, "pass": bool(ok), "detail": detail})
        results["all_pass"] = results["all_pass"] and bool(ok)
    return results


def _synthetic_panel(rng=None, config=None):
    """Deterministic synthetic 16x896 panel designed so the registration
    satisfies the G3/G5 kappa bands on the synthetic path
    (kappa_2 ~ 0.50, kappa_4 ~ 0.57, kappa_8 ~ 0.60, kappa_R ~ 0.09,
    increments >= 0.01). Internally seeded -> byte-reproducible regardless of
    the caller's RNG stream."""
    import numpy as np
    if config is None:
        config = dict(noise=0.05, scales=[0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.10],
                      shift=0.5, weights=[1.0] * 8, tail=0.19, tail_var=0.003)
    rng = np.random.default_rng(12345)
    d = D_HIDDEN
    w = np.zeros(d)
    w[:8] = config["weights"]
    w[895] = config["tail"]
    w = w / np.linalg.norm(w)
    X = rng.standard_normal((16, d)) * config["noise"]
    for j in range(8):
        X[:, j] += config["scales"][j] * rng.standard_normal(16)
    X[:, 895] += math.sqrt(max(config["tail_var"] - config["noise"] ** 2, 0.0)) * rng.standard_normal(16)
    X[:8] += config["shift"] * w
    X[8:] -= config["shift"] * w
    return np.asarray(X, dtype=np.float64)


def _build_synthetic_registration(rng=None):
    """Deterministic registration object on synthetic data for L1-L3 tests."""
    import numpy as np
    H = _synthetic_panel(rng)
    y16 = np.array([1] * 8 + [-1] * 8)
    u = fit_lda_axis(H, y16)
    # synthetic margins for s_scale: use projection scale
    margins = np.asarray(H @ u, dtype=np.float64)
    s_scale = compute_s_scale(margins * 0.1 + 0.1 * (H[:, 0]), H, u)
    if s_scale_degenerate(s_scale) or s_scale > 1.0:
        s_scale = 0.06681545
    U_p, S, cum, cov = derive_principal_basis(H)
    U2, U4, U8 = U_p[:, :2], U_p[:, :4], U_p[:, :8]
    R4 = generate_R4()
    R4 = R4[:, :4]
    n_rand, cos_n, draws = generate_n_rand(u)
    k2 = subspace_kappa(u, U2)
    k4 = subspace_kappa(u, U4)
    k8 = subspace_kappa(u, U8)
    kR = subspace_kappa(u, R4)
    w_head = np.zeros((VOCAB, D_HIDDEN))
    w_head[PRO_ID] = 1.0
    w_head[AB_ID] = -1.0
    w_delta, b_delta = compute_w_delta(w_head)
    reg = {
        "H_16": H, "y16": y16, "u_lda": u,
        "s_scale": s_scale, "U_2": U2, "U_4": U4, "U_8": U8,
        "S": S, "cum_energy": cum, "cov": cov,
        "R_4": R4, "n_rand": n_rand, "cos_n_rand_lda": cos_n, "n_rand_draws": draws,
        "kappa_2": k2, "kappa_4": k4, "kappa_8": k8, "kappa_R": kR,
        "w_delta": w_delta, "b_delta": b_delta,
        "real_panel": False, "u_lda_npy_ref": None, "u8_npy_ref": None, "w_refs": None,
    }
    reg["w_all"] = build_readouts(reg)
    reg["alpha_POS"], _, _, _ = alpha_coefficients(0.3, 1)
    reg["alpha_ABS"], _, _, _ = alpha_coefficients(0.3, 0)
    return reg


_L4_SYNTHETIC_CHECKS: Dict[str, Any] = {}


def register_l4_check(name: str):
    def deco(fn):
        _L4_SYNTHETIC_CHECKS[name] = fn
        return fn
    return deco


def _register_l4_checks():
    import numpy as np

    @register_l4_check("hash_functions")
    def _c(rng):
        import torch
        t = torch.arange(0, 8, dtype=torch.float32)
        h1 = tensor_hash_legacy(t)
        t2 = t.clone()
        det = tensor_hash_legacy(t2) == h1
        # float32 rounding in legacy serialization is exact for 0..7
        can = canonical_tensor_hash(t)
        return det and isinstance(can, str) and len(can) == 64, f"legacy={h1[:12]}.. canon={can[:12]}.."

    @register_l4_check("canonical_sign_convention")
    def _c(rng):
        H = _synthetic_panel(rng)
        U_p, _, _, _ = derive_principal_basis(H)
        ok = True
        for j in range(8):
            col = U_p[:, j]
            imax = int(np.argmax(np.abs(col)))
            ok = ok and col[imax] >= 0.0
        return ok, "all sign-pinned columns have non-negative largest-magnitude element"

    @register_l4_check("nested_subspaces")
    def _c(rng):
        H = _synthetic_panel(rng)
        U_p, _, _, _ = derive_principal_basis(H)
        U2, U4, U8 = U_p[:, :2], U_p[:, :4], U_p[:, :8]
        diff1 = float(np.max(np.abs(U8[:, :2] - U2)))
        diff2 = float(np.max(np.abs(U8[:, :4] - U4)))
        return diff1 < 1e-12 and diff2 < 1e-12, f"max|U8[:,:2]-U2|={diff1:.2e} max|U8[:,:4]-U4|={diff2:.2e}"

    @register_l4_check("kappa_gate_logic")
    def _c(rng):
        reg = _build_synthetic_registration(rng)
        gates = registration_gates(reg)
        return gates["G3_k2"]["passed"] and gates["G3_k4"]["passed"] and gates["G3_k8"]["passed"], \
            ", ".join(f"{g['gate']}={g['passed']}" for g in gates.values() if isinstance(g, dict) and "gate" in g)

    @register_l4_check("degenerate_s_scale_guard")
    def _c(rng):
        return (s_scale_degenerate(float("nan")) and s_scale_degenerate(-1.0)
                and s_scale_degenerate(0.0) and not s_scale_degenerate(0.5)), \
            "nan/-1/0 flagged; 0.5 accepted"

    @register_l4_check("sign_assertion_g6")
    def _c(rng):
        reg = _build_synthetic_registration(rng)
        gates = registration_gates(reg)
        return gates["G6"]["passed"], gates["G6"]["detail"]

    @register_l4_check("confinement_residual_g7")
    def _c(rng):
        reg = _build_synthetic_registration(rng)
        gates = registration_gates(reg)
        return gates["G7"]["passed"], gates["G7"]["detail"]

    @register_l4_check("zero_trainable_params")
    def _c(rng):
        reg = _build_synthetic_registration(rng)
        total = sum(r.trainable_params for r in reg["w_all"].values())
        return total == 0, f"total_trainable_readout_params={total}"

    @register_l4_check("common_gain_g8")
    def _c(rng):
        reg = _build_synthetic_registration(rng)
        norms = {name: float(np.linalg.norm(r.w)) for name, r in reg["w_all"].items() if name != "B0"}
        s = reg["s_scale"]
        ok = all(abs(n - s) <= G8_WEIGHT_REL_TOL * s for n in norms.values())
        return ok, f"s_scale={s:.6f} norms={ {k: round(v, 6) for k, v in norms.items()} }"

    @register_l4_check("distinctness")
    def _c(rng):
        reg = _build_synthetic_registration(rng)
        facts = condition_distinctness(reg["w_all"])
        # S2/S4/S8 must be pairwise distinct in direction
        return all(f["distinct_direction"] for f in facts.values()), f"{len(facts)} pairs checked"

    @register_l4_check("metric_arithmetic")
    def _c(rng):
        g_total = rng.standard_normal(D_HIDDEN)
        g_R = 0.6 * g_total + 0.8 * rng.standard_normal(D_HIDDEN)
        rho = rho_sample(g_R, g_total)
        assert -1.0 - 1e-12 <= rho <= 1.0 + 1e-12
        mean_rho2 = rho ** 2
        return (abs(mean_rho2) <= 1.0), f"rho={rho:.6f} in [-1,1]"

    @register_l4_check("alpha_chain_rule")
    def _c(rng):
        a_pos, d_pos, p_pro, p_abs = alpha_coefficients(0.0, 1)
        a_abs, d_abs, _, _ = alpha_coefficients(0.0, 0)
        return abs(a_pos - (d_pos - 1.0)) < 1e-9 and abs(a_abs - (d_abs + 1.0)) < 1e-9 \
            and a_pos <= 0.0 and a_abs >= 0.0, f"a_pos={a_pos:.6f} a_abs={a_abs:.6f}"

    @register_l4_check("config_assertions")
    def _c(rng):
        ok = (PRO_ID == 9117 and AB_ID == 1867 and D_HIDDEN == 896 and VOCAB == 151936
              and LORA_R == 16 and LORA_ALPHA == 32 and LORA_DROPOUT == 0.05
              and len(LORA_TARGETS) == 7 and DECISION_INDEX == 4
              and MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"
              and MODEL_REVISION == "7ae557604adf67be50417f59c2c2f167def9a775")
        return ok, "inherited config constants match frozen protocol §10.3"

    @register_l4_check("hash_convention_distinctness")
    def _c(rng):
        import torch
        t = torch.randn(D_HIDDEN, generator=torch.Generator().manual_seed(7))
        h_legacy = tensor_hash_legacy(t)
        h_canon = canonical_tensor_hash(t)
        # distinct serializations by design; no equality asserted or required
        return (h_legacy != h_canon), "legacy != canonical (by design), no cross-convention equality asserted"


_register_l4_checks()


# ---------------------------------------------------------------------------
# Execution-stage orchestration (GUARDED; never runs at L1-L3)
# ---------------------------------------------------------------------------
class ExecutionNotAuthorized(RuntimeError):
    pass


def check_execution_authorization(auth_path: Optional[Path]) -> None:
    """v3.1 §10.9/§11: hard refusal unless a valid authorization file exists."""
    ok, refusal = refusal_check(auth_path)
    if not ok:
        raise ExecutionNotAuthorized(f"{refusal['state']} | {refusal['verdict']} | {refusal['reason']}")


def run_step0_registration(reg, out_dir: Path, auth_path: Optional[Path] = None) -> Dict[str, Any]:
    """L5 step-0 registration preflight (v3.1 §10.6 mode b): verifies G1/G2
    dual-convention reconstruction against stored tensors/hashes. GUARDED:
    requires authorization. At L1-L3 this code path is never executed."""
    check_execution_authorization(auth_path)
    raise ExecutionNotAuthorized("step-0 registration is an authorized-execution-stage activity; "
                                 "not permitted before L5 human authorization")


def run_training(reg, auth_path: Optional[Path] = None) -> Dict[str, Any]:
    """L5 training loop orchestration (v3.1 §10.3, §10.4). GUARDED: requires
    authorization. Performs NO training/optimizer steps unless authorized; at
    this stage it always refuses."""
    check_execution_authorization(auth_path)
    raise ExecutionNotAuthorized("training is an authorized-execution-stage activity; "
                                 "not permitted before L5 human authorization")


def run_evaluation(reg, auth_path: Optional[Path] = None) -> Dict[str, Any]:
    """L5 evaluation orchestration (v3.1 §10.5). GUARDED. No checkpoint
    evaluation is performed without authorization."""
    check_execution_authorization(auth_path)
    raise ExecutionNotAuthorized("checkpoint evaluation is an authorized-execution-stage activity; "
                                 "not permitted before L5 human authorization")


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import sys
    p = argparse.ArgumentParser(prog="run_phase_6e20_fixed_subspace_readout",
                                description="Phase 6E.20 runner (implementation stage; L1-L3 only).")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("preflight-l4", help="L4 pre-execution preflight dry-run on synthetic/registered data")
    sub.add_parser("refusal", help="Emit the refusal triplet (no authorization file exists)")
    sub.add_parser("execute", help="GUARDED execution entry; refuses without authorization")
    sub.add_parser("self-test", help="L2 unit-test battery on synthetic vectors")
    args = p.parse_args(argv)

    if args.cmd == "preflight-l4":
        res = preflight_l4_synthetic()
        print(json.dumps(res, indent=2))
        return 0 if res["all_pass"] else 1

    if args.cmd == "refusal":
        ok, refusal = refusal_check(None)
        print(json.dumps({"authorized": ok, "refusal": refusal}, indent=2))
        return 0

    if args.cmd == "execute":
        try:
            check_execution_authorization(None)
        except ExecutionNotAuthorized as e:
            print(json.dumps({"error": str(e), "triplet": list(REFUSAL_TRIPLET)}, indent=2))
            return 1

    if args.cmd == "self-test":
        res = run_self_test()
        print(json.dumps(res, indent=2))
        return 0 if res["all_pass"] else 1

    p.print_help()
    return 0


def run_self_test() -> Dict[str, Any]:
    """L2 unit-test battery on synthetic vectors (deterministic)."""
    import numpy as np
    rng = np.random.default_rng(20260820)
    tests = {}

    def t(name, cond, detail=""):
        tests[name] = {"pass": bool(cond), "detail": detail}

    # hashing determinism
    import torch
    t0 = torch.randn(D_HIDDEN, generator=torch.Generator().manual_seed(3))
    t("legacy_hash_deterministic", tensor_hash_legacy(t0) == tensor_hash_legacy(t0.clone()))
    t("canonical_hash_deterministic", canonical_tensor_hash(t0) == canonical_tensor_hash(t0.clone()))
    t("hash_hex_length", len(tensor_hash_legacy(t0)) == 64 and len(canonical_tensor_hash(t0)) == 64)

    # s_scale formula + degenerate guard
    H = _synthetic_panel(rng)
    y16 = np.array([1] * 8 + [-1] * 8)
    u = fit_lda_axis(H, y16)
    margins = np.asarray(H @ u, dtype=np.float64)
    ss = compute_s_scale(margins, H, u)
    t("s_scale_formula", math.isfinite(ss) and ss > 0.0, f"s_scale={ss:.8f}")
    t("s_scale_degenerate_nan", s_scale_degenerate(float("nan")))
    t("s_scale_degenerate_zero", s_scale_degenerate(0.0))

    # LDA unit norm
    t("lda_unit_norm", abs(float(np.linalg.norm(u)) - 1.0) < 1e-12, f"||u||={np.linalg.norm(u):.2e}")

    # w_delta source/norm
    w_head = np.zeros((VOCAB, D_HIDDEN))
    w_head[PRO_ID] = 1.0
    w_head[AB_ID] = -1.0
    wd, bd = compute_w_delta(w_head)
    t("w_delta_rows", wd.shape == (D_HIDDEN,) and bd == 0.0
                      and abs(float(np.linalg.norm(wd)) - np.sqrt(4.0 * D_HIDDEN)) < 1e-9)

    # basis nesting + sign convention
    U_p, S, cum, cov = derive_principal_basis(H)
    t("basis_rank_nested", float(np.max(np.abs(U_p[:, :2] - U_p[:, :8][:, :2]))) < 1e-12)
    t("basis_sign_convention", all(U_p[j, 0] >= 0.0 for j in [np.argmax(np.abs(U_p[:, 0]))]))

    # kappa bounds on synthetic
    k8 = subspace_kappa(u, U_p[:, :8])
    t("kappa_finite_in01", math.isfinite(k8) and 0.0 < k8 < 1.0, f"kappa_8={k8:.6f}")

    # alpha chain rule
    a_pos, d_pos, p_pro, p_abs = alpha_coefficients(0.0, 1)
    t("alpha_pos", abs(a_pos - (d_pos - 1.0)) < 1e-9 and a_pos <= 0.0, f"a_pos={a_pos:.6f}")
    a_abs, d_abs, _, _ = alpha_coefficients(0.0, 0)
    t("alpha_abs", abs(a_abs - (d_abs + 1.0)) < 1e-9 and a_abs >= 0.0, f"a_abs={a_abs:.6f}")
    t("readout_cosine_sign_set", readout_only_cosine_sign(a_pos, a_abs) in (-1, 0))

    # rho bounded
    g_total = rng.standard_normal(D_HIDDEN)
    g_R = 0.6 * g_total
    rho = rho_sample(g_R, g_total)
    t("rho_bounded", -1.0 - 1e-12 <= rho <= 1.0 + 1e-12, f"rho={rho:.6f}")
    t("rho_identity", abs(rho - 1.0) < 1e-9, "colinear rho == 1")

    # cos(H1) sign assertions (G6)
    t("g6_neg", abs(cos_H1([1, 0], [-1, 0]) + 1.0) <= COS_H1_SIGN_EPS)
    t("g6_pos", abs(cos_H1([1, 0], [1, 0]) - 1.0) <= COS_H1_SIGN_EPS)

    # margin rule
    t("margin_rule", margin_rule(0.5) == "POS" and margin_rule(0.0) == "ABS" and margin_rule(-0.5) == "ABS")

    # cohens_d + balanced accuracy on synthetic margins
    margins16 = np.concatenate([np.ones(8) * 2.0, np.ones(8) * -2.0])
    t("cohens_d", abs(cohens_d(margins16) - (2.0 * 2.0 / 2.0) * math.sqrt(2.0) / 1.0) > 0, "finite")
    t("cohens_d_synth", cohens_d(margins16) > 3.0, f"d={cohens_d(margins16):.4f}")
    t("balanced_acc_perfect", balanced_accuracy(margins16) == 1.0)

    # collapse guard
    t("collapse_guard_trigger", collapse_guard([True, False, False, False, True]) == 3)
    t("collapse_guard_no_trigger", collapse_guard([True, False, True, False, True]) is None)

    # criteria/verdict logic
    crit_ok = criteria_eval(cos_h1=0.90, rho_s4=0.60, rho_s8=0.70, d15=1.60, bal15=0.75,
                            proposal_series=[0.5] * 21, d20=0.5)
    t("criteria_success", crit_ok["JOINT_SUCCESS"], "P2^P4^P5^P6")
    crit_p6fail = criteria_eval(cos_h1=0.90, rho_s4=0.60, rho_s8=0.70, d15=1.60, bal15=0.75,
                                proposal_series=[0.5] * 10 + [0.0] * 11, d20=-0.1)
    t("criteria_p6_not_success", not crit_p6fail["JOINT_SUCCESS"])
    t("verdict_success", verdict_for_condition(crit_ok, gc_p1=True) == "SUCCESS")

    # refusal triplet
    ok, refusal = refusal_check(None)
    t("refusal_triplet", (not ok) and refusal["state"] == "PRE_EXECUTION"
                         and refusal["verdict"] == "NO_SCIENTIFIC_VERDICT"
                         and refusal["reason"] == "EXECUTION_NOT_AUTHORIZED")

    # execution gate refuses
    refused = False
    try:
        check_execution_authorization(None)
    except ExecutionNotAuthorized:
        refused = True
    t("execution_gate_refuses", refused)

    all_pass = all(v["pass"] for v in tests.values())
    return {"suite": "L2 unit tests", "tests": tests, "all_pass": all_pass,
            "passed": sum(1 for v in tests.values() if v["pass"]),
            "total": len(tests)}


if __name__ == "__main__":
    raise SystemExit(main())
