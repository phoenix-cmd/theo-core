"""Phase 6E.17 — Causal Intervention: LDA-Aligned Trainable Decision Readout (I1)
with Matched Baseline (B) and Capacity-Matched Random-Orientation Control (C2).

Governing spec: theo-slm-causal-intervention-authorization-protocol-draft-v2.md (APPROVED 2026-08-16).
This is a CAUSAL FALSIFICATION TEST of H-Causal (readout misalignment causes gradient conflict).

Executes exactly three matched trajectories differing ONLY in the decision readout:
  B  : frozen vocabulary readout margin  Delta_z_b = w_delta^T h_L + b_delta
  I1 : trainable readout  s = a^T h_L + b ;  z_PRO = s, z_AB = -s ;  a(0) = s_scale * u_hat_LDA
  C2 : same form ;  c(0) = s_scale * n (random, |cos(n, u_hat_LDA)| <= 0.05, seed 20260816)

All frozen hyperparameters, seeds, panels, checkpoints, and thresholds follow protocol v2
sections 3-7. Primary endpoints: cos(H1) (mean pairwise label-signed cross-cosine of
decision-position gradients, LoRA params only), rho_CD, F. Verdict per section 7.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import datetime
import gc
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Frozen constants (protocol v2)
# ---------------------------------------------------------------------------
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
P1_THRESH = -0.30
C2X_THRESH = 0.25
P3_COHEN_D_THRESH = 0.2
P3_R_EFF_THRESH = 1.6
P4_KN1_RANGE = (0.65, 1.0)
P4_LOO_TOL = 0.1
P5_THRESH = 0.3

EXPECTED_BASE_HASH = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
EXPECTED_CORPUS_HASH = "a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0"
EXPECTED_COS_LDA_WDELTA = 0.0055  # +-0.001

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = BASE_DIR.parent / "theo-data" / "datasets"
CORPUS_PATH = DATA_DIR / "theo_slm_v0_deduplicated" / "candidate_records.json"
BASE_MODEL_PATH = Path(r"C:\Users\bs162\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775")
PROTOCOL_PATH = Path(r"C:\Users\bs162\Desktop\THEO\theo-core\docs\research\theo-slm-causal-intervention-authorization-protocol-draft-v2.md")
REVIEW_PATH = Path(r"C:\Users\bs162\Desktop\THEO\theo-core\docs\research\theo-slm-phase-6e17-pre-execution-authorization-review-v1.md")
ARTIFACT_DIR = DATA_DIR / "theo_slm_v0_artifacts" / "phase-6e17-causal-intervention"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t.detach().float().cpu().contiguous(), buf)
    return sha256_bytes(buf.getvalue())


def state_dict_hash(state: dict) -> str:
    buf = io.BytesIO()
    torch.save({k: v.detach().float().cpu().contiguous() for k, v in state.items()}, buf)
    return sha256_bytes(buf.getvalue())


def compute_effective_rank(gram: np.ndarray) -> float:
    eigvals = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    tr = float(np.sum(eigvals))
    tr2 = float(np.sum(eigvals ** 2))
    return 0.0 if tr2 < 1e-12 or tr < 1e-12 else float(tr ** 2 / tr2)


def get_top1_dominance(gram: np.ndarray) -> float:
    eigvals = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    tr = float(np.sum(eigvals))
    return 0.0 if tr < 1e-12 else float(np.max(eigvals) / tr)


def compute_cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    n1, n2 = v1.norm().item(), v2.norm().item()
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def format_prompt(percept: str, concepts) -> str:
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


def construct_target_object(record) -> dict:
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


class ReadoutHead(nn.Module):
    """s = a^T h_L + b ; z_PRO = s ; z_AB = -s. 897 trainable params."""

    def __init__(self, a0: np.ndarray, b0: float):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(b0), dtype=torch.float32))

    def forward_h(self, h: torch.Tensor) -> torch.Tensor:
        return h.float() @ self.a + self.b

    def get_params(self) -> dict:
        return {"a": self.a.detach().float().cpu().contiguous(), "b": self.b.detach().float().cpu().contiguous()}


def compute_weighted_loss(logits, labels, dec_token_indices, readout=None, hiddens=None, prompt_lens=None):
    """Historical weighted decision loss (6E.14) with optional readout patch at the
    decision position (valid index 4). For readout conditions the decision-position
    raw CE is computed from patched logits in fp32; everything else byte-identical."""
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
            s = readout.forward_h(hL)
            dec_logits[PRO_ID] = s
            dec_logits[AB_ID] = -s
            raw_losses[i, d_pos] = loss_fct(dec_logits.unsqueeze(0), shift_labels[i, d_pos].reshape(1)).sum()

    weighted_loss = (raw_losses * weighted_mask).sum() / max(weighted_mask.sum().item(), 1.0)
    return weighted_loss


# ---------------------------------------------------------------------------
# Data reconstruction (exact, seed 42) and panels
# ---------------------------------------------------------------------------
def reconstruct_data(corpus_path: Path):
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


# ---------------------------------------------------------------------------
# Step-0 fp32 derivation of u_hat_LDA, s_scale, a0, b0, c0
# ---------------------------------------------------------------------------
def derive_readout_initialization(data, tokenizer):
    print("\n[DERIVE] Step-0 fp32 derivation of u_hat_LDA, s_scale, a0, b0, c0", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL_PATH), torch_dtype=torch.float32,
                                                 device_map=DEVICE, trust_remote_code=True)
    model.config.use_cache = False
    model.eval()
    w_head = model.get_output_embeddings().weight.detach()
    w_delta = (w_head[PRO_ID] - w_head[AB_ID]).float().cpu().numpy()
    w_delta_unit = w_delta / (np.linalg.norm(w_delta) + 1e-12)

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
    u_lda = u_lda_raw / (np.linalg.norm(u_lda_raw) + 1e-12)
    mu_p, mu_a = H16[:8].mean(0), H16[8:].mean(0)
    u_centroid = (mu_p - mu_a) / (np.linalg.norm(mu_p - mu_a) + 1e-12)
    svm = LinearSVC(C=1.0, dual=True, max_iter=2000, random_state=42).fit(H16, y16)
    u_svm = svm.coef_[0] / (np.linalg.norm(svm.coef_[0]) + 1e-12)

    cos_lda = float(u_lda @ w_delta_unit)
    cos_cent = float(u_centroid @ w_delta_unit)
    cos_svm = float(u_svm @ w_delta_unit)

    s_scale = np.std(margins_b) / (np.std(H16 @ u_lda) + 1e-12)
    a0 = s_scale * u_lda
    b0 = -np.median(a0 @ H16[:8].T)

    # C2 random orientation, deterministic Generator seed 20260816
    gen = torch.Generator().manual_seed(SEED_C2_READOUT)
    n_draws = 0
    while True:
        n_draws += 1
        nv = torch.randn(D_HIDDEN, generator=gen)
        nv = nv / (nv.norm() + 1e-12)
        cos_n_lda = float((nv.numpy() @ u_lda))
        if abs(cos_n_lda) <= 0.05:
            break
    c0 = s_scale * nv.numpy()
    b0_c2 = -np.median(c0 @ H16[:8].T)

    s_i1 = H16 @ a0 + b0
    derive = {
        "w_delta_norm": round(float(np.linalg.norm(w_delta)), 6),
        "cos_alignment_lda_vs_w_delta": round(cos_lda, 6),
        "cos_alignment_centroid_vs_w_delta": round(cos_cent, 6),
        "cos_alignment_svm_vs_w_delta": round(cos_svm, 6),
        "cos_alignment_lda_vs_centroid": round(float(u_lda @ u_centroid), 6),
        "s_scale": round(float(s_scale), 8),
        "b0": round(float(b0), 8),
        "b0_c2": round(float(b0_c2), 8),
        "c2_readout_cos_n_lda": round(float(cos_n_lda), 8),
        "c2_readout_draws": n_draws,
        "step0_margin_b_mean": round(float(margins_b.mean()), 6),
        "step0_margin_b_propose_rate_panel": round(float((margins_b > 0).mean()), 6),
        "step0_margin_i1_mean": round(float(s_i1.mean()), 6),
        "step0_margin_i1_propose_rate_panel": round(float((s_i1 > 0).mean()), 6),
        "step0_lda_propose_rates": {
            "b_panel": round(float((margins_b > 0).mean()), 6),
            "i1_panel": round(float((s_i1 > 0).mean()), 6),
        },
        "u_hat_LDA_hash": tensor_hash(torch.tensor(u_lda)),
        "a0_hash": tensor_hash(torch.tensor(a0)),
        "b0_hash": tensor_hash(torch.tensor([b0])),
        "c0_hash": tensor_hash(torch.tensor(c0)),
        "b0_c2_hash": tensor_hash(torch.tensor([b0_c2])),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # verification assertions
    assert abs(cos_lda - EXPECTED_COS_LDA_WDELTA) <= 0.001, f"cos(u_lda,w_delta)={cos_lda} != 0.0055+-0.001"
    print(f"[DERIVE] cos(u_lda,w_delta)={cos_lda:.4f} | w_delta_norm={np.linalg.norm(w_delta):.4f} | s_scale={s_scale:.4f} | b0={b0:.4f}", flush=True)
    print(f"[DERIVE] step-0 propose: B={ (margins_b>0).mean():.3f} I1={ (s_i1>0).mean():.3f} (mismatch documented, non-voiding)", flush=True)
    return derive, u_lda, a0, b0, c0, b0_c2, H16, margins_b


# ---------------------------------------------------------------------------
# Metric probes
# ---------------------------------------------------------------------------
MARGIN_CHUNK = 8


def batched_decision_margins(model, readout, records, tokenizer):
    """Margins at the decision position (partial-prompt context), batched in chunks."""
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
                    s = readout.forward_h(out.hidden_states[-1][j, L, :]).float()
                    all_margins.append(float(2 * s))
                else:
                    all_margins.append(float(z[PRO_ID] - z[AB_ID]))
        del out
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
        s = readout.forward_h(hL)
        dec_logits = shift_logits[0, d_pos, :].float().clone()
        dec_logits[PRO_ID] = s
        dec_logits[AB_ID] = -s
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
    """Gradient of the decision margin (Delta_z_b or 2s) wrt LoRA params, partial-prompt context."""
    p = format_prompt(rec["percept"], rec.get("concepts", [])) + '{"decision": "'
    inp = tokenizer(p, return_tensors="pt").to(DEVICE)
    model.zero_grad(set_to_none=True)
    out = model(**inp, output_hidden_states=(readout is not None))
    if readout is not None:
        hL = out.hidden_states[-1][0, -1, :]
        s = readout.forward_h(hL)
        margin = 2 * s
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
    """Mean pairwise label-signed cross-cosine of decision gradients (primary) and
    aggregate batch cosine (6E.13 reproduction)."""
    signed = []
    for i in range(8):
        g = per_sample_decision_gradient(model, readout, panel_ds_items[i], lora_param_list, tokenizer)
        signed.append(g)
    for j in range(8):
        g = per_sample_decision_gradient(model, readout, panel_ds_items[8 + j], lora_param_list, tokenizer)
        signed.append(-g)
    cos_vals = []
    dots_pos = []
    for i in range(8):
        for j in range(8, 16):
            cos_vals.append(compute_cosine_similarity(signed[i], signed[j]))
            dots_pos.append(1.0 if float((signed[i] * signed[j]).sum()) > 0 else 0.0)
    mean_cos = float(np.mean(cos_vals))
    pos_mean = sum(signed[:8], torch.zeros_like(signed[0])) / 8
    abs_mean = -sum(signed[8:], torch.zeros_like(signed[0])) / 8
    agg_cos = compute_cosine_similarity(pos_mean, abs_mean)
    return {
        "cos_H1_mean_pairwise": round(mean_cos, 6),
        "cos_H1_aggregate_613": round(agg_cos, 6),
        "rho_CD": round(max(0.0, -mean_cos), 6),
        "F_simultaneous_feasibility": round(float(np.mean(dots_pos)), 6),
    }


def hidden_reps(records, model, tokenizer):
    X = []
    for rec in records:
        p = format_prompt(rec["percept"], rec.get("concepts", [])) + '{"decision": "'
        inp = tokenizer(p, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model(**inp, output_hidden_states=True)
            X.append(o.hidden_states[24][0, -1, :].float().cpu().numpy())
    return np.array(X)


def separability_suite(X, y):
    knn1 = KNeighborsClassifier(n_neighbors=1).fit(X, y).score(X, y)
    knn3 = KNeighborsClassifier(n_neighbors=3).fit(X, y).score(X, y)
    knn5 = KNeighborsClassifier(n_neighbors=5).fit(X, y).score(X, y)
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    loo_1, loo_3 = [], []
    for tr, te in loo.split(X):
        loo_1.append(KNeighborsClassifier(n_neighbors=1).fit(X[tr], y[tr]).predict(X[te])[0] == y[te][0])
        loo_3.append(KNeighborsClassifier(n_neighbors=min(3, len(tr))).fit(X[tr], y[tr]).predict(X[te])[0] == y[te][0])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    svm_accs = []
    for tr, te in skf.split(X, y):
        clf = LinearSVC(C=1.0, dual=True, max_iter=2000, random_state=42).fit(X[tr], y[tr])
        svm_accs.append(clf.score(X[te], y[te]))
    return {
        "resub_knn1": round(float(knn1), 4),
        "resub_knn3": round(float(knn3), 4),
        "resub_knn5": round(float(knn5), 4),
        "loo_knn1": round(float(np.mean(loo_1)), 4),
        "loo_knn3": round(float(np.mean(loo_3)), 4),
        "svm_cv": round(float(np.mean(svm_accs)), 4),
    }


def joint_spectrum(model, readout, panel, tokenizer, vocab_token_ids, lora_param_list):
    """B0 fingerprint: scalar-margin Gram + joint (margin + K tokens) Gram, row-normalized."""
    N = len(panel)
    K = len(vocab_token_ids)
    all_grads_f16 = []
    for rec in panel:
        p = format_prompt(rec["percept"], rec.get("concepts", [])) + '{"decision": "'
        inp = tokenizer(p, return_tensors="pt").to(DEVICE)
        model.zero_grad(set_to_none=True)
        out = model(**inp, output_hidden_states=(readout is not None))
        last_logits = out.logits[0, -1, :]
        if readout is not None:
            hL = out.hidden_states[-1][0, -1, :]
            s = readout.forward_h(hL)
            dz = 2 * s
        else:
            dz = last_logits[PRO_ID] - last_logits[AB_ID]
        sample_grads = []
        g = torch.autograd.grad(dz, lora_param_list, retain_graph=True, allow_unused=True)
        sample_grads.append(torch.cat([x.reshape(-1) for x in g]))
        for t_k in vocab_token_ids:
            g = torch.autograd.grad(last_logits[t_k], lora_param_list, retain_graph=True, allow_unused=True)
            sample_grads.append(torch.cat([x.reshape(-1) for x in g]))
        model.zero_grad(set_to_none=True)
        all_grads_f16.append(torch.stack(sample_grads).half().cpu())
        del inp, out, last_logits, sample_grads
        torch.cuda.empty_cache()
    N_all = N
    NK = N * K
    G_margin = np.zeros((N_all, N_all), dtype=np.float64)
    G_joint = np.zeros((NK, NK), dtype=np.float64)
    for i in range(N_all):
        gi = all_grads_f16[i][0].float().numpy()
        for j in range(i, N_all):
            v = float(np.dot(gi, all_grads_f16[j][0].float().numpy()))
            G_margin[i, j] = v
            G_margin[j, i] = v
    for i in range(N_all):
        Ji = all_grads_f16[i][1:].float().numpy()
        for j in range(i, N_all):
            Jj = all_grads_f16[j][1:].float().numpy()
            block = Ji @ Jj.T
            for k1 in range(K):
                for k2 in range(K):
                    G_joint[i * K + k1, j * K + k2] = block[k1, k2]
                    G_joint[j * K + k2, i * K + k1] = block[k1, k2]
            del Jj
        del Ji
    del all_grads_f16
    gc.collect()
    row_norms = np.sqrt(np.diag(G_joint))
    norm_outer = np.outer(row_norms + 1e-12, row_norms + 1e-12)
    G_joint_normed = G_joint / norm_outer
    return {
        "scalar_margin_effective_rank": round(compute_effective_rank(G_margin), 4),
        "scalar_margin_top1_dominance": round(get_top1_dominance(G_margin), 4),
        "raw_joint_effective_rank": round(compute_effective_rank(G_joint), 4),
        "row_normalized_joint_effective_rank": round(compute_effective_rank(G_joint_normed), 4),
        "row_normalized_joint_top1_dominance": round(get_top1_dominance(G_joint_normed), 4),
        "raw_joint_top5_spectral": [round(float(x), 4) for x in
                                    np.sort(np.maximum(np.linalg.eigvalsh(G_joint), 0.0))[::-1][:5]],
    }


def margin_cohen_d(margins):
    mp, ma = margins[:8], margins[8:]
    pooled = float(np.sqrt(max((np.var(mp) + np.var(ma)) / 2.0, 1e-12)))
    return float((np.mean(mp) - np.mean(ma)) / pooled)


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------
def build_model(readout_init=None, tokenizer=None):
    torch.manual_seed(SEED_DATA)
    np.random.seed(SEED_DATA)
    torch.cuda.manual_seed_all(SEED_DATA)
    base_model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL_PATH), torch_dtype=torch.bfloat16,
                                                      device_map=DEVICE, trust_remote_code=True)
    base_model.config.use_cache = False
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                             lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_config)
    readout = None
    if readout_init is not None:
        a0, b0 = readout_init
        readout = ReadoutHead(a0, b0).to(DEVICE)
    return model, readout


def lora_state(model):
    return {k: v for k, v in model.named_parameters() if "lora_" in k and v.requires_grad}


def run_condition(condition, data, tokenizer, readout_init, u_lda, derive, artifact_subdir):
    t0 = time.time()
    print(f"\n{'='*70}\nCONDITION {condition} START\n{'='*70}", flush=True)
    cond_dir = artifact_subdir / f"condition_{condition}"
    cond_dir.mkdir(parents=True, exist_ok=True)

    model, readout = build_model(readout_init=readout_init)
    lora_params = lora_state(model)
    sorted_names = sorted(lora_params.keys())
    lora_param_list = [lora_params[n] for n in sorted_names]
    lora_init_hash = state_dict_hash(lora_params)

    trainable = [p for p in model.parameters() if p.requires_grad]
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
        "step0_panel16_margin_cohen_d": round(margin_cohen_d(panel16_margins0), 6),
        "telemetry": [],
        "checkpoints": {},
        "stopping_event": None,
    }
    if readout is not None:
        run["readout_init_hash"] = state_dict_hash(readout.get_params())

    # ---- step 0 checkpoint eval ----
    ckpt = evaluate_checkpoint(model, readout, tokenizer, data, panel16_ds, panel24_ds, u_lda, lora_param_list)
    run["checkpoints"]["0"] = ckpt
    save_checkpoint(model, readout, cond_dir, 0, condition)
    print(f"[{condition}] step 0: dev_prop={dev_prop0:.3f} cosH1={ckpt['cos_H1_mean_pairwise']:+.4f} "
          f"agg={ckpt['cos_H1_aggregate_613']:+.4f}", flush=True)

    # ---- training loop ----
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
        (loss / GRAD_ACCUM).backward()
        accum_loss += loss.item()

        if (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=CLIP_NORM)
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
            })
            accum_loss = 0.0

            if global_step in CHECKPOINTS and global_step > 0:
                model.eval()
                ckpt = evaluate_checkpoint(model, readout, tokenizer, data, panel16_ds, panel24_ds, u_lda, lora_param_list)
                ckpt["dev_proposal_rate"] = round(dev_prop, 6)
                run["checkpoints"][str(global_step)] = ckpt
                save_checkpoint(model, readout, cond_dir, global_step, condition)
                print(f"[{condition}] step {global_step:02d}: dev_prop={dev_prop:.3f} "
                      f"cosH1={ckpt['cos_H1_mean_pairwise']:+.4f} agg={ckpt['cos_H1_aggregate_613']:+.4f} "
                      f"margin_d={ckpt['panel16_margin_cohen_d']:+.3f}", flush=True)
                model.train()

            if ever_proposed and consecutive_zero >= COLLAPSE_K:
                run["stopping_event"] = {
                    "type": "collapse_guard",
                    "triggered_at_step": global_step,
                    "consecutive_zero_steps": consecutive_zero,
                    "detail": f"K={COLLAPSE_K} consecutive steps with 0.0% dev proposal rate",
                }
                print(f"[{condition}] COLLAPSE GUARD triggered at step {global_step}", flush=True)
                collapsed = True
                break

    run["reached_horizon"] = global_step
    run["collapsed"] = collapsed
    # t* = first step (after positive) where dev proposal hit 0
    t_star = None
    saw_pos = False
    for rec in run["telemetry"]:
        if rec["dev_proposal_rate"] > 0:
            saw_pos = True
        elif saw_pos and t_star is None:
            t_star = rec["step"]
    run["collapse_onset_t_star"] = t_star

    # margin jacobians at t=15 (and t=0) for r_eff / P3 / fingerprint
    model.eval()
    for tt in [0, 15]:
        if str(tt) in run["checkpoints"]:
            state = load_checkpoint(model, readout, cond_dir, tt, condition)
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
            torch.cuda.empty_cache()
    if readout is not None:
        run["readout_alignment_trajectory"] = {
            str(t): run["checkpoints"][str(t)]["readout_cos_lda"] for t in CHECKPOINTS if str(t) in run["checkpoints"]
        }

    run["elapsed_seconds"] = round(time.time() - t0, 1)
    vals = [run["checkpoints"][str(t)]["cos_H1_mean_pairwise"] for t in [4, 6, 14, 15]
            if str(t) in run["checkpoints"]]
    mval = round(float(np.mean(vals)), 4) if vals else None
    print(f"[{condition}] DONE in {run['elapsed_seconds']}s | steps={global_step} | t*={t_star} | "
          f"mean_cosH1(4,6,14,15)={mval}", flush=True)
    return run


def evaluate_checkpoint(model, readout, tokenizer, data, panel16_ds, panel24_ds, u_lda, lora_param_list):
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
    rec = {
        "panel16_margin_mean": round(float(m16.mean()), 6),
        "panel16_margin_median": round(float(np.median(m16)), 6),
        "panel16_margin_std": round(float(np.std(m16)), 6),
        "panel16_proposal_rate": round(float((m16 > 0).mean()), 6),
        "panel24_proposal_rate": round(float((m24 > 0).mean()), 6),
        "panel16_margin_cohen_d": round(margin_cohen_d(m16), 6),
        "cos_H1_mean_pairwise": cosh1["cos_H1_mean_pairwise"],
        "cos_H1_aggregate_613": cosh1["cos_H1_aggregate_613"],
        "rho_CD": cosh1["rho_CD"],
        "F_simultaneous_feasibility": cosh1["F_simultaneous_feasibility"],
        "separability_N16": separability_suite(X16, y16),
        "separability_N24": separability_suite(X24, y24),
    }
    if readout is not None:
        p = readout.get_params()
        a_np = p["a"].numpy()
        rec["readout_cos_lda"] = round(float(a_np @ u_lda / (np.linalg.norm(a_np) + 1e-12)), 6)
        rec["readout_norm"] = round(float(np.linalg.norm(a_np)), 6)
    return rec


def save_checkpoint(model, readout, cond_dir, step, condition):
    st = {k: v.detach().half().cpu().contiguous() for k, v in lora_state(model).items()}
    if readout is not None:
        rp = readout.get_params()
        st["__readout_a__"] = rp["a"]
        st["__readout_b__"] = rp["b"]
    buf = io.BytesIO()
    torch.save(st, buf)
    h = sha256_bytes(buf.getvalue())
    (cond_dir / f"checkpoint_step_{step:02d}.pt").write_bytes(buf.getvalue())
    meta = {"condition": condition, "step": step, "checkpoint_state_sha256": h}
    if readout is not None:
        meta["readout_state_sha256"] = state_dict_hash(readout.get_params())
    with open(cond_dir / f"checkpoint_step_{step:02d}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta


def load_checkpoint(model, readout, cond_dir, step, condition):
    st = torch.load(cond_dir / f"checkpoint_step_{step:02d}.pt", map_location="cpu")
    with torch.no_grad():
        for name, p in lora_state(model).items():
            p.copy_(st[name].to(DEVICE).float())
        if readout is not None and "__readout_a__" in st:
            readout.a.copy_(st["__readout_a__"].to(DEVICE))
            readout.b.copy_(st["__readout_b__"].to(DEVICE))
    return True


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def compute_step0_argmax_token(model, readout, tokenizer, panel0):
    p = format_prompt(panel0["percept"], panel0.get("concepts", [])) + '{"decision": "'
    inp = tokenizer(p, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
        top = int(torch.argmax(out.logits[0, -1, :]).item())
    return top


def run_joint_spectrum_B(data, tokenizer, u_lda, cond_dir, steps=(0, 15)):
    """B0 fingerprint: joint spectra at t=0 and t=15 (margin + 8 diagnostic tokens)."""
    model, readout = build_model(readout_init=None)
    lora_params = lora_state(model)
    sorted_names = sorted(lora_params.keys())
    lora_param_list = [lora_params[n] for n in sorted_names]
    load_checkpoint(model, None, cond_dir, 0, "B")
    model.eval()
    argmax_tok = compute_step0_argmax_token(model, None, tokenizer, data["pos_panel"][0])
    vocab_token_ids = DIAGNOSTIC_TOKEN_IDS + [argmax_tok]
    top_name = tokenizer.decode([argmax_tok]).strip() or f"tok_{argmax_tok}"
    vocab_token_names = TOKEN_NAMES + [f"top_unconstrained({top_name})"]
    results = {"vocab_token_ids": vocab_token_ids, "vocab_token_names": vocab_token_names, "spectra": {}}
    for tt in steps:
        if not (cond_dir / f"checkpoint_step_{tt:02d}.pt").exists():
            print(f"[JOINT-B] skip t={tt} (no checkpoint)", flush=True)
            continue
        load_checkpoint(model, None, cond_dir, tt, "B")
        model.eval()
        spec = joint_spectrum(model, None, data["pos_panel"] + data["abs_panel"],
                              tokenizer, vocab_token_ids, lora_param_list)
        results["spectra"][str(tt)] = spec
        print(f"[JOINT-B] t={tt} scalar_r_eff={spec['scalar_margin_effective_rank']} "
              f"rownorm_joint_r_eff={spec['row_normalized_joint_effective_rank']}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def compute_verdict(runs, derive):
    def cos_t(run, t):
        return run["checkpoints"].get(str(t), {}).get("cos_H1_mean_pairwise", None)

    def mean_cos(run):
        vals = [cos_t(run, t) for t in [4, 6, 14, 15] if cos_t(run, t) is not None]
        return float(np.mean(vals)) if vals else None

    B, I1, C2 = runs["B"], runs["I1"], runs["C2"]
    mB, mI1, mC2 = mean_cos(B), mean_cos(I1), mean_cos(C2)
    P1 = (mI1 is not None) and (mI1 >= P1_THRESH)
    C2x = (mI1 - mC2) >= C2X_THRESH if (mI1 is not None and mC2 is not None) else False

    prop_ok = all(r["dev_proposal_rate"] > 0 for r in I1["telemetry"])
    P2 = prop_ok and not I1["collapsed"]
    c15 = I1["checkpoints"].get("15", {})
    cohen_d15 = c15.get("panel16_margin_cohen_d", None)
    r_eff15 = c15.get("margin_jacobian_gram", {}).get("effective_rank", None)
    P3 = (cohen_d15 is not None and cohen_d15 >= P3_COHEN_D_THRESH) and \
         (r_eff15 is not None and r_eff15 <= P3_R_EFF_THRESH)

    def sep(run, t, key, panel):
        return run["checkpoints"].get(str(t), {}).get(f"separability_{panel}", {}).get(key, None)

    shared_t = sorted({int(k) for k in B["checkpoints"] if k.isdigit()} & {int(k) for k in I1["checkpoints"] if k.isdigit()})
    k1_ok = True
    loo_ok = True
    loo_pairs = {}
    for t in shared_t:
        for panel in ("N16", "N24"):
            k1 = sep(I1, t, "resub_knn1", panel)
            if k1 is not None and not (P4_KN1_RANGE[0] <= k1 <= P4_KN1_RANGE[1]):
                k1_ok = False
            loo_i1 = sep(I1, t, "loo_knn1", panel)
            loo_b = sep(B, t, "loo_knn1", panel)
            if loo_i1 is not None and loo_b is not None:
                d = abs(loo_i1 - loo_b)
                loo_pairs[f"t{t}_{panel}"] = [loo_i1, loo_b, round(d, 4)]
                if d > P4_LOO_TOL:
                    loo_ok = False
    P4 = k1_ok and loo_ok
    k1_16 = sep(I1, 0, "resub_knn1", "N16")
    k1_24 = sep(I1, 0, "resub_knn1", "N24")

    aligns = [abs(v) for t, v in I1.get("readout_alignment_trajectory", {}).items() if int(t) >= 4]
    P5 = max(aligns) >= P5_THRESH if aligns else False

    verdict = {
        "P1_conflict_reduction": {"passed": P1, "mean_cos_H1_I1": mI1, "threshold": P1_THRESH},
        "C2x_alignment_exclusion": {"passed": C2x, "delta": (mI1 - mC2) if (mI1 is not None and mC2 is not None) else None,
                                    "mean_cos_H1_C2": mC2, "threshold": C2X_THRESH},
        "P2_collapse_prevention": {"passed": P2, "collapsed": I1["collapsed"], "t_star": I1["collapse_onset_t_star"]},
        "P3_margin_maintenance": {"passed": P3, "cohen_d_15": cohen_d15, "r_eff_15": r_eff15,
                                  "thresholds": [P3_COHEN_D_THRESH, P3_R_EFF_THRESH]},
        "P4_no_separability_degradation": {"passed": P4, "resub_knn1_N16": k1_16, "resub_knn1_N24": k1_24,
                                           "loo_pairs_by_shared_checkpoint": loo_pairs,
                                           "loo_tolerance": P4_LOO_TOL, "shared_checkpoints": shared_t},
        "P5_readout_engagement": {"passed": P5, "max_abs_cos_lda": max(aligns) if aligns else None, "threshold": P5_THRESH},
    }

    if P1 and C2x and P4:
        row = "Alignment improves + conflict decreases + simultaneous feasibility improves (I1 beats both B and C2)"
        label = "H_CAUSAL_SUPPORTED"
    elif not P1:
        row = "No alignment benefit on primary endpoint (P1 fails)"
        label = "H0_SURVIVES"
    elif not C2x:
        row = "I1 and C2 improve similarly (C2x fails); capacity explanation strengthened"
        label = "CAPACITY_EXPLANATION_STRENGTHENED"
    elif P1 and not P2 and P4:
        row = "P1 holds, P2 fails, P4 holds (partial)"
        label = "H_CAUSAL_PARTIALLY_SUPPORTED"
    else:
        row = "Mixed/inconclusive"
        label = "MIXED_INCONCLUSIVE"
    verdict["decision_row"] = row
    verdict["decision_label"] = label
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start = time.time()
    print("=" * 70)
    print("PHASE 6E.17 — CAUSAL INTERVENTION (B0 / I1 / C2)  [protocol v2, APPROVED]")
    print("=" * 70, flush=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Pre-registration snapshot (frozen BEFORE any training)
    base_hash = sha256_file(BASE_MODEL_PATH / "model.safetensors")
    corpus_hash = sha256_file(CORPUS_PATH)
    assert base_hash == EXPECTED_BASE_HASH, "base model hash mismatch"
    assert corpus_hash == EXPECTED_CORPUS_HASH, "corpus hash mismatch"
    snapshot = {
        "phase": "6E.17",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_model_sha256": base_hash,
        "corpus_sha256": corpus_hash,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "review_sha256": sha256_file(REVIEW_PATH),
        "script_sha256": sha256_file(Path(__file__)),
        "frozen_config": {
            "PRO_ID": PRO_ID, "AB_ID": AB_ID, "D_HIDDEN": D_HIDDEN,
            "checkpoints": CHECKPOINTS, "horizon": HORIZON, "collapse_K": COLLAPSE_K,
            "seed_data": SEED_DATA, "seed_c2_readout": SEED_C2_READOUT,
            "lambda_decision": LAMBDA_DECISION, "grad_accum": GRAD_ACCUM, "batch_size": BATCH_SIZE,
            "lr": LR, "wd": WD, "clip_norm": CLIP_NORM,
            "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGETS},
            "P1_threshold": P1_THRESH, "C2x_threshold": C2X_THRESH,
        },
    }
    with open(ARTIFACT_DIR / "pre_registration_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    (ARTIFACT_DIR / "code_snapshot").mkdir(exist_ok=True)
    (ARTIFACT_DIR / "code_snapshot" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print("Pre-registration snapshot frozen:", flush=True)
    for k, v in snapshot.items():
        if k.endswith("_sha256"):
            print(f"  {k}: {v[:16]}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = reconstruct_data(CORPUS_PATH)
    print(f"Data: train={data['n_train']} dev={data['n_dev']} counts={data['counts']}", flush=True)

    derive, u_lda, a0, b0, c0, b0_c2, H16, margins_b = derive_readout_initialization(data, tokenizer)
    with open(ARTIFACT_DIR / "readout_initialization.json", "w", encoding="utf-8") as f:
        json.dump(derive, f, indent=2, sort_keys=True)
    print(f"[MAIN] s_scale={derive['s_scale']} b0={derive['b0']} b0_c2={derive['b0_c2']} "
          f"cos(n,lda)={derive['c2_readout_cos_n_lda']}", flush=True)

    conditions = sys.argv[1:] if len(sys.argv) > 1 else ["B", "I1", "C2"]
    runs = {}

    if "B" not in conditions:
        partial = ARTIFACT_DIR / "runs_partial_B.json"
        if partial.exists():
            runs["B"] = json.loads(partial.read_text(encoding="utf-8"))["B"]
            print(f"[MAIN] Loaded existing B run from runs_partial_B.json "
                  f"(checkpoints={sorted(runs['B']['checkpoints'].keys())}, t*={runs['B']['collapse_onset_t_star']})",
                  flush=True)

    if "B" in conditions:
        runB = run_condition("B", data, tokenizer, None, u_lda, derive, ARTIFACT_DIR)
        runs["B"] = runB
        # B0 fingerprint gate
        jointB = run_joint_spectrum_B(data, tokenizer, u_lda, ARTIFACT_DIR / "condition_B")
        with open(ARTIFACT_DIR / "B0_joint_spectra.json", "w", encoding="utf-8") as f:
            json.dump(jointB, f, indent=2, sort_keys=True)
        fp = {}
        agg = {t: runB["checkpoints"][str(t)]["cos_H1_aggregate_613"] for t in [4, 6] if str(t) in runB["checkpoints"]}
        fp["cos_agg_t4"] = agg.get(4)
        fp["cos_agg_t6"] = agg.get(6)
        fp["collapse_onset_t_star"] = runB["collapse_onset_t_star"]
        fp["scalar_r_eff_t15"] = runB["checkpoints"].get("15", {}).get("margin_jacobian_gram", {}).get("effective_rank")
        sp15 = jointB["spectra"].get("15", {})
        sp0 = jointB["spectra"].get("0", {})
        fp["joint_row_normalized_r_eff_t15"] = sp15.get("row_normalized_joint_effective_rank")
        fp["joint_row_normalized_r_eff_t0"] = sp0.get("row_normalized_joint_effective_rank")
        fp["joint_spectra_steps_present"] = sorted(jointB["spectra"].keys())
        fp["step0_argmax_token"] = jointB["vocab_token_ids"][-1]
        fp["max_abs_cos_wdelta_udisc_step0"] = max(abs(derive["cos_alignment_lda_vs_w_delta"]),
                                                   abs(derive["cos_alignment_centroid_vs_w_delta"]),
                                                   abs(derive["cos_alignment_svm_vs_w_delta"]))
        gate_ok = (fp["cos_agg_t6"] is not None and fp["cos_agg_t6"] <= -0.90) and \
                  (fp["collapse_onset_t_star"] in (14, 15, 16)) and \
                  (fp["scalar_r_eff_t15"] is not None and 0.9 <= fp["scalar_r_eff_t15"] <= 1.8) and \
                  (fp["joint_row_normalized_r_eff_t15"] is not None and fp["joint_row_normalized_r_eff_t15"] >= 2.8) and \
                  (fp["max_abs_cos_wdelta_udisc_step0"] <= 0.06)
        fp["gate_pass"] = bool(gate_ok)
        with open(ARTIFACT_DIR / "B0_fingerprint_gate.json", "w", encoding="utf-8") as f:
            json.dump(fp, f, indent=2, sort_keys=True)
        print(f"\n[B0 FINGERPRINT] agg_cos(t4)={fp['cos_agg_t4']:+.4f} agg_cos(t6)={fp['cos_agg_t6']:+.4f} "
              f"t*={fp['collapse_onset_t_star']} scalar_r_eff(15)={fp['scalar_r_eff_t15']} "
              f"joint_r_eff(15)={fp['joint_row_normalized_r_eff_t15']} "
              f"maxcos(w_d,u_disc)={fp['max_abs_cos_wdelta_udisc_step0']:.4f} GATE={fp['gate_pass']}", flush=True)
        if not gate_ok:
            print("[HALT] B0 control fingerprint FAILED. Halting before I1/C2. Revise protocol per section 6.", flush=True)
            json.dump(runs, open(ARTIFACT_DIR / "runs_partial_B.json", "w", encoding="utf-8"), indent=2, sort_keys=True, default=str)
            sys.exit(1)

    if "I1" in conditions:
        runs["I1"] = run_condition("I1", data, tokenizer, (a0, b0), u_lda, derive, ARTIFACT_DIR)
    if "C2" in conditions:
        runs["C2"] = run_condition("C2", data, tokenizer, (c0, b0_c2), u_lda, derive, ARTIFACT_DIR)

    # serializable runs dump
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    runs_clean = clean(runs)
    with open(ARTIFACT_DIR / "runs.json", "w", encoding="utf-8") as f:
        json.dump(runs_clean, f, indent=2, sort_keys=True)

    verdict = compute_verdict(runs, derive)
    with open(ARTIFACT_DIR / "verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, sort_keys=True)
    print(f"\n[VERDICT] {verdict['decision_label']} | {verdict['decision_row']}", flush=True)
    for pid, rec in verdict.items():
        if isinstance(rec, dict) and "passed" in rec:
            print(f"  {pid}: PASS={rec['passed']}", flush=True)

    # output manifest
    manifest = {
        "phase": "6E.17",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_model_sha256_after": sha256_file(BASE_MODEL_PATH / "model.safetensors"),
        "corpus_sha256_after": sha256_file(CORPUS_PATH),
        "elapsed_seconds": round(time.time() - start, 1),
        "documented_non_voiding_discrepancies": [
            {
                "id": "STEP0_PROFILE_MISMATCH",
                "detail": "Frozen b0 formula centers I1 step-0 panel proposal at 25% vs B at 0%; "
                          "non-voiding, formula kept byte-for-byte per human decision.",
            },
            {
                "id": "B0_COLLAPSE_ONSET_DETECTOR_MISMATCH",
                "detail": "B0 margin-sign dev collapse detector fired at t*=4 (guard at step 6) vs frozen "
                          "t* in {14,15,16}; caused by 13/52 POS ceiling of margin-sign proposal on the "
                          "ABS-heavy dev split. cos(H1) fingerprint reproduced (agg t6=-0.997). "
                          "Declared non-voiding per human decision; I1/C2 executed; B truncated at step 6.",
            },
        ],
        "artifacts": [],
    }
    for p in sorted(ARTIFACT_DIR.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(ARTIFACT_DIR)).replace("\\", "/")
            manifest["artifacts"].append({"path": rel, "sha256": sha256_file(p)})
    with open(ARTIFACT_DIR / "sha256_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("\n" + "=" * 70)
    print("PHASE 6E.17 COMPLETE. HARD STOP. RETURNING FOR HUMAN REVIEW.")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
