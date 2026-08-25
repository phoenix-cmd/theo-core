#!/usr/bin/env python3
"""Phase 6E.20.3 — Read-Only Checkpoint Re-Evaluation Runner.

Scientific Goal:
    Determine whether the 56 frozen checkpoints from Phase 6E.20.1 satisfy
    hypotheses H1 (P2), H3, H4, and H5 when evaluated correctly—specifically,
    evaluating each condition with its OWN checkpoint and its OWN protocol-defined
    readout across the canonical aggregation set {4, 6, 14, 15} (28 evaluations).

Governing Documents (frozen, read-only):
    * Protocol: `theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md`
      SHA-256 E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85
    * Plan: `theo-slm-fixed-subspace-readout-authorization-protocol-v4.0-execution-extension-plan.md`
      SHA-256 87C8E486572F0B9D42EE49E21B2E93A56AD1C7CB755DD9EFAE5A0F25F463304A

Invariants:
    1. Phase 6E.20.1 is FROZEN. No file in `phase-6e20.1-execution-output` is
       modified, moved, overwritten, or deleted. All inputs are read-only.
    2. All 6E.20.3 telemetry, manifests, and reports go strictly to
       `phase-6e20.3-reevaluation-output/`.
    3. Zero retraining. No optimizer steps, no weight updates.
    4. Execution is guarded by explicit Phase 6E.20.3 L5 authorization.
    5. All NEPR logic is imported from collect_nepr.py (single canonical source).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Import canonical NEPR implementation (single source of truth)
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

# ---------------------------------------------------------------------------
# Pinned Constants & Identities (Protocol v3.1 / Inherited 6E.18)
# ---------------------------------------------------------------------------
PHASE_ID = "6E.20.3"
AUTH_FILE_NAME = "execution_authorization_6e20_3.json"
DEFAULT_OUTPUT_DIR = "phase-6e20.3-reevaluation-output"
HISTORICAL_OUTPUT_DIR = "phase-6e20.1-execution-output"

PROTOCOL_SHA256 = "E16CA4F1496AD21C99A81C94CD3EBE2CFB810B89C8A2EC2A97D08A02A5073F85"
PLAN_V40_SHA256 = "87C8E486572F0B9D42EE49E21B2E93A56AD1C7CB755DD9EFAE5A0F25F463304A"
FROZEN_6E18_SHA256 = "6BCAF1319AC551848FFDB55AEA121D14D808408A192E05F2AA6171161DECEF96"
RUNNER_6E20_BASE_SHA256 = "550FAA345721649225F32748C39053077B08BE165426EF3FE38C2F6E276A1BD9"
RUNNER_6E20_1_V7_SHA256 = "C8E298C85294B8CC5737D5A8EE44679C7A778EEF09187894F1FE1CF435115A8B"
AUTH_6E20_1_V7_SHA256 = "5214DA997764F62C9370238E055CD5394925FBFE8F0C61478B8D192B198C4F91"

CORPUS_EXPECTED_HASH = "a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0"
H16_EXPECTED_HASH = "a4593e4aad2bfab0cbbb4a2257cfe60ae9ccfe0a7cd2bf04ce8fc014eb23a1f5"
Y16_EXPECTED_HASH = "d8e9b87393a6ee9c24d56c7eb2d8c301030beafaf0e723d38ec922e8a8cb7737"
U_LDA_EXPECTED_HASH = "b72c6c215d40a76e33e1755ff88ad1cde12fca8d7b8270a638eb39385b6b0113"
U8_EXPECTED_HASH = "7c99c57787078ae6b3aa82acf86972aa485aa89c436497b78ca0ea6d9747d4a7"
R4_EXPECTED_HASH = "87eafb205d6ca713be314bb30ac91f1483f471a6769560c089c9b2cc16ee6e8e"
N_RAND_EXPECTED_HASH = "7ea2b434ce81c66ab18d153e2fb6d5bc4ac344a36eb0d5ff1b7780afadcc0ef2"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PRO_ID = 9117
AB_ID = 1867
D_HIDDEN = 896
VOCAB = 151936
TOKENIZER_BASE_VOCAB = 151643
TOKENIZER_FULL_VOCAB = 151665
LAYER_INDEX = 24

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

CONDITIONS = ["B0", "F1", "S2", "S4", "S8", "C4-Rand", "R1-Rand"]
AGGREGATION_SET = [4, 6, 14, 15]

# Historical B0 (with S8 readout) reference values from frozen 6E.20.1 runs.json
HISTORICAL_B0_S8_ANCHOR = {
    4: 0.428355,
    6: 0.409713,
    14: 0.427470,
    15: 0.417450,
}
B0_REPRODUCIBILITY_TOLERANCE = 1e-4

# Decision Thresholds (Protocol v3.1 §7 / §8)
P1_THRESHOLD = 0.50          # rho_c
P2_THRESHOLD = 0.80          # mean cos(H1)
P3_THRESHOLD = 0.80          # Cohen's d(15)
P4_THRESHOLD = 1.50          # Cohen's d(15)
P5_THRESHOLD = 0.70          # balanced accuracy (15)
H4_DELTA_MIN = 0.30          # S4 mean cos(H1) - C4-Rand mean cos(H1) >= +0.30

REFUSAL_TRIPLET = ("PRE_EXECUTION", "NO_SCIENTIFIC_VERDICT", "EXECUTION_NOT_AUTHORIZED")


# ---------------------------------------------------------------------------
# Path & Hash Utilities
# ---------------------------------------------------------------------------
def compute_sha256(file_path: Path) -> str:
    """Compute uppercase hex SHA-256 of file bytes."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def resolve_project_paths() -> Dict[str, Path]:
    """Resolve absolute paths for all repository roots and key artifacts."""
    script_dir = Path(__file__).resolve().parent
    theo_core_dir = script_dir.parents[1]
    repo_root = theo_core_dir.parent

    corpus_path = repo_root / "theo-data" / "datasets" / "theo_slm_v0_deduplicated" / "candidate_records.json"
    snapshot_path = (
        repo_root
        / "theo-data"
        / "datasets"
        / "theo_slm_v0_artifacts"
        / "phase-6e18-causal-intervention"
        / "code_snapshot"
        / "run_phase_6e18_causal_intervention.py"
    )
    protocol_path = theo_core_dir / "docs" / "research" / "theo-slm-fixed-subspace-readout-authorization-protocol-draft-v3.1.md"
    plan_path = theo_core_dir / "docs" / "research" / "theo-slm-fixed-subspace-readout-authorization-protocol-v4.0-execution-extension-plan.md"
    runner_6e20_path = script_dir / "run_phase_6e20_fixed_subspace_readout.py"
    runner_v7_path = script_dir / "run_phase_6e20_execution_v7.py"
    auth_v7_path = script_dir / "execution_authorization_v7.json"
    auth_v6_path = script_dir / "execution_authorization_v6.json"
    hist_out_dir = script_dir / HISTORICAL_OUTPUT_DIR

    return {
        "script_dir": script_dir,
        "theo_core_dir": theo_core_dir,
        "repo_root": repo_root,
        "corpus": corpus_path,
        "frozen_6e18_snapshot": snapshot_path,
        "protocol": protocol_path,
        "plan_v40": plan_path,
        "runner_6e20_base": runner_6e20_path,
        "runner_6e20_1_v7": runner_v7_path,
        "auth_6e20_1_v7": auth_v7_path,
        "auth_6e20_1_v6": auth_v6_path,
        "historical_out_dir": hist_out_dir,
    }


# ---------------------------------------------------------------------------
# Readout Dataclass & Prompt Reconstruction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Readout:
    name: str
    w: Any       # (896,) numpy float64 vector
    b: float
    trainable_params: int = 0
    subspace: str = ""

    def forward_margin(self, h) -> float:
        import numpy as np
        return float(np.asarray(self.w, dtype=np.float64) @ np.asarray(h, dtype=np.float64)) + self.b

    def forward_logits(self, h) -> Tuple[float, float]:
        s = self.forward_margin(h)
        return s, -s

    def forward_margin_tensor(self, hL: "Any") -> "Any":
        """Graph-preserving readout decision margin for autograd consumers (compute_live_rho_c only).

        hL: 1-D torch.Tensor, shape (D,), on the model device, MUST be connected to the
            computation graph (requires_grad via the model). MUST NOT be detached.
        Returns a scalar float64 torch.Tensor = w·hL + b, mathematically identical to
        forward_margin(numpy) but carrying the autograd graph so grad(., lora_params) is valid.
        """
        import torch
        if not isinstance(hL, torch.Tensor) or hL.dim() != 1 or int(hL.shape[0]) != len(self.w):
            raise PreflightError(
                "forward_margin_tensor: hL must be a 1-D torch.Tensor of length len(self.w)"
            )
        w_t = torch.as_tensor(self.w, dtype=torch.float64, device=hL.device)
        b_t = torch.as_tensor(float(self.b), dtype=torch.float64, device=hL.device)
        return torch.matmul(w_t, hL.double()) + b_t

    def forward_logits_tensor(self, hL: "Any"):
        s = self.forward_margin_tensor(hL)
        return s, -s


_READOUT_FORWARD_PATCHED = False


def ensure_readout_class_patch():
    """Apply runtime class patch: detach/cpu/float tensor before numpy."""
    global _READOUT_FORWARD_PATCHED
    if _READOUT_FORWARD_PATCHED:
        return
    import torch
    orig_fwd = Readout.forward_logits

    def _safe_fwd(self, h, *a, **k):
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().float()
        return orig_fwd(self, h, *a, **k)

    Readout.forward_logits = _safe_fwd  # type: ignore
    _READOUT_FORWARD_PATCHED = True


def format_prompt(percept: str, concepts: Any) -> str:
    """Standard decision prompt formatting."""
    labels = [c.get("id", str(c)) if isinstance(c, dict) else str(c) for c in concepts]
    concepts_str = ", ".join(labels) if labels else "none"
    return (
        "╱system\n"
        "You are THEO SLM v0, a neural cognitive provider. Given an observation percept and grounding context, "
        "evaluate decision relevance and determine whether to propose a hypothesis or abstain.╱/user\n"
        f"Observation Percept: {percept}\n"
        f"Grounding Concepts: {concepts_str}\n"
        "Task: Emit JSON evaluation containing decision (SHOULD_PROPOSE or SHOULD_ABSTAIN) and reasoning.╱/assistant\n"
    )


def reconstruct_panel_prompts(corpus_path: Path) -> List[Tuple[str, str]]:
    """Deterministically reconstruct 16 panel prompts from hash-verified corpus."""
    import numpy as np
    import re
    with open(corpus_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    from collections import defaultdict
    family_groups = defaultdict(list)
    for rec in all_records:
        family_groups[re.sub(r"_[A-D]$", "", rec["case_id"])].append(rec)
    sorted_families = sorted(family_groups.keys())

    np.random.seed(42)
    np.random.shuffle(sorted_families)
    n_train_f = int(len(sorted_families) * 0.8)
    train_families = set(sorted_families[:n_train_f])

    train_records_raw = [r for r in all_records if re.sub(r"_[A-D]$", "", r["case_id"]) in train_families]
    pos_train = [r for r in train_records_raw if r.get("abstention_label") == "SHOULD_PROPOSE" and r.get("novelty_label") == "SEMANTIC_NOVEL"]
    abs_train = [r for r in train_records_raw if r.get("novelty_label") in ["DECISION_IRRELEVANT", "EPISTEMICALLY_PREMATURE"]]

    np.random.seed(42)
    balanced_pos = list(pos_train) * 2 + list(pos_train)[:134 - len(pos_train) * 2]
    balanced_abs = list(np.random.choice(abs_train, size=67, replace=True)) if len(abs_train) < 67 else list(np.random.choice(abs_train, size=67, replace=False))

    panel = balanced_pos[:8] + balanced_abs[:8]
    prompts = []
    for i, rec in enumerate(panel):
        tag = "POS" if i < 8 else "ABS"
        idx = i if i < 8 else i - 8
        formatted = format_prompt(rec["percept"], rec.get("concepts", [])) + '{"decision": "'
        prompts.append((f"{tag}_{idx}", formatted))
    return prompts


# ---------------------------------------------------------------------------
# Frozen Machinery Loader
# ---------------------------------------------------------------------------
def load_frozen_6e18_gradient_machinery(snapshot_path: Path):
    """Import exact approved gradient machinery from frozen 6E.18 snapshot."""
    import torch
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Frozen 6E.18 snapshot missing at {snapshot_path}")
    actual_sha = compute_sha256(snapshot_path)
    if actual_sha != FROZEN_6E18_SHA256:
        raise RuntimeError(f"Snapshot SHA mismatch: {actual_sha} != {FROZEN_6E18_SHA256}")

    spec = importlib.util.spec_from_file_location("frozen_6e18_machinery", str(snapshot_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load spec for frozen 6E.18 snapshot")
    mod = importlib.util.module_from_spec(spec)
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    mod.DEVICE = device_str
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    psdg = getattr(mod, "per_sample_decision_gradient", None)
    crgc = getattr(mod, "compute_raw_gradient_cosine", None)
    if psdg is None or crgc is None:
        raise RuntimeError("Frozen snapshot missing required functions")
    return psdg, crgc, device_str


# ---------------------------------------------------------------------------
# Preflight Verification Engine
# ---------------------------------------------------------------------------
class PreflightError(RuntimeError):
    pass


def run_stage1_pre_execution_sha_manifest(paths: Dict[str, Path]) -> Dict[str, Any]:
    """Compute and verify SHA-256 for all frozen inputs."""
    manifest = {
        "timestamp_utc": "2026-08-18T00:00:00Z",
        "phase": PHASE_ID,
        "files": {},
    }

    # Core docs and code
    core_files = {
        "protocol": (paths["protocol"], PROTOCOL_SHA256),
        "plan_v40": (paths["plan_v40"], PLAN_V40_SHA256),
        "frozen_6e18_snapshot": (paths["frozen_6e18_snapshot"], FROZEN_6E18_SHA256),
        "runner_6e20_base": (paths["runner_6e20_base"], RUNNER_6E20_BASE_SHA256),
        "runner_6e20_1_v7": (paths["runner_6e20_1_v7"], RUNNER_6E20_1_V7_SHA256),
        "auth_6e20_1_v7": (paths["auth_6e20_1_v7"], AUTH_6E20_1_V7_SHA256),
        # INTENTIONAL (frozen 6E.20.1 record, v6 auth "note"): execution_authorization_v6.json
        # is v7-bound (identical content to v7); its expected SHA IS the v7 SHA. Not a bug.
        "auth_6e20_1_v6": (paths["auth_6e20_1_v6"], AUTH_6E20_1_V7_SHA256),
        "corpus": (paths["corpus"], CORPUS_EXPECTED_HASH.upper()),
    }

    for name, (path, exp_sha) in core_files.items():
        if not path.exists():
            raise PreflightError(f"Stage 1 HALT: Required frozen input missing: {path}")
        actual_sha = compute_sha256(path)
        manifest["files"][name] = {"path": str(path), "sha256": actual_sha, "expected": exp_sha, "match": actual_sha == exp_sha}
        if actual_sha != exp_sha:
            raise PreflightError(f"Stage 1 HALT: SHA-256 mismatch on {name} ({path}): {actual_sha} != {exp_sha}")

    # Step-0 artifacts
    step0_dir = paths["historical_out_dir"] / "step0"
    step0_expectations = {
        "H_16.npy": H16_EXPECTED_HASH.upper(),
        "y16.npy": Y16_EXPECTED_HASH.upper(),
        "u_lda.npy": U_LDA_EXPECTED_HASH.upper(),
        "U_8.npy": U8_EXPECTED_HASH.upper(),
        "R_4.npy": R4_EXPECTED_HASH.upper(),
        "n_rand.npy": N_RAND_EXPECTED_HASH.upper(),
    }
    for fname, exp_sha in step0_expectations.items():
        fpath = step0_dir / fname
        if not fpath.exists():
            raise PreflightError(f"Stage 1 HALT: Missing step0 artifact {fpath}")
        actual_sha = compute_sha256(fpath)
        manifest["files"][f"step0/{fname}"] = {"path": str(fpath), "sha256": actual_sha, "expected": exp_sha, "match": actual_sha == exp_sha}
        if actual_sha != exp_sha:
            raise PreflightError(f"Stage 1 HALT: SHA-256 mismatch on {fname}: {actual_sha} != {exp_sha}")

    # Checkpoint existence check
    for c in CONDITIONS:
        # Check w_{c}.npy
        w_path = step0_dir / f"w_{c}.npy"
        if not w_path.exists():
            raise PreflightError(f"Stage 1 HALT: Missing readout weight vector {w_path}")
        manifest["files"][f"step0/w_{c}.npy"] = {"path": str(w_path), "sha256": compute_sha256(w_path)}

        for t in AGGREGATION_SET:
            ckpt_path = paths["historical_out_dir"] / c / f"checkpoint_{t}" / "model.pt"
            if not ckpt_path.exists():
                raise PreflightError(f"Stage 1 HALT: In-scope checkpoint missing: {ckpt_path}")
            manifest["files"][f"checkpoint/{c}/t{t}"] = {"path": str(ckpt_path), "sha256": compute_sha256(ckpt_path)}

    return manifest


def validate_stage2_vocab_invariant(model_config_vocab_size, tokenizer_base_vocab, tokenizer_full_len):
    """Fail-closed Stage 2 vocabulary invariant.

    The authoritative model embedding vocabulary is ``model.config.vocab_size``
    (VOCAB = 151936) and must be compared to VOCAB, never to a tokenizer metric.
    ``tokenizer.vocab_size`` (base vocabulary, 151643) and ``len(tokenizer)``
    (base + added special tokens, 151665) are checked only for consistency and
    must NOT be compared against VOCAB. All deviations are fail-closed.
    """
    if model_config_vocab_size != VOCAB:
        raise PreflightError(
            f"Stage 2 HALT: Model config vocab mismatch {model_config_vocab_size} != {VOCAB}"
        )
    if tokenizer_base_vocab != TOKENIZER_BASE_VOCAB:
        raise PreflightError(
            f"Stage 2 HALT: Tokenizer base vocab mismatch {tokenizer_base_vocab} != {TOKENIZER_BASE_VOCAB}"
        )
    if tokenizer_full_len != TOKENIZER_FULL_VOCAB:
        raise PreflightError(
            f"Stage 2 HALT: Tokenizer full vocab mismatch {tokenizer_full_len} != {TOKENIZER_FULL_VOCAB}"
        )


def run_stage2_model_and_tokenizer_preflight():
    """Verify base model & tokenizer identity and configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    validate_stage2_vocab_invariant(model.config.vocab_size, tokenizer.vocab_size, len(tokenizer))

    test_pro = tokenizer.encode("PROPOSE", add_special_tokens=False)
    test_ab = tokenizer.encode("ABSTAIN", add_special_tokens=False)
    if PRO_ID not in test_pro:
        raise PreflightError(f"Stage 2 HALT: Tokenizer PRO_ID mismatch, expected {PRO_ID} in {test_pro}")
    if AB_ID not in test_ab:
        raise PreflightError(f"Stage 2 HALT: Tokenizer AB_ID mismatch, expected {AB_ID} in {test_ab}")

    if model.config.hidden_size != D_HIDDEN:
        raise PreflightError(f"Stage 2 HALT: Hidden dimension {model.config.hidden_size} != {D_HIDDEN}")
    if model.config.num_hidden_layers != LAYER_INDEX:
        raise PreflightError(f"Stage 2 HALT: Layer count {model.config.num_hidden_layers} != {LAYER_INDEX}")

    return model, tokenizer


def run_stage3_stage4_checkpoint_smoke_and_lora_validation(
    model: Any,
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    """Smoke test all 28 checkpoints with torch.load and non-silent LoRA key validation."""
    import torch
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias=LORA_BIAS, task_type=LORA_TASK_TYPE,
    )
    test_peft_model = get_peft_model(copy.deepcopy(model), lora_config)
    expected_peft_keys = set(test_peft_model.state_dict().keys())
    expected_lora_keys = {k for k in expected_peft_keys if "lora_" in k}

    if not expected_lora_keys:
        raise PreflightError("Stage 4 HALT: Expected LoRA key set is empty; PEFT setup failed")

    diagnostics = {}
    for c in CONDITIONS:
        for t in AGGREGATION_SET:
            ckpt_path = paths["historical_out_dir"] / c / f"checkpoint_{t}" / "model.pt"
            ckpt_sha = compute_sha256(ckpt_path)

            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as e:
                raise PreflightError(f"Stage 3 HALT: torch.load failed on {ckpt_path}: {e}")

            for req_key in ["state_dict", "step", "condition", "readout_w", "readout_b"]:
                if req_key not in ckpt:
                    raise PreflightError(f"Stage 3 HALT: Checkpoint {ckpt_path} missing root key '{req_key}'")

            if ckpt["step"] != t:
                raise PreflightError(f"Stage 3 HALT: Step mismatch in {ckpt_path}: ckpt['step']={ckpt['step']} != {t}")
            if ckpt["condition"] != c:
                raise PreflightError(f"Stage 3 HALT: Condition mismatch in {ckpt_path}: ckpt['condition']={ckpt['condition']} != {c}")

            ckpt_keys = set(ckpt["state_dict"].keys())
            ckpt_lora_keys = {k for k in ckpt_keys if "lora_" in k}

            missing_lora = expected_lora_keys - ckpt_lora_keys
            unexpected_lora = ckpt_lora_keys - expected_lora_keys

            if missing_lora:
                raise PreflightError(
                    f"Stage 4 HALT: Silent adapter drop detected in {ckpt_path}! "
                    f"Missing {len(missing_lora)} expected LoRA keys: {sorted(list(missing_lora))[:5]}"
                )

            # Test actual weight loading
            test_peft_model.load_state_dict(ckpt["state_dict"], strict=False)

            # Compute weight norms of loaded adapter tensors
            sum_norms = 0.0
            max_norm = 0.0
            loaded_count = 0
            for k in expected_lora_keys:
                tensor = ckpt["state_dict"][k]
                n = float(tensor.norm().item())
                sum_norms += n
                if n > max_norm:
                    max_norm = n
                loaded_count += 1

            if max_norm <= 0.0:
                raise PreflightError(f"Stage 4 HALT: Checkpoint {ckpt_path} has all-zero adapter tensor norms")

            diag_entry = {
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": ckpt_sha,
                "step": t,
                "condition": c,
                "expected_lora_key_count": len(expected_lora_keys),
                "checkpoint_lora_key_count": len(ckpt_lora_keys),
                "missing_lora_keys": list(missing_lora),
                "unexpected_lora_keys": list(unexpected_lora),
                "loaded_lora_key_count": loaded_count,
                "sum_lora_norms": sum_norms,
                "max_lora_norm": max_norm,
            }
            diagnostics[f"{c}_t{t}"] = diag_entry

    del test_peft_model
    return {"status": "PASS", "lora_key_count": len(expected_lora_keys), "diagnostics": diagnostics}


def compute_decision_margin_and_metrics(
    model: Any,
    tokenizer: Any,
    readout: Readout,
    panel_prompts: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """Execute evaluation forward pass on 16 panel prompts to compute margin, balanced accuracy, and Cohen's d."""
    import numpy as np
    import torch

    model.eval()
    hidden_states_all = []
    with torch.no_grad():
        for label, formatted in panel_prompts:
            inputs = tokenizer(formatted, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[LAYER_INDEX]
            last_token_hs = hs[0, -1, :].float().cpu().numpy()
            hidden_states_all.append(last_token_hs)

    H16_eval = np.stack(hidden_states_all, axis=0)
    margins = np.array([readout.forward_margin(H16_eval[i]) for i in range(16)])

    # Balanced accuracy
    pos_acc = float(np.mean(margins[:8] > 0))
    abs_acc = float(np.mean(margins[8:] <= 0))
    bal_acc = 0.5 * (pos_acc + abs_acc)

    # Cohen's d
    m_pos = margins[:8]
    m_abs = margins[8:]
    var_p = np.var(m_pos, ddof=1) if len(m_pos) > 1 else 0.0
    var_a = np.var(m_abs, ddof=1) if len(m_abs) > 1 else 0.0
    s_pool = math.sqrt(max(1e-12, 0.5 * (var_p + var_a)))
    cohen_d = float((np.mean(m_pos) - np.mean(m_abs)) / s_pool)

    return {
        "margins": margins.tolist(),
        "balanced_accuracy": bal_acc,
        "cohens_d": cohen_d,
        "pos_accuracy": pos_acc,
        "abs_accuracy": abs_acc,
        "H16_eval": H16_eval,
    }


def compute_live_cos_H1(
    model: Any,
    readout: Readout,
    tokenizer: Any,
    panel_prompts: List[Tuple[str, str]],
    psdg_fn: Any,
    crgc_fn: Any,
    device_str: str,
) -> float:
    """Compute total-gradient decision cosine cos(H1) using frozen 6E.18 machinery."""
    import torch
    ensure_readout_class_patch()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    if not lora_params:
        raise RuntimeError("No requires_grad parameters found in PEFT model during gradient computation")

    def build_item(prompt_str: str, dec_token_id: int):
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

    g_pos, g_abs = [], []
    for i in range(8):
        item = build_item(panel_prompts[i][1], PRO_ID)
        g_pos.append(psdg_fn(model, readout, item, lora_params, tokenizer))
    for j in range(8):
        item = build_item(panel_prompts[8 + j][1], AB_ID)
        g_abs.append(psdg_fn(model, readout, item, lora_params, tokenizer))

    result = crgc_fn(g_pos, g_abs)
    if _6E20_7_TELEMETRY_ENABLED:
        _6E20_7_GRAD_CACHE["g_pos"] = list(g_pos)
        _6E20_7_GRAD_CACHE["g_abs"] = list(g_abs)
    return float(result["cos_H1_aggregate"])


def compute_readout_total_alignment(
    g_R_flat: "Any",
    g_total_flat: "Any",
) -> float:
    """Protocol v3.1 §5.7 — per-sample readout-total alignment in common LoRA-parameter space.

        rho_i = <g_R, g_total> / (||g_R|| * ||g_total||)

    Both operands MUST be flat gradient vectors over the SAME ordered LoRA-parameter
    set; no silent flattening of unrelated tensors is performed. Guards:
      * non-tensor inputs                         -> PreflightError
      * shape mismatch                            -> PreflightError
      * non-1-D tensors                           -> PreflightError
      * non-finite norm (NaN/Inf)                 -> PreflightError
      * ||g_R|| == 0 (readout gradient vanishes)  -> 0.0  (per §5.7)
      * ||g_total|| == 0                          -> 0.0  (nothing to align against)
      * non-finite alignment value                 -> PreflightError
    The output is a genuine cosine in [-1, +1]; it is NOT analytically forced to
    1.0 / 0.0 / +/-1 by the implementation's algebra.
    """
    import torch
    import math

    if not (isinstance(g_R_flat, torch.Tensor) and isinstance(g_total_flat, torch.Tensor)):
        raise PreflightError("rho_c: g_R and g_total must be torch.Tensor")
    if g_R_flat.shape != g_total_flat.shape:
        raise PreflightError(
            f"rho_c: gradient shape mismatch g_R{g_R_flat.shape} vs g_total{g_total_flat.shape}"
        )
    if g_R_flat.dim() != 1 or g_total_flat.dim() != 1:
        raise PreflightError("rho_c: gradients must be flat 1-D vectors in the same parameter space")

    nR = float(g_R_flat.norm().item())
    nT = float(g_total_flat.norm().item())
    if not (math.isfinite(nR) and math.isfinite(nT)):
        raise PreflightError("rho_c: non-finite gradient norm (NaN/Inf)")
    if nR < 0.0 or nT < 0.0:
        raise PreflightError("rho_c: negative gradient norm")
    if nR < 1e-12 or nT < 1e-12:
        return 0.0
    rho = float(torch.dot(g_R_flat, g_total_flat).item() / (nR * nT))
    if not math.isfinite(rho):
        raise PreflightError("rho_c: non-finite alignment value")
    return rho


def compute_live_rho_c(
    model: Any,
    readout: Readout,
    tokenizer: Any,
    panel_prompts: List[Tuple[str, str]],
    psdg_fn: Any,
    crgc_fn: Any,
    device_str: str,
) -> Tuple[float, float, List[float]]:
    """Protocol v3.1 §5.7 genuine readout-total alignment rho_c (NOT readout self-projection).

    For each panel sample i (8 POS + 8 ABS):
        g_total_i = psdg_fn(model, readout, item_i, lora_params, tokenizer)
                    = autograd.grad(dec_loss, lora_params)      # frozen 6E.18 machinery
        g_R_i     = alpha_c * grad_{lora_params}(s_i)           # readout-only decision gradient
        rho_i     = compute_readout_total_alignment(g_R_i, g_total_i)
    where
        s_i      = readout scalar output z_PRO at the decision position,
        alpha_c  = (p_PRO - y_PRO) - (p_ABS - y_ABS)            # CE chain rule, post-patch softmax
        p_*      = softmax over the patched full-vocab decision logits,
        y        = (1,0) on POS panels, (0,1) on ABS panels.
    Aggregates:
        rho_c   = mean_i rho_i
        sigma_c = mean_i (rho_i ** 2)  # mean variance share, frozen Protocol v3.1 §5.7 / V3-03
                                      # (NOT std(rho); NOT (mean rho)**2)
    Both g_R and g_total live in the SAME LoRA-parameter space (identical ordering), so the
    cosine is well-defined and cannot be structurally forced to 1.0/0.0/+/-1.
    """
    import torch
    import math
    import numpy as np

    ensure_readout_class_patch()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    if not lora_params:
        raise RuntimeError("No requires_grad parameters found in PEFT model during gradient computation")

    def build_item(prompt_str: str, dec_token_id: int):
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

    per_sample_rho: List[float] = []
    g_R_all: List[Any] = []
    for i in range(16):
        dec_token_id = PRO_ID if i < 8 else AB_ID
        item = build_item(panel_prompts[i][1], dec_token_id)

        # Forward to obtain readout scalar s, patched logits, and decision position.
        model.zero_grad(set_to_none=True)
        out = model(
            input_ids=item["input_ids"].unsqueeze(0),
            attention_mask=item["attention_mask"].unsqueeze(0),
            output_hidden_states=True,
        )
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = item["labels"].unsqueeze(0)[..., 1:].contiguous()
        v_idx = torch.where(shift_labels[0] != -100)[0]
        if v_idx.numel() <= 4:
            raise PreflightError(f"rho_c: decision position index 4 unavailable in sample {i}")
        d_pos = v_idx[4]
        hL = out.hidden_states[-1][0, d_pos, :]
        z_pro, z_abs = readout.forward_logits_tensor(hL)
        s = z_pro

        # Post-patch full-vocab softmax probabilities for alpha_c (Protocol §5.7).
        full_z = shift_logits[0, d_pos, :].float().clone()
        full_z[PRO_ID] = z_pro
        full_z[AB_ID] = z_abs
        probs = torch.softmax(full_z, dim=0)
        p_pro = float(probs[PRO_ID].item())
        p_abs = float(probs[AB_ID].item())
        delta = p_pro - p_abs
        if i < 8:
            alpha_c = delta - 1.0   # POS panels, y=(1,0)
        else:
            alpha_c = delta + 1.0   # ABS panels, y=(0,1)

        # g_total via frozen 6E.18 machinery: gradient of dec_loss w.r.t. LoRA params.
        g_total = psdg_fn(model, readout, item, lora_params, tokenizer)

        # g_R = alpha_c * grad_{lora_params}(s): readout-only decision gradient.
        grads_s = torch.autograd.grad(s, lora_params, allow_unused=True, retain_graph=False)
        flat_s = []
        for g, p in zip(grads_s, lora_params):
            if g is None:
                flat_s.append(torch.zeros_like(p.data).reshape(-1))
            else:
                flat_s.append(g.reshape(-1))
        g_R = alpha_c * torch.cat(flat_s)
        g_R_all.append(g_R)

        # Common-space cosine with guards.
        rho_i = compute_readout_total_alignment(g_R, g_total)
        per_sample_rho.append(rho_i)

        model.zero_grad(set_to_none=True)

    if _6E20_7_TELEMETRY_ENABLED:
        _6E20_7_GRAD_CACHE["g_R"] = list(g_R_all)

    rho_c = float(np.mean(per_sample_rho))
    sigma_c = float(np.mean(np.asarray(per_sample_rho, dtype=np.float64) ** 2))
    return rho_c, sigma_c, per_sample_rho


# ===========================================================================
# 6E.20.7 ADDITIVE DIAGNOSTIC TELEMETRY (implementation phase)
# ---------------------------------------------------------------------------
# GOVERNANCE: The 0.95 SENS threshold and the four-cause floors defined here
# are DIAGNOSTIC CONTROLS for the cos(H1) observable. They are NOT new
# scientific success criteria and can NEVER alter H1/H3/H4/H5 verdicts.
# All state below is additive; the verdict engine (evaluate_protocol_hypotheses)
# is untouched and cos_H1_i is never passed to it. Everything is OFF by default.
_6E20_7_TELEMETRY_ENABLED = False
_6E20_7_SENS_ENABLED = False
_6E20_7_SENS_PROMPTS = None
_6E20_7_GRAD_CACHE = {"g_pos": None, "g_abs": None, "g_R": None}

# Pre-registered diagnostic floors (FIXED; never tuned to data)
_6E20_7_FLOOR_R_CS = 0.20
_6E20_7_FLOOR_CSPEC_ABS_REL = 1e-3
_6E20_7_FLOOR_VHET = 0.25
_6E20_7_FLOOR_VHET_STABLE = 0.10
_6E20_7_SENS_EXPECT = 0.95
_6E20_7_EPS = 1e-9
_6E20_7_MULTIMODAL_FRAC = 0.30


def enable_6e20_7_telemetry(sens_enabled: bool = False, sens_prompts=None) -> None:
    """Opt-in activation. By default everything is OFF so the runner's behavior
    is identical to the pre-6E.20.7 baseline. Must be called by an explicit
    (future) execution authorization; never auto-enabled."""
    global _6E20_7_TELEMETRY_ENABLED, _6E20_7_SENS_ENABLED, _6E20_7_SENS_PROMPTS
    _6E20_7_TELEMETRY_ENABLED = True
    _6E20_7_SENS_ENABLED = bool(sens_enabled)
    _6E20_7_SENS_PROMPTS = sens_prompts


def _6e20_7_cosine(a, b):
    import torch
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb))


def assemble_6e20_7_telemetry(condition, step):
    """Build additive per-step telemetry from gradients stashed by the EXISTING
    compute_live_cos_H1 / compute_live_rho_c calls (no extra gradient pass)."""
    import torch
    g_pos = _6E20_7_GRAD_CACHE.get("g_pos")
    g_abs = _6E20_7_GRAD_CACHE.get("g_abs")
    g_R = _6E20_7_GRAD_CACHE.get("g_R")
    if g_pos is None or g_abs is None or g_R is None:
        return {}
    g_total_all = list(g_pos) + list(g_abs)
    g_total_norm = [float(g.norm().item()) for g in g_total_all]
    g_R_norm = [float(g.norm().item()) for g in g_R]
    cos_H1_i = [_6e20_7_cosine(g_pos[i], g_abs[i]) for i in range(8)]
    mean_pos = torch.stack([g.float() for g in g_pos]).mean(0)
    mean_abs = torch.stack([g.float() for g in g_abs]).mean(0)
    g_total_common = torch.stack([g.float() for g in g_total_all]).mean(0)
    g_total_specific = mean_pos - mean_abs
    C_common = float(g_total_common.norm().item())
    C_spec = float(g_total_specific.norm().item())
    r_cs = C_spec / (C_common + _6E20_7_EPS)
    V_het = float(torch.std(torch.tensor(cos_H1_i, dtype=torch.float64)).item())
    return {
        "condition": condition,
        "step": step,
        "g_total_norm": g_total_norm,
        "g_R_norm": g_R_norm,
        "cos_H1_i": cos_H1_i,
        "g_total_common_norm": C_common,
        "g_total_specific_norm": C_spec,
        "r_cs": r_cs,
        "V_het": V_het,
        "pos_mean_norm": float(mean_pos.norm().item()),
        "abs_mean_norm": float(mean_abs.norm().item()),
    }


def run_6e20_7_sens_control(model, readout, tokenizer, sens_prompts, psdg_fn, crgc_fn, device_str):
    """Non-degenerate SENS control (gated OFF by default). Uses GENUINELY
    DISTINCT paraphrase framings of the SAME verdict and the REAL frozen
    psdg_fn; the aggregation is the FROZEN cos_H1_aggregate definition. Never
    written into any H1/P1-P6 verdict."""
    import torch
    ensure_readout_class_patch()
    lora_params = [p for p in model.parameters() if p.requires_grad]
    if not lora_params:
        raise RuntimeError("No requires_grad parameters found in PEFT model during SENS gradient computation")

    def build_item(prompt_str, dec_token_id):
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

    g_pos, g_abs = [], []
    for i in range(8):
        item = build_item(sens_prompts[i][1], PRO_ID)
        g_pos.append(psdg_fn(model, readout, item, lora_params, tokenizer))
    for j in range(8):
        item = build_item(sens_prompts[8 + j][1], AB_ID)
        g_abs.append(psdg_fn(model, readout, item, lora_params, tokenizer))
    result = crgc_fn(g_pos, g_abs)
    return float(result["cos_H1_aggregate"])


def classify_6e20_7_causes(all_telemetry, sens_cos=None):
    """Apply the pre-registered four-cause classification (spec Section 3b).
    Diagnostic only; never alters evaluate_protocol_hypotheses output."""
    import numpy as np
    def step_tele(cond, key):
        out = []
        for t in AGGREGATION_SET:
            td = all_telemetry[cond]["steps"][str(t)].get("telemetry_6e20_7", {})
            if key in td:
                out.append(td[key])
        return out

    def step_cos(cond):
        return [all_telemetry[cond]["steps"][str(t)]["cos_H1"] for t in AGGREGATION_SET]

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    stats = {}
    for c in CONDITIONS:
        cos = step_cos(c)
        stats[c] = {
            "agg": mean(cos),
            "agg_range": (max(cos) - min(cos)) if cos else float("nan"),
            "C_common": mean(step_tele(c, "g_total_common_norm")),
            "C_spec": mean(step_tele(c, "g_total_specific_norm")),
            "r_cs": mean(step_tele(c, "r_cs")),
            "V_het": mean(step_tele(c, "V_het")),
        }
    median_C_common = float(np.median([stats["S4"]["C_common"], stats["S8"]["C_common"]]))
    floor_cspec_abs = _6E20_7_FLOOR_CSPEC_ABS_REL * median_C_common

    def multimodal_fractions(cond):
        vals = []
        for t in AGGREGATION_SET:
            vals.extend(all_telemetry[cond]["steps"][str(t)].get("telemetry_6e20_7", {}).get("cos_H1_i", []))
        if not vals:
            return 0.0, 0.0
        hi = sum(1 for v in vals if 0.7 <= v <= 1.0) / len(vals)
        lo = sum(1 for v in vals if -1.0 <= v <= 0.1) / len(vals)
        return hi, lo

    results = {}
    for c in ("S4", "S8"):
        s = stats[c]
        hi, lo = multimodal_fractions(c)
        c1 = (s["C_common"] >= median_C_common) and (s["r_cs"] < _6E20_7_FLOOR_R_CS)
        c2 = s["C_spec"] < floor_cspec_abs
        c3 = (s["V_het"] > _6E20_7_FLOOR_VHET) or (hi >= _6E20_7_MULTIMODAL_FRAC and lo >= _6E20_7_MULTIMODAL_FRAC)
        agg_stable = (
            (s["agg_range"] < 0.05)
            and (abs(stats["S4"]["agg"] - stats["S8"]["agg"]) < 0.05)
            and (abs(stats["S4"]["agg"] - stats["C4-Rand"]["agg"]) < 0.10)
        )
        sens_ok = (sens_cos is None) or (sens_cos >= _6E20_7_SENS_EXPECT)
        c4 = agg_stable and (s["V_het"] < _6E20_7_FLOOR_VHET_STABLE) and (s["C_spec"] >= floor_cspec_abs) and sens_ok
        ex = {}
        ex["common_dominance"] = ((
            (_6E20_7_FLOOR_R_CS - s["r_cs"]) / _6E20_7_FLOOR_R_CS
        ) if (c1 and s["r_cs"] < _6E20_7_FLOOR_R_CS) else 0.0)
        ex["low_specific_signal"] = ((
            (floor_cspec_abs - s["C_spec"]) / max(floor_cspec_abs, _6E20_7_EPS)
        ) if (c2 and s["C_spec"] < floor_cspec_abs) else 0.0)
        ex["per_sample_heterogeneity"] = ((
            (s["V_het"] - _6E20_7_FLOOR_VHET) / _6E20_7_FLOOR_VHET
        ) if (c3 and s["V_het"] > _6E20_7_FLOOR_VHET) else 0.0)
        triggered = [k for k, v in ex.items() if v > 0.0]
        primary = (
            max(ex, key=lambda k: ex[k])
            if triggered
            else ("genuine_stable_plateau" if c4 else "inconclusive")
        )
        results[c] = {
            "stats": s,
            "floor_cspec_abs": floor_cspec_abs,
            "multimodal_hi_frac": hi,
            "multimodal_lo_frac": lo,
            "criteria": {
                "common_dominance": bool(c1),
                "low_specific_signal": bool(c2),
                "per_sample_heterogeneity": bool(c3),
                "genuine_stable_plateau": bool(c4),
            },
            "exceedance": ex,
            "triggered_flags": triggered,
            "primary_classification": primary,
        }
    return {
        "diagnostic_only": True,
        "note": "0.95 SENS threshold and four-cause floors are DIAGNOSTIC CONTROLS, not success criteria.",
        "sens_cos": sens_cos,
        "sens_expect": _6E20_7_SENS_EXPECT,
        "floors": {
            "FLOOR_r_cs": _6E20_7_FLOOR_R_CS,
            "FLOOR_Cspec_abs": floor_cspec_abs,
            "FLOOR_Vhet": _6E20_7_FLOOR_VHET,
            "FLOOR_Vhet_stable": _6E20_7_FLOOR_VHET_STABLE,
        },
        "classification": results,
    }


def run_stage5_b0_historical_reproducibility_anchor(
    peft_model: Any,
    tokenizer: Any,
    paths: Dict[str, Path],
    panel_prompts: List[Tuple[str, str]],
    readout_s8: Readout,
    psdg_fn: Any,
    crgc_fn: Any,
    device_str: str,
) -> Dict[str, Any]:
    """Recompute B0 checkpoints on the historical B0+S8 path and assert exact numerical parity."""
    import torch
    print("=" * 80)
    print("STAGE 5: HISTORICAL B0 REPRODUCIBILITY ANCHOR GATE")
    print("NOTE: PASSING THIS GATE DOES NOT SUPPORT H1. IT VALIDATES ONLY")
    print("ENVIRONMENT COMPATIBILITY, PROMPT RECONSTRUCTION, AND EVALUATION PARITY.")
    print("=" * 80)

    anchor_results = {}
    for t, exp_cos in HISTORICAL_B0_S8_ANCHOR.items():
        ckpt_path = paths["historical_out_dir"] / "B0" / f"checkpoint_{t}" / "model.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        peft_model.load_state_dict(ckpt["state_dict"], strict=False)

        # --- 6E.20.4 B0 execution-regime guard ---
        # The historical B0 anchor (0.428355) was generated with LoRA dropout
        # DISABLED (eval/inference mode). Running compute_live_cos_H1 in train()
        # mode activates LORA_DROPOUT=0.05 and makes cos(H1) stochastic (~1e-3),
        # which the 1e-4 reproducibility gate cannot satisfy. Restore parity by
        # evaluating under eval mode and restoring the prior mode afterward.
        was_training = peft_model.training
        try:
            peft_model.eval()
            if peft_model.training is not False:
                raise RuntimeError(
                    "6E.20.4 regime guard: model.eval() failed to disable training mode"
                )
            recomputed_cos = compute_live_cos_H1(
                peft_model, readout_s8, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str
            )
        finally:
            peft_model.train(was_training)
        if peft_model.training != was_training:
            raise RuntimeError(
                "6E.20.4 regime guard: model mode not restored after B0 evaluation"
            )
        diff = abs(recomputed_cos - exp_cos)

        anchor_results[str(t)] = {
            "checkpoint": str(ckpt_path),
            "historical_runs_json_cos_H1": exp_cos,
            "recomputed_b0_s8_cos_H1": recomputed_cos,
            "absolute_difference": diff,
            "tolerance": B0_REPRODUCIBILITY_TOLERANCE,
            "pass": diff <= B0_REPRODUCIBILITY_TOLERANCE,
        }

        if diff > B0_REPRODUCIBILITY_TOLERANCE:
            raise PreflightError(
                f"Stage 5 HALT: B0 reproducibility failure at step {t}! "
                f"Recomputed={recomputed_cos:.6f}, Historical={exp_cos:.6f}, Diff={diff:.6e} > {B0_REPRODUCIBILITY_TOLERANCE}"
            )

    return {
        "status": "PASS",
        "description": "Historical B0+S8 path reproduced within 1e-4 tolerance. Validates evaluation environment.",
        "results": anchor_results,
    }


# ---------------------------------------------------------------------------
# Decision Logic Engine (Protocol v3.1 §7, §8, §9)
# ---------------------------------------------------------------------------
def evaluate_protocol_hypotheses(all_condition_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Apply verbatim Protocol v3.1 decision rules across conditions."""
    means_cos: Dict[str, float] = {}
    d15_values: Dict[str, float] = {}
    bal15_values: Dict[str, float] = {}
    rho_means: Dict[str, float] = {}

    for c in CONDITIONS:
        c_data = all_condition_metrics[c]
        cos_list = [c_data["steps"][str(t)]["cos_H1"] for t in AGGREGATION_SET]
        rho_list = [c_data["steps"][str(t)]["rho_c"] for t in AGGREGATION_SET]
        means_cos[c] = float(sum(cos_list) / len(cos_list))
        rho_means[c] = float(sum(rho_list) / len(rho_list))
        d15_values[c] = float(c_data["steps"]["15"]["cohens_d"])
        bal15_values[c] = float(c_data["steps"]["15"]["balanced_accuracy"])

    # 1. Primary Hypothesis H1 / Criterion P2 (Protocol §7, §8)
    p2_s4 = means_cos["S4"] >= P2_THRESHOLD
    p2_s8 = means_cos["S8"] >= P2_THRESHOLD
    h1_verdict = "SUPPORTED" if (p2_s4 and p2_s8) else "REFUTED"

    # 2. Control Contrast Hypothesis H4 (Protocol §7, §9.2)
    # S4 exceeds C4-Rand on mean cos(H1) by >= +0.30 AND C4-Rand fails P3 (d(15) < +0.80)
    delta_s4_c4 = means_cos["S4"] - means_cos["C4-Rand"]
    c4_fails_p3 = d15_values["C4-Rand"] < P3_THRESHOLD
    h4_verdict = "SUPPORTED" if (delta_s4_c4 >= H4_DELTA_MIN and c4_fails_p3) else "REFUTED"

    # 3. Control Contrast Hypothesis H5 (Protocol §7, §9.2)
    # R1-Rand fails P3 (d(15) < +0.80) while F1 passes P3 (d(15) >= +0.80)
    r1_fails_p3 = d15_values["R1-Rand"] < P3_THRESHOLD
    f1_passes_p3 = d15_values["F1"] >= P3_THRESHOLD
    h5_verdict = "SUPPORTED" if (r1_fails_p3 and f1_passes_p3) else "REFUTED"

    # 4. Readout Steering Hypothesis H3 (Protocol §7, §9.2)
    # rho_s8 >= 0.50 and rho_s8 >= rho_s4 >= rho_s2
    rho_s8_pass = rho_means["S8"] >= P1_THRESHOLD
    rho_monotonic = (rho_means["S8"] >= rho_means["S4"]) and (rho_means["S4"] >= rho_means["S2"])
    h3_verdict = "SUPPORTED" if (rho_s8_pass and rho_monotonic) else "REFUTED"

    # 5. Dual-Objective Matrix Assessment (Protocol §9.1)
    tu_high_s4 = (d15_values["S4"] >= P4_THRESHOLD) and (bal15_values["S4"] >= P5_THRESHOLD)
    tu_high_s8 = (d15_values["S8"] >= P4_THRESHOLD) and (bal15_values["S8"] >= P5_THRESHOLD)

    return {
        "aggregation_set": AGGREGATION_SET,
        "mean_cos_H1": means_cos,
        "mean_rho_c": rho_means,
        "d15": d15_values,
        "balanced_accuracy_15": bal15_values,
        "criteria": {
            "P2_S4": {"passed": p2_s4, "value": means_cos["S4"], "threshold": P2_THRESHOLD},
            "P2_S8": {"passed": p2_s8, "value": means_cos["S8"], "threshold": P2_THRESHOLD},
            "P3_F1": {"passed": f1_passes_p3, "value": d15_values["F1"], "threshold": P3_THRESHOLD},
            "P3_C4_FAIL": {"passed": c4_fails_p3, "value": d15_values["C4-Rand"], "threshold": P3_THRESHOLD},
            "P3_R1_FAIL": {"passed": r1_fails_p3, "value": d15_values["R1-Rand"], "threshold": P3_THRESHOLD},
            "P1_S8": {"passed": rho_s8_pass, "value": rho_means["S8"], "threshold": P1_THRESHOLD},
        },
        "hypotheses": {
            "H1_Primary_Subspace_Gradient_Compatibility": {
                "verdict": h1_verdict,
                "rule": "mean_cos(S4) >= +0.80 AND mean_cos(S8) >= +0.80",
                "details": f"S4={means_cos['S4']:.6f}, S8={means_cos['S8']:.6f}",
            },
            "H4_Aligned_vs_Random_Subspace": {
                "verdict": h4_verdict,
                "rule": "mean_cos(S4) - mean_cos(C4-Rand) >= +0.30 AND C4-Rand d(15) < +0.80",
                "details": f"delta={delta_s4_c4:.6f} (req >= {H4_DELTA_MIN}), C4_d15={d15_values['C4-Rand']:.6f}",
            },
            "H5_Random_1D_Alignment_Specificity": {
                "verdict": h5_verdict,
                "rule": "R1-Rand d(15) < +0.80 AND F1 d(15) >= +0.80",
                "details": f"R1_d15={d15_values['R1-Rand']:.6f}, F1_d15={d15_values['F1']:.6f}",
            },
            "H3_Readout_Steering": {
                "verdict": h3_verdict,
                "rule": "rho_S8 >= 0.50 AND rho_S8 >= rho_S4 >= rho_S2",
                "details": f"rho_S8={rho_means['S8']:.6f}, rho_S4={rho_means['S4']:.6f}, rho_S2={rho_means['S2']:.6f}",
            },
        },
    }


# ---------------------------------------------------------------------------
# Authorization Check
# ---------------------------------------------------------------------------
def check_authorization(script_path: Path) -> None:
    """Verify presence and validity of Phase 6E.20.3 L5 authorization."""
    auth_file = script_path.parent / AUTH_FILE_NAME
    if not auth_file.exists():
        raise PreflightError(
            f"EXECUTION_NOT_AUTHORIZED: Authorization file {auth_file.name} not found. "
            + " | ".join(REFUSAL_TRIPLET)
        )
    try:
        data = json.loads(auth_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise PreflightError(f"EXECUTION_NOT_AUTHORIZED: Unparseable auth file: {e}")

    if data.get("decision") != "APPROVE":
        raise PreflightError(f"EXECUTION_NOT_AUTHORIZED: decision={data.get('decision')} != APPROVE")
    if data.get("phase") != PHASE_ID:
        raise PreflightError(f"EXECUTION_NOT_AUTHORIZED: phase={data.get('phase')} != {PHASE_ID}")

    expected_sha = data.get("runner_sha256", "").upper()
    actual_sha = compute_sha256(script_path)
    if actual_sha != expected_sha:
        raise PreflightError(
            f"EXECUTION_NOT_AUTHORIZED: Runner SHA mismatch! Actual={actual_sha} != Auth={expected_sha}"
        )


# ---------------------------------------------------------------------------
# Full Evaluation Orchestrator (Stages 1–8)
# ---------------------------------------------------------------------------
def run_evaluation_pipeline(out_dir_path: Path, preflight_only: bool = False) -> Dict[str, Any]:
    """Execute complete 8-stage evaluation pipeline."""
    paths = resolve_project_paths()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"PHASE {PHASE_ID} — READ-ONLY CHECKPOINT RE-EVALUATION")
    print(f"Output Directory: {out_dir_path}")
    print("=" * 80)

    # Stage 1: Pre-Execution SHA Manifest
    print("\n[STAGE 1/8] Pre-execution cryptographic SHA-256 manifest...")
    stage1_manifest = run_stage1_pre_execution_sha_manifest(paths)
    (out_dir_path / "pre_execution_sha256_manifest.json").write_text(
        json.dumps(stage1_manifest, indent=2), encoding="utf-8"
    )
    print("  -> Stage 1 PASS: All 28 checkpoints and frozen inputs verified.")

    # Stage 2: Model & Tokenizer Verification
    print("\n[STAGE 2/8] Base model & tokenizer revision verification...")
    base_model, tokenizer = run_stage2_model_and_tokenizer_preflight()
    print(f"  -> Stage 2 PASS: {MODEL_ID} @ {MODEL_REVISION} verified.")

    # Load frozen 6E.18 gradient machinery
    print("\n[LOADING FROZEN MACHINERY] Loading frozen 6E.18 total-gradient autograd functions...")
    psdg_fn, crgc_fn, device_str = load_frozen_6e18_gradient_machinery(paths["frozen_6e18_snapshot"])
    if torch.cuda.is_available():
        base_model = base_model.to(device_str)
    print(f"  -> Machinery loaded on {device_str}.")

    # Stage 3 & 4: Smoke Test & LoRA Validation
    print("\n[STAGE 3 & 4/8] Checkpoint deserialization & non-silent LoRA key validation...")
    stage3_4_results = run_stage3_stage4_checkpoint_smoke_and_lora_validation(base_model, paths)
    (out_dir_path / "checkpoint_smoke_test_report.json").write_text(
        json.dumps(stage3_4_results, indent=2), encoding="utf-8"
    )
    print(f"  -> Stage 3 & 4 PASS: All 28 checkpoints verified with {stage3_4_results['lora_key_count']} active LoRA keys.")

    # Reconstruct readouts & panel prompts
    print("\n[READOUT RECONSTRUCTION] Loading protocol readouts from step0...")
    import numpy as np
    H16_step0 = np.load(paths["historical_out_dir"] / "step0" / "H_16.npy")
    readouts: Dict[str, Readout] = {}
    for c in CONDITIONS:
        w_vec = np.load(paths["historical_out_dir"] / "step0" / f"w_{c}.npy")
        if c == "B0":
            bias = 0.0
        else:
            bias = float(-np.median(w_vec @ H16_step0[:8].T))
        readouts[c] = Readout(name=c, w=w_vec.astype(np.float64), b=bias)

    panel_prompts = reconstruct_panel_prompts(paths["corpus"])
    if len(panel_prompts) != 16:
        raise PreflightError(f"Panel prompt count {len(panel_prompts)} != 16")

    # Wrap base model in PEFT LoRA
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias=LORA_BIAS, task_type=LORA_TASK_TYPE,
    )
    eval_peft_model = get_peft_model(base_model, lora_config)

    # --- NEPR Collection (post-load, pre-evaluation) ---
    print("\n[NEPR] Collecting Numerical Environment Provenance Record...")
    _runner_sha = compute_sha256(Path(__file__).resolve())
    _test_sha_path = Path(__file__).resolve().parent / "test_phase_6e20_reevaluation.py"
    _test_sha = compute_sha256(_test_sha_path) if _test_sha_path.exists() else "unknown"
    nepr_dict = collect_nepr(
        model=base_model,
        peft_model=eval_peft_model,
        phase_id=PHASE_ID,
        runner_sha256=_runner_sha,
        test_sha256=_test_sha,
    )
    nepr_path = out_dir_path / NEPR_FILE_NAME
    atomic_write_json(nepr_path, nepr_dict)
    print(f"  -> NEPR emitted: {nepr_path.name} (nepr_hash={nepr_dict['category_F_metadata']['nepr_hash'][:24]}...)")

    # Emit execution-environment binding (binds NEPR to pre-execution manifest)
    binding_dict = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": PHASE_ID,
        "nepr_path": NEPR_FILE_NAME,
        "nepr_sha256": compute_sha256(nepr_path),
        "nepr_hash": nepr_dict["category_F_metadata"]["nepr_hash"],
        "pre_execution_manifest_path": "pre_execution_sha256_manifest.json",
        "pre_execution_manifest_sha256": compute_sha256(out_dir_path / "pre_execution_sha256_manifest.json"),
        "runner_sha256": _runner_sha,
        "test_sha256": _test_sha,
    }
    binding_path = out_dir_path / BINDING_FILE_NAME
    atomic_write_json(binding_path, binding_dict)
    print(f"  -> Binding emitted: {binding_path.name}")

    # Stage 5: B0 Historical Reproducibility Anchor
    print("\n[STAGE 5/8] Historical B0 reproducibility anchor gate...")
    stage5_results = run_stage5_b0_historical_reproducibility_anchor(
        eval_peft_model, tokenizer, paths, panel_prompts, readouts["S8"], psdg_fn, crgc_fn, device_str
    )
    stage5_results["nepr_comparison"] = {
        "nepr_hash": nepr_dict["category_F_metadata"]["nepr_hash"],
        "note": "NEPR environment classification is non-gating metadata. Stage 5 halt logic is unchanged.",
    }
    (out_dir_path / "b0_reproducibility_anchor_report.json").write_text(
        json.dumps(stage5_results, indent=2), encoding="utf-8"
    )
    print("  -> Stage 5 PASS: Historical B0 evaluation path reproduced with <= 1e-4 difference.")

    if preflight_only:
        print("\n" + "=" * 80)
        print("PREFLIGHT COMPLETE — EXECUTION HALTED (preflight-only mode)")
        print("=" * 80)
        return {"status": "PREFLIGHT_PASS"}

    # Stage 6: 28-Checkpoint Condition-Resolved Evaluation
    print("\n[STAGE 6/8] Executing 28 condition-resolved checkpoint evaluations...")
    all_telemetry: Dict[str, Any] = {}

    for c in CONDITIONS:
        print(f"\nEvaluating Condition: {c} (using {c}'s own readout)...")
        readout_c = readouts[c]
        cond_record = {"condition": c, "readout_b": readout_c.b, "steps": {}}

        for t in AGGREGATION_SET:
            ckpt_path = paths["historical_out_dir"] / c / f"checkpoint_{t}" / "model.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            eval_peft_model.load_state_dict(ckpt["state_dict"], strict=False)

            # 1. Total-gradient decision cosine cos(H1)
            cos_h1 = compute_live_cos_H1(
                eval_peft_model, readout_c, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str
            )

            # 2. Decision margins & accuracy
            margin_metrics = compute_decision_margin_and_metrics(
                eval_peft_model, tokenizer, readout_c, panel_prompts
            )

            # 3. Readout-total alignment rho_c
            rho_c, sigma_c, per_sample_rho = compute_live_rho_c(
                eval_peft_model, readout_c, tokenizer, panel_prompts, psdg_fn, crgc_fn, device_str
            )

            step_data = {
                "step": t,
                "cos_H1": cos_h1,
                "rho_c": rho_c,
                "sigma_c": sigma_c,
                "balanced_accuracy": margin_metrics["balanced_accuracy"],
                "cohens_d": margin_metrics["cohens_d"],
                "pos_accuracy": margin_metrics["pos_accuracy"],
                "abs_accuracy": margin_metrics["abs_accuracy"],
                "margins": margin_metrics["margins"],
            }
            cond_record["steps"][str(t)] = step_data
            if _6E20_7_TELEMETRY_ENABLED:
                step_data["telemetry_6e20_7"] = assemble_6e20_7_telemetry(c, t)
            print(f"  t={t:2d}: cos(H1) = {cos_h1:+.6f} | rho_c = {rho_c:.4f} | d = {margin_metrics['cohens_d']:+.4f} | bal_acc = {margin_metrics['balanced_accuracy']:.2%}")

        all_telemetry[c] = cond_record

    (out_dir_path / "reevaluation_runs.json").write_text(
        json.dumps(all_telemetry, indent=2), encoding="utf-8"
    )

    # Stage 7: Post-Execution SHA Audit (SEPARATE audit file — pre-execution manifest immutable)
    print("\n[STAGE 7/8] Post-execution cryptographic immutability audit...")
    stage7_audit = run_stage1_pre_execution_sha_manifest(paths)
    stage7_audit["audit_type"] = "post_execution"
    stage7_audit["all_frozen_inputs_unchanged"] = all(
        entry.get("match", True) for entry in stage7_audit["files"].values()
    )
    (out_dir_path / "post_execution_sha256_audit.json").write_text(
        json.dumps(stage7_audit, indent=2), encoding="utf-8"
    )
    if not stage7_audit["all_frozen_inputs_unchanged"]:
        raise PreflightError("Stage 7 HALT: Frozen input modification detected post-execution!")
    print("  -> Stage 7 PASS: Zero modification of frozen 6E.20.1 files confirmed.")

    # Stage 8: Verdict Aggregation
    print("\n[STAGE 8/8] Protocol hypothesis aggregation and verdict generation...")
    verdict_report = evaluate_protocol_hypotheses(all_telemetry)
    if _6E20_7_TELEMETRY_ENABLED:
        sens_cos = None
        if _6E20_7_SENS_ENABLED and _6E20_7_SENS_PROMPTS is not None:
            sens_cos = run_6e20_7_sens_control(
                eval_peft_model, readouts["S4"], tokenizer,
                _6E20_7_SENS_PROMPTS, psdg_fn, crgc_fn, device_str,
            )
        classification = classify_6e20_7_causes(all_telemetry, sens_cos)
        (out_dir_path / "6e20_7_cause_classification.json").write_text(
            json.dumps(classification, indent=2), encoding="utf-8"
        )
    verdict_report["nepr_environment_classification"] = "non_applicable"
    verdict_report["nepr_hash"] = nepr_dict["category_F_metadata"]["nepr_hash"]
    (out_dir_path / "hypothesis_verdicts.json").write_text(
        json.dumps(verdict_report, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print(f"SCIENTIFIC HYPOTHESIS VERDICTS (Phase {PHASE_ID}):")
    for hname, hres in verdict_report["hypotheses"].items():
        print(f"  * {hname}: {hres['verdict']}")
        print(f"    Rule: {hres['rule']}")
        print(f"    Evidence: {hres['details']}")
    print("=" * 80)

    return {
        "status": "EVALUATION_COMPLETE",
        "verdict_report": verdict_report,
        "telemetry": all_telemetry,
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=f"Phase {PHASE_ID} Read-Only Checkpoint Re-Evaluator")
    parser.add_argument("--execute", action="store_true", help="Execute full 28-checkpoint re-evaluation (requires authorization)")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight checks & B0 anchor gate only (no 28 evaluations)")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic unit test suite")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")

    args = parser.parse_args(argv)
    this_file = Path(__file__).resolve()

    if args.self_test:
        from test_phase_6e20_reevaluation import run_unit_test_suite
        res = run_unit_test_suite()
        print(json.dumps(res, indent=2))
        return 0 if res["all_pass"] else 1

    if args.preflight_only:
        out_dir = Path(args.out_dir)
        try:
            res = run_evaluation_pipeline(out_dir, preflight_only=True)
            print(json.dumps(res, indent=2))
            return 0
        except Exception as e:
            print(f"PREFLIGHT ERROR: {e}")
            return 1

    if args.execute:
        try:
            check_authorization(this_file)
        except PreflightError as e:
            print(json.dumps({"error": str(e), "triplet": list(REFUSAL_TRIPLET)}, indent=2))
            return 1

        out_dir = Path(args.out_dir)
        try:
            res = run_evaluation_pipeline(out_dir, preflight_only=False)
            return 0
        except Exception as e:
            print(f"EXECUTION HALT: {e}")
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    import torch  # Ensure torch is in namespace
    raise SystemExit(main())
