"""Patch v2 -> v3: Replace TASK_PROMPTS with corpus-derived prompts."""
from pathlib import Path

v2_path = Path("run_phase_6e20_execution_v2.py")
v3_path = Path("run_phase_6e20_execution_v3.py")
src = v2_path.read_text(encoding="utf-8")

# 1. Update docstring
src = src.replace(
    '"""Phase 6E.20.1 execution extension runner \u2014 v2 (L4 amend).',
    '"""Phase 6E.20.1 execution extension runner \u2014 v3 (corpus-derived prompts).',
)
src = src.replace(
    "NEW ARTIFACT (Phase 6E.20.1 execution extension v2).",
    "NEW ARTIFACT (Phase 6E.20.1 execution extension v3).",
)
src = src.replace(
    "Imports W1-W7 machinery from the frozen runner (never re-implements).\nPhase 6E.20.1 authorization required (NOT 6E.20).",
    "v3 fix: TASK_PROMPTS replaced with corpus-derived prompts via format_prompt(),\n"
    "matching Phase 6E.18 panel construction (seed-42, balanced 8 POS + 8 ABS).\n"
    "Resolves s_scale degenerate caused by identical POS/ABS prompt text in v2.\n"
    "Imports W1-W7 machinery from the frozen runner (never re-implements).\nPhase 6E.20.1 authorization required (NOT 6E.20).",
)

# 2. Replace TASK_PROMPTS block with corpus infrastructure
old_block = '''TASK_PROMPTS = [
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
]'''

new_block = '''CORPUS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "theo-data" / "datasets" / "theo_slm_v0_deduplicated" / "candidate_records.json"
EXPECTED_CORPUS_HASH = "a7b4e84509b1c0dc03b81425125f061cedafe288cb18f153e8399b19cf717eb0"


def format_prompt(percept, concepts):
    """Decision-position prompt template (6E.18 parity)."""
    labels = [c.get("id", str(c)) if isinstance(c, dict) else str(c) for c in concepts]
    concepts_str = ", ".join(labels) if labels else "none"
    return (
        "\u2571system\\n"
        "You are THEO SLM v0, a neural cognitive provider. Given an observation percept and grounding context, "
        "evaluate decision relevance and determine whether to propose a hypothesis or abstain.\u2571/user\\n"
        f"Observation Percept: {percept}\\n"
        f"Grounding Concepts: {concepts_str}\\n"
        "Task: Emit JSON evaluation containing decision (SHOULD_PROPOSE or SHOULD_ABSTAIN) and reasoning.\u2571/assistant\\n"
    )


def reconstruct_data(corpus_path):
    """Exact 6E.18 data reconstruction (seed 42 family split, balanced 134/67/67, panels 8/8/8)."""
    import re
    import json
    import numpy as np
    from collections import defaultdict
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
    train_records_raw = [r for r in all_records if re.sub(r"_[A-D]$", "", r["case_id"]) in train_families]
    pos_train = [r for r in train_records_raw if r.get("abstention_label") == "SHOULD_PROPOSE" and r.get("novelty_label") == "SEMANTIC_NOVEL"]
    abs_train = [r for r in train_records_raw if r.get("novelty_label") in ["DECISION_IRRELEVANT", "EPISTEMICALLY_PREMATURE"]]
    np.random.seed(42)
    balanced_pos = list(pos_train) * 2 + list(pos_train)[:134 - len(pos_train) * 2]
    balanced_abs = list(np.random.choice(abs_train, size=67, replace=True)) if len(abs_train) < 67 else list(np.random.choice(abs_train, size=67, replace=False))
    pos_panel = balanced_pos[:8]
    abs_panel = balanced_abs[:8]
    return {"pos_panel": pos_panel, "abs_panel": abs_panel}


def collect_panel_prompts():
    """Build the N=16 panel prompts from corpus records via format_prompt() (6E.18 parity)."""
    import hashlib
    import json
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}")
    actual_hash = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_CORPUS_HASH:
        raise ValueError(f"Corpus hash mismatch: {actual_hash} != {EXPECTED_CORPUS_HASH}")
    data = reconstruct_data(CORPUS_PATH)
    panel = data["pos_panel"] + data["abs_panel"]
    prompts = []
    for i, rec in enumerate(panel):
        tag = "POS" if i < 8 else "ABS"
        idx = i if i < 8 else i - 8
        formatted = format_prompt(rec["percept"], rec.get("concepts", [])) + '{\\"decision\\": \\"'
        prompts.append((f"{tag}_{idx}", formatted))
    return prompts'''

assert old_block in src, "Old TASK_PROMPTS block not found in source"
src = src.replace(old_block, new_block)

# 3. Replace collect_H16 to use corpus-derived prompts
old_collect = '''def collect_H16(model, tokenizer):
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
    return H16'''

new_collect = '''def collect_H16(model, tokenizer):
    """v3.1 3.1: collect real H16 panel from the frozen base model.
    Uses corpus-derived prompts (format_prompt from 6E.18) so POS/ABS
    panels have genuinely different hidden states. Returns H16 as (16, 896)."""
    import numpy as np
    import torch
    panel_prompts = collect_panel_prompts()
    hidden_states_all = []
    model.eval()
    with torch.no_grad():
        for label, formatted in panel_prompts:
            inputs = tokenizer(formatted, return_tensors="pt")
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[LAYER_INDEX]
            last_token_hs = hs[0, -1, :].float().cpu().numpy()
            hidden_states_all.append(last_token_hs)
    H16 = np.stack(hidden_states_all, axis=0)
    assert H16.shape == (16, D_HIDDEN), f"H16 shape {H16.shape} != (16, {D_HIDDEN})"
    return H16'''

assert old_collect in src, "Old collect_H16 not found in source"
src = src.replace(old_collect, new_collect)

# 4. Update self-test: replace TASK_PROMPTS_count with corpus check
src = src.replace(
    't("TASK_PROMPTS_count", len(TASK_PROMPTS) == 16)',
    't("corpus_panel_count", len(collect_panel_prompts()) == 16)',
)

# 5. Update auth path references
src = src.replace(
    'auth_path = Path("execution_authorization_v2.json")',
    'auth_path = Path("execution_authorization_v3.json")',
)

# 6. Update program name
src = src.replace(
    'prog="run_phase_6e20_execution_v2"',
    'prog="run_phase_6e20_execution_v3"',
)
src = src.replace(
    'description="Phase 6E.20.1 execution extension runner v2 (Plan v4.0; L1-L3 implementation)."',
    'description="Phase 6E.20.1 execution extension runner v3 (corpus prompts; Plan v4.0)."',
)

# 7. Update self-test suite name
src = src.replace(
    '"suite": "L2 execution unit tests (v2)"',
    '"suite": "L2 execution unit tests (v3)"',
)

# 8. Update self-test import (use v3 module)
src = src.replace(
    'from run_phase_6e20_execution_v2 import compute_H16_gradients',
    'from run_phase_6e20_execution_v3 import compute_H16_gradients',
)

v3_path.write_text(src, encoding="utf-8")
print(f"Written {v3_path}: {len(src.splitlines())} lines")

# Verify key changes
assert "CORPUS_PATH" in src
assert "format_prompt" in src
assert "reconstruct_data" in src
assert "collect_panel_prompts" in src
assert "TASK_PROMPTS = [" not in src
assert 'execution_authorization_v3.json' in src
print("All assertions passed")
