"""Verify v3 prompts produce non-degenerate registration."""
import sys, numpy as np
sys.path.insert(0, '.')
from run_phase_6e20_execution_v3 import collect_panel_prompts
from run_phase_6e20_fixed_subspace_readout import MODEL_ID, MODEL_REVISION, fit_lda_axis, s_scale_degenerate, compute_s_scale
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER_INDEX = 24
D_HIDDEN = 896

prompts = collect_panel_prompts()
print("Panel prompts: %d" % len(prompts))
for i, (tag, fp) in enumerate(prompts):
    print("  %d: %s (len=%d)" % (i, tag, len(fp)))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model.eval()
print("Model loaded")

hs_list = []
with torch.no_grad():
    for label, formatted in prompts:
        inp = tokenizer(formatted, return_tensors="pt")
        out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states[LAYER_INDEX]
        last_hs = hs[0, -1, :].float().cpu().numpy()
        hs_list.append(last_hs)

H16 = np.stack(hs_list, axis=0)
y16 = np.array([1]*8 + [-1]*8)

# Check that POS and ABS are different
pos_mean = H16[:8].mean(axis=0)
abs_mean = H16[8:].mean(axis=0)
diff = np.linalg.norm(pos_mean - abs_mean)
print("POS-ABS mean distance: %.6f" % diff)

# Check all 16 rows are distinct
identical_pairs = []
for i in range(16):
    for j in range(i+1, 16):
        d = np.linalg.norm(H16[i] - H16[j])
        if d < 1e-6:
            identical_pairs.append((i, j))
if identical_pairs:
    print("WARNING: identical pairs: %s" % identical_pairs)
else:
    print("All 16 rows distinct: PASS")

u_lda = fit_lda_axis(H16, y16)
print("u_lda norm: %.6f" % np.linalg.norm(u_lda))
print("u_lda nonzero: %d/%d" % (np.sum(np.abs(u_lda) > 1e-10), len(u_lda)))

margins = np.asarray(H16 @ u_lda, dtype=np.float64)
s = compute_s_scale(margins, H16, u_lda)
print("s_scale: %.6f degenerate: %s" % (s, s_scale_degenerate(s)))
print("margins: %s" % margins)
print("margins std: %.6f" % np.std(margins))

if s_scale_degenerate(s):
    print("FAIL: s_scale still degenerate")
    sys.exit(1)
else:
    print("PASS: registration non-degenerate")
