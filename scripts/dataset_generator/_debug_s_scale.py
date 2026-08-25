import sys, numpy as np, torch
sys.path.insert(0, '.')
from run_phase_6e20_execution_v2 import TASK_PROMPTS, LAYER_INDEX, D_HIDDEN
from run_phase_6e20_fixed_subspace_readout import (
    fit_lda_axis, compute_s_scale, s_scale_degenerate, MODEL_ID, MODEL_REVISION
)
from transformers import AutoModelForCausalLM, AutoTokenizer

print("MODEL_ID=%s REVISION=%s" % (MODEL_ID, MODEL_REVISION))
print("LAYER_INDEX=%d D_HIDDEN=%d TASK_PROMPTS=%d" % (LAYER_INDEX, D_HIDDEN, len(TASK_PROMPTS)))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model.eval()
print("Model loaded OK")

hs_list = []
with torch.no_grad():
    for label, prompt in TASK_PROMPTS:
        fmt = prompt + '{"decision": "'
        inp = tokenizer(fmt, return_tensors='pt')
        out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states[LAYER_INDEX]
        last_hs = hs[0, -1, :].float().cpu().numpy()
        hs_list.append(last_hs)
        print("  %s: norm=%.6f mean=%.6f" % (label, np.linalg.norm(last_hs), last_hs.mean()))

H16 = np.stack(hs_list, axis=0)
y16 = np.array([1]*8 + [-1]*8)

print("H16 shape=%s" % str(H16.shape))
print("H16 all same: %s" % np.allclose(H16, H16[0]))
print("H16 row norms: %s" % [round(np.linalg.norm(H16[i]), 6) for i in range(16)])

row_stds = [np.std(H16[i]) for i in range(16)]
print("H16 row stds: %s" % [round(x, 10) for x in row_stds])

col_stds = np.std(H16, axis=0)
print("H16 cols with nonzero std: %d / %d" % (np.sum(col_stds > 0), H16.shape[1]))
print("H16 overall std: %.12f" % np.std(H16))

u_lda = fit_lda_axis(H16, y16)
print("u_lda norm=%.6f nonzero=%d/%d" % (np.linalg.norm(u_lda), np.sum(np.abs(u_lda) > 1e-10), len(u_lda)))

margins = np.asarray(H16 @ u_lda, dtype=np.float64)
print("margins=%s" % margins)
print("margins std=%.12f" % np.std(margins))
print("margins all same: %s" % np.allclose(margins, margins[0]))

s = compute_s_scale(margins, H16, u_lda)
print("s_scale=%.12f degenerate=%s" % (s, s_scale_degenerate(s)))
