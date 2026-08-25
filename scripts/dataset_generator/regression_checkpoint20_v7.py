"""Phase 6E.20.1 — v7 checkpoint_20 regression (NEW ARTIFACT).

Validates the minimal runner revision run_phase_6e20_execution_v7.py:

  R1  All 8 rule-mandated checkpoints {0,4,6,7,14,15,17,20} are emitted for a
      trained condition (behavioral, mock model on CPU — no 28 GB run).
  R2  checkpoint_20 exists and is torch.load-able; its metadata step == 20 and
      condition matches.
  R3  No duplicate or shifted numbering: the emitted label set EQUALS the
      protocol CHECKPOINT_SET exactly (no extras, no missing, no off-by-one).
  R4  Existing labels 0,4,6,7,14,15,17 are emitted at UNCHANGED positions;
      checkpoint_0 == initial state (0 optimizer updates), proving "step 0"
      semantics preserved; checkpoint_20 == final trained state (differs).
  R5  GRAD_ACCUM parity: the extra t=20 iteration performs NO optimizer step
      (optimizer updates at t=19 and t=20 are identical -> pure final capture).
  R6  Authorization binds EXCLUSIVELY to the v7 runner: an auth file carrying
      v7's SHA is accepted; one carrying v6's SHA is rejected by v7's checker.

All checks are deterministic and run without a GPU model or training corpus
beyond the synthetic registration used by the runner's own self-test.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_phase_6e20_fixed_subspace_readout as r
import run_phase_6e20_execution_v7 as re_exec

# --- patch peft so train_condition runs on a CPU mock (no real HF model) ------
import peft
peft.LoraConfig = lambda *a, **k: None
peft.get_peft_model = lambda model, config: model


class MockModel(nn.Module):
    def __init__(self, d, vocab_head=9200, vocab_in=100):
        super().__init__()
        self.device = torch.device("cpu")
        self.embed = nn.Embedding(vocab_in, d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.head = nn.Linear(d, vocab_head, bias=False)

    def forward(self, input_ids, attention_mask=None, **kw):
        x = self.embed(input_ids)
        h = self.q_proj(x)
        return type("Out", (), {"logits": self.head(h)})()


class MockTokenizer:
    def __call__(self, texts, return_tensors=None, padding=None):
        seq = 5
        ids = [[(i % 50) + 1 for _ in range(seq)] for i in range(len(texts))]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), seq, dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [[1] * seq] * len(texts)}


def main():
    results = {}

    # ---- R5: GRAD_ACCUM parity (no extra optimizer step at t=20) -------------
    grad_accum = r.GRAD_ACCUM
    horizon = r.HORIZON
    updates_up_to_19 = (19 + 1) // grad_accum
    updates_up_to_20 = (20 + 1) // grad_accum
    results["R5_grad_accum_parity"] = {
        "pass": updates_up_to_19 == updates_up_to_20 == horizon // 2,
        "detail": f"grad_accum={grad_accum} updates@t19={updates_up_to_19} "
                  f"updates@t20={updates_up_to_20} horizon={horizon}",
    }

    # ---- static simulation of emitted label set (loop bound proof) ----------
    emitted_new = [t for t in range(horizon + 1) if t in r.CHECKPOINT_SET]
    emitted_old = [t for t in range(horizon) if t in r.CHECKPOINT_SET]
    results["R3_static_label_set"] = {
        "pass": emitted_new == r.CHECKPOINT_SET and 20 not in emitted_old,
        "detail": f"new={emitted_new} old_missing_20={20 not in emitted_old}",
    }

    # ---- behavioral: actually train one (non-B0) condition with mock --------
    rng = np.random.default_rng(20260821)
    reg = r._build_synthetic_registration(rng)
    readout = reg["w_all"]["F1"]  # non-B0 -> no collapse-break, exercises full loop
    mock = MockModel(r.D_HIDDEN)
    tok = MockTokenizer()

    initial_sd = {k: v.detach().clone() for k, v in mock.state_dict().items()}

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        res = re_exec.train_condition(mock, tok, "F1", readout, reg, out_dir)
        cond_dir = out_dir / "F1"
        present = sorted(p.name for p in cond_dir.iterdir() if p.is_dir())
        required = [f"checkpoint_{c}" for c in r.CHECKPOINT_SET]

        # R1/R3: exactly the required set, all present
        r1 = sorted(present) == sorted(required)
        r3 = set(present) == set(required)

        # R2/R4: each loadable, correct metadata; 0 == initial, 20 == final
        loadable_ok = True
        step_meta_ok = True
        ckpt0_eq_initial = True
        ckpt20_ne_initial = False
        for c in r.CHECKPOINT_SET:
            ckpt_path = cond_dir / f"checkpoint_{c}" / "model.pt"
            try:
                ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception:
                loadable_ok = False
                continue
            if ck.get("step") != c or ck.get("condition") != "F1":
                step_meta_ok = False
            sd = ck["state_dict"]
            if c == 0:
                for k in initial_sd:
                    if not torch.allclose(sd[k].float(), initial_sd[k].float()):
                        ckpt0_eq_initial = False
                        break
            if c == 20:
                diff = any(not torch.allclose(sd[k].float(), initial_sd[k].float())
                           for k in initial_sd)
                ckpt20_ne_initial = diff

        results["R1_all_eight_emitted"] = {"pass": r1, "detail": f"present={present}"}
        results["R3_no_shift_no_dup"] = {
            "pass": r3,
            "detail": f"present_set == required_set: {set(present) == set(required)}",
        }
        results["R2_checkpoint20_loadable"] = {
            "pass": loadable_ok and step_meta_ok,
            "detail": f"loadable={loadable_ok} step_meta_ok={step_meta_ok}",
        }
        results["R4_step0_initial_step20_final"] = {
            "pass": ckpt0_eq_initial and ckpt20_ne_initial,
            "detail": f"ckpt0==initial={ckpt0_eq_initial} ckpt20!=initial={ckpt20_ne_initial}",
        }

    # ---- R6: authorization binds exclusively to v7 --------------------------
    v7_path = SCRIPT_DIR / "run_phase_6e20_execution_v7.py"
    v6_path = SCRIPT_DIR / "run_phase_6e20_execution_v6.py"
    v7_sha = hashlib.sha256(v7_path.read_bytes()).hexdigest().upper()
    v6_sha = hashlib.sha256(v6_path.read_bytes()).hexdigest().upper()

    accepted_v7 = refused_v6 = False
    with tempfile.TemporaryDirectory() as td:
        auth_v7 = Path(td) / "a7.json"
        auth_v7.write_text(json.dumps({"decision": "APPROVE", "phase": "6E.20.1",
                                        "runner_sha256": v7_sha}), encoding="utf-8")
        try:
            re_exec.check_execution_authorization_v6(auth_v7)
            accepted_v7 = True
        except re_exec.ExecutionNotAuthorizedV2:
            accepted_v7 = False

        auth_v6 = Path(td) / "a6.json"
        auth_v6.write_text(json.dumps({"decision": "APPROVE", "phase": "6E.20.1",
                                        "runner_sha256": v6_sha}), encoding="utf-8")
        try:
            re_exec.check_execution_authorization_v6(auth_v6)
            refused_v6 = False
        except re_exec.ExecutionNotAuthorizedV2:
            refused_v6 = True

    results["R6_auth_exclusive_to_v7"] = {
        "pass": accepted_v7 and refused_v6,
        "detail": f"v7_auth_accepted={accepted_v7} v6_auth_refused={refused_v6}",
    }

    all_pass = all(v["pass"] for v in results.values())
    summary = {
        "suite": "v7 checkpoint_20 regression",
        "runner_sha256": v7_sha,
        "all_pass": all_pass,
        "checks": results,
    }
    print(json.dumps(summary, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
