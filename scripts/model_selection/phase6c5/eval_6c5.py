"""
6C.5 — Controlled B/C evaluation harness (PREPARATION ONLY — NOT EXECUTED).

Single code path for B and C; identical evaluation rules. Reports the relocated
gate set: accuracy, E0, Group E, grounding, ABSTAIN RECALL (denominator = 16
genuine ABSTAIN), PROPOSAL PRECISION, FALSE-PROPOSAL RATE (must-abstain),
GENERATION LATENCY (full decode) and DECISION-PATH LATENCY (deterministic gate only).

Run only after Phase 3 training AND explicit authorization.
"""
import os, sys, json, time, re, importlib.util, hashlib
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp", os.path.join(HERE, "experiment_spec.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

DEC_RE = re.compile(r'"decision"\s*:\s*"(SHOULD_PROPOSE|SHOULD_ABSTAIN)"')
REF_RE = re.compile(r'"evidence_refs"\s*:\s*(\[[^\]]*\])')

def load_adapted(arch):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(exp.M1_DIR, quantization_config=bnb,
                 torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    adir = exp.OUT_C if arch == "C" else exp.OUT_B
    model = PeftModel.from_pretrained(base, adir, local_files_only=True); model.eval()
    tok = AutoTokenizer.from_pretrained(exp.M1_DIR, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return base, model, tok

def generate(model, tok, sys_p, user, max_new=96):
    chat = tok.apply_chat_template([{"role":"system","content":sys_p},{"role":"user","content":user}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(chat, return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen, time.time() - t0

def parse_env(gen):
    m = DEC_RE.search(gen); decision = m.group(1) if m else None
    refs = []
    jm = REF_RE.search(gen)
    if jm:
        try: refs = json.loads(jm.group(1))
        except Exception: refs = []
    return decision, refs

def evaluate(arch):
    base, model, tok = load_adapted(arch)
    corpus = json.load(open(exp.FIXTURE))["cases"]
    rows = []
    for c in corpus:
        per = c["percept"]
        ev_list, snapshot = exp.evidence_payload(per, c["evidence"])
        # Firewall (R2/R3): primary user turn carries NO auxiliary-head marker, identically for B and C.
        user_turn = exp.NORM_USER.format(percept=per, evlist=ev_list)
        assert exp.firewall_user_turn(user_turn), "firewall violated: aux-head marker leaked into primary turn"
        t0 = time.time()
        gen_d, lat_d = generate(model, tok, exp.NORM_SYS, user_turn)
        decision, pred_refs = parse_env(gen_d)
        gen_lat = time.time() - t0
        ev_head_refs = []
        if arch == "C":
            gen_e, _ = generate(model, tok, exp.NORM_EVID_SYS, exp.NORM_EVID_USER.format(percept=per, evlist=ev_list), max_new=64)
            try: ev_head_refs = json.loads(re.search(r'\{.*\}', gen_e, re.S).group(0)).get("relevant_evidence", []) or []
            except Exception: ev_head_refs = []
        # decision-path gate (deterministic)
        cand = list(pred_refs) + [r for r in (ev_head_refs or []) if r not in pred_refs]
        verified = [r for r in cand if r in snapshot]
        tg0 = time.time()
        if decision == "SHOULD_PROPOSE" and not verified:
            final = "SHOULD_ABSTAIN"   # override unsupported proposal
        else:
            final = decision or "SHOULD_ABSTAIN"
        gate_dt = time.time() - tg0
        rows.append({"case_id": c["case_id"], "group": c.get("group"), "expected": c["expected_decision"],
                     "decision_raw": decision, "final_decision": final, "verified_refs": verified,
                     "gen_latency": gen_lat, "decision_path_latency": gate_dt})
    del model, base; gc.collect(); torch.cuda.empty_cache()
    return rows

def summarize(arch, rows):
    n = len(rows)
    acc = sum(r["final_decision"] == r["expected"] for r in rows) / n
    e0 = sum(1 for r in rows if r["final_decision"] in ("SHOULD_PROPOSE","SHOULD_ABSTAIN")) / n
    genuine_abstain = [r for r in rows if r["expected"] == "SHOULD_ABSTAIN"]
    n_ab = len(genuine_abstain)
    ab_recall = sum(1 for r in genuine_abstain if r["final_decision"] == "SHOULD_ABSTAIN") / n_ab
    false_prop = sum(1 for r in genuine_abstain if r["final_decision"] == "SHOULD_PROPOSE") / n_ab
    pred_prop = [r for r in rows if r["final_decision"] == "SHOULD_PROPOSE"]
    prec = (sum(1 for r in pred_prop if r["expected"] == "SHOULD_PROPOSE") / len(pred_prop)) if pred_prop else 1.0
    ge = [r for r in rows if r["group"] == "E"]
    ge_acc = sum(r["final_decision"] == r["expected"] for r in ge) / len(ge)
    grounded = all(len(r["verified_refs"]) > 0 for r in pred_prop)
    gen_lat = sum(r["gen_latency"] for r in rows) / n
    dp_lat = sum(r["decision_path_latency"] for r in rows) / n
    return {"architecture": arch, "n": n, "accuracy": round(acc,4), "E0": round(e0,4),
            "abstain_recall": round(ab_recall,4), "proposal_precision": round(prec,4),
            "false_proposal_rate": round(false_prop,4), "group_E_accuracy": round(ge_acc,4),
            "grounding_100pct": grounded, "gen_latency_s": round(gen_lat,4),
            "decision_path_latency_s": round(dp_lat,4)}

if __name__ == "__main__":
    rows_all = {}; summ = {}
    for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["B", "C"]):
        rows = evaluate(a); rows_all[a] = rows; summ[a] = summarize(a, rows)
        print(json.dumps(summ[a]))
    json.dump({"summary": summ, "rows": rows_all},
              open(os.path.join(exp.BATCH2, "6c5_evaluation_results.json"), "w"), indent=2)
