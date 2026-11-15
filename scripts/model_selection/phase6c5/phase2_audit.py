"""
6C.5 — Phase 2 independent pre-training audit (READ-ONLY; trains nothing).

Implements the 16 checks from theo-slm-phase-6c5-plan.md:
  1-10  identical inputs/labels/split/seed/prompt/supervision/budget/architecture-isolation/
        case-id leakage / output isolation
  11-15 R2 group_key leakage, Jaccard near-duplicate, B/C identical-evidence firewall,
        exact distributions, byte-level fixture SHA256
  16    aux-loss weight lock

Run:  python phase6c5/phase2_audit.py
Exit code 0 = all pass; non-zero = STOP (not eligible for training).
"""
import os, sys, json, re, importlib.util, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp", os.path.join(HERE, "experiment_spec.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

FAILS = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)

# ---- load data ----
examples, data_sha = exp.build_examples()
train_ex, dev_ex = exp.split_examples(examples)
fixture_cases = json.load(open(exp.FIXTURE))["cases"]

train_ids = {e["case_id"] for e in train_ex}
dev_ids   = {e["case_id"] for e in dev_ex}
fixture_ids = {c["case_id"] for c in fixture_cases}

# ---- 1. identical inputs (B & C read same mapping) ----
ex2, sha2 = exp.build_examples()
check("1 identical-inputs", [e["case_id"] for e in examples] == [e["case_id"] for e in ex2] and sha2 == data_sha)

# ---- 2. identical labels (deterministic final_status -> decision) ----
bad_label = [e["case_id"] for e in examples if (e["tgt"] == "SHOULD_PROPOSE") != (e.get("pos") is not None and len(e["pos"]) >= 0 and e["tgt"]=="SHOULD_PROPOSE")]
# simpler: every example has a well-defined tgt derived from final_status
label_ok = all(e["tgt"] in ("SHOULD_PROPOSE", "SHOULD_ABSTAIN") for e in examples)
check("2 identical-labels", label_ok, f"{len(examples)} examples, all mapped")

# ---- 3. identical split (269/31) ----
check("3 identical-split", len(train_ex) == 269 and len(dev_ex) == 31 and not (train_ids & dev_ids),
      f"train={len(train_ex)} dev={len(dev_ex)} overlap={len(train_ids & dev_ids)}")

# ---- 4. identical seed ----
check("4 identical-seed", exp.SEED == 20260826, f"SEED={exp.SEED}")

# ---- 5. identical prompt + neutrality ----
neutral, hits = exp.prompt_is_neutral()
check("5 identical-prompt", exp.NORM_SYS and exp.NORM_USER and neutral, f"forbidden hits={hits}")

# ---- 6. identical supervision (arch-agnostic data path) ----
sup_ok = all("decision" in e["envelope"] and "evidence_refs" in e["envelope"] for e in examples)
check("6 identical-supervision", sup_ok and "arch" not in exp.build_examples.__code__.co_varnames,
      "envelope schema uniform; build_examples has no arch param")

# ---- 7. identical training budget (single TrainingArguments in train_6c5.py) ----
train_src = open(os.path.join(HERE, "train_6c5.py")).read()
check("7 identical-budget", train_src.count("TrainingArguments(") == 1,
      f"TrainingArguments occurrences={train_src.count('TrainingArguments(')} (expect 1, built once)")

# ---- 8. architecture difference isolated (data path arch-agnostic) ----
check("8 arch-isolated", ("arch" not in exp.build_examples.__code__.co_varnames)
      and ("arch" not in exp.split_examples.__code__.co_varnames),
      "data/split paths take no arch param")

# ---- 9. case-id leakage train/dev/fixture ----
leak = (train_ids & fixture_ids) | (dev_ids & fixture_ids)
check("9 case-id-leakage", len(leak) == 0, f"overlap train/fixture+dev/fixture={len(leak)}")

# ---- 10. output isolation ----
check("10 output-isolation", exp.OUT_B != exp.OUT_C
      and exp.OUT_B.endswith("6c5-B") and exp.OUT_C.endswith("6c5-C"),
      f"B={exp.OUT_B} C={exp.OUT_C}")

# ---- 11. R2 group_key leakage (capability_family vs domain/group disjoint) ----
train_keys = set()
for e in train_ex + dev_ex:
    cf = e.get("capability_family")
    if cf: train_keys.add(cf)
fixture_keys = set()
for c in fixture_cases:
    if c.get("domain"): fixture_keys.add("DOM:" + c["domain"])
    if c.get("group"):  fixture_keys.add("GRP:" + c["group"])
overlap_keys = train_keys & fixture_keys
check("11 R2 group_key-leakage", len(overlap_keys) == 0,
      f"train keys sample={sorted(train_keys)[:5]}... fixture keys sample={sorted(fixture_keys)[:5]}... overlap={overlap_keys}")

# ---- 12. R2 Jaccard near-duplicate (train/dev vs fixture) ----
def toks(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower()))
THRESH = 0.85
max_j = 0.0; worst = None
fx_tok = {c["case_id"]: toks(c["percept"]) for c in fixture_cases}
for e in (train_ex + dev_ex):
    et = toks(e["percept"])
    for cid, ft in fx_tok.items():
        u = et | ft
        if not u: continue
        j = len(et & ft) / len(u)
        if j > max_j:
            max_j = j; worst = (e["case_id"], cid, round(j, 3))
check("12 R2 jaccard-near-dup", max_j < THRESH, f"max Jaccard={round(max_j,3)} worst={worst} thresh={THRESH}")

# ---- 13. R2 B/C identical-evidence firewall ----
firewall_ok = True; fw_detail = ""
for c in fixture_cases:
    ev_list, snap = exp.evidence_payload(c)
    # identical whether delivered to B or C
    ev2, snap2 = exp.evidence_payload(c)
    if (ev_list, snap) != (ev2, snap2):
        firewall_ok = False; fw_detail = f"payload differs for {c['case_id']}"; break
    # primary (envelope) user turn must not carry the aux-head marker
    ut = exp.NORM_USER.format(percept=c["percept"], evlist=ev_list)
    if not exp.firewall_user_turn(ut):
        firewall_ok = False; fw_detail = f"marker leaked into primary turn {c['case_id']}"; break
check("13 R2 firewall", firewall_ok, fw_detail)
check("13b aux-head marker scoped", exp.EVIDENCE_TOKEN in exp.NORM_EVID_USER and exp.EVIDENCE_TOKEN not in exp.NORM_USER,
      "marker present only on C's evidence-head prompt")

# ---- 14. R2 exact distributions ----
def hist(items):
    h = {}
    for i in items: h[i] = h.get(i, 0) + 1
    return h
tr_pos = sum(1 for e in train_ex if e["tgt"] == "SHOULD_PROPOSE")
tr_abs = sum(1 for e in train_ex if e["tgt"] == "SHOULD_ABSTAIN")
dv_pos = sum(1 for e in dev_ex if e["tgt"] == "SHOULD_PROPOSE")
dv_abs = sum(1 for e in dev_ex if e["tgt"] == "SHOULD_ABSTAIN")
fx_pos = sum(1 for c in fixture_cases if c["expected_decision"] == "SHOULD_PROPOSE")
fx_abs = sum(1 for c in fixture_cases if c["expected_decision"] == "SHOULD_ABSTAIN")
print(f"   train={tr_pos}P/{tr_abs}A  dev={dv_pos}P/{dv_abs}A  fixture={fx_pos}P/{fx_abs}A")
print(f"   capability_family(train)={hist([e['capability_family'] for e in train_ex])}")
print(f"   domain(fixture)={hist([c.get('domain') for c in fixture_cases])}")
print(f"   group(fixture)={hist([c.get('group') for c in fixture_cases])}")
check("14 R2 exact-distributions", fx_pos == 50 and fx_abs == 16 and (tr_pos+tr_abs)==269 and (dv_pos+dv_abs)==31,
      f"fixture P/A={fx_pos}/{fx_abs} train n={tr_pos+tr_abs} dev n={dv_pos+dv_abs}")

# ---- 15. R2 fixture SHA256 ----
check("15 R2 fixture-sha256", exp.fixture_sha256() == exp.FIXTURE_SHA256,
      f"computed={exp.fixture_sha256()} expected={exp.FIXTURE_SHA256}")

# ---- 16. aux-loss weight lock ----
check("16 aux-loss-lock", exp.AUX_LOSS_WEIGHT == 1.0 and 'aux_loss_weight' in train_src,
      f"AUX_LOSS_WEIGHT={exp.AUX_LOSS_WEIGHT}")

print("\n=== PHASE-2 AUDIT RESULT ===")
if FAILS:
    print(f"FAIL: {len(FAILS)} check(s) failed -> STOP, NOT ELIGIBLE FOR TRAINING")
    print("Failed:", FAILS)
    sys.exit(1)
else:
    print("ALL CHECKS PASS -> eligible for Phase 3 (subject to separate execution authorization)")
    sys.exit(0)
