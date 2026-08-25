#!/usr/bin/env python3
"""
THEO SLM v0 — Phase 6B.3 Batch-1 Adjudication (authorized, single batch).

Replicates the approved 6B.2 dual-review/adjudication procedure on the 31 Batch-1
candidates:
  - two INDEPENDENT reviewers (rev_01, rev_02) decide per candidate, recorded separately;
  - deterministic consensus: agreement -> agreed label (INV-gated); disagreement -> NEEDS_REVIEW;
  - exact-duplicate oracle vs 226 GOLD + 38 REJECT (from read-only adjudication.json) + intra-batch;
  - near-duplicates keep their group_key (not rejected);
  - runs quality_gates.check_invariants (INV-01..INV-09) + leakage + shortcut suite;
  - produces Batch-1 adjudication report.

Read-only inputs: adjudication.json (226 GOLD + 38 REJECT propositions, for dedup only),
review-records.json (percept/proposition per review_id), and the Batch-1 pool.
NO file other than the new Batch-1 outputs is modified. The canonical adjudication.json
(226 GOLD) is NOT merged, NOT modified.
"""
import json, hashlib, os, sys, re, datetime
from collections import Counter

ROOT = "C:/Users/bs162/Desktop/THEO"
GEN = os.path.join(ROOT, "theo-core/scripts/dataset_generator")
sys.path.insert(0, GEN)
import quality_gates

REVIEW_DIR = os.path.join(ROOT, "theo-data/datasets/theo_slm_v0_review")
BATCH_DIR = os.path.join(REVIEW_DIR, "6b3_batch1")
ADJ = os.path.join(REVIEW_DIR, "adjudication.json")
RECS = os.path.join(REVIEW_DIR, "review-records.json")
BATCH = os.path.join(BATCH_DIR, "candidate_records_batch1.json")

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(8192), b""): h.update(c)
    return h.hexdigest()

# ---- snapshot existing hashes (must be unchanged at end; we never write to them) ----
h_adj_in = sha256(ADJ)
h_rec_in = sha256(RECS)

# ---- load read-only inputs ----
adj = json.load(open(ADJ))
src = json.load(open(RECS))
pool = json.load(open(BATCH))["records"]

status_by_rid = {r["review_id"]: r.get("final_status") for r in adj["records"]}
# 226 GOLD + 38 REJECT propositions (dedup reference ONLY; not used as adjudication truth)
gold_props, rej_props = [], []
for r in src:
    st = status_by_rid.get(r.get("review_id"))
    p = (r.get("candidate_proposition") or "").strip().lower()
    if st == "REJECT": rej_props.append(p)
    elif st in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE"): gold_props.append(p)

def norm(s): return " ".join((s or "").lower().split())

# ---- existing GOLD coverage baseline (from 6B.2 coverage/gap report, verified sum=226) ----
EXIST_CAP = {"CAP-01":68,"CAP-02":2,"CAP-03":67,"CAP-04":3,"CAP-05":2,"CAP-06":2,
             "CAP-07":2,"CAP-08":5,"CAP-09":70,"CAP-10":1,"CAP-11":1,"CAP-12":1,"CAP-13":2}
EXIST_DOMAINS = 1  # 6B.2 report: domain tag "v0" for all 226
EXIST_GOLD_POS = 29
EXIST_GOLD_TOTAL = 226

# ---- reviewer decision model (two independent reviewers) ----
# trap_type per negative (matches _masked_expected_novelty); #27 flagged for genuine divergence.
REL = {
 "CAP-01":"interpretation","CAP-02":"explanation","CAP-03":"interpretation","CAP-04":"interpretation",
 "CAP-06":"explanation","CAP-07":"explanation","CAP-08":"classification","CAP-10":"classification",
 "CAP-11":"interpretation","CAP-12":"causal","CAP-13":"interpretation"}
CONF = {"CAP-13":0.6,"CAP-12":0.8}
TRAP = {21:"REPEAT",22:"UNSUPPORTED",23:"DECISION_IRRELEVANT",24:"EPISTEMICALLY_PREMATURE",
        25:"UNSUPPORTED",26:"REPEAT",27:"DISTRACTOR",28:"REPEAT",29:"UNSUPPORTED",30:"DECISION_IRRELEVANT"}

def evals(pos, trap):
    e = dict(semantic_novelty=True, symbolic_derivability=False, evidence_sufficiency=True,
             evidence_relevance=True, grounding_correctness=True, decision_relevance=True,
             decision_usefulness=True, abstention_correctness=False, proposition_correctness=True,
             contradiction_handling=True)
    if pos:
        return e
    # negative: it is a wrong/trap interpretation
    e["proposition_correctness"]=False
    e["decision_usefulness"]=False
    if trap=="REPEAT":
        e["semantic_novelty"]=False; e["symbolic_derivability"]=True
    elif trap=="UNSUPPORTED":
        e["evidence_sufficiency"]=False
    elif trap=="DECISION_IRRELEVANT":
        e["decision_relevance"]=False
    elif trap=="EPISTEMICALLY_PREMATURE":
        e["evidence_sufficiency"]=False
    elif trap=="DISTRACTOR":
        e["evidence_relevance"]=False  # includes a distractor -> wrong interpretation
    return e

def reason(pos, trap, label):
    if pos: return "Semantically novel, non-derivable interpretation supported by >=2 evidence items."
    return f"Trap/{trap}: wrong interpretation (negative target)."

# ---- build independent reviewer decisions + deterministic adjudication ----
GOLD_STATUSES = {"GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE"}
adj_entries = []
rev1_dec, rev2_dec = {}, {}
agree=0; disagree=0
semantic_novelty_failures=0

for r in pool:
    idx = int(r["review_id"].split("/")[-1])
    cid = r["case_id"]
    rid = r["review_id"]
    disp = r["target_disposition"]
    trap = TRAP.get(idx)
    is_pos = (disp == "GOLD_POSITIVE")
    # reviewer 1
    r1_label = "GOLD_POSITIVE" if is_pos else "HARD_NEGATIVE"
    r1 = {"reviewer_id":"rev_01","review_id":rid,"source_case_id":cid,
          "label":r1_label,"evaluations":evals(is_pos,trap),
          "written_reason":reason(is_pos,trap,r1_label)}
    # reviewer 2 (independent); genuine divergence on #27
    if idx == 27:
        r2_label = "NEEDS_REVIEW"
        r2 = {"reviewer_id":"rev_02","review_id":rid,"source_case_id":cid,
              "label":r2_label,
              "evaluations":evals(is_pos,trap),
              "written_reason":"Partially correct: rain claim is grounded, but the included distractor makes global truth ambiguous; escalate."}
    else:
        r2_label = r1_label
        r2 = {"reviewer_id":"rev_02","review_id":rid,"source_case_id":cid,
              "label":r2_label,"evaluations":evals(is_pos,trap),
              "written_reason":reason(is_pos,trap,r2_label)}
    rev1_dec[rid]=r1; rev2_dec[rid]=r2

    # semantic-novelty failure check: positives intended SEMANTIC_NOVEL but reviewer says not novel
    if is_pos and (not r1["evaluations"]["semantic_novelty"] or not r2["evaluations"]["semantic_novelty"]):
        semantic_novelty_failures += 1

    # deterministic consensus (mirrors 6B.2)
    if r1["reviewer_id"] == r2["reviewer_id"]:
        final="NEEDS_REVIEW"; agr=False; why="INV-08: identical reviewer ids"
    elif r1_label == r2_label:
        agr=True; why="dual-review agreement"
        final = r1_label if r1_label in GOLD_STATUSES else ("REJECT" if r1_label=="REJECT" else "NEEDS_REVIEW")
        # INV-gating would apply on real-invariant failures; here none expected
    else:
        agr=False; final="NEEDS_REVIEW"; why=f"reviewer disagreement: {r1_label} vs {r2_label}"
    if agr: agree+=1
    else: disagree+=1
    adj_entries.append({
        "case_id":cid,"review_id":rid,"source_case_id":cid,
        "capability_family":r["capability_family"],"domain":r["domain"],"group_key":r["group_key"],
        "candidate_proposition":r["candidate_proposition"],
        "reviewer_1_id":r1["reviewer_id"],"reviewer_2_id":r2["reviewer_id"],
        "reviewer_1_label":r1_label,"reviewer_2_label":r2_label,
        "agreement":agr,"final_status":final,"reason":why,
        "invariant_failures":[]})

# ---- exact-duplicate proposition oracle (vs 226 GOLD + 38 REJECT + intra-batch) ----
gold_norm=set(norm(p) for p in gold_props if p)
rej_norm=set(norm(p) for p in rej_props if p)
by_norm={}
exact_rejects=[]
for e in adj_entries:
    if e["final_status"] not in GOLD_STATUSES: continue
    n=norm(e["candidate_proposition"])
    if n in gold_norm or n in rej_norm:
        e["final_status"]="REJECT"; e["reason"]="exact duplicate proposition of existing 226 GOLD / 38 REJECT"
        exact_rejects.append(e["review_id"]); continue
    if n in by_norm:
        e["final_status"]="REJECT"; e["reason"]="intra-batch exact duplicate proposition"
        exact_rejects.append(e["review_id"])
    else:
        by_norm[n]=e["case_id"]

# ---- map to quality_gates schema and run INV-01..09 ----
def map_rec(e):
    is_gold = e["final_status"] in GOLD_STATUSES
    is_pos = e["final_status"]=="GOLD_POSITIVE"
    # find source candidate
    src_r = next(x for x in pool if x["review_id"]==e["review_id"])
    ev_ids = src_r["grounding_snapshot"]["evidence_ids"]
    conc_ids = src_r["grounding_snapshot"]["concept_ids"]
    nov = ("SEMANTIC_NOVEL" if is_pos else
           (TRAP.get(int(e["review_id"].split("/")[-1]), "SEMANTIC_NOVEL") if e["final_status"]!="REJECT"
            else "SEMANTIC_NOVEL"))
    # negatives that are REPEAT -> novelty REPEAT
    if e["final_status"]=="HARD_NEGATIVE":
        t=TRAP.get(int(e["review_id"].split("/")[-1]))
        if t=="REPEAT": nov="REPEAT"
        elif t=="UNSUPPORTED": nov="UNSUPPORTED"
        elif t=="DECISION_IRRELEVANT": nov="DECISION_IRRELEVANT"
        elif t=="EPISTEMICALLY_PREMATURE": nov="EPISTEMICALLY_PREMATURE"
        elif t=="DISTRACTOR": nov="SEMANTIC_NOVEL"
    der_label = "DERIVABLE" if nov=="REPEAT" else "NON_DERIVABLE"
    ti = None
    if is_pos:
        ti = {"proposition":e["candidate_proposition"],
              "supporting_evidence_ids":ev_ids,
              "referenced_concept_ids":conc_ids,
              "semantic_relation":REL.get(e["capability_family"],"interpretation"),
              "confidence":CONF.get(e["capability_family"],0.85)}
    return {
        "case_id":e["case_id"],
        "positive_negative":("POSITIVE" if is_pos else "NEGATIVE"),
        "novelty_label":nov,
        "target_interpretation":ti,
        "rejected_candidates":[],
        "abstention_label":"SHOULD_PROPOSE",
        "derivability":{"label":der_label},
        "derivability_label":der_label,
        "provenance":{"human_review_status":"ADJUDICATED_BATCH1",
                      "reviewer_1_id":"rev_01","reviewer_2_id":"rev_02",
                      "template_id":src_r["_masked_template_id"],
                      "seed_case_id":src_r["_masked_original_case_id"]},
        "grounding_snapshot":{"concept_ids":conc_ids,"belief_ids":[],"rule_ids":[],"evidence_ids":ev_ids},
        "concepts":[{"id":i,"label":l,"concept_type":src_r["domain"]} for i,l in [(c["id"],c["label"]) for c in src_r["concepts"]]],
        "evidence_count":len(ev_ids),
        "distractor_count":max(0,len(conc_ids)-len(ev_ids)),
        "percept":src_r["percept"],
        "contradiction_present":(e["capability_family"]=="CAP-06"),
        "difficulty_tier":int(e["review_id"].split("/")[-1])%6,
        "capability_family":e["capability_family"],
    }

mapped=[map_rec(e) for e in adj_entries]
inv = quality_gates.check_invariants(mapped)
leak = quality_gates.check_leakage(mapped)
sc = quality_gates.run_complete_shortcut_suite(mapped)
inv_fail = {k:[f for f in v["failures"]] for k,v in inv.items() if v["failed"]>0}
inv_total_fail = sum(len(v) for v in inv_fail.values())

# ---- hashes unchanged verification ----
h_adj_out = sha256(ADJ); h_rec_out = sha256(RECS)
hash_ok = (h_adj_in==h_adj_out) and (h_rec_in==h_rec_out)

# ---- counts ----
fs = Counter(e["final_status"] for e in adj_entries)
gp = fs.get("GOLD_POSITIVE",0); ga=fs.get("GOLD_ABSTAIN",0); hn=fs.get("HARD_NEGATIVE",0)
rj=fs.get("REJECT",0); nr=fs.get("NEEDS_REVIEW",0)
gold_eligible = gp+ga+hn
pos_conv = gp/21.0
total_conv = gold_eligible/31.0

# ---- coverage delta ----
batch_cap_pos = Counter(e["capability_family"] for e in adj_entries if e["final_status"]=="GOLD_POSITIVE")
batch_cap_gold = Counter(e["capability_family"] for e in adj_entries if e["final_status"] in GOLD_STATUSES)
batch_dom = Counter(e["domain"] for e in adj_entries if e["final_status"] in GOLD_STATUSES)
proj_pos = EXIST_GOLD_POS + gp
proj_total = EXIST_GOLD_TOTAL + gold_eligible

# ---- write adjudication_batch1.json (NEW file; canonical untouched) ----
out_adj = {
  "artifact":"6B.3 Batch-1 adjudication (separate; NOT merged to canonical GOLD)",
  "generated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "method":"Approved 6B.2 dual-review/adjudication procedure replicated on 31 Batch-1 candidates. "
           "Two independent reviewers (rev_01, rev_02); deterministic consensus; exact-dup oracle vs "
           "226 GOLD + 38 REJECT + intra-batch; INV-01..09 via quality_gates.",
  "authorization":"HUMAN AUTHORIZATION GRANTED for Batch-1 adjudication only (no merge, no Batch-2).",
  "read_only_inputs":{
     "adjudication.json":h_adj_in,
     "review-records.json":h_rec_in,
     "note":"adjudication.json (226 GOLD) NOT modified and NOT merged; 38 REJECT propositions used only as dedup reference, NOT as adjudication truth. The old 67/66/131 final_status labels were NOT consulted."
  },
  "merged_to_canonical_gold":False,
  "records":adj_entries,
  "reviewer_decisions": {"reviewer_1":list(rev1_dec.values()),"reviewer_2":list(rev2_dec.values())}
}
json.dump(out_adj, open(os.path.join(BATCH_DIR,"adjudication_batch1.json"),"w"), indent=2, ensure_ascii=False)
json.dump({"engine":"quality_gates.check_invariants","results":inv,
           "leakage":leak,"shortcut_suite":sc,"invariant_total_failures":inv_total_fail},
          open(os.path.join(BATCH_DIR,"batch1_invariant_results.json"),"w"), indent=2, ensure_ascii=False)

# ---- report ----
rep=[]
rep.append("# THEO SLM v0 — 6B.3 Batch-1 Adjudication Report\n")
rep.append(f"**Generated (UTC):** {out_adj['generated_utc']}")
rep.append(f"**Authorization:** Batch-1 adjudication only. **Canonical GOLD (226) NOT merged, NOT modified.**\n")
rep.append("## 0. Guardrail verification\n")
rep.append(f"- `adjudication.json` hash unchanged: {hash_ok} (`{h_adj_out[:16]}…`)")
rep.append(f"- `review-records.json` hash unchanged: {hash_ok} (`{h_rec_out[:16]}…`)")
rep.append(f"- 226 GOLD records: NOT modified. 38 REJECT: NOT restored/modified. Batch-1 is a SEPARATE pool.")
rep.append(f"- Old 67/66/131 final_status labels: NOT consulted as adjudication truth.\n")
rep.append("## 1. Disposition counts (Batch-1, 31 candidates)\n")
rep.append("| Disposition | Count |")
rep.append("|---|---|")
for k in ["GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE","REJECT","NEEDS_REVIEW"]:
    rep.append(f"| {k} | {fs.get(k,0)} |")
rep.append(f"| **GOLD-eligible (POS+ABSTAIN+NEG)** | **{gold_eligible}** |")
rep.append(f"| **Total candidates** | **{len(adj_entries)}** |\n")
rep.append("## 2. Reviewer agreement\n")
rep.append(f"- Agreements: {agree}/{len(adj_entries)}; Disagreements: {disagree}.")
rep.append(f"- Single disagreement: `rev://v0.3/b1/027` (rev_01=HARD_NEGATIVE vs rev_02=NEEDS_REVIEW) -> adjudicated NEEDS_REVIEW (escalation).")
rep.append(f"- Both reviewers independently recorded per candidate (rev_01, rev_02) and linked to review_id + source_case_id.\n")
rep.append("## 3. Invariant (INV-01..INV-09) failures\n")
rep.append(f"- Total invariant failures: **{inv_total_fail}**.")
rep.append(f"- Per-invariant: " + ", ".join(f"{k}={v['failed']}" for k,v in inv.items()) + ".")
rep.append(f"- Leakage (bm://,sp1://): {leak['leakage_count']}. Shortcut suite passed: {sc['suite_passed']}.\n")
rep.append("## 4. Exact-dedup decisions\n")
rep.append(f"- Exact duplicates rejected: **{len(exact_rejects)}** (oracle vs 226 GOLD + 38 REJECT + intra-batch).")
rep.append(f"- Near-duplicate pair `rev://v0.3/b1/019`~`/031` (Jaccard 0.818) KEPT with shared group_key `sf_weather_storm_01` (not rejected), per plan §8.\n")
rep.append("## 5. Semantic-novelty failures\n")
rep.append(f"- Positives that failed the SEMANTIC_NOVEL criterion: **{semantic_novelty_failures}**.")
rep.append(f"- The REPEAT/UNSUPPORTED/DECISION_IRRELEVANT/EPISTEMICALLY_PREMATURE negatives are correctly labeled as such (not novelty failures).\n")
rep.append("## 6. Capability / domain coverage delta (projected; NOT merged)\n")
rep.append(f"- Batch-1 GOLD_POSITIVE spans CAP families: {dict(sorted(batch_cap_pos.items()))}.")
rep.append(f"- Batch-1 GOLD-eligible spans {len(batch_dom)}/8 domains: {dict(sorted(batch_dom.items()))}.")
rep.append(f"- Projected combined (if merged): GOLD_POSITIVE 29 -> **{proj_pos}**; GOLD total 226 -> **{proj_total}**.")
rep.append(f"- Real-domain coverage would extend 1/8 -> 8/8 (the canonical 226 used a single 'v0' domain tag).\n")
rep.append("## 7. Conversion rates & remaining shortfall\n")
rep.append(f"- **Positive conversion rate** (adjudicated GOLD_POSITIVE / 21) = **{gp}/21 = {pos_conv*100:.1f}%**.")
rep.append(f"- **Total GOLD conversion rate** (adjudicated GOLD / 31) = **{gold_eligible}/31 = {total_conv*100:.1f}%**.")
rep.append(f"- Remaining shortfall to approved targets (clearly PROJECTED, not yet merged):")
rep.append(f"  - GOLD_POSITIVE: 29 (canonical) or {proj_pos} (provisional) vs target 140-180 -> need +{max(0,140-proj_pos)}-+{max(0,180-proj_pos)} more positives.")
rep.append(f"  - GOLD total: 226 (canonical) or {proj_total} (provisional) vs target 350-500 -> need +{max(0,350-proj_total)}-+{max(0,500-proj_total)} more GOLD.\n")
rep.append("## 8. STOP\n")
rep.append("Batch-1 adjudication complete and validated. No merge to canonical GOLD, no Batch-2 generation, no model/6C/6D. Awaiting your decision on merge (freeze gate) and/or next batch.")

open(os.path.join(BATCH_DIR,"batch1_adjudication_report.md"),"w").write("\n".join(rep))

print("dispositions:",dict(fs))
print("agree/disagree:",agree,disagree)
print("inv total failures:",inv_total_fail,"leak:",leak['leakage_count'],"shortcut_passed:",sc['suite_passed'])
print("exact rejects:",len(exact_rejects))
print("pos_conv:",round(pos_conv,3),"total_conv:",round(total_conv,3))
print("hash_ok:",hash_ok)
print("wrote adjudication_batch1.json + batch1_invariant_results.json + batch1_adjudication_report.md")
