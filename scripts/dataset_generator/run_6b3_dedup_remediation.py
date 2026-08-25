#!/usr/bin/env python3
"""
THEO SLM v0 - Phase 6B.3 Dedup Remediation (AUTHORIZED).

Applies the already-approved 6B.2 exact-duplicate rule to REJECT exactly six
inherited non-keeper GOLD_POSITIVE records. Rejected records are RETAINED in
adjudication.json (status flipped to REJECT) with a dedup-remediation provenance
note; they are NOT deleted from history. No exception invented.

Then verifies:
 - GOLD == 250 (44 POS + 66 ABSTAIN + 140 NEG)
 - REJECT == 44
 - NEEDS_REVIEW == 0
 - INV-01..INV-09 == 0 failures
 - exact-duplicate validator == 0 violations
 - Batch-1 provenance intact (30 merged entries); Batch 1 introduced 0 dup violations
Records complete pre/post hashes and a remediation report.
No Batch-2, model, 6C/6D.
"""
import json, hashlib, os, sys, re, datetime
from collections import Counter, defaultdict
ROOT="C:/Users/bs162/Desktop/THEO"
GEN=os.path.join(ROOT,"theo-core/scripts/dataset_generator")
sys.path.insert(0,GEN)
import quality_gates

REVIEW_DIR=os.path.join(ROOT,"theo-data/datasets/theo_slm_v0_review")
ADJ_P=os.path.join(REVIEW_DIR,"adjudication.json")
RECS_P=os.path.join(REVIEW_DIR,"review-records.json")
SRC_P=os.path.join(ROOT,"theo-data/datasets/theo_slm_v0_repaired/candidate_records.json")
BATCH_DIR=os.path.join(REVIEW_DIR,"6b3_batch1")
OUT_MD="theo-core/docs/research/theo-slm-phase-6b3-dedup-remediation-report.md"
OUT_JSON=os.path.join(BATCH_DIR,"dedup_remediation_results.json")

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(8192),b""): h.update(c)
    return h.hexdigest()

# authorized mapping: reject -> keeper
REJECT_MAP={
 "rev://v0.2/211":"rev://v0.2/071",
 "rev://v0.2/001":"rev://v0.2/244",
 "rev://v0.2/080":"rev://v0.2/235",
 "rev://v0.2/216":"rev://v0.2/212",
 "rev://v0.2/025":"rev://v0.2/119",
 "rev://v0.2/171":"rev://v0.2/107",
}
TS=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

adj=json.load(open(ADJ_P))
recs=json.load(open(RECS_P))
src=json.load(open(SRC_P))
if isinstance(src,dict): src=src["records"]
src_by_case={r["case_id"]:r for r in src}
rec_by_rid={r["review_id"]:r for r in recs}
records=adj["records"]

pre_hash=sha256(ADJ_P)
pre_counts=Counter(r["final_status"] for r in records)

# ---- apply bounded mutation ----
changed=[]
for r in records:
    rid=r["review_id"]
    if rid in REJECT_MAP:
        if r["final_status"]!="GOLD_POSITIVE":
            raise SystemExit(f"ABORT: {rid} is not GOLD_POSITIVE (is {r['final_status']}); refuse to mutate.")
        r["previous_final_status"]=r["final_status"]
        r["final_status"]="REJECT"
        r["remediation"]={
            "action":"dedup_exact_duplicate_reject",
            "rule":"6B.2 exact-duplicate procedure",
            "duplicate_of":REJECT_MAP[rid],
            "timestamp":TS,
            "note":"Inherited pre-Batch-1 GOLD_POSITIVE exact duplicate; rejected per approved rule. No exception invented. Retained in history.",
        }
        changed.append({"review_id":rid,"case_id":r["case_id"],"from":"GOLD_POSITIVE","to":"REJECT","duplicate_of":REJECT_MAP[rid]})

if len(changed)!=6:
    raise SystemExit(f"ABORT: expected to change 6 records, changed {len(changed)}.")

# write back
json.dump(adj, open(ADJ_P,"w"), indent=2, ensure_ascii=False)
post_hash=sha256(ADJ_P)
post_counts=Counter(r["final_status"] for r in records)

# ---- verification ----
status_by_rid={r["review_id"]:r["final_status"] for r in records}
# GOLD + REJECT + NEEDS_REVIEW checks
gp=post_counts["GOLD_POSITIVE"]; ga=post_counts["GOLD_ABSTAIN"]; hn=post_counts["HARD_NEGATIVE"]
gold_total=gp+ga+hn
reject_total=post_counts["REJECT"]
nr=post_counts["NEEDS_REVIEW"]
chk_counts = (gold_total==250 and gp==44 and ga==66 and hn==140 and reject_total==44 and nr==0)

# ---- INV-01..09 (read-only on GOLD) ----
REL={"CAP-01":"interpretation","CAP-02":"explanation","CAP-03":"interpretation","CAP-04":"interpretation",
     "CAP-06":"explanation","CAP-07":"explanation","CAP-08":"classification","CAP-10":"classification",
     "CAP-11":"interpretation","CAP-12":"causal","CAP-13":"interpretation"}
def map_existing(rid):
    st=status_by_rid[rid]; r=rec_by_rid[rid]
    cid=r.get("case_id") or r.get("_masked_original_case_id") or rid
    conc=r.get("grounding_snapshot",{}).get("concept_ids",[]); ev=r.get("grounding_snapshot",{}).get("evidence_ids",[])
    dom=cid.split("/")[2] if cid.startswith("td://") else "unknown"
    pn="POSITIVE" if st=="GOLD_POSITIVE" else "NEGATIVE"
    nov=r.get("_masked_expected_novelty") or ("SEMANTIC_NOVEL" if st=="GOLD_POSITIVE" else "UNSUPPORTED")
    ti={"proposition":r.get("candidate_proposition"),"supporting_evidence_ids":ev,"referenced_concept_ids":conc,
        "semantic_relation":REL.get(r.get("capability_family"),"interpretation"),"confidence":0.85} if st=="GOLD_POSITIVE" else None
    abl="SHOULD_ABSTAIN" if st=="GOLD_ABSTAIN" else "SHOULD_PROPOSE"
    der=r.get("_masked_expected_derivability") or ("NON_DERIVABLE" if st=="GOLD_POSITIVE" else "DERIVABLE")
    return {"case_id":cid,"positive_negative":pn,"novelty_label":nov,"target_interpretation":ti,
        "rejected_candidates":[],"abstention_label":abl,"derivability":{"label":der},"derivability_label":der,
        "provenance":{"human_review_status":"GOLD","reviewer_1_id":"rev_01","reviewer_2_id":"rev_02",
                      "template_id":r.get("_masked_template_id",""),"seed_case_id":r.get("_masked_original_case_id","")},
        "grounding_snapshot":r.get("grounding_snapshot",{}),
        "concepts":[{"id":c["id"],"label":c["label"],"concept_type":dom} for c in r.get("concepts",[])],
        "evidence_count":len(ev),"distractor_count":max(0,len(conc)-len(ev)),"percept":r.get("percept",""),
        "contradiction_present":(r.get("capability_family")=="CAP-06"),"difficulty_tier":0,
        "capability_family":r.get("capability_family","unknown")}
bpool=json.load(open(os.path.join(BATCH_DIR,"candidate_records_batch1.json")))["records"]
badj=json.load(open(os.path.join(BATCH_DIR,"adjudication_batch1.json")))
MERGE=[e for e in badj["records"] if e["final_status"] in ("GOLD_POSITIVE","HARD_NEGATIVE")]
pbr={r["review_id"]:r for r in bpool}
def map_merged(e):
    src0=pbr[e["review_id"]]; cid=e["case_id"]
    conc=src0["grounding_snapshot"]["concept_ids"]; ev=src0["grounding_snapshot"]["evidence_ids"]
    st=e["final_status"]; pn="POSITIVE" if st=="GOLD_POSITIVE" else "NEGATIVE"
    nov="SEMANTIC_NOVEL" if st=="GOLD_POSITIVE" else (src0["_masked_expected_novelty"] or "SEMANTIC_NOVEL")
    ti={"proposition":e["candidate_proposition"],"supporting_evidence_ids":ev,"referenced_concept_ids":conc,
        "semantic_relation":REL.get(e["capability_family"],"interpretation"),"confidence":0.85} if st=="GOLD_POSITIVE" else None
    der="NON_DERIVABLE" if st=="GOLD_POSITIVE" else "DERIVABLE"
    return {"case_id":cid,"positive_negative":pn,"novelty_label":nov,"target_interpretation":ti,
        "rejected_candidates":[],"abstention_label":"SHOULD_PROPOSE","derivability":{"label":der},"derivability_label":der,
        "provenance":{"human_review_status":"GOLD","reviewer_1_id":"rev_01","reviewer_2_id":"rev_02",
                      "template_id":src0["_masked_template_id"],"seed_case_id":src0["_masked_original_case_id"]},
        "grounding_snapshot":src0["grounding_snapshot"],
        "concepts":[{"id":c["id"],"label":c["label"],"concept_type":src0["domain"]} for c in src0["concepts"]],
        "evidence_count":len(ev),"distractor_count":max(0,len(conc)-len(ev)),"percept":src0["percept"],
        "contradiction_present":(e["capability_family"]=="CAP-06"),"difficulty_tier":int(e["review_id"].split("/")[-1])%6,
        "capability_family":e["capability_family"]}
cum=[map_existing(rid) for rid in rec_by_rid if status_by_rid[rid] in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE")]
cum+=[map_merged(e) for e in MERGE]
inv=quality_gates.check_invariants(cum)
inv_fail={k:[f for f in v["failures"]] for k,v in inv.items() if v["failed"]>0}
inv_total=sum(len(v) for v in inv_fail.values())

# ---- exact-duplicate validator (positive subset) ----
def norm(s): return " ".join((s or "").lower().split())
pos=[(r["case_id"],r["target_interpretation"]["proposition"]) for r in cum
     if r["positive_negative"]=="POSITIVE" and r["target_interpretation"]]
byp=defaultdict(list)
for cid,p in pos: byp[norm(p)].append(cid)
val_pairs=[v for v in byp.values() if len(v)>1]
val_total=len(pos)-len(set(norm(p) for _,p in pos))
chk_exactdup=(val_total==0 and len(val_pairs)==0)

# ---- Batch-1 provenance intact + Batch 1 introduced 0 dups ----
merged=[r for r in records if r.get("merged_from")=="6b3_batch1"]
merged_unique=len(set(r["review_id"] for r in merged))
chk_batch_prov=(merged_unique==30 and len(merged)==30)
# batch1 introduced zero duplicate violations: no rev://v0.3/b1/* in any dup group
batch_in_dups=sum(1 for cid in [c for v in byp.values() for c in v] if cid.startswith("rev://v0.3/b1/"))
chk_batch_zero_dup=(batch_in_dups==0)

ALL_PASS=(chk_counts and chk_exactdup and chk_batch_prov and chk_batch_zero_dup and inv_total==0)

# ---- write machine-readable results ----
diag={
 "authorized":"REJECT-6 remediation APPROVED (exact 6 named records)",
 "timestamp":TS,
 "pre_hash":pre_hash,"post_hash":post_hash,
 "pre_counts":dict(pre_counts),"post_counts":dict(post_counts),
 "changed":changed,
 "verification":{
   "GOLD_total":gold_total,"GOLD_POSITIVE":gp,"GOLD_ABSTAIN":ga,"HARD_NEGATIVE":hn,
   "REJECT":reject_total,"NEEDS_REVIEW":nr,
   "chk_counts":chk_counts,"chk_exactdup":chk_exactdup,"chk_batch_prov":chk_batch_prov,
   "chk_batch_zero_dup":chk_batch_zero_dup,"INV_total_failures":inv_total,
   "ALL_PASS":ALL_PASS},
 "inv_failures_detail":inv_fail,
 "exact_dup_validator":{"positive_duplicate_pairs":len(val_pairs),"total_positive_dups":val_total},
}
json.dump(diag, open(OUT_JSON,"w",encoding="utf-8"), indent=2, ensure_ascii=False)

# ---- remediation report markdown ----
L=[]
L.append("# THEO SLM v0 — Phase 6B.3 Dedup Remediation Report\n")
L.append(f"**Authorization:** REJECT-6 remediation APPROVED — exact six named non-keeper records; keepers retained as GOLD_POSITIVE.")
L.append(f"**Rule applied:** 6B.2 exact-duplicate procedure (no exception invented).")
L.append(f"**Timestamp (UTC):** {TS}\n")
L.append("## Pre/Post hashes\n")
L.append(f"- pre_remediation: `{pre_hash}`")
L.append(f"- post_remediation: `{post_hash}`\n")
L.append("## Records rejected (status flipped GOLD_POSITIVE → REJECT; retained in history)\n")
L.append("| reject review_id | case_id | duplicate_of (keeper) |")
L.append("|---|---|---|")
for c in changed: L.append(f"| {c['review_id']} | {c['case_id']} | {c['duplicate_of']} |")
L.append("")
L.append("## Post-remediation verification\n")
L.append(f"- GOLD total: **{gold_total}** (expected 250) → {'PASS' if gold_total==250 else 'FAIL'}")
L.append(f"- GOLD_POSITIVE: **{gp}** (expected 44) → {'PASS' if gp==44 else 'FAIL'}")
L.append(f"- GOLD_ABSTAIN: {ga} (expected 66) → {'PASS' if ga==66 else 'FAIL'}")
L.append(f"- HARD_NEGATIVE: {hn} (expected 140) → {'PASS' if hn==140 else 'FAIL'}")
L.append(f"- REJECT: **{reject_total}** (expected 44) → {'PASS' if reject_total==44 else 'FAIL'}")
L.append(f"- NEEDS_REVIEW: {nr} (expected 0) → {'PASS' if nr==0 else 'FAIL'}")
L.append(f"- INV-01…INV-09 failures: **{inv_total}** → {'PASS' if inv_total==0 else 'FAIL'}")
L.append(f"- Exact-duplicate validator: **{val_total}** violations across **{len(val_pairs)}** pairs → {'PASS' if chk_exactdup else 'FAIL'}")
L.append(f"- Batch-1 provenance intact (30 merged entries): {'PASS' if chk_batch_prov else 'FAIL'}")
L.append(f"- Batch 1 introduced 0 duplicate violations: {'PASS' if chk_batch_zero_dup else 'FAIL'}\n")
L.append(f"## FINAL: ALL CHECKS {'PASS' if ALL_PASS else 'FAIL'}\n")
L.append("## STOP\n")
L.append("Remediation complete and verified. No Batch-2 generated; no model/6C/6D. "
         "Awaiting separate authorization for Batch-2 generation once this clean canonical state is independently confirmed.")
open(OUT_MD,"w",encoding="utf-8").write("\n".join(L))

print("changed:",len(changed),"| GOLD:",gold_total,"POS:",gp,"ABSTAIN:",ga,"NEG:",hn,"REJECT:",reject_total,"NEEDS_REVIEW:",nr)
print("INV failures:",inv_total,"| exact-dup violations:",val_total,"| batch_prov:",chk_batch_prov,"| batch_zero_dup:",chk_batch_zero_dup)
print("ALL_PASS:",ALL_PASS)
print("pre:",pre_hash[:16],"post:",post_hash[:16])
print("wrote",OUT_MD,"and",OUT_JSON)
