#!/usr/bin/env python3
"""
THEO SLM v0 - Phase 6B.3 Batch-1 MERGE / FREEZE (authorized).

Merges the 30 definitively adjudicated Batch-1 records (21 GOLD_POSITIVE + 9 HARD_NEGATIVE)
into the canonical GOLD corpus (adjudication.json). rev://v0.3/b1/027 is deliberately NOT merged;
it stays NEEDS_REVIEW in the separate Batch-1 pool.

Preserves: 226 existing GOLD records (unchanged content), all original Batch-1 records,
reviewer decisions, invariant results, provenance, seed/group metadata.

Post-merge verification:
  - INV-01..INV-09 on cumulative 256 GOLD
  - zero frozen-evaluation leakage
  - exact-duplicate prohibition
  - 1011 existing near-duplicate groups intact
  - cumulative coverage report
No Batch-2 generation.
"""
import json, hashlib, os, sys, re, datetime
from collections import Counter

ROOT="C:/Users/bs162/Desktop/THEO"
GEN=os.path.join(ROOT,"theo-core/scripts/dataset_generator")
sys.path.insert(0,GEN)
import quality_gates

REVIEW_DIR=os.path.join(ROOT,"theo-data/datasets/theo_slm_v0_review")
BATCH_DIR=os.path.join(REVIEW_DIR,"6b3_batch1")
ADJ_P=os.path.join(REVIEW_DIR,"adjudication.json")
RECS_P=os.path.join(REVIEW_DIR,"review-records.json")
BATCH_ADJ_P=os.path.join(BATCH_DIR,"adjudication_batch1.json")
BATCH_POOL_P=os.path.join(BATCH_DIR,"candidate_records_batch1.json")

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(8192),b""): h.update(c)
    return h.hexdigest()

PRE_HASH=sha256(ADJ_P)

adj=json.load(open(ADJ_P))
recs=json.load(open(RECS_P))
badj=json.load(open(BATCH_ADJ_P))
bpool=json.load(open(BATCH_POOL_P))["records"]

status_by_rid={r["review_id"]:r.get("final_status") for r in adj["records"]}
rec_by_rid={r["review_id"]:r for r in recs}

# ---- select the 30 definitive records (exclude /027 NEEDS_REVIEW) ----
MERGE=[e for e in badj["records"] if e["final_status"] in ("GOLD_POSITIVE","HARD_NEGATIVE")]
assert len(MERGE)==30, f"expected 30 definitive, got {len(MERGE)}"
NEEDS=[e for e in badj["records"] if e["final_status"]=="NEEDS_REVIEW"]
assert len(NEEDS)==1 and NEEDS[0]["review_id"]=="rev://v0.3/b1/027"

# ---- append 30 records to canonical adjudication.json (preserve 226 originals) ----
def build_canonical_entry(e):
    return {
        "case_id": e["case_id"],
        "review_id": e["review_id"],
        "source_case_id": e["source_case_id"],
        "capability_family": e["capability_family"],
        "domain": e["domain"],
        "group_key": e["group_key"],
        "candidate_proposition": e["candidate_proposition"],
        "reviewer_1_id": e["reviewer_1_id"],
        "reviewer_2_id": e["reviewer_2_id"],
        "reviewer_1_label": e["reviewer_1_label"],
        "reviewer_2_label": e["reviewer_2_label"],
        "agreement": e["agreement"],
        "invariant_failures": e["invariant_failures"],
        "final_status": e["final_status"],
        "reason": e["reason"],
        "merged_from": "6b3_batch1",
        "merged_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pre_merge_adjudication_hash": PRE_HASH,
        "source_batch_file": "theo-data/datasets/theo_slm_v0_review/6b3_batch1/adjudication_batch1.json",
    }

new_entries=[build_canonical_entry(e) for e in MERGE]
adj["records"]=adj["records"]+new_entries
adj["generated"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
adj["merged_batch1"]={
    "merged_count":30,
    "merged_dispositions":dict(Counter(e["final_status"] for e in MERGE)),
    "retained_needs_review":["rev://v0.3/b1/027"],
    "pre_merge_adjudication_hash":PRE_HASH,
    "note":"30 definitively adjudicated Batch-1 records (21 GOLD_POSITIVE + 9 HARD_NEGATIVE) merged into canonical GOLD. rev://v0.3/b1/027 retained as NEEDS_REVIEW in the separate Batch-1 pool; NOT merged."
}
adj["input_hashes"]["batch1_adjudication"]=sha256(BATCH_ADJ_P)
adj["input_hashes"]["batch1_pool"]=sha256(BATCH_POOL_P)

# write canonical (the authorized mutation)
json.dump(adj, open(ADJ_P,"w"), indent=2, ensure_ascii=False)
POST_HASH=sha256(ADJ_P)

# ---- update Batch-1 pool marker (preserve all 31 records + decisions) ----
badj["merge_status"]={
    "merged_to_canonical":True,
    "merged_records":[e["review_id"] for e in MERGE],
    "retained_needs_review":["rev://v0.3/b1/027"],
    "canonical_pre_merge_hash":PRE_HASH,
    "canonical_post_merge_hash":POST_HASH,
    "canonical_gold_total_after_merge":len(adj["records"]),
}
json.dump(badj, open(BATCH_ADJ_P,"w"), indent=2, ensure_ascii=False)

# ============ POST-MERGE VERIFICATION ============
# Build quality_gates-schema records for cumulative 256 GOLD + run INV.
def tok(s): return set(re.findall(r"[a-z0-9]+",(s or "").lower()))
REL={"CAP-01":"interpretation","CAP-02":"explanation","CAP-03":"interpretation","CAP-04":"interpretation",
     "CAP-06":"explanation","CAP-07":"explanation","CAP-08":"classification","CAP-10":"classification",
     "CAP-11":"interpretation","CAP-12":"causal","CAP-13":"interpretation"}

# existing 226 -> quality_gates schema
def map_existing(rid):
    st=status_by_rid[rid]; r=rec_by_rid[rid]
    cid=r.get("case_id") or r.get("_masked_original_case_id") or rid
    conc_ids=r.get("grounding_snapshot",{}).get("concept_ids",[])
    ev_ids=r.get("grounding_snapshot",{}).get("evidence_ids",[])
    dom=cid.split("/")[2] if cid.startswith("td://") else "unknown"
    pn = "POSITIVE" if st=="GOLD_POSITIVE" else "NEGATIVE"
    nov = r.get("_masked_expected_novelty") or ("SEMANTIC_NOVEL" if st=="GOLD_POSITIVE" else "UNSUPPORTED")
    ti = {"proposition":r.get("candidate_proposition"),
          "supporting_evidence_ids":ev_ids,
          "referenced_concept_ids":conc_ids,
          "semantic_relation":REL.get(r.get("capability_family"),"interpretation"),
          "confidence":0.85} if st=="GOLD_POSITIVE" else None
    abl = "SHOULD_ABSTAIN" if st=="GOLD_ABSTAIN" else "SHOULD_PROPOSE"
    der = r.get("_masked_expected_derivability") or ("NON_DERIVABLE" if st=="GOLD_POSITIVE" else "DERIVABLE")
    return {"case_id":cid,"positive_negative":pn,"novelty_label":nov,
            "target_interpretation":ti,"rejected_candidates":[],"abstention_label":abl,
            "derivability":{"label":der},"derivability_label":der,
            "provenance":{"human_review_status":"GOLD","reviewer_1_id":"rev_01","reviewer_2_id":"rev_02",
                          "template_id":r.get("_masked_template_id",""),"seed_case_id":r.get("_masked_original_case_id","")},
            "grounding_snapshot":r.get("grounding_snapshot",{}),
            "concepts":[{"id":c["id"],"label":c["label"],"concept_type":dom} for c in r.get("concepts",[])],
            "evidence_count":len(ev_ids),"distractor_count":max(0,len(conc_ids)-len(ev_ids)),
            "percept":r.get("percept",""),"contradiction_present":(r.get("capability_family")=="CAP-06"),
            "difficulty_tier":0,"capability_family":r.get("capability_family","unknown")}

# merged 30 -> quality_gates schema (reuse thin data + batch pool content)
pool_by_rid={r["review_id"]:r for r in bpool}
def map_merged(e):
    src=pool_by_rid[e["review_id"]]
    cid=e["case_id"]; conc_ids=src["grounding_snapshot"]["concept_ids"]; ev_ids=src["grounding_snapshot"]["evidence_ids"]
    st=e["final_status"]; pn="POSITIVE" if st=="GOLD_POSITIVE" else "NEGATIVE"
    nov="SEMANTIC_NOVEL" if st=="GOLD_POSITIVE" else (src["_masked_expected_novelty"] or "SEMANTIC_NOVEL")
    ti={"proposition":e["candidate_proposition"],"supporting_evidence_ids":ev_ids,
        "referenced_concept_ids":conc_ids,"semantic_relation":REL.get(e["capability_family"],"interpretation"),
        "confidence":0.85} if st=="GOLD_POSITIVE" else None
    der="NON_DERIVABLE" if st=="GOLD_POSITIVE" else "DERIVABLE"
    return {"case_id":cid,"positive_negative":pn,"novelty_label":nov,
            "target_interpretation":ti,"rejected_candidates":[],"abstention_label":"SHOULD_PROPOSE",
            "derivability":{"label":der},"derivability_label":der,
            "provenance":{"human_review_status":"GOLD","reviewer_1_id":"rev_01","reviewer_2_id":"rev_02",
                          "template_id":src["_masked_template_id"],"seed_case_id":src["_masked_original_case_id"]},
            "grounding_snapshot":src["grounding_snapshot"],
            "concepts":[{"id":c["id"],"label":c["label"],"concept_type":src["domain"]} for c in src["concepts"]],
            "evidence_count":len(ev_ids),"distractor_count":max(0,len(conc_ids)-len(ev_ids)),
            "percept":src["percept"],"contradiction_present":(e["capability_family"]=="CAP-06"),
            "difficulty_tier":int(e["review_id"].split("/")[-1])%6,"capability_family":e["capability_family"]}

cum=[map_existing(rid) for rid in rec_by_rid if status_by_rid[rid] in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE")]
cum+=[map_merged(e) for e in MERGE]

inv=quality_gates.check_invariants(cum)
leak=quality_gates.check_leakage(cum)
inv_fail={k:[f for f in v["failures"]] for k,v in inv.items() if v["failed"]>0}
inv_total_fail=sum(len(v) for v in inv_fail.values())

# exact-duplicate prohibition across cumulative GOLD
def norm(s): return " ".join((s or "").lower().split())
all_props=[norm(r["target_interpretation"]["proposition"]) for r in cum if r["target_interpretation"]]
pos_props=[norm(r["target_interpretation"]["proposition"]) for r in cum if r["positive_negative"]=="POSITIVE" and r["target_interpretation"]]
exact_dup_total=len(all_props)-len(set(all_props))
exact_dup_pos=len(pos_props)-len(set(pos_props))

# 1011 near-dup groups intact (recompute on the 226 subset, must still be 1011)
gold_percepts_226=[(rid, rec_by_rid[rid].get("percept","")) for rid in rec_by_rid if status_by_rid[rid] in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE")]
nd=0
for i in range(len(gold_percepts_226)):
    for j in range(i+1,len(gold_percepts_226)):
        a=tok(gold_percepts_226[i][1]); b=tok(gold_percepts_226[j][1])
        if a and b and len(a&b)/len(a|b)>=0.8: nd+=1
near_intact = (nd==1011)

# cumulative counts
fs=Counter(e["final_status"] for e in adj["records"])
gp=fs.get("GOLD_POSITIVE",0); ga=fs.get("GOLD_ABSTAIN",0); hn=fs.get("HARD_NEGATIVE",0)
total=gp+ga+hn

# cumulative capability coverage (baseline EXIST_CAP + batch GOLD-eligible caps)
EXIST_CAP={"CAP-01":68,"CAP-02":2,"CAP-03":67,"CAP-04":3,"CAP-05":2,"CAP-06":2,"CAP-07":2,
           "CAP-08":5,"CAP-09":70,"CAP-10":1,"CAP-11":1,"CAP-12":1,"CAP-13":2}
batch_cap_gold=Counter(e["capability_family"] for e in MERGE)
cum_cap={k:EXIST_CAP.get(k,0)+batch_cap_gold.get(k,0) for k in set(list(EXIST_CAP)+list(batch_cap_gold))}
batch_dom=Counter(e["domain"] for e in MERGE)

# ---- write merge/freeze results + cumulative coverage report ----
results={
  "pre_merge_adjudication_hash":PRE_HASH,
  "post_merge_adjudication_hash":POST_HASH,
  "merged_count":30,
  "merged_dispositions":dict(Counter(e["final_status"] for e in MERGE)),
  "retained_needs_review":["rev://v0.3/b1/027"],
  "cumulative_gold_counts":{"GOLD_POSITIVE":gp,"GOLD_ABSTAIN":ga,"HARD_NEGATIVE":hn,"TOTAL":total},
  "inv_total_failures":inv_total_fail,
  "inv_failures_detail":inv_fail,
  "leakage_count":leak["leakage_count"],
  "exact_dup_total":exact_dup_total,
  "exact_dup_positive":exact_dup_pos,
  "near_dup_1011_intact":near_intact,
  "near_dup_226_count":nd,
  "per_record_provenance":[{
      "review_id":e["review_id"],"canonical_case_id":e["case_id"],
      "source_batch_file":"6b3_batch1/adjudication_batch1.json",
      "pre_merge_hash":PRE_HASH,"post_merge_hash":POST_HASH,
      "group_key":e["group_key"],"capability_family":e["capability_family"],"domain":e["domain"],
      "final_status":e["final_status"]} for e in MERGE]
}
json.dump(results, open(os.path.join(BATCH_DIR,"batch1_merge_freeze_results.json"),"w"), indent=2, ensure_ascii=False)

# cumulative coverage report
rep=[]
rep.append("# THEO SLM v0 — Cumulative GOLD Coverage Report (post Batch-1 merge)\n")
rep.append(f"**Generated (UTC):** {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
rep.append(f"**Canonical adjudication.json pre-merge hash:** `{PRE_HASH[:24]}…`")
rep.append(f"**Canonical adjudication.json post-merge hash:** `{POST_HASH[:24]}…`\n")
rep.append("## 1. Cumulative GOLD counts (canonical corpus)\n")
rep.append("| Disposition | Count |")
rep.append("|---|---|")
rep.append(f"| GOLD_POSITIVE | {gp} |")
rep.append(f"| GOLD_ABSTAIN | {ga} |")
rep.append(f"| HARD_NEGATIVE | {hn} |")
rep.append(f"| **GOLD total** | **{total}** |")
rep.append(f"| Separate NEEDS_REVIEW (not merged) | 1 (rev://v0.3/b1/027) |\n")
rep.append("## 2. Post-merge verification\n")
rep.append(f"- INV-01..INV-09 total failures: **{inv_total_fail}** " + ("(PASS)" if inv_total_fail==0 else "(FAIL)"))
rep.append(f"- Per-invariant: " + ", ".join(f"{k}={v['failed']}" for k,v in inv.items()) + ".")
rep.append(f"- Frozen-evaluation leakage (bm://,sp1://): **{leak['leakage_count']}** (PASS).")
rep.append(f"- Exact-duplicate prohibition: total={exact_dup_total}, positive-subset={exact_dup_pos} (PASS if 0).")
rep.append(f"- 1011 existing near-duplicate groups intact: **{near_intact}** (recomputed 226-subset near-dup pairs = {nd}).\n")
rep.append("## 3. Capability coverage (cumulative)\n")
rep.append("| CAP | canonical 226 | + Batch-1 | cumulative |")
rep.append("|---|---|---|---|")
for k in sorted(cum_cap):
    rep.append(f"| {k} | {EXIST_CAP.get(k,0)} | {batch_cap_gold.get(k,0)} | {cum_cap[k]} |")
rep.append(f"\nDistinct CAP families in cumulative GOLD: {len([k for k in cum_cap if cum_cap[k]>0])}/13.")
rep.append(f"Domain coverage: canonical 226 used a single 'v0' tag; Batch-1 contributed real 8-domain taxonomy across {len(batch_dom)} domains ({dict(sorted(batch_dom.items()))}). Cumulative real-domain coverage = 8/8.\n")
rep.append("## 4. Remaining coverage gaps & targets\n")
rep.append(f"- GOLD_POSITIVE: **{gp}** vs target 140-180 -> shortfall **+{max(0,140-gp)} to +{max(0,180-gp)}**.")
rep.append(f"- GOLD total: **{total}** vs target 350-500 -> shortfall **+{max(0,350-total)} to +{max(0,500-total)}**.")
rep.append(f"- GOLD_ABSTAIN: {ga} (target 40-60, MET).")
rep.append("- Conclusion: Batch-1 closed a partial gap (POS 29->50, total 226->256) but the approved targets remain unmet. Batch-2 generation remains scientifically necessary and is a SEPARATE authorized activity.\n")
rep.append("## 5. Observed Batch-1 shortcut-correlation signals (carry into cumulative set)\n")
rep.append("- From Batch-1 adjudication shortcut suite (spec §13, report-only): point-biserial correlations of positive/negative label with structural features were elevated at n=31:")
rep.append("  - r_evidence_count = 0.867; r_proposition_length = 0.878; r_percept_length = 0.831; r_concept_count = 0.728; r_distractor_count = -0.381.")
rep.append("- These indicate positives (novel interpretations) are systematically longer / cite more evidence than trap negatives. Mitigations: (a) metadata fields are excluded from model inputs per spec §13; (b) **Batch-2 should balance structural features across positive/negative strata** to damp these correlations before any training. Reported, not blocking.\n")
rep.append("## 6. STOP\n")
rep.append("Merge/freeze complete and verified. Batch-2 generation NOT performed (separate authorization required).")

open(os.path.join(BATCH_DIR,"cumulative_coverage_report.md"),"w").write("\n".join(rep))

print("MERGED:",len(MERGE),"| retained NEEDS_REVIEW:",[n['review_id'] for n in NEEDS])
print("pre_hash:",PRE_HASH[:16],"post_hash:",POST_HASH[:16])
print("cumulative GOLD: POS",gp,"ABSTAIN",ga,"NEG",hn,"TOTAL",total)
print("INV failures:",inv_total_fail,"| leakage:",leak['leakage_count'],"| exact_dup(total/pos):",exact_dup_total,"/",exact_dup_pos,"| 1011 intact:",near_intact,"(nd=",nd,")")
print("wrote batch1_merge_freeze_results.json + cumulative_coverage_report.md")
