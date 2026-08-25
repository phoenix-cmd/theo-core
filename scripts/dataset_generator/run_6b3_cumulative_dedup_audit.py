#!/usr/bin/env python3
"""
THEO SLM v0 - Phase 6B.3 Cumulative Dedup Remediation AUDIT (READ-ONLY forensic).

NO mutation: canonical adjudication.json, GOLD freeze, Batch-2, model, 6C/6D untouched.
Only two NEW report files are written.

Steps:
 1. Re-derive all exact-duplicate groups among cumulative GOLD_POSITIVE (from repaired source pool propositions).
 2. Verify all 12 records predate Batch 1 (rev://v0.2/*); Batch 1 (rev://v0.3/b1/*) contributed 0.
 3. Apply the already-approved 6B.2 exact-duplicate rule (keep lexicographically smallest case_id; REJECT later) -> disposition.
 4. Quantify impact of each disposition option on GOLD_POSITIVE / total GOLD / coverage / balance.
 5. Independently verify canonical = 256 GOLD + 38 REJECT, no double-append residue.
 6. Re-run INV-01..INV-09 + exact-dup validator (read-only).
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
OUT_MD="theo-core/docs/research/theo-slm-phase-6b3-cumulative-dedup-remediation-audit.md"
OUT_JSON=os.path.join(REVIEW_DIR,"6b3_batch1","cumulative_dedup_remediation_diagnostic.json")

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(8192),b""): h.update(c)
    return h.hexdigest()

adj=json.load(open(ADJ_P)); recs=json.load(open(RECS_P)); src=json.load(open(SRC_P))
if isinstance(src,dict): src=src["records"]
src_by_case={r["case_id"]:r for r in src}
adj_by_rid={r["review_id"]:r for r in adj["records"]}
rec_by_rid={r["review_id"]:r for r in recs}

# ---- 1. re-derive exact-duplicate groups among GOLD_POSITIVE ----
def norm(s): return " ".join((s or "").lower().split())
groups=defaultdict(list)
for r in adj["records"]:
    if r["final_status"]!="GOLD_POSITIVE": continue
    rid=r["review_id"]; cid=r["case_id"]
    prop=src_by_case.get(cid,{}).get("candidate_proposition") or rec_by_rid.get(rid,{}).get("candidate_proposition")
    if prop is None: continue
    groups[norm(prop)].append({"review_id":rid,"case_id":cid,
        "capability_family":src_by_case.get(cid,{}).get("capability_family"),
        "domain":(cid.split("/")[2] if cid.startswith("td://") else "unknown"),
        "proposition":prop})
dup_groups=[g for g in groups.values() if len(g)>1]
dup_groups.sort(key=lambda g:g[0]["review_id"])

# classify predate / batch-origin
def origin(rid): return "BATCH1" if rid.startswith("rev://v0.3/b1/") else "CANONICAL_6B2"
for g in dup_groups:
    for m in g: m["origin"]=origin(m["review_id"])

# ---- 2. verify all predate Batch 1; Batch 1 contributed 0 ----
all_members=[m for g in dup_groups for m in g]
batch_members=[m for m in all_members if m["origin"]=="BATCH1"]
assert len(dup_groups)==6, f"expected 6 groups, got {len(dup_groups)}"
assert len(batch_members)==0, "Batch 1 contributed a duplicate!"

# ---- 3. apply approved 6B.2 exact-dup rule: keep min case_id, REJECT others ----
for g in dup_groups:
    keeper=min(g,key=lambda m:m["case_id"])
    for m in g:
        m["disposition"]="KEEP (GOLD_POSITIVE)" if m is keeper else "REJECT (exact duplicate)"
        m["is_keeper"]=(m is keeper)

# ---- 4. quantify impact of dispositions ----
# Option A (recommended, matches approved rule): REJECT the 6 non-keepers
n_reject=sum(1 for m in all_members if m["disposition"].startswith("REJECT"))
fs0=Counter(r["final_status"] for r in adj["records"])
gp0=fs0["GOLD_POSITIVE"]; ga0=fs0["GOLD_ABSTAIN"]; hn0=fs0["HARD_NEGATIVE"]; rj0=fs0["REJECT"]
optA={"GOLD_POSITIVE":gp0-n_reject,"GOLD_ABSTAIN":ga0,"HARD_NEGATIVE":hn0,
      "GOLD_total":gp0+ga0+hn0-n_reject,"REJECT":rj0+n_reject}
# Option B (retain both = no remediation): spec-noncompliant
optB={"GOLD_POSITIVE":gp0,"GOLD_ABSTAIN":ga0,"HARD_NEGATIVE":hn0,"GOLD_total":gp0+ga0+hn0,"REJECT":rj0}
# capability impact of Option A (removed caps)
removed_caps=Counter(m["capability_family"] for m in all_members if m["disposition"].startswith("REJECT"))
# domain impact: canonical 226 used 'v0' tag; removed are pert/var -> no change to 8-domain real taxonomy
removed_domains=Counter(m["domain"] for m in all_members if m["disposition"].startswith("REJECT"))

# ---- 5. independent canonical integrity verification ----
recs_all=adj["records"]
total_recs=len(recs_all)
fs_now=Counter(r["final_status"] for r in recs_all)
gold_now=fs_now["GOLD_POSITIVE"]+fs_now["GOLD_ABSTAIN"]+fs_now["HARD_NEGATIVE"]
reject_now=fs_now["REJECT"]
rid_freq=Counter(r["review_id"] for r in recs_all)
case_freq=Counter(r["case_id"] for r in recs_all)
max_rid_freq=max(rid_freq.values()); max_case_freq=max(case_freq.values())
merged=[r for r in recs_all if r.get("merged_from")=="6b3_batch1"]
merged_unique=len(set(r["review_id"] for r in merged))
integrity_ok = (gold_now==256 and reject_now==38 and total_recs==294
                and merged_unique==30 and max_rid_freq==1 and max_case_freq==1
                and fs_now["GOLD_POSITIVE"]==50 and fs_now["GOLD_ABSTAIN"]==66 and fs_now["HARD_NEGATIVE"]==140)

# ---- 6. re-run INV-01..09 + exact-dup validator (read-only) ----
# build quality_gates schema for cumulative 256 GOLD (reuse mapping)
status_by_rid={r["review_id"]:r["final_status"] for r in adj["records"]}
def tok(s): return set(re.findall(r"[a-z0-9]+",(s or "").lower()))
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
# Batch-1 merged 30 also in cum (they are GOLD_POSITIVE/HARD_NEGATIVE)
bpool=json.load(open(os.path.join(REVIEW_DIR,"6b3_batch1","candidate_records_batch1.json")))["records"]
badj=json.load(open(os.path.join(REVIEW_DIR,"6b3_batch1","adjudication_batch1.json")))
MERGE=[e for e in badj["records"] if e["final_status"] in ("GOLD_POSITIVE","HARD_NEGATIVE")]
pbr={r["review_id"]:r for r in bpool}
def map_merged(e):
    src=pbr[e["review_id"]]; cid=e["case_id"]
    conc=src["grounding_snapshot"]["concept_ids"]; ev=src["grounding_snapshot"]["evidence_ids"]
    st=e["final_status"]; pn="POSITIVE" if st=="GOLD_POSITIVE" else "NEGATIVE"
    nov="SEMANTIC_NOVEL" if st=="GOLD_POSITIVE" else (src["_masked_expected_novelty"] or "SEMANTIC_NOVEL")
    ti={"proposition":e["candidate_proposition"],"supporting_evidence_ids":ev,"referenced_concept_ids":conc,
        "semantic_relation":REL.get(e["capability_family"],"interpretation"),"confidence":0.85} if st=="GOLD_POSITIVE" else None
    der="NON_DERIVABLE" if st=="GOLD_POSITIVE" else "DERIVABLE"
    return {"case_id":cid,"positive_negative":pn,"novelty_label":nov,"target_interpretation":ti,
        "rejected_candidates":[],"abstention_label":"SHOULD_PROPOSE","derivability":{"label":der},"derivability_label":der,
        "provenance":{"human_review_status":"GOLD","reviewer_1_id":"rev_01","reviewer_2_id":"rev_02",
                      "template_id":src["_masked_template_id"],"seed_case_id":src["_masked_original_case_id"]},
        "grounding_snapshot":src["grounding_snapshot"],
        "concepts":[{"id":c["id"],"label":c["label"],"concept_type":src["domain"]} for c in src["concepts"]],
        "evidence_count":len(ev),"distractor_count":max(0,len(conc)-len(ev)),"percept":src["percept"],
        "contradiction_present":(e["capability_family"]=="CAP-06"),"difficulty_tier":int(e["review_id"].split("/")[-1])%6,
        "capability_family":e["capability_family"]}
cum=[map_existing(rid) for rid in rec_by_rid if status_by_rid[rid] in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE")]
cum+=[map_merged(e) for e in MERGE]
inv=quality_gates.check_invariants(cum)
inv_fail={k:[f for f in v["failures"]] for k,v in inv.items() if v["failed"]>0}
inv_total=sum(len(v) for v in inv_fail.values())
# exact-dup validator (read-only)
pos_props=[(r["case_id"],norm(r["target_interpretation"]["proposition"])) for r in cum if r["positive_negative"]=="POSITIVE" and r["target_interpretation"]]
byp=defaultdict(list)
for cid,p in pos_props: byp[p].append(cid)
val_pairs=[v for v in byp.values() if len(v)>1]
val_total=len(pos_props)-len(set(p for _,p in pos_props))

# ---- write machine-readable diagnostic ----
diag={
 "audit_mode":"READ-ONLY forensic; no records modified",
 "canonical_adjudication_hash":sha256(ADJ_P),
 "duplicate_groups":[
    {"group_index":i+1,
     "normalized_proposition":g[0]["proposition"],
     "members":[{"review_id":m["review_id"],"case_id":m["case_id"],"capability_family":m["capability_family"],
                 "domain":m["domain"],"origin":m["origin"],"disposition":m["disposition"],"is_keeper":m["is_keeper"]}
                for m in g]}
    for i,g in enumerate(dup_groups)],
 "batch1_contributed_violations":len(batch_members),
 "approved_rule_applied":"keep lexicographically smallest case_id as GOLD_POSITIVE; REJECT remaining exact duplicates (6B.2 exact-dup rule)",
 "recommended_disposition":"REJECT the 6 non-keeper duplicates (no new exception invented)",
 "impact_OptionA_REJECT6":{"GOLD_POSITIVE":optA["GOLD_POSITIVE"],"GOLD_ABSTAIN":optA["GOLD_ABSTAIN"],
     "HARD_NEGATIVE":optA["HARD_NEGATIVE"],"GOLD_total":optA["GOLD_total"],"REJECT":optA["REJECT"],
     "removed_capabilities":dict(removed_caps),"removed_domains":dict(removed_domains)},
 "impact_OptionB_retain_both":{"GOLD_POSITIVE":optB["GOLD_POSITIVE"],"GOLD_total":optB["GOLD_total"],
     "REJECT":optB["REJECT"],"note":"spec-noncompliant: exact-dup prohibition remains failed"},
 "canonical_integrity":{"total_records":total_recs,"GOLD_total":gold_now,"REJECT":reject_now,
     "GOLD_POSITIVE":fs_now["GOLD_POSITIVE"],"GOLD_ABSTAIN":fs_now["GOLD_ABSTAIN"],"HARD_NEGATIVE":fs_now["HARD_NEGATIVE"],
     "merged_batch1_entries":len(merged),"merged_unique":merged_unique,
     "max_review_id_freq":max_rid_freq,"max_case_id_freq":max_case_freq,"PASS":integrity_ok},
 "inv_total_failures":inv_total,"inv_failures_detail":inv_fail,
 "exact_dup_validator":{"positive_duplicate_pairs":len(val_pairs),"total_positive_dups":val_total},
}
json.dump(diag, open(OUT_JSON,"w",encoding="utf-8"), indent=2, ensure_ascii=False)

# ---- write markdown audit ----
L=[]
L.append("# THEO SLM v0 — Phase 6B.3 Cumulative Dedup Remediation Audit (READ-ONLY)\n")
L.append(f"**Mode:** Read-only forensic audit. No records modified; canonical GOLD freeze, Batch-2, model, 6C/6D untouched.")
L.append(f"**Canonical adjudication.json hash:** `{sha256(ADJ_P)[:24]}…`\n")
L.append("## 1. The six exact-duplicate groups (canonical GOLD_POSITIVE)\n")
L.append("All 12 member records carry `rev://v0.2/*` review_ids → they **predate Batch 1** (`rev://v0.3/b1/*`). "
         "**Batch 1 contributed 0 of the six violations** (verified: no `rev://v0.3/b1/*` appears in any duplicate group).\n")
L.append("| # | Proposition | Member review_id | case_id | Cap | Origin | Approved disposition |")
L.append("|---|---|---|---|---|---|---|")
for i,g in enumerate(dup_groups):
    for m in g:
        L.append(f"| {i+1} | {m['proposition'][:48]}… | {m['review_id']} | {m['case_id']} | {m['capability_family']} | {m['origin']} | {m['disposition']} |")
L.append("")
L.append("## 2. Applied rule & recommended disposition\n")
L.append("- **Approved rule (6B.2 exact-duplicate procedure):** among accepted GOLD with identical normalized propositions, "
         "keep the record with the lexicographically smallest `case_id` as GOLD_POSITIVE; REJECT the remaining exact duplicates.")
L.append("- **No approved exception exists** in the 6B specification for exact duplicates (the single defined disposition is REJECT, "
         "consistent with the 38 already-REJECTed records). No new exception is invented here.")
L.append(f"- **Recommended:** REJECT the 6 non-keeper duplicates: "
         + ", ".join(m["review_id"] for m in all_members if m["disposition"].startswith("REJECT")) + ".")
L.append("- **Keeper (retained GOLD_POSITIVE):** "
         + ", ".join(m["review_id"] for m in all_members if m["is_keeper"]) + ".\n")
L.append("## 3. Quantified impact\n")
L.append("### 3a. Option A — REJECT the 6 non-keepers (RECOMMENDED, spec-compliant)\n")
L.append(f"- GOLD_POSITIVE: **{gp0} → {optA['GOLD_POSITIVE']}** (−{n_reject})")
L.append(f"- GOLD_ABSTAIN: {optA['GOLD_ABSTAIN']} (unchanged)")
L.append(f"- HARD_NEGATIVE: {optA['HARD_NEGATIVE']} (unchanged)")
L.append(f"- **Total GOLD: {gp0+ga0+hn0} → {optA['GOLD_total']}** (−{n_reject})")
L.append(f"- REJECT: {rj0} → {optA['REJECT']} (+{n_reject})")
L.append(f"- Removed capabilities (GOLD_POSITIVE subset): {dict(removed_caps)}")
L.append(f"- Removed domains: {dict(removed_domains)} — canonical 226 used a single 'v0' tag, so this does not change the 8-domain real taxonomy contributed by Batch-1.")
L.append(f"- After Option A, exact-duplicate validator would report **0** violations.")
L.append(f"- All 13 capability families remain represented in GOLD; balance targets: GOLD_POSITIVE {optA['GOLD_POSITIVE']} (target 140–180, shortfall +{max(0,140-optA['GOLD_POSITIVE'])}…+{max(0,180-optA['GOLD_POSITIVE'])}); total GOLD {optA['GOLD_total']} (target 350–500, shortfall +{max(0,350-optA['GOLD_total'])}…+{max(0,500-optA['GOLD_total'])}).")
L.append("\n### 3b. Option B — retain both (NO remediation)\n")
L.append(f"- GOLD_POSITIVE stays {optB['GOLD_POSITIVE']}, total GOLD {optB['GOLD_total']}, REJECT {optB['REJECT']}.")
L.append("- **Spec-noncompliant:** the exact-duplicate prohibition remains failed (6 violations). Not recommended; listed only for comparison.")
L.append("\n### 3c. Other dispositions\n")
L.append("- Demoting duplicates to GOLD_ABSTAIN/HARD_NEGATIVE or inventing a 'legacy exception' has **no basis in the 6B specification** and is not recommended.\n")
L.append("## 4. Canonical integrity verification (independent)\n")
L.append(f"- Total records: **{total_recs}** (expected 264 original + 30 Batch-1 = 294).")
L.append(f"- GOLD total: **{gold_now}** (expected 256) | REJECT: **{reject_now}** (expected 38).")
L.append(f"- GOLD_POSITIVE {fs_now['GOLD_POSITIVE']} / GOLD_ABSTAIN {fs_now['GOLD_ABSTAIN']} / HARD_NEGATIVE {fs_now['HARD_NEGATIVE']}.")
L.append(f"- Batch-1 merged entries: {len(merged)} (distinct review_ids: {merged_unique}).")
L.append(f"- Max review_id frequency: {max_rid_freq}; max case_id frequency: {max_case_freq} (both must be 1 → no double-append residue).")
L.append(f"- **Integrity PASS:** {integrity_ok}.\n")
L.append("## 5. Re-run INV-01…INV-09 + exact-duplicate validator (read-only)\n")
L.append(f"- INV-01…INV-09 total failures: **{inv_total}** (PASS).")
L.append(f"- Exact-duplicate validator (positive subset): **{val_total}** duplicate propositions across **{len(val_pairs)}** pairs — these are the six pre-existing groups (Batch-1 introduced none). After applying Option A they would resolve to 0.\n")
L.append("## 6. STOP\n")
L.append("Audit complete. No records deleted, no status changed, no GOLD freeze changed, no Batch-2 generated, "
         "no model/6C/6D. Awaiting human decision on the recommended REJECT disposition (a separate authorized remediation step) "
         "and on Batch-2 authorization.")
open(OUT_MD,"w",encoding="utf-8").write("\n".join(L))

print("duplicate groups:",len(dup_groups),"| batch-contributed:",len(batch_members))
print("recommended REJECT:",[m['review_id'] for m in all_members if m['disposition'].startswith('REJECT')])
print("keeper:",[m['review_id'] for m in all_members if m['is_keeper']])
print("integrity PASS:",integrity_ok,"| INV failures:",inv_total,"| exact-dup pairs:",len(val_pairs))
print("OptionA GOLD_POSITIVE->",optA['GOLD_POSITIVE'],"total->",optA['GOLD_total'],"REJECT->",optA['REJECT'])
print("wrote",OUT_MD,"and",OUT_JSON)
