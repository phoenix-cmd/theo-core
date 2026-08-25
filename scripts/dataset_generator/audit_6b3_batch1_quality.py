#!/usr/bin/env python3
"""Independent 6B.3 Batch-1 Quality Audit (READ-ONLY). No files modified."""
import json, re, collections, os

REVIEW_DIR = "theo-data/datasets/theo_slm_v0_review"
BATCH = os.path.join(REVIEW_DIR, "6b3_batch1")
adj = json.load(open(os.path.join(REVIEW_DIR, "adjudication.json")))
recs = json.load(open(os.path.join(REVIEW_DIR, "review-records.json")))
batch = json.load(open(os.path.join(BATCH, "candidate_records_batch1.json")))["records"]

status_by_rid = {r["review_id"]: r.get("final_status") for r in adj["records"]}

# existing GOLD + REJECT propositions (normalized) and GOLD percepts
gold_props, rej_props, gold_percepts = [], [], []
for r in recs:
    st = status_by_rid.get(r.get("review_id"))
    p = (r.get("candidate_proposition") or "").strip().lower()
    if st == "REJECT": rej_props.append(p)
    elif st in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE"):
        gold_props.append(p); gold_percepts.append((r.get("percept") or "").lower())

def norm(s): return re.sub(r"\s+"," ",(s or "").lower()).strip()
def toks(s): return set(re.findall(r"\w+", (s or "").lower()))
def jac(a,b):
    ta,tb=toks(a),toks(b)
    if not ta or not tb: return 0.0
    return len(ta&tb)/len(ta|tb)

N=len(batch)
print("=== B1: FIELD COMPLETENESS ===")
req_top=["review_id","case_id","percept","task","concepts","candidate_proposition",
          "grounding_snapshot","capability_family","domain","group_key","split_group",
          "human_review_status","_masked_original_case_id","_masked_generator_id",
          "_masked_template_id","_masked_expected_novelty","_masked_expected_derivability",
          "_masked_expected_abstention","target_disposition","provenance"]
req_prov=["generator_id","generator_version","dataset_version","random_seed",
          "template_id","seed_case_id","generation_timestamp","deterministic","spec_ref"]
missing=[]
for r in batch:
    for f in req_top:
        if f not in r or r[f] in ("",None,[]): missing.append((r["review_id"],f))
    for f in req_prov:
        if f not in r.get("provenance",{}) or r["provenance"][f] in ("",None): missing.append((r["review_id"],"prov."+f))
print("missing fields:", missing if missing else "NONE")

print("\n=== B2: NON_DERIVABLE for positive targets ===")
bad_der=[r["review_id"] for r in batch if r["target_disposition"]=="GOLD_POSITIVE"
         and r["_masked_expected_derivability"]!="NON_DERIVABLE"]
print("positives with wrong derivability:", bad_der if bad_der else "NONE")
# conceptual derivability: proposition verbatim in percept?
verbatim=[r["review_id"] for r in batch if r["target_disposition"]=="GOLD_POSITIVE"
          and norm(r["candidate_proposition"]) in norm(r["percept"])]
print("positives whose proposition is verbatim in percept (DERIVABLE violation):", verbatim if verbatim else "NONE")

print("\n=== B3: SEMANTIC NOVELTY vs existing 264 ===")
exact=0; paraphrase_risk=[]
gold_norm=set(norm(p) for p in gold_props); rej_norm=set(norm(p) for p in rej_props)
for r in batch:
    pn=norm(r["candidate_proposition"])
    if pn in gold_norm or pn in rej_norm: exact+=1
    # token jaccard vs every existing proposition
    best=max((jac(r["candidate_proposition"],p) for p in gold_props+rej_props), default=0)
    if best>=0.8:
        paraphrase_risk.append((r["review_id"],round(best,3)))
print("exact dup vs 264:", exact)
print("proposition token-Jaccard>=0.8 vs any existing (paraphrase risk):", paraphrase_risk if paraphrase_risk else "NONE")

print("\n=== B4: NEAR-DUP vs EXISTING GOLD percepts (cross-pool leakage risk) ===")
xrisk=[]
for r in batch:
    best=max((jac(r["percept"],gp) for gp in gold_percepts), default=0)
    if best>=0.8:
        xrisk.append((r["review_id"],round(best,3)))
print("new percept Jaccard>=0.8 vs existing GOLD percept:", xrisk if xrisk else "NONE")

print("\n=== B5: INTRA-BATCH near-dup (percept & proposition) ===")
pairs_p=[]; pairs_prop=[]
for i in range(N):
    for j in range(i+1,N):
        a,b=batch[i],batch[j]
        jp=jac(a["percept"],b["percept"])
        jq=jac(a["candidate_proposition"],b["candidate_proposition"])
        if jp>=0.8: pairs_p.append((a["review_id"],b["review_id"],round(jp,3),a["group_key"]==b["group_key"]))
        if jq>=0.8: pairs_prop.append((a["review_id"],b["review_id"],round(jq,3),a["group_key"]==b["group_key"]))
print("percept pairs>=0.8:", pairs_p if pairs_p else "NONE")
print("proposition pairs>=0.8:", pairs_prop if pairs_prop else "NONE")

print("\n=== B6: group_key integrity ===")
gk=collections.Counter(r["group_key"] for r in batch)
multi={k:v for k,v in gk.items() if v>1}
print("group_keys used >1:", multi if multi else "NONE (only intended pair)")
print("split_group == group_key for all:", all(r["split_group"]==r["group_key"] for r in batch))

print("\n=== B7: ID uniqueness & leakage ids ===")
ids_case=[r["case_id"] for r in batch]; ids_rid=[r["review_id"] for r in batch]
print("case_id unique:", len(set(ids_case))==len(ids_case), "review_id unique:", len(set(ids_rid))==len(ids_rid))
leak=[r["review_id"] for r in batch if "bm://" in (r["percept"]+r["candidate_proposition"]).lower() or "sp1://" in (r["percept"]+r["candidate_proposition"]).lower()]
print("eval-instrument id leakage:", leak if leak else "NONE")
# collision with existing 264 review_ids
exist_rid=set(status_by_rid.keys())
print("review_id collision with existing 264:", set(ids_rid)&exist_rid if set(ids_rid)&exist_rid else "NONE")

print("\n=== B8: HARD_NEGATIVE trap-type authenticity ===")
for r in batch:
    if r["target_disposition"]!="HARD_NEGATIVE": continue
    nov=r["_masked_expected_novelty"]; prop=r["candidate_proposition"]; perc=r["percept"]
    flag=""
    if nov=="REPEAT" and norm(prop) not in norm(perc): flag="REPEAT-but-not-in-percept?"
    if nov=="UNSUPPORTED" and r["grounding_snapshot"]["evidence_ids"]:
        # allowed to cite but must be unsupported; just note
        pass
    if nov=="DECISION_IRRELEVANT" and prop.lower() not in perc.lower():
        # off-task check: ensure proposition not a valid interpretation of task
        pass
    print(f"  {r['review_id']} nov={nov} cap={r['capability_family']} -> {prop[:60]}")
print("(manual authenticity review above)")

print("\n=== B9: projected GOLD contribution (NOT adjudicated) ===")
pos=sum(1 for r in batch if r["target_disposition"]=="GOLD_POSITIVE")
neg=sum(1 for r in batch if r["target_disposition"]=="HARD_NEGATIVE")
print(f"if all positives adjudicate GOLD_POSITIVE: 29+{pos}={29+pos} (target 140-180)")
print(f"if all accepted into GOLD: 226+{pos+neg}={226+pos+neg} (target 350-500)")
print("CLEARLY PROJECTION ONLY - no adjudication performed.")
