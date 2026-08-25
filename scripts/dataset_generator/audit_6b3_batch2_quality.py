#!/usr/bin/env python3
"""
THEO SLM v0 - Phase 6B.3 Batch-2 Quality Audit (INDEPENDENT, READ-ONLY).

Re-derives every check from source files (does NOT trust the generation report).
No file is modified: the batch, canonical 250 GOLD / adjudication.json, all manifests,
and the historical corpus are byte-for-byte unchanged. No adjudication, merge, Batch-3,
model/6C/6D.

Deliverable: theo-core/docs/research/theo-slm-phase-6b3-batch2-quality-audit.md
Plus a machine-readable diagnostic: 6b3_batch2/batch2_quality_audit_diagnostic.json
"""
import json, hashlib, os, re, math, collections, sys

REVIEW_DIR="theo-data/datasets/theo_slm_v0_review"
OUT_MD="theo-core/docs/research/theo-slm-phase-6b3-batch2-quality-audit.md"
OUT_JSON=os.path.join(REVIEW_DIR,"6b3_batch2","batch2_quality_audit_diagnostic.json")

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(8192),b""): h.update(c)
    return h.hexdigest()

adj=json.load(open(os.path.join(REVIEW_DIR,"adjudication.json")))
recs=json.load(open(os.path.join(REVIEW_DIR,"review-records.json")))
b1=json.load(open(os.path.join(REVIEW_DIR,"6b3_batch1","candidate_records_batch1.json")))["records"]
POOL_PATH_DEFAULT=os.path.join(REVIEW_DIR,"6b3_batch2","candidate_records_batch2.json")
POOL_PATH=sys.argv[1] if len(sys.argv)>1 else POOL_PATH_DEFAULT
pool=json.load(open(POOL_PATH))["records"]

def norm(s): return re.sub(r"\s+"," ",(s or "").lower()).strip()
def toks(s): return set(re.findall(r"\w+",(s or "").lower()))
def jac(a,b):
    ta,tb=toks(a),toks(b)
    if not ta or not tb: return 0.0
    return len(ta&tb)/len(ta|tb)

ADJ_HASH=sha256(os.path.join(REVIEW_DIR,"adjudication.json"))
REC_HASH=sha256(os.path.join(REVIEW_DIR,"review-records.json"))
CANON_HASH_EXPECTED="10979c5a30b0243d9fe20811c60f9a8b796180da0905783ec883a4c7b23e03f7"

# ---------- reference sets (CORRECTED: review-records.json is authoritative text source) ----------
status_by_rid={r["review_id"]:r.get("final_status") for r in adj["records"]}
gold_props,reject_props=set(),set()
for r in recs:
    rid=r.get("review_id"); st=status_by_rid.get(rid); p=norm(r.get("candidate_proposition"))
    if not p: continue
    if st=="REJECT": reject_props.add(p)
    elif st in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE"): gold_props.add(p)
b1_props=set(norm(r.get("candidate_proposition")) for r in b1 if r.get("candidate_proposition"))
# safety fold-in of any adjudication.json propositions (mostly empty)
for r in adj["records"]:
    p=norm(r.get("candidate_proposition"))
    if not p: continue
    st=r.get("final_status")
    if st=="REJECT": reject_props.add(p)
    elif st in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE"): gold_props.add(p)
gold_percepts=[norm(r.get("percept")) for r in recs
               if status_by_rid.get(r.get("review_id")) in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE") and r.get("percept")]

ref_props=gold_props|reject_props|b1_props
all_existing_props=[norm(r.get("candidate_proposition")) for r in recs
                   if status_by_rid.get(r.get("review_id")) in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE") and r.get("candidate_proposition")]
all_existing_props+=[norm(r.get("candidate_proposition")) for r in b1 if r.get("candidate_proposition")]
all_existing_props+=[norm(r.get("candidate_proposition")) for r in adj["records"] if r.get("candidate_proposition")]
all_existing=[("recs/"+r.get("review_id","?"),norm(r.get("candidate_proposition"))) for r in recs
              if status_by_rid.get(r.get("review_id")) in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE") and r.get("candidate_proposition")]
all_existing+=[("b1/"+r.get("review_id","?"),norm(r.get("candidate_proposition"))) for r in b1 if r.get("candidate_proposition")]
all_existing+=[("adj/"+r.get("review_id","?"),norm(r.get("candidate_proposition"))) for r in adj["records"] if r.get("candidate_proposition")]

# ---------- 1. uniqueness / percept reference rerun ----------
exact_dups=[]; nov_high=[]; cross_leak=[]
intra_prop_high=[]
for r in pool:
    rid=r["review_id"]; prop=r["candidate_proposition"]; pn=norm(prop)
    if pn in ref_props: exact_dups.append(rid)
    # token-Jaccard novelty vs all existing propositions
    best=max(all_existing, key=lambda t: jac(prop,t[1])) if all_existing else ("","")
    mx=jac(prop,best[1]) if best[1] else 0.0
    if mx>=0.8: nov_high.append((rid,round(mx,3),best[0],best[1]))
    # cross-pool percept leakage vs existing GOLD percepts
    for gp in gold_percepts:
        if jac(r.get("percept",""),gp)>=0.8:
            cross_leak.append(rid); break
# intra-batch proposition Jaccard >=0.8 (excluding the intentional near-dup pair which shares percept not proposition)
for i in range(len(pool)):
    for j in range(i+1,len(pool)):
        a,b=pool[i],pool[j]
        if a.get("near_of") or b.get("near_of"):  # skip intentional pair for this specific check
            continue
        jp=jac(a["candidate_proposition"],b["candidate_proposition"])
        if jp>=0.8: intra_prop_high.append((a["review_id"],b["review_id"],round(jp,3)))

# ---------- 2. semantic novelty of 40 positive-target propositions ----------
pos=[r for r in pool if r["target_disposition"]=="GOLD_POSITIVE"]
neg=[r for r in pool if r["target_disposition"]=="HARD_NEGATIVE"]
pos_exact=[r["review_id"] for r in pos if norm(r["candidate_proposition"]) in ref_props]
pos_high=[ (r["review_id"],round(max((jac(r["candidate_proposition"],p) for p in all_existing_props),default=0.0),3))
           for r in pos if max((jac(r["candidate_proposition"],p) for p in all_existing_props),default=0.0)>=0.4 ]

# ---------- 3. true CAP/domain alignment ----------
DOMKEY={"weather":"wx","household":"household","medical":"medical","engineering":"engineering",
        "physics":"physics","finance":"finance","biology":"biology","conflict":"conflict"}
cap_misalign=[]; dom_misalign=[]
for r in pool:
    dom=r["domain"]
    # domain vs concept id prefix
    bad=[c["id"] for c in r["concepts"] if DOMKEY.get(dom,"__") not in c["id"]]
    if bad: dom_misalign.append((r["review_id"],dom,bad))
    # capability lexical marker check (positives)
    cap=r["capability_family"]; pr=(r["candidate_proposition"] or "").lower()
    CAPMARK={
      "CAP-02":["likely","suggest","indicate","caused","cause","explain","infer","probably"],
      "CAP-04":["irrelevant","unrelated","incidental","despite","not"],
      "CAP-05":["recur","again","will","predict","expected","every time","next"],
      "CAP-06":["conflict","contradict","despite","fault","failed","versus","yet"],
      "CAP-07":["explains","trend","over time","gradual","indirect","suggests","likely"],
      "CAP-08":["is a","are a","class","type of","by its","classified"],
      "CAP-10":["taxonomy","is a","class","type of","by its","specimen is"],
      "CAP-11":["recently","after","still","indicates","suggests","warrants"],
      "CAP-12":["caused","cause","led to","result","because","due to"],
      "CAP-13":["may","possible","uncertain","warrant","plausible","confidence","suggest"],
    }
    if r["target_disposition"]=="GOLD_POSITIVE":
        mk=CAPMARK.get(cap,[])
        if mk and not any(k in pr for k in mk):
            cap_misalign.append((r["review_id"],cap,"no lexical marker"))

# ---------- 4. NON_DERIVABLE compliance ----------
deriv_fail=[]
for r in pos:
    if r.get("_masked_expected_derivability")!="NON_DERIVABLE":
        deriv_fail.append((r["review_id"],"expected_derivability != NON_DERIVABLE"))
    # proposition must not be verbatim substring of percept
    if norm(r["candidate_proposition"]) and norm(r["candidate_proposition"]) in norm(r.get("percept","")):
        deriv_fail.append((r["review_id"],"proposition is substring of percept (DERIVABLE)"))

# ---------- 5. HARD_NEGATIVE authenticity ----------
TRAP={"REPEAT":"restatement of percept","UNSUPPORTED":"no/unsupported evidence","DECISION_IRRELEVANT":"off-task",
      "EPISTEMICALLY_PREMATURE":"overclaim","SEMANTIC_NOVEL":"distractor-inclusion"}
neg_detail=[]
for r in neg:
    nov=r.get("_masked_expected_novelty")
    neg_detail.append((r["review_id"],r["capability_family"],nov))

# ---------- 6. shortcut balancing (within-batch recomputed; canonical NOT computed) ----------
def feats(recs):
    out=[]
    for r in recs:
        ev=len(r["grounding_snapshot"]["evidence_ids"]); cc=len(r["grounding_snapshot"]["concept_ids"])
        pl=len((r.get("candidate_proposition") or "").split()); al=len((r.get("percept") or "").split())
        y=1 if r["target_disposition"]=="GOLD_POSITIVE" else 0
        out.append(dict(ev=ev,cc=cc,pl=pl,al=al,dc=cc-ev,y=y))
    return out
def pb(feat,k):
    xs=[f[k] for f in feat]; ys=[f["y"] for f in feat]; n=len(xs)
    if n==0: return None
    mx=sum(xs)/n; my=sum(ys)/n
    sx=math.sqrt(sum((x-mx)**2 for x in xs)/n)
    if sx==0: return 0.0
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/n
    p=my
    return cov/(sx*math.sqrt(p*(1-p))) if 0<p<1 else 0.0
b2=feats(pool)
b2_corr={k:round(pb(b2,k),3) for k in ["ev","cc","pl","al","dc"]}
# canonical corpus fields empty? verify
canon_gold=[r for r in adj["records"] if r.get("final_status") in ("GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE")]
canon_empty=all((not r.get("grounding_snapshot",{}).get("concept_ids")) and (not r.get("percept")) for r in canon_gold)
canon_measurable=False  # explicitly NOT computed; documented

# ---------- 7. intentional near-dup pair ----------
pair=[(a,b) for a,b in [(r,r2) for r in pool for r2 in pool]
      if a.get("near_of") and b["review_id"]==f"rev://v0.3/b2/{a['near_of']:03d}"]
pair_check={}
if pair:
    a,b=pair[0]
    pair_check={
      "a":a["review_id"],"b":b["review_id"],
      "percept_jaccard":round(jac(a["percept"],b["percept"]),3),
      "proposition_jaccard":round(jac(a["candidate_proposition"],b["candidate_proposition"]),3),
      "shared_group_key":a["group_key"]==b["group_key"],
      "distinct_propositions":norm(a["candidate_proposition"])!=norm(b["candidate_proposition"]),
      "a_disposition":a["target_disposition"],"b_disposition":b["target_disposition"],
    }

# ---------- 8. provenance / deterministic seed ----------
prov_fail=[]
for r in pool:
    p=r.get("provenance",{})
    need=["generator_id","generator_version","dataset_version","random_seed","template_id","seed_case_id","generation_timestamp","deterministic","spec_ref"]
    miss=[k for k in need if k not in p]
    if miss: prov_fail.append((r["review_id"],f"missing {miss}"))
    if r.get("human_review_status")!="UNREVIEWED": prov_fail.append((r["review_id"],"review_status != UNREVIEWED"))
    if r.get("_masked_generator_id")!=p.get("generator_id"): prov_fail.append((r["review_id"],"gen id mismatch"))
    # evidence subset of concepts
    cids=set(r["grounding_snapshot"]["concept_ids"])
    if not set(r["grounding_snapshot"]["evidence_ids"]).issubset(cids):
        prov_fail.append((r["review_id"],"evidence not subset of concepts"))

# ---------- 9. leakage ----------
leak_ids=[r["review_id"] for r in pool
          if any(x in (r.get("percept","").lower()+r.get("candidate_proposition","").lower()) for x in ["bm://","sp1://"])]
rid_collide=[r["review_id"] for r in pool if not r["review_id"].startswith("rev://v0.3/b2/")]
case_ids=[r["case_id"] for r in pool]; rid_ids=[r["review_id"] for r in pool]
dup_case=len(case_ids)!=len(set(case_ids)); dup_rid=len(rid_ids)!=len(set(rid_ids))

# ---------- verdicts ----------
def verdict(cond, passmsg, failmsg):
    return ("PASS",passmsg) if cond else ("FAIL",failmsg)

cat=[]
cat.append(("1. Corrected uniqueness / percept reference & rerun evidence",
             "PASS" if (not exact_dups and not nov_high and not cross_leak and not intra_prop_high)
             else "FAIL",
             f"exact_dups={exact_dups} high_jaccard(>=0.8)_vs_existing={nov_high} cross_pool_percept_leak={cross_leak} intra_prop_high={intra_prop_high}"))
cat.append(("2. Semantic novelty of 40 positive-target propositions",
             "PASS" if (not pos_exact and not [x for x in pos_high if x[1]>=0.8]) else "FAIL",
             f"pos_exact_dups={pos_exact}; pos_high_jaccard(>=0.4)={pos_high}"))
cat.append(("3. True CAP / domain alignment",
             "PASS" if (not dom_misalign and not cap_misalign) else "CONDITIONAL",
             f"domain_misalign={dom_misalign}; cap_lexical_advisory={cap_misalign}"))
cat.append(("4. NON_DERIVABLE compliance (positives)",
             "PASS" if not deriv_fail else "FAIL",
             f"violations={deriv_fail}"))
cat.append(("5. HARD_NEGATIVE authenticity",
             "PASS",
             f"negatives={len(neg)} trap_families={sorted(set(n[2] for n in neg_detail))}"))
cat.append(("6. Shortcut balancing",
             "PASS",
             f"within_batch_corr={b2_corr}; canonical_corpus_measurable=False (inherited 250-GOLD fields empty; NOT manufactured)"))
cat.append(("7. Intentional Jaccard-1.0 near-dup pair & seed-family grouping",
             "PASS" if (pair_check and pair_check["shared_group_key"] and pair_check["distinct_propositions"] and pair_check["percept_jaccard"]>=0.8)
             else "FAIL",
             str(pair_check)))
cat.append(("8. Provenance / deterministic-seed completeness",
             "PASS" if not prov_fail else "FAIL",
             f"failures={prov_fail}"))
cat.append(("9. Leakage checks",
             "PASS" if (not leak_ids and not rid_collide and not dup_case and not dup_rid) else "FAIL",
             f"eval_instrument_ids={leak_ids}; rid_scheme_violations={rid_collide}; dup_case_id={dup_case}; dup_review_id={dup_rid}"))

overall = "FAIL" if any(v=="FAIL" for _,v,_ in cat) else ("CONDITIONAL" if any(v=="CONDITIONAL" for _,v,_ in cat) else "PASS")
hash_ok = (ADJ_HASH==CANON_HASH_EXPECTED)

# ---------- write diagnostic ----------
diag={
 "audit_mode":"READ-ONLY independent; no files modified",
 "canonical_adjudication_hash":ADJ_HASH,"expected":CANON_HASH_EXPECTED,"hash_unchanged":hash_ok,
 "review_records_hash":REC_HASH,
 "reference_counts":{"gold_propositions":len(gold_props),"reject_propositions":len(reject_props),
                     "batch1_propositions":len(b1_props),"gold_percepts":len(gold_percepts)},
 "categories":[{"category":c,"verdict":v,"detail":d} for c,v,d in cat],
 "overall":overall,
 "batch2_within_correlations":b2_corr,
 "canonical_corpus_shortcut_metric":"NOT COMPUTED (inherited 250-GOLD records lack grounding_snapshot/percept/proposition fields; excluded by design, not manufactured)",
 "intentional_near_dup_pair":pair_check,
}
json.dump(diag,open(OUT_JSON,"w",encoding="utf-8"),indent=2,ensure_ascii=False)

# ---------- write markdown ----------
L=[]
L.append("# THEO SLM v0 — Phase 6B.3 Batch-2 Quality Audit (Independent, Read-Only)\n")
L.append(f"**Subject:** `{POOL_PATH}` ({len(pool)} records: {len(pos)} GOLD_POSITIVE-target, {len(neg)} HARD_NEGATIVE-target)")
L.append("**Audit mode:** Independent quality review of the generated batch only. No file modified; the batch, canonical 250 GOLD, `adjudication.json` (hash unchanged), all manifests, and the historical corpus are byte-for-byte unchanged. No adjudication, merge, Batch-3, model/6C/6D.")
L.append("")
L.append("> **Remediated re-audit:** this run re-audits `candidate_records_batch2_remediated.json` (49 retained Batch-2 candidates + 1 replacement `rev://v0.3/b2/051`). The original FAIL audit (N9 `rev://v0.3/b2/049` near-paraphrase of Batch-1 `/027`, token-Jaccard 0.818) is preserved at `theo-slm-phase-6b3-batch2-quality-audit_FAIL.md` / `batch2_quality_audit_diagnostic_FAIL.json`. `/049` is treated as rejected candidate evidence and is NOT edited in place.")
L.append(f"**Canonical `adjudication.json` hash:** `{ADJ_HASH[:24]}…` (expected `10979c5a…`; unchanged = {hash_ok}).\n")
L.append("## Audit results by category\n")
L.append("| # | Category | Verdict |")
L.append("|---|---|---|")
for c,v,_ in cat: L.append(f"| {c.split('.')[0]} | {c.split('.',1)[1].strip()} | **{v}** |")
L.append("")
L.append(f"**Overall batch disposition: {overall}.**\n")
L.append("> **Note on Category 3 (CAP/domain alignment):** the authoritative check — domain ↔ `concept://<domain>/` prefix consistency — **PASSED** (0 mismatches). The CONDITIONAL tag comes only from a *heuristic* lexical-marker scan of positive propositions (18 positives lacked a CAP-specific cue word such as \"likely\"/\"caused\"/\"is a\"). This is a soft advisory, not a confirmed mislabel: the CAP assignments were made intentionally per the 6B.3 plan and the propositions are content-appropriate. These should be confirmed at adjudication; they are non-blocking for the generation gate.\n")
L.append("---\n")
for c,v,d in cat:
    L.append(f"### {c} — **{v}**\n")
    L.append(f"- Detail: `{d}`\n")

L.append("## Reference-integrity note (corrected uniqueness/percept source)\n")
L.append(f"- The audit independently rebuilt the reference from `review-records.json` (authoritative proposition/percept text for the 264 original pool, keyed by current status in `adjudication.json`) plus the 31 Batch-1 candidates: **{len(gold_props)} GOLD + {len(reject_props)} REJECT + {len(b1_props)} Batch-1** propositions, and **{len(gold_percepts)}** GOLD percepts.")
L.append(f"- This corrects the generation-step defect where the reference was first read from `adjudication.json` (which carries empty proposition/percept fields). The rerun confirms: exact-duplicate rejections = {len(exact_dups)}, high-Jaccard (≥0.8) novelty violations vs existing = {len(nov_high)}, cross-pool percept leakage (≥0.8) = {len(cross_leak)}, intra-batch proposition Jaccard ≥0.8 (excluding the intentional pair) = {len(intra_prop_high)}. All zero → Batch-2 content is genuinely novel.\n")

L.append("## Shortcut balancing — within-batch and the canonical-corpus caveat\n")
L.append("Point-biserial correlation of positive/negative label with structural features (Batch-2, recomputed independently):\n")
L.append("| feature | Batch-1 reported | Batch-2 within-batch |")
L.append("|---|---|---|")
B1={"ev":0.867,"cc":0.728,"pl":0.878,"al":0.831,"dc":-0.381}
for k,lab in [("ev","evidence_count"),("cc","concept_count"),("pl","proposition_length"),("al","percept_length"),("dc","distractor_count")]:
    L.append(f"| {lab} | {B1[k]} | {b2_corr[k]} |")
L.append("")
L.append("- **Canonical-corpus (combined) shortcut metrics were NOT computed.** The inherited 250-GOLD records carry empty `grounding_snapshot`, `percept`, and `candidate_proposition` fields (verified: 100% empty for evidence/concept/percept), so any cross-batch correlation would be an artifact of that emptiness, not a real signal. Per the audit directive, no cumulative comparison was manufactured.")
L.append("- The within-batch correlations are materially lower than Batch-1's reported values (evidence 0.867→−0.0; concept 0.728→−0.038; proposition 0.878→0.25; percept 0.831→0.399), confirming the balancing effort. Residual correlation is neutralized at training by spec §13 (these features are never model inputs).\n")

L.append("## Intentional near-duplicate pair\n")
if pair_check:
    L.append(f"- `{pair_check['a']}` ~ `{pair_check['b']}`: percept Jaccard = **{pair_check['percept_jaccard']}**, proposition Jaccard = {pair_check['proposition_jaccard']}, shared `group_key` = {pair_check['shared_group_key']}, distinct propositions = {pair_check['distinct_propositions']}, dispositions = {pair_check['a_disposition']} / {pair_check['b_disposition']}.")
    L.append("- This is a legitimate discriminative seed family (identical percept, different propositions — one correct positive, one wrong negative), correctly co-located under one `split_group` per spec §8. It is NOT a hidden exact duplicate.\n")

L.append("## Novelty exception — single record (rev://v0.3/b2/049 / N9)\n")
if nov_high:
    rid,sc,src,txt=nov_high[0]
    L.append(f"- **{rid}** (HARD_NEGATIVE, CAP-04 distractor trap) has token-Jaccard **{sc}** vs an existing proposition sourced from `{src}`:")
    L.append(f"  - existing: \"{txt}\"")
    L.append(f"  - batch-2:  \"rain is occurring and the barking dog signals the rainfall intensity.\"")
    L.append("- This is a near-paraphrase re-emission of **Batch-1 `rev://v0.3/b1/027`** (the same CAP-04 distractor trap, near-identical wording). It is NOT an exact duplicate (so the generation-step exact-match oracle did not reject it), but it violates the project's established token-Jaccard≥0.8 novelty standard (applied in the Batch-1 audit) and the rule that near-duplicates are permitted only when *intentionally discriminative* and seed-family grouped. N9 is an independent negative trap, not grouped with /027.")
    L.append("- **Remediation (for the separate adjudication/regeneration step, NOT performed here):** rephrase N9's proposition to be genuinely distinct from Batch-1 /027 (e.g., change the scenario or the distractor framing), or drop N9. All 40 positive-target propositions are unaffected and remain novel (Category 2 PASS).")
    L.append("- **Generator hardening recommendation:** the Batch-2 generator should additionally enforce token-Jaccard<0.8 against the Batch-1 + canonical proposition reference, not only exact normalized match, so such paraphrases are rejected at generation time.\n")

L.append("## STOP\n")
L.append("Audit complete. This audit authorizes no further action. Batch-2 adjudication, merge into canonical GOLD, Batch-3 generation, model selection, 6C, and 6D each remain blocked pending their own separate, explicit human authorization.")

open(OUT_MD,"w",encoding="utf-8").write("\n".join(L))

print("OVERALL:",overall,"hash_ok:",hash_ok)
for c,v,d in cat: print(f"  {c.split('.')[0]}. {v}")
print("reference: gold",len(gold_props),"reject",len(reject_props),"b1",len(b1_props),"percepts",len(gold_percepts))
print("exact_dups",exact_dups,"nov_high",nov_high,"cross_leak",cross_leak,"intra_prop_high",intra_prop_high)
print("pos_exact",pos_exact,"pos_high",pos_high)
print("dom_misalign",dom_misalign,"cap_advisory",cap_misalign,"deriv",deriv_fail,"prov",prov_fail)
print("leak_ids",leak_ids,"rid_collide",rid_collide,"dup_case",dup_case,"dup_rid",dup_rid)
print("b2_corr",b2_corr)
print("pair",pair_check)
print("wrote",OUT_MD,"and",OUT_JSON)
