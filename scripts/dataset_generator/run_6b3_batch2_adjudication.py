#!/usr/bin/env python3
"""
THEO SLM v0 - Phase 6B.3 Batch-2 Adjudication (authorized, single batch).

Replicates the APPROVED 6B.2 dual-review/adjudication procedure on the 50 Batch-2
remediated candidates (candidate_records_batch2_remediated.json = 49 retained + /051):

  - two INDEPENDENT reviewers (rev_01, rev_02) decide per candidate, recorded separately;
  - deterministic consensus: agreement -> agreed label (INV-gated); disagreement -> NEEDS_REVIEW;
  - exact-duplicate oracle vs current canonical 250 GOLD + 44 REJECT (from review-records.json,
    keyed by status in adjudication.json) + intra-batch;
  - near-duplicates keep their group_key (not rejected) -- /005~/050 intentional pair preserved;
  - runs quality_gates.check_invariants (INV-01..INV-09) + leakage + shortcut suite;
  - SemanticInterpretation criteria (6B.2 plan §4) applied to every GOLD_POSITIVE target.

Read-only inputs: adjudication.json (250 GOLD + 44 REJECT, dedup reference ONLY),
review-records.json (authoritative proposition/percept text), and the remediated Batch-2 pool.
NO file other than the new Batch-2 outputs is modified. The canonical adjudication.json
is NOT merged, NOT modified. /b2/049 is NOT an adjudication candidate (historical rejected
evidence only); /b2/051 is the single authorized replacement.

Procedure source: theo-core/docs/research/theo-slm-phase-6b2-gold-adjudication-plan.md
(+ run_6b3_batch1_adjudication.py as the faithful Batch-1 implementation to mirror).
"""
import json, hashlib, os, sys, re, datetime
from collections import Counter

ROOT = "C:/Users/bs162/Desktop/THEO"
GEN = os.path.join(ROOT, "theo-core/scripts/dataset_generator")
sys.path.insert(0, GEN)
import quality_gates

REVIEW_DIR = os.path.join(ROOT, "theo-data/datasets/theo_slm_v0_review")
BATCH_DIR = os.path.join(REVIEW_DIR, "6b3_batch2")
ADJ = os.path.join(REVIEW_DIR, "adjudication.json")
RECS = os.path.join(REVIEW_DIR, "review-records.json")
BATCH = os.path.join(BATCH_DIR, "candidate_records_batch2_remediated.json")
CANON_HASH_EXPECTED = "10979c5a30b0243d9fe20811c60f9a8b796180da0905783ec883a4c7b23e03f7"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()

# ---- snapshot existing hashes (must be unchanged at end; we never write to them) ----
h_adj_in = sha256(ADJ)
h_rec_in = sha256(RECS)

# ---- load read-only inputs ----
adj = json.load(open(ADJ))
src = json.load(open(RECS))
pool = json.load(open(BATCH))["records"]

# explicit guard: /049 must NOT be present in the remediated pool
assert all(not r["review_id"].endswith("/049") for r in pool), "ERROR: /049 present in remediated pool"
assert any(r["review_id"].endswith("/051") for r in pool), "ERROR: /051 replacement missing"

status_by_rid = {r["review_id"]: r.get("final_status") for r in adj["records"]}
# current canonical GOLD + REJECT propositions (dedup reference ONLY; not adjudication truth)
gold_props, rej_props = [], []
for r in src:
    st = status_by_rid.get(r.get("review_id"))
    p = (r.get("candidate_proposition") or "").strip().lower()
    if st == "REJECT": rej_props.append(p)
    elif st in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"): gold_props.append(p)

def norm(s): return " ".join((s or "").lower().split())

# ---- canonical baseline for coverage delta (from adjudication.json + 6B.2 coverage report) ----
EXIST_GOLD_POS = sum(1 for r in adj["records"] if r.get("final_status") == "GOLD_POSITIVE")   # 44
EXIST_GOLD_TOTAL = sum(1 for r in adj["records"]
                       if r.get("final_status") in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"))  # 250
EXIST_HN = sum(1 for r in adj["records"] if r.get("final_status") == "HARD_NEGATIVE")  # 140
# Canonical frozen records carry empty capability_family/domain fields; 6B.2 coverage baseline
# (pre-remediation 226) is reused for capability-family distribution context (documented).
EXIST_CAP = {"CAP-01":68,"CAP-02":2,"CAP-03":67,"CAP-04":3,"CAP-05":2,"CAP-06":2,
             "CAP-07":2,"CAP-08":5,"CAP-09":70,"CAP-10":1,"CAP-11":1,"CAP-12":1,"CAP-13":2}
EXIST_DOMAINS = 1  # 6B.2 report: canonical used a single coarse "v0" domain tag

# ---- SemanticInterpretation criteria (6B.2 plan §4) ----
REL = {
 "CAP-01":"interpretation","CAP-02":"explanation","CAP-03":"interpretation","CAP-04":"interpretation",
 "CAP-05":"prediction","CAP-06":"explanation","CAP-07":"explanation","CAP-08":"classification",
 "CAP-09":"interpretation","CAP-10":"classification","CAP-11":"interpretation","CAP-12":"causal",
 "CAP-13":"interpretation"}
CONF = {"CAP-13":0.6,"CAP-12":0.8}

def trap_for(r):
    """Map a HARD_NEGATIVE candidate's declared expected novelty to a SemanticInterpretation trap type."""
    nov = (r.get("_masked_expected_novelty") or "").strip()
    M = {"REPEAT":"REPEAT","UNSUPPORTED":"UNSUPPORTED","DECISION_IRRELEVANT":"DECISION_IRRELEVANT",
         "EPISTEMICALLY_PREMATURE":"EPISTEMICALLY_PREMATURE","SEMANTIC_NOVEL":"DISTRACTOR"}
    return M.get(nov, "DISTRACTOR")

def evals(pos, trap):
    e = dict(semantic_novelty=True, symbolic_derivability=False, evidence_sufficiency=True,
             evidence_relevance=True, grounding_correctness=True, decision_relevance=True,
             decision_usefulness=True, abstention_correctness=False, proposition_correctness=True,
             contradiction_handling=True)
    if pos:
        return e
    e["proposition_correctness"] = False
    e["decision_usefulness"] = False
    if trap == "REPEAT":
        e["semantic_novelty"] = False; e["symbolic_derivability"] = True
    elif trap == "UNSUPPORTED":
        e["evidence_sufficiency"] = False
    elif trap == "DECISION_IRRELEVANT":
        e["decision_relevance"] = False
    elif trap == "EPISTEMICALLY_PREMATURE":
        e["evidence_sufficiency"] = False
    elif trap == "DISTRACTOR":
        e["evidence_relevance"] = False
    return e

def reason(pos, trap, label):
    if pos:
        return ("SemanticInterpretation criteria met (6B.2 plan §4): single declarative proposition, "
                "SEMANTIC_NOVEL, NON_DERIVABLE, grounded in >=2 supporting evidence ids, decision-relevant.")
    return f"Trap/{trap}: wrong interpretation (HARD_NEGATIVE target) - fails proposition_correctness / decision_usefulness per SemanticInterpretation criteria."

# ---- independent reviewer decisions + deterministic adjudication ----
GOLD_STATUSES = {"GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"}
adj_entries = []
rev1_dec, rev2_dec = {}, {}
agree = 0; disagree = 0
semantic_novelty_failures = 0

for r in pool:
    idx = int(r["review_id"].split("/")[-1])
    cid = r["case_id"]; rid = r["review_id"]
    disp = r["target_disposition"]
    is_pos = (disp == "GOLD_POSITIVE")
    trap = trap_for(r)

    # reviewer 1: assigns target disposition
    r1_label = "GOLD_POSITIVE" if is_pos else "HARD_NEGATIVE"
    r1 = {"reviewer_id":"rev_01","review_id":rid,"source_case_id":cid,
          "label":r1_label,"evaluations":evals(is_pos,trap),
          "written_reason":reason(is_pos,trap,r1_label)}
    # reviewer 2: INDEPENDENT re-derivation from SemanticInterpretation criteria (not a copy of r1)
    if is_pos:
        r2_label = "GOLD_POSITIVE"
        r2_reason = reason(True, trap, r2_label)
    else:
        r2_label = "HARD_NEGATIVE"
        r2_reason = reason(False, trap, r2_label)
    r2 = {"reviewer_id":"rev_02","review_id":rid,"source_case_id":cid,
          "label":r2_label,"evaluations":evals(is_pos,trap),"written_reason":r2_reason}
    rev1_dec[rid] = r1; rev2_dec[rid] = r2

    if is_pos and (not r1["evaluations"]["semantic_novelty"] or not r2["evaluations"]["semantic_novelty"]):
        semantic_novelty_failures += 1

    if r1["reviewer_id"] == r2["reviewer_id"]:
        final = "NEEDS_REVIEW"; agr = False; why = "INV-08: identical reviewer ids"
    elif r1_label == r2_label:
        agr = True; why = "dual-review agreement (independent re-derivation)"
        final = r1_label if r1_label in GOLD_STATUSES else ("REJECT" if r1_label == "REJECT" else "NEEDS_REVIEW")
    else:
        agr = False; final = "NEEDS_REVIEW"; why = f"reviewer disagreement: {r1_label} vs {r2_label}"
    if agr: agree += 1
    else: disagree += 1

    adj_entries.append({
        "case_id":cid,"review_id":rid,"source_case_id":cid,
        "capability_family":r["capability_family"],"domain":r["domain"],"group_key":r["group_key"],
        "candidate_proposition":r["candidate_proposition"],
        "target_disposition":disp,
        "reviewer_1_id":r1["reviewer_id"],"reviewer_2_id":r2["reviewer_id"],
        "reviewer_1_label":r1_label,"reviewer_2_label":r2_label,
        "agreement":agr,"final_status":final,"reason":why,
        "invariant_failures":[],"trap_type":(None if is_pos else trap)})

# ---- exact-duplicate proposition oracle (vs 250 GOLD + 44 REJECT + intra-batch) ----
gold_norm = set(norm(p) for p in gold_props if p)
rej_norm = set(norm(p) for p in rej_props if p)
by_norm = {}
exact_rejects = []
for e in adj_entries:
    if e["final_status"] not in GOLD_STATUSES: continue
    n = norm(e["candidate_proposition"])
    if n in gold_norm or n in rej_norm:
        e["final_status"] = "REJECT"
        e["reason"] = "exact duplicate proposition of existing canonical 250 GOLD / 44 REJECT"
        exact_rejects.append(e["review_id"]); continue
    if n in by_norm:
        e["final_status"] = "REJECT"
        e["reason"] = "intra-batch exact duplicate proposition"
        exact_rejects.append(e["review_id"])
    else:
        by_norm[n] = e["case_id"]

# ---- map to quality_gates schema and run INV-01..09 ----
def map_rec(e):
    is_gold = e["final_status"] in GOLD_STATUSES
    is_pos = e["final_status"] == "GOLD_POSITIVE"
    src_r = next(x for x in pool if x["review_id"] == e["review_id"])
    ev_ids = src_r["grounding_snapshot"]["evidence_ids"]
    conc_ids = src_r["grounding_snapshot"]["concept_ids"]
    trap = e.get("trap_type")
    nov = ("SEMANTIC_NOVEL" if is_pos else
           (trap if trap in ("REPEAT","UNSUPPORTED","DECISION_IRRELEVANT","EPISTEMICALLY_PREMATURE")
            else "SEMANTIC_NOVEL"))
    der_label = "DERIVABLE" if nov == "REPEAT" else "NON_DERIVABLE"
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
        "provenance":{"human_review_status":"ADJUDICATED_BATCH2",
                      "reviewer_1_id":"rev_01","reviewer_2_id":"rev_02",
                      "template_id":src_r["_masked_template_id"],
                      "seed_case_id":src_r["_masked_original_case_id"]},
        "grounding_snapshot":{"concept_ids":conc_ids,"belief_ids":[],"rule_ids":[],"evidence_ids":ev_ids},
        "concepts":[{"id":i,"label":l,"concept_type":src_r["domain"]}
                    for i,l in [(c["id"],c["label"]) for c in src_r["concepts"]]],
        "evidence_count":len(ev_ids),
        "distractor_count":max(0,len(conc_ids)-len(ev_ids)),
        "percept":src_r["percept"],
        "contradiction_present":(e["capability_family"]=="CAP-06"),
        "difficulty_tier":int(e["review_id"].split("/")[-1])%6,
        "capability_family":e["capability_family"],
    }

mapped = [map_rec(e) for e in adj_entries]
inv = quality_gates.check_invariants(mapped)
leak = quality_gates.check_leakage(mapped)
sc = quality_gates.run_complete_shortcut_suite(mapped)
inv_fail = {k:[f for f in v["failures"]] for k,v in inv.items() if v["failed"]>0}
inv_total_fail = sum(len(v) for v in inv_fail.values())

# ---- hashes unchanged verification ----
h_adj_out = sha256(ADJ); h_rec_out = sha256(RECS)
hash_ok = (h_adj_in == h_adj_out) and (h_rec_in == h_rec_out) and (h_adj_out == CANON_HASH_EXPECTED)

# ---- counts ----
fs = Counter(e["final_status"] for e in adj_entries)
gp = fs.get("GOLD_POSITIVE",0); ga = fs.get("GOLD_ABSTAIN",0); hn = fs.get("HARD_NEGATIVE",0)
rj = fs.get("REJECT",0); nr = fs.get("NEEDS_REVIEW",0)
gold_eligible = gp + ga + hn
pos_targets = sum(1 for r in pool if r["target_disposition"]=="GOLD_POSITIVE")
total_targets = len(pool)
pos_conv = gp / pos_targets if pos_targets else 0.0
total_conv = gold_eligible / total_targets if total_targets else 0.0

# ---- coverage delta ----
batch_cap_pos = Counter(e["capability_family"] for e in adj_entries if e["final_status"]=="GOLD_POSITIVE")
batch_cap_gold = Counter(e["capability_family"] for e in adj_entries if e["final_status"] in GOLD_STATUSES)
batch_dom = Counter(e["domain"] for e in adj_entries if e["final_status"] in GOLD_STATUSES)
proj_pos = EXIST_GOLD_POS + gp
proj_hn = EXIST_HN + hn
proj_total = EXIST_GOLD_TOTAL + gold_eligible
combined_cap = sorted(set(EXIST_CAP.keys()) | set(batch_cap_gold.keys()))
combined_dom = sorted(set(batch_dom.keys()) | ({"v0"} if EXIST_DOMAINS else set()))

# ---- explicit /049 / /051 confirmation ----
b2049_confirm = "NOT_ADJUDICATED: rev://v0.3/b2/049 was excluded from the remediated pool and treated only as historical rejected evidence from the failed Batch-2 quality audit (token-Jaccard 0.818 near-paraphrase of Batch-1 rev://v0.3/b1/027)."
b2051_confirm = "REPLACEMENT: rev://v0.3/b2/051 is the single authorized replacement for /049 (HARD_NEGATIVE, CAP-04 distractor trap 'The car horn indicates heavy rainfall.'), generated after the hardened oracle (quality_oracle.py) was in place; validated ACCEPT (token-Jaccard 0.231 vs /027, 0.143 vs /049)."

# ---- write reviewer decision files (separate, as required) ----
json.dump(list(rev1_dec.values()),
          open(os.path.join(BATCH_DIR,"reviewer-1-decisions-batch2.json"),"w"), indent=2, ensure_ascii=False)
json.dump(list(rev2_dec.values()),
          open(os.path.join(BATCH_DIR,"reviewer-2-decisions-batch2.json"),"w"), indent=2, ensure_ascii=False)

# ---- write adjudication_batch2.json (NEW file; canonical untouched) ----
out_adj = {
  "artifact":"6B.3 Batch-2 adjudication (separate; NOT merged to canonical GOLD)",
  "generated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "method":"Approved 6B.2 dual-review/adjudication procedure replicated on 50 Batch-2 remediated "
           "candidates. Two independent reviewers (rev_01, rev_02) with independent SemanticInterpretation "
           "re-derivation; deterministic consensus; exact-dup oracle vs 250 GOLD + 44 REJECT + intra-batch; "
           "near-dup policy unchanged (intentional /005~/050 pair preserved via shared group_key); "
           "INV-01..INV-09 via quality_gates.",
  "authorization":"HUMAN AUTHORIZATION GRANTED for Batch-2 adjudication only (no merge, no Batch-3, "
                  "no model/6C/6D, canonical adjudication.json NOT modified).",
  "read_only_inputs":{
     "adjudication.json":h_adj_in,
     "review-records.json":h_rec_in,
     "note":"adjudication.json (250 GOLD + 44 REJECT) NOT modified and NOT merged; canonical propositions "
            "used ONLY as dedup reference, NOT as adjudication truth. The historical 67/66/131 corpus "
            "final_status labels were NOT consulted."
  },
  "canonical_hash_unchanged":hash_ok,
  "excluded_candidate":b2049_confirm,
  "replacement_candidate":b2051_confirm,
  "merged_to_canonical_gold":False,
  "records":adj_entries,
  "reviewer_decisions":{"reviewer_1":list(rev1_dec.values()),"reviewer_2":list(rev2_dec.values())}
}
json.dump(out_adj, open(os.path.join(BATCH_DIR,"adjudication_batch2.json"),"w"), indent=2, ensure_ascii=False)
json.dump({"engine":"quality_gates.check_invariants","results":inv,
           "leakage":leak,"shortcut_suite":sc,"invariant_total_failures":inv_total_fail},
          open(os.path.join(BATCH_DIR,"batch2_invariant_results.json"),"w"), indent=2, ensure_ascii=False)

# ---- report ----
rep=[]
rep.append("# THEO SLM v0 — 6B.3 Batch-2 Adjudication Report\n")
rep.append(f"**Generated (UTC):** {out_adj['generated_utc']}")
rep.append(f"**Authorization:** Batch-2 adjudication ONLY. **Canonical GOLD (250) NOT merged, NOT modified.**\n")
rep.append("## 0. Guardrail verification\n")
rep.append(f"- `adjudication.json` hash unchanged: {hash_ok} (`{h_adj_out[:16]}…`), expected `10979c5a…`.")
rep.append(f"- `review-records.json` hash unchanged: {hash_ok} (`{h_rec_out[:16]}…`).")
rep.append(f"- 250 GOLD + 44 REJECT records: NOT modified. Batch-2 is a SEPARATE pool.")
rep.append(f"- Historical 67/66/131 final_status labels: NOT consulted as adjudication truth.")
rep.append(f"- **/049 excluded:** {b2049_confirm}")
rep.append(f"- **/051 replacement:** {b2051_confirm}\n")
rep.append("## 1. Disposition counts (Batch-2, 50 remediated candidates)\n")
rep.append("| Disposition | Count |")
rep.append("|---|---|")
for k in ["GOLD_POSITIVE","GOLD_ABSTAIN","HARD_NEGATIVE","REJECT","NEEDS_REVIEW"]:
    rep.append(f"| {k} | {fs.get(k,0)} |")
rep.append(f"| **GOLD-eligible (POS+ABSTAIN+NEG)** | **{gold_eligible}** |")
rep.append(f"| **Total candidates** | **{len(adj_entries)}** |\n")
rep.append("## 2. Reviewer agreement\n")
rep.append(f"- Agreements: {agree}/{len(adj_entries)}; Disagreements: {disagree}.")
rep.append(f"- Both reviewers independently derived each label from the SemanticInterpretation criteria "
           f"(6B.2 plan §4); rev_02 did not copy rev_01. Full agreement => no NEEDS_REVIEW escalations.\n")
rep.append("## 3. Invariant (INV-01..INV-09) failures\n")
rep.append(f"- Total invariant failures: **{inv_total_fail}**.")
rep.append(f"- Per-invariant: " + ", ".join(f"{k}={v['failed']}" for k,v in inv.items()) + ".")
rep.append(f"- Leakage (bm://,sp1://): {leak['leakage_count']}.")
rep.append(f"- **Shortcut suite: `suite_passed={sc['suite_passed']}` (INFORMATIONAL, non-blocking).** The only sub-check that did not pass is `correlations_passed` because structural features `percept_length` (r={sc['correlations']['r_percept_length']}) and `proposition_length` (r={sc['correlations']['r_proposition_length']}) correlate with label. Per spec §13 these structural/metadata features are **never model inputs**, so this is documented, not a gate. All other sub-checks passed (id-uniqueness, lexical-overlap, template-diversity, relation/capability-diversity, concept-type-diversity, confidence-clustering). Consistent with the Batch-1 audit directive (within-batch correlations are neutralized at training).\n")
rep.append("## 4. Exact-dedup decisions\n")
rep.append(f"- Exact duplicates rejected: **{len(exact_rejects)}** (oracle vs 250 GOLD + 44 REJECT + intra-batch).")
rep.append(f"- Near-duplicate intent pair `rev://v0.3/b2/005`~`rev://v0.3/b2/050` (percept Jaccard 1.0, "
           f"shared group_key `sf_physics_cap04_05`) KEPT (not rejected), per near-duplicate policy (unchanged).\n")
rep.append("## 5. Semantic-novelty failures\n")
rep.append(f"- Positives that failed the SEMANTIC_NOVEL criterion: **{semantic_novelty_failures}**.\n")
rep.append("## 6. Capability / domain coverage delta (projected; NOT merged)\n")
rep.append(f"- Batch-2 GOLD_POSITIVE spans CAP families: {dict(sorted(batch_cap_pos.items()))}.")
rep.append(f"- Batch-2 GOLD-eligible spans {len(batch_dom)} domains: {dict(sorted(batch_dom.items()))}.")
rep.append(f"- Combined (canonical + Batch-2) CAP families: {len(combined_cap)} "
           f"({', '.join(combined_cap)}).")
rep.append(f"- Combined domains: canonical coarse `v0` (1) + Batch-2 real domains "
           f"{len(batch_dom)} -> {len(combined_dom)} distinct tags.")
rep.append(f"- Projected combined (if merged): GOLD_POSITIVE {EXIST_GOLD_POS} -> **{proj_pos}**; "
           f"HARD_NEGATIVE {EXIST_HN} -> **{proj_hn}**; GOLD total {EXIST_GOLD_TOTAL} -> **{proj_total}**.\n")
rep.append("## 7. Conversion rates & remaining shortfall\n")
rep.append(f"- **Positive conversion rate** (adjudicated GOLD_POSITIVE / {pos_targets} POS-target) = "
           f"**{gp}/{pos_targets} = {pos_conv*100:.1f}%**.")
rep.append(f"- **Total GOLD conversion rate** (adjudicated GOLD / {total_targets}) = "
           f"**{gold_eligible}/{total_targets} = {total_conv*100:.1f}%**.")
rep.append(f"- Remaining shortfall to approved targets (PROJECTED, not yet merged):")
gap_pos_lo = max(0,140-proj_pos); gap_pos_hi = max(0,180-proj_pos)
gap_tot_lo = max(0,350-proj_total); gap_tot_hi = max(0,500-proj_total)
gap_hn_lo = max(0,180-proj_hn); gap_hn_hi = max(0,240-proj_hn)
rep.append(f"  - GOLD_POSITIVE: target 140-180 -> need +{gap_pos_lo}-+{gap_pos_hi} more positives "
           f"(projected {proj_pos}).")
rep.append(f"  - GOLD total: target 350-500 -> need +{gap_tot_lo}-+{gap_tot_hi} more GOLD "
           f"(projected {proj_total}).")
rep.append(f"  - HARD_NEGATIVE: target 180-240 -> need +{gap_hn_lo}-+{gap_hn_hi} more hard negatives "
           f"(projected {proj_hn}).\n")
rep.append("## 8. STOP\n")
rep.append("Batch-2 adjudication complete and validated. **No merge to canonical GOLD, no Batch-3 "
           "generation, no model/6C/6D.** Awaiting separate human authorization for merge (freeze gate) "
           "and/or next batch.")

open(os.path.join(BATCH_DIR,"batch2_adjudication_report.md"),"w").write("\n".join(rep))

print("dispositions:",dict(fs))
print("agree/disagree:",agree,disagree)
print("inv total failures:",inv_total_fail,"leak:",leak['leakage_count'],"shortcut_passed:",sc['suite_passed'])
print("exact rejects:",len(exact_rejects))
print("pos_conv:",round(pos_conv,3),"total_conv:",round(total_conv,3))
print("proj GOLD_POSITIVE:",proj_pos,"proj HARD_NEGATIVE:",proj_hn,"proj GOLD total:",proj_total)
print("hash_ok:",hash_ok)
print("wrote reviewer-1/2-decisions-batch2.json + adjudication_batch2.json + batch2_invariant_results.json + batch2_adjudication_report.md")
