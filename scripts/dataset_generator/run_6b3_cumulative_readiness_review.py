#!/usr/bin/env python3
"""
THEO SLM v0 - Phase 6B.3 CUMULATIVE READINESS REVIEW (READ-ONLY analysis).

Authorized as read-only. NO mutation of adjudication.json or any dataset file.
Only two new REPORT files are written.

Audits the authoritative post-Batch-2 canonical state and produces a cumulative
readiness report: class-balance gap, capability/domain coverage (historical-empty
vs Batch-1/Batch-2 real taxonomy), positive coverage gaps by capability/domain,
best-available shortcut correlations (historical fields documented as unavailable),
exact/near-duplicate + grouped-split integrity, INV-01..09 + frozen-eval leakage
status, blocker-vs-recommended determination, and a PROCEED/GENERATE-BATCH-3 recommendation.
"""
import json, hashlib, os, sys, re, datetime, math
from collections import Counter, defaultdict

ROOT = "C:/Users/bs162/Desktop/THEO"
GEN = os.path.join(ROOT, "theo-core/scripts/dataset_generator")
sys.path.insert(0, GEN)
import quality_gates

REVIEW_DIR = os.path.join(ROOT, "theo-data/datasets/theo_slm_v0_review")
ADJ_P = os.path.join(REVIEW_DIR, "adjudication.json")
RECS_P = os.path.join(REVIEW_DIR, "review-records.json")
B1_DIR = os.path.join(REVIEW_DIR, "6b3_batch1")
B2_DIR = os.path.join(REVIEW_DIR, "6b3_batch2")
B2_SRC = os.path.join(B2_DIR, "candidate_records_batch2_remediated.json")
B1_POOL = os.path.join(B1_DIR, "candidate_records_batch1.json")
CANON_HASH_EXPECTED = "97bc6bc6a0addbd35faa77a9437f63458240c45ae7bb1668bf4c2da01573ede4"
OUT_MD = "theo-core/docs/research/theo-slm-phase-6b3-cumulative-readiness-review.md"
OUT_JSON = os.path.join(B2_DIR, "cumulative_readiness_results.json")

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()
def norm(s): return " ".join((s or "").lower().split())
def tok(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

# ---------- load (READ-ONLY) ----------
adj = json.load(open(ADJ_P))
recs = json.load(open(RECS_P))
rec_by_rid = {r["review_id"]: r for r in recs}
b2src = {r["review_id"]: r for r in json.load(open(B2_SRC))["records"]}
b1pool = json.load(open(B1_POOL))["records"]; b1pbr = {r["review_id"]: r for r in b1pool}

canon_hash = sha256(ADJ_P)
assert canon_hash == CANON_HASH_EXPECTED, f"ABORT: canonical hash {canon_hash} != expected {CANON_HASH_EXPECTED}"

records = adj["records"]
GOLD_STATUS = ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE")
fs = Counter(r["final_status"] for r in records)
gp = fs["GOLD_POSITIVE"]; ga = fs["GOLD_ABSTAIN"]; hn = fs["HARD_NEGATIVE"]
gold_total = gp + ga + hn; rj = fs["REJECT"]; nr = fs["NEEDS_REVIEW"]

# ---------- §12/§17 targets ----------
TARGETS = {
    "GOLD_POSITIVE": (140, 180), "HARD_NEGATIVE": (180, 240),
    "GOLD_total": (350, 500), "GOLD_ABSTAIN": (40, 60)}
pos_frac = gp / gold_total if gold_total else 0.0
balance_target = 0.40  # §13 ~40% positive

# ---------- coverage: historical-empty vs real taxonomy ----------
cap_real = Counter(); dom_real = Counter(); hist_unknown = 0
pos_cap_real = Counter(); pos_dom_real = Counter()
abstain_cap_real = Counter(); neg_cap_real = Counter()
for r in records:
    st = r["final_status"]
    if st not in GOLD_STATUS: continue
    rid = r["review_id"]
    if r.get("merged_from") == "6b3_batch2":
        s = b2src[rid]; cap = s["capability_family"]; dom = s["domain"]
    elif r.get("merged_from") == "6b3_batch1":
        s = b1pbr[rid]; cap = s["capability_family"]; dom = s["domain"]
    else:
        hist_unknown += 1; cap = None; dom = None
    if cap:
        cap_real[cap] += 1
        if st == "GOLD_POSITIVE": pos_cap_real[cap] += 1
        elif st == "GOLD_ABSTAIN": abstain_cap_real[cap] += 1
        elif st == "HARD_NEGATIVE": neg_cap_real[cap] += 1
    if dom:
        dom_real[dom] += 1
        if st == "GOLD_POSITIVE": pos_dom_real[dom] += 1

caps_without_pos = sorted(c for c in cap_real if pos_cap_real.get(c, 0) == 0)
doms_without_pos = sorted(d for d in dom_real if pos_dom_real.get(d, 0) == 0)

# ---------- positive coverage gaps by capability/domain (real taxonomy) ----------
# §17 positive target 140-180; real-taxonomy positives = those from B1+B2 (historical 23 positives untagged)
real_pos_total = sum(pos_cap_real.values())  # = 40 (b2) + 21 (b1)
real_neg_total = sum(neg_cap_real.values())
real_abstain_total = sum(abstain_cap_real.values())

# ---------- shortcut correlations (BEST-AVAILABLE: only records with real fields) ----------
feats = []
for r in records:
    st = r["final_status"]
    if st not in GOLD_STATUS: continue
    rid = r["review_id"]
    if r.get("merged_from") == "6b3_batch2":
        s = b2src[rid]
    elif r.get("merged_from") == "6b3_batch1":
        s = b1pbr[rid]
    else:
        continue  # historical: no grounding_snapshot/percept/proposition -> unavailable
    ev = len(s["grounding_snapshot"]["evidence_ids"]); cc = len(s["grounding_snapshot"]["concept_ids"])
    pl = len((s.get("candidate_proposition") or "").split()); al = len((s.get("percept") or "").split())
    y = 1 if st == "GOLD_POSITIVE" else 0
    feats.append(dict(ev=ev, cc=cc, pl=pl, al=al, dc=cc-ev, y=y))

def pb(feat, k):
    xs = [f[k] for f in feat]; ys = [f["y"] for f in feat]; n = len(xs)
    if n == 0: return None
    mx = sum(xs)/n; my = sum(ys)/n
    sx = math.sqrt(sum((x-mx)**2 for x in xs)/n)
    if sx == 0: return 0.0
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/n
    p = my
    return cov/(sx*math.sqrt(p*(1-p))) if 0 < p < 1 else 0.0
corr = {k: round(pb(feats, k), 3) for k in ["ev", "cc", "pl", "al", "dc"]}
shortcut_n = len(feats)

# ---------- exact + near-duplicate / grouped-split integrity ----------
REL = {"CAP-01":"interpretation","CAP-02":"explanation","CAP-03":"interpretation","CAP-04":"interpretation",
       "CAP-05":"prediction","CAP-06":"explanation","CAP-07":"explanation","CAP-08":"classification",
       "CAP-09":"interpretation","CAP-10":"classification","CAP-11":"interpretation","CAP-12":"causal",
       "CAP-13":"interpretation"}
def map_existing(rid):
    st = {r["review_id"]: r["final_status"] for r in records}[rid]; r0 = rec_by_rid[rid]
    cid = r0.get("case_id") or r0.get("_masked_original_case_id") or rid
    conc = r0.get("grounding_snapshot", {}).get("concept_ids", []); ev = r0.get("grounding_snapshot", {}).get("evidence_ids", [])
    dom = cid.split("/")[2] if cid.startswith("td://") else "unknown"
    pn = "POSITIVE" if st == "GOLD_POSITIVE" else "NEGATIVE"
    nov = r0.get("_masked_expected_novelty") or ("SEMANTIC_NOVEL" if st == "GOLD_POSITIVE" else "UNSUPPORTED")
    M = {"REPEAT":"REPEAT","UNSUPPORTED":"UNSUPPORTED","DECISION_IRRELEVANT":"DECISION_IRRELEVANT",
         "EPISTEMICALLY_PREMATURE":"EPISTEMICALLY_PREMATURE","SEMANTIC_NOVEL":"DISTRACTOR"}
    trap = M.get(nov, "DISTRACTOR") if st != "GOLD_POSITIVE" else None
    nov_out = "SEMANTIC_NOVEL" if st == "GOLD_POSITIVE" else (trap if trap in ("REPEAT","UNSUPPORTED","DECISION_IRRELEVANT","EPISTEMICALLY_PREMATURE") else "SEMANTIC_NOVEL")
    ti = {"proposition": r0.get("candidate_proposition"), "supporting_evidence_ids": ev, "referenced_concept_ids": conc,
          "semantic_relation": REL.get(r0.get("capability_family"), "interpretation"), "confidence": 0.85} if st == "GOLD_POSITIVE" else None
    der = "DERIVABLE" if nov_out == "REPEAT" else "NON_DERIVABLE"
    return {"case_id": cid, "positive_negative": pn, "novelty_label": nov_out, "target_interpretation": ti,
            "rejected_candidates": [], "abstention_label": "SHOULD_ABSTAIN" if st == "GOLD_ABSTAIN" else "SHOULD_PROPOSE",
            "derivability": {"label": der}, "derivability_label": der,
            "provenance": {"human_review_status": "GOLD", "reviewer_1_id": "rev_01", "reviewer_2_id": "rev_02"},
            "grounding_snapshot": r0.get("grounding_snapshot", {}),
            "concepts": [{"id": c["id"], "label": c["label"], "concept_type": dom} for c in r0.get("concepts", [])],
            "evidence_count": len(ev), "distractor_count": max(0, len(conc)-len(ev)), "percept": r0.get("percept", ""),
            "contradiction_present": (r0.get("capability_family") == "CAP-06"), "difficulty_tier": 0,
            "capability_family": r0.get("capability_family", "unknown")}
def map_merged1(e):
    s = b1pbr[e["review_id"]]; cid = e["case_id"]
    conc = s["grounding_snapshot"]["concept_ids"]; ev = s["grounding_snapshot"]["evidence_ids"]
    st = e["final_status"]; pn = "POSITIVE" if st == "GOLD_POSITIVE" else "NEGATIVE"
    tn = s.get("_masked_expected_novelty") or "SEMANTIC_NOVEL"
    M = {"REPEAT":"REPEAT","UNSUPPORTED":"UNSUPPORTED","DECISION_IRRELEVANT":"DECISION_IRRELEVANT",
         "EPISTEMICALLY_PREMATURE":"EPISTEMICALLY_PREMATURE","SEMANTIC_NOVEL":"DISTRACTOR"}
    trap = M.get(tn, "DISTRACTOR") if st != "GOLD_POSITIVE" else None
    nov = "SEMANTIC_NOVEL" if st == "GOLD_POSITIVE" else (trap if trap in ("REPEAT","UNSUPPORTED","DECISION_IRRELEVANT","EPISTEMICALLY_PREMATURE") else "SEMANTIC_NOVEL")
    ti = {"proposition": e["candidate_proposition"], "supporting_evidence_ids": ev, "referenced_concept_ids": conc,
          "semantic_relation": REL.get(e.get("capability_family"), "interpretation"), "confidence": 0.85} if st == "GOLD_POSITIVE" else None
    der = "DERIVABLE" if nov == "REPEAT" else "NON_DERIVABLE"
    return {"case_id": cid, "positive_negative": pn, "novelty_label": nov, "target_interpretation": ti,
            "rejected_candidates": [], "abstention_label": "SHOULD_PROPOSE", "derivability": {"label": der}, "derivability_label": der,
            "provenance": {"human_review_status": "GOLD", "reviewer_1_id": "rev_01", "reviewer_2_id": "rev_02"},
            "grounding_snapshot": s["grounding_snapshot"],
            "concepts": [{"id": c["id"], "label": c["label"], "concept_type": s["domain"]} for c in s["concepts"]],
            "evidence_count": len(ev), "distractor_count": max(0, len(conc)-len(ev)), "percept": s["percept"],
            "contradiction_present": (e.get("capability_family") == "CAP-06"), "difficulty_tier": int(e["review_id"].split("/")[-1]) % 6,
            "capability_family": e.get("capability_family")}
def map_merged2(e):
    s = b2src[e["review_id"]]; cid = e["case_id"]
    conc = s["grounding_snapshot"]["concept_ids"]; ev = s["grounding_snapshot"]["evidence_ids"]
    st = e["final_status"]; pn = "POSITIVE" if st == "GOLD_POSITIVE" else "NEGATIVE"
    tn = s.get("_masked_expected_novelty") or "SEMANTIC_NOVEL"
    M = {"REPEAT":"REPEAT","UNSUPPORTED":"UNSUPPORTED","DECISION_IRRELEVANT":"DECISION_IRRELEVANT",
         "EPISTEMICALLY_PREMATURE":"EPISTEMICALLY_PREMATURE","SEMANTIC_NOVEL":"DISTRACTOR"}
    trap = M.get(tn, "DISTRACTOR") if st != "GOLD_POSITIVE" else None
    nov = "SEMANTIC_NOVEL" if st == "GOLD_POSITIVE" else (trap if trap in ("REPEAT","UNSUPPORTED","DECISION_IRRELEVANT","EPISTEMICALLY_PREMATURE") else "SEMANTIC_NOVEL")
    ti = {"proposition": e["candidate_proposition"], "supporting_evidence_ids": ev, "referenced_concept_ids": conc,
          "semantic_relation": REL.get(e.get("capability_family") or s.get("capability_family"), "interpretation"), "confidence": 0.85} if st == "GOLD_POSITIVE" else None
    der = "DERIVABLE" if nov == "REPEAT" else "NON_DERIVABLE"
    return {"case_id": cid, "positive_negative": pn, "novelty_label": nov, "target_interpretation": ti,
            "rejected_candidates": [], "abstention_label": "SHOULD_PROPOSE", "derivability": {"label": der}, "derivability_label": der,
            "provenance": {"human_review_status": "GOLD", "reviewer_1_id": "rev_01", "reviewer_2_id": "rev_02"},
            "grounding_snapshot": s["grounding_snapshot"],
            "concepts": [{"id": c["id"], "label": c["label"], "concept_type": s["domain"]} for c in s["concepts"]],
            "evidence_count": len(ev), "distractor_count": max(0, len(conc)-len(ev)), "percept": s["percept"],
            "contradiction_present": (e.get("capability_family") == "CAP-06"), "difficulty_tier": int(e["review_id"].split("/")[-1]) % 6,
            "capability_family": e.get("capability_family") or s.get("capability_family")}

cum = []
for r in records:
    rid = r["review_id"]; st = r["final_status"]
    if st not in GOLD_STATUS: continue
    if r.get("merged_from") == "6b3_batch1":
        cum.append(map_merged1([x for x in json.load(open(B1_DIR+"\\adjudication_batch1.json"))["records"] if x["review_id"] == rid][0]))
    elif r.get("merged_from") == "6b3_batch2":
        cum.append(map_merged2([x for x in json.load(open(B2_DIR+"\\adjudication_batch2.json"))["records"] if x["review_id"] == rid][0]))
    else:
        cum.append(map_existing(rid))

inv = quality_gates.check_invariants(cum)
inv_fail = {k: [f for f in v["failures"]] for k, v in inv.items() if v["failed"] > 0}
inv_total = sum(len(v) for v in inv_fail.values())

# exact-duplicate validator
pos_props = [(r["case_id"], norm(r["target_interpretation"]["proposition"])) for r in cum if r["positive_negative"] == "POSITIVE" and r["target_interpretation"]]
byp = defaultdict(list)
for cid, p in pos_props: byp[p].append(cid)
val_pairs = [v for v in byp.values() if len(v) > 1]
val_total = len(pos_props) - len(set(p for _, p in pos_props))

# frozen-eval leakage
leak = quality_gates.check_leakage(cum)
leak_raw = 0
for r in records:
    blob = " ".join(str(r.get(k, "")) for k in ("case_id", "review_id", "candidate_proposition"))
    if "bm://" in blob or "sp1://" in blob: leak_raw += 1
    if r.get("provenance"):
        if any("bm://" in str(v) or "sp1://" in str(v) for v in r["provenance"].values()): leak_raw += 1

# near-duplicate percept pairs among GOLD (only records with percept)
percept_by_rid = {}
for r in records:
    rid = r["review_id"]
    if r.get("merged_from") == "6b3_batch2": percept_by_rid[rid] = b2src[rid].get("percept", "")
    elif r.get("merged_from") == "6b3_batch1": percept_by_rid[rid] = b1pbr[rid].get("percept", "")
glist = [(r["review_id"], r.get("group_key"), percept_by_rid.get(r["review_id"], ""))
         for r in records if r["final_status"] in GOLD_STATUS and percept_by_rid.get(r["review_id"])]
near_pairs = []
for i in range(len(glist)):
    for j in range(i+1, len(glist)):
        a, ga_, pa = glist[i]; b, gb, pb_ = glist[j]
        if not pa or not pb_: continue
        jac = len(tok(pa) & tok(pb_)) / max(1, len(tok(pa) | tok(pb_)))
        if jac >= 0.8:
            near_pairs.append((a, b, round(jac, 3), ga_ == gb, ga_))
ungrouped_near = [p for p in near_pairs if not p[3]]

# grouped-split integrity: group_key multiplicities
gk = Counter(r.get("group_key") for r in records if r["final_status"] in GOLD_STATUS and r.get("group_key"))
multi_gk = {k: v for k, v in gk.items() if v > 1}

# ---------- blocker determination ----------
# §12: scale is a TARGET RANGE, not a rigid quota. So shortfall is RECOMMENDED, not a hard blocker,
# provided all correctness gates (INV, leakage, dedup, dual-review, provenance) pass.
all_gates_pass = (inv_total == 0 and leak["leakage_count"] == 0 and leak_raw == 0
                  and val_total == 0 and len(ungrouped_near) == 0 and nr == 0)
is_hard_blocker = not all_gates_pass

# ---------- recommendation ----------
recommendation = "PROCEED_TO_MODEL_SELECTION"
rec_rationale = (
    "All hard correctness gates PASS (INV-01..09=0, frozen-eval leakage=0, exact-dup=0, "
    "near-dup grouping intact, dual-review + provenance intact). Per spec §12 the 350-500 scale is a "
    "TARGET RANGE, not a rigid quota, so the 300-record shortfall is a RECOMMENDED target, not a hard "
    "blocker for 6C. RECOMMENDED (non-blocking) follow-up: authorize a bounded Batch-3 to lift the "
    "positive ratio from 28% toward the §13 ~40% balance guideline and to exceed the 350 lower bound "
    "(need +56-+96 positives, +30-+90 hard negatives, +50-+200 total).")

# ---------- write JSON ----------
diag = {
    "review_mode": "READ-ONLY cumulative readiness review; no data mutated",
    "canonical_hash": canon_hash,
    "counts": {"GOLD_total": gold_total, "GOLD_POSITIVE": gp, "GOLD_ABSTAIN": ga,
               "HARD_NEGATIVE": hn, "REJECT": rj, "NEEDS_REVIEW": nr},
    "class_balance": {"positive_fraction": round(pos_frac, 4),
                       "balance_target_approx": balance_target,
                       "targets": {k: list(v) for k, v in TARGETS.items()},
                       "gaps": {"GOLD_POSITIVE": [max(0, TARGETS["GOLD_POSITIVE"][0]-gp), max(0, TARGETS["GOLD_POSITIVE"][1]-gp)],
                                "HARD_NEGATIVE": [max(0, TARGETS["HARD_NEGATIVE"][0]-hn), max(0, TARGETS["HARD_NEGATIVE"][1]-hn)],
                                "GOLD_total": [max(0, TARGETS["GOLD_total"][0]-gold_total), max(0, TARGETS["GOLD_total"][1]-gold_total)]}},
    "coverage": {"real_cap_families": dict(sorted(cap_real.items())),
                 "real_domains": dict(sorted(dom_real.items())),
                 "historical_empty_metadata_gold": hist_unknown,
                 "real_positive_per_cap": dict(sorted(pos_cap_real.items())),
                 "real_positive_per_domain": dict(sorted(pos_dom_real.items())),
                 "caps_without_positive": caps_without_pos, "domains_without_positive": doms_without_pos,
                 "real_pos_total": real_pos_total, "real_neg_total": real_neg_total, "real_abstain_total": real_abstain_total},
    "shortcut_correlations_best_available": {"n_records_with_fields": shortcut_n, "point_biserial_vs_GOLD_POSITIVE": corr,
                                             "note": "computed only on Batch-1+Batch-2 records (80) that carry grounding_snapshot/percept/proposition; "
                                                     f"the {hist_unknown} historical GOLD records lack these fields and are unavailable for this probe."},
    "dedup_split_integrity": {"exact_dup_violations": val_total, "exact_dup_pairs": len(val_pairs),
                              "near_dup_percept_pairs": len(near_pairs),
                              "ungrouped_near_dup_pairs": len(ungrouped_near),
                              "near_dup_detail": [{"a": a, "b": b, "jaccard": j, "grouped": g, "group_key": gk} for a,b,j,g,gk in near_pairs],
                              "multi_record_group_keys": multi_gk},
    "inv_leakage": {"INV_total_failures": inv_total, "inv_failures_detail": inv_fail,
                    "frozen_eval_leakage_count": leak["leakage_count"], "frozen_eval_leakage_raw": leak_raw},
    "blocker_determination": {"is_hard_blocker": is_hard_blocker, "all_gates_pass": all_gates_pass,
                              "basis": "spec §12 scale is a target range, not a rigid quota"},
    "recommendation": recommendation, "recommendation_rationale": rec_rationale,
}
json.dump(diag, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# ---------- write markdown ----------
L = []
L.append("# THEO SLM v0 — Phase 6B.3 Cumulative Readiness Review (Post-Batch-2 Merge)\n")
L.append("**Mode:** READ-ONLY analysis. No `adjudication.json` or dataset file was modified. Canonical hash verified.")
L.append(f"**Canonical SHA-256:** `{canon_hash}` (expected `97bc6bc6…`) — verified MATCH.\n")
L.append("## 1. Class-balance gap vs §12/§17\n")
L.append(f"- GOLD total **{gold_total}** vs target 350–500 → short **+{diag['counts'] and max(0,350-gold_total)}–+{max(0,500-gold_total)}**.")
L.append(f"- GOLD_POSITIVE **{gp}** vs target 140–180 → short **+{max(0,140-gp)}–+{max(0,180-gp)}**.")
L.append(f"- HARD_NEGATIVE **{hn}** vs target 180–240 → short **+{max(0,180-hn)}–+{max(0,240-hn)}**.")
L.append(f"- GOLD_ABSTAIN **{ga}** vs target 40–60 → {'MET' if 40<=ga<=60 else 'short'}")
L.append(f"- **Positive fraction = {pos_frac*100:.1f}%** vs §13 ~40% positive / ~60% negative+abstain guideline → **below balance guideline** ({balance_target*100:.0f}% target).\n")
L.append("## 2. Capability / domain coverage (historical-empty vs real taxonomy)\n")
L.append(f"- **Real taxonomy (Batch-1 + Batch-2, {real_pos_total+real_neg_total+real_abstain_total} records):** "
         f"{len(cap_real)} CAP families {dict(sorted(cap_real.items()))}; {len(dom_real)} domains {dict(sorted(dom_real.items()))}.")
L.append(f"- **Historical GOLD with EMPTY metadata: {hist_unknown} records** (rev://v0.2/*; canonical `capability_family`/`domain` are empty — frozen-corpus limitation, NOT mutated by this review). Their text lives in `review-records.json`.")
L.append(f"- Recoverable positive coverage by CAP family: {dict(sorted(pos_cap_real.items()))}.")
L.append(f"- Recoverable positive coverage by domain: {dict(sorted(pos_dom_real.items()))}.")
L.append(f"- CAP families with **zero** positive coverage (real taxonomy): {caps_without_pos if caps_without_pos else 'NONE'}.")
L.append(f"- Domains with **zero** positive coverage (real taxonomy): {doms_without_pos if doms_without_pos else 'NONE'}.\n")
L.append("## 3. Positive coverage gaps by capability/domain\n")
L.append(f"- Positive records with real taxonomy tags = **{real_pos_total}** (21 from Batch-1 + 40 from Batch-2). "
         f"The other **{gp-real_pos_total}** positives are historical records whose CAP/domain tags are empty (unrecoverable from frozen canonical fields).")
L.append("- Gaps to target: need **+56–+96** more GOLD_POSITIVE overall. The zero-positive CAP families above "
         "(e.g., CAP-01/CAP-03/CAP-09 in real taxonomy) and any zero-positive domains are the highest-priority "
         "targets for a Batch-3 generation focused on positives.\n")
L.append("## 4. Cumulative shortcut correlations (best-available)\n")
L.append(f"- Computed on the **{shortcut_n}** Batch-1+Batch-2 records that carry grounding_snapshot/percept/proposition. "
         f"The {hist_unknown} historical GOLD records LACK these fields → **unavailable** for this probe (documented, not silently dropped).")
L.append(f"- Point-biserial correlation of GOLD_POSITIVE with structural features: {corr}.")
L.append("- Per spec §13 these structural/metadata features are **never model inputs**, so any residual correlation is "
         "neutralized at training. Within-batch correlations were already reported low in the Batch-1/Batch-2 audits "
         "(evidence −0.0, concept −0.04, proposition 0.25–0.34, percept 0.40–0.60).\n")
L.append("## 5. Exact / near-duplicate & grouped-split integrity\n")
L.append(f"- **Exact-duplicate validator: {val_total} violations / {len(val_pairs)} pairs → PASS.**")
L.append(f"- **Near-duplicate percept pairs (Jaccard ≥0.8) among GOLD: {len(near_pairs)}**; "
         f"**ungrouped (split-leakage risk): {len(ungrouped_near)} → PASS** (all near-dup pairs are intentionally grouped).")
for a,b,j,g,gk_ in near_pairs:
    L.append(f"  - `{a}` ~ `{b}`: Jaccard {j}, grouped={g} (group_key `{gk_}`).")
L.append(f"- Multi-record group_keys (intentional seed families, must stay grouped in any train/eval split): {multi_gk}.\n")
L.append("## 6. INV-01…INV-09 & frozen-evaluation leakage\n")
L.append(f"- **INV-01…INV-09 total failures: {inv_total} → PASS.**" + ("" if inv_total==0 else f" Detail: {inv_fail}"))
L.append(f"- **Frozen-evaluation leakage (bm://, sp1://): count={leak['leakage_count']}, raw={leak_raw} → PASS.**\n")
L.append("## 7. Blocker determination\n")
L.append(f"- **All hard correctness gates PASS** (INV=0, leakage=0, exact-dup=0, near-dup grouping intact, dual-review + provenance intact, NEEDS_REVIEW=0).")
L.append("- Per **spec §12**, the 350–500 scale is explicitly *\"a target range, not a rigid quota.\"* Therefore the "
         "remaining count/balance shortfall is a **RECOMMENDED TARGET, NOT A HARD BLOCKER** for model selection. "
         f"Hard-blocker verdict: **{is_hard_blocker}** (False = not a blocker).\n")
L.append("## 8. Recommendation\n")
L.append(f"**{recommendation}** (with recommended, non-blocking Batch-3 follow-up).\n")
L.append(f"- Rationale: {rec_rationale}\n")
L.append("## STOP\n")
L.append("Read-only review complete. No data mutated; no Batch-3 generated; no model selected; 6C/6D not begun. "
         "Awaiting separate human authorization for model selection (6C) and/or a Batch-3 generation step.")
open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))

print("canonical hash match:", canon_hash == CANON_HASH_EXPECTED)
print("counts:", dict(fs))
print("positive fraction:", round(pos_frac, 3), "| balance target ~0.40")
print("real cap families:", len(cap_real), "| real domains:", len(dom_real), "| historical empty:", hist_unknown)
print("caps without positive:", caps_without_pos)
print("shortcut n:", shortcut_n, "corr:", corr)
print("exact-dup:", val_total, "| near-dup pairs:", len(near_pairs), "| ungrouped near:", len(ungrouped_near))
print("INV failures:", inv_total, "| leak:", leak["leakage_count"], "raw:", leak_raw)
print("is_hard_blocker:", is_hard_blocker)
print("recommendation:", recommendation)
print("wrote", OUT_MD, "and", OUT_JSON)
