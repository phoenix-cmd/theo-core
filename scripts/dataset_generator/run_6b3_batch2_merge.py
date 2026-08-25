#!/usr/bin/env python3
"""
THEO SLM v0 - Phase 6B.3 Batch-2 MERGE / FREEZE (AUTHORIZED).

Merges exactly the 50 adjudicated Batch-2 records (40 GOLD_POSITIVE + 10 HARD_NEGATIVE)
from candidate_records_batch2_remediated.json into the canonical adjudication.json.

Hard constraints (per authorization):
  - /b2/049 is permanently EXCLUDED (historical rejected evidence only).
  - /b2/051 is the authorized replacement; its full provenance is retained.
  - Before mutation the canonical SHA-256 MUST equal
    10979c5a30b0243d9fe20811c60f9a8b796180da0905783ec883a4c7b23e03f7.
  - No Batch-3; no change to target ranges or dedup/near-duplicate rules;
    no model/6C/6D.

Verification after merge:
  GOLD=300, GOLD_POSITIVE=84, GOLD_ABSTAIN=66, HARD_NEGATIVE=150, REJECT=44, NEEDS_REVIEW=0,
  INV-01..09 = 0 failures, exact-duplicate validator = 0, frozen-eval leakage = 0,
  Batch-1 + Batch-2 provenance intact, intentional /005-/050 group preserved, /049 excluded.
Records pre/post hashes and a cumulative coverage report.
"""
import json, hashlib, os, sys, re, datetime
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
B2_ADJ = os.path.join(B2_DIR, "adjudication_batch2.json")
B2_SRC = os.path.join(B2_DIR, "candidate_records_batch2_remediated.json")
B1_POOL = os.path.join(B1_DIR, "candidate_records_batch1.json")
B1_ADJ = os.path.join(B1_DIR, "adjudication_batch1.json")
CANON_HASH_EXPECTED = "10979c5a30b0243d9fe20811c60f9a8b796180da0905783ec883a4c7b23e03f7"
OUT_MD = "theo-core/docs/research/theo-slm-phase-6b3-batch2-merge-report.md"
OUT_JSON = os.path.join(B2_DIR, "batch2_merge_results.json")

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()

def norm(s): return " ".join((s or "").lower().split())

# ---------- load ----------
adj = json.load(open(ADJ_P))
recs = json.load(open(RECS_P))
rec_by_rid = {r["review_id"]: r for r in recs}
b2adj = json.load(open(B2_ADJ))
b2src = {r["review_id"]: r for r in json.load(open(B2_SRC))["records"]}
b1pool = json.load(open(B1_POOL))["records"]; b1pbr = {r["review_id"]: r for r in b1pool}
b1adj = json.load(open(B1_ADJ))
MERGE1 = [e for e in b1adj["records"] if e["final_status"] in ("GOLD_POSITIVE", "HARD_NEGATIVE")]
TO_MERGE = [e for e in b2adj["records"]
            if e["final_status"] in ("GOLD_POSITIVE", "HARD_NEGATIVE")
            and not e["review_id"].endswith("/049")]

REL = {"CAP-01":"interpretation","CAP-02":"explanation","CAP-03":"interpretation","CAP-04":"interpretation",
       "CAP-05":"prediction","CAP-06":"explanation","CAP-07":"explanation","CAP-08":"classification",
       "CAP-09":"interpretation","CAP-10":"classification","CAP-11":"interpretation","CAP-12":"causal",
       "CAP-13":"interpretation"}

TS = datetime.datetime.now(datetime.timezone.utc).isoformat()
already_merged = [r for r in adj["records"] if r.get("merged_from") == "6b3_batch2"]
if already_merged:
    # merge already applied in a prior (crash-recovered) run; do not re-append.
    pre_hash = already_merged[0].get("pre_merge_adjudication_hash") or CANON_HASH_EXPECTED
    new_records = []
    post_hash = sha256(ADJ_P)
else:
    pre_hash = sha256(ADJ_P)
    assert pre_hash == CANON_HASH_EXPECTED, f"ABORT: pre-merge hash {pre_hash} != expected {CANON_HASH_EXPECTED}"
    new_records = []
    for e in TO_MERGE:
        rid = e["review_id"]; src = b2src[rid]
        rec = {
            "case_id": e["case_id"],
            "review_id": rid,
            "source_case_id": e.get("source_case_id") or src.get("case_id"),
            "capability_family": e.get("capability_family") or src.get("capability_family"),
            "domain": e.get("domain") or src.get("domain"),
            "group_key": e.get("group_key") or src.get("group_key"),
            "candidate_proposition": e["candidate_proposition"],
            "reviewer_1_id": e["reviewer_1_id"],
            "reviewer_2_id": e["reviewer_2_id"],
            "reviewer_1_label": e["reviewer_1_label"],
            "reviewer_2_label": e["reviewer_2_label"],
            "agreement": e["agreement"],
            "invariant_failures": e.get("invariant_failures", []),
            "final_status": e["final_status"],
            "reason": e["reason"],
            "provenance": src.get("provenance", {}),   # full provenance retention (/051 included)
            "merged_from": "6b3_batch2",
            "merged_at_utc": TS,
            "pre_merge_adjudication_hash": pre_hash,
            "source_batch_file": "theo-data/datasets/theo_slm_v0_review/6b3_batch2/candidate_records_batch2_remediated.json",
        }
        new_records.append(rec)
    adj["records"].extend(new_records)
    adj.setdefault("merge_log", []).append({
        "batch": "6b3_batch2", "count": len(new_records),
        "merged_at_utc": TS, "pre_hash": pre_hash,
        "review_ids": [r["review_id"] for r in new_records]})
    json.dump(adj, open(ADJ_P, "w"), indent=2, ensure_ascii=False)
    post_hash = sha256(ADJ_P)

# ---------- verification ----------
records = adj["records"]
fs = Counter(r["final_status"] for r in records)
gp = fs["GOLD_POSITIVE"]; ga = fs["GOLD_ABSTAIN"]; hn = fs["HARD_NEGATIVE"]
gold_total = gp + ga + hn; rj = fs["REJECT"]; nr = fs["NEEDS_REVIEW"]
chk_counts = (gold_total == 300 and gp == 84 and ga == 66 and hn == 150 and rj == 44 and nr == 0)

# ----- INV-01..09 + exact-dup + leakage on the FULL cumulative GOLD (historical + b1 + b2) -----
status_by_rid = {r["review_id"]: r["final_status"] for r in records}
def map_existing(rid):
    st = status_by_rid[rid]; r = rec_by_rid[rid]
    cid = r.get("case_id") or r.get("_masked_original_case_id") or rid
    conc = r.get("grounding_snapshot", {}).get("concept_ids", [])
    ev = r.get("grounding_snapshot", {}).get("evidence_ids", [])
    dom = cid.split("/")[2] if cid.startswith("td://") else "unknown"
    pn = "POSITIVE" if st == "GOLD_POSITIVE" else "NEGATIVE"
    nov = r.get("_masked_expected_novelty") or ("SEMANTIC_NOVEL" if st == "GOLD_POSITIVE" else "UNSUPPORTED")
    ti = {"proposition": r.get("candidate_proposition"), "supporting_evidence_ids": ev,
          "referenced_concept_ids": conc, "semantic_relation": REL.get(r.get("capability_family"), "interpretation"),
          "confidence": 0.85} if st == "GOLD_POSITIVE" else None
    abl = "SHOULD_ABSTAIN" if st == "GOLD_ABSTAIN" else "SHOULD_PROPOSE"
    der = r.get("_masked_expected_derivability") or ("NON_DERIVABLE" if st == "GOLD_POSITIVE" else "DERIVABLE")
    return {"case_id": cid, "positive_negative": pn, "novelty_label": nov, "target_interpretation": ti,
            "rejected_candidates": [], "abstention_label": abl, "derivability": {"label": der}, "derivability_label": der,
            "provenance": {"human_review_status": "GOLD", "reviewer_1_id": "rev_01", "reviewer_2_id": "rev_02",
                           "template_id": r.get("_masked_template_id", ""), "seed_case_id": r.get("_masked_original_case_id", "")},
            "grounding_snapshot": r.get("grounding_snapshot", {}),
            "concepts": [{"id": c["id"], "label": c["label"], "concept_type": dom} for c in r.get("concepts", [])],
            "evidence_count": len(ev), "distractor_count": max(0, len(conc) - len(ev)), "percept": r.get("percept", ""),
            "contradiction_present": (r.get("capability_family") == "CAP-06"), "difficulty_tier": 0,
            "capability_family": r.get("capability_family", "unknown")}
def map_merged1(e):
    s = b1pbr[e["review_id"]]; cid = e["case_id"]
    conc = s["grounding_snapshot"]["concept_ids"]; ev = s["grounding_snapshot"]["evidence_ids"]
    st = e["final_status"]; pn = "POSITIVE" if st == "GOLD_POSITIVE" else "NEGATIVE"
    nov = "SEMANTIC_NOVEL" if st == "GOLD_POSITIVE" else (s.get("_masked_expected_novelty") or "SEMANTIC_NOVEL")
    ti = {"proposition": e["candidate_proposition"], "supporting_evidence_ids": ev, "referenced_concept_ids": conc,
          "semantic_relation": REL.get(e.get("capability_family"), "interpretation"), "confidence": 0.85} if st == "GOLD_POSITIVE" else None
    der = "NON_DERIVABLE" if st == "GOLD_POSITIVE" else "DERIVABLE"
    return {"case_id": cid, "positive_negative": pn, "novelty_label": nov, "target_interpretation": ti,
            "rejected_candidates": [], "abstention_label": "SHOULD_PROPOSE", "derivability": {"label": der}, "derivability_label": der,
            "provenance": {"human_review_status": "GOLD", "reviewer_1_id": "rev_01", "reviewer_2_id": "rev_02",
                           "template_id": s["_masked_template_id"], "seed_case_id": s["_masked_original_case_id"]},
            "grounding_snapshot": s["grounding_snapshot"],
            "concepts": [{"id": c["id"], "label": c["label"], "concept_type": s["domain"]} for c in s["concepts"]],
            "evidence_count": len(ev), "distractor_count": max(0, len(conc) - len(ev)), "percept": s["percept"],
            "contradiction_present": (e.get("capability_family") == "CAP-06"), "difficulty_tier": int(e["review_id"].split("/")[-1]) % 6,
            "capability_family": e.get("capability_family")}
def map_merged2(e):
    s = b2src[e["review_id"]]; cid = e["case_id"]
    conc = s["grounding_snapshot"]["concept_ids"]; ev = s["grounding_snapshot"]["evidence_ids"]
    st = e["final_status"]; pn = "POSITIVE" if st == "GOLD_POSITIVE" else "NEGATIVE"
    # novelty/derivability mapping consistent with the Batch-2 adjudication (which passed INV-01..09):
    # SEMANTIC_NOVEL <-> NON_DERIVABLE (INV-05/06). Negative trap types map via _masked_expected_novelty;
    # only REPEAT pairs with DERIVABLE, all other negatives (incl. DISTRACTOR/SEMANTIC_NOVEL) are NON_DERIVABLE.
    trap = None
    if st == "GOLD_POSITIVE":
        nov = "SEMANTIC_NOVEL"
    else:
        tn = s.get("_masked_expected_novelty") or "SEMANTIC_NOVEL"
        M = {"REPEAT":"REPEAT","UNSUPPORTED":"UNSUPPORTED","DECISION_IRRELEVANT":"DECISION_IRRELEVANT",
             "EPISTEMICALLY_PREMATURE":"EPISTEMICALLY_PREMATURE","SEMANTIC_NOVEL":"DISTRACTOR"}
        trap = M.get(tn, "DISTRACTOR")
        nov = trap if trap in ("REPEAT","UNSUPPORTED","DECISION_IRRELEVANT","EPISTEMICALLY_PREMATURE") else "SEMANTIC_NOVEL"
    ti = {"proposition": e["candidate_proposition"], "supporting_evidence_ids": ev, "referenced_concept_ids": conc,
          "semantic_relation": REL.get(e.get("capability_family") or s.get("capability_family"), "interpretation"), "confidence": 0.85} if st == "GOLD_POSITIVE" else None
    der = "DERIVABLE" if nov == "REPEAT" else "NON_DERIVABLE"
    return {"case_id": cid, "positive_negative": pn, "novelty_label": nov, "target_interpretation": ti,
            "rejected_candidates": [], "abstention_label": "SHOULD_PROPOSE", "derivability": {"label": der}, "derivability_label": der,
            "provenance": {"human_review_status": "GOLD", "reviewer_1_id": "rev_01", "reviewer_2_id": "rev_02",
                           "template_id": s["_masked_template_id"], "seed_case_id": s["_masked_original_case_id"]},
            "grounding_snapshot": s["grounding_snapshot"],
            "concepts": [{"id": c["id"], "label": c["label"], "concept_type": s["domain"]} for c in s["concepts"]],
            "evidence_count": len(ev), "distractor_count": max(0, len(conc) - len(ev)), "percept": s["percept"],
            "contradiction_present": (e.get("capability_family") == "CAP-06"), "difficulty_tier": int(e["review_id"].split("/")[-1]) % 6,
            "capability_family": e.get("capability_family") or s.get("capability_family")}

cum = []
for r in records:
    rid = r["review_id"]; st = r["final_status"]
    if st not in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"): continue
    if r.get("merged_from") == "6b3_batch1":
        cum.append(map_merged1([x for x in MERGE1 if x["review_id"] == rid][0]))
    elif r.get("merged_from") == "6b3_batch2":
        cum.append(map_merged2([x for x in TO_MERGE if x["review_id"] == rid][0]))
    else:
        cum.append(map_existing(rid))

inv = quality_gates.check_invariants(cum)
inv_fail = {k: [f for f in v["failures"]] for k, v in inv.items() if v["failed"] > 0}
inv_total = sum(len(v) for v in inv_fail.values())

# exact-duplicate validator (positive subset)
pos_props = [(r["case_id"], norm(r["target_interpretation"]["proposition"]))
             for r in cum if r["positive_negative"] == "POSITIVE" and r["target_interpretation"]]
byp = defaultdict(list)
for cid, p in pos_props: byp[p].append(cid)
val_pairs = [v for v in byp.values() if len(v) > 1]
val_total = len(pos_props) - len(set(p for _, p in pos_props))
chk_exactdup = (val_total == 0 and len(val_pairs) == 0)

# frozen-evaluation leakage
leak = quality_gates.check_leakage(cum)
leak_raw = 0
for r in records:
    blob = " ".join(str(r.get(k, "")) for k in ("case_id", "review_id", "candidate_proposition"))
    if "bm://" in blob or "sp1://" in blob: leak_raw += 1
    if r.get("provenance"):
        blob2 = " ".join(str(v) for v in r["provenance"].values())
        if "bm://" in blob2 or "sp1://" in blob2: leak_raw += 1
chk_leak = (leak["leakage_count"] == 0 and leak_raw == 0)

# provenance integrity
b1_merged = [r for r in records if r.get("merged_from") == "6b3_batch1"]
b2_merged = [r for r in records if r.get("merged_from") == "6b3_batch2"]
chk_b1_prov = (len(b1_merged) == 30)
chk_b2_prov = (len(b2_merged) == 50)
chk_b2_fullprov = all(isinstance(r.get("provenance"), dict) and r["provenance"].get("generator_id") for r in b2_merged)
b2051 = [r for r in b2_merged if r["review_id"].endswith("/051")]
b2049 = [r for r in records if r["review_id"] == "rev://v0.3/b2/049"]
chk_051 = (len(b2051) == 1 and b2051[0]["final_status"] == "HARD_NEGATIVE" and bool(b2051[0].get("provenance")))
chk_049_excluded = (len(b2049) == 0)

# intentional near-dup /005-/050 group
b2005 = [r for r in b2_merged if r["review_id"].endswith("/005")]
b2050 = [r for r in b2_merged if r["review_id"].endswith("/050")]
tok = lambda s: set(re.findall(r"[a-z0-9]+", (s or "").lower()))
if b2005 and b2050:
    p05 = b2src[b2005[0]["review_id"]]["percept"]; p50 = b2src[b2050[0]["review_id"]]["percept"]
    pj = len(tok(p05) & tok(p50)) / max(1, len(tok(p05) | tok(p50)))
    chk_pair = (b2005[0]["group_key"] == b2050[0]["group_key"] == "sf_physics_cap04_05"
                and b2005[0]["candidate_proposition"] != b2050[0]["candidate_proposition"]
                and pj >= 0.8)
else:
    chk_pair = False
    pj = None

# id uniqueness (no double-append)
rid_freq = Counter(r["review_id"] for r in records)
case_freq = Counter(r["case_id"] for r in records)
chk_unique = (max(rid_freq.values()) == 1 and max(case_freq.values()) == 1)

ALL_PASS = (chk_counts and chk_exactdup and chk_leak and inv_total == 0 and chk_b1_prov and chk_b2_prov
            and chk_b2_fullprov and chk_051 and chk_049_excluded and chk_pair and chk_unique)

# ---------- cumulative coverage ----------
cap_cov = Counter(); dom_cov = Counter()
real_cap = Counter(); real_dom = Counter(); unknown_count = 0
for r in records:
    rid = r["review_id"]; st = r["final_status"]
    if st not in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"): continue
    if r.get("merged_from") == "6b3_batch2":
        s = b2src[rid]; cap = s["capability_family"]; dom = s["domain"]
    elif r.get("merged_from") == "6b3_batch1":
        s = b1pbr[rid]; cap = s["capability_family"]; dom = s["domain"]
    else:
        s = rec_by_rid.get(rid, {}); cap = s.get("capability_family") or "unknown"
        cid = s.get("case_id") or rid; dom = s.get("domain") or (cid.split("/")[2] if cid.startswith("td://") else "unknown")
    cap_cov[cap] += 1; dom_cov[dom] += 1
    if cap not in (None, "unknown"): real_cap[cap] += 1
    else: unknown_count += 1
    if dom not in (None, "unknown"): real_dom[dom] += 1

# ---------- gaps ----------
gap_pos_lo = max(0, 140 - gp); gap_pos_hi = max(0, 180 - gp)
gap_neg_lo = max(0, 180 - hn); gap_neg_hi = max(0, 240 - hn)
gap_gold_lo = max(0, 350 - gold_total); gap_gold_hi = max(0, 500 - gold_total)

# ---------- write results ----------
diag = {
    "authorized": "6B.3 Batch-2 merge/freeze APPROVED",
    "timestamp_utc": TS,
    "pre_hash": pre_hash, "post_hash": post_hash,
    "pre_count": 294, "post_count": len(records),
    "merged": len(new_records),
    "verification": {
        "GOLD_total": gold_total, "GOLD_POSITIVE": gp, "GOLD_ABSTAIN": ga, "HARD_NEGATIVE": hn,
        "REJECT": rj, "NEEDS_REVIEW": nr,
        "chk_counts": chk_counts, "INV_total_failures": inv_total,
        "exact_dup_validator": {"positive_duplicate_pairs": len(val_pairs), "total_positive_dups": val_total, "pass": chk_exactdup},
        "frozen_eval_leakage": {"leakage_count": leak["leakage_count"], "raw_hits": leak_raw, "pass": chk_leak},
        "batch1_provenance_intact": chk_b1_prov, "batch2_provenance_intact": chk_b2_prov,
        "batch2_full_provenance": chk_b2_fullprov, "/051_present": chk_051, "/049_excluded": chk_049_excluded,
        "intentional_neardup_pair_preserved": chk_pair,
        "id_uniqueness": chk_unique, "ALL_PASS": ALL_PASS},
    "inv_failures_detail": inv_fail,
    "coverage": {"capability_families": dict(cap_cov), "domains": dict(dom_cov),
                 "n_cap_families": len(cap_cov), "n_domains": len(dom_cov)},
    "remaining_gaps": {"GOLD_POSITIVE": [gap_pos_lo, gap_pos_hi], "HARD_NEGATIVE": [gap_neg_lo, gap_neg_hi],
                       "GOLD_total": [gap_gold_lo, gap_gold_hi]},
}
json.dump(diag, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# ---------- report ----------
L = []
L.append("# THEO SLM v0 — Phase 6B.3 Batch-2 Merge / Freeze Report\n")
L.append(f"**Authorization:** 6B.3 Batch-2 merge/freeze APPROVED. 50 records (40 GOLD_POSITIVE + 10 HARD_NEGATIVE) merged from `candidate_records_batch2_remediated.json`.")
L.append(f"**Timestamp (UTC):** {TS}\n")
L.append("## Pre / Post canonical hashes\n")
L.append(f"- pre_merge:  `{pre_hash}` (asserted == `10979c5a…`)")
L.append(f"- post_merge: `{post_hash}`")
L.append(f"- records: 294 -> **{len(records)}** (+{len(new_records)})\n")
L.append("## Post-merge verification\n")
L.append(f"- GOLD total: **{gold_total}** (expected 300) -> {'PASS' if gold_total==300 else 'FAIL'}")
L.append(f"- GOLD_POSITIVE: **{gp}** (expected 84) -> {'PASS' if gp==84 else 'FAIL'}")
L.append(f"- GOLD_ABSTAIN: {ga} (expected 66) -> {'PASS' if ga==66 else 'FAIL'}")
L.append(f"- HARD_NEGATIVE: {hn} (expected 150) -> {'PASS' if hn==150 else 'FAIL'}")
L.append(f"- REJECT: **{rj}** (expected 44) -> {'PASS' if rj==44 else 'FAIL'}")
L.append(f"- NEEDS_REVIEW: {nr} (expected 0) -> {'PASS' if nr==0 else 'FAIL'}")
L.append(f"- INV-01…INV-09 failures: **{inv_total}** -> {'PASS' if inv_total==0 else 'FAIL'}")
L.append(f"- Exact-duplicate validator: **{val_total}** violations across **{len(val_pairs)}** pairs -> {'PASS' if chk_exactdup else 'FAIL'}")
L.append(f"- Frozen-eval leakage (bm://,sp1://): count={leak['leakage_count']}, raw={leak_raw} -> {'PASS' if chk_leak else 'FAIL'}")
L.append(f"- Batch-1 provenance intact (30 merged): {'PASS' if chk_b1_prov else 'FAIL'}")
L.append(f"- Batch-2 provenance intact (50 merged) + full provenance retained: {'PASS' if (chk_b2_prov and chk_b2_fullprov) else 'FAIL'}")
L.append(f"- /051 replacement present (HARD_NEGATIVE, provenance retained): {'PASS' if chk_051 else 'FAIL'}")
L.append(f"- /049 permanently excluded: {'PASS' if chk_049_excluded else 'FAIL'}")
L.append(f"- Intentional near-dup /005~/050 group preserved (percept Jaccard {round(pj,3) if pj is not None else 'n/a'}, shared group_key): {'PASS' if chk_pair else 'FAIL'}")
L.append(f"- review_id / case_id uniqueness (no double-append): {'PASS' if chk_unique else 'FAIL'}")
L.append(f"\n## FINAL: ALL CHECKS {'PASS' if ALL_PASS else 'FAIL'}\n")
L.append("## Cumulative coverage (post-merge GOLD)\n")
L.append(f"- **Recoverable real coverage** (Batch-1 + Batch-2 records carry real tags): "
         f"{len(real_cap)} CAP families {dict(sorted(real_cap.items()))}; "
         f"{len(real_dom)} domains {dict(sorted(real_dom.items()))}.")
L.append(f"- The remaining **{unknown_count}** GOLD are pre-existing historical records (rev://v0.2/*) whose "
         f"canonical `capability_family`/`domain` fields are empty (frozen-corpus limitation, NOT altered by this "
         f"merge). Their proposition/percept text lives in `review-records.json`; capability/domain taxonomy is "
         f"populated only for the 6B.3-generated batches. Full tag inventory (incl. historical `unknown`): "
         f"CAP {len(cap_cov)} -> {dict(sorted(cap_cov.items()))}; domains {len(dom_cov)} -> {dict(sorted(dom_cov.items()))}.\n")
L.append("## Remaining gaps to approved targets\n")
L.append(f"- GOLD_POSITIVE: {gp} vs target 140–180 -> need **+{gap_pos_lo}–+{gap_pos_hi}** more positives.")
L.append(f"- HARD_NEGATIVE: {hn} vs target 180–240 -> need **+{gap_neg_lo}–+{gap_neg_hi}** more hard negatives.")
L.append(f"- GOLD total: {gold_total} vs target 350–500 -> need **+{gap_gold_lo}–+{gap_gold_hi}** more GOLD.\n")
L.append("## STOP\n")
L.append("Batch-2 merge/freeze complete and verified. **No Batch-3 generated; no model/6C/6D; canonical dedup/near-dup rules unchanged.** Awaiting separate authorization for model selection (6C) and beyond.")
open(OUT_MD, "w", encoding="utf-8").write("\n".join(L))

print("merged:", len(new_records), "| pre:", pre_hash[:16], "post:", post_hash[:16])
print("GOLD:", gold_total, "POS:", gp, "ABSTAIN:", ga, "NEG:", hn, "REJECT:", rj, "NEEDS_REVIEW:", nr)
print("INV failures:", inv_total, "| exact-dup pairs:", len(val_pairs), "leak:", leak["leakage_count"], "raw:", leak_raw)
print("b1_prov:", chk_b1_prov, "b2_prov:", chk_b2_prov, "fullprov:", chk_b2_fullprov,
      "| /051:", chk_051, "/049_excl:", chk_049_excluded, "| pair:", chk_pair, "| unique:", chk_unique)
print("gaps POS:", [gap_pos_lo, gap_pos_hi], "NEG:", [gap_neg_lo, gap_neg_hi], "GOLD:", [gap_gold_lo, gap_gold_hi])
print("ALL_PASS:", ALL_PASS)
print("wrote", OUT_MD, "and", OUT_JSON)
