#!/usr/bin/env python3
"""
THEO SLM v0 - 6B.3 Hardened Uniqueness / Novelty Oracle (reusable module).

Project policy (enforced here):
  - EXACT duplicate (normalized proposition equals any reference proposition)
        -> ALWAYS REJECT.
  - NEAR duplicate (token-Jaccard >= 0.80 vs any reference proposition)
        -> REJECT,
           UNLESS an explicit, verified intentional seed-family relationship is
           declared (the two propositions form a declared pair AND share a
           verified group_key).
  - Legitimate intentional near-duplicates are RETAINED only with provenance + group_key.

Jaccard >= 0.80 is a *review/exception boundary*, not an automatic accept/reject in
every case. The oracle REJECTS near-duplicates by default; an intentional, grouped
pair is the only sanctioned exception. This is exactly why the /005-/050 pair is
allowed while the /049 near-paraphrase of /027 is rejected.

The authoritative reference is built from REAL proposition/percept text:
  - canonical GOLD + REJECT propositions/percepts from review-records.json
    (the authoritative text source — NOT the empty fields in adjudication.json),
  - all previously generated/adjudicated batch candidate files (e.g., Batch-1, Batch-2).
Empty adjudication.json fields are never used as a substitute for real text.
"""
import json, re, os

def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def toks(s):
    return set(re.findall(r"\w+", (s or "").lower()))

def jaccard(a, b):
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

REVIEW_DIR = "theo-data/datasets/theo_slm_v0_review"

def build_reference(review_dir=REVIEW_DIR, batch_candidate_files=None, include_adjudication_safety=True):
    """Build the authoritative novelty reference.

    batch_candidate_files: list of paths to previously generated/adjudicated batch
    candidate JSON files (each containing a 'records' list). For cumulative leakage
    these should include Batch-1, Batch-2, etc.
    """
    adj = json.load(open(os.path.join(review_dir, "adjudication.json")))
    recs = json.load(open(os.path.join(review_dir, "review-records.json")))
    status_by_rid = {r["review_id"]: r.get("final_status") for r in adj["records"]}

    gold, reject = set(), set()
    gold_percepts = []
    for r in recs:
        rid = r.get("review_id"); st = status_by_rid.get(rid)
        p = norm(r.get("candidate_proposition"))
        if p:
            if st == "REJECT":
                reject.add(p)
            elif st in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"):
                gold.add(p)
        perc = norm(r.get("percept"))
        if perc and st in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"):
            gold_percepts.append(perc)

    batch_props, batch_percepts = set(), []
    labeled = []
    for pf in (batch_candidate_files or []):
        data = json.load(open(pf))
        recs_b = data["records"] if isinstance(data, dict) else data
        for r in recs_b:
            p = norm(r.get("candidate_proposition"))
            if p:
                batch_props.add(p)
                labeled.append(("batch:" + r.get("review_id", ""), p))
            perc = norm(r.get("percept"))
            if perc:
                batch_percepts.append(perc)

    if include_adjudication_safety:
        for r in adj["records"]:
            p = norm(r.get("candidate_proposition"))
            if p:
                st = r.get("final_status")
                if st == "REJECT":
                    reject.add(p)
                elif st in ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"):
                    gold.add(p)
                labeled.append(("adj:" + r.get("review_id", ""), p))

    all_props = gold | reject | batch_props
    for p in gold:
        labeled.append(("gold", p))
    for p in reject:
        labeled.append(("reject", p))

    return {
        "gold_props": gold,
        "reject_props": reject,
        "batch_props": batch_props,
        "all_props": all_props,
        "all_props_labeled": labeled,
        "gold_percepts": gold_percepts,
        "batch_percepts": batch_percepts,
    }

def evaluate(prop, ref, intentional_pairs=None):
    """Return (decision, jaccard_or_None).

    decision in:
      EXACT_DUP_REJECT, NEAR_DUP_REJECT, NEAR_DUP_ACCEPT_INTENTIONAL, ACCEPT.
    """
    pn = norm(prop)
    if pn in ref["all_props"]:
        return ("EXACT_DUP_REJECT", None)
    best_j, best_src = 0.0, None
    for label, p in ref["all_props_labeled"]:
        j = jaccard(prop, p)
        if j > best_j:
            best_j, best_src = j, (label, p)
    if best_j >= 0.80:
        if intentional_pairs and (pn, norm(best_src[1])) in intentional_pairs:
            return ("NEAR_DUP_ACCEPT_INTENTIONAL", round(best_j, 3))
        return ("NEAR_DUP_REJECT", round(best_j, 3))
    return ("ACCEPT", round(best_j, 3))

def declare_intentional_pairs(records):
    """Build the set of intentional (prop_a_norm, prop_b_norm) pairs from records
    that carry a `near_of` pointer AND share a verified group_key."""
    by_rid = {r["review_id"]: r for r in records}
    pairs = set()
    for r in records:
        no = r.get("near_of")
        if no is None:
            continue
        other_rid = f"rev://v0.3/b2/{no:03d}" if isinstance(no, int) else no
        other = by_rid.get(other_rid)
        if not other:
            continue
        if r.get("group_key") == other.get("group_key"):
            a, b = norm(r["candidate_proposition"]), norm(other["candidate_proposition"])
            pairs.add((a, b)); pairs.add((b, a))
    return pairs
