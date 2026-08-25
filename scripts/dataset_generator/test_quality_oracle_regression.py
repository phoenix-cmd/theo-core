#!/usr/bin/env python3
"""
THEO SLM v0 - 6B.3 Batch-2 remediation REGRESSION TEST.

Guards the hardened uniqueness/novelty oracle (quality_oracle.py) against the
exact defect that caused the Batch-2 quality-audit failure:

  Batch-1 rev://v0.3/b1/027  : "Rain is occurring and the barking dog indicates rainfall intensity."
  Batch-2 rev://v0.3/b2/049  : "Rain is occurring and the barking dog signals the rainfall intensity."

The oracle MUST flag this pair at token-Jaccard ~0.818 as NEAR_DUP_REJECT
(not ACCEPT, not EXACT_DUP_REJECT). This test fails if the oracle regresses to
the old exact-match-only behavior that let /049 slip through.

Run:  py theo-core/scripts/dataset_generator/test_quality_oracle_regression.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import quality_oracle as Q

REVIEW = os.path.join(ROOT, "theo-data", "datasets", "theo_slm_v0_review")
B1 = os.path.join(REVIEW, "6b3_batch1", "candidate_records_batch1.json")
B2 = os.path.join(REVIEW, "6b3_batch2", "candidate_records_batch2.json")

def load_prop(path, suffix):
    data = json.load(open(path, encoding="utf-8"))
    recs = data["records"] if isinstance(data, dict) else data
    for r in recs:
        if r["review_id"].endswith(suffix):
            return r["candidate_proposition"]
    raise SystemExit(f"fixture record {suffix} not found in {path}")

def main():
    p027 = load_prop(B1, "/027")
    p049 = load_prop(B2, "/049")

    j = Q.jaccard(p027, p049)
    assert j >= 0.80, f"fixture invariant broken: /027 vs /049 Jaccard = {j} (expected >= 0.80)"

    # Reference = canonical (review-records) + Batch-1 (which contains /027).
    # /049 is the CANDIDATE under test, so it must NOT be in the reference yet.
    ref = Q.build_reference(review_dir=REVIEW, batch_candidate_files=[B1])
    dec, score = Q.evaluate(p049, ref)

    assert dec == "NEAR_DUP_REJECT", f"oracle must REJECT /049 as near-dup; got {dec} (j={score})"
    assert score is not None and score >= 0.80, f"oracle near-dup score = {score} (expected >= 0.80)"

    # Sanity: the remediated /051 must PASS (ACCEPT) against the same reference.
    p051 = "The car horn indicates heavy rainfall."
    dec2, score2 = Q.evaluate(p051, ref)
    assert dec2 == "ACCEPT", f"oracle must ACCEPT remediated /051; got {dec2} (j={score2})"

    print(f"PASS: /027 vs /049 Jaccard = {round(j,3)} -> {dec} @ {score}")
    print(f"PASS: remediated /051 -> {dec2} @ {score2}")

if __name__ == "__main__":
    main()
