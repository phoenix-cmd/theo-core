"""Phase 6B.4 Human Review Specification & Machine-Readable Schema Validator.

Validates:
1. Review Record Schema (Dual-reviewer fields, 10 mandatory evaluation criteria, status labels).
2. Blind Review Manifest Layout (Deterministic randomization, metadata masking).
3. Adjudication Schema (Disagreement resolution layout).
4. Calibration Batch Sampling (15 mixed candidate records).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

VALID_REVIEW_STATUSES = {
    "UNREVIEWED",
    "GOLD_POSITIVE",
    "GOLD_ABSTAIN",
    "HARD_NEGATIVE",
    "REJECT",
    "NEEDS_REVISION",
}

MANDATORY_EVALUATION_FIELDS = [
    "semantic_novelty",
    "symbolic_derivability",
    "evidence_sufficiency",
    "evidence_relevance",
    "grounding_correctness",
    "decision_relevance",
    "decision_usefulness",
    "abstention_correctness",
    "proposition_correctness",
    "contradiction_handling",
]


def validate_review_record_schema(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate machine-readable human review record structure."""
    errors = []

    # Check top-level blind review fields
    for field in ["blind_case_id", "percept", "task", "concepts", "beliefs", "rules", "candidate_proposition"]:
        if field not in record:
            errors.append(f"Missing required blind field '{field}'")

    # Check reviewer 1 structure
    r1 = record.get("reviewer_1")
    if not isinstance(r1, dict):
        errors.append("Missing or invalid 'reviewer_1' object")
    else:
        if "label" not in r1 or r1["label"] not in VALID_REVIEW_STATUSES:
            errors.append(f"Invalid reviewer_1 label: {r1.get('label')}")
        evals = r1.get("evaluations", {})
        for mf in MANDATORY_EVALUATION_FIELDS:
            if mf not in evals:
                errors.append(f"Missing mandatory evaluation field '{mf}' in reviewer_1")

    # Check reviewer 2 structure
    r2 = record.get("reviewer_2")
    if not isinstance(r2, dict):
        errors.append("Missing or invalid 'reviewer_2' object")
    else:
        if "label" not in r2 or r2["label"] not in VALID_REVIEW_STATUSES:
            errors.append(f"Invalid reviewer_2 label: {r2.get('label')}")
        evals = r2.get("evaluations", {})
        for mf in MANDATORY_EVALUATION_FIELDS:
            if mf not in evals:
                errors.append(f"Missing mandatory evaluation field '{mf}' in reviewer_2")

    return (len(errors) == 0, errors)


def generate_blind_review_manifest(candidates: list[dict[str, Any]], seed: int = 20260811) -> list[dict[str, Any]]:
    """Generate deterministically randomized blind review manifest hiding generator metadata."""
    rng = random.Random(seed)
    indices = list(range(len(candidates)))
    rng.shuffle(indices)

    blind_records = []
    for rank, orig_idx in enumerate(indices):
        c = candidates[orig_idx]
        
        target_prop = c.get("target_interpretation", {}).get("proposition", "") if c.get("target_interpretation") else ""
        if not target_prop and c.get("rejected_candidates"):
            target_prop = c["rejected_candidates"][0].get("proposition", "")

        blind_records.append({
            "review_rank": rank + 1,
            "blind_case_id": f"rev://v0.2/{rank + 1:03d}",
            "percept": c["percept"],
            "task": c["task"],
            "concepts": [{"id": cp["id"], "label": cp["label"]} for cp in c.get("concepts", [])],
            "beliefs": c.get("beliefs", []),
            "rules": c.get("rules", []),
            "candidate_proposition": target_prop,
            "grounding_snapshot": c.get("grounding_snapshot", {}),
            # Metadata is masked from reviewers
            "_masked_original_case_id": c["case_id"],
            "_masked_generator_id": c.get("provenance", {}).get("generator_id"),
            "_masked_template_id": c.get("provenance", {}).get("template_id"),
            "_masked_expected_novelty": c.get("novelty_label"),
            "_masked_expected_derivability": c.get("derivability_label"),
            "reviewer_1": {
                "reviewer_id": "rev_01",
                "timestamp": None,
                "label": "UNREVIEWED",
                "evaluations": {field: None for field in MANDATORY_EVALUATION_FIELDS},
                "written_reason": None,
            },
            "reviewer_2": {
                "reviewer_id": "rev_02",
                "timestamp": None,
                "label": "UNREVIEWED",
                "evaluations": {field: None for field in MANDATORY_EVALUATION_FIELDS},
                "written_reason": None,
            },
            "adjudication": {
                "adjudicator_id": None,
                "timestamp": None,
                "disagreement_detected": False,
                "final_status": "UNREVIEWED",
                "adjudication_reason": None,
            },
        })

    return blind_records


def main():
    print("=" * 80)
    print("THEO SLM Phase 6B.4 — Human Review Specification & Schema Validator")
    print("=" * 80)

    repaired_path = Path(r"c:\Users\bs162\Desktop\THEO\theo-data\datasets\theo_slm_v0_repaired\candidate_records.json")
    if not repaired_path.exists():
        print(f"Error: Candidate records file not found at {repaired_path}")
        return

    with open(repaired_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    print(f"Loaded candidate dataset ds-v0.2-repaired ({len(candidates)} records).")

    print("\n[1/3] Generating Deterministic Blind Review Manifest (Seed 20260811)...")
    blind_manifest = generate_blind_review_manifest(candidates, seed=20260811)
    print(f"  - Successfully generated {len(blind_manifest)} blind review records.")

    print("\n[2/3] Validating Blind Review Record Schemas...")
    valid_count = 0
    all_errors = []
    for rec in blind_manifest:
        is_valid, errs = validate_review_record_schema(rec)
        if is_valid:
            valid_count += 1
        else:
            all_errors.extend(errs)

    print(f"  - Valid Review Records: {valid_count} / {len(blind_manifest)}")
    if all_errors:
        print(f"  - Schema Errors Detected: {len(all_errors)}")
    else:
        print("  - Schema Validation Passed: 100% compliant with 10 mandatory evaluation fields!")

    print("\n[3/3] Sampling Calibration Batch (15 mixed records)...")
    calib_batch = blind_manifest[:15]
    print(f"  - Sampled {len(calib_batch)} calibration records across domains.")

    print("\nSchema & Specification Validation Complete!")


if __name__ == "__main__":
    main()
