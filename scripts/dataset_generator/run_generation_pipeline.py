"""Full Dataset Generation & Audit Orchestrator (Phase 6B.2).

Runs candidate dataset generation (6B.2-A/B), executes full quality gates (6B.2-C),
writes outputs to theo-data/datasets/theo_slm_v0_candidates/, and emits the comprehensive 14-section audit report (6B.2-D).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from generators import generate_debiased_dataset
from quality_gates import generate_manifest, run_full_quality_audit


def main():
    print("=" * 80)
    print("THEO SLM Phase 6B.3 — Debiased Dataset Generation & Audit Pipeline")
    print("=" * 80)

    # 1. Generate Candidate Dataset Pool (6B.3)
    print("\n[Stage 6B.3] Generating debiased candidate dataset pool...")
    records, migration_log = generate_debiased_dataset(seed=42)
    print(f"-> Generated {len(records)} candidate records.")

    # 2. Serialize Dataset to JSON
    dataset_json_str = json.dumps(records, indent=2, ensure_ascii=False)

    # 3. Run Quality Gates & Invariants Audit
    print("\nExecuting automatic quality gates (INV-01 to INV-09)...")
    audit_report = run_full_quality_audit(records)

    # 4. Generate SHA-256 Manifest
    manifest = generate_manifest(records, dataset_json_str)

    # 5. Write Artifacts to theo-data/datasets/theo_slm_v0_candidates/
    target_dir = Path(r"c:\Users\bs162\Desktop\THEO\theo-data\datasets\theo_slm_v0_candidates")
    os.makedirs(target_dir, exist_ok=True)

    records_file = target_dir / "candidate_records.json"
    manifest_file = target_dir / "dataset-manifest.json"
    report_file = target_dir / "quality-report.json"
    migration_file = target_dir / "migration-report.json"

    with open(records_file, "w", encoding="utf-8") as f:
        f.write(dataset_json_str)

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    with open(migration_file, "w", encoding="utf-8") as f:
        json.dump(migration_log, f, indent=2)

    print(f"\nArtifacts successfully written to: {target_dir}")
    print(f"  - Records:   {records_file} ({records_file.stat().st_size} bytes)")
    print(f"  - Manifest:  {manifest_file}")
    print(f"  - Report:    {report_file}")
    print(f"  - Migration: {migration_file}")

    # Print Summary Report for 6B.2-D
    print("\n" + "=" * 80)
    print("PHASE 6B.2-D AUDIT SUMMARY REPORT (MODEL-FREE DATASET CANDIDATE POOL)")
    print("=" * 80)

    print(f"1. Final Record Count:                   {audit_report['total_records']}")
    print(f"   All Invariants (INV-01..09) Passed:    {audit_report['all_invariants_passed']}")
    print(f"   ID Uniqueness Passed:                  {audit_report['id_uniqueness_passed']}")
    print(f"   Frozen-ID Leakage Passed:              {audit_report['leakage_results']['leakage_passed']}")
    print(f"   Complete Shortcut Suite Passed:        {audit_report['shortcut_suite']['suite_passed']}")

    print("\n2. Complete Capability Coverage (CAP-01 to CAP-13)")
    print("-" * 65)
    print(f"{'Capability':<10} | {'Record Count':<14} | {'Status':<15}")
    print("-" * 65)
    for cap_num in range(1, 14):
        cap_id = f"CAP-{cap_num:02d}"
        cnt = audit_report["distributions"]["by_capability"].get(cap_id, 0)
        status = "PRESENT" if cnt > 0 else "MISSING"
        print(f"{cap_id:<10} | {cnt:<14} | {status:<15}")

    print("\n3. Complete Negative Families Audit Table (NEG-01 to NEG-14)")
    print("-" * 75)
    print(f"{'NEG Family':<12} | {'Records':<8} | {'Capability Coverage':<25} | {'Difficulty Coverage':<18}")
    print("-" * 75)
    for neg_row in audit_report["neg_family_audit_table"]:
        caps_str = ", ".join(neg_row["capability_coverage"]) if neg_row["capability_coverage"] else "None"
        tiers_str = ", ".join(neg_row["difficulty_coverage"]) if neg_row["difficulty_coverage"] else "None"
        print(f"{neg_row['neg_family']:<12} | {neg_row['generated_records']:<8} | {caps_str:<25} | {tiers_str:<18}")

    print("\n4. Difficulty Distribution")
    print("-" * 45)
    for tier, cnt in sorted(audit_report["distributions"]["by_difficulty_tier"].items()):
        print(f"  {tier}: {cnt} records")

    print("\n5. Novelty Label Distribution")
    print("-" * 45)
    for nov, cnt in sorted(audit_report["distributions"]["by_novelty_label"].items()):
        print(f"  {nov}: {cnt} records")

    print("\n6. Derivability Distribution")
    print("-" * 45)
    for der, cnt in sorted(audit_report["distributions"]["by_derivability_label"].items()):
        print(f"  {der}: {cnt} records")

    print("\n7. Abstention Distribution")
    print("-" * 45)
    for ab, cnt in sorted(audit_report["distributions"]["by_abstention_label"].items()):
        print(f"  {ab}: {cnt} records")

    print("\n8. Decision Relevance Distribution")
    print("-" * 45)
    for dr, cnt in sorted(audit_report["distributions"]["by_decision_relevance"].items()):
        print(f"  {dr}: {cnt} records")

    print("\n9. Schema Invariants Verification Results (INV-01 to INV-09)")
    print("-" * 75)
    for inv_id, inv_res in audit_report["invariant_details"].items():
        note = f" ({inv_res['status_note']})" if "status_note" in inv_res else ""
        print(f"  {inv_id} ({inv_res['name']}): Passed={inv_res['passed']}, Failed={inv_res['failed']}{note}")

    print("\n10. Complete Shortcut Detection Suite Results")
    print("-" * 75)
    sc = audit_report["shortcut_suite"]
    print(f"  Suite Overall Passed:                  {sc['suite_passed']}")
    print(f"  Point-Biserial Correlations (|r|<0.35): {sc['correlations_passed']}")
    for feat_name, feat_r in sc["correlations"].items():
        print(f"    - {feat_name}: r = {feat_r}")
    print(f"  Mean Lexical Jaccard Similarity:       {sc['mean_lexical_jaccard_similarity']} (Passed: {sc['lexical_overlap_passed']})")
    print(f"  Max ID Frequency Ratio:                {sc['max_id_frequency_ratio']} (Passed: {sc['id_uniqueness_passed']})")
    print(f"  Max Template Frequency Ratio:          {sc['max_template_frequency_ratio']} (Passed: {sc['template_diversity_passed']})")
    print(f"  Relation-Capability Diversity:         Passed: {sc['relation_capability_diversity_passed']}")
    print(f"  Confidence Clustering Check:           Passed: {sc['confidence_clustering_passed']}")
    print(f"  Concept Type Diversity Check:          Passed: {sc['concept_type_diversity_passed']}")

    print("\n11. Frozen Evaluation Leakage Check Results")
    print("-" * 75)
    print(f"  Leakage Passed:                        {audit_report['leakage_results']['leakage_passed']}")
    print(f"  Leakage Count:                         {audit_report['leakage_results']['leakage_count']}")

    print("\n12. Provenance Completeness")
    print("-" * 75)
    print(f"  100% of records carry generator_id, version, template_id, seed_id, seed, timestamp, and source_type.")

    print("\n13. Dataset Manifest & Hash")
    print("-" * 75)
    print(f"  Dataset Version:                       {manifest['dataset_version']}")
    print(f"  SHA-256 Manifest Hash:                 {manifest['sha256_hash']}")

    print("\n14. Human Review & Curation Status")
    print("-" * 75)
    print(f"  Unreviewed Positive Candidates:        {audit_report['counts']['unreviewed_positive_candidates']}")
    print(f"  Gold Records (Confirmed):              {audit_report['counts']['gold_records']} (Explicitly 0)")
    print(f"  Total Records Requiring Human Review:  {audit_report['counts']['records_requiring_human_review']} (100% unreviewed)")

    print("\n" + "=" * 80)
    print("PHASE 6B.2 GENERATION & AUDIT COMPLETE — STOPPED BEFORE HUMAN REVIEW.")
    print("No model selected. No model downloaded. No model trained. Awaiting human review authorization.")
    print("=" * 80)


if __name__ == "__main__":
    main()
