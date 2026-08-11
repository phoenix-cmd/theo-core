"""Orchestrator for Phase 6B.3-D Repaired Dataset Generation & Quality Audit.

Generates `ds-v0.2-repaired` revision into `theo-data/datasets/theo_slm_v0_repaired/`.
Preserves `theo-data/datasets/theo_slm_v0_candidates/` as an immutable forensic checkpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from quality_gates import generate_manifest, run_full_quality_audit
from repaired_generator import generate_repaired_dataset


def main():
    print("=" * 80)
    print("THEO SLM Phase 6B.3-D — Repaired Dataset Generation & Audit Pipeline")
    print("=" * 80)

    # 1. Generate Repaired Dataset Pool (ds-v0.2-repaired)
    print("\n[Stage 6B.3-D] Generating repaired candidate dataset pool...")
    records, migration_log = generate_repaired_dataset(seed=42)
    print(f"-> Generated {len(records)} candidate records.")

    # 2. Serialize Dataset to JSON
    dataset_json_str = json.dumps(records, indent=2, ensure_ascii=False)

    # 3. Run Quality Gates & Invariants Audit
    print("\nExecuting automatic quality gates (INV-01 to INV-09)...")
    audit_report = run_full_quality_audit(records)

    # 4. Generate SHA-256 Manifest
    manifest = generate_manifest(records, dataset_json_str)

    # 5. Write Artifacts to NEW revision directory: theo-data/datasets/theo_slm_v0_repaired/
    target_dir = Path(r"c:\Users\bs162\Desktop\THEO\theo-data\datasets\theo_slm_v0_repaired")
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

    print(f"\nRepaired artifacts successfully written to: {target_dir}")
    print(f"  - Records:   {records_file} ({records_file.stat().st_size} bytes)")
    print(f"  - Manifest:  {manifest_file}")
    print(f"  - Report:    {report_file}")
    print(f"  - Migration: {migration_file}")


if __name__ == "__main__":
    main()
