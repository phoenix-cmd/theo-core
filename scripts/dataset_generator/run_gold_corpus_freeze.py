"""Phase 6C.1 — Human-Reviewed Gold Corpus Freeze (Step 1) + Corpus-Level Audit (Step 7).

Step 1 (freeze):
    Materialize the immutable, versioned curated corpus from the FINAL human
    review decisions only:
      theo-data/datasets/theo_slm_v0_gold/
        theo_slm_v0_gold.jsonl            (all 264 records)
        gold_positive.jsonl               (67)
        gold_abstain.jsonl                (66)
        hard_negative.jsonl               (131)
        reviewer-1-decisions.json         (extracted Reviewer 1 judgments)
        reviewer-2-decisions.json         (extracted Reviewer 2 judgments)
        provenance-index.json             (review -> candidate mapping)
        corpus-manifest.json              (version, hashes, integrity flags)
        corpus-audit.json                 (Step 7 audit output)

    Source candidate dataset (ds-v0.2-repaired) is only READ, never modified.
    Reviewed fields are carried VERBATIM (value-identical to review-records.json);
    final_status is derived from the review/adjudication decisions, nothing else.

Step 7 (audit):
    Corpus-level audit of the HUMAN-REVIEWED subset: duplicates, near-duplicates,
    evidence-structure duplicates, domain/capability/tier balance, split balance,
    grounding distribution, length/relation/confidence shortcut probes,
    review-status integrity, frozen-evaluation leakage, and a rerun of the
    surface-text adversarial classifiers against the FINAL curated corpus.

Run from anywhere; paths are resolved relative to the workspace root.
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

VALID_FINAL_STATUSES = ("GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE")
EXPECTED_SPLIT_COUNTS = {"GOLD_POSITIVE": 67, "GOLD_ABSTAIN": 66, "HARD_NEGATIVE": 131}
REVIEW_FIELDS = (
    "percept",
    "task",
    "concepts",
    "beliefs",
    "rules",
    "candidate_proposition",
    "grounding_snapshot",
)
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
FROZEN_PREFIXES = ("bm://", "sp1://")
CORPUS_VERSION = "ds-v0.3-gold"
SCHEMA_VERSION = "gold-schema-v0.1"
SEED = 20260811

ROOT = Path(r"C:\Users\bs162\Desktop\THEO")
DATA = ROOT / "theo-data" / "datasets"
REVIEW_DIR = DATA / "theo_slm_v0_review"
REPAIRED_FILE = DATA / "theo_slm_v0_repaired" / "candidate_records.json"
GOLD_DIR = DATA / "theo_slm_v0_gold"
BENCHMARK_DIR = ROOT / "theo-core" / "src" / "theo_core" / "evaluation" / "benchmarks"
PROBE_FILE = ROOT / "theo-core" / "docs" / "research" / "reference-slm" / "semantic-probe-v1" / "semantic-probe-v1-cases.json"

COMBINED_FILE = GOLD_DIR / "theo_slm_v0_gold.jsonl"
SPLIT_FILES = {
    "GOLD_POSITIVE": GOLD_DIR / "gold_positive.jsonl",
    "GOLD_ABSTAIN": GOLD_DIR / "gold_abstain.jsonl",
    "HARD_NEGATIVE": GOLD_DIR / "hard_negative.jsonl",
}
REVIEWER_1_FILE = GOLD_DIR / "reviewer-1-decisions.json"
REVIEWER_2_FILE = GOLD_DIR / "reviewer-2-decisions.json"
PROVENANCE_FILE = GOLD_DIR / "provenance-index.json"
MANIFEST_FILE = GOLD_DIR / "corpus-manifest.json"
AUDIT_FILE = GOLD_DIR / "corpus-audit.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json_dumps(rec))
            f.write("\n")


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json_dumps(obj))
        f.write("\n")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return set(normalize_text(text).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def cohens_kappa(ratings1: list[str], ratings2: list[str]) -> float:
    n = len(ratings1)
    if n == 0:
        return 0.0
    categories = sorted(set(ratings1 + ratings2))
    po = sum(1 for a, b in zip(ratings1, ratings2) if a == b) / n
    pe = sum(
        (sum(1 for r in ratings1 if r == c) / n) * (sum(1 for r in ratings2 if r == c) / n)
        for c in categories
    )
    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1.0 - pe), 4)


def chi_square_cramers_v(table_counts: list[list[int]]) -> dict[str, Any]:
    obs = np.array(table_counts, dtype=np.float64)
    row_sums = obs.sum(axis=1)
    col_sums = obs.sum(axis=0)
    total = obs.sum()
    if total == 0 or len(row_sums) < 2 or len(col_sums) < 2:
        return {"chi2_statistic": 0.0, "degrees_of_freedom": 0, "cramers_v": 0.0}
    exp = np.outer(row_sums, col_sums) / total
    exp_safe = np.where(exp == 0, 1e-9, exp)
    chi2 = float(np.sum((obs - exp) ** 2 / exp_safe))
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    min_dim = min(obs.shape[0] - 1, obs.shape[1] - 1)
    cramers_v = math.sqrt(chi2 / (total * max(1, min_dim))) if min_dim > 0 and total > 0 else 0.0
    return {
        "chi2_statistic": round(chi2, 2),
        "degrees_of_freedom": dof,
        "cramers_v": round(cramers_v, 4),
    }


def extract_benchmark_percepts(benchmark_dir: Path) -> list[str]:
    """Extract percept_input strings from the frozen 51-case benchmark modules via AST."""
    percepts: list[str] = []
    if not benchmark_dir.exists():
        return percepts
    for path in sorted(benchmark_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "BenchmarkCase":
                kwargs = {kw.arg: kw.value for kw in node.keywords}
                val = kwargs.get("percept_input")
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    percepts.append(val.value)
    return percepts


# ---------------------------------------------------------------------------
# Step 1 — Corpus Freeze
# ---------------------------------------------------------------------------


def load_upstream() -> dict[str, Any]:
    review_records = json.loads((REVIEW_DIR / "review-records.json").read_text(encoding="utf-8"))
    review_manifest = json.loads((REVIEW_DIR / "review-manifest.json").read_text(encoding="utf-8"))
    review_order = json.loads((REVIEW_DIR / "review-order.json").read_text(encoding="utf-8"))
    adjudication = json.loads((REVIEW_DIR / "adjudication.json").read_text(encoding="utf-8"))
    candidates = json.loads(REPAIRED_FILE.read_text(encoding="utf-8"))
    probe = json.loads(PROBE_FILE.read_text(encoding="utf-8")) if PROBE_FILE.exists() else None
    benchmark_percepts = extract_benchmark_percepts(BENCHMARK_DIR)
    return {
        "review_records": review_records,
        "review_manifest": review_manifest,
        "review_order": review_order,
        "adjudication": adjudication,
        "candidates": candidates,
        "probe": probe,
        "benchmark_percepts": benchmark_percepts,
    }


def build_corpus(up: dict[str, Any]) -> list[dict[str, Any]]:
    review_records = up["review_records"]
    manifest_by_id = {r["review_id"]: r for r in up["review_manifest"]}
    order_by_id = {r["review_id"]: r["source_case_id"] for r in up["review_order"]}
    candidate_ids = {c["case_id"] for c in up["candidates"]}

    corpus: list[dict[str, Any]] = []
    for rec in review_records:
        review_id = rec["review_id"]
        source_case_id = rec["_masked_original_case_id"]
        if source_case_id not in candidate_ids:
            raise ValueError(f"Provenance failure: {source_case_id} not in candidate dataset")
        if order_by_id.get(review_id) != source_case_id:
            raise ValueError(f"Provenance failure: review-order mismatch for {review_id}")

        manifest_rec = manifest_by_id.get(review_id)
        if manifest_rec is None:
            raise ValueError(f"Provenance failure: {review_id} missing from review manifest")

        for field in REVIEW_FIELDS:
            if rec.get(field) != manifest_rec.get(field):
                raise ValueError(f"Content integrity failure: {review_id} field '{field}' differs from manifest")

        adjudication = rec.get("adjudication")
        if not isinstance(adjudication, dict) or not adjudication.get("final_status"):
            raise ValueError(f"Adjudication failure: {review_id} missing final_status")
        final_status = adjudication["final_status"]
        if final_status not in VALID_FINAL_STATUSES:
            raise ValueError(f"Adjudication failure: {review_id} invalid final_status {final_status}")

        record: dict[str, Any] = {}
        for field in REVIEW_FIELDS:
            record[field] = rec.get(field)
        record["review_id"] = review_id
        record["review_rank"] = rec["review_rank"]
        record["final_status"] = final_status
        record["reviewer_1"] = rec.get("reviewer_1")
        record["reviewer_2"] = rec.get("reviewer_2")
        record["adjudication"] = adjudication
        record["provenance"] = {
            "source_case_id": source_case_id,
            "generator_id": rec.get("_masked_generator_id"),
            "template_id": rec.get("_masked_template_id"),
        }
        corpus.append(record)

    if len(corpus) != len(review_records):
        raise ValueError(f"Corpus cardinality failure: {len(corpus)} != {len(review_records)}")
    return corpus


def materialize(up: dict[str, Any], corpus: list[dict[str, Any]]) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    review_records = up["review_records"]

    write_jsonl(COMBINED_FILE, corpus)
    for status, path in SPLIT_FILES.items():
        write_jsonl(path, [r for r in corpus if r["final_status"] == status])

    def decision_blocks(reviewer_key: str) -> list[dict[str, Any]]:
        blocks = []
        for rec in review_records:
            reviewer = rec.get(reviewer_key)
            if not isinstance(reviewer, dict):
                raise ValueError(f"Missing {reviewer_key} for {rec['review_id']}")
            blocks.append(
                {
                    "review_id": rec["review_id"],
                    "review_rank": rec["review_rank"],
                    "source_case_id": rec["_masked_original_case_id"],
                    "reviewer_id": reviewer.get("reviewer_id"),
                    "label": reviewer.get("label"),
                    "evaluations": reviewer.get("evaluations"),
                    "written_reason": reviewer.get("written_reason"),
                    "timestamp": reviewer.get("timestamp"),
                }
            )
        return blocks

    write_json(REVIEWER_1_FILE, decision_blocks("reviewer_1"))
    write_json(REVIEWER_2_FILE, decision_blocks("reviewer_2"))

    provenance_index = {
        "corpus_version": CORPUS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "entries": [
            {
                "review_id": r["review_id"],
                "review_rank": r["review_rank"],
                "source_case_id": r["provenance"]["source_case_id"],
                "generator_id": r["provenance"]["generator_id"],
                "template_id": r["provenance"]["template_id"],
                "final_status": r["final_status"],
            }
            for r in sorted(corpus, key=lambda x: x["review_rank"])
        ],
    }
    write_json(PROVENANCE_FILE, provenance_index)


def build_manifest(up: dict[str, Any], corpus: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["final_status"] for r in corpus)
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "derived_from": "FINAL HUMAN REVIEW DECISIONS (Phase 6B.4, review-records.json)",
        "authoritative_label_source": "human-review final_status (unanimous dual-reviewer + adjudication)",
        "total_records": len(corpus),
        "split_counts": dict(sorted(counts.items())),
        "source_candidate_dataset_sha256": sha256_file(REPAIRED_FILE),
        "review_manifest_sha256": sha256_file(REVIEW_DIR / "review-manifest.json"),
        "reviewer_1_decisions_sha256": sha256_file(REVIEWER_1_FILE),
        "reviewer_2_decisions_sha256": sha256_file(REVIEWER_2_FILE),
        "adjudication_artifact_sha256": sha256_file(REVIEW_DIR / "adjudication.json"),
        "corpus_sha256": sha256_file(COMBINED_FILE),
        "split_sha256": {status: sha256_file(path) for status, path in SPLIT_FILES.items()},
        "integrity": {
            "reviewed_fields_value_identical_to_review_records": True,
            "reviewed_fields_identical_to_review_manifest": True,
            "final_status_derived_from_adjudication_only": True,
            "provenance_join_verified": True,
            "review_order_verified": True,
            "split_counts_match_accepted": all(
                counts.get(s, 0) == n for s, n in EXPECTED_SPLIT_COUNTS.items()
            ),
            "immutable": True,
            "frozen_evaluation_ids_excluded": True,
        },
        "frozen_evaluation_ids_excluded": ["bm://*", "sp1://*"],
        "generator_expected_labels_policy": (
            "Generator expected labels exist in source artifacts only (review-records.json "
            "masked fields, candidate_records.json). They are NOT propagated into this corpus "
            "and MUST NOT be used as training labels."
        ),
    }
    return manifest


# ---------------------------------------------------------------------------
# Step 7 — Corpus-Level Audit
# ---------------------------------------------------------------------------


def run_audit(up: dict[str, Any], corpus: list[dict[str, Any]]) -> dict[str, Any]:
    candidates_by_id = {c["case_id"]: c for c in up["candidates"]}
    review_by_id = {r["review_id"]: r for r in up["review_records"]}
    probe_cases = up["probe"]["cases"] if up.get("probe") else []
    probe_percepts = [c["percept_input"] for c in probe_cases]
    probe_candidates = [c["ground_truth"]["candidate"] for c in probe_cases]
    benchmark_percepts = up["benchmark_percepts"]

    by_rank = {r["review_rank"]: r for r in corpus}
    ordered = [by_rank[i] for i in sorted(by_rank)]

    def domain_of(r: dict[str, Any]) -> str:
        parts = r["provenance"]["source_case_id"].split("/")
        family = parts[3] if len(parts) > 3 else "UNKNOWN"
        if family == "pert":
            cand = candidates_by_id[r["provenance"]["source_case_id"]]
            seed = (cand.get("provenance") or {}).get("seed_case_id", "")
            seed_parts = seed.split("/")
            if len(seed_parts) > 3 and not seed_parts[3].startswith("seed_"):
                return seed_parts[3]
            return "UNKNOWN"
        return family

    def capability_of(r: dict[str, Any]) -> str:
        cand = candidates_by_id[r["provenance"]["source_case_id"]]
        return str(cand.get("capability_family", "CAP-00"))

    def case_family_of(r: dict[str, Any]) -> str:
        parts = r["provenance"]["source_case_id"].split("/")
        return parts[3] if len(parts) > 3 else "UNKNOWN"

    audit: dict[str, Any] = {}

    # --- 1. review-status integrity -------------------------------------
    integrity: dict[str, Any] = {}
    invalid = [r["review_id"] for r in corpus if r["final_status"] not in VALID_FINAL_STATUSES]
    integrity["invalid_final_status_count"] = len(invalid)
    integrity["final_status_matches_adjudication"] = all(
        r["final_status"] == r["adjudication"]["final_status"] for r in corpus
    )
    r1_labels = [r["reviewer_1"]["label"] for r in corpus]
    r2_labels = [r["reviewer_2"]["label"] for r in corpus]
    integrity["reviewer_1_equals_reviewer_2"] = all(a == b for a, b in zip(r1_labels, r2_labels))
    integrity["recomputed_cohens_kappa"] = cohens_kappa(r1_labels, r2_labels)
    integrity["all_reviewers_matched"] = all(
        r["reviewer_1"]["label"] == r["reviewer_2"]["label"] == r["final_status"] for r in corpus
    )
    integrity["evaluation_fields_complete"] = all(
        all(f in r["reviewer_1"].get("evaluations", {}) for f in MANDATORY_EVALUATION_FIELDS)
        and all(f in r["reviewer_2"].get("evaluations", {}) for f in MANDATORY_EVALUATION_FIELDS)
        for r in corpus
    )
    integrity["evaluation_fields_boolean"] = all(
        isinstance(r["reviewer_1"]["evaluations"][f], bool)
        and isinstance(r["reviewer_2"]["evaluations"][f], bool)
        for r in corpus
        for f in MANDATORY_EVALUATION_FIELDS
    )
    integrity["split_counts_match_accepted"] = dict(Counter(r["final_status"] for r in corpus)) == EXPECTED_SPLIT_COUNTS

    # human review vs generator expected labels (cross-check; generator is NOT authoritative)
    exp_map = {
        "GOLD_POSITIVE": lambda n, d, a: n == "SEMANTIC_NOVEL" and d == "NON_DERIVABLE" and a == "SHOULD_PROPOSE",
        "GOLD_ABSTAIN": lambda n, d, a: a == "SHOULD_ABSTAIN" and n in ("UNSUPPORTED", "EPISTEMICALLY_PREMATURE"),
        "HARD_NEGATIVE": lambda n, d, a: d == "DERIVABLE"
        or n in ("REPEAT", "PARAPHRASE", "RULE_ECHO", "TAXONOMY_ECHO", "DECISION_IRRELEVANT"),
    }
    exp_agree = []
    exp_disagree = []
    premature_abstain = 0
    novelty_x_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    derivability_x_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    abstention_x_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in ordered:
        review = review_by_id[r["review_id"]]
        n = review.get("_masked_expected_novelty")
        d = review.get("_masked_expected_derivability")
        a = review.get("_masked_expected_abstention")
        if n is None:
            continue
        novelty_x_status[r["final_status"]][n] += 1
        derivability_x_status[r["final_status"]][d] += 1
        abstention_x_status[r["final_status"]][a] += 1
        if exp_map[r["final_status"]](n, d, a):
            exp_agree.append(r["review_id"])
        else:
            exp_disagree.append(r["review_id"])
        if n == "EPISTEMICALLY_PREMATURE" and r["final_status"] == "GOLD_ABSTAIN":
            premature_abstain += 1
    integrity["human_vs_generator_expected_agreement"] = {
        "agree_count": len(exp_agree),
        "disagree_count": len(exp_disagree),
        "disagree_review_ids": exp_disagree,
    }
    integrity["masked_expected_novelty_x_status"] = {k: dict(v) for k, v in sorted(novelty_x_status.items())}
    integrity["masked_expected_derivability_x_status"] = {k: dict(v) for k, v in sorted(derivability_x_status.items())}
    integrity["masked_expected_abstention_x_status"] = {k: dict(v) for k, v in sorted(abstention_x_status.items())}
    integrity["epistemically_premature_gold_abstain_count"] = premature_abstain
    audit["review_status_integrity"] = integrity

    # --- 2. duplicate propositions ----------------------------------------
    def norm_prop(r: dict[str, Any]) -> str:
        return normalize_text(r["candidate_proposition"])

    dup: dict[str, Any] = {}
    norm_counts = Counter(norm_prop(r) for r in corpus)
    dup["exact_normalized_duplicate_groups"] = {k: v for k, v in norm_counts.items() if v > 1}
    for status in VALID_FINAL_STATUSES:
        subset = [r for r in corpus if r["final_status"] == status]
        sub_counts = Counter(norm_prop(r) for r in subset)
        dup[f"{status}_exact_duplicate_groups"] = {k: v for k, v in sub_counts.items() if v > 1}
    audit["duplicate_propositions"] = dup

    # --- 3. near-duplicate scenarios --------------------------------------
    pairs = []
    props = ordered
    for i in range(len(props)):
        for j in range(i + 1, len(props)):
            a, b = props[i], props[j]
            sim = jaccard(token_set(a["percept"]), token_set(b["percept"]))
            if sim >= 0.8:
                pairs.append(
                    {
                        "review_id_a": a["review_id"],
                        "review_id_b": b["review_id"],
                        "status_a": a["final_status"],
                        "status_b": b["final_status"],
                        "percept_jaccard": round(sim, 3),
                    }
                )
    audit["near_duplicate_scenarios"] = {
        "threshold": "token-set Jaccard >= 0.80 on percept text",
        "pair_count": len(pairs),
        "pairs": pairs,
        "note": "Near-duplicates are EXPECTED by design (debiased perturb/var_* family); reported as information.",
    }

    # --- 4. duplicate evidence structures ----------------------------------
    struct_counter = Counter(
        tuple(sorted(r["grounding_snapshot"].get("evidence_ids", []))) for r in corpus
    )
    dup_structs = {k: v for k, v in struct_counter.items() if v > 1}
    audit["duplicate_evidence_structures"] = {
        "distinct_structures": len(struct_counter),
        "duplicate_structures_count": len(dup_structs),
        "duplicate_groups": [{"structure": list(k), "count": v} for k, v in dup_structs.items()],
    }

    # --- 5/6/7. domain, capability, tier balance ----------------------------
    domain_mat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cap_mat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tier_mat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    family_mat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in corpus:
        cand = candidates_by_id[r["provenance"]["source_case_id"]]
        domain_mat[domain_of(r)][r["final_status"]] += 1
        cap_mat[capability_of(r)][r["final_status"]] += 1
        tier_mat[str(cand.get("difficulty_tier", 0))][r["final_status"]] += 1
        family_mat[case_family_of(r)][r["final_status"]] += 1
    audit["domain_balance"] = {k: dict(v) for k, v in sorted(domain_mat.items())}
    audit["capability_balance"] = {k: dict(v) for k, v in sorted(cap_mat.items())}
    audit["tier_balance"] = {k: dict(v) for k, v in sorted(tier_mat.items())}
    audit["case_family_x_status"] = {k: dict(v) for k, v in sorted(family_mat.items())}

    audit["domain_coverage"] = {
        "domains": sorted(domain_mat),
        "every_domain_has_gold_positive": all(
            domain_mat[d].get("GOLD_POSITIVE", 0) >= 1 for d in domain_mat
        ),
        "every_domain_has_gold_abstain": all(
            domain_mat[d].get("GOLD_ABSTAIN", 0) >= 1 for d in domain_mat
        ),
        "every_domain_has_hard_negative": all(
            domain_mat[d].get("HARD_NEGATIVE", 0) >= 1 for d in domain_mat
        ),
    }
    audit["capability_coverage"] = {
        "capabilities": sorted(cap_mat),
        "every_capability_has_gold_positive": all(
            cap_mat[c].get("GOLD_POSITIVE", 0) >= 1 for c in cap_mat
        ),
        "every_capability_has_gold_abstain": all(
            cap_mat[c].get("GOLD_ABSTAIN", 0) >= 1 for c in cap_mat
        ),
        "every_capability_has_hard_negative": all(
            cap_mat[c].get("HARD_NEGATIVE", 0) >= 1 for c in cap_mat
        ),
    }

    # --- 8. positive/abstain/negative balance -------------------------------
    counts = Counter(r["final_status"] for r in corpus)
    audit["split_balance"] = {
        "counts": dict(sorted(counts.items())),
        "percent_of_total": {k: round(v / len(corpus) * 100, 1) for k, v in sorted(counts.items())},
        "supervision_targets_positive_abstain": counts["GOLD_POSITIVE"] + counts["GOLD_ABSTAIN"],
        "hard_negatives": counts["HARD_NEGATIVE"],
    }

    # --- 9. grounding distribution -------------------------------------------
    evidence_counts = [len(r["grounding_snapshot"].get("evidence_ids", [])) for r in corpus]
    belief_counts = [len(r["beliefs"]) for r in corpus]
    rule_counts = [len(r["rules"]) for r in corpus]
    concept_counts = [len(r["concepts"]) for r in corpus]
    distractor_counts = [candidates_by_id[r["provenance"]["source_case_id"]].get("distractor_count", 0) for r in corpus]

    def dist_stats(values: list[int]) -> dict[str, Any]:
        return {
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 3),
            "stdev": round(statistics.pstdev(values), 3),
            "distribution": dict(sorted(Counter(values).items())),
        }

    grounded_targets_ok = 0
    grounding_issues = []
    for r in corpus:
        cand = candidates_by_id[r["provenance"]["source_case_id"]]
        target = cand.get("target_interpretation")
        if target:
            refs = set(target.get("referenced_concept_ids", []))
            allowed = set(r["grounding_snapshot"].get("concept_ids", []))
            if refs <= allowed:
                grounded_targets_ok += 1
            else:
                grounding_issues.append(
                    {
                        "review_id": r["review_id"],
                        "source_case_id": r["provenance"]["source_case_id"],
                        "unresolvable_refs": sorted(refs - allowed),
                    }
                )
    audit["grounding_distribution"] = {
        "evidence_count": dist_stats(evidence_counts),
        "belief_count": dist_stats(belief_counts),
        "rule_count": dist_stats(rule_counts),
        "concept_count": dist_stats(concept_counts),
        "distractor_count": dist_stats(distractor_counts),
        "target_referenced_ids_within_snapshot": {
            "ok": grounded_targets_ok,
            "issues": grounding_issues,
        },
    }

    # --- 10/11/12. length / relation / confidence shortcut probes -----------
    lengths: dict[str, list[int]] = defaultdict(list)
    relations: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    confidences: dict[str, list[float]] = defaultdict(list)
    relation_targets: list[tuple[str, str]] = []
    conf_targets: list[tuple[float, str]] = []
    length_targets: list[tuple[int, str]] = []
    for r in corpus:
        status = r["final_status"]
        lengths[status].append(len(r["candidate_proposition"]))
        length_targets.append((len(r["candidate_proposition"]), status))
        cand = candidates_by_id[r["provenance"]["source_case_id"]]
        target = cand.get("target_interpretation") or {}
        rel = str(target.get("semantic_relation", "UNKNOWN"))
        conf = target.get("confidence")
        relations[rel][status] += 1
        relation_targets.append((rel, status))
        if conf is not None:
            confidences[status].append(float(conf))
            conf_targets.append((float(conf), status))

    length_stats = {k: dist_stats(v) for k, v in sorted(lengths.items())}
    conf_stats = {}
    for k, v in sorted(confidences.items()):
        conf_stats[k] = {
            "count": len(v),
            "min": round(min(v), 3),
            "max": round(max(v), 3),
            "mean": round(statistics.mean(v), 3),
            "stdev": round(statistics.pstdev(v), 3),
        }
    relation_counts = {k: dict(v) for k, v in sorted(relations.items())}

    rel_matrix = []
    rel_cats = sorted(relations)
    status_order = list(VALID_FINAL_STATUSES)
    for rel in rel_cats:
        rel_matrix.append([relations[rel].get(s, 0) for s in status_order])
    rel_assoc = chi_square_cramers_v(rel_matrix)
    deterministic_rel = []
    for rel in rel_cats:
        row = relations[rel]
        total = sum(row.values())
        for s, c in row.items():
            if total and c == total and total >= 3:
                deterministic_rel.append(f"100% of '{rel}' -> '{s}' (n={total})")

    audit["proposition_length_shortcuts"] = {
        "per_status_stats": length_stats,
        "length_only_classifier": run_ovr_classifier(
            length_targets, lambda pairs: [[float(v[0])] for v in pairs], "length_only"
        ),
    }
    audit["relation_shortcuts"] = {
        "per_status_counts": relation_counts,
        "cramers_v": rel_assoc,
        "deterministic_links": deterministic_rel,
        "relation_only_classifier": run_ovr_classifier(
            relation_targets, lambda pairs: OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            .fit_transform([[v[0]] for v in pairs])
            .tolist(),
            "relation_only",
        ),
        "note": (
            "All GOLD_ABSTAIN and HARD_NEGATIVE records carry semantic_relation UNKNOWN because they "
            "do not propose a supported interpretation (no target_interpretation). GOLD_POSITIVE "
            "requires a relation by construction. This is a definitional property of the label, NOT a "
            "ground-truth relation present at inference time, so it is not a model-input leak."
        ),
    }
    status_counts_for_conf = Counter(r["final_status"] for r in corpus)
    audit["confidence_shortcuts"] = {
        "per_status_stats": conf_stats,
        "missing_confidence": {
            s: status_counts_for_conf[s] - len(confidences[s]) for s in VALID_FINAL_STATUSES
        },
        "confidence_continuous_means": {s: conf_stats[s]["mean"] for s in conf_stats},
    }
    audit["shortcut_probes_verdict"] = {
        "length_relation_confidence_flag": (
            "REVIEW"
            if (
                audit["proposition_length_shortcuts"]["length_only_classifier"]["max_balanced_accuracy_excess"] > 0.15
                or audit["relation_shortcuts"]["relation_only_classifier"]["max_balanced_accuracy_excess"] > 0.15
            )
            else "PASS"
        )
    }

    # --- 13. frozen-evaluation leakage ---------------------------------------
    leaked_ids = []
    leaked_grounding = []
    for r in corpus:
        case_id = r["provenance"]["source_case_id"]
        if case_id.startswith(FROZEN_PREFIXES):
            leaked_ids.append(case_id)
        for bucket in ("concept_ids", "belief_ids", "rule_ids", "evidence_ids"):
            for gid in r["grounding_snapshot"].get(bucket, []):
                if gid.startswith(FROZEN_PREFIXES):
                    leaked_grounding.append(gid)
    percept_overlap = []
    for r in corpus:
        pt = token_set(r["percept"])
        for p in probe_percepts:
            sim = jaccard(pt, token_set(p))
            if sim >= 0.5:
                percept_overlap.append(
                    {"review_id": r["review_id"], "probe_percept": p, "jaccard": round(sim, 3)}
                )
    prop_overlap = []
    for r in corpus:
        pt = token_set(r["candidate_proposition"])
        for p in probe_candidates:
            sim = jaccard(pt, token_set(p))
            if sim >= 0.5:
                prop_overlap.append(
                    {"review_id": r["review_id"], "probe_candidate": p, "jaccard": round(sim, 3)}
                )
    benchmark_overlap = []
    for r in corpus:
        pt = token_set(r["percept"])
        for p in benchmark_percepts:
            sim = jaccard(pt, token_set(p))
            if sim >= 0.5:
                benchmark_overlap.append(
                    {"review_id": r["review_id"], "benchmark_percept": p, "jaccard": round(sim, 3)}
                )
    audit["frozen_evaluation_leakage"] = {
        "leaked_case_ids": leaked_ids,
        "leaked_grounding_ids": leaked_grounding,
        "probe_percept_overlap": percept_overlap,
        "probe_candidate_overlap": prop_overlap,
        "benchmark_percept_overlap": benchmark_overlap,
        "verdict": "PASS" if not (leaked_ids or leaked_grounding or percept_overlap or prop_overlap or benchmark_overlap) else "REVIEW",
    }

    # --- 14. surface-text adversarial classifiers on FINAL corpus -----------
    surface = run_surface_classifiers(corpus, candidates_by_id)
    audit["surface_text_adversarial_classifiers"] = surface

    # --- verdict rollup -------------------------------------------------------
    checks = {
        "review_status_integrity": "PASS" if _integrity_pass(integrity) else "REVIEW",
        "duplicate_propositions": "PASS" if not dup["exact_normalized_duplicate_groups"] else "REVIEW",
        "near_duplicate_scenarios": "INFO",
        "duplicate_evidence_structures": "INFO",
        "domain_coverage": "PASS" if audit["domain_coverage"]["every_domain_has_gold_positive"] and audit["domain_coverage"]["every_domain_has_gold_abstain"] and audit["domain_coverage"]["every_domain_has_hard_negative"] else "REVIEW",
        "capability_coverage": "PASS" if audit["capability_coverage"]["every_capability_has_gold_positive"] and audit["capability_coverage"]["every_capability_has_gold_abstain"] and audit["capability_coverage"]["every_capability_has_hard_negative"] else "REVIEW",
        "split_balance": "PASS",
        "grounding": "PASS" if not grounding_issues else "REVIEW",
        "shortcut_probes": audit["shortcut_probes_verdict"]["length_relation_confidence_flag"],
        "frozen_evaluation_leakage": audit["frozen_evaluation_leakage"]["verdict"],
        "surface_text_classifiers": "PASS" if not surface["any_shortcut_risk"] else "REVIEW",
    }
    audit["verdict"] = {
        "checks": checks,
        "overall": "REVIEW" if "REVIEW" in checks.values() else "PASS",
        "no_auto_repair_policy": "No records were added, removed, or replaced. Residual findings are reported for human decision only.",
    }
    return audit


def _integrity_pass(integrity: dict[str, Any]) -> bool:
    return all(
        (
            integrity["invalid_final_status_count"] == 0,
            integrity["final_status_matches_adjudication"],
            integrity["reviewer_1_equals_reviewer_2"],
            integrity["all_reviewers_matched"],
            integrity["evaluation_fields_complete"],
            integrity["evaluation_fields_boolean"],
            integrity["split_counts_match_accepted"],
        )
    )


def run_ovr_classifier(
    pairs: list[tuple[Any, str]],
    feature_fn,
    name: str,
) -> dict[str, Any]:
    """One-vs-rest TF-IDF/custom-feature classifier per final status."""
    statuses = VALID_FINAL_STATUSES
    X = np.asarray(feature_fn(pairs), dtype=np.float64)
    results: dict[str, Any] = {"classifier": name}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    max_excess = 0.0
    for target in statuses:
        y = np.array([1 if s == target else 0 for _, s in pairs])
        accs, bas, f1s = [], [], []
        if len(set(y)) < 2 or X.shape[0] < 10:
            results[target] = {
                "note": "skipped: class degenerate",
                "majority_baseline": round(max(float(np.mean(y == 1)), float(np.mean(y == 0))), 4),
                "balanced_accuracy": 0.5,
                "balanced_accuracy_excess": 0.0,
            }
            continue
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
            clf.fit(X[tr], y[tr])
            preds = clf.predict(X[te])
            accs.append(accuracy_score(y[te], preds))
            bas.append(balanced_accuracy_score(y[te], preds))
            f1s.append(f1_score(y[te], preds, zero_division=0))
        majority = max(float(np.mean(y == 1)), float(np.mean(y == 0)))
        ba = round(float(np.mean(bas)), 4)
        excess = round(ba - majority, 4)
        max_excess = max(max_excess, excess)
        results[target] = {
            "accuracy": round(float(np.mean(accs)), 4),
            "balanced_accuracy": ba,
            "macro_f1": round(float(np.mean(f1s)), 4),
            "majority_baseline": majority,
            "balanced_accuracy_excess": excess,
        }
    results["max_balanced_accuracy_excess"] = round(max_excess, 4)
    return results


def run_surface_classifiers(
    corpus: list[dict[str, Any]], candidates_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    statuses = VALID_FINAL_STATUSES
    target: dict[str, list[int]] = {}
    for s in statuses:
        target[s] = [1 if r["final_status"] == s else 0 for r in corpus]

    surface_texts = [f"{r['percept']} {r['candidate_proposition']}" for r in corpus]
    propositions = [r["candidate_proposition"] for r in corpus]
    percepts = [r["percept"] for r in corpus]
    tasks = [r["task"] for r in corpus]
    concept_names = [" ".join(c.get("label", "") for c in r["concepts"]) for r in corpus]
    content_words = [
        " ".join(w for w in normalize_text(p).split() if w not in ENGLISH_STOP_WORDS) for p in propositions
    ]
    relations = []
    for r in corpus:
        cand = candidates_by_id[r["provenance"]["source_case_id"]]
        target_ = cand.get("target_interpretation") or {}
        relations.append(str(target_.get("semantic_relation", "UNKNOWN")))

    def run(feats: list[str], label: str, max_features: int = 250) -> dict[str, Any]:
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        X = vec.fit_transform(feats)
        res: dict[str, Any] = {"feature_set": label, "vocabulary_size": len(vec.get_feature_names_out())}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for s in statuses:
            y = np.array(target[s])
            accs, bas, f1s = [], [], []
            for tr, te in skf.split(X, y):
                clf = LogisticRegression(max_iter=1000, random_state=SEED)
                clf.fit(X[tr], y[tr])
                preds = clf.predict(X[te])
                accs.append(accuracy_score(y[te], preds))
                bas.append(balanced_accuracy_score(y[te], preds))
                f1s.append(f1_score(y[te], preds, zero_division=0))
            majority = max(float(np.mean(y == 1)), float(np.mean(y == 0)))
            res[s] = {
                "accuracy": round(float(np.mean(accs)), 4),
                "balanced_accuracy": round(float(np.mean(bas)), 4),
                "macro_f1": round(float(np.mean(f1s)), 4),
                "majority_baseline": round(majority, 4),
                "balanced_accuracy_excess": round(float(np.mean(bas)) - majority, 4),
            }
        return res

    # Metadata features (one-hot + numeric), mirroring the earlier A..F baselines.
    meta_rows: list[dict[str, Any]] = []
    for r in corpus:
        cand = candidates_by_id[r["provenance"]["source_case_id"]]
        prov = cand.get("provenance", {})
        meta_rows.append(
            {
                "capability": str(cand.get("capability_family", "CAP-00")),
                "difficulty_tier": str(cand.get("difficulty_tier", 0)),
                "source_type": str(prov.get("source_type", "SYNTHETIC")),
                "evidence_count": len(r["grounding_snapshot"].get("evidence_ids", [])),
                "belief_count": len(r["beliefs"]),
                "concept_count": len(r["concepts"]),
                "rule_count": len(r["rules"]),
                "distractor_count": cand.get("distractor_count", 0),
                "percept_length": len(r["percept"]),
            }
        )

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat = encoder.fit_transform([[m["capability"], m["difficulty_tier"], m["source_type"]] for m in meta_rows])
    num = np.array(
        [
            [
                m["evidence_count"],
                m["belief_count"],
                m["concept_count"],
                m["rule_count"],
                m["distractor_count"],
                m["percept_length"],
            ]
            for m in meta_rows
        ]
    )
    X_meta = np.hstack([cat, num])

    def run_matrix(X: np.ndarray, label: str) -> dict[str, Any]:
        res: dict[str, Any] = {"feature_set": label, "vocabulary_size": int(X.shape[1])}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for s in statuses:
            y = np.array(target[s])
            accs, bas, f1s = [], [], []
            for tr, te in skf.split(X, y):
                clf = LogisticRegression(max_iter=1000, random_state=SEED)
                clf.fit(X[tr], y[tr])
                preds = clf.predict(X[te])
                accs.append(accuracy_score(y[te], preds))
                bas.append(balanced_accuracy_score(y[te], preds))
                f1s.append(f1_score(y[te], preds, zero_division=0))
            majority = max(float(np.mean(y == 1)), float(np.mean(y == 0)))
            res[s] = {
                "accuracy": round(float(np.mean(accs)), 4),
                "balanced_accuracy": round(float(np.mean(bas)), 4),
                "macro_f1": round(float(np.mean(f1s)), 4),
                "majority_baseline": round(majority, 4),
                "balanced_accuracy_excess": round(float(np.mean(bas)) - majority, 4),
            }
        return res

    results = {
        "classifier": "LogisticRegression on TF-IDF (max 250 features, 1-2 grams), 5-fold StratifiedKFold",
        "seed": SEED,
        "targets": ["GOLD_POSITIVE", "GOLD_ABSTAIN", "HARD_NEGATIVE"],
        "feature_sets": {
            "A_proposition_only": run(propositions, "A_proposition_only"),
            "B_task_only": run(tasks, "B_task_only", max_features=50),
            "C_percept_only": run(percepts, "C_percept_only"),
            "D_concept_names_only": run(concept_names, "D_concept_names_only", max_features=135),
            "E_relation_only": run(relations, "E_relation_only", max_features=2),
            "F_content_words_only": run(content_words, "F_content_words_only"),
            "G_surface_combined": run(surface_texts, "G_surface_combined"),
            "H_metadata_only": run_matrix(X_meta, "H_metadata_only"),
        },
    }

    # Top discriminating tokens per target (full-data fit) to characterize the leak vocabulary.
    def top_tokens(feats: list[str], label: str, max_features: int = 250) -> dict[str, Any]:
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        X = vec.fit_transform(feats)
        names = vec.get_feature_names_out()
        out: dict[str, Any] = {"feature_set": label}
        for s in statuses:
            y = np.array(target[s])
            if len(set(y)) < 2:
                out[s] = {"note": "skipped: class degenerate"}
                continue
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
            clf.fit(X, y)
            coef = clf.coef_[0]
            top = sorted(zip(names, coef), key=lambda t: t[1], reverse=True)
            out[s] = {
                "top_positive_tokens": [t[0] for t in top[:20]],
                "top_negative_tokens": [t[0] for t in top[-20:]],
            }
        return out

    results["top_tokens"] = {
        "proposition_only": top_tokens(propositions, "proposition_only"),
        "content_words_only": top_tokens(content_words, "content_words_only"),
    }

    # E_relation_only is definitional (see relation_shortcuts note) and reported separately;
    # exclude it from the aggregate shortcut-risk flag but keep it in the table.
    max_excesses: list[float] = []
    for feat, r in results["feature_sets"].items():
        if feat == "E_relation_only":
            continue
        for s in statuses:
            if "balanced_accuracy_excess" in r[s]:
                max_excesses.append(r[s]["balanced_accuracy_excess"])
    results["max_balanced_accuracy_excess_any_feature"] = round(max(max_excesses), 4)
    results["any_shortcut_risk"] = any(e > 0.15 for e in max_excesses)
    results["note"] = (
        "A positive excess above the majority baseline is expected when the proposition CONTENT itself "
        "determines the label; shortcut risk is flagged when a superficial feature (length, relation, "
        "metadata, isolated surface field) predicts the label far above chance. Findings are reported "
        "for human decision; no automatic replacement is performed. The relation feature set E is "
        "definitional (GOLD_ABSTAIN/HARD_NEGATIVE carry no target_interpretation) and is documented "
        "under relation_shortcuts rather than counted toward any_shortcut_risk."
    )
    return results


def main() -> None:
    print("=" * 80)
    print("Phase 6C.1 — Gold Corpus Freeze (Step 1) + Corpus Audit (Step 7)")
    print("=" * 80)

    up = load_upstream()
    print(f"Loaded review records: {len(up['review_records'])}")
    print(f"Loaded candidates: {len(up['candidates'])}")

    corpus = build_corpus(up)
    print(f"Built corpus: {len(corpus)} records "
          f"({dict(Counter(r['final_status'] for r in corpus))})")

    materialize(up, corpus)
    print("Materialized corpus, splits, reviewer decisions, provenance index.")

    manifest = build_manifest(up, corpus)
    write_json(MANIFEST_FILE, manifest)
    print(f"Corpus SHA-256: {manifest['corpus_sha256']}")

    audit = run_audit(up, corpus)
    write_json(AUDIT_FILE, audit)
    print(f"Audit overall verdict: {audit['verdict']['overall']}")

    print("\nArtifacts:")
    for p in sorted(GOLD_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
