"""Port contract tests — GapAnalyzer bucketing (ADR-0028)."""

from __future__ import annotations

from decimal import Decimal

from theo_core.models.ports.snapshots import BenchmarkFailureSnapshot
from theo_core.symbolic.analysis.gap_analyzer import GapAnalyzer


class TestGapAnalyzer:
    def test_empty_input_yields_empty_report(self) -> None:
        report = GapAnalyzer().analyze(failures=())
        assert report.benchmark_failures == ()
        assert report.weak_rule_coverage == ()
        assert report.missing_premises == ()

    def test_decision_mismatch_buckets_missing_rules_and_premises(self) -> None:
        failure = BenchmarkFailureSnapshot(
            case_id="case://01",
            domain="causal_reasoning",
            failure_type="decision_mismatch",
            expected_decision="expected",
            actual_decision="actual",
            missing_rules=("rule://r1",),
            missing_premises=("premise://p1",),
        )
        report = GapAnalyzer().analyze(failures=(failure,))
        assert len(report.weak_rule_coverage) == 1
        assert len(report.missing_premises) == 1
        assert report.weak_rule_coverage[0].domain == "causal_reasoning"
        assert report.weak_rule_coverage[0].severity == "high"

    def test_confidence_mismatch_buckets_low_confidence_regions(self) -> None:
        failure = BenchmarkFailureSnapshot(
            case_id="case://02",
            domain="uncertainty",
            failure_type="confidence_mismatch",
            expected_decision="expected",
            actual_decision="actual",
            confidence_delta=Decimal("0.20"),
        )
        report = GapAnalyzer().analyze(failures=(failure,))
        assert len(report.low_confidence_regions) == 1
        assert "0.20" in report.low_confidence_regions[0].description

    def test_missing_concepts_bucket_retrieval_failures(self) -> None:
        failure = BenchmarkFailureSnapshot(
            case_id="case://03",
            domain="taxonomy",
            failure_type="decision_mismatch",
            expected_decision="expected",
            actual_decision="actual",
            missing_concepts=("concept://c1",),
        )
        report = GapAnalyzer().analyze(failures=(failure,))
        assert len(report.retrieval_failures) == 1
        assert report.retrieval_failures[0].severity == "high"
