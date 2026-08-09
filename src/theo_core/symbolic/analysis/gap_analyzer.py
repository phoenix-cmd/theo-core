"""Gap analysis (ADR-0028) — theo-core owned.

``GapAnalyzer`` consumes benchmark outcomes and emits a ``KnowledgeGapReport``
structured by remediation action. It never exposes raw runtime state. The
bucketing below is a deterministic Phase 0 default; gap-detection heuristics are
deliberately out of ADR-0028 scope (see the v0.5 implementation plan).
"""

from __future__ import annotations

from theo_core.models.ports.snapshots import (
    BenchmarkFailureSnapshot,
    GapItem,
    KnowledgeGapReport,
)


class GapAnalyzer:
    """Groups benchmark failures into remediation-oriented gap sections."""

    def analyze(
        self,
        failures: tuple[BenchmarkFailureSnapshot, ...],
    ) -> KnowledgeGapReport:
        """Bucket benchmark failures into ``KnowledgeGapReport`` sections.

        Args:
            failures: Canonical failure snapshots from a benchmark run.

        Returns:
            A ``KnowledgeGapReport`` grouped by remediation action. Empty input
            yields an empty report.

        """
        missing_premises: list[GapItem] = []
        weak_rule_coverage: list[GapItem] = []
        unresolved_ambiguities: list[GapItem] = []
        low_confidence_regions: list[GapItem] = []
        retrieval_failures: list[GapItem] = []
        contradiction_patterns: list[GapItem] = []

        for failure in failures:
            references = self._references(failure)
            if failure.failure_type == "decision_mismatch":
                if failure.missing_rules:
                    weak_rule_coverage.append(
                        GapItem(
                            description=(
                                f"{failure.case_id} needs rules: "
                                + ", ".join(failure.missing_rules)
                            ),
                            domain=failure.domain,
                            severity=self._severity(failure),
                            references=references,
                        )
                    )
                if failure.missing_premises:
                    missing_premises.append(
                        GapItem(
                            description=(
                                f"{failure.case_id} missing premises: "
                                + ", ".join(failure.missing_premises)
                            ),
                            domain=failure.domain,
                            severity=self._severity(failure),
                            references=references,
                        )
                    )
            elif failure.failure_type == "confidence_mismatch":
                low_confidence_regions.append(
                    GapItem(
                        description=(
                            f"{failure.case_id} confidence mismatch "
                            f"(delta={self._format_delta(failure)})"
                        ),
                        domain=failure.domain,
                        severity=self._severity(failure),
                        references=references,
                    )
                )
            elif failure.failure_type == "trace_mismatch":
                unresolved_ambiguities.append(
                    GapItem(
                        description=f"{failure.case_id} trace divergence",
                        domain=failure.domain,
                        severity=self._severity(failure),
                        references=references,
                    )
                )
            elif failure.failure_type == "state_mismatch":
                contradiction_patterns.append(
                    GapItem(
                        description=f"{failure.case_id} state divergence",
                        domain=failure.domain,
                        severity=self._severity(failure),
                        references=references,
                    )
                )
            if failure.missing_concepts:
                retrieval_failures.append(
                    GapItem(
                        description=(
                            f"{failure.case_id} missing concepts: "
                            + ", ".join(failure.missing_concepts)
                        ),
                        domain=failure.domain,
                        severity=self._severity(failure),
                        references=references,
                    )
                )

        return KnowledgeGapReport(
            missing_premises=tuple(missing_premises),
            weak_rule_coverage=tuple(weak_rule_coverage),
            unresolved_ambiguities=tuple(unresolved_ambiguities),
            low_confidence_regions=tuple(low_confidence_regions),
            retrieval_failures=tuple(retrieval_failures),
            contradiction_patterns=tuple(contradiction_patterns),
            benchmark_failures=failures,
        )

    @staticmethod
    def _references(failure: BenchmarkFailureSnapshot) -> tuple[str, ...]:
        return (
            failure.missing_rules
            + failure.missing_concepts
            + failure.missing_premises
        )

    @staticmethod
    def _severity(failure: BenchmarkFailureSnapshot) -> str:
        if failure.failure_type == "decision_mismatch":
            return "high"
        if failure.failure_type in {"trace_mismatch", "state_mismatch"}:
            return "medium"
        return "low"

    @staticmethod
    def _format_delta(failure: BenchmarkFailureSnapshot) -> str:
        delta = failure.confidence_delta
        if delta is None:
            return "unknown"
        return str(delta)
