"""BenchmarkHarness — deterministic execution of BenchmarkCases against the symbolic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from theo_core.evaluation.benchmark_schema import BenchmarkCase, GoldenTrace
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline

if TYPE_CHECKING:
    from theo_core.runtime.providers.coordinator import ProviderCoordinator
    from theo_core.symbolic.decisions.models import DecisionRecord


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Outcome of executing a single benchmark case."""

    case_id: str
    passed: bool
    failures: tuple[str, ...] = field(default_factory=tuple)
    decision: DecisionRecord | None = None
    golden_trace: GoldenTrace | None = None


class BenchmarkHarness:
    """Runs a BenchmarkCase through the canonical symbolic pipeline and verifies it.

    Checks, in order:
    1. Decision type matches ``expected_decision_type``.
    2. Decision confidence falls within the declared bounds.
    3. Decision action text matches ``expected_action_text``.
    4. Every ``expected_beliefs`` proposition is active post-cycle.
    5. No ``excluded_beliefs`` proposition remains active post-cycle.
    6. Every explicitly-specified GoldenTrace field matches the produced trace
       (fields left at their defaults are not asserted).
    """

    @staticmethod
    def run(
        case: BenchmarkCase,
        coordinator: ProviderCoordinator | None = None,
    ) -> BenchmarkResult:
        """Execute one benchmark case and return its result.

        Args:
            case: The benchmark case definition.
            coordinator: Optional provider hook coordinator (ADR-0028). When
                provided, the pipeline consults provider hooks (Phase 1:
                provenance only, outputs not consumed).

        Returns:
            A BenchmarkResult describing pass/fail and any mismatches.

        """
        failures: list[str] = []

        concepts = ConceptGraph()
        for concept in case.initial_concepts:
            concepts.add_concept(concept)
        for edge in case.initial_concept_edges:
            concepts.add_edge(edge)

        beliefs = BeliefGraph()
        for belief in case.initial_beliefs:
            beliefs.add_belief(belief)
        for belief_edge in case.initial_belief_edges:
            beliefs.add_edge(belief_edge)

        pipeline = SymbolicCognitivePipeline(
            concepts=concepts,
            beliefs=beliefs,
            rules=list(case.rules),
            coordinator=coordinator,
        )
        decision, _trace, golden_trace = pipeline.execute_cycle(case.percept_input)

        if decision.type.value != case.expected_decision_type:
            failures.append(
                f"decision.type: expected {case.expected_decision_type!r}, "
                f"got {decision.type.value!r}"
            )

        if not (case.min_confidence <= decision.confidence <= case.max_confidence):
            failures.append(
                f"decision.confidence: {decision.confidence} outside "
                f"[{case.min_confidence}, {case.max_confidence}]"
            )

        if decision.action_text != case.expected_action_text:
            failures.append(
                f"decision.action_text: expected {case.expected_action_text!r}, "
                f"got {decision.action_text!r}"
            )

        active_propositions = [b.proposition for b in pipeline.beliefs.get_active_beliefs()]
        for expected in case.expected_beliefs:
            if expected not in active_propositions:
                failures.append(f"expected_beliefs: {expected!r} not active after cycle")

        for excluded in case.excluded_beliefs:
            if excluded in active_propositions:
                failures.append(f"excluded_beliefs: {excluded!r} still active after cycle")

        if case.golden_trace is not None:
            failures.extend(
                BenchmarkHarness._golden_trace_mismatches(case.golden_trace, golden_trace)
            )

        return BenchmarkResult(
            case_id=case.id.value,
            passed=not failures,
            failures=tuple(failures),
            decision=decision,
            golden_trace=golden_trace,
        )

    @staticmethod
    def run_all(
        cases: tuple[BenchmarkCase, ...] | list[BenchmarkCase],
        coordinator: ProviderCoordinator | None = None,
    ) -> list[BenchmarkResult]:
        """Execute every case and return results in input order.

        Args:
            cases: Iterable of benchmark cases.
            coordinator: Optional provider hook coordinator (ADR-0028).

        Returns:
            A list of BenchmarkResult instances, one per case.

        """
        return [BenchmarkHarness.run(case, coordinator) for case in cases]

    @staticmethod
    def _golden_trace_mismatches(expected: GoldenTrace, produced: GoldenTrace) -> list[str]:
        """Compare only the fields explicitly specified on the expected trace."""
        defaults = GoldenTrace().model_dump(mode="json")
        expected_dump = expected.model_dump(mode="json")
        produced_dump = produced.model_dump(mode="json")

        failures: list[str] = []
        for key, expected_value in expected_dump.items():
            if defaults[key] == expected_value:
                continue
            produced_value = produced_dump[key]
            if produced_value != expected_value:
                failures.append(
                    f"golden_trace.{key}: expected {expected_value!r}, got {produced_value!r}"
                )
        return failures
