"""ConstraintEngine — system-wide invariant enforcement engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.symbolic._graph.traversal import CycleDetector
from theo_core.symbolic._primitives.errors import ConstraintViolationError
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.constraints.models import (
    ConstraintId,
    ConstraintRule,
    ConstraintSeverity,
    ConstraintViolation,
)

if TYPE_CHECKING:
    from theo_core.symbolic.beliefs.graph import BeliefGraph
    from theo_core.symbolic.concepts.graph import ConceptGraph
    from theo_core.symbolic.thoughts.graph import ThoughtGraph


class ConstraintEngine:
    """Deterministic constraint validation engine enforcing Canon invariants.

    Complexity Contract:
        Time: O(V + E)
        Memory: O(V)
        Deterministic: YES
    """

    @staticmethod
    def validate_all(
        _concepts: ConceptGraph,
        beliefs: BeliefGraph,
        thoughts: ThoughtGraph,
        custom_rules: list[ConstraintRule] | None = None,
    ) -> list[ConstraintViolation]:
        """Validate system invariants across all symbolic graphs.

        Custom rules declare a named check via ``metadata["predicate"]``. Known
        predicates are evaluated deterministically; unknown predicates produce an
        ADVISORY violation so misconfiguration fails loudly rather than silently.

        Args:
            _concepts: Active ConceptGraph.
            beliefs: Active BeliefGraph.
            thoughts: Active ThoughtGraph.
            custom_rules: Optional list of additional ConstraintRules.

        Returns:
            List of detected ConstraintViolation objects.

        """
        violations: list[ConstraintViolation] = []

        # 1. Invariant Check: Thought Graph DAG integrity (no cycles allowed)
        if CycleDetector.has_cycle(thoughts.raw_graph):
            cid = ConstraintId.of("constraint://canon/thought_dag_cycle")
            violations.append(
                ConstraintViolation(
                    constraint_id=cid,
                    target_id=SymbolicId.of("thought://dag"),
                    reason="ThoughtGraph contains a cycle violating DAG invariant.",
                    severity=ConstraintSeverity.FATAL,
                )
            )

        # 2. Invariant Check: Belief grounded provenance (Canon Law 4)
        for belief in beliefs.get_active_beliefs():
            if not belief.source:
                cid = ConstraintId.of("constraint://canon/belief_source_required")
                violations.append(
                    ConstraintViolation(
                        constraint_id=cid,
                        target_id=belief.id.to_symbolic_id(),
                        reason=f"Belief {belief.id} has no valid BeliefSource.",
                        severity=ConstraintSeverity.WARNING,
                    )
                )

        # 3. Custom rules validation (if supplied)
        if custom_rules:
            for rule in sorted(custom_rules, key=lambda r: r.id.value):
                predicate = rule.metadata.get("predicate")
                if predicate == "no_thought_graph_cycles":
                    if CycleDetector.has_cycle(thoughts.raw_graph):
                        violations.append(
                            ConstraintViolation(
                                constraint_id=rule.id,
                                target_id=SymbolicId.of("thought://dag"),
                                reason=rule.description,
                                severity=rule.severity,
                            )
                        )
                elif predicate == "beliefs_have_sources":
                    ungrounded = [b for b in beliefs.get_active_beliefs() if not b.source]
                    if ungrounded:
                        violations.append(
                            ConstraintViolation(
                                constraint_id=rule.id,
                                target_id=ungrounded[0].id.to_symbolic_id(),
                                reason=rule.description,
                                severity=rule.severity,
                            )
                        )
                elif predicate is None:
                    violations.append(
                        ConstraintViolation(
                            constraint_id=rule.id,
                            target_id=rule.id.to_symbolic_id(),
                            reason=f"Custom rule {rule.name!r} has no predicate defined.",
                            severity=ConstraintSeverity.ADVISORY,
                        )
                    )
                else:
                    violations.append(
                        ConstraintViolation(
                            constraint_id=rule.id,
                            target_id=rule.id.to_symbolic_id(),
                            reason=f"Unknown custom predicate {predicate!r} on rule {rule.name!r}.",
                            severity=ConstraintSeverity.ADVISORY,
                        )
                    )

        return sorted(violations, key=lambda v: (v.severity.value, v.constraint_id.value))

    @staticmethod
    def assert_valid(
        _concepts: ConceptGraph,
        beliefs: BeliefGraph,
        thoughts: ThoughtGraph,
        custom_rules: list[ConstraintRule] | None = None,
    ) -> None:
        """Validate and raise on any FATAL violation.

        Args:
            _concepts: Active ConceptGraph.
            beliefs: Active BeliefGraph.
            thoughts: Active ThoughtGraph.
            custom_rules: Optional list of additional ConstraintRules.

        Raises:
            ConstraintViolationError: If any FATAL violation is detected.

        """
        fatal = [
            v
            for v in ConstraintEngine.validate_all(
                _concepts, beliefs, thoughts, custom_rules=custom_rules
            )
            if v.severity == ConstraintSeverity.FATAL
        ]
        if fatal:
            reasons = "; ".join(f"{v.constraint_id.value}: {v.reason}" for v in fatal)
            raise ConstraintViolationError(reasons)
