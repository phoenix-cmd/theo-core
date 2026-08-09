"""Snapshot converters from internal symbolic models (ADR-0028).

Theo-core owned adapter that maps internal runtime models to snapshot DTOs,
stripping timestamps, metadata, and mutable lifecycle fields. This module is NOT
part of the provider-visible ``theo_core.models.ports`` surface: providers may
import only the DTOs and protocols, never these adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.models.ports.snapshots import (
    BeliefSnapshot,
    BeliefSnapshotCollection,
    ConceptSnapshot,
    ConceptSnapshotCollection,
    DecisionSnapshot,
    GoalSnapshot,
    GoalSnapshotCollection,
    GroundingSnapshot,
    HypothesisSnapshot,
    HypothesisSnapshotCollection,
    RuleSnapshot,
    RuleSnapshotCollection,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from theo_core.domain.runtime.entities.goal import Goal
    from theo_core.symbolic.beliefs.models import Belief
    from theo_core.symbolic.concepts.models import Concept
    from theo_core.symbolic.decisions.models import DecisionRecord
    from theo_core.symbolic.hypotheses.models import Hypothesis
    from theo_core.symbolic.inference.models import InferenceRule


def belief_to_snapshot(belief: Belief) -> BeliefSnapshot:
    """Convert an internal ``Belief`` to a semantic ``BeliefSnapshot``."""
    return BeliefSnapshot(
        belief_id=belief.id.to_symbolic_id().value,
        proposition=belief.proposition,
        source=belief.source.value,
        confidence=belief.confidence,
    )


def beliefs_to_collection(beliefs: Iterable[Belief]) -> BeliefSnapshotCollection:
    """Convert an iterable of ``Belief`` objects to a snapshot collection."""
    return tuple(belief_to_snapshot(b) for b in beliefs)


def concept_to_snapshot(concept: Concept) -> ConceptSnapshot:
    """Convert an internal ``Concept`` to a semantic ``ConceptSnapshot``."""
    return ConceptSnapshot(
        concept_id=concept.id.to_symbolic_id().value,
        name=concept.label,
        definition=concept.label,
    )


def concepts_to_collection(concepts: Iterable[Concept]) -> ConceptSnapshotCollection:
    """Convert an iterable of ``Concept`` objects to a snapshot collection."""
    return tuple(concept_to_snapshot(c) for c in concepts)


def rule_to_snapshot(rule: InferenceRule) -> RuleSnapshot:
    """Convert an internal ``InferenceRule`` to a semantic ``RuleSnapshot``."""
    premise_text = " AND ".join(c.premise_predicate for c in rule.conditions)
    return RuleSnapshot(
        rule_id=rule.id.to_symbolic_id().value,
        name=rule.name,
        premise_text=premise_text,
        conclusion_text=rule.conclusion_template,
        salience=rule.salience,
    )


def rules_to_collection(rules: Iterable[InferenceRule]) -> RuleSnapshotCollection:
    """Convert an iterable of ``InferenceRule`` objects to a snapshot collection."""
    return tuple(rule_to_snapshot(r) for r in rules)


def hypothesis_to_snapshot(hypothesis: Hypothesis) -> HypothesisSnapshot:
    """Convert an internal ``Hypothesis`` to a semantic ``HypothesisSnapshot``."""
    return HypothesisSnapshot(
        hypothesis_id=hypothesis.id.to_symbolic_id().value,
        content=hypothesis.interpretation,
        confidence=hypothesis.score,
    )


def hypotheses_to_collection(
    hypotheses: Iterable[Hypothesis],
) -> HypothesisSnapshotCollection:
    """Convert an iterable of ``Hypothesis`` objects to a snapshot collection."""
    return tuple(hypothesis_to_snapshot(h) for h in hypotheses)


def decision_to_snapshot(decision: DecisionRecord) -> DecisionSnapshot:
    """Convert an internal ``DecisionRecord`` to a semantic ``DecisionSnapshot``."""
    return DecisionSnapshot(
        decision_id=decision.id.to_symbolic_id().value,
        action_text=decision.action_text,
        intent=decision.intent.value,
        confidence=decision.confidence,
        referenced_goal=decision.referenced_goal.value,
    )


def goals_to_collection(goals: Iterable[Goal]) -> GoalSnapshotCollection:
    """Convert an iterable of ``Goal`` objects to a snapshot collection."""
    return tuple(goal_to_snapshot(g) for g in goals)


def goal_to_snapshot(goal: Goal) -> GoalSnapshot:
    """Convert an internal ``Goal`` to a semantic ``GoalSnapshot``.

    The ``Goal`` type carries ``created_at`` and ``metadata``; those are
    stripped here.

    """
    goal_id = goal.goal_id.value if goal.goal_id is not None else str(goal.id)
    return GoalSnapshot(goal_id=goal_id, description=goal.description)


def build_grounding(
    beliefs: Iterable[Belief],
    concepts: Iterable[Concept] | None = None,
    rules: Iterable[InferenceRule] | None = None,
) -> GroundingSnapshot:
    """Build a ``GroundingSnapshot`` from live runtime state.

    Evidence identifiers are collected from the support traces of every belief,
    so proposals may reference percepts and other evidence.

    """
    belief_list = list(beliefs)
    concept_list = list(concepts or ())
    rule_list = list(rules or ())
    evidence_ids = frozenset(
        trace.evidence_id.value
        for belief in belief_list
        for trace in belief.support
    )
    return GroundingSnapshot(
        belief_ids=frozenset(b.id.to_symbolic_id().value for b in belief_list),
        concept_ids=frozenset(c.id.to_symbolic_id().value for c in concept_list),
        rule_ids=frozenset(r.id.to_symbolic_id().value for r in rule_list),
        evidence_ids=evidence_ids,
    )
