"""Port contract tests — snapshot converters (ADR-0028).

Converters strip timestamps, metadata, and mutable lifecycle fields from
internal runtime models.
"""

from __future__ import annotations

from decimal import Decimal

from theo_core.domain.runtime.entities.goal import Goal
from theo_core.models.ports.converters import (
    belief_to_snapshot,
    build_grounding,
    concept_to_snapshot,
    decision_to_snapshot,
    goal_to_snapshot,
    hypothesis_to_snapshot,
    rule_to_snapshot,
)
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import Belief, BeliefId, BeliefSource, EvidenceTrace
from theo_core.symbolic.concepts.models import Concept, ConceptId
from theo_core.symbolic.decisions.models import (
    ActionSpec,
    DecisionId,
    DecisionRecord,
    Intent,
)
from theo_core.symbolic.hypotheses.models import Hypothesis, HypothesisId
from theo_core.symbolic.inference.models import (
    InferenceRule,
    RuleCondition,
    RuleId,
)


class TestConverters:
    def test_belief_snapshot_strips_lifecycle_state(self) -> None:
        belief = Belief(
            id=BeliefId.of("belief://sun_rises_east"),
            proposition="The sun rises in the east",
            confidence=Decimal("0.99"),
            support=(
                EvidenceTrace(
                    evidence_id=SymbolicId.of("memory://obs_1"),
                    source_type="perception",
                ),
            ),
            source=BeliefSource.MEMORY,
            metadata={"irrelevant": True},
        )
        snapshot = belief_to_snapshot(belief)
        assert snapshot.belief_id == "belief://sun_rises_east"
        assert snapshot.proposition == "The sun rises in the east"
        assert snapshot.confidence == Decimal("0.99")
        assert snapshot.source == "memory"
        assert not hasattr(snapshot, "last_verified")
        assert not hasattr(snapshot, "metadata")

    def test_rule_snapshot_flattens_conditions(self) -> None:
        rule = InferenceRule(
            id=RuleId.of("rule://r1"),
            name="mammal_rule",
            conditions=(
                RuleCondition(premise_predicate="is_mammal"),
                RuleCondition(premise_predicate="is_warm_blooded"),
            ),
            conclusion_template="classifies_as_mammal",
            salience=Decimal("0.8"),
        )
        snapshot = rule_to_snapshot(rule)
        assert snapshot.rule_id == "rule://r1"
        assert snapshot.premise_text == "is_mammal AND is_warm_blooded"
        assert snapshot.conclusion_text == "classifies_as_mammal"
        assert snapshot.salience == Decimal("0.8")

    def test_decision_snapshot_strips_timestamps(self) -> None:
        decision = DecisionRecord(
            id=DecisionId.of("decision://d1"),
            action_text="respond",
            intent=Intent.MAINTAIN_CONVERSATION,
            referenced_goal=SymbolicId.of("goal://maintain_conversation"),
            action_spec=ActionSpec(capability="respond"),
            confidence=Decimal("0.9"),
        )
        snapshot = decision_to_snapshot(decision)
        assert snapshot.decision_id == "decision://d1"
        assert snapshot.intent == "maintain_conversation"
        assert snapshot.confidence == Decimal("0.9")
        assert not hasattr(snapshot, "created_at")
        assert not hasattr(snapshot, "metadata")

    def test_goal_snapshot_uses_deterministic_goal_id(self) -> None:
        goal = Goal(description="MaintainConversation")
        snapshot = goal_to_snapshot(goal)
        assert snapshot.goal_id == "goal://maintainconversation"
        assert snapshot.description == "MaintainConversation"

    def test_concept_and_hypothesis_converters(self) -> None:
        concept = Concept(id=ConceptId.of("concept://sun"), label="sun")
        concept_snapshot = concept_to_snapshot(concept)
        assert concept_snapshot.concept_id == "concept://sun"
        assert concept_snapshot.name == "sun"

        hypothesis = Hypothesis(
            id=HypothesisId.of("hypothesis://h1"),
            interpretation="The sun is a star",
            score=Decimal("0.7"),
        )
        hypothesis_snapshot = hypothesis_to_snapshot(hypothesis)
        assert hypothesis_snapshot.hypothesis_id == "hypothesis://h1"
        assert hypothesis_snapshot.content == "The sun is a star"
        assert hypothesis_snapshot.confidence == Decimal("0.7")

    def test_build_grounding_collects_all_entity_kinds(self) -> None:
        belief = Belief(
            id=BeliefId.of("belief://b1"),
            proposition="the sky is blue",
            support=(
                EvidenceTrace(
                    evidence_id=SymbolicId.of("percept://sky"),
                    source_type="perception",
                ),
            ),
        )
        concept = Concept(id=ConceptId.of("concept://sky"), label="sky")
        rule = InferenceRule(
            id=RuleId.of("rule://r1"),
            name="sky_rule",
            conditions=(RuleCondition(premise_predicate="is_sky"),),
            conclusion_template="is_blue",
        )
        grounding = build_grounding(
            beliefs=(belief,), concepts=(concept,), rules=(rule,)
        )
        assert grounding.belief_ids == frozenset({"belief://b1"})
        assert grounding.concept_ids == frozenset({"concept://sky"})
        assert grounding.rule_ids == frozenset({"rule://r1"})
        assert grounding.evidence_ids == frozenset({"percept://sky"})
