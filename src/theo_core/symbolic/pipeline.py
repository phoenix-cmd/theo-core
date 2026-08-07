"""SymbolicCognitivePipeline — end-to-end cognitive runtime orchestrator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from theo_core.domain.runtime.entities.percept import Percept as DomainPercept
from theo_core.domain.runtime.entities.percept import PerceptModality
from theo_core.evaluation.benchmark_schema import GoldenTrace
from theo_core.goals.manager.goal_manager import GoalManager
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId, BeliefSource, EvidenceTrace
from theo_core.symbolic.beliefs.revision import BeliefRevision
from theo_core.symbolic.concepts.activation import ActivationEngine
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.conflict.models import ConflictPolicy
from theo_core.symbolic.conflict.resolver import ConflictResolver
from theo_core.symbolic.constraints.engine import ConstraintEngine
from theo_core.symbolic.decisions.engine import DecisionEngine
from theo_core.symbolic.decisions.models import DecisionRecord  # noqa: TC001
from theo_core.symbolic.hypotheses.engine import HypothesisEngine
from theo_core.symbolic.inference.engine import InferenceEngine
from theo_core.symbolic.inference.models import (
    InferenceRule,  # noqa: TC001
    InferenceTrace,  # noqa: TC001
)
from theo_core.symbolic.perception.models import Percept, PerceptId
from theo_core.symbolic.scheduler.models import ComputeBudget, CycleStage, SchedulerTrace
from theo_core.symbolic.scheduler.scheduler import CognitiveScheduler
from theo_core.symbolic.thoughts.graph import ThoughtGraph

if TYPE_CHECKING:
    from theo_core.domain.runtime.ports.goal import GoalPort


@dataclass(frozen=True, slots=True)
class CycleState:
    """The cognitive state threaded through and published by a cycle.

    Per the Canon functional computation model (§6), stages compute over a
    working copy of the state and only the Learning stage publishes it as the
    committed state ``S_{t+1}``.
    """

    concepts: ConceptGraph = field(default_factory=ConceptGraph)
    beliefs: BeliefGraph = field(default_factory=BeliefGraph)
    thoughts: ThoughtGraph = field(default_factory=ThoughtGraph)
    percept: Percept | None = None


class SymbolicCognitivePipeline:
    """End-to-end deterministic symbolic cognitive pipeline orchestrator.

    Integrates Concept System, Belief System, Thought Graph, Inference Engine,
    Hypothesis Engine, Constraint Engine, Conflict Resolution, Decision Engine,
    and Cognitive Scheduler into a single verifiable execution cycle.

    Guarantees:
    - 100% Deterministic execution (Canon Invariant 8 & Invariant 2).
    - Full traceability (Canon Law 5).
    - Immutable data structures (Canon Invariant 1).
    - Side effects confined to the Learning stage (Canon §6): pre-L stages
      operate on a working copy, and ``S_{t+1}`` is published once at LEARNING.
    """

    def __init__(
        self,
        concepts: ConceptGraph | None = None,
        beliefs: BeliefGraph | None = None,
        thoughts: ThoughtGraph | None = None,
        rules: list[InferenceRule] | None = None,
        goal_manager: GoalPort | None = None,
    ) -> None:
        """Initialize cognitive pipeline state containers.

        Args:
            concepts: Initial concept graph.
            beliefs: Initial belief graph.
            thoughts: Initial thought graph.
            rules: Initial inference rules.
            goal_manager: Goal port used to resolve the active goal
                (Canon Invariant 7). Defaults to a fresh ``GoalManager``.

        """
        self.concepts = concepts or ConceptGraph()
        self.beliefs = beliefs or BeliefGraph()
        self.thoughts = thoughts or ThoughtGraph()
        self.rules = rules or []
        self._goal_manager = goal_manager or GoalManager()
        self.state: CycleState = CycleState(
            concepts=self.concepts,
            beliefs=self.beliefs,
            thoughts=self.thoughts,
        )

    def execute_cycle(
        self,
        percept_input: str,
        budget: ComputeBudget | None = None,
    ) -> tuple[DecisionRecord, SchedulerTrace, GoldenTrace]:
        """Execute one 9-stage deterministic cognitive cycle.

        Args:
            percept_input: Raw percept string input.
            budget: Optional compute budget.

        Returns:
            A tuple of (DecisionRecord, SchedulerTrace, GoldenTrace).

        """
        scheduler = CognitiveScheduler(budget)
        scheduler.start_cycle()

        # Snapshot the committed state BEFORE the cycle: it is the memory
        # retrieved into working memory and the baseline for derived beliefs.
        pre_cycle_beliefs = self.beliefs.get_active_beliefs()
        pre_cycle_belief_ids = {b.id for b in pre_cycle_beliefs}
        retrieved_memory_ids = tuple(b.id.to_symbolic_id() for b in pre_cycle_beliefs)

        # 1. PERCEPTION — build an immutable Percept (evidence, not a belief)
        scheduler.record_stage(CycleStage.PERCEPTION)
        percept_hash = hashlib.sha256(percept_input.encode("utf-8")).hexdigest()[:8]
        percept = Percept(
            id=PerceptId.of(f"percept://{percept_hash}"),
            content=percept_input,
        )
        percept_belief_id = BeliefId.of(f"belief://percept/{percept_hash}")
        percept_belief = Belief(
            id=percept_belief_id,
            proposition=percept_input,
            source=BeliefSource.INFERENCE,
            support=(
                EvidenceTrace(
                    evidence_id=percept.id.to_symbolic_id(),
                    source_type="perception",
                ),
            ),
        )

        # Functional model: stages compute over a working copy (Canon §6).
        working = CycleState(
            concepts=self.concepts.copy(),
            beliefs=self.beliefs.copy(),
            thoughts=self.thoughts.copy(),
            percept=percept,
        )
        # The perceptual belief (derived from the Percept as evidence) enters the
        # working state during Perception so downstream stages consume it; the
        # committed state is only published once, at the Learning stage.
        if not working.beliefs.has_belief(percept_belief_id):
            working.beliefs.add_belief(percept_belief)

        # 2. CONCEPT ACTIVATION
        scheduler.record_stage(CycleStage.ACTIVATION)
        concept_ids = working.concepts.get_concepts()
        if concept_ids:
            seed = {concept_ids[0].id: Decimal("1.0")}
            ActivationEngine.activate(working.concepts, seeds=seed)

        # 3. MEMORY RETRIEVAL & REVISION
        scheduler.record_stage(CycleStage.REVISION)
        contradictions = BeliefRevision.detect_contradictions(working.beliefs)
        for b_id_1, b_id_2 in contradictions:
            BeliefRevision.reconcile_contradiction(working.beliefs, b_id_1, b_id_2)

        # 4. INFERENCE & RULE DEDUCTION (consumes the perceptual belief as evidence)
        scheduler.record_stage(CycleStage.INFERENCE)
        inference_trace: InferenceTrace | None = None
        if self.rules:
            inference_trace = InferenceEngine.forward_chain(
                working.concepts,
                working.beliefs,
                working.thoughts,
                self.rules,
            )

        # 5. HYPOTHESIS GENERATION & EVALUATION
        scheduler.record_stage(CycleStage.HYPOTHESIS)
        cands = HypothesisEngine.generate_hypotheses(
            percept_input, working.concepts, working.beliefs, working.thoughts
        )
        evaluated_cands = HypothesisEngine.evaluate_hypotheses(
            cands, working.thoughts, working.beliefs
        )

        # 6. CONFLICT RESOLUTION
        scheduler.record_stage(CycleStage.CONFLICT_RESOLUTION)
        resolved_cands, _conflicts = ConflictResolver.resolve_hypotheses_conflict(
            evaluated_cands, policy=ConflictPolicy.HIGHER_CONFIDENCE
        )

        # 7. DECISION SELECTION — resolve the active goal (Canon Invariant 7)
        scheduler.record_stage(CycleStage.DECISION)
        domain_percept = DomainPercept(
            modality=PerceptModality.TEXT,
            content=percept_input,
        )
        goal = self._goal_manager.select_top_goal(domain_percept)
        assert goal.goal_id is not None, "GoalId is always derived from the description"
        referenced_goal = SymbolicId.of(goal.goal_id.value)
        decision = DecisionEngine.make_decision(
            resolved_cands, working.thoughts, referenced_goal=referenced_goal
        )

        # 8. REALIZATION & CONSTRAINT VALIDATION
        scheduler.record_stage(CycleStage.REALIZATION)
        ConstraintEngine.assert_valid(working.concepts, working.beliefs, working.thoughts)

        # 9. LEARNING — publish the computed state (sole side-effect point, Canon §6)
        scheduler.record_stage(CycleStage.LEARNING)
        self.concepts = working.concepts
        self.beliefs = working.beliefs
        self.thoughts = working.thoughts
        self.state = working

        trace = scheduler.finalize_trace()
        active_concepts = self.concepts.get_concepts()
        fired_rule_ids = (
            tuple(
                sorted(
                    {step.rule_id.to_symbolic_id() for step in inference_trace.steps},
                    key=lambda sid: sid.value,
                )
            )
            if inference_trace is not None
            else ()
        )
        derived_belief_ids = tuple(
            b.id.to_symbolic_id()
            for b in self.beliefs.get_active_beliefs()
            if b.id not in pre_cycle_belief_ids
        )
        golden_trace = GoldenTrace(
            activated_concept_ids=tuple(c.id.to_symbolic_id() for c in active_concepts),
            retrieved_memory_ids=retrieved_memory_ids,
            generated_hypothesis_ids=tuple(h.id.to_symbolic_id() for h in evaluated_cands),
            fired_rule_ids=fired_rule_ids,
            derived_belief_ids=derived_belief_ids,
            resolved_conflict_ids=tuple(c.conflict_id for c in _conflicts),
            thought_dag_node_count=self.thoughts.node_count,
            decision_id=decision.id.to_symbolic_id(),
            response_text=decision.action_text,
        )
        return decision, trace, golden_trace
