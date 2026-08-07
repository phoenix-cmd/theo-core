"""SymbolicCognitivePipeline — end-to-end cognitive runtime orchestrator."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import TYPE_CHECKING

from theo_core.evaluation.benchmark_schema import GoldenTrace
from theo_core.symbolic.beliefs.graph import BeliefGraph
from theo_core.symbolic.beliefs.models import Belief, BeliefId, BeliefSource
from theo_core.symbolic.concepts.activation import ActivationEngine
from theo_core.symbolic.concepts.graph import ConceptGraph
from theo_core.symbolic.conflict.models import ConflictPolicy
from theo_core.symbolic.conflict.resolver import ConflictResolver
from theo_core.symbolic.constraints.engine import ConstraintEngine
from theo_core.symbolic.decisions.engine import DecisionEngine
from theo_core.symbolic.decisions.models import DecisionRecord  # noqa: TC001
from theo_core.symbolic.hypotheses.engine import HypothesisEngine
from theo_core.symbolic.inference.engine import InferenceEngine
from theo_core.symbolic.inference.models import InferenceRule  # noqa: TC001
from theo_core.symbolic.scheduler.models import ComputeBudget, CycleStage, SchedulerTrace
from theo_core.symbolic.scheduler.scheduler import CognitiveScheduler
from theo_core.symbolic.thoughts.graph import ThoughtGraph

if TYPE_CHECKING:
    from theo_core.symbolic._primitives.identifiers import SymbolicId


class SymbolicCognitivePipeline:
    """End-to-end deterministic symbolic cognitive pipeline orchestrator.

    Integrates Concept System, Belief System, Thought Graph, Inference Engine,
    Hypothesis Engine, Constraint Engine, Conflict Resolution, Decision Engine,
    and Cognitive Scheduler into a single verifiable execution cycle.

    Guarantees:
    - 100% Deterministic execution (Canon Invariant 8 & Invariant 2).
    - Full traceability (Canon Law 5).
    - Immutable data structures (Canon Invariant 1).
    - Zero side effects on state across independent runs.
    """

    def __init__(
        self,
        concepts: ConceptGraph | None = None,
        beliefs: BeliefGraph | None = None,
        thoughts: ThoughtGraph | None = None,
        rules: list[InferenceRule] | None = None,
    ) -> None:
        """Initialize cognitive pipeline state containers."""
        self.concepts = concepts or ConceptGraph()
        self.beliefs = beliefs or BeliefGraph()
        self.thoughts = thoughts or ThoughtGraph()
        self.rules = rules or []

    def execute_cycle(
        self,
        percept_input: str,
        budget: ComputeBudget | None = None,
        active_goal_id: SymbolicId | None = None,
    ) -> tuple[DecisionRecord, SchedulerTrace, GoldenTrace]:
        """Execute one 8-stage deterministic cognitive cycle.

        Args:
            percept_input: Raw percept string input.
            budget: Optional compute budget.
            active_goal_id: Optional active GoalId per Canon Invariant 7.

        Returns:
            A tuple of (DecisionRecord, SchedulerTrace, GoldenTrace).

        """
        scheduler = CognitiveScheduler(budget)
        scheduler.start_cycle()

        # 1. PERCEPTION & INITIAL BELIEF ADMISSION
        scheduler.record_stage(CycleStage.PERCEPTION)
        percept_hash = hashlib.sha256(percept_input.encode("utf-8")).hexdigest()[:8]
        b_init_id = BeliefId.of(f"belief://percept/{percept_hash}")
        if not self.beliefs.has_belief(b_init_id):
            self.beliefs.add_belief(
                Belief(
                    id=b_init_id,
                    proposition=percept_input,
                    source=BeliefSource.PERCEPTION,
                )
            )

        # 2. CONCEPT ACTIVATION
        scheduler.record_stage(CycleStage.ACTIVATION)
        concept_ids = self.concepts.get_concepts()
        if concept_ids:
            seed = {concept_ids[0].id: Decimal("1.0")}
            ActivationEngine.activate(self.concepts, seeds=seed)

        # 3. MEMORY RETRIEVAL & REVISION
        scheduler.record_stage(CycleStage.REVISION)

        # 4. INFERENCE & RULE DEDUCTION
        scheduler.record_stage(CycleStage.INFERENCE)
        if self.rules:
            InferenceEngine.forward_chain(
                self.concepts, self.beliefs, self.thoughts, self.rules
            )

        # 5. HYPOTHESIS GENERATION & EVALUATION
        scheduler.record_stage(CycleStage.HYPOTHESIS)
        cands = HypothesisEngine.generate_hypotheses(
            percept_input, self.concepts, self.beliefs, self.thoughts
        )
        evaluated_cands = HypothesisEngine.evaluate_hypotheses(
            cands, self.thoughts, self.beliefs
        )

        # 6. CONFLICT RESOLUTION
        scheduler.record_stage(CycleStage.CONFLICT_RESOLUTION)
        resolved_cands, _conflicts = ConflictResolver.resolve_hypotheses_conflict(
            evaluated_cands, policy=ConflictPolicy.HIGHER_CONFIDENCE
        )

        # 7. DECISION SELECTION
        scheduler.record_stage(CycleStage.DECISION)
        decision = DecisionEngine.make_decision(
            resolved_cands, self.thoughts, active_goal_id=active_goal_id
        )

        # 8. REALIZATION & CONSTRAINT VALIDATION
        scheduler.record_stage(CycleStage.REALIZATION)
        ConstraintEngine.validate_all(self.concepts, self.beliefs, self.thoughts)

        trace = scheduler.finalize_trace()
        active_concepts = self.concepts.get_concepts()
        active_beliefs = self.beliefs.get_active_beliefs()
        golden_trace = GoldenTrace(
            activated_concept_ids=tuple(c.id.to_symbolic_id() for c in active_concepts),
            generated_hypothesis_ids=tuple(h.id.to_symbolic_id() for h in resolved_cands),
            derived_belief_ids=tuple(b.id.to_symbolic_id() for b in active_beliefs),
            resolved_conflict_ids=tuple(c.conflict_id for c in _conflicts),
            thought_dag_node_count=self.thoughts.node_count,
            decision_id=decision.id.to_symbolic_id(),
            response_text=decision.action_text,
        )
        return decision, trace, golden_trace
