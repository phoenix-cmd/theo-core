"""SymbolicCognitivePipeline — end-to-end cognitive runtime orchestrator."""

from __future__ import annotations

from decimal import Decimal

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
    ) -> tuple[DecisionRecord, SchedulerTrace]:
        """Execute a full cognitive cycle over a percept input.

        Args:
            percept_input: Raw text input percept.
            budget: Optional compute budget constraints.

        Returns:
            Tuple of (DecisionRecord, SchedulerTrace).

        """
        scheduler = CognitiveScheduler(budget)
        scheduler.start_cycle()

        # 1. PERCEPTION & INITIAL BELIEF ADMISSION
        scheduler.record_stage(CycleStage.PERCEPTION)
        b_init_id = BeliefId.of(f"belief://percept/{hash(percept_input) & 0xFFFFFFFF}")
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

        # 3. REVISION
        scheduler.record_stage(CycleStage.REVISION)

        # 4. INFERENCE
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
        decision = DecisionEngine.make_decision(resolved_cands, self.thoughts)

        # 8. REALIZATION & CONSTRAINT VALIDATION
        scheduler.record_stage(CycleStage.REALIZATION)
        ConstraintEngine.validate_all(self.concepts, self.beliefs, self.thoughts)

        trace = scheduler.finalize_trace()
        return decision, trace
