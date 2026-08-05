"""CognitiveEngine — 12-stage deterministic cognitive pipeline orchestrator with tracing.

Executes the full 12-stage cognitive pipeline:
Perception -> Context -> Memory Classifier -> Memory Retrieval -> Knowledge Retrieval
-> Goal Selection -> Planning -> Cognitive Inference -> Reflection -> Decision -> Response Generator
-> Learning & Trace
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from theo_core.domain.runtime.entities.cognitive_state import CognitiveState
from theo_core.domain.runtime.entities.decision import Decision
from theo_core.domain.runtime.entities.decision_record import DecisionRecord
from theo_core.events.events import (
    ContextUpdatedV1,
    DecisionMadeV1,
    GoalSelectedV1,
    InferenceCompletedV1,
    KnowledgeRetrievedV1,
    MemoryRetrievedV1,
    MemoryStoredV1,
    PerceptAnalyzedV1,
    PlanGeneratedV1,
    ResponseGeneratedV1,
    TraceRecordedV1,
)

if TYPE_CHECKING:
    from theo_core.cognition.inference.engine import InferenceEngine
    from theo_core.cognition.planning.planner import RuleBasedPlanner
    from theo_core.context.session.active_context import InMemoryContextManager
    from theo_core.events.bus import EventBus
    from theo_core.goals.manager.goal_manager import GoalManager
    from theo_core.knowledge.graph.engine import KnowledgeGraphEngine
    from theo_core.memory.classifier.memory_classifier import MemoryClassifier
    from theo_core.memory.engine.deterministic_memory import DeterministicMemoryEngine
    from theo_core.perception.text.data_driven_processor import DataDrivenPerceptionProcessor
    from theo_core.response.template.generator import TemplateResponseGenerator
    from theo_core.telemetry.tracing.recorder import TraceRecorder

logger = logging.getLogger(__name__)


class CognitiveEngine:
    """The 12-stage cognitive pipeline orchestrator.

    Executes all cognitive stages in sequence while passing a single CognitiveState
    instance, collecting trace spans, and emitting V1 domain events to the EventBus.
    """

    def __init__(
        self,
        perception: DataDrivenPerceptionProcessor,
        context_mgr: InMemoryContextManager,
        memory_engine: DeterministicMemoryEngine,
        memory_classifier: MemoryClassifier,
        knowledge_engine: KnowledgeGraphEngine,
        goal_mgr: GoalManager,
        planner: RuleBasedPlanner,
        inference_engine: InferenceEngine,
        response_generator: TemplateResponseGenerator,
        trace_recorder: TraceRecorder,
        event_bus: EventBus,
    ) -> None:
        """Initialize the CognitiveEngine with required stage implementations.

        Args:
            perception: Perception Engine.
            context_mgr: Ephemeral Context Manager.
            memory_engine: Deterministic Memory Engine.
            memory_classifier: Memory Classifier.
            knowledge_engine: Knowledge Graph Engine.
            goal_mgr: Goal Stack Manager.
            planner: Cognitive Planner.
            inference_engine: Cognitive Inference Engine.
            response_generator: Response Generator.
            trace_recorder: Cognitive Trace Recorder.
            event_bus: Event Bus.

        """
        self._perception = perception
        self._context_mgr = context_mgr
        self._memory_engine = memory_engine
        self._memory_classifier = memory_classifier
        self._knowledge_engine = knowledge_engine
        self._goal_mgr = goal_mgr
        self._planner = planner
        self._inference_engine = inference_engine
        self._response_generator = response_generator
        self._trace_recorder = trace_recorder
        self._event_bus = event_bus
        self._last_state: CognitiveState | None = None
        self._last_record: DecisionRecord | None = None

    def process(self, raw_input: str) -> CognitiveState:
        """Execute one complete 12-stage cognitive iteration.

        Args:
            raw_input: Raw text string from the user.

        Returns:
            The final CognitiveState object containing decision, response text, and visited stages.

        """
        state = CognitiveState(raw_input=raw_input)
        self._trace_recorder.start_cycle()

        # 1. Perception Stage
        state.visit_stage("perception")
        self._trace_recorder.start_stage("perception")
        state.percept = self._perception.perceive(raw_input)
        intent = state.percept.metadata.get("intent", "unknown")
        self._trace_recorder.end_stage(
            "perception",
            input_summary=raw_input,
            output_summary=f"intent={intent}",
        )
        self._event_bus.publish(
            PerceptAnalyzedV1(
                source="perception",
                percept_id=state.percept.id,
                intent=intent,
                fact_count=len(state.percept.metadata.get("facts", {})),
            )
        )

        # 2. Context Stage
        state.visit_stage("context")
        self._trace_recorder.start_stage("context")
        turn_count = self._context_mgr.increment_turns()
        state.context = self._context_mgr.snapshot()
        self._trace_recorder.end_stage(
            "context",
            input_summary=f"turn={turn_count}",
            output_summary=f"user={state.context.get('active_user')}",
        )
        self._event_bus.publish(
            ContextUpdatedV1(
                source="context",
                turn_count=turn_count,
                active_user=state.context.get("active_user", "anonymous"),
            )
        )

        # 3. Memory Classification Stage
        state.visit_stage("memory_classifier")
        self._trace_recorder.start_stage("memory_classifier")
        category = self._memory_classifier.classify(state.percept)
        state.memory_classification = category
        self._trace_recorder.end_stage(
            "memory_classifier",
            output_summary=f"category={category}",
        )

        # 4. Memory Retrieval Stage
        state.visit_stage("memory_retrieval")
        self._trace_recorder.start_stage("memory_retrieval")
        scored_memories = self._memory_engine.retrieve_scored(raw_input, top_k=5)
        state.retrieved_memories = [rm.entry.model_dump(mode="json") for rm in scored_memories]
        state.workspace.retrieved_memory_ids = tuple(rm.memory_id for rm in scored_memories)
        self._trace_recorder.end_stage(
            "memory_retrieval",
            output_summary=f"retrieved_count={len(scored_memories)}",
        )
        self._event_bus.publish(
            MemoryRetrievedV1(
                source="memory",
                query=raw_input,
                retrieved_count=len(scored_memories),
            )
        )

        # 5. Knowledge Retrieval Stage (Multi-hop concept traversal)
        state.visit_stage("knowledge_retrieval")
        self._trace_recorder.start_stage("knowledge_retrieval")
        state.retrieved_knowledge = self._knowledge_engine.search_concepts(raw_input)
        self._trace_recorder.end_stage(
            "knowledge_retrieval",
            output_summary=f"knowledge_facts_count={len(state.retrieved_knowledge)}",
        )
        self._event_bus.publish(
            KnowledgeRetrievedV1(
                source="knowledge",
                concept=raw_input,
                facts_count=len(state.retrieved_knowledge),
            )
        )

        # 6. Goal Selection Stage
        state.visit_stage("goal_selection")
        self._trace_recorder.start_stage("goal_selection")
        state.active_goal = self._goal_mgr.select_top_goal(state.percept)
        state.workspace.active_goal_id = state.active_goal.id
        self._trace_recorder.end_stage(
            "goal_selection",
            output_summary=f"goal={state.active_goal.description}",
        )
        self._event_bus.publish(
            GoalSelectedV1(
                source="goals",
                goal_id=state.active_goal.id,
                goal_description=state.active_goal.description,
                priority=state.active_goal.priority.value,
            )
        )

        # 7. Planning Stage
        state.visit_stage("planning")
        self._trace_recorder.start_stage("planning")
        state.plan = self._planner.plan(state.active_goal.description)
        state.workspace.current_plan_id = state.plan.id
        self._trace_recorder.end_stage(
            "planning",
            output_summary=f"actions_count={state.plan.action_count}",
        )
        self._event_bus.publish(
            PlanGeneratedV1(
                source="planning",
                plan_id=state.plan.id,
                action_count=state.plan.action_count,
            )
        )

        # 8. Cognitive Inference Stage
        state.visit_stage("inference")
        self._trace_recorder.start_stage("inference")
        state.inference_result = self._inference_engine.infer(state)
        self._trace_recorder.end_stage(
            "inference",
            output_summary=f"strategy={self._inference_engine.active_strategy_name}",
        )
        self._event_bus.publish(
            InferenceCompletedV1(
                source="inference",
                strategy_name=self._inference_engine.active_strategy_name,
                confidence=state.inference_result.get("confidence", 1.0),
            )
        )

        # 9. Reflection Stage
        state.visit_stage("reflection")
        self._trace_recorder.start_stage("reflection")
        state.reflection_result = {
            "satisfied": True,
            "confidence": state.inference_result.get("confidence", 1.0),
        }
        state.workspace.reflection_notes = state.reflection_result
        self._trace_recorder.end_stage("reflection", output_summary="satisfied=True")

        # 10. Decision Stage
        state.visit_stage("decision")
        self._trace_recorder.start_stage("decision")
        cand_resp = state.inference_result.get("candidate_response", "I understand.")
        conf = state.inference_result.get("confidence", 1.0)
        summary = state.inference_result.get("reasoning_summary", "")
        actions = tuple(state.inference_result.get("actions_executed", []))
        used_mem_ids = tuple(state.inference_result.get("used_memory_ids", []))
        used_rule_ids = tuple(state.inference_result.get("used_rule_ids", []))
        state.workspace.used_memory_ids = used_mem_ids

        decision_obj = Decision(
            response=cand_resp,
            confidence=conf,
            goal=state.active_goal.description,
            actions_taken=actions,
            used_memory_ids=used_mem_ids,
            reasoning_summary=summary,
        )
        state.decision = decision_obj.model_dump()

        record_obj = DecisionRecord(
            selected_option=cand_resp,
            selection_reason=summary,
            confidence=conf,
            used_memory_ids=used_mem_ids,
            used_rule_ids=used_rule_ids,
            used_goal=state.active_goal.description,
        )
        self._last_record = record_obj

        self._trace_recorder.end_stage(
            "decision",
            output_summary=f"decision_id={decision_obj.id}",
        )
        self._event_bus.publish(
            DecisionMadeV1(
                source="decision",
                decision_id=decision_obj.id,
                response_summary=summary,
                confidence=conf,
            )
        )

        # 11. Response Generator Stage
        state.visit_stage("response_generator")
        self._trace_recorder.start_stage("response_generator")
        state.response_text = self._response_generator.generate(decision_obj)
        self._trace_recorder.end_stage(
            "response_generator",
            output_summary=f"length={len(state.response_text)}",
        )
        self._event_bus.publish(
            ResponseGeneratedV1(
                source="response",
                response_length=len(state.response_text),
                generator_type="TemplateResponseGenerator",
            )
        )

        # 12. Learning & Persistence Stage
        state.visit_stage("learning")
        self._trace_recorder.start_stage("learning")
        self._context_mgr.set("last_percept", state.percept.content)
        self._context_mgr.set("last_response", state.response_text)

        provenance_map = {
            "percept_id": str(state.percept.id),
            "decision_id": str(decision_obj.id),
            "cycle_id": str(state.cycle_id),
        }

        facts = state.percept.metadata.get("facts", {})
        for k, v in facts.items():
            self._context_mgr.set(k, v)
            mem_entry = self._memory_engine.store_fact(
                key=k,
                value=v,
                category=category,
                provenance=provenance_map,
            )
            self._event_bus.publish(
                MemoryStoredV1(
                    source="memory",
                    memory_key=mem_entry.key,
                    category=mem_entry.memory_type,
                )
            )

        self._memory_engine.add_working(
            {
                "input": raw_input,
                "response": state.response_text,
                "intent": intent,
                "used_memory_ids": list(used_mem_ids),
            }
        )

        self._trace_recorder.end_stage("learning", output_summary="facts_stored")

        # Close trace & publish event
        stats = {
            "retrieved_memories_count": len(scored_memories),
            "retrieved_knowledge_count": len(state.retrieved_knowledge),
            "used_memory_ids_count": len(used_mem_ids),
            "used_rule_ids_count": len(used_rule_ids),
        }
        trace = self._trace_recorder.close_trace(
            cycle_id=state.cycle_id,
            raw_input=raw_input,
            response_text=state.response_text,
            execution_stats=stats,
        )
        self._last_state = state
        self._last_record.trace_id = trace.trace_id

        self._event_bus.publish(
            TraceRecordedV1(
                source="tracing",
                trace_id=trace.trace_id,
                file_path=f"data/traces/{trace.trace_id}.json",
                cognitive_depth=state.cognitive_depth,
            )
        )

        return state

    @property
    def last_state(self) -> CognitiveState | None:
        """Return the CognitiveState of the last executed cycle."""
        return self._last_state

    @property
    def last_record(self) -> DecisionRecord | None:
        """Return the DecisionRecord of the last executed cycle."""
        return self._last_record
