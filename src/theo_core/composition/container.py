"""TheoContainer — holds all live service instances for the THEO system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.cognition.inference.engine import InferenceEngine
    from theo_core.cognition.planning.planner import RuleBasedPlanner
    from theo_core.cognitive_cycle.engine.cognitive_engine import CognitiveEngine
    from theo_core.context.session.active_context import InMemoryContextManager
    from theo_core.domain.research.ports.experiment_tracker import ExperimentTrackerPort
    from theo_core.events.bus import EventBus
    from theo_core.explanation.engine.explain_engine import ExplainEngine
    from theo_core.explanation.replay.replay_engine import ReplayEngine
    from theo_core.goals.manager.goal_manager import GoalManager
    from theo_core.infrastructure.config import TheoSettings
    from theo_core.kernel.boot import Kernel
    from theo_core.knowledge.graph.engine import KnowledgeGraphEngine
    from theo_core.memory.classifier.memory_classifier import MemoryClassifier
    from theo_core.memory.engine.deterministic_memory import DeterministicMemoryEngine
    from theo_core.perception.text.data_driven_processor import DataDrivenPerceptionProcessor
    from theo_core.response.template.generator import TemplateResponseGenerator
    from theo_core.symbolic.pipeline import SymbolicCognitivePipeline
    from theo_core.symbolic.response.renderer import TemplateResponseRenderer
    from theo_core.symbolic.runtime import SymbolicRuntime
    from theo_core.telemetry.tracing.recorder import TraceRecorder


@dataclass
class TheoContainer:
    """Holds all live instances for the THEO cognitive system.

    Created once at startup by the bootstrap function.

    Attributes:
        settings: The root configuration.
        event_bus: The central event bus.
        kernel: The cognitive kernel.
        experiment_tracker: The active experiment tracking backend.
        perception: Data-driven perception processor.
        context_manager: Ephemeral session context manager.
        memory_engine: Deterministic 4-layer memory engine.
        memory_classifier: Memory classifier.
        knowledge_engine: Deterministic graph knowledge engine.
        goal_manager: Priority goal stack manager.
        planner: Cognitive planner.
        inference_engine: Cognitive inference engine.
        response_generator: Template response generator.
        trace_recorder: Cognitive trace recorder.
        explain_engine: Explainability engine.
        replay_engine: Trace replay engine.
        cognitive_engine: 12-stage cognitive pipeline orchestrator.
        symbolic_pipeline: Canonical symbolic cognitive pipeline.
        response_renderer: Boundary response renderer (Canon Law 6).
        symbolic_runtime: Canonical symbolic runtime boundary service.

    """

    settings: TheoSettings
    event_bus: EventBus
    kernel: Kernel
    experiment_tracker: ExperimentTrackerPort
    perception: DataDrivenPerceptionProcessor
    context_manager: InMemoryContextManager
    memory_engine: DeterministicMemoryEngine
    memory_classifier: MemoryClassifier
    knowledge_engine: KnowledgeGraphEngine
    goal_manager: GoalManager
    planner: RuleBasedPlanner
    inference_engine: InferenceEngine
    response_generator: TemplateResponseGenerator
    trace_recorder: TraceRecorder
    explain_engine: ExplainEngine
    replay_engine: ReplayEngine
    cognitive_engine: CognitiveEngine
    symbolic_pipeline: SymbolicCognitivePipeline
    response_renderer: TemplateResponseRenderer
    symbolic_runtime: SymbolicRuntime
