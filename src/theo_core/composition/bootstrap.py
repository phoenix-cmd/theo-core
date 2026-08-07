"""Bootstrap — builds the entire THEO object graph from configuration.

Wires all concrete implementations to abstractions.
"""

from __future__ import annotations

from theo_core.cognition.inference.engine import InferenceEngine
from theo_core.cognition.inference.strategies.rule_based import RuleBasedStrategy
from theo_core.cognition.planning.planner import RuleBasedPlanner
from theo_core.cognitive_cycle.engine.cognitive_engine import CognitiveEngine
from theo_core.composition.container import TheoContainer
from theo_core.context.session.active_context import InMemoryContextManager
from theo_core.events.bus import EventBus
from theo_core.explanation.engine.explain_engine import ExplainEngine
from theo_core.explanation.replay.replay_engine import ReplayEngine
from theo_core.goals.manager.goal_manager import GoalManager
from theo_core.infrastructure.config import TheoSettings
from theo_core.infrastructure.experiment_tracking import ExperimentTrackerFactory
from theo_core.infrastructure.logging import configure_logging
from theo_core.kernel.boot import Kernel
from theo_core.kernel.lifecycle import LifecycleManager
from theo_core.kernel.registry import SubsystemRegistry
from theo_core.knowledge.graph.engine import KnowledgeGraphEngine
from theo_core.memory.classifier.memory_classifier import MemoryClassifier
from theo_core.memory.engine.deterministic_memory import DeterministicMemoryEngine
from theo_core.memory.storage.json_repository import JSONMemoryRepository
from theo_core.perception.text.data_driven_processor import DataDrivenPerceptionProcessor
from theo_core.response.template.generator import TemplateResponseGenerator
from theo_core.symbolic.persistence.store import SymbolicStateStore
from theo_core.symbolic.pipeline import SymbolicCognitivePipeline
from theo_core.symbolic.response.renderer import TemplateResponseRenderer
from theo_core.symbolic.runtime import SymbolicRuntime
from theo_core.telemetry.tracing.recorder import TraceRecorder


def bootstrap(
    settings: TheoSettings | None = None,
    memory_file: str = "data/memory_store.json",
    knowledge_file: str = "data/knowledge_graph.json",
    trace_dir: str = "data/traces",
    state_file: str = "data/symbolic_state.json",
) -> TheoContainer:
    """Build and wire the entire THEO cognitive system.

    Args:
        settings: Optional settings override. Uses defaults if not provided.
        memory_file: Path to JSON memory store file.
        knowledge_file: Path to JSON knowledge graph file.
        trace_dir: Directory path for trace JSON files.
        state_file: Path for persisted symbolic committed state.

    Returns:
        A fully wired TheoContainer ready for operation.

    """
    if settings is None:
        settings = TheoSettings()

    # 1. Configure logging first
    configure_logging(
        level=settings.logging.level,
        format_style=settings.logging.format,
    )

    # 2. Central Event Bus
    event_bus = EventBus()

    # 3. Experiment Tracker & Trace Recorder
    tracker = ExperimentTrackerFactory.create(
        backend=settings.experiment_tracking.backend,
    )
    trace_recorder = TraceRecorder(trace_dir=trace_dir)

    # 4. v0.2 Cognitive Subsystems
    perception = DataDrivenPerceptionProcessor()
    context_mgr = InMemoryContextManager()
    repo = JSONMemoryRepository(file_path=memory_file)
    memory_engine = DeterministicMemoryEngine(repository=repo)
    memory_classifier = MemoryClassifier()
    knowledge_engine = KnowledgeGraphEngine(file_path=knowledge_file)
    goal_mgr = GoalManager()
    planner = RuleBasedPlanner()
    inference_strategy = RuleBasedStrategy()
    inference_engine = InferenceEngine(strategy=inference_strategy)
    response_generator = TemplateResponseGenerator()
    explain_engine = ExplainEngine()

    # 5. Cognitive Pipeline Orchestrators
    cognitive_engine = CognitiveEngine(
        perception=perception,
        context_mgr=context_mgr,
        memory_engine=memory_engine,
        memory_classifier=memory_classifier,
        knowledge_engine=knowledge_engine,
        goal_mgr=goal_mgr,
        planner=planner,
        inference_engine=inference_engine,
        response_generator=response_generator,
        trace_recorder=trace_recorder,
        event_bus=event_bus,
    )

    symbolic_pipeline = SymbolicCognitivePipeline()
    symbolic_state_store = SymbolicStateStore(state_file)
    response_renderer = TemplateResponseRenderer()
    symbolic_runtime = SymbolicRuntime(
        pipeline=symbolic_pipeline,
        renderer=response_renderer,
        store=symbolic_state_store,
        recorder=trace_recorder,
    )

    replay_engine = ReplayEngine(
        recorder=trace_recorder,
        engine_factory=lambda: SymbolicRuntime(
            pipeline=SymbolicCognitivePipeline(rules=list(symbolic_pipeline.rules))
        ),
    )

    # 6. Kernel & Registry
    registry = SubsystemRegistry()
    lifecycle = LifecycleManager()

    registry.register("event_bus", event_bus)
    registry.register("experiment_tracker", tracker)
    registry.register("perception", perception)
    registry.register("context_manager", context_mgr)
    registry.register("memory_engine", memory_engine)
    registry.register("memory_classifier", memory_classifier)
    registry.register("knowledge_engine", knowledge_engine)
    registry.register("goal_manager", goal_mgr)
    registry.register("planner", planner)
    registry.register("inference_engine", inference_engine)
    registry.register("response_generator", response_generator)
    registry.register("trace_recorder", trace_recorder)
    registry.register("explain_engine", explain_engine)
    registry.register("replay_engine", replay_engine)
    registry.register("cognitive_engine", cognitive_engine)
    registry.register("symbolic_pipeline", symbolic_pipeline)
    registry.register("response_renderer", response_renderer)
    registry.register("symbolic_runtime", symbolic_runtime)

    kernel = Kernel(
        registry=registry,
        event_bus=event_bus,
        lifecycle=lifecycle,
        start_order=settings.kernel.subsystem_start_order,
    )

    return TheoContainer(
        settings=settings,
        event_bus=event_bus,
        kernel=kernel,
        experiment_tracker=tracker,
        perception=perception,
        context_manager=context_mgr,
        memory_engine=memory_engine,
        memory_classifier=memory_classifier,
        knowledge_engine=knowledge_engine,
        goal_manager=goal_mgr,
        planner=planner,
        inference_engine=inference_engine,
        response_generator=response_generator,
        trace_recorder=trace_recorder,
        explain_engine=explain_engine,
        replay_engine=replay_engine,
        cognitive_engine=cognitive_engine,
        symbolic_pipeline=symbolic_pipeline,
        response_renderer=response_renderer,
        symbolic_runtime=symbolic_runtime,
    )
