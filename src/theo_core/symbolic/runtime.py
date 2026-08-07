"""SymbolicRuntime — canonical runtime boundary for the symbolic cognitive pipeline.

The pipeline produces a structured Decision (Intent + ActionSpec) and never renders
language (Canon Law 6). This runtime boundary owns: rendering the final response
via a ``ResponseRendererPort``, persisting committed state via ``SymbolicStateStore``,
and surfacing a boundary-populated ``GoldenTrace``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from theo_core.symbolic.decisions.models import DecisionRecord  # noqa: TC001
from theo_core.symbolic.response.renderer import TemplateResponseRenderer

if TYPE_CHECKING:
    from theo_core.evaluation.benchmark_schema import GoldenTrace
    from theo_core.symbolic._primitives.identifiers import SymbolicId
    from theo_core.symbolic.persistence.store import SymbolicStateStore
    from theo_core.symbolic.pipeline import SymbolicCognitivePipeline
    from theo_core.symbolic.response.port import ResponseRendererPort
    from theo_core.symbolic.scheduler.models import ComputeBudget, SchedulerTrace


@dataclass(frozen=True, slots=True)
class SymbolicRuntimeResult:
    """Outcome of one symbolic runtime turn."""

    decision: DecisionRecord
    scheduler_trace: SchedulerTrace
    golden_trace: GoldenTrace
    response_text: str
    referenced_goal: SymbolicId


class SymbolicRuntime:
    """Boundary service orchestrating the symbolic pipeline, renderer, and store.

    Implements the ``Startable``/``Stoppable`` kernel lifecycle: ``start`` restores
    committed state from the store, ``stop`` persists it back.
    """

    def __init__(
        self,
        pipeline: SymbolicCognitivePipeline | None = None,
        renderer: ResponseRendererPort | None = None,
        store: SymbolicStateStore | None = None,
    ) -> None:
        """Initialize the runtime.

        Args:
            pipeline: The symbolic pipeline to drive. Created fresh if omitted.
            renderer: Boundary response renderer. Defaults to TemplateResponseRenderer.
            store: Optional persistent state store.

        """
        from theo_core.symbolic.pipeline import SymbolicCognitivePipeline

        self._pipeline = pipeline or SymbolicCognitivePipeline()
        self._renderer = renderer or TemplateResponseRenderer()
        self._store = store
        self._started = False

    @property
    def pipeline(self) -> SymbolicCognitivePipeline:
        """Return the underlying symbolic pipeline."""
        return self._pipeline

    @property
    def is_started(self) -> bool:
        """Return True if the runtime has been started."""
        return self._started

    def start(self) -> None:
        """Restore committed state from the store and mark the runtime started."""
        if self._store is not None:
            state = self._store.load()
            if state is not None:
                self._pipeline.concepts = state.concepts
                self._pipeline.beliefs = state.beliefs
                self._pipeline.thoughts = state.thoughts
                self._pipeline.state = state
        self._started = True

    def stop(self) -> None:
        """Persist committed state to the store and mark the runtime stopped."""
        if self._store is not None:
            self._store.save(self._pipeline.state)
        self._started = False

    def persist(self) -> None:
        """Persist committed state without stopping the runtime."""
        if self._store is not None:
            self._store.save(self._pipeline.state)

    def process(
        self,
        percept_input: str,
        budget: ComputeBudget | None = None,
    ) -> SymbolicRuntimeResult:
        """Execute one cognitive cycle and render the boundary response.

        Args:
            percept_input: Raw percept string input.
            budget: Optional compute budget.

        Returns:
            A SymbolicRuntimeResult with the decision, trace, boundary-populated
            GoldenTrace, and rendered response text.

        """
        decision, scheduler_trace, golden_trace = self._pipeline.execute_cycle(
            percept_input, budget
        )
        response_text = self._renderer.render(decision)
        boundary_golden_trace = golden_trace.model_copy(update={"response_text": response_text})
        return SymbolicRuntimeResult(
            decision=decision,
            scheduler_trace=scheduler_trace,
            golden_trace=boundary_golden_trace,
            response_text=response_text,
            referenced_goal=decision.referenced_goal,
        )
