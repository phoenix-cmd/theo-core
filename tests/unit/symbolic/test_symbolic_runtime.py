"""Unit tests for the SymbolicRuntime boundary service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.symbolic.persistence.store import SymbolicStateStore
from theo_core.symbolic.runtime import SymbolicRuntime

if TYPE_CHECKING:
    from pathlib import Path


class TestSymbolicRuntime:
    def test_process_renders_and_populates_boundary_golden_trace(self) -> None:
        """The boundary MUST render text and populate GoldenTrace.response_text."""
        runtime = SymbolicRuntime()
        result = runtime.process("hello theo")

        assert result.response_text is not None
        assert result.golden_trace.response_text == result.response_text
        assert result.decision.referenced_goal.value.startswith("goal://")
        assert result.referenced_goal == result.decision.referenced_goal

    def test_start_restores_persisted_state(self, tmp_path: Path) -> None:
        """start() should restore committed state from the store."""
        path = tmp_path / "state.json"
        runtime = SymbolicRuntime(store=SymbolicStateStore(path))
        runtime.process("first input")
        runtime.persist()

        second = SymbolicRuntime(store=SymbolicStateStore(path))
        assert not second.is_started
        second.start()
        assert second.is_started
        assert second.pipeline.beliefs.node_count == runtime.pipeline.beliefs.node_count

    def test_stop_persists_state(self, tmp_path: Path) -> None:
        """stop() should persist committed state and mark the runtime stopped."""
        path = tmp_path / "state.json"
        runtime = SymbolicRuntime(store=SymbolicStateStore(path))
        runtime.process("one")
        runtime.stop()

        assert not runtime.is_started
        assert path.exists()
