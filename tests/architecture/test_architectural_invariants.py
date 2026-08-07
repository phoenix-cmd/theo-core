"""Architectural Compliance Test Suite enforcing ARCHITECTURAL_INVARIANTS.md."""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING

from theo_core.composition.bootstrap import bootstrap

if TYPE_CHECKING:
    from pathlib import Path


class TestArchitecturalInvariants:
    """Automated verification of architectural invariants."""

    def test_all_12_subsystems_registered(self, tmp_path: Path) -> None:
        """Kernel registry must contain all registered subsystems."""
        container = bootstrap(
            memory_file=str(tmp_path / "memory_store.json"),
            knowledge_file=str(tmp_path / "knowledge_graph.json"),
            trace_dir=str(tmp_path / "traces"),
            state_file=str(tmp_path / "symbolic_state.json"),
        )
        subsystems = [e.name for e in container.kernel._registry.all_entries()]
        assert "event_bus" in subsystems
        assert "perception" in subsystems
        assert "context_manager" in subsystems
        assert "memory_engine" in subsystems
        assert "knowledge_engine" in subsystems
        assert "goal_manager" in subsystems
        assert "planner" in subsystems
        assert "inference_engine" in subsystems
        assert "response_generator" in subsystems
        assert "trace_recorder" in subsystems
        assert "explain_engine" in subsystems
        assert "replay_engine" in subsystems
        assert "cognitive_engine" in subsystems

    def test_no_direct_cross_subsystem_imports(self) -> None:
        """Subsystem packages must not directly import internals of other subsystems."""
        perception_mod = importlib.import_module(
            "theo_core.perception.text.data_driven_processor"
        )
        source = inspect.getsource(perception_mod)
        assert "from theo_core.cognition" not in source
        assert "from theo_core.memory" not in source
        assert "from theo_core.goals" not in source
