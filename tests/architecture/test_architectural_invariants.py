"""Architectural Compliance Test Suite enforcing ARCHITECTURAL_INVARIANTS.md."""

from __future__ import annotations

import importlib
import inspect

from theo_core.composition.bootstrap import bootstrap


class TestArchitecturalInvariants:
    """Automated verification of architectural invariants."""

    def test_all_12_subsystems_registered(self) -> None:
        """Kernel registry must contain all registered subsystems."""
        container = bootstrap()
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
