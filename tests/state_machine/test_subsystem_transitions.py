"""State machine — subsystem lifecycle transition validation.

The registry enforces a legal transition graph; illegal jumps MUST raise
``InvalidStateTransitionError`` and leave the current state unchanged.
"""

from __future__ import annotations

import pytest

from theo_core.kernel.registry import (
    InvalidStateTransitionError,
    SubsystemRegistry,
    SubsystemState,
)


class StubSubsystem:
    """Minimal subsystem instance for registry tests."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        """Mark as started."""
        self.started = True

    def stop(self) -> None:
        """Mark as stopped."""
        self.stopped = True


def _registry() -> SubsystemRegistry:
    registry = SubsystemRegistry()
    registry.register("sub", StubSubsystem())
    return registry


class TestLegalTransitions:
    def test_full_lifecycle_chain(self) -> None:
        registry = _registry()
        registry.transition("sub", SubsystemState.STARTING)
        registry.transition("sub", SubsystemState.RUNNING)
        registry.transition("sub", SubsystemState.STOPPING)
        registry.transition("sub", SubsystemState.STOPPED)
        assert registry.get("sub") is not None
        assert registry.all_entries()[0].state == SubsystemState.STOPPED

    def test_failure_from_running_is_legal(self) -> None:
        registry = _registry()
        registry.transition("sub", SubsystemState.STARTING)
        registry.transition("sub", SubsystemState.RUNNING)
        registry.transition("sub", SubsystemState.FAILED)
        assert registry.all_entries()[0].state == SubsystemState.FAILED

    def test_retry_from_failed_is_legal(self) -> None:
        registry = _registry()
        registry.transition("sub", SubsystemState.STARTING)
        registry.transition("sub", SubsystemState.FAILED)
        registry.transition("sub", SubsystemState.STARTING)
        registry.transition("sub", SubsystemState.RUNNING)
        assert registry.all_entries()[0].state == SubsystemState.RUNNING


class TestIllegalTransitions:
    def test_registered_to_running_raises(self) -> None:
        registry = _registry()
        with pytest.raises(InvalidStateTransitionError):
            registry.transition("sub", SubsystemState.RUNNING)

    def test_running_to_starting_raises(self) -> None:
        registry = _registry()
        registry.transition("sub", SubsystemState.STARTING)
        registry.transition("sub", SubsystemState.RUNNING)
        with pytest.raises(InvalidStateTransitionError):
            registry.transition("sub", SubsystemState.STARTING)

    def test_stopped_is_terminal(self) -> None:
        registry = _registry()
        registry.transition("sub", SubsystemState.STARTING)
        registry.transition("sub", SubsystemState.RUNNING)
        registry.transition("sub", SubsystemState.STOPPING)
        registry.transition("sub", SubsystemState.STOPPED)
        with pytest.raises(InvalidStateTransitionError):
            registry.transition("sub", SubsystemState.STARTING)

    def test_illegal_transition_leaves_state_unchanged(self) -> None:
        registry = _registry()
        registry.transition("sub", SubsystemState.STARTING)
        registry.transition("sub", SubsystemState.RUNNING)
        with pytest.raises(InvalidStateTransitionError):
            registry.transition("sub", SubsystemState.STOPPED)
        assert registry.all_entries()[0].state == SubsystemState.RUNNING

    def test_unknown_subsystem_raises_keyerror(self) -> None:
        registry = _registry()
        with pytest.raises(KeyError):
            registry.transition("missing", SubsystemState.STARTING)


class TestTransitionEdgeCases:
    def test_self_transition_is_noop(self) -> None:
        registry = _registry()
        registry.transition("sub", SubsystemState.STARTING)
        registry.transition("sub", SubsystemState.STARTING)
        assert registry.all_entries()[0].state == SubsystemState.STARTING

    def test_registered_to_starting_is_legal(self) -> None:
        registry = _registry()
        registry.transition("sub", SubsystemState.STARTING)
        assert registry.all_entries()[0].state == SubsystemState.STARTING
