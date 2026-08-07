"""SubsystemRegistry — tracks all registered subsystems and their lifecycle state."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class SubsystemState(StrEnum):
    """Lifecycle state of a registered subsystem."""

    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class InvalidStateTransitionError(RuntimeError):
    """Raised when a subsystem attempts an illegal lifecycle transition."""

    def __init__(self, name: str, current: SubsystemState, target: SubsystemState) -> None:
        """Initialize with the offending subsystem and states."""
        super().__init__(
            f"Illegal subsystem state transition for '{name}': "
            f"{current.value} -> {target.value}"
        )
        self.name = name
        self.current = current
        self.target = target


LEGAL_TRANSITIONS: dict[SubsystemState, frozenset[SubsystemState]] = {
    SubsystemState.REGISTERED: frozenset({SubsystemState.STARTING}),
    SubsystemState.STARTING: frozenset(
        {SubsystemState.RUNNING, SubsystemState.STOPPING, SubsystemState.FAILED}
    ),
    SubsystemState.RUNNING: frozenset(
        {SubsystemState.STOPPING, SubsystemState.FAILED}
    ),
    SubsystemState.STOPPING: frozenset(
        {SubsystemState.STOPPED, SubsystemState.FAILED}
    ),
    SubsystemState.STOPPED: frozenset(),
    SubsystemState.FAILED: frozenset({SubsystemState.STARTING}),
}


class SubsystemEntry:
    """A registered subsystem with its state and metadata.

    Attributes:
        name: Unique subsystem name.
        instance: The live subsystem object.
        state: Current lifecycle state.
        version: Subsystem version string.

    """

    def __init__(self, name: str, instance: Any, version: str = "0.1.0") -> None:
        """Initialize a subsystem entry.

        Args:
            name: Unique subsystem name.
            instance: The subsystem object.
            version: Version string.

        """
        self.name = name
        self.instance = instance
        self.state = SubsystemState.REGISTERED
        self.version = version


class SubsystemRegistry:
    """Registry of all subsystems known to the kernel.

    Provides registration, lookup, and state management for subsystems.
    """

    def __init__(self) -> None:
        """Initialize an empty subsystem registry."""
        self._entries: dict[str, SubsystemEntry] = {}

    def register(self, name: str, instance: Any, version: str = "0.1.0") -> None:
        """Register a subsystem.

        Args:
            name: Unique subsystem name.
            instance: The subsystem object.
            version: Version string.

        """
        self._entries[name] = SubsystemEntry(name, instance, version)
        logger.info("Subsystem registered: %s (v%s)", name, version)

    def get(self, name: str) -> Any | None:
        """Get a subsystem instance by name.

        Args:
            name: The subsystem name.

        Returns:
            The subsystem instance, or None if not found.

        """
        entry = self._entries.get(name)
        return entry.instance if entry else None

    def set_state(self, name: str, state: SubsystemState) -> None:
        """Update the state of a registered subsystem.

        Low-level permissive setter; prefer :meth:`transition` for lifecycle
        enforcement.

        Args:
            name: The subsystem name.
            state: The new state.

        """
        if name in self._entries:
            self._entries[name].state = state
            logger.debug("Subsystem %s → %s", name, state)

    def transition(self, name: str, target: SubsystemState) -> None:
        """Transition a subsystem to a new state, validating legality.

        Raises:
            KeyError: If the subsystem is not registered.
            InvalidStateTransitionError: If the transition is not legal.

        Args:
            name: The subsystem name.
            target: The target state.

        """
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"Unknown subsystem: {name}")
        if entry.state == target:
            return
        if target not in LEGAL_TRANSITIONS[entry.state]:
            raise InvalidStateTransitionError(name, entry.state, target)
        entry.state = target
        logger.debug("Subsystem %s → %s", name, target)

    def all_entries(self) -> list[SubsystemEntry]:
        """Return all registered subsystem entries.

        Returns:
            A list of all SubsystemEntry objects.

        """
        return list(self._entries.values())

    @property
    def count(self) -> int:
        """Return the number of registered subsystems."""
        return len(self._entries)

    def is_all_running(self) -> bool:
        """Check whether all subsystems are in the RUNNING state.

        Returns:
            True if all subsystems are running.

        """
        return all(e.state == SubsystemState.RUNNING for e in self._entries.values())
