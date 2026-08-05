"""Lifecycle — subsystem start/stop hooks and lifecycle management."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Startable(Protocol):
    """Protocol for subsystems that support a start lifecycle hook."""

    def start(self) -> None:
        """Start the subsystem."""
        ...


@runtime_checkable
class Stoppable(Protocol):
    """Protocol for subsystems that support a stop lifecycle hook."""

    def stop(self) -> None:
        """Stop the subsystem."""
        ...


class LifecycleManager:
    """Manages the start and stop lifecycle of subsystems.

    Subsystems that implement the Startable/Stoppable protocols
    have their lifecycle hooks called during boot and shutdown.
    """

    def start_subsystem(self, name: str, instance: Any) -> bool:
        """Start a subsystem if it implements Startable.

        Args:
            name: The subsystem name.
            instance: The subsystem object.

        Returns:
            True if started successfully, False if start failed.

        """
        if isinstance(instance, Startable):
            try:
                instance.start()
                logger.info("Subsystem started: %s", name)
                return True
            except Exception:
                logger.exception("Failed to start subsystem: %s", name)
                return False
        logger.debug("Subsystem %s has no start hook, skipping.", name)
        return True

    def stop_subsystem(self, name: str, instance: Any) -> bool:
        """Stop a subsystem if it implements Stoppable.

        Args:
            name: The subsystem name.
            instance: The subsystem object.

        Returns:
            True if stopped successfully, False if stop failed.

        """
        if isinstance(instance, Stoppable):
            try:
                instance.stop()
                logger.info("Subsystem stopped: %s", name)
                return True
            except Exception:
                logger.exception("Failed to stop subsystem: %s", name)
                return False
        logger.debug("Subsystem %s has no stop hook, skipping.", name)
        return True
