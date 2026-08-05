"""ContextPort — interface for managing ephemeral active state.

Context is temporary and session-bound. Memory is persistent.
Working memory consumes Context but does not own it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ContextPort(ABC):
    """Abstract interface for active context management."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Retrieve a value from the active context.

        Args:
            key: The context key.

        Returns:
            The value, or None if not present.

        """

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set a value in the active context.

        Args:
            key: The context key.
            value: The value to store.

        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all values from the active context."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the current context state.

        Returns:
            A dictionary of all key-value pairs in the context.

        """
