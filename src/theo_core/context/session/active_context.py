"""InMemoryContextManager — ephemeral session context management.

Maintains current turn history, active user ID, current goals, and temporary execution state.
Purely in-memory, non-persistent.
"""

from __future__ import annotations

from typing import Any

from theo_core.domain.runtime.ports.context import ContextPort


class InMemoryContextManager(ContextPort):
    """In-memory active context manager.

    Holds ephemeral session variables for the current cognitive session.
    Is not saved to disk — persistent facts belong in Memory/Knowledge.
    """

    def __init__(self, active_user: str = "anonymous") -> None:
        """Initialize ephemeral context.

        Args:
            active_user: Initial active user identifier.

        """
        self._store: dict[str, Any] = {
            "active_user": active_user,
            "turn_count": 0,
            "last_percept": None,
            "last_response": None,
        }

    def get(self, key: str) -> Any | None:
        """Retrieve a value from the active context.

        Args:
            key: The context key.

        Returns:
            The value, or None if not found.

        """
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the active context.

        Args:
            key: The context key.
            value: The value to store.

        """
        self._store[key] = value

    def clear(self) -> None:
        """Clear all context entries except active_user."""
        user = self._store.get("active_user", "anonymous")
        self._store.clear()
        self._store["active_user"] = user
        self._store["turn_count"] = 0

    def increment_turns(self) -> int:
        """Increment and return the current conversation turn count.

        Returns:
            The updated integer turn count.

        """
        count = int(self._store.get("turn_count", 0)) + 1
        self._store["turn_count"] = count
        return count

    def snapshot(self) -> dict[str, Any]:
        """Return a copy snapshot of the current active context.

        Returns:
            A dictionary of all active context key-value pairs.

        """
        return dict(self._store)
