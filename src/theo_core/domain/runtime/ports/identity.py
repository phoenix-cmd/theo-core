"""IdentityPort — interface for Theo's persistent self-model.

Identity is not a behavior — it is what Theo is. It governs persona,
values, alignment rules, and long-term behavioral consistency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.cognitive_state import CognitiveState


class IdentityPort(ABC):
    """Abstract interface for Theo's identity system."""

    @abstractmethod
    def get_state(self) -> CognitiveState:
        """Return a snapshot of the current cognitive/identity state.

        Returns:
            The current CognitiveState.

        """

    @abstractmethod
    def update(self, event_data: dict[str, Any]) -> None:
        """Update the identity state based on a cognitive event.

        Args:
            event_data: Data from the event that triggered the update.

        """

    @abstractmethod
    def describe(self) -> str:
        """Return a natural language description of Theo's current identity.

        Returns:
            A string describing the current identity state.

        """
