"""ResponseGeneratorPort — abstract interface for response generation subsystems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.decision import Decision


class ResponseGeneratorPort(ABC):
    """Abstract interface for generating user-visible response text from a Decision."""

    @abstractmethod
    def generate(self, decision: Decision) -> str:
        """Generate formatted response text from a Decision object.

        Args:
            decision: The decision formulated by the Decision Engine.

        Returns:
            The final formatted response text string.

        """
