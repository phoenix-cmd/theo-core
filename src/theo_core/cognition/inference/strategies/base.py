"""InferenceStrategy — abstract port for cognitive inference strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.cognitive_state import CognitiveState


class InferenceStrategy(ABC):
    """Abstract interface for cognitive inference strategies.

    Concrete implementations execute policy evaluation over CognitiveState.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this inference strategy."""

    @abstractmethod
    def evaluate(self, state: CognitiveState) -> dict[str, Any]:
        """Evaluate cognitive policy over the current state.

        Args:
            state: The current CognitiveState object.

        Returns:
            A dictionary containing candidate thoughts, actions, and response template parameters.

        """
