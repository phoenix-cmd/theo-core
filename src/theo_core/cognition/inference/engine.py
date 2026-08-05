"""InferenceEngine — orchestrates cognitive inference execution via InferenceStrategy.

Decides 'What follows from available information?' by delegating policy evaluation
to the configured InferenceStrategy (e.g. RuleBasedStrategy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from theo_core.cognition.inference.strategies.base import InferenceStrategy
    from theo_core.domain.runtime.entities.cognitive_state import CognitiveState


class InferenceEngine:
    """Orchestrates cognitive inference strategy execution.

    Delegates decision policy evaluation to the registered strategy.
    """

    def __init__(self, strategy: InferenceStrategy) -> None:
        """Initialize InferenceEngine with a strategy.

        Args:
            strategy: The active InferenceStrategy implementation.

        """
        self._strategy = strategy

    def infer(self, state: CognitiveState) -> dict[str, Any]:
        """Execute cognitive inference over the current state.

        Args:
            state: The current CognitiveState object.

        Returns:
            Dictionary containing inferred decision results and reasoning details.

        """
        return self._strategy.evaluate(state)

    @property
    def active_strategy_name(self) -> str:
        """Return the name of the active strategy."""
        return self._strategy.name
