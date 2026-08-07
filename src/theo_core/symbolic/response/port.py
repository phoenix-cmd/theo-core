"""ResponseRendererPort — boundary interface for rendering Decisions into text.

The pipeline never generates language (Canon Law 6); it produces a structured
DecisionRecord (Intent + ActionSpec). This port owns the text-rendering concern
at the runtime boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theo_core.symbolic.decisions.models import DecisionRecord


class ResponseRendererPort(ABC):
    """Boundary adapter interface: converts a DecisionRecord into user-facing text."""

    @abstractmethod
    def render(self, decision: DecisionRecord) -> str:
        """Render user-facing response text from a DecisionRecord.

        Args:
            decision: The immutable DecisionRecord produced by the pipeline.

        Returns:
            The final response text string.

        """
