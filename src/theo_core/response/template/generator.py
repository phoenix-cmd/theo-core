"""TemplateResponseGenerator — deterministic template-rendered response generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.response.ports.response_generator import ResponseGeneratorPort

if TYPE_CHECKING:
    from theo_core.domain.runtime.entities.decision import Decision


class TemplateResponseGenerator(ResponseGeneratorPort):
    """Deterministic template-based response generator for v0.2.

    Formats decision payloads into final output strings.
    """

    def generate(self, decision: Decision) -> str:
        """Generate response text string from a Decision object.

        Args:
            decision: The decision formulated by the Decision Engine.

        Returns:
            Formatted response text string.

        """
        return decision.response.strip()
