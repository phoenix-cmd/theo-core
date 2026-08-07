"""TemplateResponseRenderer — deterministic response renderer for the symbolic runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from theo_core.symbolic.decisions.models import DecisionType
from theo_core.symbolic.response.port import ResponseRendererPort

if TYPE_CHECKING:
    from theo_core.symbolic.decisions.models import DecisionRecord

_DEFER_RESPONSE = "I'm not confident enough to respond to that yet."


class TemplateResponseRenderer(ResponseRendererPort):
    """Deterministic template-based response renderer (Canon Law 6).

    Deferrals use a fixed template; response decisions pass through the
    structured content selected by the decision engine.
    """

    def render(self, decision: DecisionRecord) -> str:
        """Render user-facing response text from a DecisionRecord.

        Args:
            decision: The immutable DecisionRecord produced by the pipeline.

        Returns:
            The final response text string.

        """
        if decision.type == DecisionType.DEFER:
            return _DEFER_RESPONSE
        content = decision.action_spec.parameters.get("content")
        if isinstance(content, str) and content:
            return content
        return decision.action_text
