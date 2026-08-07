"""Unit tests for the boundary ResponseRenderer (Canon Law 6)."""

from decimal import Decimal

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.decisions.models import (
    ActionSpec,
    DecisionId,
    DecisionRecord,
    DecisionType,
    Intent,
)
from theo_core.symbolic.response.renderer import TemplateResponseRenderer
from theo_core.symbolic.thoughts.models import ThoughtId


def _decision(
    *,
    decision_type: DecisionType = DecisionType.RESPONSE,
    action_text: str = "Hello Falcon!",
    capability: str = "respond",
    parameters: dict[str, object] | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        id=DecisionId.of("decision://test/1"),
        type=decision_type,
        action_text=action_text,
        referenced_goal=SymbolicId.of("goal://maintainconversation"),
        intent=Intent.MAINTAIN_CONVERSATION,
        action_spec=ActionSpec(
            capability=capability,
            parameters=parameters or {},
        ),
        confidence=Decimal("0.9"),
        referenced_thoughts=(ThoughtId.of("thought://sys/inference_rule"),),
    )


class TestTemplateResponseRenderer:
    def test_render_defer_uses_fixed_template(self) -> None:
        """Deferred decisions MUST render a fixed deferral template."""
        renderer = TemplateResponseRenderer()
        defer = _decision(
            decision_type=DecisionType.DEFER,
            action_text="Insufficient confidence; deferring decision.",
            capability="defer",
        )

        out = renderer.render(defer)

        assert out == "I'm not confident enough to respond to that yet."

    def test_render_response_passes_through_action_spec_content(self) -> None:
        """Response decisions MUST render the structured content from the ActionSpec."""
        renderer = TemplateResponseRenderer()
        decision = _decision(
            action_text="fallback text",
            parameters={"content": "Hello Falcon!"},
        )

        out = renderer.render(decision)

        assert out == "Hello Falcon!"

    def test_render_response_falls_back_to_action_text(self) -> None:
        """Response decisions without content MUST fall back to action_text."""
        renderer = TemplateResponseRenderer()
        decision = _decision(parameters={})

        out = renderer.render(decision)

        assert out == "Hello Falcon!"
