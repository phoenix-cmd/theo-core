"""Unit tests for Inference Engine domain models and rules."""

from decimal import Decimal

import pytest

from theo_core.symbolic.inference.models import (
    InferenceRule,
    RuleCondition,
    RuleId,
)


class TestInferenceRules:
    def test_rule_id_factory(self) -> None:
        rid = RuleId.of("rule://modus_ponens")
        assert rid.value == "rule://modus_ponens"
        assert str(rid) == "rule://modus_ponens"

    def test_invalid_rule_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with 'rule://'"):
            RuleId.of("invalid://scheme")

    def test_rule_creation_and_immutability(self) -> None:
        rid = RuleId.of("rule://r1")
        cond = RuleCondition(premise_predicate="user_likes_astronomy")
        rule = InferenceRule(
            id=rid,
            name="Astronomy interest rule",
            conditions=(cond,),
            conclusion_template="Recommend astronomy content",
            confidence_multiplier=Decimal("0.95"),
        )

        assert rule.id == rid
        assert rule.name == "Astronomy interest rule"
        assert len(rule.conditions) == 1
        assert rule.confidence_multiplier == Decimal("0.95")

        with pytest.raises((TypeError, Exception)):
            rule.name = "Modified name"  # type: ignore[misc]
