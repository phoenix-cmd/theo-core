"""Unit tests for Hypothesis Engine domain models."""

from decimal import Decimal

import pytest

from theo_core.symbolic.hypotheses.models import (
    Hypothesis,
    HypothesisId,
    HypothesisState,
)


class TestHypothesisModels:
    def test_hypothesis_id_factory(self) -> None:
        hid = HypothesisId.of("hypothesis://cand_1")
        assert hid.value == "hypothesis://cand_1"
        assert str(hid) == "hypothesis://cand_1"

    def test_invalid_hypothesis_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with 'hypothesis://'"):
            HypothesisId.of("invalid://scheme")

    def test_hypothesis_creation_and_immutability(self) -> None:
        hid = HypothesisId.of("hypothesis://cand_1")
        h = Hypothesis(
            id=hid,
            interpretation="User wants code refactoring",
            score=Decimal("0.85"),
            state=HypothesisState.CANDIDATE,
        )

        assert h.id == hid
        assert h.interpretation == "User wants code refactoring"
        assert h.score == Decimal("0.85")
        assert h.state == HypothesisState.CANDIDATE

        with pytest.raises((TypeError, Exception)):
            h.score = Decimal("0.9")  # type: ignore[misc]
