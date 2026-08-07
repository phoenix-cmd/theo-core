"""Property tests — identifier determinism and ID scheme invariants.

Identifiers MUST be pure functions of their content: identical content always
yields an identical identifier, and the URI scheme invariants MUST hold for
all well-formed inputs.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic.beliefs.models import BeliefId
from theo_core.symbolic.perception.models import PerceptId

_ANY_URI = st.from_regex(r"[a-z][a-z0-9_]*://[a-z0-9_./-]+", fullmatch=True)
_BELIEF_URI = st.from_regex(r"belief://[a-z0-9_./-]+", fullmatch=True)
_Percept_URI = st.from_regex(r"percept://[a-z0-9_./-]+", fullmatch=True)
_MAX_EXAMPLES = settings(max_examples=50)


class TestIdentifierDeterminism:
    @given(_ANY_URI)
    @_MAX_EXAMPLES
    def test_symbolic_id_is_pure(self, value: str) -> None:
        assert SymbolicId.of(value).value == SymbolicId.of(value).value

    @given(_ANY_URI)
    @_MAX_EXAMPLES
    def test_symbolic_id_equality_is_value_based(self, value: str) -> None:
        assert SymbolicId.of(value) == SymbolicId.of(value)

    @given(_BELIEF_URI)
    @_MAX_EXAMPLES
    def test_belief_id_is_pure(self, value: str) -> None:
        assert BeliefId.of(value).value == BeliefId.of(value).value

    @given(_Percept_URI)
    @_MAX_EXAMPLES
    def test_percept_id_uri_scheme(self, value: str) -> None:
        pid = PerceptId.of(value)
        assert pid.value.startswith("percept://")

    @given(st.text(min_size=1, max_size=64))
    @_MAX_EXAMPLES
    def test_percept_id_rejects_non_uri(self, content: str) -> None:
        assume(not content.startswith("percept://"))
        try:
            PerceptId.of(content)
        except ValueError:
            return
        raise AssertionError(
            f"PerceptId.of({content!r}) should have raised ValueError"
        )
