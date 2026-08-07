"""Unit tests for shared symbolic primitives (_primitives)."""

import pytest

from theo_core.symbolic._primitives.errors import (
    DuplicateIdError,
    SerializationError,
    SymbolicError,
    ValidationError,
)
from theo_core.symbolic._primitives.identifiers import SymbolicId
from theo_core.symbolic._primitives.ordering import sorted_items, sorted_keys, sorted_values
from theo_core.symbolic._primitives.versioning import SchemaVersion


class TestSymbolicId:
    def test_factory_valid_uri(self) -> None:
        sid = SymbolicId.of("concept://animal/dog")
        assert sid.value == "concept://animal/dog"
        assert str(sid) == "concept://animal/dog"

    def test_factory_invalid_uri_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid SymbolicId URI"):
            SymbolicId.of("INVALID_URI")

    def test_immutability(self) -> None:
        sid = SymbolicId.of("concept://animal/dog")
        with pytest.raises((TypeError, Exception)):
            sid.value = "concept://cat"  # type: ignore[misc]

    def test_equality_and_hashing(self) -> None:
        sid1 = SymbolicId.of("concept://animal/dog")
        sid2 = SymbolicId.of("concept://animal/dog")
        sid3 = SymbolicId.of("concept://animal/cat")

        assert sid1 == sid2
        assert sid1 != sid3
        assert hash(sid1) == hash(sid2)
        assert len({sid1, sid2, sid3}) == 2


class TestSchemaVersion:
    def test_version_string(self) -> None:
        sv = SchemaVersion(1, 0)
        assert str(sv) == "1.0"

    def test_compatibility(self) -> None:
        sv1 = SchemaVersion(1, 0)
        sv2 = SchemaVersion(1, 1)
        sv3 = SchemaVersion(2, 0)

        assert sv1.is_compatible_with(sv2)
        assert not sv1.is_compatible_with(sv3)


class TestOrderingUtilities:
    def test_sorted_items(self) -> None:
        d = {"c": 3, "a": 1, "b": 2}
        assert sorted_items(d) == [("a", 1), ("b", 2), ("c", 3)]

    def test_sorted_keys(self) -> None:
        d = {"c": 3, "a": 1, "b": 2}
        assert sorted_keys(d) == ["a", "b", "c"]

    def test_sorted_values(self) -> None:
        d = {"c": 3, "a": 1, "b": 2}
        assert sorted_values(d) == [1, 2, 3]


class TestErrorHierarchy:
    def test_inheritance(self) -> None:
        err = DuplicateIdError("duplicate id")
        assert isinstance(err, ValidationError)
        assert isinstance(err, SymbolicError)

        ser_err = SerializationError("failed")
        assert isinstance(ser_err, SymbolicError)
        assert not isinstance(ser_err, ValidationError)
