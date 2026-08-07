"""Stable identity primitives for all symbolic subsystems."""

from __future__ import annotations

import re
from dataclasses import dataclass

_URI_PATTERN = re.compile(r"^[a-z][a-z0-9_]*://[a-z0-9_./-]+$")


@dataclass(frozen=True, slots=True)
class SymbolicId:
    """Immutable, hashable URI-style identifier.

    Validation happens via the ``of`` factory method.
    Direct construction is permitted for trusted internal callers
    (e.g., deserialization of already-validated data).

    Equality is identity-based: two ``SymbolicId`` instances are equal
    if and only if their ``value`` fields are equal.
    """

    value: str

    @classmethod
    def of(cls, value: str) -> SymbolicId:
        """Create a ``SymbolicId`` with URI validation.

        Args:
            value: A URI string matching ``scheme://path`` format.
                   Scheme and path must be lowercase alphanumeric with
                   underscores, dots, hyphens, and slashes.

        Returns:
            A validated ``SymbolicId``.

        Raises:
            ValueError: If the URI format is invalid.

        """
        if not _URI_PATTERN.match(value):
            msg = (
                f"Invalid SymbolicId URI: {value!r}. "
                f"Expected format: scheme://path (lowercase alphanumeric, _, ., -, /)"
            )
            raise ValueError(msg)
        return cls(value=value)

    def __str__(self) -> str:
        """Return the raw URI string."""
        return self.value
