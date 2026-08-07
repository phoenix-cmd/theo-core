"""Schema versioning primitives for serialized symbolic artifacts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    """Immutable schema version tag embedded in all serialized artifacts.

    Attributes:
        major: Breaking changes increment this.
        minor: Backward-compatible changes increment this.

    """

    major: int
    minor: int

    def __str__(self) -> str:
        """Return ``major.minor`` string."""
        return f"{self.major}.{self.minor}"

    def is_compatible_with(self, other: SchemaVersion) -> bool:
        """Check backward compatibility (same major version).

        Args:
            other: The schema version to compare against.

        Returns:
            True if the major versions match.

        """
        return self.major == other.major
