"""CapabilityPort — interface for pluggable cognitive capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CapabilityPort(ABC):
    """Abstract interface for a cognitive capability.

    Capabilities are broad functional contracts (planning, retrieval, etc.).
    Concrete skills implement specific instances of a capability.
    """

    @abstractmethod
    def execute(self, input_data: Any, context: dict[str, Any] | None = None) -> Any:
        """Execute this capability on the given input.

        Args:
            input_data: The input to process.
            context: Optional execution context.

        Returns:
            The capability output.

        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this capability."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version of this capability."""
