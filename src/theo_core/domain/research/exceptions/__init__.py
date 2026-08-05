"""Research domain exceptions."""

from __future__ import annotations

from theo_core.domain.runtime.exceptions import TheoError


class ExperimentError(TheoError):
    """Raised when an experiment operation fails."""


class DatasetError(TheoError):
    """Raised when a dataset operation fails."""


class RegistryError(TheoError):
    """Raised when a registry operation fails."""


class CheckpointError(TheoError):
    """Raised when checkpointing fails."""
