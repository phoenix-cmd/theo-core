"""Runtime domain exceptions — all errors originating from cognitive operations."""

from __future__ import annotations


class TheoError(Exception):
    """Base exception for all THEO cognitive errors."""


class MemoryError(TheoError):
    """Raised when a memory operation fails."""


class InferenceError(TheoError):
    """Raised when model inference fails."""


class ConfigurationError(TheoError):
    """Raised when configuration is invalid or missing."""


class TrainingError(TheoError):
    """Raised when a training operation fails."""


class EvaluationError(TheoError):
    """Raised when an evaluation operation fails."""


class PerceptionError(TheoError):
    """Raised when perceptual processing fails."""


class KnowledgeError(TheoError):
    """Raised when a knowledge graph operation fails."""


class GoalError(TheoError):
    """Raised when goal management encounters an error."""


class PluginError(TheoError):
    """Raised when a plugin fails to load or execute."""
