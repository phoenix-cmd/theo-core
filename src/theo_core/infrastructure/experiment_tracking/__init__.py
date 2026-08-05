"""Experiment tracking adapters — NoOp, MLflow, and W&B implementations."""

from __future__ import annotations

import logging
from typing import Any

from theo_core.domain.research.ports.experiment_tracker import ExperimentTrackerPort

logger = logging.getLogger(__name__)


class NoOpExperimentTracker(ExperimentTrackerPort):
    """Silent experiment tracker that discards all tracking calls.

    Used as the default in testing and when no backend is configured.
    """

    def start_run(self, name: str, config: dict[str, Any] | None = None) -> str:
        """Start a no-op run. Returns a placeholder run ID."""
        del config  # Unused in NoOp implementation
        logger.debug("NoOp tracker: start_run(%s)", name)
        return "noop-run"

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Discard the metric."""
        logger.debug("NoOp tracker: log_metric(%s, %s, step=%s)", name, value, step)

    def log_param(self, name: str, value: Any) -> None:
        """Discard the parameter."""
        logger.debug("NoOp tracker: log_param(%s, %s)", name, value)

    def log_artifact(self, path: str) -> None:
        """Discard the artifact."""
        logger.debug("NoOp tracker: log_artifact(%s)", path)

    def end_run(self) -> None:
        """End the no-op run."""
        logger.debug("NoOp tracker: end_run()")


class ExperimentTrackerFactory:
    """Factory for creating experiment tracker instances from configuration.

    Supports 'noop', 'mlflow', and 'wandb' backends.
    MLflow and W&B are optional dependencies.
    """

    @staticmethod
    def create(backend: str = "noop", **kwargs: Any) -> ExperimentTrackerPort:
        """Create an experiment tracker from the backend name.

        Args:
            backend: The backend name (noop, mlflow, wandb).
            **kwargs: Backend-specific configuration.

        Returns:
            An ExperimentTrackerPort implementation.

        Raises:
            ValueError: If the backend is not recognized.
            ImportError: If the backend's optional dependency is missing.

        """
        if backend == "noop":
            return NoOpExperimentTracker()
        if backend == "mlflow":
            try:
                from theo_core.infrastructure.experiment_tracking.mlflow_tracker import (
                    MLflowExperimentTracker,
                )

                return MLflowExperimentTracker(**kwargs)
            except ImportError as e:
                msg = "MLflow is not installed. Install with: pip install theo-core[tracking]"
                raise ImportError(msg) from e
        if backend == "wandb":
            try:
                from theo_core.infrastructure.experiment_tracking.wandb_tracker import (
                    WandbExperimentTracker,
                )

                return WandbExperimentTracker(**kwargs)
            except ImportError as e:
                msg = "W&B is not installed. Install with: pip install theo-core[tracking]"
                raise ImportError(msg) from e

        msg = f"Unknown experiment tracking backend: {backend}"
        raise ValueError(msg)
