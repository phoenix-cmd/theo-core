"""ExperimentTrackerPort — interface for experiment tracking backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExperimentTrackerPort(ABC):
    """Abstract interface for experiment tracking.

    Implementations may wrap MLflow, W&B, TensorBoard, or a NoOp logger.
    """

    @abstractmethod
    def start_run(self, name: str, config: dict[str, Any] | None = None) -> str:
        """Start a new experiment run.

        Args:
            name: Name of the run.
            config: Configuration parameters to log.

        Returns:
            A unique run ID.

        """

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step number.

        """

    @abstractmethod
    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter.

        Args:
            name: Parameter name.
            value: Parameter value.

        """

    @abstractmethod
    def log_artifact(self, path: str) -> None:
        """Log an artifact file.

        Args:
            path: Path to the artifact file.

        """

    @abstractmethod
    def end_run(self) -> None:
        """End the current experiment run."""
