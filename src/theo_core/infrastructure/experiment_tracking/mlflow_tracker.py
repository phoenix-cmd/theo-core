"""MLflow experiment tracker adapter — optional dependency."""

from __future__ import annotations

from typing import Any

from theo_core.domain.research.ports.experiment_tracker import ExperimentTrackerPort


class MLflowExperimentTracker(ExperimentTrackerPort):
    """Experiment tracker backed by MLflow.

    Requires the 'tracking' extra: pip install theo-core[tracking]
    """

    def __init__(self, tracking_uri: str = "", experiment_name: str = "theo") -> None:
        """Initialize the MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: Name of the MLflow experiment.

        """
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._mlflow = mlflow

    def start_run(self, name: str, config: dict[str, Any] | None = None) -> str:
        """Start an MLflow run."""
        run = self._mlflow.start_run(run_name=name)
        if config:
            self._mlflow.log_params(config)
        return str(run.info.run_id)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a metric to MLflow."""
        self._mlflow.log_metric(name, value, step=step)

    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter to MLflow."""
        self._mlflow.log_param(name, value)

    def log_artifact(self, path: str) -> None:
        """Log an artifact file to MLflow."""
        self._mlflow.log_artifact(path)

    def end_run(self) -> None:
        """End the current MLflow run."""
        self._mlflow.end_run()
