"""Weights & Biases experiment tracker adapter — optional dependency."""

from __future__ import annotations

from typing import Any

from theo_core.domain.research.ports.experiment_tracker import ExperimentTrackerPort


class WandbExperimentTracker(ExperimentTrackerPort):
    """Experiment tracker backed by Weights & Biases.

    Requires the 'tracking' extra: pip install theo-core[tracking]
    """

    def __init__(self, project: str = "theo", **kwargs: Any) -> None:
        """Initialize the W&B tracker.

        Args:
            project: W&B project name.
            **kwargs: Additional wandb.init parameters.

        """
        import wandb

        self._wandb = wandb
        self._project = project
        self._init_kwargs = kwargs
        self._run: Any = None

    def start_run(self, name: str, config: dict[str, Any] | None = None) -> str:
        """Start a W&B run."""
        self._run = self._wandb.init(
            project=self._project, name=name, config=config, **self._init_kwargs
        )
        return str(self._run.id)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a metric to W&B."""
        log_data: dict[str, Any] = {name: value}
        if step is not None:
            log_data["step"] = step
        self._wandb.log(log_data)

    def log_param(self, name: str, value: Any) -> None:
        """Log a parameter to W&B config."""
        if self._run:
            self._run.config[name] = value

    def log_artifact(self, path: str) -> None:
        """Log an artifact to W&B."""
        artifact = self._wandb.Artifact(name="artifact", type="file")
        artifact.add_file(path)
        if self._run:
            self._run.log_artifact(artifact)

    def end_run(self) -> None:
        """End the current W&B run."""
        self._wandb.finish()
