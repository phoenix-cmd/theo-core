"""Research ports — TrainerPort, DatasetPort, EvaluatorPort, BenchmarkPort, RegistryPort."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from theo_core.domain.research.entities.checkpoint import Checkpoint
    from theo_core.domain.research.entities.evaluation import Benchmark, EvaluationResult
    from theo_core.domain.research.entities.training_run import TrainingRun


class TrainerPort(ABC):
    """Abstract interface for model training."""

    @abstractmethod
    def train(self, config: dict[str, Any]) -> TrainingRun:
        """Execute a training run.

        Args:
            config: Training configuration.

        Returns:
            The completed TrainingRun.

        """

    @abstractmethod
    def checkpoint(self) -> Checkpoint:
        """Save a checkpoint of the current model state.

        Returns:
            The saved Checkpoint.

        """


class DatasetPort(ABC):
    """Abstract interface for dataset operations."""

    @abstractmethod
    def load(self) -> Any:
        """Load the dataset.

        Returns:
            The loaded dataset object.

        """

    @abstractmethod
    def split(self, ratios: dict[str, float]) -> dict[str, Any]:
        """Split the dataset into named subsets.

        Args:
            ratios: A mapping of split name to fraction (e.g. {"train": 0.8}).

        Returns:
            A dictionary mapping split name to dataset subset.

        """

    @abstractmethod
    def iter_batches(self, batch_size: int = 32) -> Iterator[Any]:
        """Iterate over the dataset in batches.

        Args:
            batch_size: Number of samples per batch.

        Yields:
            Batches of samples.

        """


class EvaluatorPort(ABC):
    """Abstract interface for model evaluation."""

    @abstractmethod
    def evaluate(self, model: Any, benchmark: Benchmark) -> EvaluationResult:
        """Evaluate a model against a benchmark.

        Args:
            model: The model to evaluate.
            benchmark: The benchmark definition.

        Returns:
            The evaluation result.

        """

    @abstractmethod
    def compare(self, result_a: EvaluationResult, result_b: EvaluationResult) -> dict[str, float]:
        """Compare two evaluation results.

        Args:
            result_a: First result.
            result_b: Second result.

        Returns:
            A dictionary of metric deltas.

        """


class BenchmarkPort(ABC):
    """Abstract interface for benchmark management."""

    @abstractmethod
    def load(self, name: str) -> Benchmark:
        """Load a benchmark by name.

        Args:
            name: The benchmark name.

        Returns:
            The loaded Benchmark definition.

        """

    @abstractmethod
    def list_benchmarks(self) -> list[str]:
        """List all available benchmark names.

        Returns:
            A list of benchmark name strings.

        """


class RegistryPort(ABC):
    """Abstract interface for the artifact registry."""

    @abstractmethod
    def register(self, entry: Any) -> str:
        """Register an artifact in the registry.

        Args:
            entry: The artifact entry to register.

        Returns:
            The unique ID of the registered entry.

        """

    @abstractmethod
    def get(self, artifact_id: str) -> Any | None:
        """Retrieve an artifact by ID.

        Args:
            artifact_id: The unique artifact ID.

        Returns:
            The artifact entry, or None if not found.

        """

    @abstractmethod
    def list_entries(self, artifact_type: str | None = None) -> list[Any]:
        """List registry entries, optionally filtered by type.

        Args:
            artifact_type: Optional filter by artifact type.

        Returns:
            A list of registry entries.

        """

    @abstractmethod
    def update_status(self, artifact_id: str, status: str) -> None:
        """Update the status of a registry entry.

        Args:
            artifact_id: The artifact ID.
            status: The new status string.

        """


class SchedulerPort(ABC):
    """Abstract interface for background job scheduling."""

    @abstractmethod
    def schedule(self, job_id: str, func: Any, trigger: str, **kwargs: Any) -> None:
        """Schedule a background job.

        Args:
            job_id: Unique job identifier.
            func: The callable to execute.
            trigger: Trigger type (e.g. "interval", "cron").
            **kwargs: Trigger-specific parameters.

        """

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a scheduled job.

        Args:
            job_id: The job ID to cancel.

        Returns:
            True if the job was cancelled.

        """

    @abstractmethod
    def list_jobs(self) -> list[dict[str, Any]]:
        """List all scheduled jobs.

        Returns:
            A list of job info dicts.

        """


class SecretManagerPort(ABC):
    """Abstract interface for secret management."""

    @abstractmethod
    def get(self, key: str) -> str:
        """Retrieve a secret value.

        Args:
            key: The secret key.

        Returns:
            The secret value string.

        Raises:
            KeyError: If the secret is not found.

        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether a secret exists.

        Args:
            key: The secret key.

        Returns:
            True if the secret exists.

        """
