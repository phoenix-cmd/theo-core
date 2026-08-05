"""Tests for research domain entities."""

from __future__ import annotations

from theo_core.domain.research.entities.checkpoint import Checkpoint
from theo_core.domain.research.entities.dataset import DatasetManifest, DatasetSample
from theo_core.domain.research.entities.evaluation import Benchmark, EvaluationResult, Metric
from theo_core.domain.research.entities.experiment import Experiment, ExperimentStatus
from theo_core.domain.research.entities.training_run import TrainingRun


class TestExperiment:
    """Tests for the Experiment aggregate."""

    def test_create_experiment(self) -> None:
        """An experiment should start in PLANNED status."""
        exp = Experiment(name="test-experiment")
        assert exp.status == ExperimentStatus.PLANNED
        assert exp.name == "test-experiment"


class TestTrainingRun:
    """Tests for the TrainingRun entity."""

    def test_log_metric(self) -> None:
        """Logging a metric should append to the metrics dict."""
        run = TrainingRun()
        run.log_metric("loss", 0.5, step=1)
        run.log_metric("loss", 0.3, step=2)
        assert len(run.metrics["loss"]) == 2
        assert run.metrics["loss"][0] == (1, 0.5)


class TestCheckpoint:
    """Tests for the Checkpoint entity."""

    def test_create_checkpoint(self) -> None:
        """A checkpoint should record step and path."""
        cp = Checkpoint(step=1000, path="/checkpoints/model_1000.pt")
        assert cp.step == 1000


class TestDataset:
    """Tests for DatasetManifest and DatasetSample."""

    def test_create_manifest(self) -> None:
        """A manifest should capture versioning metadata."""
        manifest = DatasetManifest(
            name="test-dataset",
            version="1.0.0",
            checksum="abc123",
            source="synthetic",
            license="MIT",
        )
        assert manifest.version == "1.0.0"

    def test_create_sample(self) -> None:
        """A sample should hold features and labels."""
        sample = DatasetSample(
            features={"text": "hello"},
            labels={"sentiment": "positive"},
            split="train",
        )
        assert sample.split == "train"


class TestEvaluation:
    """Tests for Metric, EvaluationResult, and Benchmark."""

    def test_create_metric(self) -> None:
        """A metric should be immutable with a value and direction."""
        metric = Metric(name="accuracy", value=0.95, higher_is_better=True)
        assert metric.value == 0.95

    def test_create_evaluation_result(self) -> None:
        """An evaluation result should contain metrics."""
        metric = Metric(name="accuracy", value=0.95)
        result = EvaluationResult(
            benchmark_name="test-bench",
            metrics=(metric,),
        )
        assert len(result.metrics) == 1

    def test_create_benchmark(self) -> None:
        """A benchmark should define expected metric names."""
        bench = Benchmark(
            name="reasoning-v1",
            metric_names=["accuracy", "consistency"],
        )
        assert len(bench.metric_names) == 2
