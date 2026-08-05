"""Tests for optional experiment tracking adapters and CLI."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from theo_core.__main__ import app, boot, main
from theo_core.infrastructure.experiment_tracking import ExperimentTrackerFactory
from theo_core.infrastructure.experiment_tracking.mlflow_tracker import MLflowExperimentTracker
from theo_core.infrastructure.experiment_tracking.wandb_tracker import WandbExperimentTracker


class TestMLflowTracker:
    """Tests for MLflowExperimentTracker with mocked mlflow module."""

    def test_mlflow_tracker_methods(self) -> None:
        """MLflow tracker should invoke underlying mlflow functions."""
        mock_mlflow = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "mlflow-run-123"
        mock_mlflow.start_run.return_value = mock_run

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            tracker = MLflowExperimentTracker(
                tracking_uri="http://localhost:5000", experiment_name="test"
            )
            mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
            mock_mlflow.set_experiment.assert_called_once_with("test")

            run_id = tracker.start_run("my_run", {"lr": 0.01})
            assert run_id == "mlflow-run-123"

            tracker.log_metric("acc", 0.9, step=1)
            mock_mlflow.log_metric.assert_called_once_with("acc", 0.9, step=1)

            tracker.log_param("optimizer", "adam")
            mock_mlflow.log_param.assert_called_once_with("optimizer", "adam")

            tracker.log_artifact("/path/to/file")
            mock_mlflow.log_artifact.assert_called_once_with("/path/to/file")

            tracker.end_run()
            mock_mlflow.end_run.assert_called_once()

    def test_factory_creates_mlflow(self) -> None:
        """Factory should create MLflow tracker when backend='mlflow'."""
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            tracker = ExperimentTrackerFactory.create("mlflow")
            assert isinstance(tracker, MLflowExperimentTracker)


class TestWandbTracker:
    """Tests for WandbExperimentTracker with mocked wandb module."""

    def test_wandb_tracker_methods(self) -> None:
        """W&B tracker should invoke underlying wandb functions."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_run.id = "wandb-run-456"
        mock_run.config = {}
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            tracker = WandbExperimentTracker(project="test_proj")
            run_id = tracker.start_run("my_wandb_run", {"lr": 0.01})
            assert run_id == "wandb-run-456"

            tracker.log_metric("loss", 0.1, step=5)
            mock_wandb.log.assert_called_once_with({"loss": 0.1, "step": 5})

            tracker.log_param("epochs", 10)
            assert mock_run.config["epochs"] == 10

            tracker.log_artifact("/path/to/artifact")
            mock_wandb.Artifact.assert_called_once()

            tracker.end_run()
            mock_wandb.finish.assert_called_once()

    def test_factory_creates_wandb(self) -> None:
        """Factory should create W&B tracker when backend='wandb'."""
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            tracker = ExperimentTrackerFactory.create("wandb")
            assert isinstance(tracker, WandbExperimentTracker)


class TestCLI:
    """Tests for Typer CLI app."""

    def test_app_help(self) -> None:
        """App should be a valid Typer instance."""
        assert app.info.name == "theo"

    def test_boot_function(self) -> None:
        """Boot function should execute kernel boot."""
        boot()

    def test_main_function(self) -> None:
        """Main function should be callable."""
        assert callable(main)
