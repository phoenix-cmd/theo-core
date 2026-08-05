"""Tests for infrastructure — logging, config, and experiment tracking."""

from __future__ import annotations

import pytest

from theo_core.infrastructure.config import TheoSettings
from theo_core.infrastructure.experiment_tracking import (
    ExperimentTrackerFactory,
    NoOpExperimentTracker,
)


class TestTheoSettings:
    """Tests for the configuration system."""

    def test_default_settings(self) -> None:
        """Default settings should create valid nested configs."""
        settings = TheoSettings()
        assert settings.logging.level == "INFO"
        assert settings.model.device == "cpu"
        assert settings.experiment_tracking.backend == "noop"

    def test_override_settings(self) -> None:
        """Settings should accept nested overrides."""
        settings = TheoSettings(
            logging={"level": "DEBUG", "format": "console"},  # type: ignore[arg-type]
            model={"device": "cuda", "dtype": "bfloat16"},  # type: ignore[arg-type]
        )
        assert settings.logging.level == "DEBUG"
        assert settings.model.device == "cuda"

    def test_kernel_boot_order(self) -> None:
        """Kernel settings should include a default boot order."""
        settings = TheoSettings()
        assert "event_bus" in settings.kernel.subsystem_start_order


class TestNoOpTracker:
    """Tests for the NoOp experiment tracker."""

    def test_full_lifecycle(self) -> None:
        """NoOp tracker should complete a full run lifecycle silently."""
        tracker = NoOpExperimentTracker()
        run_id = tracker.start_run("test-run", {"lr": 0.001})
        assert run_id == "noop-run"
        tracker.log_metric("loss", 0.5, step=1)
        tracker.log_param("lr", 0.001)
        tracker.log_artifact("/path/to/file")
        tracker.end_run()


class TestExperimentTrackerFactory:
    """Tests for the tracker factory."""

    def test_create_noop(self) -> None:
        """Factory should create a NoOp tracker."""
        tracker = ExperimentTrackerFactory.create("noop")
        assert isinstance(tracker, NoOpExperimentTracker)

    def test_create_unknown_raises(self) -> None:
        """Factory should raise ValueError for unknown backends."""
        with pytest.raises(ValueError, match="Unknown"):
            ExperimentTrackerFactory.create("unknown_backend")
