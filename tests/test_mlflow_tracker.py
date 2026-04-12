"""
Tests for MLflow Tracker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestMLflowTracker:
    """Validate MLflow experiment tracking."""

    def test_tracker_initialization(self):
        from src.mlops.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker(experiment_name="test_exp")
        assert tracker.experiment_name == "test_exp"
        assert tracker._initialised is False

    def test_log_param_without_mlflow_does_not_crash(self):
        from src.mlops.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker()
        # Should not raise even when MLflow is not installed
        tracker.log_param("temperature", 1.5)
        assert True  # No exception

    def test_log_metric_without_mlflow_does_not_crash(self):
        from src.mlops.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker()
        tracker.log_metric("points", 85.0)
        assert True

    def test_start_run_context_manager_without_mlflow(self):
        from src.mlops.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker()
        with tracker.start_run("test_run") as ctx:
            assert ctx is tracker
        assert True

    def test_get_tracker_returns_singleton(self):
        from src.mlops.mlflow_tracker import get_tracker
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    @patch("src.mlops.mlflow_tracker.MLflowTracker._ensure_mlflow")
    def test_log_params_calls_mlflow(self, mock_ensure):
        from src.mlops.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker()
        tracker._initialised = True
        mock_mlflow = MagicMock()
        tracker._mlflow = mock_mlflow
        tracker.log_params({"a": 1, "b": 2})
        mock_mlflow.log_params.assert_called_once_with({"a": 1, "b": 2})

    @patch("src.mlops.mlflow_tracker.MLflowTracker._ensure_mlflow")
    def test_register_model_calls_mlflow(self, mock_ensure):
        from src.mlops.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker()
        tracker._initialised = True
        mock_mlflow = MagicMock()
        tracker._mlflow = mock_mlflow
        tracker.register_model("test_model", "runs:/abc/model")
        mock_mlflow.register_model.assert_called_once()

    def test_transition_model_stage_without_mlflow(self):
        from src.mlops.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker()
        # Should not raise
        tracker.transition_model_stage("test_model", "Production")
        assert True
