"""
MLflow Experiment Tracker — Ch. 11 §11.4.

Tracks experiments, logs metrics/params/artifacts, and manages
model registry for staging/promoting FPL decision engines.

Usage
-----
>>> from src.mlops.mlflow_tracker import MLflowTracker
>>> tracker = MLflowTracker()
>>> with tracker.start_run("thompson_sampling_gw12"):
...     tracker.log_param("temperature", 1.5)
...     tracker.log_metric("cumulative_points", 850)
...     tracker.log_model("thompson_sampling", model_path="model.pkl")
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Wrapper around MLflow for experiment tracking and model registry.

    Parameters
    ----------
    tracking_uri : str | None
        MLflow tracking server URI. Defaults to local file store.
    experiment_name : str
        Name of the MLflow experiment.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "fpl_sequential_optimisation",
    ) -> None:
        self.tracking_uri = tracking_uri or f"file://{os.path.abspath('mlruns')}"
        self.experiment_name = experiment_name
        self._initialised = False

    def _ensure_mlflow(self) -> None:
        """Lazy-initialise MLflow to avoid import errors when not installed."""
        if self._initialised:
            return

        try:
            import mlflow
            import mlflow.tracking
        except ImportError:
            logger.warning(
                "MLflow not installed. Install with: pip install mlflow"
            )
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._initialised = True
        self._mlflow = mlflow

    @contextmanager
    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None):
        """
        Context manager for an MLflow run.

        Parameters
        ----------
        run_name : str | None
            Name of the run.
        tags : dict | None
            Tags to attach to the run.

        Yields
        ------
        MLflowTracker — self for chaining.
        """
        self._ensure_mlflow()
        if not self._initialised:
            yield self
            return

        with self._mlflow.start_run(run_name=run_name) as run:
            if tags:
                for key, value in tags.items():
                    self._mlflow.set_tag(key, value)
            logger.info("Started MLflow run: %s", run.info.run_id)
            yield self

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self._ensure_mlflow()
        if not self._initialised:
            return
        self._mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters."""
        self._ensure_mlflow()
        if not self._initialised:
            return
        self._mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric."""
        self._ensure_mlflow()
        if not self._initialised:
            return
        self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics."""
        self._ensure_mlflow()
        if not self._initialised:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file as an artifact."""
        self._ensure_mlflow()
        if not self._initialised:
            return
        self._mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, name: str, model_path: str) -> None:
        """
        Log a model artifact.

        Parameters
        ----------
        name : str
            Name of the model (for registry).
        model_path : str
            Path to the saved model file.
        """
        self._ensure_mlflow()
        if not self._initialised:
            return
        self._mlflow.log_artifact(model_path, artifact_path="models")
        logger.info("Logged model %s from %s", name, model_path)

    def register_model(self, model_name: str, model_uri: str) -> None:
        """
        Register a model in the MLflow Model Registry.

        Parameters
        ----------
        model_name : str
            Name for the registered model.
        model_uri : str
            URI of the logged model (e.g., "runs:/<run_id>/models").
        """
        self._ensure_mlflow()
        if not self._initialised:
            return
        try:
            self._mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info("Registered model: %s", model_name)
        except Exception as e:
            logger.warning("Failed to register model %s: %s", model_name, e)

    def transition_model_stage(self, model_name: str, stage: str) -> None:
        """
        Transition a model version to a new stage.

        Parameters
        ----------
        model_name : str
            Name of the registered model.
        stage : str
            Target stage: "Staging", "Production", or "Archived".
        """
        self._ensure_mlflow()
        if not self._initialised:
            return
        try:
            client = self._mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage=stage,
            )
            logger.info("Transitioned %s to %s", model_name, stage)
        except Exception as e:
            logger.warning("Failed to transition model %s: %s", model_name, e)

    def load_model_from_registry(self, model_name: str, stage: str = "Production"):
        """
        Load a model from the MLflow Model Registry.

        Parameters
        ----------
        model_name : str
            Name of the registered model.
        stage : str
            Stage to load from.

        Returns
        -------
        The loaded model (or None if MLflow not available).
        """
        self._ensure_mlflow()
        if not self._initialised:
            return None
        try:
            model_uri = f"models:/{model_name}/{stage}"
            return self._mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.warning("Failed to load model %s: %s", model_name, e)
            return None


# Global tracker instance
_tracker: MLflowTracker | None = None


def get_tracker(
    tracking_uri: str | None = None,
    experiment_name: str = "fpl_sequential_optimisation",
) -> MLflowTracker:
    """Get or create the global MLflow tracker."""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker(tracking_uri, experiment_name)
    return _tracker
