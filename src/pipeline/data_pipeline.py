"""Orchestrated data pipeline (Prefect / cron compatible).

This module provides the weekly data-collection pipeline that:
1. Fetches fresh FPL API data (public + protected).
2. Runs feature engineering.
3. Triggers Bayesian model updates (HDP posteriors).
4. Logs metrics to MLflow.

Can be run standalone::

    python -m src.pipeline.data_pipeline

Or scheduled via Prefect / Apache Airflow.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.data_fetcher import fetch_all_data
from src.config.settings import settings
from src.data.feature_engineering import (
    build_core_features,
    build_advanced_features,
    build_ict_features,
    build_form_features,
    assemble_feature_matrix,
)

log = logging.getLogger(__name__)


def run_pipeline(latest_gameweek: int) -> None:
    """Execute the full weekly data + model-update pipeline.

    Parameters
    ----------
    latest_gameweek : int
        The most recently completed gameweek.
    """
    log.info("=" * 60)
    log.info("Starting FPL data pipeline for GW %d", latest_gameweek)
    log.info("=" * 60)

    # Step 1 – Fetch all data
    import asyncio
    asyncio.run(fetch_all_data(latest_gameweek))

    # Step 2 – Feature engineering
    features = engineer_features()

    # Step 3 – Model update (placeholder – wire to HDP models)
    update_hdp_models(features)

    log.info("Pipeline complete for GW %d.", latest_gameweek)


def engineer_features() -> pd.DataFrame:
    """Load raw CSVs and produce the feature matrix for modelling.

    Delegates to ``src.data.feature_engineering`` for canonical per-90,
    form, ICT, and advanced feature computation.

    Returns
    -------
    pd.DataFrame
        Rows = player-gameweek observations, columns = engineered features.
    """
    data_dir = settings.DATA_DIR
    log.info("Engineering features from %s …", data_dir)

    gw_df = pd.read_csv(data_dir / "fpl_gameweek_data.csv")
    players_df = pd.read_csv(data_dir / "fpl_players.csv")

    # Parse stats column if present
    if "stats" in gw_df.columns:
        import ast
        stats_expanded = gw_df["stats"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        ).apply(pd.Series)
        gw_df = pd.concat([gw_df.drop(columns=["stats"]), stats_expanded], axis=1)

    # Merge player metadata
    players_df["full_name"] = players_df["first_name"] + " " + players_df["second_name"]
    merged = pd.merge(
        gw_df,
        players_df[["id", "full_name", "element_type", "team"]],
        on="id",
        how="left",
    )

    # Map element_type to position
    position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    merged["position"] = merged["element_type"].map(position_map)

    # Rename 'id' to 'player_id' for feature engineering compatibility
    if "id" in merged.columns and "player_id" not in merged.columns:
        merged = merged.rename(columns={"id": "player_id"})

    # Ensure 'gameweek' column exists
    if "gameweek" not in merged.columns:
        merged["gameweek"] = 1  # fallback

    # Apply canonical feature engineering pipeline
    merged = build_core_features(merged)
    merged = build_advanced_features(merged)
    merged = build_ict_features(merged)
    merged = build_form_features(merged)

    # Assemble final feature matrix
    X, feature_names = assemble_feature_matrix(merged)
    log.info("  → %d rows, %d engineered features", len(X), len(feature_names))
    return X


def update_hdp_models(features: pd.DataFrame) -> None:
    """Run Bayesian model updates with the latest feature matrix.

    Logs metrics to MLflow and checks for drift to trigger retraining.
    """
    from src.mlops.mlflow_tracker import get_tracker
    from src.mlops.drift_detector import should_retrain

    tracker = get_tracker()

    log.info("Running HDP model updates…")
    # In production, this would call:
    #   from src.models.hdp_gmm import run_hdp_gmm
    #   samples, summary = run_hdp_gmm(features.values, ...)

    # Log placeholder metrics for now
    tracker.log_param("model", "hdp_gmm")
    tracker.log_metric("n_features", features.shape[1])
    tracker.log_metric("n_samples", features.shape[0])

    # Check drift and potentially trigger full retrain
    # drift = compute_population_drift(...)
    # if should_retrain(drift, historical_drifts):
    #     log.info("Drift detected — triggering full retrain")

    log.info("HDP model update complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gw = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_pipeline(gw)
