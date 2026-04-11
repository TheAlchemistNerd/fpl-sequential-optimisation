"""FastAPI inference service for FPL decision recommendations.

Production-grade endpoints:
    GET  /health          – liveness check
    GET  /beliefs         – current HDP belief state for all players
    POST /recommendations – optimal transfers, lineup, and captain
    POST /retrain         – trigger a model retrain (admin only)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException

from src.config.settings import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="FPL Prediction API",
    description="Bayesian HDP + optimisation service for FPL decisions.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict[str, Any]:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@app.get("/beliefs")
def get_beliefs(
    player_id: Optional[int] = None,
    model: str = "hdp_gmm",
) -> dict[str, Any]:
    """Return the current belief state from the HDP model.

    Parameters
    ----------
    player_id : int, optional
        If provided, return beliefs for a single player.
    model : str
        Which HDP variant to query (``hdp_gmm``, ``hdp_poisson_gamma``,
        ``hdp_negbin``).
    """
    # TODO: Load posterior samples from MLflow / disk cache
    raise HTTPException(501, "Belief endpoint not yet implemented.")


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

class RecommendationRequest:
    """Schema for incoming recommendation requests."""

    gameweek: int
    transfers_allowed: int = 1
    use_chip: Optional[str] = None
    team_id: Optional[int] = None


@app.post("/recommendations")
def get_recommendations(req: RecommendationRequest) -> dict[str, Any]:
    """Return optimal transfers, lineup, and captain for *gameweek*.

    The service:
    1. Loads latest HDP posteriors.
    2. Runs Thompson Sampling (sample from posteriors).
    3. Solves the knapsack optimisation for the sampled scores.
    """
    # TODO: Wire up Thompson Sampling + knapsack solver
    raise HTTPException(501, "Recommendation endpoint not yet implemented.")


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

@app.post("/retrain")
def trigger_retrain() -> dict[str, str]:
    """Trigger a full model retrain pipeline.

    In production this would enqueue a Prefect/Airflow flow run.
    """
    log.info("Retrain requested – not yet wired to pipeline.")
    return {"status": "not_implemented"}
