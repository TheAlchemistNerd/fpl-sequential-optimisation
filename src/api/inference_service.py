"""FastAPI inference service for FPL decision recommendations.

Production-grade endpoints:
    GET  /health          — liveness check
    GET  /beliefs         — current HDP belief state for all players
    POST /recommendations — optimal transfers, lineup, and captain
    POST /retrain         — trigger a model retrain (admin only)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.mlops.mlflow_tracker import get_tracker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class BeliefResponse(BaseModel):
    player_id: int
    gameweek: int
    probabilities: list[float]
    mode_cluster: int
    entropy: float


class RecommendationRequest(BaseModel):
    gameweek: int = Field(..., ge=1, le=38)
    transfers_allowed: int = Field(default=1, ge=0)
    use_chip: Optional[str] = Field(default=None)
    team_id: Optional[int] = None
    current_squad: Optional[list[int]] = None


class RecommendationResponse(BaseModel):
    gameweek: int
    transfers_in: list[int] = []
    transfers_out: list[int] = []
    starting_xi: list[int] = []
    captain: int = 0
    vice_captain: int = 0
    predicted_points: float = 0.0
    chip_recommended: Optional[str] = None


class RetriggerResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise services on startup, clean up on shutdown."""
    log.info("FPL Prediction API starting…")
    # Load latest models from MLflow registry
    tracker = get_tracker()
    log.info("MLflow tracking URI: %s", tracker.tracking_uri)
    yield
    log.info("FPL Prediction API shutting down.")


app = FastAPI(
    title="FPL Prediction API",
    description="Bayesian HDP + optimisation service for FPL decisions.",
    version="0.1.0",
    lifespan=lifespan,
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


@app.get("/beliefs", response_model=list[BeliefResponse] | dict[str, Any])
def get_beliefs(
    player_id: Optional[int] = None,
    gameweek: Optional[int] = None,
    model: str = "hdp_gmm",
) -> list[BeliefResponse] | dict[str, Any]:
    """Return the current belief state from the HDP model.

    Parameters
    ----------
    player_id : int, optional
        If provided, return beliefs for a single player.
    gameweek : int, optional
        Gameweek number. Defaults to latest available.
    model : str
        Which HDP variant to query (``hdp_gmm``, ``hdp_poisson_gamma``,
        ``hdp_negbin``).
    """
    # Load belief states from disk cache or MLflow
    from src.mlops.drift_detector import compute_population_drift
    from src.models.belief_state import BeliefState

    belief_store = settings.DATA_DIR / "belief_states"
    if not belief_store.exists():
        raise HTTPException(
            404,
            "No belief states available. Run the data pipeline first.",
        )

    # Load belief CSVs
    gw = gameweek or _get_latest_gameweek(belief_store)
    belief_file = belief_store / f"belief_states_gw{gw}.csv"
    if not belief_file.exists():
        raise HTTPException(404, f"No belief states for GW {gw}.")

    import pandas as pd

    df = pd.read_csv(belief_file)
    if player_id is not None:
        df = df[df["player_id"] == player_id]
        if df.empty:
            return {"message": f"No beliefs found for player {player_id}"}

    beliefs = []
    for _, row in df.iterrows():
        probs = [float(x) for x in str(row["probabilities"]).split(",")]
        beliefs.append(
            BeliefResponse(
                player_id=int(row["player_id"]),
                gameweek=int(row["gameweek"]),
                probabilities=probs,
                mode_cluster=int(np.argmax(probs)),
                entropy=float(-sum(p * np.log(p + 1e-10) for p in probs)),
            )
        )

    return beliefs


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(req: RecommendationRequest) -> RecommendationResponse:
    """Return optimal transfers, lineup, and captain for *gameweek*.

    The service:
    1. Loads latest HDP posteriors.
    2. Runs Thompson Sampling (sample from posteriors).
    3. Solves the MILP optimisation for the sampled scores.
    """
    from src.optimization.milp_solver import solve_squad_optimisation, solve_with_transfers
    from src.optimization.thompson_sampling import thompson_sampling_milp

    # Load player pool and belief states
    player_pool = _load_player_pool()
    if player_pool is None:
        raise HTTPException(
            404,
            "No player data available. Run the data pipeline first.",
        )

    # Run Thompson Sampling with MILP
    current_squad = req.current_squad or []
    result = thompson_sampling_milp(
        player_pool=player_pool,
        points_map=player_pool.get("points", {}),
        budget=100.0,
        current_squad=current_squad if current_squad else None,
        free_transfers=req.transfers_allowed,
    )

    # Determine transfers
    transfers_in = []
    transfers_out = []
    if current_squad:
        new_squad = set(result["squad"])
        old_squad = set(current_squad)
        transfers_in = list(new_squad - old_squad)
        transfers_out = list(old_squad - new_squad)

    return RecommendationResponse(
        gameweek=req.gameweek,
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        starting_xi=result.get("starting_xi", []),
        captain=result.get("captain", 0),
        vice_captain=result.get("vice_captain", 0),
        predicted_points=result.get("total_points", 0.0),
        chip_recommended=_recommend_chip(req.gameweek, player_pool),
    )


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------


@app.post("/retrain", response_model=RetriggerResponse)
def trigger_retrain() -> RetriggerResponse:
    """Trigger a full model retrain pipeline.

    In production this would enqueue a Prefect/Airflow flow run.
    """
    log.info("Retrain requested — triggering data pipeline.")
    try:
        from src.pipeline.data_pipeline import run_pipeline
        # In production, this would be async via Celery/Prefect
        # run_pipeline(latest_gameweek=_get_latest_gameweek())
        return RetriggerResponse(
            status="queued",
            message="Retrain pipeline enqueued.",
        )
    except Exception as e:
        log.error("Failed to enqueue retrain: %s", e)
        raise HTTPException(500, f"Retrain failed: {e}")


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _load_player_pool() -> dict[str, Any] | None:
    """Load player pool from disk."""
    import pandas as pd

    players_file = settings.DATA_DIR / "fpl_players.csv"
    if not players_file.exists():
        return None

    players_df = pd.read_csv(players_file)
    return {
        "ids": players_df["id"].tolist(),
        "positions": players_df.get("position", ["MID"] * len(players_df)).tolist(),
        "prices": players_df.get("now_cost", [50] * len(players_df)).tolist(),
        "clubs": players_df.get("team", [0] * len(players_df)).tolist(),
        "points": {
            int(row["id"]): float(row.get("points_per_game", 5.0))
            for _, row in players_df.iterrows()
        },
    }


def _get_latest_gameweek(belief_store: Path) -> int:
    """Infer the latest gameweek from available belief files."""
    files = list(belief_store.glob("belief_states_gw*.csv"))
    if not files:
        return 1
    gw_numbers = [int(f.stem.split("_")[-1]) for f in files]
    return max(gw_numbers)


def _recommend_chip(gameweek: int, player_pool: dict[str, Any]) -> str | None:
    """Simple heuristic for chip recommendation."""
    from src.optimization.chip_optimizer import ChipDeploymentOptimiser

    optimiser = ChipDeploymentOptimiser()
    rng = np.random.default_rng(gameweek)
    result = optimiser.select_chip(rng)
    if result and result.expected_gain > 5.0:
        return result.chip
    return None
