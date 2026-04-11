"""
Belief State Construction — Chapter 5 §5.7, Chapter 7 §7.9.

The belief state is the mathematical bridge between the HDP posterior samples
and the decision engines (Thompson Sampling, MCTS, DDQN). For each player j
at gameweek t, the belief state is a probability distribution over the K
latent form clusters:

    b_j(t) = (P(z_{j,t} = 1 | x_{j,1:t}), ..., P(z_{j,t} = K | x_{j,1:t}))

This module provides:
    - Belief state computation from MCMC cluster assignments
    - Feature-to-points mapping (linear + Gradient Boosting)
    - Posterior predictive sampling (Negative-Binomial)
    - Belief state persistence and retrieval
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Belief State
# ---------------------------------------------------------------------------


@dataclass
class BeliefState:
    """
    Represents the posterior belief distribution over latent form clusters
    for a single player at a single gameweek.

    Attributes
    ----------
    player_id : int
        FPL element ID.
    gameweek : int
        Gameweek number.
    probabilities : np.ndarray
        Shape (K,) — P(z = k | data) for each cluster k.
    cluster_means : np.ndarray | None
        Shape (K, D) — μ_k for each cluster (from MCMC posterior mean).
    cluster_covs : np.ndarray | None
        Shape (K, D, D) — Σ_k for each cluster (from MCMC posterior mean).
    expected_features : np.ndarray | None
        Shape (D,) — E[x] = Σ_k b^{(k)} · μ_k.
    """

    player_id: int
    gameweek: int
    probabilities: np.ndarray
    cluster_means: np.ndarray | None = None
    cluster_covs: np.ndarray | None = None
    expected_features: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.cluster_means is not None and self.expected_features is None:
            self.expected_features = self.probabilities @ self.cluster_means

    @property
    def mode_cluster(self) -> int:
        """Most likely cluster index."""
        return int(np.argmax(self.probabilities))

    @property
    def entropy(self) -> float:
        """Shannon entropy of the belief distribution (nats)."""
        p = np.clip(self.probabilities, 1e-10, 1.0)
        return float(-np.sum(p * np.log(p)))

    @property
    def is_concentrated(self, threshold: float = 0.8) -> bool:
        """Whether the belief is concentrated on a single cluster."""
        return bool(np.max(self.probabilities) > threshold)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "player_id": self.player_id,
            "gameweek": self.gameweek,
            "probabilities": self.probabilities.tolist(),
            "mode_cluster": self.mode_cluster,
            "entropy": self.entropy,
            "is_concentrated": self.is_concentrated,
        }


# ---------------------------------------------------------------------------
# Belief State Builder
# ---------------------------------------------------------------------------


def build_belief_states_from_mcmc(
    cluster_assignments: np.ndarray,
    player_ids: np.ndarray,
    gameweeks: np.ndarray,
    cluster_means: np.ndarray | None = None,
    cluster_covs: np.ndarray | None = None,
) -> dict[tuple[int, int], BeliefState]:
    """
    Compute belief states from MCMC cluster assignment samples.

    Parameters
    ----------
    cluster_assignments : np.ndarray
        Shape (S, N) — cluster assignment for each of S MCMC samples and N observations.
    player_ids : np.ndarray
        Shape (N,) — player ID for each observation.
    gameweeks : np.ndarray
        Shape (N,) — gameweek for each observation.
    cluster_means : np.ndarray | None
        Shape (K, D) — mean of each cluster (from MCMC posterior).
    cluster_covs : np.ndarray | None
        Shape (K, D, D) — covariance of each cluster.

    Returns
    -------
    dict mapping (player_id, gameweek) → BeliefState
    """
    S, N = cluster_assignments.shape
    unique_players = np.unique(player_ids)
    unique_gameweeks = np.unique(gameweeks)
    K = int(cluster_assignments.max()) + 1

    belief_states: dict[tuple[int, int], BeliefState] = {}

    for pid in unique_players:
        player_mask = player_ids == pid
        player_gws = gameweeks[player_mask]

        for gw in unique_gameweeks:
            gw_mask = player_gws == gw
            if not gw_mask.any():
                continue

            obs_indices = np.where(player_mask)[0][gw_mask]
            # Average cluster assignments across observations at this gameweek
            counts = np.zeros(K)
            for s in range(S):
                for idx in obs_indices:
                    counts[cluster_assignments[s, idx]] += 1

            probs = counts / (counts.sum() + 1e-10)
            probs /= probs.sum()  # normalise

            bs = BeliefState(
                player_id=int(pid),
                gameweek=int(gw),
                probabilities=probs,
                cluster_means=cluster_means,
                cluster_covs=cluster_covs,
            )
            belief_states[(int(pid), int(gw))] = bs

    logger.info(
        "Built %d belief states for %d players across %d gameweeks",
        len(belief_states), len(unique_players), len(unique_gameweeks),
    )
    return belief_states


def build_belief_states_from_posterior_predictive(
    posterior_samples: np.ndarray,
    player_ids: np.ndarray,
    gameweeks: np.ndarray,
) -> dict[tuple[int, int], BeliefState]:
    """
    Build belief states from posterior predictive point samples.

    This is a simplified version when only point predictions are available
    (e.g., from a trained model rather than full MCMC).

    Parameters
    ----------
    posterior_samples : np.ndarray
        Shape (S, N) — sampled point predictions.
    player_ids : np.ndarray
        Shape (N,)
    gameweeks : np.ndarray
        Shape (N,)

    Returns
    -------
    dict mapping (player_id, gameweek) → BeliefState
    """
    # Discretise posterior samples into K bins via quantile-based clustering
    S, N = posterior_samples.shape
    K = 5  # Fixed: Low, Below-Avg, Average, Above-Avg, High

    # Global quantile thresholds
    all_values = posterior_samples.flatten()
    thresholds = np.quantile(all_values, np.linspace(0, 1, K + 1))

    # Assign each sample to a cluster
    cluster_assignments = np.zeros_like(posterior_samples, dtype=int)
    for s in range(S):
        for k in range(K):
            mask = (posterior_samples[s] >= thresholds[k]) & (posterior_samples[s] < thresholds[k + 1])
            cluster_assignments[s, mask] = k

    return build_belief_states_from_mcmc(
        cluster_assignments, player_ids, gameweeks,
    )


# ---------------------------------------------------------------------------
# Feature-to-Points Mapping
# ---------------------------------------------------------------------------


class FeatureToPointsMapper:
    """
    Maps feature vectors to expected FPL points.

    Supports two modes:
        - 'linear': Ridge regression with L2 regularisation.
        - 'gbm': Gradient Boosting Regressor (captures non-linearities).

    The mapper is trained on historical (features, points) pairs and then
    applied to the expected features from belief states:

        E[points_j] = f(E[x_j]) = f(Σ_k b_j^{(k)} · μ_k)
    """

    def __init__(self, method: str = "linear", **kwargs: Any) -> None:
        self.method = method
        if method == "linear":
            alpha = kwargs.get("alpha", 1.0)
            self.model: Ridge | GradientBoostingRegressor = Ridge(alpha=alpha)
        elif method == "gbm":
            n_estimators = kwargs.get("n_estimators", 200)
            max_depth = kwargs.get("max_depth", 4)
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'linear' or 'gbm'.")
        self.feature_names: list[str] = []
        self._is_fit = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureToPointsMapper":
        """Train the mapper on historical data."""
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        self._is_fit = True
        logger.info(
            "FeatureToPointsMapper (%s) fitted on %d samples, %d features",
            self.method, len(X), len(self.feature_names),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected points for feature vectors."""
        if not self._is_fit:
            raise RuntimeError("Mapper must be fitted before predict.")
        X_subset = X[self.feature_names]
        return self.model.predict(X_subset.values)  # type: ignore[return-value]

    def predict_from_belief_state(self, belief: BeliefState) -> float:
        """
        Predict expected points from a single belief state.

        Computes E[x] = Σ_k b^{(k)} · μ_k, then f(E[x]).
        """
        if belief.expected_features is None:
            raise ValueError("Belief state has no expected_features.")
        x = belief.expected_features.reshape(1, -1)
        x_df = pd.DataFrame(x, columns=self.feature_names)
        return float(self.predict(x_df)[0])


# ---------------------------------------------------------------------------
# Posterior Predictive (Negative-Binomial)
# ---------------------------------------------------------------------------


@dataclass
class NBPointsPosterior:
    """
    Negative-Binomial posterior over FPL points for a single player.

    Parameterised by (r, p) such that:
        E[Y] = r(1-p)/p
        Var[Y] = r(1-p)/p²

    Given an expected points estimate μ, we solve for p:
        p = r / (r + μ)
    """

    player_id: int
    r: float  # dispersion (learned from historical data)
    expected_points: float  # μ

    @property
    def p(self) -> float:
        """Success probability derived from expected points."""
        return self.r / (self.r + self.expected_points + 1e-10)

    def sample(self, rng: np.random.Generator | None = None) -> int:
        """Sample one realisation from the posterior predictive."""
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.negative_binomial(self.r, self.p))

    def pmf(self, k: int) -> float:
        """Probability mass at k points."""
        from scipy.stats import nbinom
        return float(nbinom.pmf(k, self.r, self.p))

    def mean(self) -> float:
        """Expected points (by construction, equals expected_points)."""
        return self.expected_points

    def variance(self) -> float:
        """Variance of points."""
        return self.r * (1 - self.p) / (self.p ** 2 + 1e-10)


def fit_nb_dispersion(
    historical_points: np.ndarray,
    method: str = "method_of_moments",
) -> float:
    """
    Estimate the NB dispersion parameter r from historical data.

    Method of moments:
        r = μ² / (σ² - μ)

    where μ is the sample mean and σ² is the sample variance.
    """
    mu = historical_points.mean()
    var = historical_points.var()

    if var <= mu:
        # Under-dispersed — fall back to Poisson (large r)
        logger.warning(
            "Data is under-dispersed (var=%.2f <= mean=%.2f); using large r.",
            var, mu,
        )
        return 100.0

    r = mu ** 2 / (var - mu)
    logger.info("NB dispersion fitted: r=%.2f (mean=%.2f, var=%.2f)", r, mu, var)
    return max(r, 0.5)  # floor at 0.5


# ---------------------------------------------------------------------------
# Belief State Store (persistence)
# ---------------------------------------------------------------------------


@dataclass
class BeliefStateStore:
    """
    Simple file-based persistence for belief states.

    Stores belief states as CSV files, one per gameweek.
    """

    base_dir: Path = Path("data/belief_states")

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        belief_states: dict[tuple[int, int], BeliefState],
        gameweek: int,
    ) -> Path:
        """Save belief states for a gameweek to CSV."""
        rows = []
        for (pid, gw), bs in belief_states.items():
            if gw != gameweek:
                continue
            row = bs.to_dict()
            row["probabilities"] = ",".join(f"{p:.4f}" for p in bs.probabilities)
            rows.append(row)

        df = pd.DataFrame(rows)
        path = self.base_dir / f"belief_states_gw{gameweek}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %d belief states to %s", len(rows), path)
        return path

    def load(self, gameweek: int) -> dict[tuple[int, int], BeliefState]:
        """Load belief states for a gameweek from CSV."""
        path = self.base_dir / f"belief_states_gw{gameweek}.csv"
        if not path.exists():
            logger.warning("No belief state file for GW %d", gameweek)
            return {}

        df = pd.read_csv(path)
        belief_states: dict[tuple[int, int], BeliefState] = {}
        for _, row in df.iterrows():
            probs = np.array([float(x) for x in str(row["probabilities"]).split(",")])
            bs = BeliefState(
                player_id=int(row["player_id"]),
                gameweek=int(row["gameweek"]),
                probabilities=probs,
            )
            belief_states[(bs.player_id, bs.gameweek)] = bs

        logger.info("Loaded %d belief states for GW %d", len(belief_states), gameweek)
        return belief_states


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def get_belief_state(
    player_id: int,
    gameweek: int,
    belief_states: dict[tuple[int, int], BeliefState],
) -> BeliefState | None:
    """Retrieve a single player's belief state."""
    return belief_states.get((player_id, gameweek))


def get_top_form_players(
    gameweek: int,
    belief_states: dict[tuple[int, int], BeliefState],
    position: str | None = None,
    top_k: int = 10,
) -> list[BeliefState]:
    """
    Get players with the highest mode-cluster probability at a gameweek.

    Useful for quick scouting: which players are most confidently in good form?
    """
    candidates = [
        bs for (pid, gw), bs in belief_states.items()
        if gw == gameweek
    ]
    candidates.sort(key=lambda bs: bs.mode_cluster, reverse=True)
    return candidates[:top_k]


def get_high_uncertainty_players(
    gameweek: int,
    belief_states: dict[tuple[int, int], BeliefState],
    top_k: int = 10,
) -> list[BeliefState]:
    """
    Get players with the highest belief entropy (most uncertain form).

    These are prime candidates for Thompson Sampling exploration.
    """
    candidates = [
        bs for (pid, gw), bs in belief_states.items()
        if gw == gameweek
    ]
    candidates.sort(key=lambda bs: bs.entropy, reverse=True)
    return candidates[:top_k]
