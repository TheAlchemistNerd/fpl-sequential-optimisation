"""
Drift Detection & ASUGS — Chapter 11 §11.6, Chapter 7 §7.8.

Monitors KL-divergence between belief state distributions across consecutive
gameweeks to detect concept drift. Triggers automated retraining when
structural shifts exceed historical thresholds.

ASUGS (Adaptive SUGS) provides online updates for the HDP concentration
parameter γ, adapting the number of latent clusters to streaming data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KL Divergence (Ch. 11 §11.6.2)
# ---------------------------------------------------------------------------


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL(p || q) = Σ_k p_k · log(p_k / q_k).

    Parameters
    ----------
    p : (K,) — source distribution (must sum to 1).
    q : (K,) — target distribution (must sum to 1).
    epsilon : float — smoothing for numerical stability.

    Returns
    -------
    Scalar KL divergence in nats.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # Clip to avoid log(0)
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    # Renormalise
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_population_drift(
    beliefs_t: dict[int, np.ndarray],
    beliefs_t_prev: dict[int, np.ndarray],
) -> float:
    """
    Compute population-level KL drift between consecutive gameweeks.

    KL_pop(t) = (1/J) Σ_j KL(b_j(t) || b_j(t-1))

    Parameters
    ----------
    beliefs_t : dict mapping player_id → belief vector (K,)
    beliefs_t_prev : dict mapping player_id → belief vector (K,) for previous GW

    Returns
    -------
    Scalar population drift value.
    """
    common_players = set(beliefs_t.keys()) & set(beliefs_t_prev.keys())
    if not common_players:
        logger.warning("No common players between gameweeks; drift = 0")
        return 0.0

    total_kl = 0.0
    for pid in common_players:
        total_kl += kl_divergence(beliefs_t[pid], beliefs_t_prev[pid])

    return float(total_kl / len(common_players))


def compute_drift_timeseries(
    belief_series: list[dict[int, np.ndarray]],
) -> list[float]:
    """
    Compute KL drift for all consecutive gameweek pairs.

    Parameters
    ----------
    belief_series : list of {player_id: belief_vector}, one per gameweek.

    Returns
    -------
    List of drift values (len = len(belief_series) - 1).
    """
    drifts = []
    for t in range(1, len(belief_series)):
        d = compute_population_drift(belief_series[t], belief_series[t - 1])
        drifts.append(d)
    return drifts


# ---------------------------------------------------------------------------
# Threshold Computation (Ch. 11 §11.6.2)
# ---------------------------------------------------------------------------


def compute_threshold(
    historical_drifts: list[float],
    percentile: float = 95.0,
) -> float:
    """
    Compute drift threshold from historical percentile.

    Parameters
    ----------
    historical_drifts : list of past drift values.
    percentile : float — threshold percentile (default 95).

    Returns
    -------
    Scalar threshold value.
    """
    if len(historical_drifts) < 2:
        logger.warning(
            "Insufficient history (%d values) for threshold computation",
            len(historical_drifts),
        )
        return float("inf")

    return float(np.percentile(historical_drifts, percentile))


# ---------------------------------------------------------------------------
# Drift Detection (Ch. 11 §11.6.2, §11.7.2)
# ---------------------------------------------------------------------------


def detect_drift(
    current_drift: float,
    historical_drifts: list[float],
    percentile: float = 95.0,
    min_history: int = 5,
) -> tuple[bool, dict[str, Any]]:
    """
    Detect whether current drift exceeds historical threshold.

    Trigger conditions (Ch. 11 §11.7.2):
    - KL_pop(t) > threshold (95th percentile of historical values)
    - Requires at least min_history historical values

    Parameters
    ----------
    current_drift : float — current KL_pop(t).
    historical_drifts : list of past drift values.
    percentile : float — threshold percentile.
    min_history : int — minimum history required.

    Returns
    -------
    (is_drift, details) where details contains threshold, current value, etc.
    """
    details = {
        "current_drift": current_drift,
        "n_history": len(historical_drifts),
    }

    if len(historical_drifts) < min_history:
        details["reason"] = f"Insufficient history ({len(historical_drifts)} < {min_history})"
        return False, details

    threshold = compute_threshold(historical_drifts, percentile)
    details["threshold"] = threshold

    is_drift = current_drift > threshold
    details["is_drift"] = is_drift
    if is_drift:
        details["reason"] = (
            f"Drift {current_drift:.4f} exceeds {percentile}th percentile "
            f"threshold {threshold:.4f}"
        )
        logger.warning("DRIFT DETECTED: %s", details["reason"])
    else:
        details["reason"] = f"Drift {current_drift:.4f} within normal range (threshold {threshold:.4f})"

    return is_drift, details


# ---------------------------------------------------------------------------
# Performance Degradation Check (Ch. 11 §11.7.2)
# ---------------------------------------------------------------------------


def check_performance_degradation(
    historical_points: list[float],
    recent_points: list[float],
    percentile: float = 25.0,
) -> bool:
    """
    Check if recent performance is below the 25th percentile of historical.

    Parameters
    ----------
    historical_points : list of past gameweek point totals.
    recent_points : list of recent gameweek point totals.
    percentile : float — degradation threshold percentile.

    Returns
    -------
    True if recent performance is significantly degraded.
    """
    if len(historical_points) < 4 or len(recent_points) < 1:
        return False

    threshold = float(np.percentile(historical_points, percentile))
    recent_avg = float(np.mean(recent_points))

    is_degraded = recent_avg < threshold
    if is_degraded:
        logger.warning(
            "PERFORMANCE DEGRADATION: recent avg %.1f < %.1fth percentile (%.1f)",
            recent_avg, percentile, threshold,
        )
    return is_degraded


# ---------------------------------------------------------------------------
# ASUGS: Adaptive Concentration Parameter Updates (Ch. 7 §7.8)
# ---------------------------------------------------------------------------


def asugs_gamma_update(
    gamma_old: float,
    n_clusters: int,
    n_observations: int,
    step_size: float | None = None,
) -> float:
    """
    ASUGS online update for the global concentration parameter γ.

    The gradient of the log-likelihood w.r.t. γ is:
        ∂/∂γ log p(z | γ) = 1/(γ + n - 1) - ψ(γ + n - 1) + ψ(γ)

    where ψ is the digamma function.

    Update: γ_n = γ_{n-1} + η_n · gradient

    Parameters
    ----------
    gamma_old : float — previous concentration value.
    n_clusters : int — current number of active clusters.
    n_observations : int — total observations seen so far.
    step_size : float | None — η_n; defaults to 1/n.

    Returns
    -------
    Updated γ value (guaranteed positive).
    """
    from scipy.special import digamma

    if step_size is None:
        step_size = 1.0 / max(n_observations, 1)

    # Approximate gradient using the expected number of new clusters
    # E[new clusters at step n] ≈ γ / (γ + n - 1)
    n_prev = max(n_observations - 1, 1)
    expected_new = gamma_old / (gamma_old + n_prev)

    # Simplified gradient: (observed_new - expected_new) / (γ + n - 1)
    # We approximate observed_new from the cluster-to-observation ratio
    observed_new = min(n_clusters / max(n_observations, 1), 1.0)
    gradient = (observed_new - expected_new) / (gamma_old + n_prev)

    gamma_new = gamma_old + step_size * gradient
    # Ensure positivity
    return max(gamma_new, 0.1)


def asugs_alpha_update(
    alpha_old: float,
    n_player_clusters: int,
    n_player_observations: int,
    step_size: float | None = None,
) -> float:
    """
    ASUGS update for the player-specific concentration parameter α.

    Same logic as gamma_update but applied per-player.

    Parameters
    ----------
    alpha_old : float — previous player-level concentration.
    n_player_clusters : int — active clusters for this player.
    n_player_observations : int — gameweeks seen for this player.
    step_size : float | None — defaults to 1/n.

    Returns
    -------
    Updated α value (guaranteed positive).
    """
    return asugs_gamma_update(
        alpha_old, n_player_clusters, n_player_observations, step_size
    )


# ---------------------------------------------------------------------------
# Retraining Decision (Ch. 11 §11.7)
# ---------------------------------------------------------------------------


def should_retrain(
    current_drift: float,
    historical_drifts: list[float],
    historical_points: list[float],
    recent_points: list[float] | None = None,
    force_schedule: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """
    Comprehensive retraining decision combining drift and performance signals.

    Triggers retraining when ANY of:
    1. KL divergence exceeds 95th percentile threshold
    2. Recent performance < 25th percentile of historical
    3. Force schedule (e.g., every 4 gameweeks)

    Parameters
    ----------
    current_drift : float — current KL_pop(t).
    historical_drifts : list of past drift values.
    historical_points : list of past gameweek point totals.
    recent_points : list | None — recent gameweek points.
    force_schedule : bool — forced periodic retraining.

    Returns
    -------
    (should_retrain, reasons_dict)
    """
    reasons: dict[str, Any] = {}

    # Check drift
    drift_detected, drift_details = detect_drift(current_drift, historical_drifts)
    reasons["drift"] = drift_details

    # Check performance
    perf_degraded = False
    if recent_points is not None:
        perf_degraded = check_performance_degradation(historical_points, recent_points)
        reasons["performance"] = {
            "degraded": perf_degraded,
            "recent_avg": float(np.mean(recent_points)),
            "historical_25th": float(np.percentile(historical_points, 25)),
        }

    # Decision
    do_retrain = drift_detected or perf_degraded or force_schedule
    reasons["retrain"] = do_retrain
    reasons["drift_triggered"] = drift_detected
    reasons["performance_triggered"] = perf_degraded
    reasons["schedule_triggered"] = force_schedule

    if do_retrain:
        logger.info("RETRAINING TRIGGERED: %s", reasons)
    else:
        logger.debug("No retraining needed. Drift=%.4f, Perf OK", current_drift)

    return do_retrain, reasons
