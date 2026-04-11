"""
Thompson Sampling (Step A) — Chapter 8 §8.2–8.7.

Bayesian sequential decision-making for single-gameweek FPL optimisation.
Samples from the posterior predictive distribution over player points,
then solves a deterministic knapsack problem on the sampled values.

Usage
-----
>>> from src.optimization.thompson_sampling import thompson_sampling_select
>>> squad = thompson_sampling_select(belief_states, points_map, squad_size=15, ...)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.models.belief_state import NBPointsPosterior

logger = logging.getLogger(__name__)

# FPL squad formation requirements
POSITION_REQUIREMENTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


# ---------------------------------------------------------------------------
# Temperature Annealing
# ---------------------------------------------------------------------------


def anneal_temperature(
    current_gameweek: int,
    total_gameweeks: int = 38,
    t_init: float = 2.0,
    t_final: float = 0.1,
) -> float:
    """
    Linearly anneal temperature from t_init to t_final over the season.

    High temperature early → more exploration (wider posteriors).
    Low temperature late → more exploitation (narrower posteriors).

    τ_t = τ_0 · (1 - t/T) + τ_min · (t/T)
    """
    fraction = current_gameweek / total_gameweeks
    return t_init * (1 - fraction) + t_final * fraction


# ---------------------------------------------------------------------------
# Thompson Sampling: Single-GW Selection
# ---------------------------------------------------------------------------


def thompson_sampling_select(
    belief_states: dict[tuple[int, int], Any],
    points_map: dict[int, float],
    squad_size: int,
    budget: float,
    price_map: dict[int, float],
    rng: np.random.Generator | None = None,
    temperature: float = 1.0,
) -> list[int]:
    """
    Thompson Sampling for single-gameweek squad selection.

    Algorithm (Step A):
    1. For each player, sample points from their posterior predictive.
    2. Apply temperature scaling to widen/narrow posteriors.
    3. Select the top squad_size players by sampled points, respecting budget.

    Parameters
    ----------
    belief_states : dict
        Mapping (player_id, gameweek) → BeliefState.
    points_map : dict
        Mapping player_id → expected points (pre-computed from belief states).
    squad_size : int
        Number of players to select.
    budget : float
        Total budget in £m.
    price_map : dict
        Mapping player_id → price in £m.
    rng : np.random.Generator
    temperature : float
        Temperature scaling factor (>1 = more exploration).

    Returns
    -------
    List of selected player IDs.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample perturbed points for each player
    perturbed_points = {}
    for pid, expected in points_map.items():
        # Use Negative-Binomial for realistic point distributions
        r = max(expected * 2, 2.0)  # dispersion scales with expectation
        nb = NBPointsPosterior(pid, r=r, expected_points=expected)
        sampled = nb.sample(rng)
        # Temperature scaling: divide by τ to widen/narrow
        perturbed = sampled / max(temperature, 0.1)
        perturbed_points[pid] = perturbed

    # Greedy selection with budget constraint
    # Sort by perturbed points descending
    sorted_players = sorted(perturbed_points.items(), key=lambda x: x[1], reverse=True)

    selected = []
    remaining_budget = budget
    for pid, pts in sorted_players:
        if len(selected) >= squad_size:
            break
        price = price_map.get(pid, 4.0)
        if price <= remaining_budget:
            selected.append(pid)
            remaining_budget -= price

    logger.info(
        "Thompson Sampling: selected %d players, budget used: £%.1f/%.1f",
        len(selected), budget - remaining_budget, budget,
    )
    return selected


# ---------------------------------------------------------------------------
# GenTS-Explore: Combinatorial Pure Exploration
# ---------------------------------------------------------------------------


def gents_explore(
    player_ids: list[int],
    points_posterior: dict[int, float],
    price_map: dict[int, float],
    position_map: dict[int, str],
    budget: float,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    GenTS-Explore for initial squad selection or Wildcard optimisation.

    Solves the constrained combinatorial selection problem:
    - Select 15 players (2 GK, 5 DEF, 5 MID, 3 FWD)
    - Total price ≤ budget
    - Maximise sum of sampled points

    Uses a greedy heuristic per position group with budget feasibility.

    Parameters
    ----------
    player_ids : list[int]
        All available player IDs.
    points_posterior : dict
        Mapping player_id → sampled expected points.
    price_map : dict
        Mapping player_id → price.
    position_map : dict
        Mapping player_id → position string.
    budget : float
        Total budget in £m.
    rng : np.random.Generator

    Returns
    -------
    List of 15 selected player IDs.
    """
    if rng is None:
        rng = np.random.default_rng()

    squad = []
    remaining_budget = budget

    for position, required in POSITION_REQUIREMENTS.items():
        # Filter players by position
        candidates = [
            pid for pid in player_ids
            if position_map.get(pid) == position and pid not in squad
        ]
        # Sort by sampled points descending
        candidates.sort(key=lambda pid: points_posterior.get(pid, 0), reverse=True)

        selected_pos = 0
        for pid in candidates:
            if selected_pos >= required:
                break
            price = price_map.get(pid, 4.0)
            if price <= remaining_budget:
                squad.append(pid)
                remaining_budget -= price
                selected_pos += 1

    # Fill remaining spots with best available players if budget allows
    while len(squad) < 15:
        available = [pid for pid in player_ids if pid not in squad]
        if not available:
            break
        # Sort by value-per-cost ratio
        available.sort(
            key=lambda pid: points_posterior.get(pid, 0) / max(price_map.get(pid, 4.0), 0.1),
            reverse=True,
        )
        pid = available[0]
        price = price_map.get(pid, 4.0)
        if price <= remaining_budget:
            squad.append(pid)
            remaining_budget -= price
        else:
            break  # Can't afford any more players

    logger.info(
        "GenTS-Explore: selected %d players, budget used: £%.1f",
        len(squad), budget - remaining_budget,
    )
    return squad


# ---------------------------------------------------------------------------
# Regret Computation
# ---------------------------------------------------------------------------


def compute_bayes_regret(
    selected_points: list[float],
    oracle_points: list[float],
) -> float:
    """
    Compute cumulative Bayes regret.

    Regret(T) = Σ_t (μ* - μ_{a_t})

    where μ* is the oracle (best possible) points and μ_{a_t} is the
    selected squad's points.
    """
    return sum(oracle_points) - sum(selected_points)
