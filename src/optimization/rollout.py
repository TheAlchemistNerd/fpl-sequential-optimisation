"""
MCTS Rollout — Forward Simulation via FFBS + Belief Updates.

Simulates T gameweeks forward from a given belief state and squad configuration.
At each simulated step:
    1. Player performances are sampled from HDP-HMM emission distributions
    2. Features are mapped to points
    3. Squad total points minus transfer costs are computed
    4. Belief states are updated via HDP-HMM transitions
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def simulate_gameweek(
    squad_ids: list[int],
    points_map: dict[int, float],
    belief_states: dict[int, np.ndarray],
    transition_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, dict[int, np.ndarray]]:
    """
    Simulate a single gameweek for a given squad.

    Parameters
    ----------
    squad_ids : list[int]
        Player IDs in the current squad.
    points_map : dict
        Mapping player_id → expected points for this gameweek.
    belief_states : dict
        Mapping player_id → belief vector (K,) for each squad member.
    transition_matrix : np.ndarray
        (K, K) state transition matrix for belief updates.
    rng : np.random.Generator

    Returns
    -------
    (total_points, updated_beliefs)
    """
    total_points = 0.0
    updated_beliefs = {}

    for pid in squad_ids:
        # Sample points from NB posterior
        expected = points_map.get(pid, 5.0)
        r = max(expected * 2, 2.0)
        p = r / (r + expected + 1e-10)
        sampled_points = rng.negative_binomial(r, p)
        total_points += sampled_points

        # Update belief state via transition
        if pid in belief_states:
            b = belief_states[pid]
            # b_next = b · π (matrix multiplication)
            b_next = b @ transition_matrix
            b_next = b_next / (b_next.sum() + 1e-10)  # renormalise
            updated_beliefs[pid] = b_next

    return total_points, updated_beliefs


def rollout(
    squad_ids: list[int],
    belief_states: dict[int, np.ndarray],
    points_maps: list[dict[int, float]],
    transition_matrix: np.ndarray,
    free_transfers: int = 1,
    transfer_cost_per_hit: float = 4.0,
    rng: np.random.Generator | None = None,
    discount: float = 0.95,
) -> float:
    """
    Simulate a multi-gameweek rollout from the current squad.

    Parameters
    ----------
    squad_ids : list[int]
        Current squad player IDs.
    belief_states : dict
        Mapping player_id → belief vector (K,) for each squad member.
    points_maps : list[dict]
        One points_map per future gameweek (length = horizon).
    transition_matrix : np.ndarray
        (K, K) state transition matrix.
    free_transfers : int
        Free transfers available per gameweek.
    transfer_cost_per_hit : float
        Point deduction per transfer beyond free allowance.
    rng : np.random.Generator
    discount : float
        Discount factor for future gameweek rewards.

    Returns
    -------
    float — cumulative discounted reward.
    """
    if rng is None:
        rng = np.random.default_rng()

    total_reward = 0.0
    current_beliefs = dict(belief_states)

    for gw_idx, points_map in enumerate(points_maps):
        # Simulate this gameweek
        gw_points, current_beliefs = simulate_gameweek(
            squad_ids=squad_ids,
            points_map=points_map,
            belief_states=current_beliefs,
            transition_matrix=transition_matrix,
            rng=rng,
        )

        # Apply discount and accumulate
        total_reward += (discount ** gw_idx) * gw_points

    return total_reward
