"""
State Representation — Ch. 9 §9.5.3.

Assembles the 757-dimensional state vector for the DDQN from:
    - Squad binary encoding (N ≈ 720): one-hot for players in the squad
    - Belief states (K = 30): mean belief across squad members
    - Chips (4): binary flags for available chips
    - Metadata (3): budget, gameweek, free transfers

Delegates to assemble_757_state from belief_state.py.
"""

from __future__ import annotations

import numpy as np

from src.models.belief_state import assemble_757_state


def build_state(
    squad_ids: list[int],
    all_player_ids: list[int],
    belief_vecs: dict[int, np.ndarray],
    budget: float,
    chips: dict[str, bool],
    gameweek: int,
    free_transfers: int,
) -> np.ndarray:
    """
    Build the 757-dimensional state vector.

    Parameters
    ----------
    squad_ids : list[int]
        Current squad player IDs.
    all_player_ids : list[int]
        All player IDs in the FPL registry.
    belief_vecs : dict
        Mapping player_id → belief vector (K,).
    budget : float
        Current budget in £m.
    chips : dict
        Chip availability flags.
    gameweek : int
        Current gameweek number.
    free_transfers : int
        Free transfers available.

    Returns
    -------
    np.ndarray — state vector.
    """
    state = assemble_757_state(
        squad_ids=squad_ids,
        all_player_ids=all_player_ids,
        belief_vecs=belief_vecs,
        budget=budget,
        chips=chips,
        gameweek=gameweek,
        free_transfers=free_transfers,
    )
    return np.array(state)


def state_dim(
    n_players: int,
    n_clusters: int = 30,
) -> int:
    """
    Compute the state dimensionality.

    dim = N (squad binary) + K (mean beliefs) + 4 (chips) + 3 (meta)

    Parameters
    ----------
    n_players : int
        Number of players in the registry.
    n_clusters : int
        Number of HDP clusters.

    Returns
    -------
    int — state dimensionality.
    """
    return n_players + n_clusters + 4 + 3
