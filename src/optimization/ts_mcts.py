"""
Thompson Sampling + MCTS (Step B) — Chapter 8 §8.8.

Multi-gameweek tactical planning via Monte Carlo Tree Search.
Samples a world from the HDP-HMM posterior, builds a search tree over
transfer sequences, and selects the action with the highest visit count.

Usage
-----
>>> from src.optimization.ts_mcts import ts_mcts_select
>>> result = ts_mcts_select(player_pool, current_squad, belief_states, ...)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.models.belief_state import NBPointsPosterior
from src.optimization.mcts_node import MCTSNode
from src.optimization.rollout import rollout

logger = logging.getLogger(__name__)

# MCTS hyperparameters
DEFAULT_SIMULATIONS = 1000
DEFAULT_HORIZON = 3  # gameweeks to lookahead
DEFAULT_EXPLORATION = 1.414  # √2
DEFAULT_DISCOUNT = 0.95
TOP_K_CANDIDATES = 100  # Belief-state pruning


def _generate_transfer_actions(
    current_squad: list[int],
    player_pool: dict[str, list],
    top_k: int = TOP_K_CANDIDATES,
    max_transfers: int = 2,
) -> list[dict[str, Any]]:
    """
    Generate candidate transfer actions for MCTS expansion.

    Uses belief-state pruning: only consider top-K players by expected points.

    Parameters
    ----------
    current_squad : list[int]
        Current squad player IDs.
    player_pool : dict
        Must contain 'ids', 'positions', 'prices', 'clubs', 'points'.
    top_k : int
        Number of top candidates to consider (pruning).
    max_transfers : int
        Maximum number of simultaneous transfers.

    Returns
    -------
    list of action dicts: {'in': [...], 'out': [...]}
    """
    all_ids = player_pool["ids"]
    points = player_pool.get("points", {})

    # Prune to top-K candidates by expected points
    sorted_candidates = sorted(all_ids, key=lambda pid: points.get(pid, 0), reverse=True)
    top_candidates = sorted_candidates[:top_k]

    actions: list[dict[str, Any]] = []

    # No transfer action
    actions.append({"in": [], "out": [], "label": "no_transfer"})

    # Single transfer actions
    for out_pid in current_squad:
        for in_pid in top_candidates:
            if in_pid not in current_squad:
                actions.append({
                    "in": [in_pid],
                    "out": [out_pid],
                    "label": f"{in_pid}_in_{out_pid}_out",
                })

    # Limit action space for tractability
    if len(actions) > 500:
        # Keep best single transfers + no transfer
        actions = actions[:500]

    return actions


def _apply_action(
    current_squad: list[int],
    action: dict[str, Any],
    player_pool: dict[str, list],
    budget: float,
) -> tuple[list[int], float] | None:
    """
    Apply a transfer action to the current squad.

    Returns (new_squad, transfer_cost) or None if infeasible.
    """
    if not action["in"]:
        return current_squad, 0  # No transfer

    new_squad = list(current_squad)
    cost = 0.0

    for in_pid, out_pid in zip(action["in"], action["out"]):
        if out_pid in new_squad:
            new_squad.remove(out_pid)
            new_squad.append(in_pid)
            cost += 4.0  # Transfer cost per hit

    # Check budget
    prices = {pid: player_pool["prices"][i] for i, pid in enumerate(player_pool["ids"])}
    total_cost = sum(prices.get(pid, 5.0) for pid in new_squad)
    if total_cost > budget:
        return None  # Infeasible

    return new_squad, cost


def _sample_points_map(
    player_pool: dict[str, list],
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> dict[int, float]:
    """
    Sample a points map from posterior predictive distributions.

    Parameters
    ----------
    player_pool : dict
        Must contain 'ids', 'points' (expected points per player).
    rng : np.random.Generator
    temperature : float
        Temperature scaling (>1 = more exploration).

    Returns
    -------
    dict mapping player_id → sampled points.
    """
    ids = player_pool["ids"]
    expected_points = player_pool.get("points", {})
    sampled = {}

    for pid in ids:
        expected = expected_points.get(pid, 5.0)
        adjusted = max(expected / max(temperature, 0.1), 0.5)
        r = max(adjusted * 2, 2.0)
        nb = NBPointsPosterior(pid, r=r, expected_points=adjusted)
        sampled[pid] = float(nb.sample(rng))

    return sampled


def ts_mcts_select(
    player_pool: dict[str, list],
    current_squad: list[int],
    belief_states: dict[int, np.ndarray],
    transition_matrix: np.ndarray,
    budget: float,
    horizon: int = DEFAULT_HORIZON,
    n_simulations: int = DEFAULT_SIMULATIONS,
    exploration_weight: float = DEFAULT_EXPLORATION,
    discount: float = DEFAULT_DISCOUNT,
    temperature: float = 1.0,
    free_transfers: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Thompson Sampling + MCTS for multi-gameweek tactical planning.

    Algorithm (Ch. 8 §8.8.1):
    1. Sample a world from HDP-HMM posterior
    2. Build MCTS tree over transfer actions
    3. Rollout: simulate T gameweeks via FFBS + belief updates
    4. Backpropagate rewards
    5. Select action with highest visit count

    Parameters
    ----------
    player_pool : dict
        Must contain 'ids', 'positions', 'prices', 'clubs', 'points'.
    current_squad : list[int]
        Current squad player IDs.
    belief_states : dict
        Mapping player_id → belief vector (K,).
    transition_matrix : np.ndarray
        (K, K) state transition matrix.
    budget : float
        Total budget in £m.
    horizon : int
        Number of gameweeks to lookahead.
    n_simulations : int
        Number of MCTS simulations.
    exploration_weight : float
        UCB1 exploration constant.
    discount : float
        Discount factor for future rewards.
    temperature : float
        Temperature for posterior sampling.
    free_transfers : int
        Free transfers per gameweek.
    rng : np.random.Generator

    Returns
    -------
    dict with keys:
        - best_action: the selected transfer action
        - best_squad: resulting squad after best action
        - squad: current squad (before action)
        - simulations: number of simulations run
        - root_visits: visit count at root
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate pruned action space
    actions = _generate_transfer_actions(
        current_squad=current_squad,
        player_pool=player_pool,
        top_k=TOP_K_CANDIDATES,
    )

    # Create root node
    root_state = {
        "squad": current_squad,
        "belief_states": belief_states,
        "budget": budget,
        "free_transfers": free_transfers,
    }
    root = MCTSNode(
        state=root_state,
        untried_actions=list(actions),
    )

    # Sample future points maps for rollouts
    n_future_maps = max(horizon, 1)
    future_points_maps = [
        _sample_points_map(player_pool, rng, temperature)
        for _ in range(n_future_maps)
    ]

    # MCTS main loop
    for sim in range(n_simulations):
        # 1. Selection: traverse tree using UCB1
        node = root
        while node.is_fully_expanded and not node.is_leaf:
            node = node.select_best_child(exploration_weight)

        # 2. Expansion: add a new child if possible
        if node.untried_actions and not node.is_leaf:
            action = rng.choice(node.untried_actions)
            new_squad_result = _apply_action(
                node.state["squad"], action, player_pool, budget
            )
            if new_squad_result is not None:
                new_squad, transfer_cost = new_squad_result
                child_state = {
                    "squad": new_squad,
                    "belief_states": dict(node.state["belief_states"]),
                    "budget": budget,
                    "transfer_cost": transfer_cost,
                }
                node = node.expand(action, child_state)

        # 3. Rollout: simulate future gameweeks
        reward = rollout(
            squad_ids=node.state["squad"],
            belief_states=node.state["belief_states"],
            points_maps=future_points_maps,
            transition_matrix=transition_matrix,
            free_transfers=free_transfers,
            discount=discount,
            rng=rng,
        )

        # Penalise transfer costs
        transfer_cost = node.state.get("transfer_cost", 0.0)
        reward -= transfer_cost

        # 4. Backpropagation
        node.backpropagate(reward, discount)

    # Select action with highest visit count
    if root.children:
        best_action_node = max(root.children.values(), key=lambda n: n.visit_count)
        best_action = best_action_node.action
        best_squad = best_action_node.state["squad"]
    else:
        best_action = {"in": [], "out": [], "label": "no_transfer"}
        best_squad = current_squad

    logger.info(
        "TS+MCTS: simulations=%d, best_action=%s, root_visits=%d",
        n_simulations, best_action.get("label", "unknown"), root.visit_count,
    )

    return {
        "best_action": best_action,
        "best_squad": best_squad,
        "squad": current_squad,
        "simulations": n_simulations,
        "root_visits": root.visit_count,
    }
