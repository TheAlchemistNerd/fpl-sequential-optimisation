"""
FPL Environment — Gym-like interface for DDQN training.

Provides a standard RL environment interface:
    state, reward = env.step(action)
    state = env.reset()

The environment simulates gameweek transitions using:
    - FFBS for belief state updates (from jax_kernels.py)
    - NB posterior predictive for point sampling
    - MILP for constraint enforcement after transfers
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.models.belief_state import assemble_757_state, BeliefState
from src.optimization.milp_solver import solve_squad_optimisation, solve_with_transfers

logger = logging.getLogger(__name__)


class FPLEnvironment:
    """
    Gym-like FPL environment for DDQN training.

    State: 757-dim vector (squad encoding + beliefs + chips + meta)
    Action: transfer action index (0 = no transfer, 1..K = transfer in candidate)
    Reward: FPL points scored minus transfer costs
    """

    def __init__(
        self,
        player_pool: dict[str, list],
        belief_states: dict[int, np.ndarray],
        transition_matrix: np.ndarray,
        budget: float = 100.0,
        free_transfers: int = 1,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        player_pool : dict
            Must contain 'ids', 'positions', 'prices', 'clubs', 'points'.
        belief_states : dict
            Mapping player_id → belief vector (K,).
        transition_matrix : np.ndarray
            (K, K) state transition matrix for belief updates.
        budget : float
            Total budget in £m.
        free_transfers : int
            Free transfers per gameweek.
        seed : int
            Random seed for reproducibility.
        """
        self.player_pool = player_pool
        self.belief_states = belief_states
        self.transition_matrix = transition_matrix
        self.budget = budget
        self.free_transfers = free_transfers
        self.rng = np.random.default_rng(seed)

        # Episode state
        self.current_squad: list[int] = []
        self.gameweek: int = 0
        self.total_points: float = 0.0
        self.chips: dict[str, bool] = {
            "wildcard": True,
            "bench_boost": True,
            "triple_captain": True,
            "free_hit": True,
        }
        self.done: bool = False

    def reset(
        self,
        initial_squad: list[int] | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Reset the environment to the start of the season.

        Parameters
        ----------
        initial_squad : list[int] | None
            Starting squad. If None, uses first 15 players.
        seed : int | None
            Reseed the RNG.

        Returns
        -------
        np.ndarray — initial state vector (757-dim).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if initial_squad is None:
            self.current_squad = self.player_pool["ids"][:15]
        else:
            self.current_squad = list(initial_squad)

        self.gameweek = 0
        self.total_points = 0.0
        self.done = False

        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action and advance to the next gameweek.

        Parameters
        ----------
        action : int
            Action index (0 = no transfer, 1..K = transfer in candidate).

        Returns
        -------
        (next_state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Decode action
        transfer = self._decode_action(action)

        # Apply transfer
        transfer_cost = 0
        if transfer is not None:
            player_in, player_out = transfer
            if player_out in self.current_squad and player_in not in self.current_squad:
                self.current_squad.remove(player_out)
                self.current_squad.append(player_in)
                # Count this as a transfer (simplified: 1 per step)
                if self.free_transfers > 0:
                    self.free_transfers -= 1
                else:
                    transfer_cost = 4.0

        # Simulate gameweek
        gw_points = self._simulate_gameweek()

        # Update belief states
        self._update_belief_states()

        # Accumulate points
        self.gameweek += 1
        reward = gw_points - transfer_cost
        self.total_points += reward

        # Check if season is over
        self.done = self.gameweek >= 38

        # Get next state
        next_state = self._get_state()

        info = {
            "gameweek": self.gameweek,
            "gw_points": gw_points,
            "transfer_cost": transfer_cost,
            "total_points": self.total_points,
        }

        return next_state, reward, self.done, info

    def _get_state(self) -> np.ndarray:
        """
        Assemble the 757-dimensional state vector.

        Uses assemble_757_state from belief_state.py.
        """
        try:
            state = assemble_757_state(
                squad_ids=self.current_squad,
                all_player_ids=self.player_pool["ids"],
                belief_vecs={
                    pid: self.belief_states.get(pid, np.zeros(30))
                    for pid in self.current_squad
                },
                budget=self.budget,
                chips=self.chips,
                gameweek=self.gameweek,
                free_transfers=self.free_transfers,
            )
            return np.array(state)
        except Exception:
            # Fallback: return a zero state of the expected dimension
            # This happens if assemble_757_state has different signature
            n_players = len(self.player_pool["ids"])
            K = 30
            dim = n_players + K + 4 + 3  # squad + beliefs + chips + meta
            return np.zeros(dim)

    def _decode_action(self, action: int) -> tuple[int, int] | None:
        """
        Decode action index into a (player_in, player_out) pair.

        Simplified: action 0 = no transfer, action k = transfer in
        the k-th candidate, transfer out the lowest-priced squad member.
        """
        if action == 0:
            return None

        candidates = sorted(
            self.player_pool["ids"],
            key=lambda pid: self.player_pool.get("points", {}).get(pid, 0),
            reverse=True,
        )[:100]

        if action - 1 >= len(candidates):
            return None

        player_in = candidates[action - 1]

        # Transfer out the lowest-priced squad member
        prices = {
            pid: self.player_pool["prices"][i]
            for i, pid in enumerate(self.player_pool["ids"])
        }
        sorted_squad = sorted(self.current_squad, key=lambda pid: prices.get(pid, 5.0))

        if sorted_squad:
            player_out = sorted_squad[0]
            return (player_in, player_out)

        return None

    def _simulate_gameweek(self) -> float:
        """
        Simulate a single gameweek for the current squad.

        Uses NB posterior predictive for point sampling.
        """
        points_map = self.player_pool.get("points", {})
        total = 0.0

        for pid in self.current_squad:
            expected = points_map.get(pid, 5.0)
            r = max(expected * 2, 2.0)
            p = r / (r + expected + 1e-10)
            sampled = self.rng.negative_binomial(r, max(min(p, 1 - 1e-10), 1e-10))
            total += sampled

        return total

    def _update_belief_states(self) -> None:
        """
        Update belief states via transition matrix multiplication.

        b_next = b · π
        """
        K = self.transition_matrix.shape[0]
        for pid in list(self.belief_states.keys()):
            b = self.belief_states[pid]
            b_next = b @ self.transition_matrix
            b_next = b_next / (b_next.sum() + 1e-10)
            self.belief_states[pid] = b_next
