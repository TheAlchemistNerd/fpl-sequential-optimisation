"""
Mean-Field Games (MFG) — Ch. 9 §9.10.

Implements the coupled HJB-FP system for classical FPL leagues:
    - HJB equation (backward): Optimal control for a representative agent
    - Fokker-Planck equation (forward): Population distribution evolution
    - Fixed-point iteration between HJB and FP until equilibrium
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MFGState:
    """Represents the mean-field state at a given gameweek."""
    population_dist: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    value_function: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    optimal_policy: np.ndarray = field(default_factory=lambda: np.array([0]))


class MeanFieldGame:
    """
    Mean-Field Game solver for classical FPL leagues.

    Solves the coupled HJB-FP system via fixed-point iteration.
    """

    def __init__(
        self,
        n_states: int = 50,
        n_actions: int = 101,
        gamma: float = 0.99,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

    def solve_hjb(
        self,
        reward_matrix: np.ndarray,
        transition_matrix: np.ndarray,
        terminal_value: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve HJB backward in time: V(s,t) = max_a { r(s,a) + γ·E[V(s',t+1)] }

        Parameters
        ----------
        reward_matrix : np.ndarray (T, n_states, n_actions)
        transition_matrix : np.ndarray (n_states, n_states, n_actions)
        terminal_value : np.ndarray (n_states,)

        Returns
        -------
        V : np.ndarray (n_states, T)
        policy : np.ndarray (n_states, T)
        """
        T = reward_matrix.shape[0]
        V = np.zeros((self.n_states, T))
        policy = np.zeros((self.n_states, T), dtype=int)

        if terminal_value is not None:
            V[:, -1] = terminal_value

        for t in range(T - 2, -1, -1):
            r_t = reward_matrix[t]  # (S, A)
            for s in range(self.n_states):
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    expected_future = transition_matrix[s, :, a] @ V[:, t + 1]
                    q_values[a] = r_t[s, a] + self.gamma * expected_future
                V[s, t] = np.max(q_values)
                policy[s, t] = np.argmax(q_values)

        return V, policy

    def solve_fokker_planck(
        self,
        optimal_policy: np.ndarray,
        transition_matrix: np.ndarray,
        initial_distribution: np.ndarray,
    ) -> np.ndarray:
        """
        Solve FP forward in time: m_{t+1} = (L^{a*})^* m_t

        Parameters
        ----------
        optimal_policy : np.ndarray (n_states, T)
        transition_matrix : np.ndarray (n_states, n_states, n_actions)
        initial_distribution : np.ndarray (n_states,)

        Returns
        -------
        m : np.ndarray (T, n_states)
        """
        T = optimal_policy.shape[1]
        m = np.zeros((T, self.n_states))
        m[0] = initial_distribution / (initial_distribution.sum() + 1e-10)

        for t in range(T - 1):
            next_m = np.zeros(self.n_states)
            for s_prime in range(self.n_states):
                a_star = optimal_policy[s_prime, t]
                next_m += m[t, s_prime] * transition_matrix[s_prime, :, a_star]
            m[t + 1] = next_m / (next_m.sum() + 1e-10)

        return m

    def solve_equilibrium(
        self,
        reward_matrix: np.ndarray,
        transition_matrix: np.ndarray,
        initial_distribution: np.ndarray | None = None,
        terminal_value: np.ndarray | None = None,
        n_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> dict[str, np.ndarray]:
        """
        Solve coupled HJB-FP via fixed-point iteration.
        """
        T = reward_matrix.shape[0]
        if initial_distribution is None:
            initial_distribution = np.ones(self.n_states) / self.n_states

        population_dist = np.ones((T, self.n_states)) / self.n_states
        converged = False
        n_iters = 0

        for iteration in range(n_iterations):
            V, policy = self.solve_hjb(
                reward_matrix=reward_matrix,
                transition_matrix=transition_matrix,
                terminal_value=terminal_value,
            )

            new_population = self.solve_fokker_planck(
                optimal_policy=policy,
                transition_matrix=transition_matrix,
                initial_distribution=initial_distribution,
            )

            alpha = 0.5
            population_dist = (1 - alpha) * population_dist + alpha * new_population

            diff = np.max(np.abs(new_population - population_dist))
            n_iters = iteration + 1

            if diff < tolerance:
                converged = True
                logger.info("MFG converged in %d iters (diff=%.6f)", n_iters, diff)
                break

            if iteration % 10 == 0:
                logger.info("MFG iter %d: diff=%.6f", iteration, diff)

        return {
            "population_dist": population_dist,
            "value_function": V,
            "optimal_policy": policy,
            "converged": converged,
            "n_iters": n_iters,
        }
