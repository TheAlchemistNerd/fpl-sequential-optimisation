"""
Tests for Mean-Field Games (MFG) module.

Tests cover:
    - HJB equation solving (backward induction)
    - Fokker-Planck equation solving (forward evolution)
    - Fixed-point equilibrium convergence
    - Edge cases and boundary conditions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def simple_reward_matrix():
    """Simple reward matrix for testing."""
    # 10 timesteps, 5 states, 3 actions
    T, S, A = 10, 5, 3
    # Reward increases with state index (higher budget → more points)
    base_reward = np.linspace(1, 5, S)
    reward = np.tile(base_reward[:, np.newaxis], (1, A))
    # Add small action-specific bonuses
    reward[:, 0] += 0.1  # No transfer
    reward[:, 1] += 0.2  # Conservative transfer
    reward[:, 2] += 0.0  # Aggressive transfer (risky)
    return np.tile(reward[np.newaxis, :, :], (T, 1, 1))


@pytest.fixture
def simple_transition_matrix():
    """Simple transition matrix for testing."""
    S, A = 5, 3
    P = np.zeros((S, S, A))
    for a in range(A):
        for s in range(S):
            # Higher actions lead to more state transitions
            if a == 0:  # No transfer: stay in current state
                P[s, s, a] = 0.9
                if s > 0:
                    P[s, s - 1, a] = 0.1
            elif a == 1:  # Conservative: slight improvement
                P[s, s, a] = 0.7
                if s < S - 1:
                    P[s, s + 1, a] = 0.3
            else:  # Aggressive: high variance
                P[s, s, a] = 0.5
                if s < S - 1:
                    P[s, s + 1, a] = 0.3
                if s > 0:
                    P[s, s - 1, a] = 0.2
    return P


# ===========================================================================
# HJB Tests
# ===========================================================================


class TestHJBEquation:
    """Validate Hamilton-Jacobi-Bellman equation solving."""

    def test_hjb_returns_correct_shapes(self, simple_reward_matrix, simple_transition_matrix):
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3, gamma=0.95)
        V, policy = mfg.solve_hjb(
            reward_matrix=simple_reward_matrix,
            transition_matrix=simple_transition_matrix,
        )
        assert V.shape == (5, 10)
        assert policy.shape == (5, 10)
        assert np.all(policy >= 0)
        assert np.all(policy < 3)

    def test_hjb_value_increases_backward(self, simple_reward_matrix, simple_transition_matrix):
        """Value function should generally increase as we go backward in time."""
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3, gamma=0.95)
        V, _ = mfg.solve_hjb(
            reward_matrix=simple_reward_matrix,
            transition_matrix=simple_transition_matrix,
        )
        # Earlier times should have higher values (more future rewards)
        assert V[:, 0].mean() >= V[:, -1].mean()

    def test_hjb_with_terminal_value(self, simple_reward_matrix, simple_transition_matrix):
        """Terminal value should be respected."""
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3, gamma=0.95)
        terminal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        V, _ = mfg.solve_hjb(
            reward_matrix=simple_reward_matrix,
            transition_matrix=simple_transition_matrix,
            terminal_value=terminal,
        )
        assert np.allclose(V[:, -1], terminal)


# ===========================================================================
# Fokker-Planck Tests
# ===========================================================================


class TestFokkerPlanck:
    """Validate Fokker-Planck equation solving."""

    def test_fp_returns_correct_shape(self, simple_transition_matrix):
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3)
        T = 10
        policy = np.zeros((5, T), dtype=int)
        m_dist = mfg.solve_fokker_planck(
            optimal_policy=policy,
            transition_matrix=simple_transition_matrix,
            initial_distribution=np.ones(5) / 5,
        )
        assert m_dist.shape == (T, 5)

    def test_fp_preserves_mass(self, simple_transition_matrix):
        """Total population mass should be preserved (≈1.0)."""
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3)
        T = 10
        policy = np.zeros((5, T), dtype=int)
        m_dist = mfg.solve_fokker_planck(
            optimal_policy=policy,
            transition_matrix=simple_transition_matrix,
            initial_distribution=np.ones(5) / 5,
        )
        # Each timestep should sum to 1.0 (within numerical tolerance)
        assert np.allclose(m_dist.sum(axis=1), 1.0, atol=1e-6)

    def test_fp_evolution_changes_distribution(self, simple_transition_matrix):
        """Population distribution should evolve over time."""
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3)
        T = 20
        # Policy that always selects action 1 (conservative improvement)
        policy = np.ones((5, T), dtype=int)
        initial = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
        m_dist = mfg.solve_fokker_planck(
            optimal_policy=policy,
            transition_matrix=simple_transition_matrix,
            initial_distribution=initial,
        )
        # Distribution should change from initial
        assert not np.allclose(m_dist[0], m_dist[-1])


# ===========================================================================
# Equilibrium Tests
# ===========================================================================


class TestMFGEquilibrium:
    """Validate MFG fixed-point equilibrium solving."""

    def test_equilibrium_returns_correct_keys(self, simple_reward_matrix, simple_transition_matrix):
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3, gamma=0.95)
        result = mfg.solve_equilibrium(
            reward_matrix=simple_reward_matrix,
            transition_matrix=simple_transition_matrix,
            n_iterations=20,
        )
        assert "population_dist" in result
        assert "value_function" in result
        assert "optimal_policy" in result
        assert "converged" in result
        assert "n_iters" in result

    def test_equilibrium_population_sums_to_one(self, simple_reward_matrix, simple_transition_matrix):
        """Population distribution should sum to 1 at each timestep."""
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3, gamma=0.95)
        result = mfg.solve_equilibrium(
            reward_matrix=simple_reward_matrix,
            transition_matrix=simple_transition_matrix,
            n_iterations=20,
        )
        pop = result["population_dist"]
        assert np.allclose(pop.sum(axis=1), 1.0, atol=1e-5)

    def test_equilibrium_value_function_is_finite(self, simple_reward_matrix, simple_transition_matrix):
        """Value function should be finite everywhere."""
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3, gamma=0.95)
        result = mfg.solve_equilibrium(
            reward_matrix=simple_reward_matrix,
            transition_matrix=simple_transition_matrix,
            n_iterations=20,
        )
        assert np.all(np.isfinite(result["value_function"]))


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestMFGEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_state_single_action(self):
        """Trivial case: 1 state, 1 action."""
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=1, n_actions=1)
        reward = np.ones((5, 1, 1)) * 2.0
        P = np.ones((1, 1, 1))
        result = mfg.solve_equilibrium(reward, P, n_iterations=10)
        assert result["population_dist"].shape == (5, 1)
        assert np.allclose(result["population_dist"], 1.0)

    def test_uniform_initial_distribution(self, simple_reward_matrix, simple_transition_matrix):
        """Should work with default uniform initial distribution."""
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg = MeanFieldGame(n_states=5, n_actions=3, gamma=0.95)
        result = mfg.solve_equilibrium(
            reward_matrix=simple_reward_matrix,
            transition_matrix=simple_transition_matrix,
            initial_distribution=None,
            n_iterations=10,
        )
        assert result["population_dist"] is not None

    def test_high_gamma_increases_value(self, simple_reward_matrix, simple_transition_matrix):
        """Higher discount factor should generally increase value function."""
        from src.optimization.mfg.mean_field_game import MeanFieldGame
        mfg_low_gamma = MeanFieldGame(n_states=5, n_actions=3, gamma=0.5)
        mfg_high_gamma = MeanFieldGame(n_states=5, n_actions=3, gamma=0.99)

        result_low = mfg_low_gamma.solve_equilibrium(
            simple_reward_matrix, simple_transition_matrix, n_iterations=10,
        )
        result_high = mfg_high_gamma.solve_equilibrium(
            simple_reward_matrix, simple_transition_matrix, n_iterations=10,
        )
        # Higher gamma should lead to higher values (more future rewards counted)
        assert result_high["value_function"].mean() >= result_low["value_function"].mean()
