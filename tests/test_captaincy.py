"""
Tests for Captaincy MAB (LinUCB + Contextual Thompson Sampling).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestLinUCBArm:
    """Validate individual LinUCB arm."""

    def test_initialization(self):
        from src.optimization.selection.captaincy import LinUCBArm
        arm = LinUCBArm(d=5)
        assert arm.A.shape == (5, 5)
        assert arm.b.shape == (5,)
        assert arm.mu.shape == (5,)
        # Initial A should be identity
        assert np.allclose(arm.A, np.eye(5))

    def test_update_changes_posterior(self):
        from src.optimization.selection.captaincy import LinUCBArm
        arm = LinUCBArm(d=3)
        context = np.array([1.0, 2.0, 3.0])
        arm.update(context, reward=10.0)
        # A should have increased
        assert not np.allclose(arm.A, np.eye(3))
        # mu should be non-zero after update
        assert not np.allclose(arm.mu, np.zeros(3))

    def test_ucb_score_with_exploration(self):
        from src.optimization.selection.captaincy import LinUCBArm
        arm = LinUCBArm(d=3)
        context = np.array([1.0, 0.0, 0.0])
        # Fresh arm should have high uncertainty
        score = arm.ucb_score(context, alpha=1.0)
        assert score > 0  # Uncertainty bonus ensures positive score

    def test_ucb_score_converges(self):
        from src.optimization.selection.captaincy import LinUCBArm
        arm = LinUCBArm(d=3)
        context = np.array([1.0, 0.0, 0.0])
        # Simulate many observations
        for _ in range(100):
            arm.update(context, reward=5.0)
        score = arm.ucb_score(context, alpha=1.0)
        # After many observations, score should be close to expected reward
        assert abs(score - 5.0) < 1.0


class TestCaptaincyBandit:
    """Validate full captaincy bandit."""

    @pytest.fixture
    def simple_bandit(self):
        """11 candidates with 10-dim context."""
        from src.optimization.selection.captaincy import CaptaincyBandit
        return CaptaincyBandit(n_candidates=11, context_dim=10, alpha=1.0)

    def test_select_linucb_returns_valid_index(self, simple_bandit):
        contexts = np.random.randn(11, 10)
        action = simple_bandit.select_linucb(contexts)
        assert 0 <= action < 11

    def test_select_thompson_returns_valid_index(self, simple_bandit):
        contexts = np.random.randn(11, 10)
        rng = np.random.default_rng(42)
        action = simple_bandit.select_thompson(contexts, rng)
        assert 0 <= action < 11

    def test_update_increases_total_pulls(self, simple_bandit):
        simple_bandit.update(0, np.zeros(10), reward=10.0)
        assert simple_bandit.total_pulls == 1

    def test_build_context_shape(self, simple_bandit):
        belief = np.array([0.3, 0.5, 0.2])  # K=3
        context = simple_bandit.build_context(
            belief_state=belief,
            fdr=3.0,
            is_home=True,
            form_3=6.5,
            form_5=5.8,
            sentiment=0.2,
        )
        # K + 5 (fdr, home, form3, form5, sentiment)
        expected_dim = 3 + 5
        assert context.shape == (expected_dim,)

    def test_linucb_prefers_higher_reward(self):
        """After many observations, LinUCB should prefer the better arm."""
        from src.optimization.selection.captaincy import CaptaincyBandit
        bandit = CaptaincyBandit(n_candidates=3, context_dim=5, alpha=0.1)
        contexts = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ])
        # Simulate arm 1 consistently giving higher rewards
        for _ in range(50):
            bandit.update(1, contexts[1], reward=10.0)
            bandit.update(0, contexts[0], reward=5.0)
            bandit.update(2, contexts[2], reward=3.0)

        action = bandit.select_linucb(contexts)
        assert action == 1  # Should prefer arm 1

    def test_thompson_sampling_exploration(self):
        """Thompson Sampling should explore different arms."""
        from src.optimization.selection.captaincy import CaptaincyBandit
        bandit = CaptaincyBandit(n_candidates=3, context_dim=5, alpha=0.1)
        contexts = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ])
        # With no observations, TS should sample different arms
        actions = set()
        for seed in range(20):
            rng = np.random.default_rng(seed)
            actions.add(bandit.select_thompson(contexts, rng))
        # Should have sampled at least 2 different arms
        assert len(actions) >= 2
