"""
Tests for Chip Deployment Optimiser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestChipDeploymentOptimiser:
    """Validate chip deployment optimiser."""

    def test_initialization(self):
        from src.optimization.chip_optimizer import ChipDeploymentOptimiser, ALL_CHIPS
        opt = ChipDeploymentOptimiser(n_gameweeks=38)
        assert opt.n_gameweeks == 38
        assert set(opt.available_chips) == set(ALL_CHIPS)
        for chip in ALL_CHIPS:
            assert opt.mu[chip].shape == (38,)
            assert opt.tau[chip].shape == (38,)

    def test_select_chip_returns_valid_result(self):
        from src.optimization.chip_optimizer import ChipDeploymentOptimiser
        opt = ChipDeploymentOptimiser(n_gameweeks=10)
        rng = np.random.default_rng(42)
        result = opt.select_chip(rng)
        assert result is not None
        assert result.chip in opt.available_chips
        assert 0 <= result.gameweek < 10

    def test_select_chip_none_when_all_used(self):
        from src.optimization.chip_optimizer import ChipDeploymentOptimiser, ALL_CHIPS
        opt = ChipDeploymentOptimiser(n_gameweeks=10)
        for chip in ALL_CHIPS:
            opt.mark_chip_used(chip)
        rng = np.random.default_rng(42)
        result = opt.select_chip(rng)
        assert result is None

    def test_update_changes_posterior(self):
        from src.optimization.chip_optimizer import ChipDeploymentOptimiser
        opt = ChipDeploymentOptimiser(n_gameweeks=10)
        opt.update("bench_boost", gameweek=5, observed_gain=8.0)
        # Posterior mean should move toward observed gain
        assert abs(opt.mu["bench_boost"][5]) > 0
        # Precision should increase
        assert opt.tau["bench_boost"][5] > 1.0

    def test_update_converges_to_observed(self):
        from src.optimization.chip_optimizer import ChipDeploymentOptimiser
        opt = ChipDeploymentOptimiser(n_gameweeks=10)
        true_gain = 10.0
        for _ in range(100):
            opt.update("triple_captain", gameweek=3, observed_gain=true_gain + np.random.randn() * 0.1)
        # Posterior mean should converge to true gain
        assert abs(opt.mu["triple_captain"][3] - true_gain) < 1.0

    def test_mark_chip_used_removes_from_available(self):
        from src.optimization.chip_optimizer import ChipDeploymentOptimiser
        opt = ChipDeploymentOptimiser(n_gameweeks=10)
        assert "wildcard" in opt.available_chips
        opt.mark_chip_used("wildcard")
        assert "wildcard" not in opt.available_chips

    def test_estimate_double_gw_reward(self):
        from src.optimization.chip_optimizer import ChipDeploymentOptimiser
        opt = ChipDeploymentOptimiser()
        # Bench Boost gain = bench points
        bb_reward = opt.estimate_double_gw_reward("bench_boost", 10, 50.0, bench_points=8.0)
        assert bb_reward == pytest.approx(8.0)
        # Triple Captain gain = captain points
        tc_reward = opt.estimate_double_gw_reward("triple_captain", 10, 50.0, captain_points=12.0)
        assert tc_reward == pytest.approx(12.0)
        # Free Hit gain ≈ 20% of base
        fh_reward = opt.estimate_double_gw_reward("free_hit", 10, 50.0)
        assert fh_reward == pytest.approx(10.0, abs=0.1)
        # Wildcard gain ≈ 15% of base
        wc_reward = opt.estimate_double_gw_reward("wildcard", 10, 50.0)
        assert wc_reward == pytest.approx(7.5, abs=0.1)

    def test_invalid_chip_raises(self):
        from src.optimization.chip_optimizer import ChipDeploymentOptimiser
        opt = ChipDeploymentOptimiser(n_gameweeks=10)
        with pytest.raises(ValueError):
            opt.update("invalid_chip", gameweek=5, observed_gain=5.0)

    def test_invalid_gameweek_raises(self):
        from src.optimization.chip_optimizer import ChipDeploymentOptimiser
        opt = ChipDeploymentOptimiser(n_gameweeks=10)
        with pytest.raises(ValueError):
            opt.update("bench_boost", gameweek=15, observed_gain=5.0)
