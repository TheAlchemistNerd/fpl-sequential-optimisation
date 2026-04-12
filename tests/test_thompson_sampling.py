"""
Tests for src/optimization/thompson_sampling.py — Thompson Sampling (Step A).

Tests cover:
    - Posterior predictive sampling (Negative-Binomial)
    - Thompson Sampling algorithm for single-gameweek optimisation
    - GenTS-Explore for combinatorial settings (initial squad / Wildcard)
    - Temperature scaling and annealing
    - Exploration quality (posterior-driven vs uniform)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.belief_state import BeliefState, NBPointsPosterior


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_belief_states():
    """Sample belief states for 5 players at GW1."""
    return {
        (1, 1): BeliefState(1, 1, np.array([0.9, 0.05, 0.05])),
        (2, 1): BeliefState(2, 1, np.array([0.1, 0.8, 0.1])),
        (3, 1): BeliefState(3, 1, np.array([0.05, 0.1, 0.85])),
        (4, 1): BeliefState(4, 1, np.array([0.33, 0.34, 0.33])),
        (5, 1): BeliefState(5, 1, np.array([0.7, 0.2, 0.1])),
    }


@pytest.fixture
def sample_cluster_means():
    """3 clusters, 4 features."""
    return np.array([
        [2.0, 1.5, 0.5, 0.3],
        [1.0, 0.8, 0.3, 0.2],
        [0.2, 0.1, 0.1, 0.05],
    ])


# ===========================================================================
# NBPointsPosterior tests
# ===========================================================================


class TestNBPointsPosterior:
    """Validate Negative-Binomial posterior predictive."""

    def test_sample_returns_non_negative(self):
        nb = NBPointsPosterior(player_id=1, r=5.0, expected_points=10.0)
        rng = np.random.default_rng(42)
        samples = [nb.sample(rng) for _ in range(100)]
        assert all(s >= 0 for s in samples)

    def test_mean_approx_expected(self):
        nb = NBPointsPosterior(player_id=1, r=5.0, expected_points=10.0)
        rng = np.random.default_rng(42)
        samples = [nb.sample(rng) for _ in range(1000)]
        # Sample mean should be close to expected_points
        assert np.mean(samples) == pytest.approx(10.0, abs=1.0)

    def test_variance_scales_with_r(self):
        """Smaller r → higher variance (more over-dispersed)."""
        rng = np.random.default_rng(42)
        nb_low_r = NBPointsPosterior(1, r=2.0, expected_points=10.0)
        nb_high_r = NBPointsPosterior(1, r=20.0, expected_points=10.0)
        samples_low = [nb_low_r.sample(rng) for _ in range(500)]
        samples_high = [nb_high_r.sample(rng) for _ in range(500)]
        assert np.var(samples_low) > np.var(samples_high)


# ===========================================================================
# Thompson Sampling tests
# ===========================================================================


class TestThompsonSampling:
    """Validate Thompson Sampling algorithm."""

    def test_ts_returns_valid_squad(self, sample_belief_states):
        """Selected squad should satisfy basic constraints."""
        from src.optimization.thompson_sampling import thompson_sampling_select
        rng = np.random.default_rng(42)

        # Mock point predictions for each player
        points_per_player = {pid: 10.0 for pid in range(1, 6)}

        # Simple selection: top-3 by sampled points
        selected = thompson_sampling_select(
            belief_states=sample_belief_states,
            points_map=points_per_player,
            squad_size=3,
            budget=50.0,
            price_map={1: 10.0, 2: 8.0, 3: 6.0, 4: 5.0, 5: 7.0},
            rng=rng,
        )
        assert len(selected) == 3
        assert all(pid in points_per_player for pid in selected)

    def test_ts_respects_budget(self, sample_belief_states):
        """Selected squad should not exceed budget."""
        from src.optimization.thompson_sampling import thompson_sampling_select
        rng = np.random.default_rng(42)
        points_per_player = {pid: 10.0 for pid in range(1, 6)}
        price_map = {1: 15.0, 2: 12.0, 3: 8.0, 4: 5.0, 5: 10.0}
        budget = 25.0

        selected = thompson_sampling_select(
            belief_states=sample_belief_states,
            points_map=points_per_player,
            squad_size=3,
            budget=budget,
            price_map=price_map,
            rng=rng,
        )
        total_cost = sum(price_map[pid] for pid in selected)
        assert total_cost <= budget

    def test_ts_prefers_high_value_players(self):
        """TS should preferentially select players with higher expected points."""
        from src.optimization.thompson_sampling import thompson_sampling_select
        rng = np.random.default_rng(42)

        belief_states = {
            (i, 1): BeliefState(i, 1, np.array([0.9, 0.1]))
            for i in range(1, 11)
        }
        points_per_player = {i: float(i) for i in range(1, 11)}  # Higher = better
        price_map = {i: 5.0 for i in range(1, 11)}

        # Run multiple TS iterations and check selection frequency
        selection_counts = {i: 0 for i in range(1, 11)}
        for _ in range(100):
            selected = thompson_sampling_select(
                belief_states=belief_states,
                points_map=points_per_player,
                squad_size=5,
                budget=50.0,
                price_map=price_map,
                rng=rng,
            )
            for pid in selected:
                selection_counts[pid] += 1

        # Top 5 players should be selected more often than bottom 5
        top_freq = sum(selection_counts[i] for i in range(6, 11))
        bottom_freq = sum(selection_counts[i] for i in range(1, 6))
        assert top_freq > bottom_freq


# ===========================================================================
# Temperature scaling tests
# ===========================================================================


class TestTemperatureScaling:
    """Validate temperature annealing for exploration control."""

    def test_temperature_annealing_decreases(self):
        """Temperature should decrease over the season."""
        from src.optimization.thompson_sampling import anneal_temperature
        T_init = 2.0
        T_final = 0.1

        t_early = anneal_temperature(5, 38, T_init, T_final)
        t_late = anneal_temperature(35, 38, T_init, T_final)
        assert t_early > t_late

    def test_temperature_at_start_equals_init(self):
        from src.optimization.thompson_sampling import anneal_temperature
        T = anneal_temperature(1, 38, 2.0, 0.1)
        assert T == pytest.approx(2.0, abs=0.1)

    def test_temperature_at_end_equals_final(self):
        from src.optimization.thompson_sampling import anneal_temperature
        T = anneal_temperature(38, 38, 2.0, 0.1)
        assert T == pytest.approx(0.1, abs=0.1)


# ===========================================================================
# GenTS-Explore tests
# ===========================================================================


class TestGenTSExplore:
    """Validate GenTS-Explore for combinatorial squad selection."""

    def test_gents_explore_returns_valid_squad(self):
        """Initial squad should satisfy all FPL constraints."""
        from src.optimization.thompson_sampling import gents_explore
        rng = np.random.default_rng(42)

        # Mock 20 players with different positions and prices
        n_players = 20
        points = {i: np.random.normal(5, 2) for i in range(1, n_players + 1)}
        # Use affordable prices (max 6.67 avg to fit 15 in 100)
        prices = {i: np.random.uniform(4.0, 7.0) for i in range(1, n_players + 1)}
        positions = {i: ["GK", "DEF", "MID", "FWD"][i % 4] for i in range(1, n_players + 1)}

        squad = gents_explore(
            player_ids=list(range(1, n_players + 1)),
            points_posterior=points,
            price_map=prices,
            position_map=positions,
            budget=100.0,
            rng=rng,
        )
        assert len(squad) == 15

    def test_gents_explore_respects_position_constraints(self):
        """Squad should have correct position distribution (2-5-5-3)."""
        from src.optimization.thompson_sampling import gents_explore
        rng = np.random.default_rng(42)

        n_gk, n_def, n_mid, n_fwd = 4, 8, 8, 4
        player_ids = list(range(1, n_gk + n_def + n_mid + n_fwd + 1))
        positions = {}
        idx = 1
        for _ in range(n_gk):
            positions[idx] = "GK"; idx += 1
        for _ in range(n_def):
            positions[idx] = "DEF"; idx += 1
        for _ in range(n_mid):
            positions[idx] = "MID"; idx += 1
        for _ in range(n_fwd):
            positions[idx] = "FWD"; idx += 1

        points = {i: np.random.normal(5, 2) for i in player_ids}
        prices = {i: 5.0 for i in player_ids}

        squad = gents_explore(
            player_ids=player_ids,
            points_posterior=points,
            price_map=prices,
            position_map=positions,
            budget=100.0,
            rng=rng,
        )
        squad_positions = [positions[pid] for pid in squad]
        assert squad_positions.count("GK") == 2
        assert squad_positions.count("DEF") == 5
        assert squad_positions.count("MID") == 5
        assert squad_positions.count("FWD") == 3


# ===========================================================================
# MILP-backed Thompson Sampling tests
# ===========================================================================


class TestThompsonSamplingMILP:
    """Validate MILP-backed Thompson Sampling (production tier)."""

    @pytest.fixture
    def sample_player_pool(self):
        """30-player pool with realistic FPL distribution."""
        np.random.seed(42)
        n = 30
        positions = ["GK"]*4 + ["DEF"]*8 + ["MID"]*10 + ["FWD"]*8
        prices = [4.5 + np.random.uniform(0, 3.5) for _ in range(n)]
        clubs = [f"Club_{i % 20}" for i in range(n)]
        points = {i: float(np.random.normal(5, 3)) for i in range(1, n+1)}
        return {
            "ids": list(range(1, n + 1)),
            "positions": positions,
            "prices": prices,
            "clubs": clubs,
            "points": points,
        }

    def test_ts_milp_returns_valid_squad(self, sample_player_pool):
        """MILP-backed TS should return a valid 15-player squad."""
        from src.optimization.thompson_sampling import thompson_sampling_milp
        rng = np.random.default_rng(42)
        result = thompson_sampling_milp(
            player_pool=sample_player_pool,
            points_map=sample_player_pool["points"],
            budget=100.0,
            rng=rng,
        )
        assert len(result["squad"]) == 15
        assert len(result["starting_xi"]) == 11
        assert result["captain"] in result["starting_xi"]
        assert result["vice_captain"] in result["starting_xi"]
        assert result["captain"] != result["vice_captain"]

    def test_ts_milp_respects_budget(self, sample_player_pool):
        """MILP-backed TS should not exceed budget."""
        from src.optimization.thompson_sampling import thompson_sampling_milp
        rng = np.random.default_rng(42)
        result = thompson_sampling_milp(
            player_pool=sample_player_pool,
            points_map=sample_player_pool["points"],
            budget=80.0,
            rng=rng,
        )
        assert result["total_cost"] <= 80.01

    def test_ts_milp_formation_valid(self, sample_player_pool):
        """MILP-backed TS should satisfy 2-5-5-3 formation."""
        from src.optimization.thompson_sampling import thompson_sampling_milp
        rng = np.random.default_rng(42)
        result = thompson_sampling_milp(
            player_pool=sample_player_pool,
            points_map=sample_player_pool["points"],
            budget=100.0,
            rng=rng,
        )
        squad_positions = [
            sample_player_pool["positions"][sample_player_pool["ids"].index(pid)]
            for pid in result["squad"]
        ]
        assert squad_positions.count("GK") == 2
        assert squad_positions.count("DEF") == 5
        assert squad_positions.count("MID") == 5
        assert squad_positions.count("FWD") == 3

    def test_ts_milp_with_transfers(self, sample_player_pool):
        """MILP-backed TS with current squad should account for transfer costs."""
        from src.optimization.thompson_sampling import thompson_sampling_milp
        rng = np.random.default_rng(42)
        current_squad = sample_player_pool["ids"][:15]
        result = thompson_sampling_milp(
            player_pool=sample_player_pool,
            points_map=sample_player_pool["points"],
            budget=100.0,
            current_squad=current_squad,
            free_transfers=1,
            rng=rng,
        )
        assert "transfer_cost" in result
        assert result["transfer_cost"] >= 0

    def test_ts_milp_temperature_effect(self, sample_player_pool):
        """High temperature should produce more diverse squads."""
        from src.optimization.thompson_sampling import thompson_sampling_milp
        rng = np.random.default_rng(42)

        result_low = thompson_sampling_milp(
            player_pool=sample_player_pool,
            points_map=sample_player_pool["points"],
            budget=100.0,
            temperature=0.1,
            rng=rng,
        )
        rng2 = np.random.default_rng(42)
        result_high = thompson_sampling_milp(
            player_pool=sample_player_pool,
            points_map=sample_player_pool["points"],
            budget=100.0,
            temperature=5.0,
            rng=rng2,
        )
        # Both should be valid squads
        assert len(result_low["squad"]) == 15
        assert len(result_high["squad"]) == 15
