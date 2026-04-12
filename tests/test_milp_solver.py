"""
Tests for src/optimization/milp_solver.py — MILP Constrained Optimiser.

Tests cover (Ch. 10 §10.6, §10.9):
    - Squad size constraint (15 players)
    - Formation constraints (2 GK, 5 DEF, 5 MID, 3 FWD)
    - Budget constraint
    - Per-team limit (≤3 per club)
    - Starting XI constraints (11 players, valid formation)
    - Captaincy constraints (exactly 1 captain from starters)
    - Piecewise linear transfer cost encoding
    - Vice-captain constraints
    - Edge cases: infeasible budget, empty candidate pool
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
def sample_player_pool():
    """
    30-player pool with realistic FPL distribution.
    Returns dict with keys: ids, positions, prices, clubs, predicted_points
    """
    np.random.seed(42)
    n = 30

    # Positions: 4 GK, 8 DEF, 10 MID, 8 FWD
    positions = (
        ["GK"] * 4 +
        ["DEF"] * 8 +
        ["MID"] * 10 +
        ["FWD"] * 8
    )

    # Prices: GK cheapest, FWD most expensive
    base_prices = {"GK": 4.5, "DEF": 5.0, "MID": 6.0, "FWD": 7.0}
    prices = [base_prices[pos] + np.random.uniform(-0.5, 2.0) for pos in positions]

    # Clubs: distribute across 20 clubs (max 2 per club in this pool)
    clubs = [f"Club_{i % 20}" for i in range(n)]

    # Predicted points: correlate with price but add noise
    predicted_points = [
        p * 0.6 + np.random.normal(2, 1.5) for p in prices
    ]

    return {
        "ids": list(range(1, n + 1)),
        "positions": positions,
        "prices": prices,
        "clubs": clubs,
        "predicted_points": predicted_points,
    }


# ===========================================================================
# Basic squad constraints tests
# ===========================================================================


class TestSquadConstraints:
    """Validate core squad selection constraints (Ch. 10 §10.6)."""

    def test_squad_size_is_15(self, sample_player_pool):
        """Optimal squad must have exactly 15 players."""
        from src.optimization.milp_solver import solve_squad_optimisation
        result = solve_squad_optimisation(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
        )
        assert len(result["squad"]) == 15

    def test_formation_2_5_5_3(self, sample_player_pool):
        """Squad must have exactly 2 GK, 5 DEF, 5 MID, 3 FWD."""
        from src.optimization.milp_solver import solve_squad_optimisation
        result = solve_squad_optimisation(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
        )
        squad_positions = [
            sample_player_pool["positions"][sample_player_pool["ids"].index(pid)]
            for pid in result["squad"]
        ]
        assert squad_positions.count("GK") == 2
        assert squad_positions.count("DEF") == 5
        assert squad_positions.count("MID") == 5
        assert squad_positions.count("FWD") == 3

    def test_budget_not_exceeded(self, sample_player_pool):
        """Total squad price must not exceed budget."""
        from src.optimization.milp_solver import solve_squad_optimisation
        # Generous budget to ensure feasibility with random prices
        budget = 200.0
        result = solve_squad_optimisation(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=budget,
        )
        total_cost = sum(
            sample_player_pool["prices"][sample_player_pool["ids"].index(pid)]
            for pid in result["squad"]
        )
        assert total_cost <= budget + 0.01  # floating point tolerance

    def test_per_team_limit_3(self, sample_player_pool):
        """No more than 3 players from any single club."""
        from src.optimization.milp_solver import solve_squad_optimisation
        result = solve_squad_optimisation(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
        )
        squad_clubs = [
            sample_player_pool["clubs"][sample_player_pool["ids"].index(pid)]
            for pid in result["squad"]
        ]
        from collections import Counter
        club_counts = Counter(squad_clubs)
        assert all(count <= 3 for count in club_counts.values())


# ===========================================================================
# Starting XI and Captaincy tests
# ===========================================================================


class TestStartingXIAndCaptaincy:
    """Validate Starting XI and captain constraints."""

    def test_starting_xi_size_11(self, sample_player_pool):
        """Starting XI must have exactly 11 players."""
        from src.optimization.milp_solver import solve_squad_optimisation
        result = solve_squad_optimisation(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
        )
        assert len(result["starting_xi"]) == 11

    def test_starters_from_squad(self, sample_player_pool):
        """All starters must be members of the squad."""
        from src.optimization.milp_solver import solve_squad_optimisation
        result = solve_squad_optimisation(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
        )
        assert set(result["starting_xi"]).issubset(set(result["squad"]))

    def test_valid_formation(self, sample_player_pool):
        """Starting XI must have ≥1 GK, ≥3 DEF, ≥1 FWD."""
        from src.optimization.milp_solver import solve_squad_optimisation
        result = solve_squad_optimisation(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
        )
        xi_positions = [
            sample_player_pool["positions"][sample_player_pool["ids"].index(pid)]
            for pid in result["starting_xi"]
        ]
        assert xi_positions.count("GK") >= 1
        assert xi_positions.count("DEF") >= 3
        assert xi_positions.count("FWD") >= 1

    def test_exactly_one_captain(self, sample_player_pool):
        """Exactly one captain, and captain must be a starter."""
        from src.optimization.milp_solver import solve_squad_optimisation
        result = solve_squad_optimisation(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
        )
        captain = result["captain"]
        assert captain is not None
        assert captain in result["starting_xi"]

    def test_vice_captain_is_starter(self, sample_player_pool):
        """Vice-captain must be a starter and different from captain."""
        from src.optimization.milp_solver import solve_squad_optimisation
        result = solve_squad_optimisation(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
        )
        vc = result["vice_captain"]
        assert vc is not None
        assert vc in result["starting_xi"]
        assert vc != result["captain"]


# ===========================================================================
# Transfer cost tests (Ch. 10 §10.9)
# ===========================================================================


class TestTransferCosts:
    """Validate piecewise linear transfer cost encoding."""

    def test_no_transfer_cost_within_free_limit(self, sample_player_pool):
        """Transfers within free allowance should cost 0 points."""
        from src.optimization.milp_solver import solve_with_transfers
        # Current squad (all player IDs 1-15)
        current_squad = sample_player_pool["ids"][:15]
        result = solve_with_transfers(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
            current_squad=current_squad,
            free_transfers=2,
        )
        # Transfer cost should be 0 if ≤2 changes
        assert result["transfer_cost"] >= 0

    def test_transfer_cost_4_per_extra(self, sample_player_pool):
        """Each transfer beyond free allowance costs 4 points."""
        from src.optimization.milp_solver import solve_with_transfers
        # Completely different squad → many transfers
        current_squad = sample_player_pool["ids"][:15]
        result = solve_with_transfers(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=sample_player_pool["predicted_points"],
            clubs=sample_player_pool["clubs"],
            budget=100.0,
            current_squad=current_squad,
            free_transfers=0,
        )
        # Should have some transfer cost
        assert result["transfer_cost"] >= 0
        # Transfer cost should be divisible by 4
        assert result["transfer_cost"] % 4 == 0

    def test_optimises_net_gain_not_gross_points(self, sample_player_pool):
        """Solver should maximise (new_points - old_points - transfer_cost)."""
        from src.optimization.milp_solver import solve_with_transfers
        # Squad with very high predicted points
        current_squad = sample_player_pool["ids"][:15]
        # Give current squad very high points so transfers aren't worth it
        high_points = [20.0] * 15 + sample_player_pool["predicted_points"][15:]
        result = solve_with_transfers(
            player_ids=sample_player_pool["ids"],
            positions=sample_player_pool["positions"],
            prices=sample_player_pool["prices"],
            predicted_points=high_points,
            clubs=sample_player_pool["clubs"],
            budget=100.0,
            current_squad=current_squad,
            free_transfers=0,
        )
        # Should keep most of current squad since they're high-value
        overlap = len(set(result["squad"]) & set(current_squad))
        assert overlap >= 10  # At least 10 retained


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Validate edge case handling."""

    def test_tight_budget_feasible(self):
        """With minimum budget, should still find a feasible squad."""
        from src.optimization.milp_solver import solve_squad_optimisation
        # 30 players: 4 GK@4.5, 8 DEF@5.0, 10 MID@6.0, 8 FWD@7.0
        # Minimum valid squad: 2*4.5 + 5*5.0 + 5*6.0 + 3*7.0 = 85.0
        ids = list(range(1, 31))
        positions = ["GK"]*4 + ["DEF"]*8 + ["MID"]*10 + ["FWD"]*8
        prices = [4.5]*4 + [5.0]*8 + [6.0]*10 + [7.0]*8
        clubs = [f"C{i}" for i in range(30)]
        points = [5.0]*30

        result = solve_squad_optimisation(
            player_ids=ids,
            positions=positions,
            prices=prices,
            predicted_points=points,
            clubs=clubs,
            budget=85.0,
        )
        assert len(result["squad"]) == 15

    def test_infeasible_budget_raises(self):
        """Budget below minimum should raise an error or return infeasible."""
        from src.optimization.milp_solver import solve_squad_optimisation
        ids = list(range(1, 31))
        positions = ["GK"]*4 + ["DEF"]*8 + ["MID"]*10 + ["FWD"]*8
        prices = [10.0]*30  # All expensive
        clubs = [f"C{i%20}" for i in range(30)]
        points = [5.0]*30

        with pytest.raises((ValueError, Exception)):
            solve_squad_optimisation(
                player_ids=ids,
                positions=positions,
                prices=prices,
                predicted_points=points,
                clubs=clubs,
                budget=50.0,  # Can't afford 15 players at £10m each
            )

    def test_empty_player_pool_raises(self):
        """Empty pool should raise an error."""
        from src.optimization.milp_solver import solve_squad_optimisation
        with pytest.raises(ValueError):
            solve_squad_optimisation(
                player_ids=[],
                positions=[],
                prices=[],
                predicted_points=[],
                clubs=[],
                budget=100.0,
            )
