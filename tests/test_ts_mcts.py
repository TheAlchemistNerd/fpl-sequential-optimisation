"""
Tests for MCTS modules (mcts_node.py, rollout.py, ts_mcts.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# MCTS Node Tests
# ===========================================================================


class TestMCTSNode:
    """Validate MCTS tree node operations."""

    def test_node_creation(self):
        from src.optimization.mcts_node import MCTSNode
        node = MCTSNode(state={"squad": [1, 2, 3]})
        assert node.state == {"squad": [1, 2, 3]}
        assert node.visit_count == 0
        assert node.total_reward == 0.0
        assert node.is_leaf
        assert node.is_fully_expanded  # No untried actions = fully expanded

    def test_ucb1_unvisited_returns_inf(self):
        from src.optimization.mcts_node import MCTSNode
        parent = MCTSNode(state={})
        child = MCTSNode(state={}, parent=parent)
        parent.children["a"] = child
        # Unvisited child should have infinite UCB1
        assert child.ucb1() == float("inf")

    def test_ucb1_visited_returns_finite(self):
        from src.optimization.mcts_node import MCTSNode
        parent = MCTSNode(state={})
        parent.visit_count = 10
        child = MCTSNode(state={}, parent=parent)
        child.visit_count = 5
        child.total_reward = 25.0
        parent.children["a"] = child
        ucb = child.ucb1()
        assert np.isfinite(ucb)
        # exploitation = 25/5 = 5.0, exploration = √2 · √(log(10)/5) ≈ 0.96
        assert ucb == pytest.approx(5.0 + 1.414 * np.sqrt(np.log(10) / 5), abs=0.01)

    def test_select_best_child(self):
        from src.optimization.mcts_node import MCTSNode
        parent = MCTSNode(state={})
        parent.visit_count = 100
        c1 = MCTSNode(state={}, parent=parent, visit_count=10, total_reward=50.0)
        c2 = MCTSNode(state={}, parent=parent, visit_count=10, total_reward=30.0)
        parent.children["a"] = c1
        parent.children["b"] = c2
        best = parent.select_best_child()
        assert best is c1  # c1 has higher exploitation (5.0 vs 3.0)

    def test_expand(self):
        from src.optimization.mcts_node import MCTSNode
        node = MCTSNode(state={}, untried_actions=["a", "b", "c"])
        child = node.expand("b", {"new": "state"})
        assert "b" not in node.untried_actions
        assert "b" in node.children
        assert child.parent is node
        assert child.state == {"new": "state"}

    def test_backpropagate(self):
        from src.optimization.mcts_node import MCTSNode
        root = MCTSNode(state={})
        child = MCTSNode(state={}, parent=root)
        root.children["a"] = child
        child.visit_count = 0
        child.total_reward = 0.0

        child.backpropagate(100.0, discount=0.9)
        assert child.visit_count == 1
        assert child.total_reward == pytest.approx(100.0)
        assert root.visit_count == 1
        assert root.total_reward == pytest.approx(90.0)  # 100 * 0.9

    def test_repr(self):
        from src.optimization.mcts_node import MCTSNode
        node = MCTSNode(state={}, visit_count=5, total_reward=25.0)
        r = repr(node)
        assert "visits=5" in r
        assert "reward=25.0" in r


# ===========================================================================
# Rollout Tests
# ===========================================================================


class TestRollout:
    """Validate forward simulation rollouts."""

    def test_simulate_gameweek_returns_points(self):
        from src.optimization.rollout import simulate_gameweek
        rng = np.random.default_rng(42)
        squad = [1, 2, 3]
        points_map = {1: 5.0, 2: 3.0, 3: 7.0}
        beliefs = {1: np.array([0.8, 0.2]), 2: np.array([0.5, 0.5]), 3: np.array([0.1, 0.9])}
        trans = np.array([[0.8, 0.2], [0.3, 0.7]])

        total_points, updated = simulate_gameweek(squad, points_map, beliefs, trans, rng)
        assert total_points >= 0  # Points are non-negative
        assert len(updated) == 3  # All players updated

    def test_simulate_gameweek_belief_update(self):
        from src.optimization.rollout import simulate_gameweek
        rng = np.random.default_rng(42)
        squad = [1]
        points_map = {1: 5.0}
        beliefs = {1: np.array([0.8, 0.2])}
        trans = np.array([[0.9, 0.1], [0.2, 0.8]])

        _, updated = simulate_gameweek(squad, points_map, beliefs, trans, rng)
        # b_next = [0.8, 0.2] · [[0.9, 0.1], [0.2, 0.8]] = [0.76, 0.24]
        assert np.allclose(updated[1], [0.76, 0.24], atol=0.01)

    def test_rollout_multi_gameweek(self):
        from src.optimization.rollout import rollout
        rng = np.random.default_rng(42)
        squad = [1, 2, 3]
        beliefs = {1: np.array([0.8, 0.2]), 2: np.array([0.5, 0.5]), 3: np.array([0.1, 0.9])}
        trans = np.array([[0.8, 0.2], [0.3, 0.7]])
        points_maps = [{1: 5.0, 2: 3.0, 3: 7.0} for _ in range(3)]

        reward = rollout(squad, beliefs, points_maps, trans, rng=rng, discount=0.95)
        assert reward >= 0  # Cumulative reward should be non-negative


# ===========================================================================
# TS+MCTS Integration Tests
# ===========================================================================


class TestTSMCTS:
    """Validate TS+MCTS orchestration."""

    @pytest.fixture
    def sample_scenario(self):
        """Sample MCTS scenario."""
        np.random.seed(42)
        n = 30
        positions = ["GK"]*4 + ["DEF"]*8 + ["MID"]*10 + ["FWD"]*8
        prices = [4.5 + np.random.uniform(0, 3.5) for _ in range(n)]
        clubs = [f"Club_{i % 20}" for i in range(n)]
        points = {i: float(np.random.normal(5, 3)) for i in range(1, n+1)}

        K = 4
        beliefs = {i: np.random.dirichlet([2.0]*K) for i in range(1, n+1)}
        trans = np.eye(K) * 0.7 + (1 - np.eye(K)) * 0.3 / (K - 1)

        return {
            "player_pool": {
                "ids": list(range(1, n + 1)),
                "positions": positions,
                "prices": prices,
                "clubs": clubs,
                "points": points,
            },
            "current_squad": list(range(1, 16)),
            "belief_states": beliefs,
            "transition_matrix": trans,
            "budget": 100.0,
        }

    def test_ts_mcts_returns_valid_result(self, sample_scenario):
        """TS+MCTS should return a valid result dict."""
        from src.optimization.ts_mcts import ts_mcts_select
        rng = np.random.default_rng(42)

        result = ts_mcts_select(
            player_pool=sample_scenario["player_pool"],
            current_squad=sample_scenario["current_squad"],
            belief_states=sample_scenario["belief_states"],
            transition_matrix=sample_scenario["transition_matrix"],
            budget=sample_scenario["budget"],
            n_simulations=50,  # Small for test speed
            horizon=2,
            rng=rng,
        )
        assert "best_action" in result
        assert "best_squad" in result
        assert "simulations" in result
        assert result["simulations"] == 50
        assert result["root_visits"] > 0

    def test_ts_mcts_best_action_is_dict(self, sample_scenario):
        """Best action should be a transfer action dict."""
        from src.optimization.ts_mcts import ts_mcts_select
        rng = np.random.default_rng(42)

        result = ts_mcts_select(
            player_pool=sample_scenario["player_pool"],
            current_squad=sample_scenario["current_squad"],
            belief_states=sample_scenario["belief_states"],
            transition_matrix=sample_scenario["transition_matrix"],
            budget=sample_scenario["budget"],
            n_simulations=50,
            rng=rng,
        )
        assert isinstance(result["best_action"], dict)
        assert "in" in result["best_action"] or "label" in result["best_action"]

    def test_ts_mcts_no_transfer_action_exists(self, sample_scenario):
        """Should always include a no-transfer action."""
        from src.optimization.ts_mcts import _generate_transfer_actions
        actions = _generate_transfer_actions(
            current_squad=sample_scenario["current_squad"],
            player_pool=sample_scenario["player_pool"],
        )
        # Should have at least the no_transfer action
        no_transfer = [a for a in actions if a.get("label") == "no_transfer"]
        assert len(no_transfer) >= 1

    def test_ts_mcts_action_pruning(self, sample_scenario):
        """Action space should be pruned to top-K candidates."""
        from src.optimization.ts_mcts import _generate_transfer_actions
        actions = _generate_transfer_actions(
            current_squad=sample_scenario["current_squad"],
            player_pool=sample_scenario["player_pool"],
            top_k=10,
        )
        # Should be limited (no_transfer + single transfers from top-10)
        assert len(actions) <= 500  # Reasonable upper bound
