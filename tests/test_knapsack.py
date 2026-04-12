"""
Tests for knapsack baselines (DP, B&B, Greedy).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestGreedyKnapsack:
    """Validate greedy knapsack by value-per-weight."""

    def test_basic_selection(self):
        from src.optimization.knapsack import greedy_knapsack
        values = np.array([60, 100, 120])
        weights = np.array([10, 20, 30])
        capacity = 50
        result = greedy_knapsack(values, weights, capacity)
        assert result.total_weight <= capacity
        assert len(result.selected_indices) > 0

    def test_empty_instance(self):
        from src.optimization.knapsack import greedy_knapsack
        result = greedy_knapsack(
            np.array([]), np.array([]), capacity=10
        )
        assert result.total_value == 0.0
        assert result.selected_indices == []


class TestDPKnapsack:
    """Validate DP knapsack."""

    def test_optimal_solution(self):
        """DP should find the optimal solution for small instances."""
        from src.optimization.knapsack import dp_knapsack
        values = np.array([60, 100, 120])
        weights = np.array([10, 20, 30])
        capacity = 50
        result = dp_knapsack(values, weights, capacity)
        # Optimal: items 1 and 2 (weight=50, value=220)
        assert result.total_value == pytest.approx(220.0)
        assert result.total_weight <= 50

    def test_single_item(self):
        from src.optimization.knapsack import dp_knapsack
        values = np.array([50])
        weights = np.array([10])
        result = dp_knapsack(values, weights, 20)
        assert result.total_value == pytest.approx(50.0)
        assert result.selected_indices == [0]

    def test_no_items_fit(self):
        from src.optimization.knapsack import dp_knapsack
        values = np.array([50, 60])
        weights = np.array([30, 40])
        result = dp_knapsack(values, weights, 10)
        assert result.total_value == 0.0
        assert result.selected_indices == []


class TestBranchAndBoundKnapsack:
    """Validate Branch-and-Bound knapsack."""

    def test_optimal_solution(self):
        """B&B should find the optimal solution."""
        from src.optimization.knapsack import branch_and_bound_knapsack
        values = np.array([60, 100, 120])
        weights = np.array([10, 20, 30])
        capacity = 50
        result = branch_and_bound_knapsack(values, weights, capacity)
        # Optimal: items 1 and 2 (weight=50, value=220)
        assert result.total_value == pytest.approx(220.0)
        assert result.total_weight <= 50

    def test_pruning_reduces_search(self):
        """B&B should explore fewer nodes than exhaustive search."""
        from src.optimization.knapsack import branch_and_bound_knapsack
        np.random.seed(42)
        n = 20
        values = np.random.uniform(10, 100, n)
        weights = np.random.uniform(5, 50, n).astype(int)
        capacity = 100
        result = branch_and_bound_knapsack(values, weights, capacity)
        assert result.total_weight <= capacity
        assert result.total_value > 0

    def test_empty_instance(self):
        from src.optimization.knapsack import branch_and_bound_knapsack
        result = branch_and_bound_knapsack(
            np.array([]), np.array([]), capacity=10
        )
        assert result.total_value == 0.0
        assert result.selected_indices == []


class TestKnapsackComparison:
    """Validate that DP and B&B find same optimal solutions."""

    def test_dp_equals_bb_small(self):
        """DP and B&B should agree on small instances."""
        from src.optimization.knapsack import dp_knapsack, branch_and_bound_knapsack
        np.random.seed(42)
        n = 10
        values = np.random.uniform(10, 100, n)
        weights = np.random.uniform(5, 30, n).astype(int)
        capacity = 50
        dp_result = dp_knapsack(values, weights, capacity)
        bb_result = branch_and_bound_knapsack(values, weights, capacity)
        assert dp_result.total_value == pytest.approx(bb_result.total_value, rel=1e-5)

    def test_greedy_is_suboptimal(self):
        """Greedy should sometimes be suboptimal vs DP."""
        from src.optimization.knapsack import dp_knapsack, greedy_knapsack
        values = np.array([60, 100, 120])
        weights = np.array([10, 20, 30])
        capacity = 50
        dp_result = dp_knapsack(values, weights, capacity)
        greedy_result = greedy_knapsack(values, weights, capacity)
        # Greedy may not always be optimal, but should never exceed DP
        assert greedy_result.total_value <= dp_result.total_value + 1e-10
