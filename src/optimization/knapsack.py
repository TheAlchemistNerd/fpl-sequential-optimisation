"""
Knapsack Baselines — Ch. 10 §10.3–10.5.

Implements three baseline algorithms for the FPL squad selection knapsack:
    1. Dynamic Programming (exact, O(N·W))
    2. Branch-and-Bound with fractional relaxation (exact, typically fast)
    3. Greedy by value-per-weight (approximate, O(N log N))

These serve as baselines for evaluating the MILP solver and for ablation
studies (Chapter 12).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KnapsackResult:
    """Result of a knapsack optimisation."""
    selected_indices: list[int] = field(default_factory=list)
    total_value: float = 0.0
    total_weight: float = 0.0
    method: str = ""


def greedy_knapsack(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
) -> KnapsackResult:
    """
    Greedy knapsack by value-per-weight ratio.

    Sorts items by v_i/w_i descending and selects until capacity is exhausted.
    Does NOT guarantee optimality for 0/1 knapsack.

    Complexity: O(N log N) for sorting.

    Parameters
    ----------
    values : np.ndarray (N,)
        Value (predicted points) for each item.
    weights : np.ndarray (N,)
        Weight (price) for each item.
    capacity : float
        Maximum total weight allowed.

    Returns
    -------
    KnapsackResult
    """
    n = len(values)
    ratios = values / (weights + 1e-10)
    sorted_indices = np.argsort(ratios)[::-1]

    selected = []
    total_value = 0.0
    total_weight = 0.0

    for idx in sorted_indices:
        if total_weight + weights[idx] <= capacity:
            selected.append(int(idx))
            total_value += values[idx]
            total_weight += weights[idx]

    return KnapsackResult(
        selected_indices=selected,
        total_value=total_value,
        total_weight=total_weight,
        method="greedy",
    )


def dp_knapsack(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: int,
) -> KnapsackResult:
    """
    0/1 Knapsack via Dynamic Programming.

    Uses the Bellman recursion:
        V[i, w] = max(V[i-1, w], v_i + V[i-1, w - w_i])

    Complexity: O(N·W) time, O(N·W) space.

    Parameters
    ----------
    values : np.ndarray (N,)
    weights : np.ndarray (N,)
    capacity : int
        Maximum total weight (must be integer for DP table indexing).

    Returns
    -------
    KnapsackResult
    """
    n = len(values)
    W = int(capacity)

    # DP table
    V = np.zeros((n + 1, W + 1))
    for i in range(1, n + 1):
        v_i = values[i - 1]
        w_i = int(weights[i - 1])
        for w in range(W + 1):
            if w_i > w:
                V[i, w] = V[i - 1, w]
            else:
                V[i, w] = max(V[i - 1, w], v_i + V[i - 1, w - w_i])

    # Reconstruct solution
    selected = []
    w = W
    for i in range(n, 0, -1):
        if V[i, w] != V[i - 1, w]:
            selected.append(i - 1)
            w -= int(weights[i - 1])

    selected.reverse()
    total_value = V[n, W]
    total_weight = sum(weights[i] for i in selected)

    return KnapsackResult(
        selected_indices=selected,
        total_value=float(total_value),
        total_weight=float(total_weight),
        method="dp",
    )


def _fractional_knapsack_bound(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    fixed_items: list[int],
    excluded_items: set[int],
) -> float:
    """
    Compute upper bound via fractional knapsack relaxation.

    Allows partial items, which provides an upper bound for the 0/1 problem.
    Used by Branch-and-Bound for pruning.
    """
    remaining_capacity = capacity - sum(weights[i] for i in fixed_items)
    current_value = sum(values[i] for i in fixed_items)

    if remaining_capacity < 0:
        return -1.0  # Infeasible

    # Sort remaining items by ratio
    candidates = [
        i for i in range(len(values))
        if i not in fixed_items and i not in excluded_items
    ]
    ratios = [(values[i] / (weights[i] + 1e-10), i) for i in candidates]
    ratios.sort(reverse=True)

    for ratio, idx in ratios:
        if weights[idx] <= remaining_capacity:
            current_value += values[idx]
            remaining_capacity -= weights[idx]
        else:
            # Take fraction
            current_value += ratio * remaining_capacity
            break

    return current_value


def branch_and_bound_knapsack(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    time_limit_steps: int = 100000,
) -> KnapsackResult:
    """
    0/1 Knapsack via Branch-and-Bound with fractional relaxation.

    Uses a priority queue (sorted list) ordered by upper bound.
    Prunes branches whose upper bound is worse than the best known solution.

    Parameters
    ----------
    values : np.ndarray (N,)
    weights : np.ndarray (N,)
    capacity : float
    time_limit_steps : int
        Maximum number of nodes to explore (safety limit).

    Returns
    -------
    KnapsackResult
    """
    n = len(values)

    # Sort items by ratio for better branching
    ratios = values / (weights + 1e-10)
    sorted_order = np.argsort(ratios)[::-1]
    sorted_values = values[sorted_order]
    sorted_weights = weights[sorted_order]

    best_value = 0.0
    best_selection = []

    # Stack: (level, current_value, current_weight, selected_indices)
    stack = [(0, 0.0, 0.0, [])]
    steps = 0

    while stack and steps < time_limit_steps:
        steps += 1
        level, curr_val, curr_wt, selected = stack.pop()

        if level == n or curr_wt >= capacity:
            if curr_val > best_value:
                best_value = curr_val
                best_selection = list(selected)
            continue

        # Upper bound via fractional relaxation
        fixed = selected
        excluded = set()
        bound = _fractional_knapsack_bound(
            sorted_values, sorted_weights, capacity, fixed, excluded
        )

        if bound <= best_value:
            continue  # Prune

        # Branch: include item at `level`
        idx = level
        if curr_wt + sorted_weights[idx] <= capacity:
            stack.append((
                level + 1,
                curr_val + sorted_values[idx],
                curr_wt + sorted_weights[idx],
                selected + [idx],
            ))

        # Branch: exclude item at `level`
        stack.append((level + 1, curr_val, curr_wt, selected))

    # Map back to original indices
    original_indices = [int(sorted_order[i]) for i in best_selection]
    total_weight = sum(weights[i] for i in original_indices)

    logger.info(
        "B&B knapsack: selected=%d, value=%.1f, weight=%.1f, steps=%d",
        len(original_indices), best_value, total_weight, steps,
    )

    return KnapsackResult(
        selected_indices=original_indices,
        total_value=best_value,
        total_weight=total_weight,
        method="branch_and_bound",
    )
