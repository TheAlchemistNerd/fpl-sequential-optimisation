"""
MILP Constrained Optimiser — Chapter 10 §10.6, §10.9.

Solves the FPL squad selection problem as a Mixed-Integer Linear Program:
    max  Σ v_pred[i] · s_i + v_pred[captain] - transfer_cost
    s.t. squad size, formation, budget, per-team, captaincy constraints.

Uses PuLP with CBC solver (open-source, included with PuLP).

Decision Variables (Ch. 10 §10.6.1):
    x_i ∈ {0,1}  — player i is in the squad
    s_i ∈ {0,1}  — player i is in the starting XI
    c_i ∈ {0,1}  — player i is captain
    v_i ∈ {0,1}  — player i is vice-captain
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import pulp
except ImportError:
    pulp = None

logger = logging.getLogger(__name__)

# FPL constraints (Ch. 10 §10.6.2)
SQUAD_SIZE = 15
STARTING_XI_SIZE = 11
POSITION_REQUIREMENTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
TRANSFER_COST_PER_EXTRA = 4


def _check_pulp() -> None:
    """Raise informative error if PuLP not installed."""
    if pulp is None:
        raise ImportError(
            "PuLP is required for MILP solving. Install with: pip install pulp"
        )


def _add_squad_constraints(
    prob: Any,
    x: dict[int, Any],
    s: dict[int, Any],
    c: dict[int, Any],
    vc: dict[int, Any],
    player_count: int,
    positions: list[str],
    prices: list[float],
    clubs: list[str],
    unique_clubs: list[str],
    budget: float,
) -> None:
    """Add the squad, XI, formation, budget, and captaincy constraints."""
    prob += pulp.lpSum(x[i] for i in range(player_count)) == SQUAD_SIZE, "Squad_Size"

    for pos, required in POSITION_REQUIREMENTS.items():
        pos_indices = [i for i in range(player_count) if positions[i] == pos]
        prob += (
            pulp.lpSum(x[i] for i in pos_indices) == required,
            f"Formation_{pos}",
        )

    prob += (
        pulp.lpSum(prices[i] * x[i] for i in range(player_count)) <= budget,
        "Budget",
    )

    for club in unique_clubs:
        club_indices = [i for i in range(player_count) if clubs[i] == club]
        if club_indices:
            prob += (
                pulp.lpSum(x[i] for i in club_indices) <= MAX_PER_CLUB,
                f"PerTeam_{club}",
            )

    prob += (
        pulp.lpSum(s[i] for i in range(player_count)) == STARTING_XI_SIZE,
        "XI_Size",
    )
    for i in range(player_count):
        prob += s[i] <= x[i], f"Starter_From_Squad_{i}"

    gk_indices = [i for i in range(player_count) if positions[i] == "GK"]
    def_indices = [i for i in range(player_count) if positions[i] == "DEF"]
    fwd_indices = [i for i in range(player_count) if positions[i] == "FWD"]
    prob += pulp.lpSum(s[i] for i in gk_indices) >= 1, "XI_Min_GK"
    prob += pulp.lpSum(s[i] for i in def_indices) >= 3, "XI_Min_DEF"
    prob += pulp.lpSum(s[i] for i in fwd_indices) >= 1, "XI_Min_FWD"

    prob += pulp.lpSum(c[i] for i in range(player_count)) == 1, "One_Captain"
    prob += pulp.lpSum(vc[i] for i in range(player_count)) == 1, "One_Vice_Captain"
    for i in range(player_count):
        prob += c[i] <= s[i], f"Captain_Is_Starter_{i}"
        prob += vc[i] <= s[i], f"Vice_Captain_Is_Starter_{i}"
        prob += c[i] + vc[i] <= 1, f"Captain_Ne_Vice_{i}"


# ---------------------------------------------------------------------------
# Core Squad Optimisation (Ch. 10 §10.6)
# ---------------------------------------------------------------------------


def solve_squad_optimisation(
    player_ids: list[int],
    positions: list[str],
    prices: list[float],
    predicted_points: list[float],
    clubs: list[str],
    budget: float,
) -> dict[str, Any]:
    """
    Solve the FPL squad selection MILP.

    Parameters
    ----------
    player_ids : list[int]
        All available player IDs (typically ~600-720).
    positions : list[str]
        Position for each player ("GK", "DEF", "MID", "FWD").
    prices : list[float]
        Price in £m for each player.
    predicted_points : list[float]
        Predicted points for the upcoming gameweek (from Thompson Sampling
        or DDQN).
    clubs : list[str]
        Club name for each player.
    budget : float
        Total budget in £m.

    Returns
    -------
    dict with keys:
        - squad: list[int] — selected player IDs
        - starting_xi: list[int] — starting 11 player IDs
        - captain: int — captain player ID
        - vice_captain: int — vice-captain player ID
        - total_points: float — predicted total points
        - total_cost: float — squad total price
    """
    _check_pulp()

    if not player_ids:
        raise ValueError("Player pool is empty.")

    N = len(player_ids)
    unique_clubs = sorted(set(clubs))

    # -- Problem definition ------------------------------------------------
    prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)

    # -- Decision variables ------------------------------------------------
    x = pulp.LpVariable.dicts("squad", range(N), cat=pulp.LpBinary)
    s = pulp.LpVariable.dicts("starter", range(N), cat=pulp.LpBinary)
    c = pulp.LpVariable.dicts("captain", range(N), cat=pulp.LpBinary)
    vc = pulp.LpVariable.dicts("vice_captain", range(N), cat=pulp.LpBinary)

    # -- Objective: maximise starting XI points + captain bonus ------------
    prob += (
        pulp.lpSum(predicted_points[i] * s[i] for i in range(N))
        + pulp.lpSum(predicted_points[i] * c[i] for i in range(N)),
        "Total_Points",
    )

    _add_squad_constraints(
        prob, x, s, c, vc, N, positions, prices, clubs, unique_clubs, budget
    )

    # -- Solve -------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(
            f"MILP solver returned {pulp.LpStatus[status]}. "
            "The problem may be infeasible (check budget vs minimum squad cost)."
        )

    # -- Extract solution --------------------------------------------------
    squad = [player_ids[i] for i in range(N) if pulp.value(x[i]) > 0.5]
    starting_xi = [player_ids[i] for i in range(N) if pulp.value(s[i]) > 0.5]
    captain = next(player_ids[i] for i in range(N) if pulp.value(c[i]) > 0.5)
    vice_captain = next(player_ids[i] for i in range(N) if pulp.value(vc[i]) > 0.5)
    total_points = pulp.value(prob.objective)
    total_cost = sum(
        prices[i] for i in range(N) if pulp.value(x[i]) > 0.5
    )

    logger.info(
        "MILP solved: squad=%d, XI=%d, captain=%d, points=%.1f, cost=%.1f",
        len(squad), len(starting_xi), captain, total_points, total_cost,
    )

    return {
        "squad": squad,
        "starting_xi": starting_xi,
        "captain": captain,
        "vice_captain": vice_captain,
        "total_points": total_points,
        "total_cost": total_cost,
    }


# ---------------------------------------------------------------------------
# Transfer-Aware Optimisation (Ch. 10 §10.9)
# ---------------------------------------------------------------------------


def solve_with_transfers(
    player_ids: list[int],
    positions: list[str],
    prices: list[float],
    predicted_points: list[float],
    clubs: list[str],
    budget: float,
    current_squad: list[int],
    free_transfers: int = 1,
) -> dict[str, Any]:
    """
    Solve squad optimisation with piecewise linear transfer costs.

    Transfer cost encoding (Ch. 10 §10.9.2):
        cost ≥ 0
        cost ≥ 4 · (k - f)
    where k = number of transfers, f = free transfers.

    Parameters
    ----------
    current_squad : list[int]
        Current squad player IDs.
    free_transfers : int
        Number of free transfers available (typically 1 or 2).

    Returns
    -------
    dict with keys from solve_squad_optimisation plus:
        - transfer_cost: int — points deducted for transfers
        - n_transfers: int — number of players changed
    """
    _check_pulp()

    if not player_ids:
        raise ValueError("Player pool is empty.")

    N = len(player_ids)
    unique_clubs = sorted(set(clubs))
    current_set = set(current_squad)

    # -- Problem definition ------------------------------------------------
    prob = pulp.LpProblem("FPL_Transfer_Optimisation", pulp.LpMaximize)

    # -- Decision variables ------------------------------------------------
    x = pulp.LpVariable.dicts("squad", range(N), cat=pulp.LpBinary)
    s = pulp.LpVariable.dicts("starter", range(N), cat=pulp.LpBinary)
    c = pulp.LpVariable.dicts("captain", range(N), cat=pulp.LpBinary)
    vc = pulp.LpVariable.dicts("vice_captain", range(N), cat=pulp.LpBinary)

    # Transfer count auxiliary (linearised absolute difference)
    # k = (1/2) · Σ |x_i - x_i_prev|
    # For binary variables: |x_i - x_i_prev| = x_i + x_i_prev - 2·x_i·x_i_prev
    # We use indicator variables: t_i ≥ x_i - x_i_prev, t_i ≥ x_i_prev - x_i
    t = pulp.LpVariable.dicts("transfer", range(N), cat=pulp.LpBinary)
    for i in range(N):
        is_current = 1.0 if player_ids[i] in current_set else 0.0
        prob += t[i] >= x[i] - is_current, f"Transfer_LB_{i}"
        prob += t[i] >= is_current - x[i], f"Transfer_UB_{i}"

    # Total transfers
    k = pulp.lpSum(t[i] for i in range(N)) / 2

    # Transfer cost auxiliary variables (Ch. 10 §10.9.2)
    cost = pulp.LpVariable("transfer_cost", lowBound=0, cat=pulp.LpInteger)
    prob += cost >= 0, "Transfer_Cost_NonNeg"
    prob += cost >= TRANSFER_COST_PER_EXTRA * (k - free_transfers), "Transfer_Cost_Piecewise"

    # -- Objective: net gain = new_points - transfer_cost -------------------
    prob += (
        pulp.lpSum(predicted_points[i] * s[i] for i in range(N))
        + pulp.lpSum(predicted_points[i] * c[i] for i in range(N))
        - cost,
        "Net_Gain",
    )

    # Shared squad/lineup constraints are identical to the non-transfer model.
    _add_squad_constraints(
        prob, x, s, c, vc, N, positions, prices, clubs, unique_clubs, budget
    )

    # -- Solve -------------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(
            f"MILP solver returned {pulp.LpStatus[status]}. "
            "Problem may be infeasible."
        )

    # -- Extract solution --------------------------------------------------
    squad = [player_ids[i] for i in range(N) if pulp.value(x[i]) > 0.5]
    starting_xi = [player_ids[i] for i in range(N) if pulp.value(s[i]) > 0.5]
    captain = next(player_ids[i] for i in range(N) if pulp.value(c[i]) > 0.5)
    vice_captain = next(player_ids[i] for i in range(N) if pulp.value(vc[i]) > 0.5)
    total_points = pulp.value(prob.objective)
    n_transfers = int(round(pulp.value(k)))
    transfer_cost_val = int(round(pulp.value(cost)))

    logger.info(
        "MILP with transfers: squad=%d, transfers=%d, cost=%dpts, net=%.1f",
        len(squad), n_transfers, transfer_cost_val, total_points,
    )

    return {
        "squad": squad,
        "starting_xi": starting_xi,
        "captain": captain,
        "vice_captain": vice_captain,
        "total_points": total_points,
        "transfer_cost": transfer_cost_val,
        "n_transfers": n_transfers,
        "total_cost": sum(
            prices[i] for i in range(N) if pulp.value(x[i]) > 0.5
        ),
    }
