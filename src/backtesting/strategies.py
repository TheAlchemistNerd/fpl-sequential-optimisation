"""Thesis-linked FPL challengers and legal-squad action construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.optimization.selection.milp_solver import (
    solve_squad_optimisation,
    solve_with_transfers,
)


StrategyName = Literal[
    "hold",
    "form_milp",
    "xg_milp",
    "thompson_nb_milp",
    "random_valid",
    "hdp_gmm_milp",
    "hdp_hmm_milp",
]
STRATEGIES: tuple[StrategyName, ...] = (
    "hold",
    "form_milp",
    "xg_milp",
    "thompson_nb_milp",
    "random_valid",
    "hdp_gmm_milp",
    "hdp_hmm_milp",
)
PREDICTION_STRATEGIES = {
    "hdp_gmm_milp": "hdp_gmm",
    "hdp_hmm_milp": "hdp_hmm",
}
POSITION_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


@dataclass(frozen=True)
class Decision:
    """A complete squad and gameweek action returned by a challenger."""

    squad: list[int]
    starting_xi: list[int]
    captain: int
    vice_captain: int
    bench_order: list[int]
    objective_value: float | None = None


def make_decision(
    strategy: StrategyName,
    players: pd.DataFrame,
    current_squad: list[int],
    budget: float | None,
    free_transfers: int,
    rng: np.random.Generator,
    *,
    bank_balance: float | None = None,
    current_sell_prices: dict[int, float] | None = None,
    predicted_points: dict[int, float] | None = None,
) -> Decision:
    """Choose this GW's action from a point-in-time player feature table."""
    if strategy == "random_valid" and bank_balance is None:
        if budget is None:
            raise ValueError("A budget is required to initialize a random-valid squad")
        return random_valid_decision(players, budget, rng)

    if strategy == "hold":
        pool = players.loc[players["player_id"].isin(current_squad)].copy()
        if len(pool) != len(current_squad):
            raise ValueError("The hold policy requires metadata for every current player")
        result = solve_squad_optimisation(
            player_ids=pool["player_id"].astype(int).tolist(),
            positions=pool["position"].tolist(),
            prices=pool["price"].astype(float).tolist(),
            predicted_points=pool["form_points"].astype(float).tolist(),
            clubs=pool["club"].tolist(),
            budget=float(pool["price"].sum()),
        )
        squad = list(current_squad)
        scores = pool.set_index("player_id")["form_points"]
        return _decision_from_result(result, squad, pool, scores)

    score_column = {
        "form_milp": "form_points",
        "xg_milp": "xgi_points",
    }.get(strategy, "form_points")
    if strategy in PREDICTION_STRATEGIES:
        if predicted_points is None:
            raise ValueError(f"{strategy} requires a point-in-time prediction map")
        missing_predictions = set(players["player_id"].astype(int)) - set(predicted_points)
        if missing_predictions:
            raise ValueError(
                f"Missing {strategy} predictions for players: {sorted(missing_predictions)}"
            )
        score_column = "_model_prediction"
        players = players.copy()
        players[score_column] = players["player_id"].map(predicted_points)
        point_map = {
            int(player_id): float(score)
            for player_id, score in zip(players["player_id"], players[score_column])
        }
    else:
        point_map = {
            int(row.player_id): float(getattr(row, score_column))
            for row in players.itertuples(index=False)
        }
    if strategy == "random_valid":
        # Random objective weights make this a feasible stochastic policy
        # under the same squad, wallet, transfer, and hit constraints.
        point_map = {
            int(player_id): float(rng.uniform(0.0, 12.0))
            for player_id in players["player_id"]
        }
        score_column = "_random_score"
        players = players.copy()
        players[score_column] = players["player_id"].map(point_map)
    player_pool = {
        "ids": players["player_id"].astype(int).tolist(),
        "positions": players["position"].tolist(),
        "prices": players["price"].astype(float).tolist(),
        "clubs": players["club"].astype(str).tolist(),
    }

    if strategy == "thompson_nb_milp":
        # This invokes the repository's implemented NB sampling + MILP wrapper.
        # Its dispersion is heuristic; it is not the thesis's HDP/MCMC posterior.
        from src.optimization.selection.thompson_sampling import thompson_sampling_milp

        result = thompson_sampling_milp(
            player_pool=player_pool,
            points_map=point_map,
            budget=budget,
            current_squad=current_squad,
            free_transfers=free_transfers,
            rng=rng,
            bank_balance=bank_balance,
            current_sell_prices=current_sell_prices,
        )
    else:
        result = solve_with_transfers(
            player_ids=player_pool["ids"],
            positions=player_pool["positions"],
            prices=player_pool["prices"],
            predicted_points=[point_map[player_id] for player_id in player_pool["ids"]],
            clubs=player_pool["clubs"],
            budget=budget,
            current_squad=current_squad,
            free_transfers=free_transfers,
            bank_balance=bank_balance,
            current_sell_prices=current_sell_prices,
        )

    ordering_scores = players.set_index("player_id")[score_column]
    return _decision_from_result(result, list(result["squad"]), players, ordering_scores)


def random_valid_decision(
    players: pd.DataFrame,
    budget: float,
    rng: np.random.Generator,
    *,
    attempts: int = 2_000,
) -> Decision:
    """Draw a seeded feasible squad, then a legal XI and captain pair.

    Sampling is rejection-based, so it is a reproducible random-valid baseline,
    not a claim of uniform probability over all feasible squads.
    """
    required = set(POSITION_COUNTS)
    if not required.issubset(set(players["position"])):
        raise ValueError("Player pool must contain players from all four positions")

    grouped = {
        position: group.reset_index(drop=True)
        for position, group in players.groupby("position", sort=False)
    }
    shortages = {
        position: count - len(grouped.get(position, ()))
        for position, count in POSITION_COUNTS.items()
        if len(grouped.get(position, ())) < count
    }
    if shortages:
        raise ValueError(f"Not enough candidates by position: {shortages}")
    squad: list[int] | None = None
    for _ in range(attempts):
        selected_parts = [
            grouped[position].iloc[
                rng.choice(len(grouped[position]), size=count, replace=False)
            ]
            for position, count in POSITION_COUNTS.items()
        ]
        selected = pd.concat(selected_parts, ignore_index=True)
        if selected["club"].value_counts().max() > 3:
            continue
        if float(selected["price"].sum()) > budget + 1e-9:
            continue
        squad = selected["player_id"].astype(int).tolist()
        break

    if squad is None:
        # A random objective keeps the fallback stochastic while using the
        # exact constraints to find a feasible squad in unusually tight pools.
        random_values = rng.random(len(players))
        fallback = solve_squad_optimisation(
            player_ids=players["player_id"].astype(int).tolist(),
            positions=players["position"].tolist(),
            prices=players["price"].astype(float).tolist(),
            predicted_points=random_values.tolist(),
            clubs=players["club"].astype(str).tolist(),
            budget=budget,
        )
        squad = list(fallback["squad"])

    selected = players.loc[players["player_id"].isin(squad)].copy()
    by_position = {
        position: selected.loc[selected["position"] == position, "player_id"].astype(int).tolist()
        for position in POSITION_COUNTS
    }
    formations = [
        (defenders, midfielders, forwards)
        for defenders in range(3, 6)
        for midfielders in range(2, 6)
        for forwards in range(1, 4)
        if defenders + midfielders + forwards == 10
    ]
    n_def, n_mid, n_fwd = formations[int(rng.integers(len(formations)))]
    starting_xi = (
        rng.choice(by_position["GK"], size=1, replace=False).astype(int).tolist()
        + rng.choice(by_position["DEF"], size=n_def, replace=False).astype(int).tolist()
        + rng.choice(by_position["MID"], size=n_mid, replace=False).astype(int).tolist()
        + rng.choice(by_position["FWD"], size=n_fwd, replace=False).astype(int).tolist()
    )
    captain, vice_captain = rng.choice(starting_xi, size=2, replace=False).astype(int).tolist()
    bench_order = [player_id for player_id in squad if player_id not in starting_xi]
    rng.shuffle(bench_order)
    return Decision(squad, starting_xi, captain, vice_captain, bench_order)


def _decision_from_result(
    result: dict,
    squad: list[int],
    players: pd.DataFrame,
    scores: pd.Series,
) -> Decision:
    starters = [int(player_id) for player_id in result["starting_xi"]]
    bench = [player_id for player_id in squad if player_id not in starters]
    bench.sort(key=lambda player_id: float(scores.loc[player_id]), reverse=True)
    return Decision(
        squad=[int(player_id) for player_id in squad],
        starting_xi=starters,
        captain=int(result["captain"]),
        vice_captain=int(result["vice_captain"]),
        bench_order=bench,
        objective_value=float(result["total_points"]),
    )
