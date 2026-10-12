"""Sequential replay engine for historical FPL challengers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.backtesting.data import build_as_of_features, outcomes_for_gameweek
from src.backtesting.strategies import (
    STRATEGIES,
    Decision,
    StrategyName,
    make_decision,
    random_valid_decision,
)

STARTING_BUDGET_TENTHS = 1_000
MAX_FREE_TRANSFERS_2023_24 = 2
HIT_PER_EXTRA_TRANSFER = 4
MINIMUM_OUTFIELD = {"DEF": 3, "MID": 2, "FWD": 1}


@dataclass(frozen=True)
class BacktestResult:
    """Detailed weekly ledger, challenger summary, and replay metadata."""

    gameweeks: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict[str, object]


def market_sell_price_tenths(purchase_price_tenths: int, market_price_tenths: int) -> int:
    """Apply the FPL half-profit rule using integer tenths of a million."""
    if market_price_tenths <= purchase_price_tenths:
        return market_price_tenths
    return purchase_price_tenths + (market_price_tenths - purchase_price_tenths) // 2


def score_lineup(
    decision: Decision,
    positions: dict[int, str],
    outcomes: dict[int, dict[str, float]],
) -> tuple[float, list[int]]:
    """Score a submitted XI, captaincy, and bench with FPL-style autosubs."""
    starting = list(decision.starting_xi)
    bench_goalkeepers = [pid for pid in decision.bench_order if positions.get(pid) == "GK"]
    bench_outfield = [
        pid for pid in decision.bench_order
        if positions.get(pid) != "GK" and outcomes.get(pid, {}).get("minutes", 0.0) > 0
    ]
    played_starters = {
        pid for pid in starting if outcomes.get(pid, {}).get("minutes", 0.0) > 0
    }
    missed_outfield = [
        pid
        for pid in starting
        if positions.get(pid) != "GK" and pid not in played_starters
    ]
    final_xi = set(played_starters)

    starting_gks = [pid for pid in starting if positions.get(pid) == "GK"]
    if starting_gks and starting_gks[0] not in played_starters:
        for goalkeeper in bench_goalkeepers:
            if outcomes.get(goalkeeper, {}).get("minutes", 0.0) > 0:
                final_xi.add(goalkeeper)
                break

    # Enumerate legal replacement assignments; at most three outfield substitutes.
    candidates: list[tuple[int, tuple[int, ...], set[int]]] = []
    for count in range(0, min(len(bench_outfield), len(missed_outfield)) + 1):
        for priorities in combinations(range(len(bench_outfield)), count):
            additions = {bench_outfield[index] for index in priorities}
            for assignment in permutations(missed_outfield, count):
                lineup = (final_xi - set(assignment)) | additions
                counts = {
                    position: sum(positions.get(pid) == position for pid in lineup)
                    for position in ("GK", "DEF", "MID", "FWD")
                }
                if counts["GK"] == 1 and all(
                    counts[position] >= required
                    for position, required in MINIMUM_OUTFIELD.items()
                ):
                    candidates.append((count, priorities, lineup))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        final_xi = candidates[0][2]

    points = sum(outcomes.get(pid, {}).get("points", 0.0) for pid in final_xi)
    captain_scored = decision.captain in final_xi
    doubled_player = decision.captain if captain_scored else decision.vice_captain
    if doubled_player in final_xi:
        points += outcomes.get(doubled_player, {}).get("points", 0.0)
    return float(points), sorted(final_xi)


def run_backtest(
    gameweeks: pd.DataFrame,
    *,
    strategies: Iterable[StrategyName] = STRATEGIES,
    seeds: Iterable[int] = (42,),
    season: str = "unknown",
    lookback: int = 5,
    max_stale_gameweeks: int = 5,
    free_transfer_cap: int = MAX_FREE_TRANSFERS_2023_24,
    dataset_sha256: str | None = None,
) -> BacktestResult:
    """Replay strategies in chronological order using strictly lagged features.

    The merged archive does not contain official deadline snapshots or purchase
    prices. This pilot approximates the player pool and market values with the
    latest observation before each GW and applies the published half-profit
    selling-price rule to each simulated manager's own purchase price.
    """
    required = {
        "player_id", "gameweek", "name", "position", "club", "price_tenths",
        "price", "minutes", "points", "xg", "xa", "clean_sheets",
    }
    missing = required - set(gameweeks.columns)
    if missing:
        raise ValueError(f"Normalized data is missing columns: {sorted(missing)}")
    weeks = sorted(int(gw) for gw in gameweeks["gameweek"].unique())
    if len(weeks) < 2:
        raise ValueError("At least two gameweeks are required for a sequential replay")
    if weeks != list(range(weeks[0], weeks[-1] + 1)):
        raise ValueError("Gameweek data must be contiguous for transfer rollover replay")
    if weeks[0] != 1:
        raise ValueError("The pilot requires GW1 to construct the shared initial squad")

    strategy_list = tuple(strategies)
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ValueError("At least one random seed is required")
    if not strategy_list:
        raise ValueError("At least one strategy is required")
    if free_transfer_cap < 1:
        raise ValueError("free_transfer_cap must be at least one")
    unknown = set(strategy_list) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    if len(set(strategy_list)) != len(strategy_list):
        raise ValueError("Each strategy may appear only once")

    rows: list[dict[str, object]] = []
    for seed in seed_list:
        gw1 = gameweeks.loc[gameweeks["gameweek"] == 1].copy()
        initial_players = gw1[
            ["player_id", "name", "position", "club", "price", "price_tenths"]
        ].drop_duplicates("player_id")
        initial_rng = np.random.default_rng(int(seed))
        initial_decision = random_valid_decision(
            initial_players, STARTING_BUDGET_TENTHS / 10.0, initial_rng
        )
        initial_outcomes = outcomes_for_gameweek(gameweeks, 1)
        initial_positions = dict(
            zip(initial_players["player_id"].astype(int), initial_players["position"])
        )
        initial_points, _ = score_lineup(
            initial_decision, initial_positions, initial_outcomes
        )
        initial_prices = initial_players.set_index("player_id")["price_tenths"].astype(int)
        initial_cost_tenths = int(initial_prices.loc[initial_decision.squad].sum())

        for strategy_index, strategy in enumerate(strategy_list):
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), strategy_index + 1]))
            squad = list(initial_decision.squad)
            purchase_prices = {pid: int(initial_prices.loc[pid]) for pid in squad}
            bank_tenths = STARTING_BUDGET_TENTHS - initial_cost_tenths
            free_transfers = 1
            cumulative_points = 0.0

            for gw in weeks:
                if gw == 1:
                    decision = initial_decision
                    n_transfers = 0
                    hit = 0
                    gross_points = initial_points
                    market_prices: dict[int, int] = {}
                    positions = initial_positions
                    free_transfers = free_transfer_cap
                else:
                    players = build_as_of_features(
                        gameweeks,
                        target_gameweek=gw,
                        current_squad=squad,
                        lookback=lookback,
                        max_stale_gameweeks=max_stale_gameweeks,
                    )
                    market_prices = {
                        int(row.player_id): int(round(float(row.price) * 10))
                        for row in players.itertuples(index=False)
                    }
                    current_sell_prices = {
                        pid: market_sell_price_tenths(
                            purchase_prices[pid], market_prices[pid]
                        ) / 10.0
                        for pid in squad
                    }
                    decision = make_decision(
                        strategy,
                        players,
                        squad,
                        budget=None,
                        free_transfers=free_transfers,
                        rng=rng,
                        bank_balance=bank_tenths / 10.0,
                        current_sell_prices=current_sell_prices,
                    )
                    n_transfers = len(set(squad) ^ set(decision.squad)) // 2
                    hit = HIT_PER_EXTRA_TRANSFER * max(0, n_transfers - free_transfers)
                    sold = set(squad) - set(decision.squad)
                    bought = set(decision.squad) - set(squad)
                    bank_tenths += sum(
                        int(round(current_sell_prices[pid] * 10)) for pid in sold
                    )
                    bank_tenths -= sum(market_prices[pid] for pid in bought)
                    purchase_prices = {
                        pid: (market_prices[pid] if pid in bought else purchase_prices[pid])
                        for pid in decision.squad
                    }
                    squad = list(decision.squad)
                    free_transfers = min(
                        free_transfer_cap,
                        max(0, free_transfers - n_transfers) + 1,
                    )
                    positions = dict(
                        zip(players["player_id"].astype(int), players["position"])
                    )
                    outcomes = outcomes_for_gameweek(gameweeks, gw)
                    gross_points, _ = score_lineup(decision, positions, outcomes)

                net_points = gross_points - hit
                cumulative_points += net_points
                squad_value_tenths = sum(
                    market_prices.get(pid, int(initial_prices.loc[pid]) if pid in initial_prices.index else 0)
                    for pid in squad
                )
                rows.append(
                    {
                        "season": season,
                        "seed": int(seed),
                        "strategy": strategy,
                        "gameweek": gw,
                        "gross_points": float(gross_points),
                        "transfer_hits": int(hit),
                        "net_points": float(net_points),
                        "cumulative_points": float(cumulative_points),
                        "transfers": int(n_transfers),
                        "free_transfers_next_gw": int(free_transfers),
                        "bank_tenths": int(bank_tenths),
                        "squad_value_tenths": int(squad_value_tenths),
                        "squad": ",".join(str(pid) for pid in sorted(squad)),
                    }
                )

    weekly = pd.DataFrame(rows)
    summary = (
        weekly.groupby(["season", "seed", "strategy"], as_index=False)
        .agg(
            total_net_points=("net_points", "sum"),
            total_gross_points=("gross_points", "sum"),
            total_transfer_hits=("transfer_hits", "sum"),
            total_transfers=("transfers", "sum"),
            mean_weekly_points=("net_points", "mean"),
            gameweeks=("gameweek", "count"),
        )
        .sort_values(["seed", "total_net_points"], ascending=[True, False])
        .reset_index(drop=True)
    )
    metadata = {
        "season": season,
        "strategies": list(strategy_list),
        "seeds": list(seed_list),
        "first_gameweek": weeks[0],
        "last_gameweek": weeks[-1],
        "lookback_gameweeks": lookback,
        "max_stale_gameweeks": max_stale_gameweeks,
        "starting_budget_tenths": STARTING_BUDGET_TENTHS,
        "free_transfer_cap": free_transfer_cap,
        "dataset_sha256": dataset_sha256,
        "caveats": [
            "Vaastav merged data does not contain official deadline availability snapshots.",
            "The latest pre-GW row proxies each player market price and candidate eligibility.",
            "Purchase prices and bank are simulated; historical manager ownership is not replayed.",
            "The xGI challenger is a simple expected-points proxy, not a trained points model.",
            "The Thompson challenger uses the repository's heuristic NB sampling, not thesis HDP/MCMC.",
        ],
    }
    return BacktestResult(weekly, summary, metadata)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 fingerprint of an input archive."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
