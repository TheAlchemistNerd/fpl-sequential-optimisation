from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _small_history(weeks: int = 3) -> pd.DataFrame:
    positions = ["GK"] * 4 + ["DEF"] * 8 + ["MID"] * 10 + ["FWD"] * 8
    rows = []
    for gw in range(1, weeks + 1):
        for idx, position in enumerate(positions):
            rows.append(
                {
                    "player_id": idx + 1,
                    "gameweek": gw,
                    "name": f"Player {idx + 1}",
                    "position": position,
                    "club": f"Club {idx + 1}",
                    "price_tenths": 45 if position in {"GK", "DEF"} else 50,
                    "price": 4.5 if position in {"GK", "DEF"} else 5.0,
                    "minutes": 90.0,
                    "points": float((idx % 8) + gw),
                    "xg": 0.05 * (idx % 4),
                    "xa": 0.03 * (idx % 3),
                    "clean_sheets": float(position in {"GK", "DEF"} and gw % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_vaastav_combines_double_gameweek_rows():
    from src.backtesting.data import aggregate_vaastav_gameweeks

    raw = pd.DataFrame(
        [
            {"element": 1, "position": "GKP", "team": "A", "value": 45,
             "minutes": 90, "total_points": 6, "expected_goals": 0.1,
             "expected_assists": 0.0, "clean_sheets": 1, "GW": 4, "fixture": 1},
            {"element": 1, "position": "GKP", "team": "A", "value": 46,
             "minutes": 90, "total_points": 8, "expected_goals": 0.0,
             "expected_assists": 0.1, "clean_sheets": 0, "GW": 4, "fixture": 2},
        ]
    )
    result = aggregate_vaastav_gameweeks(raw).iloc[0]
    assert result["points"] == 14
    assert result["minutes"] == 180
    assert result["xg"] == pytest.approx(0.1)
    assert result["price_tenths"] == 46


def test_as_of_features_do_not_depend_on_target_gameweek_outcomes():
    from src.backtesting.data import build_as_of_features

    history = _small_history()
    altered = history.copy()
    altered.loc[altered["gameweek"] == 3, ["points", "minutes", "xg", "xa"]] = 999
    before = build_as_of_features(history, 3)
    after = build_as_of_features(altered, 3)
    pd.testing.assert_frame_equal(before, after)


def test_sell_price_credits_half_of_profit_in_tenths():
    from src.backtesting.engine import market_sell_price_tenths

    assert market_sell_price_tenths(70, 73) == 71
    assert market_sell_price_tenths(70, 69) == 69


def test_scoring_applies_captain_and_legal_outfield_substitution():
    from src.backtesting.engine import score_lineup
    from src.backtesting.strategies import Decision

    positions = {
        1: "GK", 12: "GK", 2: "DEF", 3: "DEF", 4: "DEF",
        13: "DEF", 14: "DEF", 5: "MID", 6: "MID", 7: "MID",
        8: "MID", 15: "MID", 9: "FWD", 10: "FWD", 11: "FWD",
    }
    decision = Decision(
        squad=list(positions),
        starting_xi=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        captain=9,
        vice_captain=10,
        bench_order=[15, 13, 14, 12],
    )
    outcomes = {pid: {"points": 2.0, "minutes": 90.0} for pid in positions}
    outcomes[4] = {"points": 0.0, "minutes": 0.0}
    outcomes[13] = {"points": 7.0, "minutes": 20.0}
    # A playing midfielder cannot replace the missed defender because the
    # resulting XI would fall below three defenders, so the next defender enters.
    outcomes[15] = {"points": 5.0, "minutes": 90.0}
    points, final_xi = score_lineup(decision, positions, outcomes)
    assert 4 not in final_xi
    assert 13 in final_xi
    assert 15 not in final_xi
    assert len(final_xi) == 11
    assert points == pytest.approx(2 * 10 + 7 + 2)


def test_scoring_substitutes_the_reserve_goalkeeper():
    from src.backtesting.engine import score_lineup
    from src.backtesting.strategies import Decision

    positions = {1: "GK", 12: "GK"}
    positions.update({pid: "DEF" for pid in range(2, 7)})
    positions.update({pid: "MID" for pid in range(7, 12)})
    positions.update({pid: "FWD" for pid in range(13, 16)})
    decision = Decision(
        squad=list(positions),
        starting_xi=[1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15],
        captain=13,
        vice_captain=14,
        bench_order=[12, 5, 6, 11],
    )
    outcomes = {pid: {"points": 2.0, "minutes": 90.0} for pid in positions}
    outcomes[1] = {"points": 0.0, "minutes": 0.0}
    outcomes[12] = {"points": 4.0, "minutes": 90.0}
    points, final_xi = score_lineup(decision, positions, outcomes)
    assert 1 not in final_xi and 12 in final_xi
    assert len(final_xi) == 11
    assert points == pytest.approx(10 * 2 + 4 + 2)


def test_random_valid_is_legal_and_seeded():
    import numpy as np

    from src.backtesting.strategies import random_valid_decision

    players = _small_history(weeks=1).drop_duplicates("player_id")
    first = random_valid_decision(players, 100.0, np.random.default_rng(17))
    second = random_valid_decision(players, 100.0, np.random.default_rng(17))
    assert first == second
    assert len(first.squad) == 15
    assert len(first.starting_xi) == 11
    assert len(set(first.squad)) == 15
    assert first.captain in first.starting_xi
    assert first.vice_captain in first.starting_xi
    positions = players.set_index("player_id")["position"]
    assert sum(positions.loc[first.starting_xi] == "GK") == 1


def test_hold_policy_runs_chronological_backtest():
    from src.backtesting.engine import run_backtest

    result = run_backtest(_small_history(weeks=2), strategies=("hold",), seeds=(17,))
    assert result.gameweeks["gameweek"].tolist() == [1, 2]
    assert result.summary.iloc[0]["gameweeks"] == 2
    assert result.gameweeks.iloc[1]["transfers"] == 0
    assert result.gameweeks.iloc[1]["free_transfers_next_gw"] == 2
    assert result.metadata["dataset_sha256"] is None


def test_model_score_backtest_uses_saved_predictions_from_common_evaluation_start():
    from src.backtesting.engine import run_backtest

    history = _small_history(weeks=3)
    ids = sorted(history["player_id"].unique())
    predictions = pd.DataFrame(
        [
            {
                "gameweek": gw,
                "player_id": player_id,
                "hdp_gmm": float(player_id),
                "hdp_hmm": float(30 - player_id),
            }
            for gw in (2, 3)
            for player_id in ids
        ]
    )
    result = run_backtest(
        history,
        strategies=("hdp_gmm_milp",),
        seeds=(17,),
        prediction_ledger=predictions,
        evaluation_start_gameweek=2,
    )
    assert result.summary.iloc[0]["evaluation_gameweeks"] == 2
    assert result.gameweeks["is_evaluation_gameweek"].tolist() == [False, True, True]
    assert result.gameweeks.iloc[1]["transfers"] <= 2
