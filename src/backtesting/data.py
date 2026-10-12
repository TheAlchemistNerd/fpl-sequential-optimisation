"""Vaastav historical data loading and strictly as-of feature construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "element",
    "position",
    "team",
    "value",
    "minutes",
    "total_points",
    "GW",
}
POSITION_MAP = {
    "GK": "GK",
    "GKP": "GK",
    "GOALKEEPER": "GK",
    "DEF": "DEF",
    "DEFENDER": "DEF",
    "MID": "MID",
    "MIDFIELDER": "MID",
    "FWD": "FWD",
    "FORWARD": "FWD",
}


def load_vaastav_merged(path: str | Path) -> pd.DataFrame:
    """Load and aggregate a Vaastav ``gws/merged_gw.csv`` file.

    Fixture rows are collapsed to one row per player and gameweek. Points,
    minutes, and expected-goal metrics are summed so double gameweeks retain
    their full outcomes. Prices and identity metadata use the final fixture
    row in that gameweek.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Historical FPL CSV not found: {path}")
    raw = pd.read_csv(path)
    return aggregate_vaastav_gameweeks(raw)


def aggregate_vaastav_gameweeks(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize Vaastav fixture rows to a player-gameweek table."""
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"Vaastav data is missing required columns: {missing}")

    frame = raw.copy()
    for column in ("element", "GW", "value", "minutes", "total_points"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["element", "GW", "value", "minutes", "total_points"]].isna().any().any():
        raise ValueError("Player IDs, gameweeks, prices, minutes, and points must be numeric")

    frame["element"] = frame["element"].astype(int)
    frame["GW"] = frame["GW"].astype(int)
    frame["position"] = frame["position"].astype(str).str.upper().str.strip()
    frame["position"] = frame["position"].map(POSITION_MAP)
    if frame["position"].isna().any():
        unknown = sorted(raw.loc[frame["position"].isna(), "position"].astype(str).unique())
        raise ValueError(f"Unrecognized FPL positions: {unknown}")

    if "name" not in frame:
        frame["name"] = frame["element"].astype(str)
    for source, target in (
        ("expected_goals", "xg"),
        ("expected_assists", "xa"),
        ("clean_sheets", "clean_sheets"),
    ):
        if source not in frame:
            frame[source] = 0.0
        frame[source] = pd.to_numeric(frame[source], errors="coerce").fillna(0.0)

    frame["team"] = frame["team"].fillna("").astype(str)
    sort_columns = ["GW"]
    if "kickoff_time" in frame:
        sort_columns.append("kickoff_time")
    if "fixture" in frame:
        sort_columns.append("fixture")
    frame = frame.sort_values(sort_columns, kind="stable")

    grouped = (
        frame.groupby(["element", "GW"], as_index=False, sort=True)
        .agg(
            name=("name", "last"),
            position=("position", "last"),
            club=("team", "last"),
            price_tenths=("value", "last"),
            minutes=("minutes", "sum"),
            points=("total_points", "sum"),
            xg=("expected_goals", "sum"),
            xa=("expected_assists", "sum"),
            clean_sheets=("clean_sheets", "sum"),
        )
    )
    grouped["price"] = grouped["price_tenths"] / 10.0
    grouped["player_id"] = grouped["element"].astype(int)
    grouped["gameweek"] = grouped["GW"].astype(int)
    grouped["minutes"] = grouped["minutes"].astype(float)
    grouped["points"] = grouped["points"].astype(float)
    grouped["xg"] = grouped["xg"].astype(float)
    grouped["xa"] = grouped["xa"].astype(float)
    grouped["clean_sheets"] = grouped["clean_sheets"].astype(float)
    return grouped[
        [
            "player_id",
            "gameweek",
            "name",
            "position",
            "club",
            "price_tenths",
            "price",
            "minutes",
            "points",
            "xg",
            "xa",
            "clean_sheets",
        ]
    ].sort_values(["gameweek", "player_id"], kind="stable").reset_index(drop=True)


def build_as_of_features(
    gameweeks: pd.DataFrame,
    target_gameweek: int,
    current_squad: list[int] | None = None,
    *,
    lookback: int = 5,
    max_stale_gameweeks: int = 5,
) -> pd.DataFrame:
    """Build a player table using only records strictly before the target GW.

    The candidate pool is approximated from players observed within the last
    ``max_stale_gameweeks``. A manager's current players are retained even if
    their latest historical row is older. The archive has no per-deadline
    availability snapshots, so this rule is an explicit proxy rather than an
    exact reconstruction of the official player pool.
    """
    if lookback < 1 or max_stale_gameweeks < 1:
        raise ValueError("lookback and max_stale_gameweeks must be positive")
    if "gameweek" not in gameweeks or "player_id" not in gameweeks:
        raise ValueError("Expected normalized player-gameweek data")

    history = gameweeks.loc[gameweeks["gameweek"] < target_gameweek].copy()
    if history.empty:
        raise ValueError(f"No historical observations precede GW{target_gameweek}")

    history = history.sort_values(["gameweek", "player_id"], kind="stable")
    latest = history.groupby("player_id", as_index=False, sort=False).tail(1).set_index("player_id")
    current_ids = {int(player_id) for player_id in (current_squad or [])}
    fresh_ids = set(
        latest.index[latest["gameweek"] >= target_gameweek - max_stale_gameweeks]
    )
    eligible_ids = fresh_ids | current_ids
    missing_current = current_ids - set(latest.index)
    if missing_current:
        raise ValueError(f"Current squad has no prior data for players: {sorted(missing_current)}")

    window_start = max(int(gameweeks["gameweek"].min()), target_gameweek - lookback)
    window = history.loc[history["gameweek"] >= window_start]
    window_stats = window.groupby("player_id", sort=False).agg(
        xg=("xg", "sum"),
        xa=("xa", "sum"),
        minutes=("minutes", "sum"),
        clean_sheets=("clean_sheets", "sum"),
    )
    window_rows = {
        int(pid): rows.set_index("gameweek")
        for pid, rows in window.groupby("player_id", sort=False)
    }
    first_seen_by_player = history.groupby("player_id", sort=False)["gameweek"].min()

    records: list[dict[str, object]] = []
    for player_id in sorted(eligible_ids):
        latest_row = latest.loc[player_id]
        first_seen = int(first_seen_by_player.loc[player_id])
        period_start = max(window_start, first_seen)
        n_periods = max(target_gameweek - period_start, 1)
        prior_rows = window_rows.get(player_id)
        stats = window_stats.loc[player_id] if player_id in window_stats.index else None
        weekly_points = np.asarray(
            [
                float(prior_rows.loc[gw, "points"])
                if prior_rows is not None and gw in prior_rows.index
                else 0.0
                for gw in range(period_start, target_gameweek)
            ],
            dtype=float,
        )
        weekly_minutes = np.asarray(
            [
                float(prior_rows.loc[gw, "minutes"])
                if prior_rows is not None and gw in prior_rows.index
                else 0.0
                for gw in range(period_start, target_gameweek)
            ],
            dtype=float,
        )
        xg_total = float(stats["xg"]) if stats is not None else 0.0
        xa_total = float(stats["xa"]) if stats is not None else 0.0
        minutes_total = float(stats["minutes"]) if stats is not None else 0.0
        clean_sheets = float(stats["clean_sheets"]) if stats is not None else 0.0
        position = str(latest_row["position"])
        appearance_points = float(np.mean(weekly_minutes > 0)) + float(
            np.mean(weekly_minutes >= 60)
        )
        goal_points = {"GK": 6.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}[position]
        clean_sheet_points = {"GK": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}[position]
        expected_minutes = float(weekly_minutes.mean()) if weekly_minutes.size else 0.0
        xgi_points = (
            appearance_points
            + (xg_total * goal_points + xa_total * 3.0) / n_periods
            + clean_sheets * clean_sheet_points / n_periods
        )
        records.append(
            {
                "player_id": player_id,
                "name": str(latest_row["name"]),
                "position": position,
                "club": str(latest_row["club"]),
                "price": float(latest_row["price"]),
                "last_seen_gameweek": int(latest_row["gameweek"]),
                "history_gameweeks": n_periods,
                "form_points": float(weekly_points.mean()) if weekly_points.size else 0.0,
                "form_std": float(weekly_points.std(ddof=1)) if weekly_points.size > 1 else 0.0,
                "xg_per_90": 90.0 * xg_total / minutes_total if minutes_total > 0 else 0.0,
                "xgi_points": float(xgi_points),
                "appearance_probability": float(np.mean(weekly_minutes > 0)),
                "expected_minutes": expected_minutes,
            }
        )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError(f"No eligible players found before GW{target_gameweek}")
    return result.sort_values("player_id", kind="stable").reset_index(drop=True)


def outcomes_for_gameweek(gameweeks: pd.DataFrame, gameweek: int) -> dict[int, dict[str, float]]:
    """Return realized points and minutes by player for one target GW."""
    rows = gameweeks.loc[gameweeks["gameweek"] == gameweek]
    return {
        int(row.player_id): {"points": float(row.points), "minutes": float(row.minutes)}
        for row in rows.itertuples(index=False)
    }
