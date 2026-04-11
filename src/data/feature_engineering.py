"""
Feature Engineering Pipeline — Chapter 3, Section 3.4.

Transforms raw data from all sources into a unified feature vector for each
player-gameweek observation (p, t).

Feature categories:
    - Core performance (per-90): goals, assists, clean_sheets, saves, bonus, bps
    - Advanced performance (per-90): xG, xA, npxG, shots, key_passes, xGChain, xGBuildup
    - ICT (per-90): influence, creativity, threat
    - Form: rolling windows (3, 5, 10 GW), xGForm(5), xAForm(5)
    - Fixture: FDR, RollingFDR(3), RollingFDR(5), is_home, rest_days
    - Contextual: Injured, Sentiment, SentimentVol, ownership_percent, price, price_change
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def per90(series: pd.Series, minutes: pd.Series) -> pd.Series:
    """S^{/90} = S / max(minutes, 1) * 90

    Canonical per-90 normalisation — all modules should import from here.
    """
    return series / minutes.clip(lower=1) * 90.0


# Backward-compatible alias (private name kept for internal use)
_per90 = per90


# ---------------------------------------------------------------------------
# Core performance features  (§3.4.1)
# ---------------------------------------------------------------------------


def build_core_features(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-90 rates for standard FPL output metrics.

    Expected columns in history_df (cumulative season stats):
        minutes, goals_scored, assists, clean_sheets, saves,
        bonus, bps, yellow_cards, red_cards

    Returns a DataFrame with per-90 columns.
    """
    df = history_df.copy()
    mins = df["minutes"].astype(float)

    core_cols = [
        "goals_scored", "assists", "clean_sheets", "saves",
        "bonus", "bps", "yellow_cards", "red_cards",
    ]
    for col in core_cols:
        if col in df.columns:
            df[f"{col}_per_90"] = _per90(df[col].astype(float), mins)

    logger.info("Core features built: %d columns", len(core_cols))
    return df


# ---------------------------------------------------------------------------
# Advanced performance features  (§3.4.2)
# ---------------------------------------------------------------------------


def build_advanced_features(
    history_df: pd.DataFrame,
    understat_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute per-90 rates for advanced metrics (from Understat if available,
    otherwise from FPL expected_goals / expected_assists).

    Returns DataFrame with xG_per_90, xA_per_90, npxG_per_90, shots_per_90,
    key_passes_per_90, xGChain_per_90, xGBuildup_per_90.
    """
    df = history_df.copy()
    mins = df["minutes"].astype(float)

    # FPL-native expected metrics
    fpl_adv_map = {
        "expected_goals": "xG",
        "expected_assists": "xA",
        "expected_goal_involvements": "xGI",
        "expected_goals_conceded": "xGC",
    }
    for fpl_col, short in fpl_adv_map.items():
        if fpl_col in df.columns:
            df[f"{short}_per_90"] = _per90(df[fpl_col].astype(float), mins)

    # If Understat data is merged in, override with higher-quality xG
    if understat_df is not None and not understat_df.empty:
        if "xG" in understat_df.columns:
            df["xG_per_90"] = _per90(understat_df["xG"].astype(float), mins)
        if "xA" in understat_df.columns:
            df["xA_per_90"] = _per90(understat_df["xA"].astype(float), mins)
        if "npxG" in understat_df.columns:
            df["npxG_per_90"] = _per90(understat_df["npxG"].astype(float), mins)
        if "shots" in understat_df.columns:
            df["shots_per_90"] = _per90(understat_df["shots"].astype(float), mins)
        if "keys" in understat_df.columns:
            df["key_passes_per_90"] = _per90(understat_df["keys"].astype(float), mins)

    logger.info("Advanced features built")
    return df


# ---------------------------------------------------------------------------
# ICT features  (§3.4.3)
# ---------------------------------------------------------------------------


def build_ict_features(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    ICT components per 90: influence, creativity, threat.
    """
    df = history_df.copy()
    mins = df["minutes"].astype(float)

    for col in ["influence", "creativity", "threat"]:
        if col in df.columns:
            df[f"{col}_per_90"] = _per90(df[col].astype(float), mins)

    return df


# ---------------------------------------------------------------------------
# Form features  (§3.4.4)
# ---------------------------------------------------------------------------


def build_form_features(
    player_gw_df: pd.DataFrame,
    windows: tuple[int, ...] = (3, 5, 10),
) -> pd.DataFrame:
    """
    Compute rolling-average form over multiple windows.

    Expected columns: player_id, gameweek, total_points,
                      expected_goals (or xG_per_90), expected_assists (or xA_per_90)

    Produces:
        Form_3, Form_5, Form_10
        xGForm_5, xAForm_5
    """
    df = player_gw_df.copy()
    df = df.sort_values(["player_id", "gameweek"])

    # Points-based form
    for w in windows:
        df[f"Form_{w}"] = df.groupby("player_id")["total_points"].transform(
            lambda s: s.rolling(window=w, min_periods=1).mean()
        )

    # xG / xA form (5-game window)
    xg_col = "xG_per_90" if "xG_per_90" in df.columns else "expected_goals"
    xa_col = "xA_per_90" if "xA_per_90" in df.columns else "expected_assists"

    if xg_col in df.columns:
        df["xGForm_5"] = df.groupby("player_id")[xg_col].transform(
            lambda s: s.rolling(window=5, min_periods=1).mean()
        )
    if xa_col in df.columns:
        df["xAForm_5"] = df.groupby("player_id")[xa_col].transform(
            lambda s: s.rolling(window=5, min_periods=1).mean()
        )

    logger.info("Form features built for windows %s", windows)
    return df


# ---------------------------------------------------------------------------
# Fixture and opponent features  (§3.4.5)
# ---------------------------------------------------------------------------


def build_fixture_features(
    history_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fixture difficulty features:
        FDR, RollingFDR_3, RollingFDR_5, is_home, rest_days
    """
    df = history_df.copy()

    # Build a team-strength lookup
    team_lookup = {}
    for _, row in teams_df.iterrows():
        tid = int(row["id"])
        team_lookup[tid] = {
            "strength_defence_home": row.get("strength_defence_home", 3),
            "strength_defence_away": row.get("strength_defence_away", 3),
        }

    # FDR for each row
    fdr_values = []
    for _, row in df.iterrows():
        opp_team = int(row.get("opponent_team", 0))
        was_home = row.get("was_home", True)
        if opp_team not in team_lookup:
            fdr_values.append(3)  # default neutral
            continue
        strengths = team_lookup[opp_team]
        if was_home:
            fdr_values.append(strengths["strength_defence_away"])
        else:
            fdr_values.append(strengths["strength_defence_home"])

    df["FDR"] = fdr_values

    # Rolling FDR
    df = df.sort_values(["player_id", "gameweek"])
    for w in [3, 5]:
        df[f"RollingFDR_{w}"] = df.groupby("player_id")["FDR"].transform(
            lambda s: s.rolling(window=w, min_periods=1).mean()
        )

    # is_home
    if "was_home" in df.columns:
        df["is_home"] = df["was_home"].astype(int)

    # rest_days (from kickoff_time)
    if "kickoff_time" in df.columns:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
        df["rest_days"] = df.groupby("player_id")["kickoff_time"].diff().dt.total_seconds() / 86400.0
        df["rest_days"] = df["rest_days"].fillna(7.0)  # default one week

    logger.info("Fixture features built")
    return df


# ---------------------------------------------------------------------------
# Contextual features  (§3.4.6)
# ---------------------------------------------------------------------------


def build_contextual_features(
    history_df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add contextual features:
        Injured, Sentiment, SentimentVol, ownership_percent,
        price, price_change
    """
    df = history_df.copy()

    # Injured (from FPL status field: i=injured, d=doubtful, s=suspended)
    if "status" in df.columns:
        df["Injured"] = df["status"].isin(["i", "d", "s", "u"]).astype(int)
    else:
        df["Injured"] = 0

    # Merge sentiment if available
    if sentiment_df is not None and not sentiment_df.empty:
        sentiment_cols = {"Sentiment", "SentimentVol"}
        available = sentiment_cols & set(sentiment_df.columns)
        merge_cols = ["player_name", "gameweek"]
        if "player_name" not in df.columns and "player_id" in df.columns:
            # Cannot merge without player_name — skip
            logger.warning("Cannot merge sentiment: missing player_name column")
        else:
            df = df.merge(
                sentiment_df[merge_cols + list(available)],
                on=merge_cols,
                how="left",
            )
            if "Sentiment" in df.columns:
                df["Sentiment"] = df["Sentiment"].fillna(0.0)
            if "SentimentVol" in df.columns:
                df["SentimentVol"] = df["SentimentVol"].fillna(0.0)

    # Ownership percentage
    if "selected" in df.columns and "total_players" not in df.columns:
        # selected is raw count; compute percentage if total_players known
        # For now, store raw selected as proxy
        df["ownership_percent"] = df.get("selected", pd.Series(0, index=df.index))

    # Price and price change
    if "value" in df.columns:
        df["price"] = df["value"] / 10.0  # convert from tenths to £m
        df["price_change"] = df.groupby("player_id")["value"].diff().fillna(0)

    logger.info("Contextual features built")
    return df


# ---------------------------------------------------------------------------
# Full feature vector assembly  (§3.4.7)
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "core": [
        "goals_scored_per_90", "assists_per_90", "clean_sheets_per_90",
        "saves_per_90", "bonus_per_90", "bps_per_90",
        "yellow_cards_per_90", "red_cards_per_90",
    ],
    "advanced": [
        "xG_per_90", "xA_per_90", "npxG_per_90",
        "shots_per_90", "key_passes_per_90",
        "xGChain_per_90", "xGBuildup_per_90",
    ],
    "ict": [
        "influence_per_90", "creativity_per_90", "threat_per_90",
    ],
    "form": [
        "Form_3", "Form_5", "Form_10", "xGForm_5", "xAForm_5",
    ],
    "fixture": [
        "FDR", "RollingFDR_3", "RollingFDR_5", "is_home", "rest_days",
    ],
    "contextual": [
        "Injured", "Sentiment", "SentimentVol",
        "ownership_percent", "price", "price_change",
    ],
}


def get_feature_names(group: str | None = None) -> list[str]:
    """Return the list of feature column names for *group* or all groups."""
    if group is not None:
        return FEATURE_GROUPS.get(group, [])
    names: list[str] = []
    for cols in FEATURE_GROUPS.values():
        names.extend(cols)
    return names


def assemble_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract the final feature matrix from the enriched DataFrame.

    Returns (X, feature_names) where X contains only the numeric feature columns
    that exist in the DataFrame.
    """
    all_features = get_feature_names()
    available = [c for c in all_features if c in df.columns]
    X = df[available].copy()
    logger.info(
        "Feature matrix assembled: %d rows x %d columns",
        len(X), len(available),
    )
    return X, available
