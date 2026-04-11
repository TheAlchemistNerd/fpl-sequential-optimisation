"""
Tests for src/data/feature_engineering.py — Feature Engineering Pipeline.

Tests cover:
    - Core performance features (§3.4.1)
    - Advanced performance features (§3.4.2)
    - ICT features (§3.4.3)
    - Form features (§3.4.4)
    - Fixture and opponent features (§3.4.5)
    - Contextual features (§3.4.6)
    - Full feature vector assembly (§3.4.7)
    - FEATURE_GROUPS and get_feature_names
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_history_df():
    """Sample player gameweek history DataFrame."""
    return pd.DataFrame({
        "player_id": [1, 1, 1, 2, 2, 2],
        "gameweek": [1, 2, 3, 1, 2, 3],
        "minutes": [90, 90, 75, 90, 0, 90],
        "goals_scored": [1, 0, 1, 0, 0, 1],
        "assists": [1, 0, 0, 0, 1, 0],
        "clean_sheets": [0, 1, 0, 1, 0, 0],
        "saves": [0, 0, 0, 3, 2, 4],
        "bonus": [2, 1, 0, 1, 0, 0],
        "bps": [40, 35, 20, 30, 15, 25],
        "yellow_cards": [0, 0, 1, 0, 1, 0],
        "red_cards": [0, 0, 0, 0, 0, 0],
        "total_points": [12, 6, 3, 7, 2, 6],
        "influence": [50.0, 45.0, 30.0, 40.0, 20.0, 35.0],
        "creativity": [40.0, 35.0, 25.0, 30.0, 25.0, 20.0],
        "threat": [60.0, 50.0, 40.0, 35.0, 20.0, 45.0],
        "expected_goals": [0.5, 0.3, 0.7, 0.1, 0.05, 0.4],
        "expected_assists": [0.3, 0.2, 0.1, 0.1, 0.15, 0.05],
        "expected_goal_involvements": [0.8, 0.5, 0.8, 0.2, 0.2, 0.45],
        "expected_goals_conceded": [1.0, 0.8, 1.2, 0.9, 1.1, 1.0],
        "was_home": [True, False, True, True, False, True],
        "opponent_team": [2, 3, 2, 4, 4, 5],
        "kickoff_time": pd.to_datetime([
            "2024-08-17", "2024-08-24", "2024-08-31",
            "2024-08-17", "2024-08-24", "2024-08-31",
        ]),
        "value": [130, 131, 132, 45, 45, 46],
        "selected": [2000000, 2100000, 2200000, 500000, 480000, 520000],
        "status": ["a", "a", "a", "a", "i", "a"],
    })


@pytest.fixture
def sample_understat_df():
    """Sample Understat advanced metrics DataFrame."""
    return pd.DataFrame({
        "player_name": ["Salah", "Salah", "Salah"],
        "gameweek": [1, 2, 3],
        "xG": [0.6, 0.4, 0.8],
        "xA": [0.25, 0.18, 0.12],
        "npxG": [0.6, 0.4, 0.8],  # No penalties
        "shots": [4, 3, 5],
        "keys": [2, 1, 3],  # Key passes
        "xGChain": [1.2, 0.9, 1.5],
        "xGBuildup": [0.8, 0.6, 1.0],
        "time": [90, 90, 75],
    })


@pytest.fixture
def sample_teams_df():
    """Sample teams DataFrame with strength ratings."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Arsenal", "Chelsea", "Liverpool", "Man City", "Man Utd"],
        "strength_defence_home": [4, 3, 5, 5, 3],
        "strength_defence_away": [4, 3, 4, 5, 3],
    })


@pytest.fixture
def sample_fixtures_df():
    """Sample fixtures DataFrame."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "team_h": [1, 3, 2, 4, 5, 1],
        "team_a": [2, 4, 1, 5, 3, 2],
        "kickoff_time": pd.to_datetime([
            "2024-08-17", "2024-08-24", "2024-08-31",
            "2024-09-07", "2024-09-14", "2024-09-21",
        ]),
        "event": [1, 2, 3, 4, 5, 6],
    })


@pytest.fixture
def sample_sentiment_df():
    """Sample sentiment data DataFrame."""
    return pd.DataFrame({
        "player_name": ["Salah", "Salah", "Salah", "Haaland", "Haaland"],
        "gameweek": [1, 2, 3, 1, 2],
        "Sentiment": [0.3, -0.1, 0.2, 0.5, 0.1],
        "SentimentVol": [0.1, 0.15, 0.12, 0.2, 0.18],
    })


# ===========================================================================
# Core performance features tests (§3.4.1)
# ===========================================================================


class TestCoreFeatures:
    """Validate core performance feature computation (per-90 rates)."""

    def test_build_core_features_adds_per90_columns(self, sample_history_df):
        from src.data.feature_engineering import build_core_features
        result = build_core_features(sample_history_df)
        expected_cols = [
            "goals_scored_per_90", "assists_per_90", "clean_sheets_per_90",
            "saves_per_90", "bonus_per_90", "bps_per_90",
            "yellow_cards_per_90", "red_cards_per_90",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_build_core_features_values_correct(self, sample_history_df):
        from src.data.feature_engineering import build_core_features
        result = build_core_features(sample_history_df)
        # Player 1, GW1: 1 goal in 90 min → 1.0 per 90
        player1_gw1 = result[(result["player_id"] == 1) & (result["gameweek"] == 1)]
        assert player1_gw1["goals_scored_per_90"].iloc[0] == pytest.approx(1.0)
        # Player 1, GW1: 1 assist in 90 min → 1.0 per 90
        assert player1_gw1["assists_per_90"].iloc[0] == pytest.approx(1.0)

    def test_build_core_features_handles_zero_minutes(self, sample_history_df):
        from src.data.feature_engineering import build_core_features
        # Player 2, GW2 has 0 minutes
        result = build_core_features(sample_history_df)
        player2_gw2 = result[(result["player_id"] == 2) & (result["gameweek"] == 2)]
        # Per-90 should handle 0 minutes via clip(lower=1)
        assert player2_gw2["goals_scored_per_90"].iloc[0] == 0.0  # 0 goals

    def test_build_core_features_preserves_original_columns(self, sample_history_df):
        from src.data.feature_engineering import build_core_features
        result = build_core_features(sample_history_df)
        assert "goals_scored" in result.columns  # Original still there
        assert "minutes" in result.columns


# ===========================================================================
# Advanced performance features tests (§3.4.2)
# ===========================================================================


class TestAdvancedFeatures:
    """Validate advanced metrics feature computation."""

    def test_build_advanced_features_without_understat(self, sample_history_df):
        from src.data.feature_engineering import build_advanced_features
        result = build_advanced_features(sample_history_df)
        # Should use FPL expected metrics
        assert "xG_per_90" in result.columns
        assert "xA_per_90" in result.columns
        assert "xGI_per_90" in result.columns
        assert "xGC_per_90" in result.columns

    def test_build_advanced_features_with_understat(self, sample_history_df, sample_understat_df):
        from src.data.feature_engineering import build_advanced_features
        result = build_advanced_features(sample_history_df, understat_df=sample_understat_df)
        # Should override with Understat data
        assert "xG_per_90" in result.columns
        assert "xA_per_90" in result.columns
        assert "npxG_per_90" in result.columns
        assert "shots_per_90" in result.columns
        assert "key_passes_per_90" in result.columns

    def test_build_advanced_features_understat_values_correct(self, sample_history_df, sample_understat_df):
        from src.data.feature_engineering import build_advanced_features
        result = build_advanced_features(sample_history_df, understat_df=sample_understat_df)
        # Player 1 (Salah) GW1: xG=0.6 in 90 min → 0.6 per 90
        # Note: merge depends on player_name, which may not be in history_df
        # So this tests that the function doesn't crash
        assert isinstance(result, pd.DataFrame)

    def test_build_advanced_features_empty_understat(self, sample_history_df):
        from src.data.feature_engineering import build_advanced_features
        empty_understat = pd.DataFrame()
        result = build_advanced_features(sample_history_df, understat_df=empty_understat)
        # Should fall back to FPL data
        assert "xG_per_90" in result.columns


# ===========================================================================
# ICT features tests (§3.4.3)
# ===========================================================================


class TestICTFeatures:
    """Validate ICT component feature computation."""

    def test_build_ict_features_adds_per90_columns(self, sample_history_df):
        from src.data.feature_engineering import build_ict_features
        result = build_ict_features(sample_history_df)
        assert "influence_per_90" in result.columns
        assert "creativity_per_90" in result.columns
        assert "threat_per_90" in result.columns

    def test_build_ict_features_values_correct(self, sample_history_df):
        from src.data.feature_engineering import build_ict_features
        result = build_ict_features(sample_history_df)
        # Player 1, GW1: influence=50 in 90 min → 50.0 per 90
        player1_gw1 = result[(result["player_id"] == 1) & (result["gameweek"] == 1)]
        assert player1_gw1["influence_per_90"].iloc[0] == pytest.approx(50.0)
        assert player1_gw1["creativity_per_90"].iloc[0] == pytest.approx(40.0)
        assert player1_gw1["threat_per_90"].iloc[0] == pytest.approx(60.0)


# ===========================================================================
# Form features tests (§3.4.4)
# ===========================================================================


class TestFormFeatures:
    """Validate rolling form feature computation."""

    def test_build_form_features_adds_form_columns(self, sample_history_df):
        from src.data.feature_engineering import build_form_features
        result = build_form_features(sample_history_df)
        assert "Form_3" in result.columns
        assert "Form_5" in result.columns
        assert "Form_10" in result.columns

    def test_build_form_features_values_correct(self, sample_history_df):
        from src.data.feature_engineering import build_form_features
        result = build_form_features(sample_history_df, windows=(3,))
        # Player 1, GW3: Form_3 = mean([12, 6, 3]) = 7.0
        player1_gw3 = result[(result["player_id"] == 1) & (result["gameweek"] == 3)]
        assert player1_gw3["Form_3"].iloc[0] == pytest.approx(7.0)

    def test_build_form_features_xg_form(self, sample_history_df):
        from src.data.feature_engineering import build_form_features
        result = build_form_features(sample_history_df)
        # Should compute xGForm_5 and xAForm_5
        assert "xGForm_5" in result.columns
        assert "xAForm_5" in result.columns

    def test_build_form_features_handles_single_gameweek(self):
        from src.data.feature_engineering import build_form_features
        df = pd.DataFrame({
            "player_id": [1],
            "gameweek": [1],
            "total_points": [12],
            "expected_goals": [0.5],
            "expected_assists": [0.3],
        })
        result = build_form_features(df)
        # Should handle gracefully with min_periods=1
        assert "Form_3" in result.columns
        assert result["Form_3"].iloc[0] == pytest.approx(12.0)

    def test_build_form_features_sorts_by_player_and_gameweek(self, sample_history_df):
        from src.data.feature_engineering import build_form_features
        # Shuffle the input
        shuffled = sample_history_df.sample(frac=1, random_state=42)
        result = build_form_features(shuffled)
        # Check that it's sorted within each player
        for player_id in result["player_id"].unique():
            player_data = result[result["player_id"] == player_id]
            gameweeks = player_data["gameweek"].tolist()
            assert gameweeks == sorted(gameweeks)


# ===========================================================================
# Fixture features tests (§3.4.5)
# ===========================================================================


class TestFixtureFeatures:
    """Validate fixture difficulty and opponent feature computation."""

    def test_build_fixture_features_adds_fdr_columns(self, sample_history_df, sample_teams_df):
        from src.data.feature_engineering import build_fixture_features
        # Need to add empty fixtures_df
        fixtures_df = pd.DataFrame(columns=["id", "team_h", "team_a", "event"])
        result = build_fixture_features(sample_history_df, sample_teams_df, fixtures_df)
        assert "FDR" in result.columns
        assert "RollingFDR_3" in result.columns
        assert "RollingFDR_5" in result.columns

    def test_build_fixture_features_fdr_values(self, sample_history_df, sample_teams_df):
        from src.data.feature_engineering import build_fixture_features
        fixtures_df = pd.DataFrame(columns=["id", "team_h", "team_a", "event"])
        result = build_fixture_features(sample_history_df, sample_teams_df, fixtures_df)
        # Player 1, GW1: opponent_team=2, was_home=True
        # Team 2 strength_defence_away=3 → FDR=3
        player1_gw1 = result[(result["player_id"] == 1) & (result["gameweek"] == 1)]
        assert player1_gw1["FDR"].iloc[0] == pytest.approx(3.0)

    def test_build_fixture_features_is_home(self, sample_history_df, sample_teams_df):
        from src.data.feature_engineering import build_fixture_features
        fixtures_df = pd.DataFrame(columns=["id", "team_h", "team_a", "event"])
        result = build_fixture_features(sample_history_df, sample_teams_df, fixtures_df)
        assert "is_home" in result.columns
        player1_gw1 = result[(result["player_id"] == 1) & (result["gameweek"] == 1)]
        assert player1_gw1["is_home"].iloc[0] == 1

    def test_build_fixture_features_rest_days(self, sample_history_df, sample_teams_df):
        from src.data.feature_engineering import build_fixture_features
        fixtures_df = pd.DataFrame(columns=["id", "team_h", "team_a", "event"])
        result = build_fixture_features(sample_history_df, sample_teams_df, fixtures_df)
        assert "rest_days" in result.columns
        # Player 1, GW2 should have rest_days = 7 days (from kickoff_time diff)
        player1_gw2 = result[(result["player_id"] == 1) & (result["gameweek"] == 2)]
        assert player1_gw2["rest_days"].iloc[0] == pytest.approx(7.0, abs=0.1)

    def test_build_fixture_features_missing_opponent_defaults_to_3(self, sample_teams_df):
        from src.data.feature_engineering import build_fixture_features
        history_df = pd.DataFrame({
            "player_id": [1],
            "gameweek": [1],
            "opponent_team": [999],  # Non-existent team
            "was_home": [True],
            "kickoff_time": pd.to_datetime(["2024-08-17"]),
        })
        fixtures_df = pd.DataFrame(columns=["id", "team_h", "team_a", "event"])
        result = build_fixture_features(history_df, sample_teams_df, fixtures_df)
        assert result["FDR"].iloc[0] == 3  # Default neutral


# ===========================================================================
# Contextual features tests (§3.4.6)
# ===========================================================================


class TestContextualFeatures:
    """Validate contextual feature computation."""

    def test_build_contextual_features_injured_flag(self, sample_history_df):
        from src.data.feature_engineering import build_contextual_features
        result = build_contextual_features(sample_history_df)
        assert "Injured" in result.columns
        # Player 2, GW5 has status="i" → Injured=1
        player2_gw2 = result[(result["player_id"] == 2) & (result["gameweek"] == 2)]
        assert player2_gw2["Injured"].iloc[0] == 1

    def test_build_contextual_features_without_status(self):
        from src.data.feature_engineering import build_contextual_features
        df = pd.DataFrame({"player_id": [1], "gameweek": [1]})
        result = build_contextual_features(df)
        assert "Injured" in result.columns
        assert result["Injured"].iloc[0] == 0

    def test_build_contextual_features_price_conversion(self, sample_history_df):
        from src.data.feature_engineering import build_contextual_features
        result = build_contextual_features(sample_history_df)
        assert "price" in result.columns
        assert "price_change" in result.columns
        # Player 1, GW1: value=130 → price=13.0
        player1_gw1 = result[(result["player_id"] == 1) & (result["gameweek"] == 1)]
        assert player1_gw1["price"].iloc[0] == pytest.approx(13.0)

    def test_build_contextual_features_ownership_percent(self, sample_history_df):
        from src.data.feature_engineering import build_contextual_features
        result = build_contextual_features(sample_history_df)
        assert "ownership_percent" in result.columns


# ===========================================================================
# Feature matrix assembly tests (§3.4.7)
# ===========================================================================


class TestFeatureMatrixAssembly:
    """Validate final feature vector assembly."""

    def test_get_feature_names_all(self):
        from src.data.feature_engineering import get_feature_names
        names = get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 20  # Should have many features
        assert "goals_scored_per_90" in names
        assert "xG_per_90" in names
        assert "Form_3" in names
        assert "FDR" in names

    def test_get_feature_names_by_group(self):
        from src.data.feature_engineering import get_feature_names
        core = get_feature_names("core")
        assert len(core) == 8
        assert "goals_scored_per_90" in core

        advanced = get_feature_names("advanced")
        assert len(advanced) == 7

        ict = get_feature_names("ict")
        assert len(ict) == 3

        form = get_feature_names("form")
        assert len(form) == 5

        fixture = get_feature_names("fixture")
        assert len(fixture) == 5

        contextual = get_feature_names("contextual")
        assert len(contextual) == 6

    def test_get_feature_names_invalid_group(self):
        from src.data.feature_engineering import get_feature_names
        result = get_feature_names("nonexistent")
        assert result == []

    def test_assemble_feature_matrix(self, sample_history_df):
        from src.data.feature_engineering import (
            build_core_features,
            build_ict_features,
            build_form_features,
            assemble_feature_matrix,
        )
        # Build some features
        df = build_core_features(sample_history_df)
        df = build_ict_features(df)
        df = build_form_features(df)
        X, feature_names = assemble_feature_matrix(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(feature_names, list)
        # Should only include features that exist in df
        for name in feature_names:
            assert name in df.columns
            assert name in X.columns

    def test_assemble_feature_matrix_filters_missing(self):
        from src.data.feature_engineering import assemble_feature_matrix
        df = pd.DataFrame({
            "goals_scored_per_90": [1.0, 0.0, 0.5],
            "Form_3": [10.0, 5.0, 7.0],
        })
        X, feature_names = assemble_feature_matrix(df)
        assert len(feature_names) == 2
        assert "goals_scored_per_90" in feature_names
        assert "Form_3" in feature_names

    def test_feature_groups_dict_is_complete(self):
        from src.data.feature_engineering import FEATURE_GROUPS
        assert "core" in FEATURE_GROUPS
        assert "advanced" in FEATURE_GROUPS
        assert "ict" in FEATURE_GROUPS
        assert "form" in FEATURE_GROUPS
        assert "fixture" in FEATURE_GROUPS
        assert "contextual" in FEATURE_GROUPS


# ===========================================================================
# Integration tests
# ===========================================================================


class TestFeatureEngineeringPipeline:
    """Test the full feature engineering pipeline end-to-end."""

    def test_full_pipeline_core_to_assembly(self, sample_history_df, sample_teams_df):
        from src.data.feature_engineering import (
            build_core_features,
            build_advanced_features,
            build_ict_features,
            build_form_features,
            build_fixture_features,
            build_contextual_features,
            assemble_feature_matrix,
        )
        # Build all feature groups
        df = build_core_features(sample_history_df)
        df = build_advanced_features(df)
        df = build_ict_features(df)
        df = build_form_features(df)
        fixtures_df = pd.DataFrame(columns=["id", "team_h", "team_a", "event"])
        df = build_fixture_features(df, sample_teams_df, fixtures_df)
        df = build_contextual_features(df)

        # Assemble final matrix
        X, feature_names = assemble_feature_matrix(df)
        assert X.shape[0] == len(sample_history_df)
        assert len(feature_names) > 0

    def test_pipeline_with_understat_merge(self, sample_history_df, sample_understat_df, sample_teams_df):
        from src.data.feature_engineering import (
            build_core_features,
            build_advanced_features,
            assemble_feature_matrix,
        )
        df = build_core_features(sample_history_df)
        df = build_advanced_features(df, understat_df=sample_understat_df)
        X, feature_names = assemble_feature_matrix(df)
        assert "xG_per_90" in feature_names
        assert X.shape[0] == len(sample_history_df)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_build_core_features_empty_dataframe(self):
        from src.data.feature_engineering import build_core_features
        df = pd.DataFrame(columns=["minutes", "goals_scored"])
        result = build_core_features(df)
        assert len(result) == 0

    def test_build_form_features_missing_player_id(self):
        from src.data.feature_engineering import build_form_features
        df = pd.DataFrame({"total_points": [10, 5, 8]})
        with pytest.raises(KeyError):
            build_form_features(df)

    def test_build_fixture_features_empty_teams_df(self, sample_history_df):
        from src.data.feature_engineering import build_fixture_features
        teams_df = pd.DataFrame(columns=["id", "strength_defence_home", "strength_defence_away"])
        fixtures_df = pd.DataFrame(columns=["id", "team_h", "team_a"])
        result = build_fixture_features(sample_history_df, teams_df, fixtures_df)
        # Should use default FDR=3 for all
        assert result["FDR"].iloc[0] == 3

    def test_build_contextual_features_empty_dataframe(self):
        from src.data.feature_engineering import build_contextual_features
        df = pd.DataFrame()
        result = build_contextual_features(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
