"""
Tests for src/data/preprocessing.py — Data Preprocessing Pipeline.

Tests cover:
    - Missing value imputation (§3.5.1)
    - Feature standardisation (§3.5.2)
    - Position-based splitting (§3.5.3)
    - Edge cases and error handling
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_player_data():
    """Create a sample player dataset with some missing values."""
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5, 6],
        "position": ["GK", "DEF", "MID", "FWD", "MID", "FWD"],
        "minutes": [1800, 1620, 900, 0, 450, 1200],
        "goals_scored_per_90": [0.0, 0.15, np.nan, np.nan, 0.3, 0.5],
        "assists_per_90": [0.0, 0.1, 0.2, np.nan, np.nan, 0.15],
        "xG_per_90": [0.0, 0.05, 0.15, np.nan, 0.25, np.nan],
        "total_points": [120, 95, 60, 0, 45, 85],
        "Sentiment": [0.3, np.nan, -0.1, 0.0, np.nan, 0.2],
        "SentimentVol": [0.1, np.nan, 0.2, 0.0, np.nan, 0.15],
    })


@pytest.fixture
def sample_feature_data():
    """Create sample feature data for standardisation tests."""
    return pd.DataFrame({
        "xG_per_90": [0.5, 0.3, 0.7, 0.2, 0.6],
        "xA_per_90": [0.2, 0.15, 0.25, 0.1, 0.18],
        "Form_5": [6.5, 4.2, 7.8, 3.5, 5.9],
    })


# ===========================================================================
# Imputation tests (§3.5.1)
# ===========================================================================


class TestImputation:
    """Validate missing value imputation strategy."""

    def test_impute_zero_minutes_sets_per90_to_zero(self, sample_player_data):
        from src.data.preprocessing import impute_missing_values
        result = impute_missing_values(sample_player_data)
        # Player 4 has 0 minutes, all per-90 should be 0
        player4 = result[result["player_id"] == 4]
        assert player4["goals_scored_per_90"].iloc[0] == 0.0
        assert player4["assists_per_90"].iloc[0] == 0.0
        assert player4["xG_per_90"].iloc[0] == 0.0

    def test_impute_fills_nan_with_median(self, sample_player_data):
        from src.data.preprocessing import impute_missing_values
        result = impute_missing_values(sample_player_data)
        # Should have no NaN in numeric columns (except possibly non-per-90)
        per90_cols = [c for c in result.columns if c.endswith("_per_90")]
        assert result[per90_cols].isna().sum().sum() == 0

    def test_impute_sentiment_defaults_to_zero(self, sample_player_data):
        from src.data.preprocessing import impute_missing_values
        result = impute_missing_values(sample_player_data)
        # NaN sentiment should default to 0
        assert result["Sentiment"].isna().sum() == 0
        assert result["SentimentVol"].isna().sum() == 0
        # Check that previously NaN values are now 0
        player2 = result[result["player_id"] == 2]
        assert player2["Sentiment"].iloc[0] == 0.0
        assert player2["SentimentVol"].iloc[0] == 0.0

    def test_impute_preserves_non_missing_values(self, sample_player_data):
        from src.data.preprocessing import impute_missing_values
        result = impute_missing_values(sample_player_data)
        # Player 1 had no missing values, should be unchanged
        player1_before = sample_player_data[sample_player_data["player_id"] == 1]
        player1_after = result[result["player_id"] == 1]
        assert player1_after["goals_scored_per_90"].iloc[0] == player1_before["goals_scored_per_90"].iloc[0]
        assert player1_after["Sentiment"].iloc[0] == player1_before["Sentiment"].iloc[0]

    def test_impute_returns_new_dataframe(self, sample_player_data):
        from src.data.preprocessing import impute_missing_values
        result = impute_missing_values(sample_player_data)
        assert result is not sample_player_data

    def test_impute_median_within_position_group(self):
        from src.data.preprocessing import impute_missing_values
        df = pd.DataFrame({
            "position": ["MID", "MID", "MID", "FWD", "FWD"],
            "minutes": [900, 800, 700, 600, 500],
            "xG_per_90": [0.5, np.nan, 0.3, 0.6, 0.4],
        })
        result = impute_missing_values(df)
        # MID median is 0.4 (between 0.5 and 0.3), should fill NaN with that
        mid_player = result[(result["position"] == "MID") & (result["xG_per_90"].notna())]
        # The previously NaN MID player should now have the median
        assert result.loc[(result["position"] == "MID") & (result["minutes"] == 800), "xG_per_90"].iloc[0] == pytest.approx(0.4)

    def test_impute_falls_back_to_global_median(self):
        from src.data.preprocessing import impute_missing_values
        # All values in a position group are NaN
        df = pd.DataFrame({
            "position": ["GK", "GK"],
            "minutes": [1800, 1620],
            "xG_per_90": [np.nan, np.nan],
        })
        result = impute_missing_values(df)
        # Should fall back to global median (which is also NaN, but we handle it)
        assert result["xG_per_90"].isna().sum() == 0

    def test_impute_no_remaining_nans_in_numeric_cols(self, sample_player_data):
        from src.data.preprocessing import impute_missing_values
        result = impute_missing_values(sample_player_data)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        total_nans = result[numeric_cols].isna().sum().sum()
        assert total_nans == 0


# ===========================================================================
# Standardisation tests (§3.5.2)
# ===========================================================================


class TestFeatureStandardiser:
    """Validate feature standardisation to prevent data leakage."""

    def test_standardiser_fit(self, sample_feature_data):
        from src.data.preprocessing import FeatureStandardiser
        scaler = FeatureStandardiser()
        scaler.fit(sample_feature_data)
        assert scaler._is_fit is True
        assert len(scaler.feature_names) == 3

    def test_standardiser_transform(self, sample_feature_data):
        from src.data.preprocessing import FeatureStandardiser
        scaler = FeatureStandardiser()
        scaler.fit(sample_feature_data)
        result = scaler.transform(sample_feature_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_feature_data.shape
        # After standardisation, mean should be ~0
        means = result.mean()
        assert means.abs().max() < 1e-10

    def test_standardiser_fit_transform(self, sample_feature_data):
        from src.data.preprocessing import FeatureStandardiser
        scaler = FeatureStandardiser()
        result = scaler.fit_transform(sample_feature_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_feature_data.shape
        # Standardised to mean ~0, std ~1
        assert result.mean().abs().max() < 1e-10
        assert (result.std() - 1.0).abs().max() < 1e-10

    def test_standardiser_transform_without_fit_raises(self, sample_feature_data):
        from src.data.preprocessing import FeatureStandardiser
        scaler = FeatureStandardiser()
        with pytest.raises(RuntimeError, match="must be fitted"):
            scaler.transform(sample_feature_data)

    def test_standardiser_column_order_preserved(self, sample_feature_data):
        from src.data.preprocessing import FeatureStandardiser
        scaler = FeatureStandardiser()
        scaler.fit(sample_feature_data)
        result = scaler.transform(sample_feature_data)
        assert list(result.columns) == list(sample_feature_data.columns)

    def test_standardiser_transform_with_extra_columns(self, sample_feature_data):
        from src.data.preprocessing import FeatureStandardiser
        scaler = FeatureStandardiser()
        scaler.fit(sample_feature_data)
        # New data with additional columns (should only transform fitted ones)
        new_data = sample_feature_data.copy()
        new_data["extra_col"] = [1, 2, 3, 4, 5]
        result = scaler.transform(new_data)
        assert list(result.columns) == list(sample_feature_data.columns)
        assert "extra_col" not in result.columns

    def test_standardiser_preserves_index(self, sample_feature_data):
        from src.data.preprocessing import FeatureStandardiser
        scaler = FeatureStandardiser()
        custom_index = [10, 20, 30, 40, 50]
        sample_feature_data.index = custom_index
        result = scaler.fit_transform(sample_feature_data)
        assert list(result.index) == custom_index


# ===========================================================================
# Position-based splitting tests (§3.5.3)
# ===========================================================================


class TestPositionSplitting:
    """Validate position-based data splitting for separate models."""

    def test_split_by_position_returns_dict(self, sample_player_data):
        from src.data.preprocessing import split_by_position
        result = split_by_position(sample_player_data)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"GK", "DEF", "MID", "FWD"}

    def test_split_by_position_correct_counts(self, sample_player_data):
        from src.data.preprocessing import split_by_position
        result = split_by_position(sample_player_data)
        assert len(result["GK"]) == 1
        assert len(result["DEF"]) == 1
        assert len(result["MID"]) == 2
        assert len(result["FWD"]) == 2

    def test_split_by_position_returns_copies(self, sample_player_data):
        from src.data.preprocessing import split_by_position
        result = split_by_position(sample_player_data)
        # Modifying result shouldn't affect original
        result["MID"].iloc[0, 0] = 999
        assert sample_player_data.iloc[0, 0] != 999

    def test_split_by_position_empty_position(self):
        from src.data.preprocessing import split_by_position
        df = pd.DataFrame({
            "player_id": [1, 2],
            "position": ["MID", "MID"],
        })
        result = split_by_position(df)
        assert len(result["GK"]) == 0
        assert len(result["DEF"]) == 0
        assert len(result["FWD"]) == 0
        assert len(result["MID"]) == 2

    def test_build_position_standardisers(self, sample_player_data):
        from src.data.preprocessing import split_by_position, build_position_standardisers
        position_dfs = split_by_position(sample_player_data)
        feature_cols = ["xG_per_90", "total_points"]
        standardisers = build_position_standardisers(position_dfs, feature_cols)
        assert isinstance(standardisers, dict)
        assert "MID" in standardisers
        assert "FWD" in standardisers
        assert isinstance(standardisers["MID"], type(FeatureStandardiser()))

    def test_build_position_standardisers_skips_empty(self):
        from src.data.preprocessing import build_position_standardisers
        position_dfs = {
            "GK": pd.DataFrame(),
            "DEF": pd.DataFrame(),
            "MID": pd.DataFrame({"xG_per_90": [0.5, 0.3]}),
            "FWD": pd.DataFrame({"xG_per_90": [0.6, 0.4]}),
        }
        standardisers = build_position_standardisers(position_dfs, ["xG_per_90"])
        assert "GK" not in standardisers
        assert "DEF" not in standardisers
        assert "MID" in standardisers
        assert "FWD" in standardisers


# ===========================================================================
# Integration tests
# ===========================================================================


class TestPreprocessingPipeline:
    """Test the full preprocessing pipeline end-to-end."""

    def test_full_pipeline_impute_then_standardise(self, sample_player_data):
        from src.data.preprocessing import (
            impute_missing_values,
            FeatureStandardiser,
            split_by_position,
            build_position_standardisers,
        )
        # Step 1: Impute
        imputed = impute_missing_values(sample_player_data)
        assert imputed.isna().sum().sum() == 0

        # Step 2: Split by position
        position_dfs = split_by_position(imputed)
        assert all(len(df) > 0 for df in position_dfs.values())

        # Step 3: Build standardisers per position
        feature_cols = ["xG_per_90", "total_points"]
        standardisers = build_position_standardisers(position_dfs, feature_cols)
        assert len(standardisers) == 4

        # Step 4: Transform each position
        for pos, scaler in standardisers.items():
            transformed = scaler.transform(position_dfs[pos][feature_cols])
            assert transformed.shape[1] == len(feature_cols)
            assert transformed.mean().abs().max() < 1e-6

    def test_pipeline_with_zero_minute_players(self):
        from src.data.preprocessing import impute_missing_values, FeatureStandardiser
        df = pd.DataFrame({
            "player_id": [1, 2],
            "position": ["MID", "MID"],
            "minutes": [0, 900],
            "goals_scored_per_90": [np.nan, 0.3],
            "total_points": [0, 50],
        })
        imputed = impute_missing_values(df)
        assert imputed.loc[0, "goals_scored_per_90"] == 0.0  # Zero minutes → 0
        assert imputed.loc[1, "goals_scored_per_90"] == pytest.approx(0.3)

        # Standardise
        scaler = FeatureStandardiser()
        features = imputed[["goals_scored_per_90", "total_points"]]
        transformed = scaler.fit_transform(features)
        assert transformed.shape == (2, 2)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_impute_empty_dataframe(self):
        from src.data.preprocessing import impute_missing_values
        df = pd.DataFrame({"position": [], "minutes": []})
        result = impute_missing_values(df)
        assert len(result) == 0

    def test_standardiser_single_row(self):
        from src.data.preprocessing import FeatureStandardiser
        df = pd.DataFrame({"xG_per_90": [0.5]})
        scaler = FeatureStandardiser()
        # Should not crash, but std will be 0, so division by zero handled
        result = scaler.fit_transform(df)
        assert result.shape == (1, 1)

    def test_split_by_position_missing_column(self):
        from src.data.preprocessing import split_by_position
        df = pd.DataFrame({"player_id": [1, 2]})
        with pytest.raises(KeyError):
            split_by_position(df, position_col="position")

    def test_impute_missing_minutes_column(self):
        from src.data.preprocessing import impute_missing_values
        df = pd.DataFrame({
            "position": ["MID"],
            "goals_scored_per_90": [0.3],
        })
        with pytest.raises(KeyError):
            impute_missing_values(df)
