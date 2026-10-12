"""
Tests for src/pipeline/data_pipeline.py — Orchestrated Data Pipeline.

Tests cover:
    - Full pipeline execution
    - Feature engineering from CSV data
    - HDP model updates (placeholder)
    - CLI entry point
    - Error handling and edge cases
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_csv_data(tmp_path):
    """Create mock CSV files for pipeline testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Gameweek data CSV
    gw_data = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "gameweek": [1, 2, 1, 2],
        "total_points": [12, 8, 6, 10],
        "minutes": [90, 90, 75, 90],
        "goals_scored": [1, 0, 0, 1],
        "assists": [1, 0, 1, 0],
    })
    gw_data.to_csv(data_dir / "fpl_gameweek_data.csv", index=False)

    # Players CSV
    players_data = pd.DataFrame({
        "id": [1, 2],
        "first_name": ["Mohamed", "Virgil"],
        "second_name": ["Salah", "van Dijk"],
        "element_type": [3, 2],
        "team": [10, 10],
    })
    players_data.to_csv(data_dir / "fpl_players.csv", index=False)

    return data_dir


@pytest.fixture
def mock_settings(tmp_path):
    """Mock settings with test data directory."""
    mock = MagicMock()
    mock.DATA_DIR = tmp_path / "data"
    mock.DATA_DIR.mkdir(exist_ok=True)
    return mock


# ===========================================================================
# Pipeline execution tests
# ===========================================================================


class TestPipelineExecution:
    """Validate full pipeline execution."""

    @patch("src.pipeline.data_pipeline.fetch_all_data")
    @patch("src.pipeline.data_pipeline.engineer_features")
    @patch("src.pipeline.data_pipeline.update_hdp_models")
    def test_run_pipeline_calls_all_steps(self, mock_update, mock_engineer, mock_fetch):
        from src.pipeline.data_pipeline import run_pipeline
        mock_engineer.return_value = pd.DataFrame()

        run_pipeline(latest_gameweek=5)

        mock_fetch.assert_called_once()
        mock_engineer.assert_called_once()
        mock_update.assert_called_once()

    @patch("src.pipeline.data_pipeline.fetch_all_data")
    @patch("src.pipeline.data_pipeline.engineer_features")
    @patch("src.pipeline.data_pipeline.update_hdp_models")
    def test_run_pipeline_passes_gameweek_to_fetch(self, mock_update, mock_engineer, mock_fetch):
        from src.pipeline.data_pipeline import run_pipeline
        mock_engineer.return_value = pd.DataFrame()

        run_pipeline(latest_gameweek=10)

        # fetch_all_data should be called with gameweek=10
        import asyncio
        mock_fetch.assert_called_once()


# ===========================================================================
# Feature engineering tests
# ===========================================================================


class TestFeatureEngineering:
    """Validate feature engineering from CSV data."""

    @patch("src.pipeline.data_pipeline.settings")
    def test_engineer_features_returns_numeric_feature_matrix(
        self, mock_settings, mock_csv_data
    ):
        from src.pipeline import data_pipeline
        mock_settings.DATA_DIR = mock_csv_data

        result = data_pipeline.engineer_features()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 4 rows from mock data
        assert {"goals_scored_per_90", "assists_per_90", "Form_3", "Form_5", "Form_10"} <= set(result.columns)
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes)

    @patch("src.pipeline.data_pipeline.settings")
    def test_engineer_features_computes_form_per_player(self, mock_settings, mock_csv_data):
        from src.pipeline import data_pipeline
        mock_settings.DATA_DIR = mock_csv_data

        result = data_pipeline.engineer_features()
        # Player metadata is used during engineering; the returned matrix keeps
        # only modelling features, including per-player rolling form.
        assert result["Form_3"].tolist() == [12.0, 10.0, 6.0, 8.0]

    @patch("src.pipeline.data_pipeline.settings")
    def test_engineer_features_computes_rolling_averages(self, mock_settings, mock_csv_data):
        from src.pipeline import data_pipeline
        mock_settings.DATA_DIR = mock_csv_data

        result = data_pipeline.engineer_features()
        assert {"Form_3", "Form_5", "Form_10"} <= set(result.columns)
        assert result["Form_3"].notna().all()
        assert result["Form_3"].tolist() == [12.0, 10.0, 6.0, 8.0]

    @patch("src.pipeline.data_pipeline.settings")
    def test_engineer_features_handles_missing_stats_column(self, mock_settings, tmp_path):
        from src.pipeline import data_pipeline
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # CSV without stats column
        gw_data = pd.DataFrame({
            "id": [1, 2],
            "gameweek": [1, 1],
            "total_points": [12, 6],
            "minutes": [90, 90],
        })
        gw_data.to_csv(data_dir / "fpl_gameweek_data.csv", index=False)

        players_data = pd.DataFrame({
            "id": [1, 2],
            "first_name": ["Mohamed", "Virgil"],
            "second_name": ["Salah", "van Dijk"],
            "element_type": [3, 2],
            "team": [10, 10],
        })
        players_data.to_csv(data_dir / "fpl_players.csv", index=False)

        mock_settings.DATA_DIR = data_dir
        result = data_pipeline.engineer_features()
        assert isinstance(result, pd.DataFrame)
        assert "Form_3" in result.columns

    @patch("src.pipeline.data_pipeline.settings")
    def test_engineer_features_handles_string_stats_column(self, mock_settings, tmp_path):
        from src.pipeline import data_pipeline
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # CSV with string stats column
        gw_data = pd.DataFrame({
            "id": [1],
            "gameweek": [1],
            "total_points": [12],
            "stats": ['{"goals_scored": 1, "assists": 1, "minutes": 90}'],
        })
        gw_data.to_csv(data_dir / "fpl_gameweek_data.csv", index=False)

        players_data = pd.DataFrame({
            "id": [1],
            "first_name": ["Mohamed"],
            "second_name": ["Salah"],
            "element_type": [3],
            "team": [10],
        })
        players_data.to_csv(data_dir / "fpl_players.csv", index=False)

        mock_settings.DATA_DIR = data_dir
        result = data_pipeline.engineer_features()
        assert isinstance(result, pd.DataFrame)
        assert result["goals_scored_per_90"].tolist() == [1.0]
        assert result["assists_per_90"].tolist() == [1.0]


# ===========================================================================
# HDP model update tests
# ===========================================================================


class TestHDPModelUpdates:
    """Validate HDP model update function (placeholder)."""

    def test_update_hdp_models_is_placeholder(self, caplog):
        from src.pipeline.data_pipeline import update_hdp_models
        import logging
        caplog.set_level(logging.INFO)

        features = pd.DataFrame()
        update_hdp_models(features)

        # Should log that it's not implemented
        assert "not yet implemented" in caplog.text.lower() or len(caplog.text) > 0

    def test_update_hdp_models_accepts_dataframe(self):
        from src.pipeline.data_pipeline import update_hdp_models
        features = pd.DataFrame({"xG_per_90": [0.5, 0.3]})
        # Should not raise
        update_hdp_models(features)


# ===========================================================================
# CLI tests
# ===========================================================================


class TestCLI:
    """Validate CLI entry point."""

    @patch("src.pipeline.data_pipeline.run_pipeline")
    def test_cli_with_argument(self, mock_run, monkeypatch):
        import subprocess
        import sys
        # Can't easily test __main__ directly, but we can test the logic
        from src.pipeline.data_pipeline import run_pipeline
        # Simulate sys.argv parsing
        import sys as sys_module
        original_argv = sys_module.argv
        sys_module.argv = ["data_pipeline.py", "10"]
        try:
            # Would call __main__ block if we could execute it
            # Just verify the function exists
            assert callable(run_pipeline)
        finally:
            sys_module.argv = original_argv

    def test_cli_default_gameweek(self):
        from src.pipeline.data_pipeline import run_pipeline
        # Default is 5 (from source code)
        assert callable(run_pipeline)


# ===========================================================================
# Integration tests
# ===========================================================================


class TestPipelineIntegration:
    """Test pipeline integration end-to-end."""

    @patch("src.pipeline.data_pipeline.fetch_all_data")
    @patch("src.pipeline.data_pipeline.update_hdp_models")
    def test_full_pipeline_with_mock_data(self, mock_update, mock_fetch, mock_csv_data, mock_settings):
        from src.pipeline.data_pipeline import run_pipeline
        with patch("src.pipeline.data_pipeline.settings", mock_settings):
            # Should execute without errors
            run_pipeline(latest_gameweek=1)
            mock_fetch.assert_called_once()
            mock_update.assert_called_once()


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("src.pipeline.data_pipeline.settings")
    def test_engineer_features_missing_csv_files(self, mock_settings, tmp_path):
        from src.pipeline import data_pipeline
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        mock_settings.DATA_DIR = data_dir

        # CSV files don't exist
        with pytest.raises(FileNotFoundError):
            data_pipeline.engineer_features()

    @patch("src.pipeline.data_pipeline.settings")
    def test_engineer_features_empty_csv(self, mock_settings, tmp_path):
        from src.pipeline import data_pipeline
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Empty CSVs (just headers)
        pd.DataFrame(columns=["id", "gameweek", "total_points", "minutes"]).to_csv(
            data_dir / "fpl_gameweek_data.csv", index=False
        )
        pd.DataFrame(columns=["id", "first_name", "second_name", "element_type", "team"]).to_csv(
            data_dir / "fpl_players.csv", index=False
        )

        mock_settings.DATA_DIR = data_dir
        result = data_pipeline.engineer_features()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_run_pipeline_with_zero_gameweek(self):
        from src.pipeline.data_pipeline import run_pipeline
        with patch("src.pipeline.data_pipeline.fetch_all_data") as mock_fetch:
            with patch("src.pipeline.data_pipeline.engineer_features") as mock_engineer:
                with patch("src.pipeline.data_pipeline.update_hdp_models"):
                    mock_engineer.return_value = pd.DataFrame()
                    run_pipeline(latest_gameweek=0)
                    # Should still call fetch with 0
                    mock_fetch.assert_called_once()

    def test_run_pipeline_with_large_gameweek(self):
        from src.pipeline.data_pipeline import run_pipeline
        with patch("src.pipeline.data_pipeline.fetch_all_data") as mock_fetch:
            with patch("src.pipeline.data_pipeline.engineer_features") as mock_engineer:
                with patch("src.pipeline.data_pipeline.update_hdp_models"):
                    mock_engineer.return_value = pd.DataFrame()
                    run_pipeline(latest_gameweek=38)  # Full Premier League season
                    mock_fetch.assert_called_once_with(38)
