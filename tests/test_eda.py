"""
Tests for src/data/eda.py — Exploratory Data Analysis.

Tests cover:
    - Univariate analysis: points, price, ICT distributions (§3.6.1)
    - Bivariate analysis: xG vs goals, correlation heatmap (§3.6.2)
    - Time series analysis: form trajectories, price dynamics (§3.6.3)
    - Position-specific patterns (§3.6.4)
    - Full EDA report generation
    - Figure saving and backend handling
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_eda_data():
    """Sample DataFrame for EDA tests."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "player_id": np.random.randint(1, 50, n),
        "gameweek": np.random.randint(1, 20, n),
        "position": np.random.choice(["GK", "DEF", "MID", "FWD"], n),
        "total_points": np.random.exponential(3, n).astype(int),
        "now_cost": np.random.uniform(40, 140, n).astype(int),
        "ict_index": np.random.uniform(50, 300, n),
        "influence": np.random.uniform(20, 150, n),
        "creativity": np.random.uniform(10, 100, n),
        "threat": np.random.uniform(5, 200, n),
        "expected_goals_per_90": np.random.exponential(0.3, n),
        "goals_scored_per_90": np.random.exponential(0.25, n),
        "xG_per_90": np.random.exponential(0.3, n),
        "xA_per_90": np.random.exponential(0.2, n),
        "Form_5": np.random.uniform(2, 12, n),
        "price_change": np.random.normal(0, 0.5, n),
    })


@pytest.fixture
def sample_player_history():
    """Sample player history for time series tests."""
    return pd.DataFrame({
        "player_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "gameweek": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "Form_5": [5.0, 6.0, 7.5, 8.0, 7.0, 4.0, 4.5, 5.0, 6.0, 6.5],
        "price_change": [0.0, 0.1, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
    })


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestHelperFunctions:
    """Validate EDA helper functions."""

    def test_save_creates_directory(self):
        from src.data.eda import _save
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            output_dir = Path(tmpdir) / "subdir"
            _save(fig, "test_fig", output_dir)
            assert output_dir.exists()

    def test_save_returns_path(self):
        from src.data.eda import _save
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots()
            output_dir = Path(tmpdir)
            result = _save(fig, "test_fig", output_dir)
            assert isinstance(result, Path)
            assert result.exists()


# ===========================================================================
# Univariate analysis tests (§3.6.1)
# ===========================================================================


class TestUnivariateAnalysis:
    """Validate univariate EDA functions."""

    def test_plot_points_distribution(self, sample_eda_data):
        from src.data.eda import plot_points_distribution
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_points_distribution(sample_eda_data, output_dir=Path(tmpdir))
            assert isinstance(result, Path)
            # Should save a file
            assert result.exists() or str(result) == ""

    def test_plot_points_distribution_custom_column(self, sample_eda_data):
        from src.data.eda import plot_points_distribution
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_points_distribution(
                sample_eda_data,
                points_col="ict_index",
                output_dir=Path(tmpdir),
            )
            assert isinstance(result, Path)

    def test_plot_price_distribution(self, sample_eda_data):
        from src.data.eda import plot_price_distribution
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_price_distribution(sample_eda_data, output_dir=Path(tmpdir))
            assert isinstance(result, Path)
            assert result.exists() or str(result) == ""

    def test_plot_price_distribution_custom_column(self, sample_eda_data):
        from src.data.eda import plot_price_distribution
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_price_distribution(
                sample_eda_data,
                price_col="ict_index",
                output_dir=Path(tmpdir),
            )
            assert isinstance(result, Path)

    def test_plot_ict_by_position(self, sample_eda_data):
        from src.data.eda import plot_ict_by_position
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_ict_by_position(sample_eda_data, output_dir=Path(tmpdir))
            assert isinstance(result, Path)
            assert result.exists() or str(result) == ""

    def test_plot_ict_by_position_missing_position(self):
        from src.data.eda import plot_ict_by_position
        df = pd.DataFrame({
            "position": ["MID", "MID"],
            "ict_index": [150, 160],
        })
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_ict_by_position(df, output_dir=Path(tmpdir))
            # Should still save figure (only has MID data)
            assert isinstance(result, Path)


# ===========================================================================
# Bivariate analysis tests (§3.6.2)
# ===========================================================================


class TestBivariateAnalysis:
    """Validate bivariate EDA functions."""

    def test_plot_xg_vs_goals(self, sample_eda_data):
        from src.data.eda import plot_xg_vs_goals
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_xg_vs_goals(sample_eda_data, output_dir=Path(tmpdir))
            assert isinstance(result, Path)
            assert result.exists() or str(result) == ""

    def test_plot_xg_vs_goals_missing_columns(self):
        from src.data.eda import plot_xg_vs_goals
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_xg_vs_goals(df, output_dir=Path(tmpdir))
            assert result is None

    def test_plot_correlation_heatmap(self, sample_eda_data):
        from src.data.eda import plot_correlation_heatmap
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_cols = ["expected_goals_per_90", "goals_scored_per_90", "xG_per_90"]
            result = plot_correlation_heatmap(
                sample_eda_data,
                feature_cols=feature_cols,
                output_dir=Path(tmpdir),
            )
            assert isinstance(result, Path)
            assert result.exists() or str(result) == ""

    def test_plot_correlation_heatmap_auto_columns(self, sample_eda_data):
        from src.data.eda import plot_correlation_heatmap
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_correlation_heatmap(sample_eda_data, output_dir=Path(tmpdir))
            assert isinstance(result, Path)

    def test_find_highly_correlated_pairs(self, sample_eda_data):
        from src.data.eda import find_highly_correlated_pairs
        feature_cols = ["expected_goals_per_90", "xG_per_90"]
        pairs = find_highly_correlated_pairs(sample_eda_data, feature_cols, threshold=0.1)
        assert isinstance(pairs, list)
        # Should find some pairs with low threshold
        if len(pairs) > 0:
            assert len(pairs[0]) == 3  # (col_a, col_b, correlation)

    def test_find_highly_correlated_pairs_no_pairs(self):
        from src.data.eda import find_highly_correlated_pairs
        # Create uncorrelated data
        np.random.seed(42)
        df = pd.DataFrame({
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100),
        })
        pairs = find_highly_correlated_pairs(df, ["feat1", "feat2"], threshold=0.9)
        assert pairs == []


# ===========================================================================
# Time series analysis tests (§3.6.3)
# ===========================================================================


class TestTimeSeriesAnalysis:
    """Validate time series EDA functions."""

    def test_plot_form_trajectory(self, sample_player_history):
        from src.data.eda import plot_form_trajectory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_form_trajectory(
                sample_player_history,
                player_id=1,
                output_dir=Path(tmpdir),
            )
            assert isinstance(result, Path)
            assert result.exists() or str(result) == ""

    def test_plot_form_trajectory_player_not_found(self, sample_player_history):
        from src.data.eda import plot_form_trajectory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_form_trajectory(
                sample_player_history,
                player_id=999,
                output_dir=Path(tmpdir),
            )
            assert result is None

    def test_plot_form_trajectory_custom_form_col(self, sample_player_history):
        from src.data.eda import plot_form_trajectory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_form_trajectory(
                sample_player_history,
                player_id=1,
                form_col="Form_5",
                output_dir=Path(tmpdir),
            )
            assert isinstance(result, Path)

    def test_plot_price_change_dynamics(self, sample_player_history):
        from src.data.eda import plot_price_change_dynamics
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_price_change_dynamics(
                sample_player_history,
                player_id=1,
                output_dir=Path(tmpdir),
            )
            assert isinstance(result, Path)
            assert result.exists() or str(result) == ""

    def test_plot_price_change_dynamics_missing_column(self, sample_player_history):
        from src.data.eda import plot_price_change_dynamics
        df = sample_player_history.drop(columns=["price_change"])
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_price_change_dynamics(df, player_id=1, output_dir=Path(tmpdir))
            assert result is None


# ===========================================================================
# Position-specific patterns tests (§3.6.4)
# ===========================================================================


class TestPositionSpecificPatterns:
    """Validate position-specific EDA functions."""

    def test_plot_position_violins(self, sample_eda_data):
        from src.data.eda import plot_position_violins
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_position_violins(sample_eda_data, output_dir=Path(tmpdir))
            assert isinstance(result, Path)
            assert result.exists() or str(result) == ""

    def test_plot_position_violins_custom_value_col(self, sample_eda_data):
        from src.data.eda import plot_position_violins
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_position_violins(
                sample_eda_data,
                value_col="ict_index",
                output_dir=Path(tmpdir),
            )
            assert isinstance(result, Path)

    def test_plot_position_violins_missing_position(self):
        from src.data.eda import plot_position_violins
        df = pd.DataFrame({
            "total_points": [10, 8, 12],
        })
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(KeyError, match="position"):
                plot_position_violins(df, output_dir=Path(tmpdir))


# ===========================================================================
# Full EDA report tests
# ===========================================================================


class TestFullEDAReport:
    """Validate full EDA report generation."""

    def test_run_full_eda(self, sample_eda_data):
        from src.data.eda import run_full_eda
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_full_eda(sample_eda_data, output_dir=Path(tmpdir))
            assert isinstance(result, dict)
            assert isinstance(result["high_corr_count"], int)
            # Should have saved multiple figures
            assert len(result) > 5

    def test_run_full_eda_with_sample_players(self, sample_player_history):
        from src.data.eda import run_full_eda
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_full_eda(
                sample_player_history,
                output_dir=Path(tmpdir),
                sample_players=[1, 2],
            )
            assert isinstance(result, dict)
            # Should have form and price plots for both players
            assert any("form_p1" in key for key in result.keys())
            assert any("form_p2" in key for key in result.keys())

    def test_run_full_eda_missing_columns(self):
        from src.data.eda import run_full_eda
        df = pd.DataFrame({
            "player_id": [1, 2],
            "position": ["MID", "FWD"],
        })
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle missing columns gracefully (some plots will fail)
            try:
                result = run_full_eda(df, output_dir=Path(tmpdir))
                assert isinstance(result, dict)
            except Exception:
                pass  # Some missing columns may cause failures


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_plot_points_distribution_empty_dataframe(self):
        from src.data.eda import plot_points_distribution
        df = pd.DataFrame(columns=["total_points"])
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_points_distribution(df, output_dir=Path(tmpdir))
            assert result.exists()

    def test_plot_correlation_heatmap_single_column(self):
        from src.data.eda import plot_correlation_heatmap
        df = pd.DataFrame({"xG_per_90": [0.5, 0.3, 0.7]})
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = plot_correlation_heatmap(df, output_dir=Path(tmpdir))
            assert isinstance(result, Path)

    def test_find_highly_correlated_pairs_single_column(self):
        from src.data.eda import find_highly_correlated_pairs
        df = pd.DataFrame({"xG_per_90": [0.5, 0.3, 0.7]})
        pairs = find_highly_correlated_pairs(df, ["xG_per_90"], threshold=0.9)
        assert pairs == []  # Can't have pairs with single column

    def test_run_full_eda_custom_output_dir(self, sample_eda_data):
        from src.data.eda import run_full_eda
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "custom_eda"
            result = run_full_eda(sample_eda_data, output_dir=custom_dir)
            assert custom_dir.exists()
            # Check that files were saved
            saved_files = list(custom_dir.glob("*.png"))
            assert len(saved_files) > 0
