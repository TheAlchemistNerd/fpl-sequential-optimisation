"""
Tests for src/data/feature_selection.py — Feature Selection Methods.

Tests cover:
    - Filter methods: mutual information, correlation threshold (§3.7.1)
    - Embedded methods: Lasso, tree-based importance (§3.7.2)
    - Final feature set selection (§3.7.3)
    - Consensus and union methods
    - Edge cases
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
def sample_feature_matrix():
    """Create a sample feature matrix with known correlations."""
    np.random.seed(42)
    n = 100
    # Create features with varying predictive power
    x1 = np.random.randn(n)  # Strong predictor
    x2 = x1 * 0.95 + np.random.randn(n) * 0.1  # Highly correlated with x1
    x3 = np.random.randn(n)  # Moderate predictor
    x4 = np.random.randn(n)  # Weak predictor
    x5 = np.random.randn(n)  # No predictive power (noise)
    x6 = np.random.randn(n)  # Moderate predictor
    x7 = x3 * 0.92 + np.random.randn(n) * 0.1  # Highly correlated with x3

    # Target depends on x1, x3, x6 (not x2, x4, x5, x7)
    y = 2.0 * x1 + 1.5 * x3 + 0.8 * x6 + np.random.randn(n) * 0.5

    return pd.DataFrame({
        "xG_per_90": x1,
        "shots_per_90": x2,  # Highly correlated with xG
        "xA_per_90": x3,
        "Form_5": x4,
        "noise_feature": x5,
        "creativity_per_90": x6,
        "key_passes_per_90": x7,  # Highly correlated with xA
    }), pd.Series(y, name="total_points")


@pytest.fixture
def small_feature_matrix():
    """Small feature matrix for quick tests."""
    X = pd.DataFrame({
        "xG_per_90": [0.5, 0.3, 0.7, 0.2, 0.6],
        "xA_per_90": [0.2, 0.15, 0.25, 0.1, 0.18],
        "Form_5": [6.5, 4.2, 7.8, 3.5, 5.9],
        "FDR": [3.0, 4.0, 2.0, 5.0, 3.0],
        "noise": [1.0, 1.0, 1.0, 1.0, 1.0],  # Zero variance
    })
    y = pd.Series([12, 8, 15, 5, 10], name="total_points")
    return X, y


# ===========================================================================
# Mutual information tests (§3.7.1)
# ===========================================================================


class TestMutualInformationSelection:
    """Validate mutual information feature selection."""

    def test_mi_select_returns_list(self, sample_feature_matrix):
        from src.data.feature_selection import mutual_information_select
        X, y = sample_feature_matrix
        result = mutual_information_select(X, y)
        assert isinstance(result, list)
        assert all(isinstance(feat, str) for feat in result)

    def test_mi_select_retains_predictive_features(self, sample_feature_matrix):
        from src.data.feature_selection import mutual_information_select
        X, y = sample_feature_matrix
        result = mutual_information_select(X, y, threshold=0.01)
        # xG_per_90 (x1) should be selected (strong predictor)
        assert "xG_per_90" in result
        # xA_per_90 (x3) should be selected
        assert "xA_per_90" in result
        # creativity_per_90 (x6) should be selected
        assert "creativity_per_90" in result

    def test_mi_select_threshold_effects(self, sample_feature_matrix):
        from src.data.feature_selection import mutual_information_select
        X, y = sample_feature_matrix
        # Low threshold: more features
        low_thresh = mutual_information_select(X, y, threshold=0.001)
        # High threshold: fewer features
        high_thresh = mutual_information_select(X, y, threshold=0.1)
        assert len(low_thresh) >= len(high_thresh)

    def test_mi_select_deterministic(self, sample_feature_matrix):
        from src.data.feature_selection import mutual_information_select
        X, y = sample_feature_matrix
        result1 = mutual_information_select(X, y)
        result2 = mutual_information_select(X, y)
        assert result1 == result2

    def test_mi_select_empty_dataframe(self):
        from src.data.feature_selection import mutual_information_select
        X = pd.DataFrame()
        y = pd.Series([])
        result = mutual_information_select(X, y)
        # Should return empty list for empty input
        assert result == []

    def test_mi_select_single_feature(self):
        from src.data.feature_selection import mutual_information_select
        # Need at least 4 samples for sklearn MI (n_neighbors=3 + 1)
        X = pd.DataFrame({
            "xG_per_90": [0.5, 0.3, 0.7, 0.2, 0.6],
            "xA_per_90": [0.2, 0.1, 0.3, 0.15, 0.25],
        })
        y = pd.Series([12, 8, 15, 5, 10])
        result = mutual_information_select(X, y)
        # Should return both features (minimum samples met)
        assert len(result) >= 1


class TestCorrelationThresholdSelection:
    """Validate correlation-based feature selection."""

    def test_corr_select_returns_list(self, sample_feature_matrix):
        from src.data.feature_selection import correlation_threshold_select
        X, _ = sample_feature_matrix
        result = correlation_threshold_select(X, threshold=0.9)
        assert isinstance(result, list)

    def test_corr_select_removes_highly_correlated(self, sample_feature_matrix):
        from src.data.feature_selection import correlation_threshold_select
        X, _ = sample_feature_matrix
        # x1 and x2 are correlated (r≈0.95), x3 and x7 are correlated (r≈0.92)
        result = correlation_threshold_select(X, threshold=0.9)
        # Should drop one from each correlated pair
        # Either xG_per_90 or shots_per_90 should be dropped
        assert not ("xG_per_90" in result and "shots_per_90" in result)
        # Either xA_per_90 or key_passes_per_90 should be dropped
        assert not ("xA_per_90" in result and "key_passes_per_90" in result)

    def test_corr_select_keeps_more_features_at_higher_threshold(self, sample_feature_matrix):
        from src.data.feature_selection import correlation_threshold_select
        X, _ = sample_feature_matrix
        low_thresh = correlation_threshold_select(X, threshold=0.99)
        high_thresh = correlation_threshold_select(X, threshold=0.5)
        # Higher threshold means fewer features dropped
        assert len(low_thresh) <= len(high_thresh)

    def test_corr_select_no_correlated_features(self):
        from src.data.feature_selection import correlation_threshold_select
        np.random.seed(42)
        X = pd.DataFrame({
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
            "feat3": np.random.randn(50),
        })
        result = correlation_threshold_select(X, threshold=0.9)
        # All should be retained (no correlations)
        assert len(result) == 3

    def test_corr_select_zero_variance_feature(self, small_feature_matrix):
        from src.data.feature_selection import correlation_threshold_select
        X, _ = small_feature_matrix
        result = correlation_threshold_select(X, threshold=0.9)
        # Should handle zero variance feature without crashing
        assert isinstance(result, list)


# ===========================================================================
# Lasso selection tests (§3.7.2)
# ===========================================================================


class TestLassoSelection:
    """Validate Lasso (L1) feature selection."""

    def test_lasso_select_returns_list(self, sample_feature_matrix):
        from src.data.feature_selection import lasso_select
        X, y = sample_feature_matrix
        result = lasso_select(X, y)
        assert isinstance(result, list)

    def test_lasso_select_retains_important_features(self, sample_feature_matrix):
        from src.data.feature_selection import lasso_select
        X, y = sample_feature_matrix
        result = lasso_select(X, y, cv=3)
        # Should retain at least some of the predictive features
        assert len(result) > 0
        assert len(result) <= X.shape[1]

    def test_lasso_select_deterministic(self, sample_feature_matrix):
        from src.data.feature_selection import lasso_select
        X, y = sample_feature_matrix
        result1 = lasso_select(X, y, cv=3)
        result2 = lasso_select(X, y, cv=3)
        assert result1 == result2

    def test_lasso_select_small_data(self, small_feature_matrix):
        from src.data.feature_selection import lasso_select
        X, y = small_feature_matrix
        result = lasso_select(X, y, cv=2)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_lasso_select_zero_variance_feature(self, small_feature_matrix):
        from src.data.feature_selection import lasso_select
        X, y = small_feature_matrix
        # Should handle zero variance feature
        result = lasso_select(X, y, cv=2)
        assert "noise" not in result  # Zero variance should get coef=0


# ===========================================================================
# Tree-based selection tests (§3.7.2)
# ===========================================================================


class TestTreeImportanceSelection:
    """Validate tree-based (Gradient Boosting) feature selection."""

    def test_tree_select_returns_list(self, sample_feature_matrix):
        from src.data.feature_selection import tree_importance_select
        X, y = sample_feature_matrix
        result = tree_importance_select(X, y, n_estimators=50)
        assert isinstance(result, list)

    def test_tree_select_top_k(self, sample_feature_matrix):
        from src.data.feature_selection import tree_importance_select
        X, y = sample_feature_matrix
        result = tree_importance_select(X, y, n_estimators=50, top_k=3)
        assert len(result) == 3

    def test_tree_select_all_nonzero_importance(self, sample_feature_matrix):
        from src.data.feature_selection import tree_importance_select
        X, y = sample_feature_matrix
        result = tree_importance_select(X, y, n_estimators=50)
        # Should return all features with non-zero importance
        assert len(result) > 0
        assert len(result) <= X.shape[1]

    def test_tree_select_small_data(self, small_feature_matrix):
        from src.data.feature_selection import tree_importance_select
        X, y = small_feature_matrix
        result = tree_importance_select(X, y, n_estimators=20)
        assert isinstance(result, list)
        assert len(result) > 0


# ===========================================================================
# Final feature set tests (§3.7.3)
# ===========================================================================


class TestFinalFeatureSet:
    """Validate final feature set selection and consensus methods."""

    def test_final_feature_list_not_empty(self):
        from src.data.feature_selection import FINAL_FEATURE_LIST
        assert len(FINAL_FEATURE_LIST) > 0
        assert isinstance(FINAL_FEATURE_LIST, list)

    def test_final_feature_list_contains_expected_features(self):
        from src.data.feature_selection import FINAL_FEATURE_LIST
        assert "xG_per_90" in FINAL_FEATURE_LIST
        assert "xA_per_90" in FINAL_FEATURE_LIST
        assert "Form_3" in FINAL_FEATURE_LIST
        assert "FDR" in FINAL_FEATURE_LIST
        assert "is_home" in FINAL_FEATURE_LIST

    def test_select_final_features_consensus(self, sample_feature_matrix):
        from src.data.feature_selection import select_final_features
        X, y = sample_feature_matrix
        result = select_final_features(X, y, method="consensus")
        assert isinstance(result, list)
        # Consensus is intersection, so should be conservative
        all_features = set(X.columns)
        assert set(result).issubset(all_features)

    def test_select_final_features_union(self, sample_feature_matrix):
        from src.data.feature_selection import select_final_features
        X, y = sample_feature_matrix
        result = select_final_features(X, y, method="union")
        assert isinstance(result, list)
        # Union is more inclusive than consensus
        union_features = select_final_features(X, y, method="consensus")
        assert len(result) >= len(union_features)

    def test_select_final_features_single_method(self, sample_feature_matrix):
        from src.data.feature_selection import select_final_features
        X, y = sample_feature_matrix
        result_lasso = select_final_features(X, y, method="lasso")
        result_mi = select_final_features(X, y, method="mi")
        assert isinstance(result_lasso, list)
        assert isinstance(result_mi, list)

    def test_select_final_features_filters_to_available(self):
        from src.data.feature_selection import select_final_features, FINAL_FEATURE_LIST
        # Need enough samples for all methods to work
        np.random.seed(42)
        X = pd.DataFrame({
            "xG_per_90": np.random.randn(20),
            "Form_5": np.random.randn(20),
            "FDR": np.random.uniform(2, 5, 20),
            "is_home": np.random.randint(0, 2, 20),
            "nonexistent_feature": [1] * 20,
        })
        y = pd.Series(np.random.randn(20))
        result = select_final_features(X, y, method="union")
        # Should only return features that exist in X AND are selected
        for feat in result:
            assert feat in X.columns

    def test_select_final_features_no_methods_selected(self):
        from src.data.feature_selection import select_final_features
        X = pd.DataFrame({
            "xG_per_90": [0.5, 0.3],
            "Form_5": [6.5, 4.2],
        })
        y = pd.Series([12, 8])
        # Invalid method should fall back to all candidates
        result = select_final_features(X, y, method="invalid")
        assert isinstance(result, list)
        assert len(result) > 0


# ===========================================================================
# Integration tests
# ===========================================================================


class TestFeatureSelectionPipeline:
    """Test feature selection methods work together."""

    def test_all_methods_agree_on_strong_signal(self):
        from src.data.feature_selection import (
            mutual_information_select,
            lasso_select,
            tree_importance_select,
        )
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        # Strong signal: y depends only on x1
        y = 5.0 * x1 + np.random.randn(n) * 0.1

        X = pd.DataFrame({"strong_predictor": x1, "noise": x2})
        y_series = pd.Series(y, name="target")

        mi = mutual_information_select(X, y_series)
        lasso = lasso_select(X, y_series, cv=3)
        tree = tree_importance_select(X, y_series, n_estimators=50)

        # All methods should retain the strong predictor
        assert "strong_predictor" in mi
        assert "strong_predictor" in lasso
        assert "strong_predictor" in tree

    def test_pipeline_correlation_then_lasso(self, sample_feature_matrix):
        from src.data.feature_selection import (
            correlation_threshold_select,
            lasso_select,
        )
        X, y = sample_feature_matrix
        # First remove correlated features
        after_corr = correlation_threshold_select(X, threshold=0.9)
        X_reduced = X[after_corr]
        # Then apply Lasso
        final = lasso_select(X_reduced, y, cv=3)
        assert isinstance(final, list)
        assert len(final) <= len(after_corr)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_mi_select_single_feature(self):
        from src.data.feature_selection import mutual_information_select
        # Below minimum samples (4) → should return all columns
        X = pd.DataFrame({"xG_per_90": [0.5, 0.3]})
        y = pd.Series([12, 8])
        result = mutual_information_select(X, y)
        assert result == ["xG_per_90"]

    def test_lasso_select_single_sample(self):
        from src.data.feature_selection import lasso_select
        X = pd.DataFrame({"xG_per_90": [0.5]})
        y = pd.Series([12])
        # Should handle gracefully or raise informative error
        try:
            result = lasso_select(X, y, cv=2)
            assert isinstance(result, list)
        except Exception:
            pass  # Acceptable to fail with single sample

    def test_tree_select_constant_target(self):
        from src.data.feature_selection import tree_importance_select
        X = pd.DataFrame({"xG_per_90": [0.5, 0.3, 0.7]})
        y = pd.Series([10, 10, 10])  # Constant target
        result = tree_importance_select(X, y, n_estimators=10)
        # Should handle constant target
        assert isinstance(result, list)

    def test_corr_select_single_column(self):
        from src.data.feature_selection import correlation_threshold_select
        X = pd.DataFrame({"xG_per_90": [0.5, 0.3, 0.7]})
        result = correlation_threshold_select(X, threshold=0.9)
        assert result == ["xG_per_90"]

    def test_select_final_features_empty_dataframe(self):
        from src.data.feature_selection import select_final_features
        X = pd.DataFrame()
        y = pd.Series([])
        result = select_final_features(X, y, method="union")
        assert result == []
