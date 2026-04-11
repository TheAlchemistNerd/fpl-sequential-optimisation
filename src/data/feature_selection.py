"""
Feature Selection — Chapter 3, Section 3.7.

Methods:
    - Filter: mutual information, correlation threshold
    - Embedded: Lasso (L1), tree-based importance (XGBoost / RandomForest)
    - Final feature set: ~15-20 features suitable for HDP-GMM
"""

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter methods (§3.7.1)
# ---------------------------------------------------------------------------


def mutual_information_select(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.01,
) -> list[str]:
    """
    Compute MI(X^{(d)}; Y) for each feature and discard those below *threshold*.

    MI(X; Y) = sum_{x,y} P(x,y) log(P(x,y) / (P(x) P(y)))
    """
    mi_scores = mutual_info_regression(X.values, y.values, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    selected = mi_series[mi_series >= threshold].index.tolist()
    logger.info(
        "Mutual information selection: %d / %d features retained (threshold=%.3f)",
        len(selected), len(X.columns), threshold,
    )
    for feat, score in mi_series.items():
        logger.debug("  MI(%s) = %.4f %s", feat, score, "✓" if score >= threshold else "✗")
    return selected


def correlation_threshold_select(
    X: pd.DataFrame,
    threshold: float = 0.9,
) -> list[str]:
    """
    For any pair with |r| > threshold, keep the more predictive feature
    (the one with higher variance as a proxy for information content).

    Returns the list of retained feature names.
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    to_drop: set[str] = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold]
        for other in high_corr.index:
            # Drop the one with lower variance
            if X[col].var() < X[other].var():
                to_drop.add(col)
            else:
                to_drop.add(other)

    retained = [c for c in X.columns if c not in to_drop]
    logger.info(
        "Correlation threshold selection: %d / %d features retained (|r| > %.2f dropped)",
        len(retained), len(X.columns), threshold,
    )
    return retained


# ---------------------------------------------------------------------------
# Embedded methods (§3.7.2)
# ---------------------------------------------------------------------------


def lasso_select(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> list[str]:
    """
    Fit Lasso (L1 regularisation):
        min_β  (1/2N) Σ (y_i - β^T x_i)^2 + λ ||β||_1

    Features with β^{(d)} = 0 are eliminated.
    λ is tuned via cross-validation.
    """
    lasso = LassoCV(cv=cv, random_state=42, max_iter=10000)
    lasso.fit(X.values, y.values)

    selected = X.columns[lasso.coef_ != 0].tolist()
    logger.info(
        "Lasso selection: %d / %d features retained (λ = %.4f)",
        len(selected), len(X.columns), lasso.alpha_,
    )
    for feat, coef in zip(X.columns, lasso.coef_):
        if coef != 0:
            logger.debug("  β(%s) = %.4f", feat, coef)
    return selected


def tree_importance_select(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 200,
    top_k: int | None = None,
) -> list[str]:
    """
    Train Gradient Boosting and rank by mean decrease in impurity.

    Returns the top *top_k* features (or all with non-zero importance).
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=4,
        random_state=42,
    )
    model.fit(X.values, y.values)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    if top_k is not None:
        selected = importances.head(top_k).index.tolist()
    else:
        selected = importances[importances > 0].index.tolist()

    logger.info(
        "Tree-based selection: %d / %d features retained",
        len(selected), len(X.columns),
    )
    for feat, imp in importances.head(15).items():
        logger.debug("  Importance(%s) = %.4f", feat, imp)
    return selected


# ---------------------------------------------------------------------------
# Final feature set (§3.7.3)
# ---------------------------------------------------------------------------

# The canonical final feature set after selection
FINAL_FEATURE_LIST: list[str] = [
    # Advanced (per 90)
    "xG_per_90", "xA_per_90", "npxG_per_90",
    "shots_per_90", "key_passes_per_90",
    # ICT (per 90)
    "influence_per_90", "creativity_per_90", "threat_per_90",
    # Form
    "Form_3", "Form_5", "Form_10",
    # Fixture
    "FDR", "RollingFDR_3",
    # Contextual
    "is_home", "rest_days",
    "Sentiment", "ownership_percent", "Injured",
]


def select_final_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "consensus",
) -> list[str]:
    """
    Apply multiple selection methods and return the consensus (intersection
    or union) of selected features.

    Parameters
    ----------
    X : DataFrame
        Feature matrix (already imputed).
    y : Series
        Target variable (total FPL points).
    method : str
        "consensus" → intersection of all methods (most conservative)
        "union" → union of all methods (most inclusive)
        "lasso" / "mi" / "tree" / "corr" → single method

    Returns
    -------
    List of selected feature names that exist in X.
    """
    available = [c for c in FINAL_FEATURE_LIST if c in X.columns]
    # Also consider all available numeric columns as candidates
    all_candidates = [c for c in X.select_dtypes(include=[np.number]).columns if c not in ["player_id", "gameweek"]]

    selections: dict[str, list[str]] = {}

    if method in ("consensus", "union", "mi"):
        selections["mi"] = mutual_information_select(X[all_candidates], y)
    if method in ("consensus", "union", "corr"):
        selections["corr"] = correlation_threshold_select(X[all_candidates])
    if method in ("consensus", "union", "lasso"):
        selections["lasso"] = lasso_select(X[all_candidates], y)
    if method in ("consensus", "union", "tree"):
        selections["tree"] = tree_importance_select(X[all_candidates], y)

    if not selections:
        logger.warning("No selection method applied; returning all available columns.")
        return all_candidates

    if method == "consensus":
        # Intersection of all sets
        selected = set(selections.popitem()[1])
        for s in selections.values():
            selected &= set(s)
    elif method == "union":
        selected = set()
        for s in selections.values():
            selected |= set(s)
    else:
        selected = set(selections.get(method, all_candidates))

    # Ensure we only return columns that actually exist
    final = [c for c in all_candidates if c in selected]
    logger.info(
        "Final feature set (%s): %d features",
        method, len(final),
    )
    return final
