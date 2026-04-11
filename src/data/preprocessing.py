"""
Data Preprocessing Pipeline — Chapter 3, Section 3.5.

Handles:
    - Missing-value imputation (§3.5.1)
    - Feature standardisation (§3.5.2)
    - Position-encoding (separate models per position group) (§3.5.3)
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

POSITIONS = ["GK", "DEF", "MID", "FWD"]


# ---------------------------------------------------------------------------
# Imputation (§3.5.1)
# ---------------------------------------------------------------------------


def impute_missing_values(
    df: pd.DataFrame,
    position_col: str = "position",
) -> pd.DataFrame:
    """
    Apply the imputation strategy:
        1. Zero-minutes players: all per-90 features → 0 (structural zeros).
        2. Numerical features with some minutes: median imputation within
           the player's position group.
        3. Sentiment: if no articles, Sentiment → 0, SentimentVol → 0.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'minutes' column and the target numeric columns.
    position_col : str
        Column name for position (GK/DEF/MID/FWD).

    Returns
    -------
    DataFrame with missing values filled.
    """
    df = df.copy()

    # 1. Structural zeros for 0-minute players
    zero_min_mask = df["minutes"] == 0
    per90_cols = [c for c in df.columns if c.endswith("_per_90")]
    df.loc[zero_min_mask, per90_cols] = 0.0

    # 2. Median imputation within position group for remaining NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col == "minutes":
            continue
        if df[col].isna().any():
            df[col] = df.groupby(position_col)[col].transform(
                lambda s: s.fillna(s.median())
            )
            # If entire position group is NaN, fall back to global median
            if df[col].isna().any():
                global_median = df[col].median()
                if not np.isnan(global_median):
                    df[col] = df[col].fillna(global_median)
                else:
                    # All values are NaN, fill with 0
                    df[col] = df[col].fillna(0.0)

    # 3. Sentiment defaults
    if "Sentiment" in df.columns:
        df["Sentiment"] = df["Sentiment"].fillna(0.0)
    if "SentimentVol" in df.columns:
        df["SentimentVol"] = df["SentimentVol"].fillna(0.0)

    logger.info("Imputation complete. Remaining NaNs: %d", df.isna().sum().sum())
    return df


# ---------------------------------------------------------------------------
# Standardisation (§3.5.2)
# ---------------------------------------------------------------------------


@dataclass
class FeatureStandardiser:
    """
    Wraps sklearn StandardScaler.  Computes μ_d and σ_d on the training set
    and applies them to all subsequent data to prevent data leakage.
    """

    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_names: list[str] = field(default_factory=list)
    _is_fit: bool = False

    def fit(self, X: pd.DataFrame) -> "FeatureStandardiser":
        """Fit on the training set only."""
        self.feature_names = list(X.columns)
        self.scaler.fit(X.values)
        self._is_fit = True
        logger.info(
            "Standardiser fitted on %d features (mean range: %.3f–%.3f)",
            len(self.feature_names),
            self.scaler.mean_.min(),  # type: ignore[union-attr]
            self.scaler.mean_.max(),  # type: ignore[union-attr]
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply standardisation."""
        if not self._is_fit:
            raise RuntimeError("Standardiser must be fitted before transform.")
        # Ensure columns match
        X_subset = X[self.feature_names]
        transformed = self.scaler.transform(X_subset.values)
        return pd.DataFrame(transformed, columns=self.feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# ---------------------------------------------------------------------------
# Position-based splitting (§3.5.3)
# ---------------------------------------------------------------------------


def split_by_position(
    df: pd.DataFrame,
    position_col: str = "position",
) -> dict[str, pd.DataFrame]:
    """
    Split the dataset into separate DataFrames per position group.

    Returns a dict: {"GK": df_gk, "DEF": df_def, "MID": df_mid, "FWD": df_fwd}
    """
    result: dict[str, pd.DataFrame] = {}
    for pos in POSITIONS:
        pos_df = df[df[position_col] == pos].copy()
        result[pos] = pos_df
        logger.info("Position %s: %d rows", pos, len(pos_df))
    return result


def build_position_standardisers(
    position_dfs: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> dict[str, FeatureStandardiser]:
    """
    Fit a separate FeatureStandardiser for each position group.

    This is equivalent to conditioning the model on position:
        P(x | z=k) = P(x | z=k, position(p))
    """
    standardisers: dict[str, FeatureStandardiser] = {}
    for pos, pos_df in position_dfs.items():
        if pos_df.empty:
            logger.warning("Position %s has no data; skipping standardiser.", pos)
            continue
        scaler = FeatureStandardiser()
        scaler.fit(pos_df[feature_cols])
        standardisers[pos] = scaler
    return standardisers
