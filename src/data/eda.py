"""
Exploratory Data Analysis (EDA) — Chapter 3, Section 3.6.

Analyses:
    - Univariate: distribution of points, prices, ICT by position
    - Bivariate: xG vs goals, correlation heatmap
    - Time series: rolling form trajectories, price-change dynamics
    - Position-specific patterns

All plotting functions save figures to *output_dir*.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, name: str, output_dir: Path) -> Path:
    """Save and close a figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved EDA figure: %s", path)
    return path


# ---------------------------------------------------------------------------
# Univariate (§3.6.1)
# ---------------------------------------------------------------------------


def plot_points_distribution(
    df: pd.DataFrame,
    points_col: str = "total_points",
    output_dir: Path = Path("reports/figures"),
) -> Path:
    """
    Distribution of total points per gameweek.
    Expected: heavily right-skewed (most 1-3 pts, long tail).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df[points_col], bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("FPL Points (per GW)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of FPL Points")

    # Log-scale to see the tail
    nonzero = df[df[points_col] > 0][points_col]
    axes[1].hist(np.log1p(nonzero), bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_xlabel("log(1 + FPL Points)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Log-scale Distribution")

    fig.suptitle("Univariate: FPL Points Distribution", fontsize=14)
    return _save(fig, "eda_points_distribution", output_dir)


def plot_price_distribution(
    df: pd.DataFrame,
    price_col: str = "now_cost",
    output_dir: Path = Path("reports/figures"),
) -> Path:
    """
    Player price distribution.
    Expected: bimodal (cheap backups vs expensive elites).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    prices = df[price_col] / 10.0  # convert tenths to £m
    ax.hist(prices, bins=40, edgecolor="black", alpha=0.7, color="green")
    ax.set_xlabel("Price (£m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Player Price Distribution (Bimodal)")
    fig.tight_layout()
    return _save(fig, "eda_price_distribution", output_dir)


def plot_ict_by_position(
    df: pd.DataFrame,
    position_col: str = "position",
    output_dir: Path = Path("reports/figures"),
) -> Path:
    """
    ICT Index distribution by position.
    Expected: Forwards/MIDs dominate upper tail; DEF/GK cluster low.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = ["GK", "DEF", "MID", "FWD"]
    for pos in positions:
        subset = df[df[position_col] == pos]
        if subset.empty:
            continue
        ict_col = "ict_index" if "ict_index" in subset.columns else "influence"
        ax.hist(subset[ict_col], bins=30, alpha=0.5, label=pos)
    ax.set_xlabel("ICT Index")
    ax.set_ylabel("Frequency")
    ax.set_title("ICT Index by Position")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "eda_ict_by_position", output_dir)


# ---------------------------------------------------------------------------
# Bivariate (§3.6.2)
# ---------------------------------------------------------------------------


def plot_xg_vs_goals(
    df: pd.DataFrame,
    output_dir: Path = Path("reports/figures"),
) -> Path:
    """
    Scatter: xG per 90 vs goals per 90.
    Players above diagonal: over-performing (clinical / lucky).
    Players below: under-performing.
    Expected correlation: r ≈ 0.6-0.7.
    """
    xg_col = "expected_goals_per_90" if "expected_goals_per_90" in df.columns else "xG_per_90"
    goals_col = "goals_scored_per_90"

    if xg_col not in df.columns or goals_col not in df.columns:
        logger.warning("Cannot plot xG vs goals: missing columns.")
        return Path("")

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = (df[xg_col] >= 0) & (df[goals_col] >= 0)
    ax.scatter(df.loc[mask, xg_col], df.loc[mask, goals_col], alpha=0.3, s=10)

    # Diagonal
    lims = [
        min(df.loc[mask, xg_col].min(), df.loc[mask, goals_col].min()),
        max(df.loc[mask, xg_col].max(), df.loc[mask, goals_col].max()),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, label="y = x")
    ax.set_xlabel("xG per 90")
    ax.set_ylabel("Goals per 90")
    ax.set_title("xG vs Actual Goals (per 90)")
    ax.legend()

    corr = df.loc[mask, xg_col].corr(df.loc[mask, goals_col])
    ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    return _save(fig, "eda_xg_vs_goals", output_dir)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    output_dir: Path = Path("reports/figures"),
) -> Path:
    """
    Pearson correlation matrix heatmap.
    Identifies multicollinearity: |r| > 0.9 → candidate for removal.
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    available = [c for c in feature_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        annot=False,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return _save(fig, "eda_correlation_heatmap", output_dir)


def find_highly_correlated_pairs(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.9,
) -> list[tuple[str, str, float]]:
    """
    Return pairs of features with |r| > threshold.
    """
    available = [c for c in feature_cols if c in df.columns]
    corr = df[available].corr()
    pairs: list[tuple[str, str, float]] = []
    for i, col_a in enumerate(available):
        for j, col_b in enumerate(available):
            if j <= i:
                continue
            r = corr.loc[col_a, col_b]
            if abs(r) > threshold:
                pairs.append((col_a, col_b, r))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for a, b, r in pairs:
        logger.info("Highly correlated: %s <-> %s (r = %.3f)", a, b, r)
    return pairs


# ---------------------------------------------------------------------------
# Time Series (§3.6.3)
# ---------------------------------------------------------------------------


def plot_form_trajectory(
    df: pd.DataFrame,
    player_id: int,
    form_col: str = "Form_5",
    output_dir: Path = Path("reports/figures"),
) -> Path:
    """
    Rolling 5-game average points for a single player.
    Reveals hot streaks and slumps.
    """
    player_df = df[df["player_id"] == player_id].sort_values("gameweek")
    if player_df.empty:
        logger.warning("No data for player %d", player_id)
        return Path("")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(player_df["gameweek"], player_df[form_col], marker="o", markersize=4)
    ax.set_xlabel("Gameweek")
    ax.set_ylabel(form_col)
    ax.set_title(f"Rolling Form Trajectory — Player {player_id}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, f"eda_form_trajectory_p{player_id}", output_dir)


def plot_price_change_dynamics(
    df: pd.DataFrame,
    player_id: int,
    output_dir: Path = Path("reports/figures"),
) -> Path:
    """
    Player price changes over time.
    Reveals the lag structure of the FPL pricing algorithm (2-4 GW lag).
    """
    player_df = df[df["player_id"] == player_id].sort_values("gameweek")
    if player_df.empty or "price_change" not in player_df.columns:
        return Path("")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(player_df["gameweek"], player_df["price_change"], width=0.8, alpha=0.7)
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Price Change (£m)")
    ax.set_title(f"Price Change Dynamics — Player {player_id}")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    return _save(fig, f"eda_price_dynamics_p{player_id}", output_dir)


# ---------------------------------------------------------------------------
# Position-specific patterns (§3.6.4)
# ---------------------------------------------------------------------------


def plot_position_violins(
    df: pd.DataFrame,
    value_col: str = "total_points",
    output_dir: Path = Path("reports/figures"),
) -> Path:
    """
    Violin plot of points by position.
    Shows: GK low variance, FWD high variance with occasional massive hauls.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=df, x="position", y=value_col, ax=ax, inner="quartile")
    ax.set_title("Points Distribution by Position")
    ax.set_ylabel("FPL Points")
    fig.tight_layout()
    return _save(fig, "eda_position_violins", output_dir)


# ---------------------------------------------------------------------------
# Full EDA report
# ---------------------------------------------------------------------------


def run_full_eda(
    df: pd.DataFrame,
    output_dir: Path = Path("reports/figures"),
    sample_players: list[int] | None = None,
) -> dict[str, Path]:
    """
    Run the complete EDA suite and save all figures.

    Returns a dict mapping figure names to saved file paths.
    """
    results: dict[str, Path] = {}

    # Univariate
    results["points_dist"] = plot_points_distribution(df, output_dir=output_dir)
    results["price_dist"] = plot_price_distribution(df, output_dir=output_dir)
    results["ict_by_position"] = plot_ict_by_position(df, output_dir=output_dir)

    # Bivariate
    results["xg_vs_goals"] = plot_xg_vs_goals(df, output_dir=output_dir)
    feature_cols = [c for c in df.columns if c.endswith("_per_90")]
    results["correlation"] = plot_correlation_heatmap(df, feature_cols, output_dir=output_dir)
    high_corr = find_highly_correlated_pairs(df, feature_cols)
    results["high_corr_count"] = Path(f"{len(high_corr)} pairs > 0.9")

    # Position-specific
    results["position_violins"] = plot_position_violins(df, output_dir=output_dir)

    # Time series (sample players)
    if sample_players is None:
        sample_players = df["player_id"].head(5).tolist()
    for pid in sample_players:
        path = plot_form_trajectory(df, pid, output_dir=output_dir)
        if path:
            results[f"form_p{pid}"] = path
        path = plot_price_change_dynamics(df, pid, output_dir=output_dir)
        if path:
            results[f"price_p{pid}"] = path

    logger.info("Full EDA complete: %d figures saved", len(results))
    return results
