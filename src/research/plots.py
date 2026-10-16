"""Publication-ready plots for walk-forward prediction and squad ledgers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "rolling_form": "Rolling form",
    "hdp_gmm": "HDP-GMM",
    "hdp_hmm": "HDP-HMM",
    "direct_ridge": "Direct Ridge",
    "form_milp": "Rolling-form MILP",
    "hdp_gmm_milp": "HDP-GMM MILP",
    "hdp_hmm_milp": "HDP-HMM MILP",
    "direct_ridge_milp": "Direct Ridge MILP",
}
MODEL_COLORS = {
    "rolling_form": "#0072B2",
    "hdp_gmm": "#D55E00",
    "hdp_hmm": "#009E73",
    "direct_ridge": "#CC79A7",
    "form_milp": "#0072B2",
    "hdp_gmm_milp": "#D55E00",
    "hdp_hmm_milp": "#009E73",
    "direct_ridge_milp": "#CC79A7",
}


def plot_prediction_artifacts(output_dir: str | Path) -> list[Path]:
    """Create forecast charts from saved metrics, without requiring a replay."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_csv(output_dir / "prediction_week_metrics.csv")
    positions = pd.read_csv(output_dir / "prediction_position_metrics.csv")

    return [
        _plot_weekly_errors(weekly, figures_dir / "prediction_error_by_gameweek.png"),
        _plot_weekly_rank(weekly, figures_dir / "prediction_rank_correlation_by_gameweek.png"),
        _plot_position_error(positions, figures_dir / "prediction_mae_by_position.png"),
    ]


def plot_study_artifacts(output_dir: str | Path) -> list[Path]:
    """Create forecast and squad charts from persisted evaluation artifacts.

    Inputs are only the saved CSV artifacts, so plots can be regenerated
    without fitting models or joining new outcomes.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    paths = plot_prediction_artifacts(output_dir)
    squad = pd.read_csv(output_dir / "squad_gameweeks.csv")
    paths.append(_plot_cumulative_squad(squad, figures_dir / "cumulative_squad_points.png"))
    return paths


def _style_axis(axis: plt.Axes, *, gameweeks: list[int]) -> None:
    axis.set_xlabel("Target gameweek")
    axis.set_xticks(gameweeks)
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_weekly_errors(weekly: pd.DataFrame, destination: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True, constrained_layout=True)
    gameweeks = sorted(weekly["gameweek"].astype(int).unique())
    interval = max(1, len(gameweeks) // 12)
    ticks = gameweeks[::interval]
    for axis, metric, title in zip(
        axes,
        ("mae", "rmse"),
        ("Mean absolute error", "Root mean squared error"),
    ):
        for model, rows in weekly.groupby("model", sort=False):
            rows = rows.sort_values("gameweek")
            axis.plot(
                rows["gameweek"],
                rows[metric],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.set_ylabel("FPL points")
        axis.set_title(title, loc="left", fontsize=11, weight="semibold")
        _style_axis(axis, gameweeks=ticks)
    axes[0].legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.suptitle("Next-gameweek forecast error", fontsize=15, weight="bold")
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def _plot_weekly_rank(weekly: pd.DataFrame, destination: Path) -> Path:
    fig, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    gameweeks = sorted(weekly["gameweek"].astype(int).unique())
    interval = max(1, len(gameweeks) // 12)
    for model, rows in weekly.groupby("model", sort=False):
        rows = rows.sort_values("gameweek")
        axis.plot(
            rows["gameweek"],
            rows["spearman"],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axis.axhline(0, color="#656D76", linewidth=0.9, linestyle="--")
    axis.set_ylabel("Spearman correlation")
    axis.set_title("Player ranking agreement by target gameweek", loc="left", fontsize=11, weight="semibold")
    _style_axis(axis, gameweeks=gameweeks[::interval])
    axis.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def _plot_position_error(positions: pd.DataFrame, destination: Path) -> Path:
    position_order = [position for position in ("GK", "DEF", "MID", "FWD") if position in set(positions["position"])]
    models = [
        model
        for model in ("rolling_form", "direct_ridge", "hdp_gmm", "hdp_hmm")
        if model in set(positions["model"])
    ]
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    positions = positions.copy()
    positions["position"] = pd.Categorical(positions["position"], position_order, ordered=True)
    width = 0.24
    x = range(len(position_order))
    for model_index, model in enumerate(models):
        values = (
            positions.loc[positions["model"] == model]
            .set_index("position")["mae"]
            .reindex(position_order)
        )
        offsets = [value + (model_index - (len(models) - 1) / 2) * width for value in x]
        axis.bar(
            offsets,
            values,
            width=width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    axis.set_xticks(list(x), position_order)
    axis.set_xlabel("Player position")
    axis.set_ylabel("Mean absolute error (FPL points)")
    axis.set_title("Forecast error by position", loc="left", fontsize=12, weight="semibold")
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=len(models))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def _plot_cumulative_squad(squad: pd.DataFrame, destination: Path) -> Path:
    evaluation = squad.loc[squad["is_evaluation_gameweek"].astype(bool)].copy()
    fig, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    for (seed, strategy), rows in evaluation.groupby(["seed", "strategy"], sort=False):
        rows = rows.sort_values("gameweek")
        axis.plot(
            rows["gameweek"],
            rows["evaluation_net_points"].cumsum(),
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=MODEL_COLORS[strategy],
            label=f"{MODEL_LABELS[strategy]} (seed {seed})",
        )
    gameweeks = sorted(evaluation["gameweek"].astype(int).unique())
    interval = max(1, len(gameweeks) // 12)
    axis.set_ylabel("Cumulative net points")
    axis.set_title("Constrained squad replay", loc="left", fontsize=12, weight="semibold")
    _style_axis(axis, gameweeks=gameweeks[::interval])
    axis.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination
