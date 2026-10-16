"""Walk-forward point forecasts from lagged form, HDP-GMM, and HDP-HMM.

Each fold creates predictor snapshots using only gameweeks before its target.
Historical target points are used for fitting only after those gameweeks have
completed. Forecast rows are written and flushed before the scoring stage reads
the target outcomes.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from src.backtesting.data import build_as_of_features, outcomes_for_gameweek
from src.backtesting.engine import run_backtest, sha256_file
from src.models.hdp_gmm import run_hdp_gmm
from src.models.hdp_hmm import run_hdp_hmm
from src.research.plots import plot_study_artifacts


BASE_FEATURES = (
    "form_points",
    "form_std",
    "xgi_points",
    "appearance_probability",
    "expected_minutes",
)
POSITION_FEATURES = ("pos_GK", "pos_DEF", "pos_MID", "pos_FWD")
MODEL_FEATURES = BASE_FEATURES + POSITION_FEATURES
PREDICTION_COLUMNS = ("rolling_form", "hdp_gmm", "hdp_hmm")
MODEL_STRATEGIES = ("form_milp", "hdp_gmm_milp", "hdp_hmm_milp")


@dataclass(frozen=True)
class FoldPanels:
    """Training snapshots/labels and the target-GW candidates for one fold."""

    target_gameweek: int
    train: pd.DataFrame
    target: pd.DataFrame


def _with_model_features(players: pd.DataFrame) -> pd.DataFrame:
    """Add role indicators and ensure the fixed numerical model schema."""
    result = players.copy()
    for position, column in zip(("GK", "DEF", "MID", "FWD"), POSITION_FEATURES):
        result[column] = (result["position"] == position).astype(float)
    for column in MODEL_FEATURES:
        if column not in result:
            result[column] = 0.0
    result.loc[:, list(MODEL_FEATURES)] = (
        result.loc[:, list(MODEL_FEATURES)].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    return result


def _json_safe(value: Any) -> Any:
    """Convert metadata to strict JSON values, representing non-finite values as null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Atomically write strict JSON metadata so partial files are not mistaken for results."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_safe(metadata), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _runtime_versions() -> dict[str, str]:
    """Record the runtime and numerical packages that affect model results."""
    import jax

    packages = ("jax", "jaxlib", "numpyro", "numpy", "pandas", "scipy", "scikit-learn")
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    versions["jax_local_device_count"] = str(jax.local_device_count())
    return versions


def build_fold_panels(
    gameweeks: pd.DataFrame,
    target_gameweek: int,
    *,
    lookback: int = 5,
    max_stale_gameweeks: int = 5,
) -> FoldPanels:
    """Build a leakage-safe fold without reading target-GW outcomes.

    Training feature rows are point-in-time snapshots from GW2 through
    ``target_gameweek - 1``. Their labels are the corresponding realized
    points. The target panel contains only pre-GW features and no target label.
    The archive lacks official availability snapshots, so player eligibility
    remains the documented recent-observation proxy.
    """
    if target_gameweek < 3:
        raise ValueError("Walk-forward model fitting requires a target of GW3 or later")
    max_gw = int(gameweeks["gameweek"].max())
    train_records: list[pd.DataFrame] = []
    for history_gw in range(2, target_gameweek):
        historical = build_as_of_features(
            gameweeks,
            history_gw,
            lookback=lookback,
            max_stale_gameweeks=max_gw + 1,
        )
        outcomes = outcomes_for_gameweek(gameweeks, history_gw)
        historical["gameweek"] = history_gw
        historical["actual_points"] = historical["player_id"].map(
            {player_id: values["points"] for player_id, values in outcomes.items()}
        ).fillna(0.0)
        train_records.append(_with_model_features(historical))

    target = build_as_of_features(
        gameweeks,
        target_gameweek,
        lookback=lookback,
        max_stale_gameweeks=max_stale_gameweeks,
    )
    return FoldPanels(
        target_gameweek=target_gameweek,
        train=pd.concat(train_records, ignore_index=True),
        target=_with_model_features(target),
    )


def choose_fit_roster(
    gameweeks: pd.DataFrame,
    *,
    players_per_position: int = 12,
    seed: int = 42,
) -> list[int]:
    """Choose a stratified, reproducible model roster from GW1 information."""
    if players_per_position < 1:
        raise ValueError("players_per_position must be at least one")
    gw1 = gameweeks.loc[gameweeks["gameweek"] == 1, ["player_id", "position"]]
    gw1 = gw1.drop_duplicates("player_id").sort_values(["position", "player_id"])
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for position in ("GK", "DEF", "MID", "FWD"):
        ids = gw1.loc[gw1["position"] == position, "player_id"].to_numpy(dtype=int)
        if len(ids) == 0:
            raise ValueError(f"GW1 does not contain any {position} players")
        count = min(players_per_position, len(ids))
        selected.extend(rng.choice(ids, size=count, replace=False).astype(int).tolist())
    return sorted(selected)


def _numpy_stick_breaking(raw: np.ndarray) -> np.ndarray:
    """Map raw break proportions to normalized component weights."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:
        raw = raw[None, :]
    breaks = np.clip(raw, 1e-8, 1.0 - 1e-8)
    remaining_before = np.concatenate(
        [np.ones((*breaks.shape[:-1], 1)), np.cumprod(1.0 - breaks, axis=-1)[..., :-1]],
        axis=-1,
    )
    leading = breaks * remaining_before
    residual = np.prod(1.0 - breaks, axis=-1, keepdims=True)
    return np.concatenate([leading, residual], axis=-1)


def _gaussian_log_density(
    observations: np.ndarray,
    means: np.ndarray,
    precisions: np.ndarray,
) -> np.ndarray:
    """Log N(x | mu, precision^-1) for rows x and all component states."""
    observations = np.asarray(observations, dtype=float)
    means = np.asarray(means, dtype=float)
    precisions = np.asarray(precisions, dtype=float)
    dimension = observations.shape[1]
    signs, log_determinants = np.linalg.slogdet(precisions)
    if np.any(signs <= 0):
        raise ValueError("HDP posterior contains a non-positive precision matrix")
    differences = observations[:, None, :] - means[None, :, :]
    quadratic = np.einsum("nkd,kde,nke->nk", differences, precisions, differences)
    return 0.5 * log_determinants[None, :] - 0.5 * dimension * np.log(2.0 * np.pi) - 0.5 * quadratic


def _responsibilities(
    observations: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    precisions: np.ndarray,
) -> np.ndarray:
    log_weights = np.log(np.clip(np.asarray(weights, dtype=float), 1e-12, None))
    log_joint = _gaussian_log_density(observations, means, precisions) + log_weights
    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


def _state_rewards(responsibilities: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Estimate held-out-point reward by state with global-mean smoothing."""
    points = np.asarray(points, dtype=float)
    global_mean = float(points.mean()) if len(points) else 0.0
    state_mass = responsibilities.sum(axis=0)
    smoothing = 2.0
    return (responsibilities.T @ points + smoothing * global_mean) / (state_mass + smoothing)


def predict_hdp_gmm(
    samples: dict[str, np.ndarray],
    train_x: np.ndarray,
    train_codes: np.ndarray,
    train_points: np.ndarray,
    target_x: np.ndarray,
    target_player_ids: np.ndarray,
    player_to_code: dict[int, int],
) -> np.ndarray:
    """Map the GMM posterior state mixture to next-GW points.

    State-to-points means are learned only from earlier realized outcomes.
    Players outside the fixed GW1 fit roster use the posterior global mixture.
    """
    beta_weights = _numpy_stick_breaking(samples["beta_raw"])
    local_weights = np.asarray(samples["player_weights"], dtype=float)
    draws = len(beta_weights)
    train_membership = np.zeros((len(train_x), samples["mu"].shape[1]), dtype=float)
    target_membership = np.zeros(
        (len(target_x), samples["mu"].shape[1]), dtype=float
    )
    for draw in range(draws):
        means = np.asarray(samples["mu"][draw], dtype=float)
        precisions = np.asarray(samples["precision"][draw], dtype=float)
        train_mix = local_weights[draw, train_codes]
        train_membership += _responsibilities(
            train_x, train_mix, means, precisions
        ) / draws
        target_mix = np.vstack(
            [
                local_weights[draw, player_to_code[int(player_id)]]
                if int(player_id) in player_to_code
                else beta_weights[draw]
                for player_id in target_player_ids
            ]
        )
        target_membership += _responsibilities(
            target_x, target_mix, means, precisions
        ) / draws
    rewards = _state_rewards(train_membership, train_points)
    return target_membership @ rewards


def _forward_state_probabilities(
    observations: np.ndarray,
    means: np.ndarray,
    precisions: np.ndarray,
    global_weights: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    """Filtered HMM state probabilities, one row per observed week."""
    log_emissions = _gaussian_log_density(observations, means, precisions)
    return _forward_from_log_emissions(log_emissions, global_weights, transition)


def _forward_from_log_emissions(
    log_emissions: np.ndarray,
    global_weights: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    """Filter a sequence from precomputed Gaussian log-emission values."""
    filtered = np.zeros_like(log_emissions)
    log_state = np.log(np.clip(global_weights, 1e-12, None)) + log_emissions[0]
    current = np.exp(log_state - logsumexp(log_state))
    filtered[0] = current
    for index in range(1, len(log_emissions)):
        prior = current @ transition
        log_state = np.log(np.clip(prior, 1e-12, None)) + log_emissions[index]
        current = np.exp(log_state - logsumexp(log_state))
        filtered[index] = current
    return filtered


def predict_hdp_hmm(
    samples: dict[str, np.ndarray],
    train: pd.DataFrame,
    train_x: np.ndarray,
    train_points: np.ndarray,
    target_x: np.ndarray,
    target_player_ids: np.ndarray,
    *,
    filter_history: pd.DataFrame | None = None,
    filter_history_x: np.ndarray | None = None,
) -> np.ndarray:
    """Filter each player through observed feature history and one target row."""
    ordered = train.reset_index(drop=True)
    grouped_indices = {
        int(player_id): group.index.to_numpy(dtype=int)
        for player_id, group in ordered.groupby("player_id", sort=False)
    }
    player_order = {
        int(player_id): group["gameweek"].to_numpy(dtype=int)
        for player_id, group in ordered.groupby("player_id", sort=False)
    }
    for player_id, weeks in player_order.items():
        if len(weeks) > 1 and not np.all(np.diff(weeks) == 1):
            raise ValueError(f"HDP-HMM training sequence for player {player_id} has gaps")
    history = ordered if filter_history is None else filter_history.reset_index(drop=True)
    history_x = train_x if filter_history_x is None else filter_history_x
    if len(history) != len(history_x):
        raise ValueError("filter_history_x must align one-to-one with filter_history rows")
    history_groups = {
        int(player_id): group.index.to_numpy(dtype=int)
        for player_id, group in history.groupby("player_id", sort=False)
    }
    history_weeks = {
        int(player_id): group["gameweek"].to_numpy(dtype=int)
        for player_id, group in history.groupby("player_id", sort=False)
    }
    for player_id, weeks in history_weeks.items():
        if len(weeks) > 1 and not np.all(np.diff(weeks) == 1):
            raise ValueError(f"HDP-HMM filter history for player {player_id} has gaps")

    state_count = samples["mu"].shape[1]
    train_membership = np.zeros((len(train_x), state_count), dtype=float)
    target_membership = np.zeros((len(target_x), state_count), dtype=float)
    draws = len(samples["beta_raw"])
    for draw in range(draws):
        beta = _numpy_stick_breaking(samples["beta_raw"][draw])[0]
        transition = np.asarray(samples["trans_probs"][draw, 0], dtype=float)
        means = np.asarray(samples["mu"][draw], dtype=float)
        precisions = np.asarray(samples["precision"][draw], dtype=float)
        train_log_emissions = _gaussian_log_density(train_x, means, precisions)
        history_log_emissions = _gaussian_log_density(history_x, means, precisions)
        target_log_emissions = _gaussian_log_density(target_x, means, precisions)

        for player_id, indices in grouped_indices.items():
            filtered = _forward_from_log_emissions(
                train_log_emissions[indices], beta, transition
            )
            train_membership[indices] += filtered / draws

        for index, player_id_value in enumerate(target_player_ids):
            player_id = int(player_id_value)
            if player_id in history_groups:
                indices = history_groups[player_id]
                history_belief = _forward_from_log_emissions(
                    history_log_emissions[indices], beta, transition
                )[-1]
                prior = history_belief @ transition
            else:
                # No fitted sequence for this player: use the global initial state.
                prior = beta
            target_log = np.log(np.clip(prior, 1e-12, None)) + target_log_emissions[index]
            target_membership[index] += np.exp(target_log - logsumexp(target_log)) / draws

    rewards = _state_rewards(train_membership, train_points)
    return target_membership @ rewards


def _fit_gmm_arrays(
    train: pd.DataFrame,
    train_x: np.ndarray,
    fit_roster: list[int],
    max_timesteps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    player_to_code = {player_id: code for code, player_id in enumerate(fit_roster)}
    dimension = train_x.shape[1]
    padded = np.zeros((len(fit_roster), max_timesteps, dimension), dtype=np.float32)
    mask = np.zeros((len(fit_roster), max_timesteps), dtype=np.float32)
    for row_index, row in train.reset_index(drop=True).iterrows():
        code = player_to_code[int(row.player_id)]
        offset = int(row.gameweek) - 2
        if not 0 <= offset < max_timesteps:
            raise ValueError(f"Unexpected training gameweek {row.gameweek}")
        padded[code, offset] = train_x[row_index]
        mask[code, offset] = 1.0
    ids = np.repeat(np.arange(len(fit_roster), dtype=np.int32), max_timesteps)
    return padded.reshape(-1, dimension), mask.reshape(-1), ids, player_to_code


def _metrics_rows(scored: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_key, group in scored.groupby(grouper, sort=True):
        key = group_key if isinstance(group_key, tuple) else (group_key,)
        predictions = group["prediction"].to_numpy(dtype=float)
        actual = group["actual_points"].to_numpy(dtype=float)
        rho = (
            spearmanr(predictions, actual).statistic
            if len(group) > 1 and np.ptp(predictions) > 0 and np.ptp(actual) > 0
            else np.nan
        )
        record = dict(zip(group_columns, key))
        record.update(
            n=int(len(group)),
            mae=float(np.mean(np.abs(predictions - actual))),
            rmse=float(np.sqrt(np.mean(np.square(predictions - actual)))),
            mean_error=float(np.mean(predictions - actual)),
            spearman=float(rho) if np.isfinite(rho) else np.nan,
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _sampler_diagnostic_record(
    *,
    gameweek: int,
    model: str,
    training_rows: int,
    fit_players: int,
    summary: dict[str, Any],
    retained_draws: int,
    max_divergence_rate: float,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    target_accept_prob: float,
) -> dict[str, Any]:
    """Summarize one fit and screen severe NUTS failures before forecasting."""
    divergences = int(summary["num_divergences"])
    divergence_rate = divergences / retained_draws
    mean_accept_prob = float(summary["mean_accept_prob"])
    passed = divergence_rate <= max_divergence_rate and mean_accept_prob >= 0.5
    return {
        "gameweek": gameweek,
        "model": model,
        "training_rows": training_rows,
        "fit_players": fit_players,
        "active_states": summary["active_clusters"],
        "num_divergences": divergences,
        "divergence_rate": divergence_rate,
        "mean_accept_prob": mean_accept_prob,
        "diagnostic_gate_passed": passed,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "target_accept_prob": target_accept_prob,
        "max_divergence_rate": max_divergence_rate,
    }


def _require_sampler_diagnostics(
    diagnostic: dict[str, Any],
    *,
    diagnostics_rows: list[dict[str, Any]],
    diagnostics_path: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
) -> None:
    """Persist failed-fit evidence and stop before writing its forecasts."""
    pd.DataFrame(diagnostics_rows).to_csv(diagnostics_path, index=False)
    if diagnostic["diagnostic_gate_passed"]:
        return
    metadata["status"] = "sampling_diagnostics_failed"
    metadata["failure"] = {
        "gameweek": diagnostic["gameweek"],
        "model": diagnostic["model"],
        "reason": "NUTS divergence rate or mean acceptance probability failed the configured screening gate",
        "diagnostic": diagnostic,
    }
    _write_metadata(metadata_path, metadata)
    raise RuntimeError(
        f"{diagnostic['model']} GW{diagnostic['gameweek']} failed NUTS diagnostics; "
        "the forecast fold was not written"
    )


def _load_resume_state(
    *,
    ledger_path: Path,
    diagnostics_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, Any],
    header: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], int]:
    """Load a contiguous, screened forecast prefix for an explicit resume."""
    required_paths = (ledger_path, diagnostics_path, metadata_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise ValueError(f"Cannot resume; missing run artifacts: {missing_paths}")

    previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    previous_status = previous_metadata.get("status")
    retrying_failed_fit = previous_status == "sampling_diagnostics_failed"
    compared_keys = (
        "season",
        "source_file_name",
        "source_url",
        "source_sha256",
        "source_coverage",
        "first_gameweek",
        "feature_lookback_gameweeks",
        "max_stale_gameweeks",
        "fit_roster_selected_from_gw1",
        "players_per_position",
        "seed",
        "num_samples",
        "num_chains",
        "truncation",
        "model_features",
    )
    if not retrying_failed_fit:
        compared_keys += ("num_warmup", "nuts_target_accept_prob")
    for key in compared_keys:
        if previous_metadata.get(key) != expected_metadata.get(key):
            raise ValueError(
                f"Cannot resume; saved configuration differs for {key}: "
                f"{previous_metadata.get(key)!r} != {expected_metadata.get(key)!r}"
            )
    previous_gate = previous_metadata.get("max_divergence_rate")
    if previous_gate is None:
        previous_gate = previous_metadata.get("diagnostic_gate", {}).get(
            "maximum_divergence_rate"
        )
    if previous_gate != expected_metadata.get("max_divergence_rate"):
        raise ValueError("Cannot resume; saved maximum divergence rate differs")

    previous_last = int(previous_metadata.get("last_forecast_gameweek_written", 0))
    first_gameweek = int(expected_metadata["first_gameweek"])
    requested_last = int(expected_metadata["last_gameweek"])
    saved_last = int(previous_metadata.get("last_gameweek", previous_last))
    if previous_last < first_gameweek or previous_last > requested_last:
        raise ValueError("Cannot resume; saved forecast prefix is outside the requested range")
    if saved_last > requested_last:
        raise ValueError("Cannot resume; saved run extends beyond the requested last gameweek")

    predictions = pd.read_csv(ledger_path)
    if list(predictions.columns) != header:
        raise ValueError("Cannot resume; forecast ledger columns do not match this runner")
    expected_weeks = list(range(first_gameweek, previous_last + 1))
    saved_weeks = sorted(predictions["gameweek"].astype(int).unique().tolist())
    if saved_weeks != expected_weeks:
        raise ValueError(
            f"Cannot resume; forecast ledger is not a contiguous prefix: {saved_weeks}"
        )
    if predictions.duplicated(["gameweek", "player_id"]).any():
        raise ValueError("Cannot resume; forecast ledger contains duplicate player-week rows")
    if not (
        predictions["feature_cutoff_gameweek"].astype(int)
        == predictions["gameweek"].astype(int) - 1
    ).all():
        raise ValueError("Cannot resume; forecast ledger contains an invalid feature cutoff")
    if not np.isfinite(predictions.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Cannot resume; forecast ledger contains non-finite predictions")

    diagnostics = pd.read_csv(diagnostics_path)
    expected_models = {"hdp_gmm", "hdp_hmm"}
    expected_draws = int(expected_metadata["num_samples"]) * int(
        expected_metadata["num_chains"]
    )
    diagnostic_weeks = diagnostics["gameweek"].astype(int)
    orphan_mask = diagnostic_weeks > previous_last
    trim_orphan_diagnostics = orphan_mask.any()
    failed_attempt_rows: list[dict[str, Any]] = []
    if retrying_failed_fit and not trim_orphan_diagnostics:
        raise ValueError("Cannot retry; saved sampler failure has no orphan diagnostic rows")
    if trim_orphan_diagnostics:
        orphan_diagnostics = diagnostics.loc[orphan_mask]
        orphan_weeks = set(orphan_diagnostics["gameweek"].astype(int))
        orphan_models = orphan_diagnostics["model"].astype(str)
        if (
            previous_status not in {"running", "sampling_diagnostics_failed"}
            or previous_last >= requested_last
            or orphan_weeks != {previous_last + 1}
            or len(orphan_diagnostics) > len(expected_models)
            or orphan_models.duplicated().any()
            or not set(orphan_models).issubset(expected_models)
        ):
            raise ValueError(
                "Cannot resume; diagnostics extend beyond the saved forecast prefix"
            )
        if retrying_failed_fit:
            failure = previous_metadata.get("failure", {})
            failure_gameweek = int(failure.get("gameweek", -1))
            failure_model = str(failure.get("model", ""))
            failed_rows = orphan_diagnostics.loc[
                (orphan_diagnostics["model"].astype(str) == failure_model)
                & ~orphan_diagnostics["diagnostic_gate_passed"].astype(bool)
            ]
            if (
                failure_gameweek != previous_last + 1
                or failure_model not in expected_models
                or len(failed_rows) != 1
                or float(failed_rows.iloc[0]["num_divergences"])
                != float(failure.get("diagnostic", {}).get("num_divergences", -1))
            ):
                raise ValueError(
                    "Cannot retry; saved sampler failure does not match the orphan diagnostics"
                )
            failed_attempt_rows = orphan_diagnostics.to_dict(orient="records")
        diagnostics = diagnostics.loc[~orphan_mask].copy()

    # Older checkpoints predate per-fold sampler settings. Backfill from their
    # recorded run configuration before a resumed run adds rows under new settings.
    legacy_sampler_settings = {
        "num_warmup": "num_warmup",
        "num_samples": "num_samples",
        "num_chains": "num_chains",
        "target_accept_prob": "nuts_target_accept_prob",
        "max_divergence_rate": "max_divergence_rate",
    }
    for column, metadata_key in legacy_sampler_settings.items():
        if column not in diagnostics:
            diagnostics[column] = previous_metadata.get(metadata_key)
        else:
            diagnostics[column] = diagnostics[column].fillna(
                previous_metadata.get(metadata_key)
            )

    for gameweek in expected_weeks:
        fold = diagnostics.loc[diagnostics["gameweek"].astype(int) == gameweek]
        if set(fold["model"].astype(str)) != expected_models or len(fold) != 2:
            raise ValueError(
                f"Cannot resume; GW{gameweek} lacks one diagnostic row per model"
            )
        for row in fold.to_dict(orient="records"):
            divergence_rate = float(row.get("divergence_rate", row["num_divergences"] / expected_draws))
            mean_accept_prob = float(row["mean_accept_prob"])
            gate_passed = bool(row.get("diagnostic_gate_passed", False))
            if (
                not gate_passed
                or divergence_rate > float(expected_metadata["max_divergence_rate"])
                or mean_accept_prob < 0.5
            ):
                raise ValueError(
                    f"Cannot resume; GW{gameweek} {row['model']} did not pass the saved diagnostic gate"
                )
    if set(diagnostics["gameweek"].astype(int)) != set(expected_weeks):
        raise ValueError("Cannot resume; diagnostic rows include an incomplete or failed fold")
    if trim_orphan_diagnostics:
        diagnostic_tmp_path = diagnostics_path.with_name(
            f".{diagnostics_path.name}.resume-tmp"
        )
        diagnostics.to_csv(diagnostic_tmp_path, index=False)
        os.replace(diagnostic_tmp_path, diagnostics_path)

    if retrying_failed_fit:
        retry_history = list(previous_metadata.get("sampler_retry_history", []))
        retry_history.append(
            {
                "gameweek": previous_last + 1,
                "failure": previous_metadata.get("failure"),
                "failed_sampler_settings": {
                    key: previous_metadata.get(key)
                    for key in (
                        "num_warmup",
                        "num_samples",
                        "num_chains",
                        "nuts_target_accept_prob",
                        "max_divergence_rate",
                    )
                },
                "failed_diagnostics": failed_attempt_rows,
                "retry_sampler_settings": {
                    key: expected_metadata.get(key)
                    for key in (
                        "num_warmup",
                        "num_samples",
                        "num_chains",
                        "nuts_target_accept_prob",
                        "max_divergence_rate",
                    )
                },
            }
        )
        previous_metadata["sampler_retry_history"] = retry_history

    return (
        previous_metadata,
        predictions,
        diagnostics.to_dict(orient="records"),
        previous_last,
    )


def generate_forecast_ledger(
    gameweeks: pd.DataFrame,
    *,
    output_dir: str | Path,
    season: str,
    source_path: str | Path,
    first_gameweek: int = 7,
    last_gameweek: int | None = None,
    lookback: int = 5,
    max_stale_gameweeks: int = 5,
    players_per_position: int = 12,
    seed: int = 42,
    num_warmup: int = 50,
    num_samples: int = 50,
    num_chains: int = 2,
    target_accept_prob: float = 0.95,
    max_divergence_rate: float = 0.01,
    truncation: int = 6,
    resume: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit sequential folds and durably write each forecast before scoring.

    The file `predictions.csv` is appended and flushed once per target GW. No
    outcome lookup for that target occurs in this function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_gameweek = int(last_gameweek or gameweeks["gameweek"].max())
    if first_gameweek < 7:
        raise ValueError("first_gameweek must leave at least five completed training GWs")
    if last_gameweek < first_gameweek:
        raise ValueError("last_gameweek must be at least first_gameweek")
    if last_gameweek > int(gameweeks["gameweek"].max()):
        raise ValueError("last_gameweek is outside the source archive")
    if num_warmup < 1 or num_samples < 2 or num_chains < 1 or truncation < 2:
        raise ValueError("MCMC settings require warmup>=1, samples>=2, chains>=1, truncation>=2")
    if not 0 <= max_divergence_rate < 1:
        raise ValueError("max_divergence_rate must be in [0, 1)")

    source_hash = sha256_file(source_path)
    max_gw = int(gameweeks["gameweek"].max())
    fit_roster = choose_fit_roster(
        gameweeks, players_per_position=players_per_position, seed=seed
    )
    max_training_timesteps = last_gameweek - 2
    ledger_path = output_dir / "predictions.csv"
    diagnostics_path = output_dir / "fit_diagnostics.csv"
    metadata_path = output_dir / "run_metadata.json"
    metadata: dict[str, Any] = {
        "status": "running",
        "season": season,
        "source_file_name": Path(source_path).name,
        "source_url": (
            "https://github.com/vaastav/Fantasy-Premier-League/blob/master/"
            f"data/{season}/gws/merged_gw.csv"
        ),
        "source_sha256": source_hash,
        "source_coverage": {
            "normalized_player_gameweek_rows": int(len(gameweeks)),
            "unique_players": int(gameweeks["player_id"].nunique()),
            "first_gameweek": int(gameweeks["gameweek"].min()),
            "last_gameweek": int(gameweeks["gameweek"].max()),
        },
        "first_gameweek": first_gameweek,
        "last_gameweek": last_gameweek,
        "maximum_training_sequence_length": max_training_timesteps,
        "training_padding_rule": "each fold is padded only to the latest completed training week",
        "feature_cutoff_rule": "strictly before each target gameweek",
        "feature_lookback_gameweeks": lookback,
        "max_stale_gameweeks": max_stale_gameweeks,
        "fit_roster_selected_from_gw1": fit_roster,
        "players_per_position": players_per_position,
        "seed": seed,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "nuts_target_accept_prob": target_accept_prob,
        "max_divergence_rate": max_divergence_rate,
        "chain_method_requested": "parallel",
        "truncation": truncation,
        "runtime_versions": _runtime_versions(),
        "inference": {
            "sampler": "NumPyro NUTS",
            "gmm_likelihood": "exact finite-mixture marginal; categorical states integrated out",
            "hmm_likelihood": "masked forward algorithm",
        },
        "model_features": list(MODEL_FEATURES),
        "caveats": [
            "Vaastav merged rows proxy information available before each deadline; no official dated deadline snapshots are present.",
            "HDP models are fitted to a reproducible GW1-stratified player cohort; other candidates use population-level state weights.",
            "The HMM uses one neutral action category because historical manager-action sequences are unavailable.",
            "State-conditioned points are estimated from earlier realized player-gameweek outcomes, not from a direct points-emission model.",
            "One archived season is a pilot and cannot establish cross-season generalization.",
        ],
    }
    header = [
        "season", "seed", "gameweek", "feature_cutoff_gameweek", "player_id",
        "position", "price", "candidate_fresh", *PREDICTION_COLUMNS,
        "fit_roster_member", "training_rows",
    ]
    diagnostics_rows: list[dict[str, Any]] = []
    next_gameweek = first_gameweek
    if resume:
        previous_metadata, _, diagnostics_rows, last_completed = _load_resume_state(
            ledger_path=ledger_path,
            diagnostics_path=diagnostics_path,
            metadata_path=metadata_path,
            expected_metadata=metadata,
            header=header,
        )
        resume_record = {
            "previous_status": previous_metadata.get("status"),
            "previous_last_gameweek": last_completed,
            "previous_sampler_settings": {
                key: previous_metadata.get(key)
                for key in (
                    "num_warmup",
                    "num_samples",
                    "num_chains",
                    "nuts_target_accept_prob",
                    "max_divergence_rate",
                )
            },
            "current_sampler_settings": {
                key: metadata.get(key)
                for key in (
                    "num_warmup",
                    "num_samples",
                    "num_chains",
                    "nuts_target_accept_prob",
                    "max_divergence_rate",
                )
            },
        }
        metadata = {**previous_metadata, **metadata}
        metadata["status"] = "running"
        metadata["last_forecast_gameweek_written"] = last_completed
        metadata["resume_history"] = [
            *previous_metadata.get("resume_history", []),
            resume_record,
        ]
        metadata.pop("failure", None)
        for stale_result in (
            "scored_gameweeks",
            "prediction_metrics",
            "squad_evaluation_summary",
            "figures",
        ):
            metadata.pop(stale_result, None)
        next_gameweek = last_completed + 1

    _write_metadata(metadata_path, metadata)
    mode = "a" if resume else "w"
    with ledger_path.open(mode, encoding="utf-8", newline="") as ledger_stream:
        if not resume:
            pd.DataFrame(columns=header).to_csv(ledger_stream, index=False)
            ledger_stream.flush()
            os.fsync(ledger_stream.fileno())

        for target_gameweek in range(next_gameweek, last_gameweek + 1):
            fold_timesteps = target_gameweek - 2
            fold = build_fold_panels(
                gameweeks,
                target_gameweek,
                lookback=lookback,
                max_stale_gameweeks=max_gw + 1,
            )
            train = fold.train.loc[fold.train["player_id"].isin(fit_roster)].copy()
            train = train.sort_values(["player_id", "gameweek"], kind="stable").reset_index(drop=True)
            if train.empty or train["actual_points"].nunique() < 2:
                raise ValueError(f"Insufficient training outcomes before GW{target_gameweek}")

            target = fold.target.sort_values("player_id", kind="stable").reset_index(drop=True)
            raw_train_x = train.loc[:, list(MODEL_FEATURES)].to_numpy(dtype=np.float64)
            raw_target_x = target.loc[:, list(MODEL_FEATURES)].to_numpy(dtype=np.float64)
            scaler = StandardScaler().fit(raw_train_x)
            train_x = scaler.transform(raw_train_x).astype(np.float32)
            target_x = scaler.transform(raw_target_x).astype(np.float32)
            train_points = train["actual_points"].to_numpy(dtype=np.float64)
            train_codes = np.asarray(
                [fit_roster.index(int(player_id)) for player_id in train["player_id"]],
                dtype=np.int32,
            )
            target_ids = target["player_id"].to_numpy(dtype=int)
            all_history = fold.train.sort_values(
                ["player_id", "gameweek"], kind="stable"
            ).reset_index(drop=True)
            all_history_x = scaler.transform(
                all_history.loc[:, list(MODEL_FEATURES)].to_numpy(dtype=np.float64)
            ).astype(np.float32)

            gmm_data, gmm_mask, gmm_player_ids, player_to_code = _fit_gmm_arrays(
                train, train_x, fit_roster, fold_timesteps
            )
            gmm_samples, gmm_summary = run_hdp_gmm(
                gmm_data,
                gmm_player_ids,
                len(fit_roster),
                mask=gmm_mask,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
                truncation=truncation,
                rng_key=seed * 100_000 + target_gameweek * 10 + 1,
                progress_bar=False,
                print_summary=False,
            )
            gmm_prediction = predict_hdp_gmm(
                gmm_samples,
                train_x,
                train_codes,
                train_points,
                target_x,
                target_ids,
                player_to_code,
            )
            gmm_diagnostic = _sampler_diagnostic_record(
                gameweek=target_gameweek,
                model="hdp_gmm",
                training_rows=len(train),
                fit_players=len(fit_roster),
                summary=gmm_summary,
                retained_draws=num_samples * num_chains,
                max_divergence_rate=max_divergence_rate,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
            )
            diagnostics_rows.append(gmm_diagnostic)
            _require_sampler_diagnostics(
                gmm_diagnostic,
                diagnostics_rows=diagnostics_rows,
                diagnostics_path=diagnostics_path,
                metadata=metadata,
                metadata_path=metadata_path,
            )

            actions = np.ones((len(train), 1), dtype=np.float32)
            hmm_samples, hmm_summary = run_hdp_hmm(
                train_x,
                train_codes,
                actions,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
                truncation=truncation,
                rng_key=seed * 100_000 + target_gameweek * 10 + 2,
                max_timesteps=fold_timesteps,
                progress_bar=False,
                print_summary=False,
            )
            hmm_prediction = predict_hdp_hmm(
                hmm_samples,
                train,
                train_x,
                train_points,
                target_x,
                target_ids,
                filter_history=all_history,
                filter_history_x=all_history_x,
            )
            hmm_diagnostic = _sampler_diagnostic_record(
                gameweek=target_gameweek,
                model="hdp_hmm",
                training_rows=len(train),
                fit_players=len(fit_roster),
                summary=hmm_summary,
                retained_draws=num_samples * num_chains,
                max_divergence_rate=max_divergence_rate,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                target_accept_prob=target_accept_prob,
            )
            diagnostics_rows.append(hmm_diagnostic)
            _require_sampler_diagnostics(
                hmm_diagnostic,
                diagnostics_rows=diagnostics_rows,
                diagnostics_path=diagnostics_path,
                metadata=metadata,
                metadata_path=metadata_path,
            )

            fold_rows = pd.DataFrame(
                {
                    "season": season,
                    "seed": seed,
                    "gameweek": target_gameweek,
                    "feature_cutoff_gameweek": target_gameweek - 1,
                    "player_id": target_ids,
                    "position": target["position"].to_numpy(),
                    "price": target["price"].to_numpy(dtype=float),
                    "candidate_fresh": (
                        target["last_seen_gameweek"].astype(int)
                        >= target_gameweek - max_stale_gameweeks
                    ).to_numpy(),
                    "rolling_form": target["form_points"].to_numpy(dtype=float),
                    "hdp_gmm": gmm_prediction,
                    "hdp_hmm": hmm_prediction,
                    "fit_roster_member": [int(pid) in player_to_code for pid in target_ids],
                    "training_rows": len(train),
                })
            if not np.isfinite(fold_rows.loc[:, PREDICTION_COLUMNS].to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite forecast generated for GW{target_gameweek}")
            fold_rows.to_csv(ledger_stream, index=False, header=False)
            ledger_stream.flush()
            os.fsync(ledger_stream.fileno())

            pd.DataFrame(diagnostics_rows).to_csv(diagnostics_path, index=False)
            metadata["last_forecast_gameweek_written"] = target_gameweek
            _write_metadata(metadata_path, metadata)

    predictions = pd.read_csv(ledger_path)
    diagnostics = pd.DataFrame(diagnostics_rows)
    metadata["status"] = "forecasts_complete"
    _write_metadata(metadata_path, metadata)
    return predictions, diagnostics, metadata


def evaluate_forecast_ledger(
    gameweeks: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    output_dir: str | Path,
    prediction_columns: tuple[str, ...] = PREDICTION_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join outcomes only after forecast persistence, then score the ledger."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_predictions = set(prediction_columns) - set(predictions.columns)
    if missing_predictions:
        raise ValueError(
            f"Forecast ledger is missing prediction columns: {sorted(missing_predictions)}"
        )
    evaluated_predictions = predictions
    if "candidate_fresh" in evaluated_predictions:
        evaluated_predictions = evaluated_predictions.loc[
            evaluated_predictions["candidate_fresh"].astype(bool)
        ].copy()
    actuals: list[dict[str, float | int]] = []
    for gameweek in sorted(evaluated_predictions["gameweek"].astype(int).unique()):
        outcomes = outcomes_for_gameweek(gameweeks, int(gameweek))
        fold = evaluated_predictions.loc[
            evaluated_predictions["gameweek"].astype(int) == int(gameweek)
        ]
        for player_id in fold["player_id"].astype(int):
            actuals.append(
                {
                    "gameweek": int(gameweek),
                    "player_id": int(player_id),
                    "actual_points": float(outcomes.get(int(player_id), {}).get("points", 0.0)),
                }
            )
    scored_wide = evaluated_predictions.merge(
        pd.DataFrame(actuals), on=["gameweek", "player_id"], how="left", validate="one_to_one"
    )
    scored = scored_wide.melt(
        id_vars=["season", "seed", "gameweek", "feature_cutoff_gameweek", "player_id", "position", "actual_points"],
        value_vars=list(prediction_columns),
        var_name="model",
        value_name="prediction",
    )
    scored["error"] = scored["prediction"] - scored["actual_points"]
    summary = _metrics_rows(scored, ["model"])
    by_week = _metrics_rows(scored, ["gameweek", "model"])
    by_position = _metrics_rows(scored, ["position", "model"])
    scored_wide.to_csv(output_dir / "predictions_scored.csv", index=False)
    summary.to_csv(output_dir / "prediction_metrics.csv", index=False)
    by_week.to_csv(output_dir / "prediction_week_metrics.csv", index=False)
    by_position.to_csv(output_dir / "prediction_position_metrics.csv", index=False)
    return scored_wide, summary, by_week


def run_walk_forward_study(
    gameweeks: pd.DataFrame,
    *,
    data_path: str | Path,
    output_dir: str | Path,
    season: str = "2023-24",
    first_gameweek: int = 7,
    last_gameweek: int | None = None,
    lookback: int = 5,
    max_stale_gameweeks: int = 5,
    players_per_position: int = 12,
    seed: int = 42,
    num_warmup: int = 50,
    num_samples: int = 50,
    num_chains: int = 2,
    target_accept_prob: float = 0.95,
    max_divergence_rate: float = 0.01,
    truncation: int = 6,
    resume: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run forecasts, score them, and replay model scores through the MILP."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, diagnostics, metadata = generate_forecast_ledger(
        gameweeks,
        output_dir=output_dir,
        season=season,
        source_path=data_path,
        first_gameweek=first_gameweek,
        last_gameweek=last_gameweek,
        lookback=lookback,
        max_stale_gameweeks=max_stale_gameweeks,
        players_per_position=players_per_position,
        seed=seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        target_accept_prob=target_accept_prob,
        max_divergence_rate=max_divergence_rate,
        truncation=truncation,
        resume=resume,
    )
    scored, prediction_metrics, week_metrics = evaluate_forecast_ledger(
        gameweeks, predictions, output_dir=output_dir
    )
    position_metrics = pd.read_csv(output_dir / "prediction_position_metrics.csv")
    replay = run_backtest(
        gameweeks,
        strategies=MODEL_STRATEGIES,
        seeds=(seed,),
        season=season,
        lookback=lookback,
        max_stale_gameweeks=max_stale_gameweeks,
        dataset_sha256=str(metadata["source_sha256"]),
        prediction_ledger=predictions,
        evaluation_start_gameweek=first_gameweek,
    )
    replay.gameweeks.to_csv(output_dir / "squad_gameweeks.csv", index=False)
    replay.summary.to_csv(output_dir / "squad_summary.csv", index=False)
    figure_paths = plot_study_artifacts(output_dir)
    metadata.update(
        {
            "status": "complete",
            "prediction_metrics": prediction_metrics.to_dict(orient="records"),
            "squad_metrics": replay.summary.to_dict(orient="records"),
            "squad_metadata": replay.metadata,
            "visualizations": [str(path.relative_to(output_dir)) for path in figure_paths],
        }
    )
    _write_metadata(output_dir / "run_metadata.json", metadata)
    return {
        "predictions": predictions,
        "scored_predictions": scored,
        "prediction_metrics": prediction_metrics,
        "prediction_week_metrics": week_metrics,
        "prediction_position_metrics": position_metrics,
        "fit_diagnostics": diagnostics,
        "squad_gameweeks": replay.gameweeks,
        "squad_summary": replay.summary,
    }
