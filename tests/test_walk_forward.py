from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_backtesting import _small_history


def test_fold_panels_keep_target_outcomes_out_of_features_and_training():
    from src.research.walk_forward import build_fold_panels

    history = _small_history(weeks=5)
    original = build_fold_panels(history, 4)
    changed = history.copy()
    future = changed["gameweek"] >= 4
    changed.loc[future, ["points", "minutes", "xg", "xa", "clean_sheets"]] = 999
    repeated = build_fold_panels(changed, 4)
    pd.testing.assert_frame_equal(original.train, repeated.train)
    pd.testing.assert_frame_equal(original.target, repeated.target)
    assert set(original.train["gameweek"]) == {2, 3}
    assert "actual_points" not in original.target


def test_fit_roster_is_seeded_and_uses_only_gw1_players():
    from src.research.walk_forward import choose_fit_roster

    history = _small_history(weeks=4)
    first = choose_fit_roster(history, players_per_position=2, seed=31)
    second = choose_fit_roster(history, players_per_position=2, seed=31)
    assert first == second
    assert len(first) == 8
    assert set(first).issubset(set(history.loc[history["gameweek"] == 1, "player_id"]))


def test_numpy_stick_breaking_and_gmm_responsibilities_are_probabilities():
    from src.research.walk_forward import _numpy_stick_breaking, _responsibilities

    weights = _numpy_stick_breaking(np.array([[0.2, 0.5], [0.8, 0.3]]))
    assert weights.shape == (2, 3)
    assert np.allclose(weights.sum(axis=1), 1.0)
    responsibilities = _responsibilities(
        np.array([[0.0, 0.0], [3.0, 3.0]]),
        np.array([[0.5, 0.5], [0.5, 0.5]]),
        np.array([[0.0, 0.0], [3.0, 3.0]]),
        np.stack([np.eye(2), np.eye(2)]),
    )
    assert responsibilities.shape == (2, 2)
    assert np.allclose(responsibilities.sum(axis=1), 1.0)
    assert responsibilities[0, 0] > responsibilities[0, 1]
    assert responsibilities[1, 1] > responsibilities[1, 0]


def test_gmm_state_reward_mapping_returns_finite_player_forecasts():
    from src.research.walk_forward import predict_hdp_gmm

    samples = {
        "beta_raw": np.array([[0.5], [0.6]]),
        "player_weights": np.array(
            [[[0.6, 0.4], [0.5, 0.5]], [[0.7, 0.3], [0.4, 0.6]]]
        ),
        "mu": np.array(
            [
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-0.9, -1.1], [1.1, 0.9]],
            ]
        ),
        "precision": np.tile(np.eye(2), (2, 2, 1, 1)),
    }
    prediction = predict_hdp_gmm(
        samples,
        train_x=np.array([[-1.0, -1.0], [-0.8, -1.1], [1.0, 1.0], [0.8, 1.1]]),
        train_codes=np.array([0, 0, 1, 1]),
        train_points=np.array([2.0, 3.0, 8.0, 9.0]),
        target_x=np.array([[-1.0, -0.9], [1.1, 1.0]]),
        target_player_ids=np.array([10, 30]),
        player_to_code={10: 0, 20: 1},
    )
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert prediction[0] < prediction[1]


def test_hmm_forward_filter_maps_state_history_to_points():
    from src.research.walk_forward import predict_hdp_hmm

    samples = {
        "beta_raw": np.array([[0.5], [0.6]]),
        "trans_probs": np.array(
            [
                [[[0.9, 0.1], [0.1, 0.9]]],
                [[[0.85, 0.15], [0.15, 0.85]]],
            ]
        ),
        "mu": np.array(
            [
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-0.9, -1.1], [1.1, 0.9]],
            ]
        ),
        "precision": np.tile(np.eye(2), (2, 2, 1, 1)),
    }
    train = pd.DataFrame(
        [
            {"player_id": pid, "gameweek": gw}
            for pid in (10, 20)
            for gw in (2, 3, 4)
        ]
    )
    train_x = np.array(
        [[-1, -1], [-0.9, -1], [-1.1, -0.8], [1, 1], [0.9, 1], [1.1, 0.8]],
        dtype=float,
    )
    prediction = predict_hdp_hmm(
        samples,
        train,
        train_x,
        train_points=np.array([2, 2, 3, 8, 9, 9], dtype=float),
        target_x=np.array([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]),
        target_player_ids=np.array([10, 20, 30]),
    )
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
    assert prediction[0] < prediction[1]


def test_hmm_rejects_training_gaps_in_weekly_sequences():
    from src.research.walk_forward import predict_hdp_hmm

    samples = {
        "beta_raw": np.array([[0.5]]),
        "trans_probs": np.array([[[[0.8, 0.2], [0.2, 0.8]]]]),
        "mu": np.array([[[-1.0], [1.0]]]),
        "precision": np.ones((1, 2, 1, 1)),
    }
    train = pd.DataFrame(
        [{"player_id": 1, "gameweek": 2}, {"player_id": 1, "gameweek": 4}]
    )
    with pytest.raises(ValueError, match="has gaps"):
        predict_hdp_hmm(
            samples, train, np.array([[-1.0], [1.0]]), np.array([1.0, 2.0]),
            np.array([[0.0]]), np.array([1]),
        )


def test_forecast_evaluation_uses_shared_fresh_pool_and_writes_metrics(tmp_path):
    from src.research.walk_forward import evaluate_forecast_ledger

    history = _small_history(weeks=3)
    ids = sorted(history["player_id"].unique())
    predictions = pd.DataFrame(
        [
            {
                "season": "test",
                "seed": 1,
                "gameweek": 3,
                "feature_cutoff_gameweek": 2,
                "player_id": player_id,
                "position": history.loc[history["player_id"] == player_id, "position"].iloc[0],
                "candidate_fresh": player_id != ids[0],
                "rolling_form": 4.0,
                "hdp_gmm": 5.0,
                "hdp_hmm": 6.0,
            }
            for player_id in ids
        ]
    )

    scored, metrics, by_week = evaluate_forecast_ledger(
        history, predictions, output_dir=tmp_path
    )

    assert len(scored) == len(ids) - 1
    assert set(metrics["model"]) == {"rolling_form", "hdp_gmm", "hdp_hmm"}
    assert set(by_week["gameweek"]) == {3}
    assert (tmp_path / "predictions_scored.csv").is_file()
    assert (tmp_path / "prediction_position_metrics.csv").is_file()


def test_run_metadata_is_strict_json_and_atomic(tmp_path):
    import json

    from src.research.walk_forward import _write_metadata

    destination = tmp_path / "run_metadata.json"
    _write_metadata(destination, {"value": np.float64(np.nan), "seed": np.int64(42)})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "value": None,
        "seed": 42,
    }
    assert not destination.with_suffix(".json.tmp").exists()


def test_sampler_diagnostic_gate_stops_before_forecast_and_saves_failure(tmp_path):
    import json

    from src.research.walk_forward import (
        _require_sampler_diagnostics,
        _sampler_diagnostic_record,
    )

    diagnostic = _sampler_diagnostic_record(
        gameweek=7,
        model="hdp_gmm",
        training_rows=120,
        fit_players=24,
        summary={
            "num_divergences": 4,
            "mean_accept_prob": 0.91,
            "active_clusters": 4,
        },
        retained_draws=50,
        max_divergence_rate=0.01,
    )
    metadata = {"status": "running"}

    with pytest.raises(RuntimeError, match="failed NUTS diagnostics"):
        _require_sampler_diagnostics(
            diagnostic,
            diagnostics_rows=[diagnostic],
            diagnostics_path=tmp_path / "fit_diagnostics.csv",
            metadata=metadata,
            metadata_path=tmp_path / "run_metadata.json",
        )

    assert not diagnostic["diagnostic_gate_passed"]
    assert "divergence_rate" in pd.read_csv(tmp_path / "fit_diagnostics.csv")
    assert json.loads((tmp_path / "run_metadata.json").read_text())["status"] == "sampling_diagnostics_failed"


def test_plot_study_artifacts_regenerates_all_figures_from_saved_csvs(tmp_path):
    from src.research.plots import plot_study_artifacts

    pd.DataFrame(
        [
            {"gameweek": gw, "model": model, "mae": 1.0 + i, "rmse": 2.0 + i, "spearman": 0.4 - i * 0.1}
            for gw in (7, 8)
            for i, model in enumerate(("rolling_form", "hdp_gmm", "hdp_hmm"))
        ]
    ).to_csv(tmp_path / "prediction_week_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"position": position, "model": model, "mae": 1.0 + i}
            for position in ("GK", "DEF", "MID", "FWD")
            for i, model in enumerate(("rolling_form", "hdp_gmm", "hdp_hmm"))
        ]
    ).to_csv(tmp_path / "prediction_position_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": 42,
                "strategy": strategy,
                "gameweek": gw,
                "is_evaluation_gameweek": True,
                "evaluation_net_points": points,
            }
            for strategy, points in (("form_milp", 50), ("hdp_gmm_milp", 55), ("hdp_hmm_milp", 48))
            for gw in (7, 8)
        ]
    ).to_csv(tmp_path / "squad_gameweeks.csv", index=False)

    paths = plot_study_artifacts(tmp_path)

    assert [path.name for path in paths] == [
        "prediction_error_by_gameweek.png",
        "prediction_rank_correlation_by_gameweek.png",
        "prediction_mae_by_position.png",
        "cumulative_squad_points.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 5_000 for path in paths)
