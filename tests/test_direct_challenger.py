from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_backtesting import _small_history


def test_direct_ridge_fold_is_cutoff_safe_and_handles_signed_targets():
    from src.research.direct_challenger import forecast_direct_ridge_fold
    from src.research.walk_forward import build_fold_panels

    history = _small_history(weeks=6)
    history.loc[
        (history["player_id"] == 1) & (history["gameweek"] == 3), "points"
    ] = -3.0
    target_gameweek = 5
    base_fold = build_fold_panels(
        history,
        target_gameweek,
        max_stale_gameweeks=int(history["gameweek"].max()) + 1,
    )
    saved_rows = base_fold.target[["player_id"]].copy()
    saved_rows["feature_cutoff_gameweek"] = target_gameweek - 1
    before = forecast_direct_ridge_fold(
        history, saved_rows, target_gameweek=target_gameweek
    )

    altered = history.copy()
    altered.loc[altered["gameweek"] >= target_gameweek, [
        "points", "minutes", "xg", "xa", "clean_sheets"
    ]] = -999.0
    altered_fold = build_fold_panels(
        altered,
        target_gameweek,
        max_stale_gameweeks=int(altered["gameweek"].max()) + 1,
    )
    altered_saved_rows = altered_fold.target[["player_id"]].copy()
    altered_saved_rows["feature_cutoff_gameweek"] = target_gameweek - 1
    after = forecast_direct_ridge_fold(
        altered, altered_saved_rows, target_gameweek=target_gameweek
    )

    assert base_fold.train["gameweek"].max() < target_gameweek
    assert (before["feature_cutoff_gameweek"] == target_gameweek - 1).all()
    assert np.isfinite(before["direct_ridge"]).all()
    assert before["direct_ridge_training_rows"].nunique() == 1
    pd.testing.assert_frame_equal(before, after)


def test_predict_direct_ridge_rejects_misaligned_and_nonfinite_training_data():
    import pytest

    from src.research.direct_challenger import predict_direct_ridge

    with pytest.raises(ValueError, match="aligned"):
        predict_direct_ridge(
            np.ones((3, 2)), np.ones(2), np.ones((1, 2))
        )
    with pytest.raises(ValueError, match="finite"):
        predict_direct_ridge(
            np.array([[0.0], [np.nan]]), np.array([1.0, 2.0]), np.array([[1.0]])
        )


def test_supplementary_comparison_keeps_completed_source_run_unchanged(tmp_path):
    import json

    from src.backtesting.engine import sha256_file
    from src.research.direct_challenger import add_direct_ridge_comparison
    from src.research.walk_forward import build_fold_panels

    history = _small_history(weeks=3)
    data_path = tmp_path / "source.csv"
    data_path.write_text("fixed archive fingerprint", encoding="utf-8")
    source_dir = tmp_path / "base_study"
    source_dir.mkdir()
    fold = build_fold_panels(
        history,
        3,
        max_stale_gameweeks=int(history["gameweek"].max()) + 1,
    )
    target = fold.target.sort_values("player_id", kind="stable").reset_index(drop=True)
    base_predictions = pd.DataFrame(
        {
            "season": "test",
            "seed": 17,
            "gameweek": 3,
            "feature_cutoff_gameweek": 2,
            "player_id": target["player_id"].astype(int),
            "position": target["position"],
            "price": target["price"],
            "candidate_fresh": True,
            "rolling_form": target["form_points"],
            "hdp_gmm": target["form_points"] + 0.5,
            "hdp_hmm": target["form_points"] - 0.5,
        }
    )
    predictions_path = source_dir / "predictions.csv"
    metadata_path = source_dir / "run_metadata.json"
    base_predictions.to_csv(predictions_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "source_sha256": sha256_file(data_path),
                "source_coverage": {"normalized_player_gameweek_rows": len(history)},
                "season": "test",
                "seed": 17,
                "first_gameweek": 3,
                "last_gameweek": 3,
                "feature_lookback_gameweeks": 5,
                "max_stale_gameweeks": 5,
            }
        ),
        encoding="utf-8",
    )
    original_predictions = predictions_path.read_bytes()
    original_metadata = metadata_path.read_bytes()

    result = add_direct_ridge_comparison(
        history,
        data_path=data_path,
        source_output_dir=source_dir,
        output_dir=tmp_path / "comparison",
        alpha=1.0,
    )

    assert set(result["prediction_metrics"]["model"]) == {
        "rolling_form", "direct_ridge", "hdp_gmm", "hdp_hmm"
    }
    assert set(result["squad_summary"]["strategy"]) == {
        "form_milp", "direct_ridge_milp", "hdp_gmm_milp", "hdp_hmm_milp"
    }
    assert (tmp_path / "comparison" / "direct_ridge_predictions.csv").is_file()
    assert (tmp_path / "comparison" / "predictions_with_direct.csv").is_file()
    assert predictions_path.read_bytes() == original_predictions
    assert metadata_path.read_bytes() == original_metadata
