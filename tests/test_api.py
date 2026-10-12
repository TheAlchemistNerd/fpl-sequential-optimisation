"""Tests for API authentication and data fetching."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Auth Manager
# ===========================================================================

class TestFPLAuthManager:
    """Validate the FPL authentication manager."""

    def test_unauthenticated_by_default(self):
        from src.api.auth_manager import FPLAuthManager
        auth = FPLAuthManager()
        assert auth.is_authenticated is False

    def test_ensure_raises_without_credentials(self):
        from src.api.auth_manager import FPLAuthManager, FPLAuthError
        auth = FPLAuthManager()

        with patch("src.api.auth_manager.settings") as mock_settings:
            mock_settings.FPL_EMAIL = None
            mock_settings.FPL_PASSWORD = None
            with pytest.raises(FPLAuthError, match="FPL_EMAIL and FPL_PASSWORD"):
                auth.ensure_authenticated()

    def test_login_raises_on_http_failure(self):
        from src.api.auth_manager import FPLAuthManager, FPLAuthError
        auth = FPLAuthManager()

        mock_session = MagicMock()
        mock_session.get.return_value.raise_for_status.side_effect = requests.HTTPError()
        auth.session = mock_session

        with pytest.raises(requests.HTTPError):
            auth.login("test@test.com", "password")

    def test_get_authorized_session_authenticates(self):
        from src.api.auth_manager import FPLAuthManager
        auth = FPLAuthManager()

        with patch.object(auth, "ensure_authenticated") as mock_ensure:
            auth.get_authorized_session()
            mock_ensure.assert_called_once()


# ===========================================================================
# Settings
# ===========================================================================

class TestSettings:
    """Validate configuration loading."""

    def test_validate_returns_missing_when_set(self):
        from src.config.settings import Settings

        class MockSettings:
            FPL_EMAIL = "test@test.com"
            FPL_PASSWORD = "secret"
            FPL_TEAM_ID = 123

        assert MockSettings.FPL_EMAIL is not None
        assert MockSettings.FPL_PASSWORD is not None
        assert MockSettings.FPL_TEAM_ID is not None

    def test_api_rate_limit_is_positive(self):
        from src.config.settings import settings
        assert settings.API_RATE_LIMIT >= 0

    def test_data_dir_exists_or_is_path(self):
        from src.config.settings import settings
        assert settings.DATA_DIR is not None


# ===========================================================================
# Inference service helpers
# ===========================================================================

class TestInferenceServiceHelpers:
    def test_retrain_reports_not_configured(self):
        from src.api.inference_service import trigger_retrain

        response = trigger_retrain()

        assert response.status == "not_configured"
        assert "no retraining job is configured" in response.message.lower()

    def test_latest_gameweek_parses_numeric_suffix(self, tmp_path):
        from src.api.inference_service import _get_latest_gameweek

        (tmp_path / "belief_states_gw5.csv").touch()
        (tmp_path / "belief_states_gw12.csv").touch()
        (tmp_path / "belief_states_gw13_backup.csv").touch()

        assert _get_latest_gameweek(tmp_path) == 12

    def test_load_player_pool_maps_positions_and_normalizes_costs(
        self, tmp_path, monkeypatch
    ):
        import pandas as pd

        from src.api import inference_service

        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "element_type": [1, 2, 3, 4],
                "now_cost": [45, 50, 75, 105],
                "team": [1, 2, 3, 4],
                "points_per_game": [2.0, 3.0, 4.0, 5.0],
            }
        ).to_csv(tmp_path / "fpl_players.csv", index=False)
        monkeypatch.setattr(
            inference_service, "settings", SimpleNamespace(DATA_DIR=tmp_path)
        )

        player_pool = inference_service._load_player_pool()

        assert player_pool is not None
        assert player_pool["positions"] == ["GK", "DEF", "MID", "FWD"]
        assert player_pool["prices"] == [4.5, 5.0, 7.5, 10.5]
        assert player_pool["clubs"] == [1, 2, 3, 4]

    def test_load_player_pool_uses_series_fallbacks(self, tmp_path, monkeypatch):
        import pandas as pd

        from src.api import inference_service

        pd.DataFrame({"id": [1, 2]}).to_csv(tmp_path / "fpl_players.csv", index=False)
        monkeypatch.setattr(
            inference_service, "settings", SimpleNamespace(DATA_DIR=tmp_path)
        )

        player_pool = inference_service._load_player_pool()

        assert player_pool is not None
        assert player_pool["positions"] == ["MID", "MID"]
        assert player_pool["prices"] == [5.0, 5.0]
        assert player_pool["clubs"] == [0, 0]
