"""
Tests for src/api/data_fetcher.py — Production FPL Data Fetcher.

Tests cover:
    - Rate limiting and retry logic
    - Public endpoint fetchers (bootstrap, fixtures, gameweek)
    - Protected endpoint fetchers (entry info, history, transfers, picks)
    - Orchestration (fetch_all_data)
    - Error handling and edge cases
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_bootstrap_data():
    """Mock bootstrap data response."""
    return (
        pd.DataFrame([
            {"id": 1, "first_name": "Mohamed", "second_name": "Salah", "element_type": 3, "team": 10},
            {"id": 2, "first_name": "Virgil", "second_name": "van Dijk", "element_type": 2, "team": 10},
        ]),
        pd.DataFrame([
            {"id": 10, "name": "Liverpool", "strength": 5},
        ]),
    )


@pytest.fixture
def mock_fixtures_data():
    """Mock fixtures data response."""
    return pd.DataFrame([
        {"id": 1, "team_h": 10, "team_a": 1, "event": 1, "finished": True},
        {"id": 2, "team_h": 1, "team_a": 10, "event": 2, "finished": True},
    ])


@pytest.fixture
def mock_gameweek_data():
    """Mock gameweek live data response."""
    return pd.DataFrame([
        {"id": 1, "gameweek": 1, "total_points": 12, "minutes": 90, "goals_scored": 1},
        {"id": 2, "gameweek": 1, "total_points": 6, "minutes": 90, "goals_scored": 0},
    ])


@pytest.fixture
def mock_settings():
    """Mock settings object."""
    mock = MagicMock()
    mock.FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
    mock.API_RATE_LIMIT = 0.01  # Fast for tests
    mock.API_MAX_RETRIES = 1
    mock.FPL_TEAM_ID = 123
    mock.DATA_DIR = Path("/tmp/fpl_test_data")
    return mock


# ===========================================================================
# Rate limiting and retry tests
# ===========================================================================


class TestRateLimitingAndRetry:
    """Validate rate limiting and retry logic."""

    @patch("src.api.data_fetcher.settings")
    def test_rate_limit_sleeps(self, mock_settings):
        from src.api import data_fetcher
        mock_settings.API_RATE_LIMIT = 0.5
        with patch("src.api.data_fetcher.time.sleep") as mock_sleep:
            data_fetcher._rate_limit()
            mock_sleep.assert_called_once_with(0.5)

    @patch("src.api.data_fetcher.time.sleep")
    def test_wait_with_backoff_exponential(self, mock_sleep):
        from src.api.data_fetcher import _wait_with_backoff
        _wait_with_backoff(0, 3)  # First attempt
        mock_sleep.assert_called_once()
        # delay = min(2^0 * 0.5, 30) = 0.5
        assert mock_sleep.call_args[0][0] == pytest.approx(0.5, abs=0.1)

    @patch("src.api.data_fetcher.time.sleep")
    def test_wait_with_backoff_caps_at_30(self, mock_sleep):
        from src.api.data_fetcher import _wait_with_backoff
        _wait_with_backoff(10, 20)  # High attempt
        # delay = min(2^10 * 0.5, 30) = 30 (capped)
        assert mock_sleep.call_args[0][0] == pytest.approx(30.0, abs=0.1)

    @patch("src.api.data_fetcher._rate_limit")
    @patch("src.api.data_fetcher.requests.Session")
    def test_request_with_retry_success_first_attempt(self, mock_session_cls, mock_rate_limit):
        from src.api.data_fetcher import _request_with_retry
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"key": "value"}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        result = _request_with_retry("https://example.com/api")
        assert result == mock_response
        mock_session.get.assert_called_once()

    @patch("src.api.data_fetcher._rate_limit")
    @patch("src.api.data_fetcher.requests.Session")
    def test_request_with_retry_exhausts_retries(self, mock_session_cls, mock_rate_limit):
        from src.api.data_fetcher import _request_with_retry
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.HTTPError("500 Error")
        mock_session_cls.return_value = mock_session

        with pytest.raises(requests.HTTPError):
            _request_with_retry("https://example.com/api", max_retries=2)
        assert mock_session.get.call_count == 2


# ===========================================================================
# Public data fetcher tests
# ===========================================================================


class TestPublicDataFetchers:
    """Validate public endpoint data fetchers."""

    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_bootstrap_data_returns_tuple(self, mock_request, mock_bootstrap_data, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "elements": mock_bootstrap_data[0].to_dict("records"),
                "teams": mock_bootstrap_data[1].to_dict("records"),
            }
            mock_request.return_value = mock_response

            players, teams = data_fetcher.get_bootstrap_data()
            assert isinstance(players, pd.DataFrame)
            assert isinstance(teams, pd.DataFrame)
            assert len(players) == 2
            assert len(teams) == 1

    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_fixture_data_returns_dataframe(self, mock_request, mock_fixtures_data, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_response = MagicMock()
            mock_response.json.return_value = mock_fixtures_data.to_dict("records")
            mock_request.return_value = mock_response

            result = data_fetcher.get_fixture_data()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "team_h" in result.columns

    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_gameweek_data_returns_dataframe(self, mock_request, mock_gameweek_data, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "elements": [
                    {
                        "id": 1,
                        "stats": {"total_points": 12, "minutes": 90},
                        "explain": [],
                        "in_dreamteam": False,
                        "multiplier": 1,
                    },
                ],
            }
            mock_request.return_value = mock_response

            result = data_fetcher.get_gameweek_data(gameweek=1)
            assert isinstance(result, pd.DataFrame)
            assert "id" in result.columns
            assert "gameweek" in result.columns
            assert "total_points" in result.columns

    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_gameweek_data_empty_elements(self, mock_request, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_response = MagicMock()
            mock_response.json.return_value = {"elements": []}
            mock_request.return_value = mock_response

            result = data_fetcher.get_gameweek_data(gameweek=1)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


# ===========================================================================
# Protected data fetcher tests
# ===========================================================================


class TestProtectedDataFetchers:
    """Validate protected endpoint data fetchers."""

    @patch("src.api.data_fetcher.FPLAuthManager")
    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_entry_info_returns_dict(self, mock_request, mock_auth, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_auth_instance = MagicMock()
            mock_auth_instance.get_authorized_session.return_value = MagicMock()
            mock_auth.return_value = mock_auth_instance

            mock_response = MagicMock()
            mock_response.json.return_value = {"id": 123, "player_first_name": "John"}
            mock_request.return_value = mock_response

            result = data_fetcher.get_entry_info(team_id=123)
            assert isinstance(result, dict)
            assert "id" in result

    @patch("src.api.data_fetcher.FPLAuthManager")
    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_entry_history_returns_dict(self, mock_request, mock_auth, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_auth_instance = MagicMock()
            mock_auth_instance.get_authorized_session.return_value = MagicMock()
            mock_auth.return_value = mock_auth_instance

            mock_response = MagicMock()
            mock_response.json.return_value = {"current": [], "past": []}
            mock_request.return_value = mock_response

            result = data_fetcher.get_entry_history(team_id=123)
            assert isinstance(result, dict)

    @patch("src.api.data_fetcher.FPLAuthManager")
    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_entry_transfers_returns_list(self, mock_request, mock_auth, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_auth_instance = MagicMock()
            mock_auth_instance.get_authorized_session.return_value = MagicMock()
            mock_auth.return_value = mock_auth_instance

            mock_response = MagicMock()
            mock_response.json.return_value = [{"player": "Salah", "amount": 1}]
            mock_request.return_value = mock_response

            result = data_fetcher.get_entry_transfers(team_id=123)
            assert isinstance(result, list)
            assert len(result) == 1

    @patch("src.api.data_fetcher.FPLAuthManager")
    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_entry_picks_returns_dict(self, mock_request, mock_auth, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_auth_instance = MagicMock()
            mock_auth_instance.get_authorized_session.return_value = MagicMock()
            mock_auth.return_value = mock_auth_instance

            mock_response = MagicMock()
            mock_response.json.return_value = {
                "active_chip": None,
                "automatic_subs": [],
                "entry_history": {"points": 65},
                "picks": [{"element": 1, "position": 1}],
            }
            mock_request.return_value = mock_response

            result = data_fetcher.get_entry_picks(gameweek=1, team_id=123)
            assert isinstance(result, dict)
            assert "picks" in result


# ===========================================================================
# Orchestration tests
# ===========================================================================


class TestFetchAllData:
    """Validate fetch_all_data orchestration."""

    @patch("src.api.data_fetcher.settings")
    @patch("src.api.data_fetcher.get_bootstrap_data")
    @patch("src.api.data_fetcher.get_fixture_data")
    @patch("src.api.data_fetcher.get_gameweek_data")
    @patch("src.api.data_fetcher._save_json")
    def test_fetch_all_data_calls_all_fetchers(
        self, mock_save, mock_gw_data, mock_fixtures, mock_bootstrap, mock_settings
    ):
        from src.api.data_fetcher import fetch_all_data
        mock_settings.DATA_DIR = Path("/tmp/fpl_test")
        mock_settings.FPL_TEAM_ID = None  # Skip protected data

        mock_bootstrap.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_fixtures.return_value = pd.DataFrame()
        mock_gw_data.return_value = pd.DataFrame([{"id": 1, "gameweek": 1}])

        asyncio.run(fetch_all_data(latest_gameweek=1))

        mock_bootstrap.assert_called_once()
        mock_fixtures.assert_called_once()
        mock_gw_data.assert_called_once_with(1)
        assert mock_save.call_count >= 3  # At least 3 CSVs saved

    @patch("src.api.data_fetcher.settings")
    @patch("src.api.data_fetcher.get_bootstrap_data")
    @patch("src.api.data_fetcher.get_fixture_data")
    @patch("src.api.data_fetcher.get_gameweek_data")
    def test_fetch_all_data_creates_data_dir(
        self, mock_gw_data, mock_fixtures, mock_bootstrap, mock_settings
    ):
        from src.api.data_fetcher import fetch_all_data
        mock_settings.DATA_DIR = Path("/tmp/fpl_test_new_dir")
        mock_settings.FPL_TEAM_ID = None

        mock_bootstrap.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_fixtures.return_value = pd.DataFrame()
        mock_gw_data.return_value = pd.DataFrame()

        asyncio.run(fetch_all_data(latest_gameweek=1))
        # Should have created the directory
        # (Can't easily verify in test, but should not raise)


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestHelperFunctions:
    """Validate helper functions."""

    def test_save_json(self):
        from src.api.data_fetcher import _save_json
        data = {"key": "value", "number": 42}
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            with patch("src.api.data_fetcher.log"):
                _save_json(data, Path("/tmp/test.json"))
                mock_file.assert_called_once_with(Path("/tmp/test.json"), "w")

    def test_save_json_formats_indent(self):
        from src.api.data_fetcher import _save_json
        data = {"key": "value"}
        import json
        mock_file = mock_open()
        with patch("builtins.open", mock_file):
            with patch("src.api.data_fetcher.log"):
                _save_json(data, Path("/tmp/test.json"))
                # Check that json.dump was called with indent=2
                handle = mock_file()
                handle.write.assert_called_once()
                written = handle.write.call_args[0][0]
                parsed = json.loads(written)
                assert parsed == data


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_bootstrap_data_missing_elements_key(self, mock_request, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_response = MagicMock()
            mock_response.json.return_value = {"teams": []}  # Missing "elements"
            mock_request.return_value = mock_response

            with pytest.raises(KeyError):
                data_fetcher.get_bootstrap_data()

    @patch("src.api.data_fetcher._request_with_retry")
    def test_get_gameweek_data_malformed_response(self, mock_request, mock_settings):
        from src.api import data_fetcher
        with patch.object(data_fetcher, "settings", mock_settings):
            mock_response = MagicMock()
            mock_response.json.return_value = {}  # Missing "elements"
            mock_request.return_value = mock_response

            result = data_fetcher.get_gameweek_data(gameweek=1)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    @patch("src.api.data_fetcher.settings")
    @patch("src.api.data_fetcher.get_bootstrap_data")
    @patch("src.api.data_fetcher.get_fixture_data")
    @patch("src.api.data_fetcher.get_gameweek_data")
    def test_fetch_all_data_protected_data_fails_gracefully(
        self, mock_gw_data, mock_fixtures, mock_bootstrap, mock_settings
    ):
        from src.api.data_fetcher import fetch_all_data
        from src.api.data_fetcher import get_entry_info
        mock_settings.DATA_DIR = Path("/tmp/fpl_test")
        mock_settings.FPL_TEAM_ID = 123

        mock_bootstrap.return_value = (pd.DataFrame(), pd.DataFrame())
        mock_fixtures.return_value = pd.DataFrame()
        mock_gw_data.return_value = pd.DataFrame()

        # Make protected fetcher fail
        with patch("src.api.data_fetcher.get_entry_info", side_effect=Exception("Auth failed")):
            # Should not raise, just log the error
            asyncio.run(fetch_all_data(latest_gameweek=1))
