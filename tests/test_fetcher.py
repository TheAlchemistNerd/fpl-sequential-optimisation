"""
Tests for src/data/fetcher.py — FPL API Data Acquisition.

Tests cover:
    - Session creation and retry configuration
    - Bootstrap static data fetching
    - Fixtures data fetching
    - Element summary fetching (single and multiple players)
    - Event live data fetching
    - Entry history fetching
    - Derived quantities (per-90, form, enrichment)
    - Error handling and edge cases
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_bootstrap_response():
    """Mock response for bootstrap-static endpoint."""
    return {
        "elements": [
            {
                "id": 1,
                "first_name": "Mohamed",
                "second_name": "Salah",
                "element_type": 3,
                "team": 10,
                "now_cost": 130,
                "total_points": 200,
                "points_per_game": 6.5,
                "selected_by_percent": 45.2,
                "form": 7.5,
                "ict_index": 250,
                "minutes": 1800,
                "goals_scored": 18,
                "assists": 10,
                "clean_sheets": 8,
                "goals_conceded": 20,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 1,
                "yellow_cards": 2,
                "red_cards": 0,
                "saves": 0,
                "bonus": 20,
                "bps": 600,
                "influence": "800.0",
                "creativity": "600.0",
                "threat": "1200.0",
                "starts": 20,
                "expected_goals": "15.5",
                "expected_assists": "8.2",
                "expected_goal_involvements": "23.7",
                "expected_goals_conceded": "20.0",
                "status": "a",
            },
            {
                "id": 2,
                "first_name": "Virgil",
                "second_name": "van Dijk",
                "element_type": 2,
                "team": 10,
                "now_cost": 65,
                "total_points": 120,
                "points_per_game": 4.0,
                "selected_by_percent": 15.3,
                "form": 4.5,
                "ict_index": 180,
                "minutes": 1620,
                "goals_scored": 3,
                "assists": 2,
                "clean_sheets": 10,
                "goals_conceded": 18,
                "yellow_cards": 3,
                "red_cards": 0,
                "saves": 0,
                "bonus": 15,
                "bps": 450,
                "influence": "600.0",
                "creativity": "300.0",
                "threat": "400.0",
                "starts": 18,
                "expected_goals": "2.5",
                "expected_assists": "1.8",
                "expected_goal_involvements": "4.3",
                "expected_goals_conceded": "18.0",
                "status": "a",
            },
        ],
        "teams": [
            {
                "id": 10,
                "name": "Liverpool",
                "strength": 5,
                "strength_overall_home": 5,
                "strength_overall_away": 5,
                "strength_attack_home": 5,
                "strength_attack_away": 5,
                "strength_defence_home": 5,
                "strength_defence_away": 4,
                "position": 1,
                "played": 25,
                "win": 18,
                "loss": 3,
                "draw": 4,
                "points": 58,
            },
        ],
        "events": [
            {
                "id": 1,
                "name": "Gameweek 1",
                "deadline_time": "2024-08-17T11:00:00Z",
                "average_entry_score": 55,
                "highest_score": 120,
                "is_current": False,
                "is_next": False,
                "is_previous": True,
            },
        ],
    }


@pytest.fixture
def mock_fixtures_response():
    """Mock response for fixtures endpoint."""
    return [
        {
            "id": 1,
            "code": 10,
            "team_h": 10,
            "team_a": 1,
            "team_h_score": 2,
            "team_a_score": 1,
            "finished": True,
            "minutes": 90,
            "provisional_start_time": True,
            "kickoff_time": "2024-08-17T11:00:00Z",
            "event": 1,
            "stats": [],
            "team_h_difficulty": 3,
            "team_a_difficulty": 4,
        },
        {
            "id": 2,
            "code": 11,
            "team_h": 1,
            "team_a": 10,
            "team_h_score": 0,
            "team_a_score": 3,
            "finished": True,
            "minutes": 90,
            "provisional_start_time": True,
            "kickoff_time": "2024-08-24T14:00:00Z",
            "event": 2,
            "stats": [],
            "team_h_difficulty": 4,
            "team_a_difficulty": 2,
        },
    ]


@pytest.fixture
def mock_element_summary_response():
    """Mock response for element-summary endpoint."""
    return {
        "history": [
            {
                "element": 1,
                "fixture": 1,
                "opponent_team": 1,
                "total_points": 12,
                "was_home": True,
                "kickoff_time": "2024-08-17T11:00:00Z",
                "team_h_score": 2,
                "team_a_score": 1,
                "round": 1,
                "minutes": 90,
                "goals_scored": 1,
                "assists": 1,
                "clean_sheets": 0,
                "goals_conceded": 1,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "saves": 0,
                "bonus": 2,
                "bps": 40,
                "influence": "50.0",
                "creativity": "40.0",
                "threat": "60.0",
                "ict_index": "15.0",
                "starts": 1,
                "expected_goals": "0.5",
                "expected_assists": "0.3",
                "expected_goal_involvements": "0.8",
                "expected_goals_conceded": "1.0",
                "value": 130,
                "transfers_balance": 50000,
                "selected": 2000000,
                "transfers_in": 100000,
                "transfers_out": 50000,
            },
            {
                "element": 1,
                "fixture": 2,
                "opponent_team": 2,
                "total_points": 8,
                "was_home": False,
                "kickoff_time": "2024-08-24T14:00:00Z",
                "team_h_score": 0,
                "team_a_score": 3,
                "round": 2,
                "minutes": 90,
                "goals_scored": 1,
                "assists": 0,
                "clean_sheets": 1,
                "goals_conceded": 0,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "saves": 0,
                "bonus": 1,
                "bps": 35,
                "influence": "45.0",
                "creativity": "30.0",
                "threat": "55.0",
                "ict_index": "13.0",
                "starts": 1,
                "expected_goals": "0.7",
                "expected_assists": "0.2",
                "expected_goal_involvements": "0.9",
                "expected_goals_conceded": "0.5",
                "value": 131,
                "transfers_balance": 60000,
                "selected": 2100000,
                "transfers_in": 120000,
                "transfers_out": 60000,
            },
        ],
        "history_past": [],
        "fixtures": [],
    }


@pytest.fixture
def mock_event_live_response():
    """Mock response for event/{GW}/live endpoint."""
    return {
        "elements": [
            {
                "id": 1,
                "stats": {
                    "minutes": 90,
                    "goals_scored": 1,
                    "assists": 1,
                    "clean_sheets": 0,
                    "goals_conceded": 1,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "saves": 0,
                    "bonus": 2,
                    "bps": 40,
                    "influence": "50.0",
                    "creativity": "40.0",
                    "threat": "60.0",
                    "ict_index": "15.0",
                    "starts": 1,
                    "expected_goals": "0.5",
                    "expected_assists": "0.3",
                    "expected_goal_involvements": "0.8",
                    "expected_goals_conceded": "1.0",
                    "total_points": 12,
                    "in_dreamteam": False,
                },
                "explain": [["Points for goal", 4], ["Points for assist", 3]],
                "in_dreamteam": False,
                "special": False,
                "multiplier": 1,
            },
        ],
    }


@pytest.fixture
def mock_entry_history_response():
    """Mock response for entry/{TID}/history endpoint."""
    return {
        "current": [
            {
                "event": 1,
                "points": 65,
                "total_points": 65,
                "rank": 1500000,
                "rank_sort": 1450000,
                "overall_rank": 1500000,
                "bank": 5,
                "value": 1000,
                "event_transfers": 1,
                "event_transfers_cost": 0,
                "active_chip": None,
            },
            {
                "event": 2,
                "points": 72,
                "total_points": 137,
                "rank": 1200000,
                "rank_sort": 1150000,
                "overall_rank": 1200000,
                "bank": 3,
                "value": 1005,
                "event_transfers": 0,
                "event_transfers_cost": 0,
                "active_chip": None,
            },
        ],
        "past": [],
        "chips": [],
    }


# ===========================================================================
# Session and helper tests
# ===========================================================================


class TestSessionManagement:
    """Validate session creation and retry configuration."""

    def test_get_session_returns_session(self):
        from src.data.fetcher import _get_session
        session = _get_session()
        assert isinstance(session, requests.Session)

    def test_get_session_caches_session(self):
        from src.data.fetcher import _get_session
        session1 = _get_session()
        session2 = _get_session()
        assert session1 is session2

    def test_session_has_retry_adapter(self):
        from src.data.fetcher import _get_session
        session = _get_session()
        # Check that adapters are mounted
        assert len(session.adapters) > 0


class TestFetchJSON:
    """Validate JSON fetching with error handling."""

    @patch("src.data.fetcher._get_session")
    def test_fetch_json_success(self, mock_get_session):
        from src.data.fetcher import _fetch_json
        mock_response = MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_get_session.return_value = mock_session

        result = _fetch_json("https://example.com/api")
        assert result == {"key": "value"}
        mock_session.get.assert_called_once()

    @patch("src.data.fetcher._get_session")
    def test_fetch_json_raises_on_http_error(self, mock_get_session):
        from src.data.fetcher import _fetch_json
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.HTTPError("404 Not Found")
        mock_get_session.return_value = mock_session

        with pytest.raises(requests.HTTPError):
            _fetch_json("https://example.com/api")


# ===========================================================================
# Bootstrap static tests
# ===========================================================================


class TestBootstrapStatic:
    """Validate bootstrap data fetching and enrichment."""

    @patch("src.data.fetcher._fetch_json")
    def test_get_bootstrap_data_returns_dict(self, mock_fetch, mock_bootstrap_response):
        from src.data.fetcher import get_bootstrap_data
        mock_fetch.return_value = mock_bootstrap_response

        result = get_bootstrap_data()
        assert isinstance(result, dict)
        assert "players" in result
        assert "teams" in result
        assert "events" in result

    @patch("src.data.fetcher._fetch_json")
    def test_bootstrap_data_players_dataframe(self, mock_fetch, mock_bootstrap_response):
        from src.data.fetcher import get_bootstrap_data
        mock_fetch.return_value = mock_bootstrap_response

        result = get_bootstrap_data()
        players = result["players"]
        assert isinstance(players, pd.DataFrame)
        assert len(players) == 2
        assert "id" in players.columns
        assert "position" in players.columns
        assert "status_label" in players.columns

    @patch("src.data.fetcher._fetch_json")
    def test_bootstrap_data_position_mapping(self, mock_fetch, mock_bootstrap_response):
        from src.data.fetcher import get_bootstrap_data
        mock_fetch.return_value = mock_bootstrap_response

        result = get_bootstrap_data()
        players = result["players"]
        # element_type 3 -> MID, 2 -> DEF
        assert players.loc[0, "position"] == "MID"
        assert players.loc[1, "position"] == "DEF"

    @patch("src.data.fetcher._fetch_json")
    def test_bootstrap_data_status_mapping(self, mock_fetch, mock_bootstrap_response):
        from src.data.fetcher import get_bootstrap_data
        mock_fetch.return_value = mock_bootstrap_response

        result = get_bootstrap_data()
        players = result["players"]
        # status "a" -> "available"
        assert players.loc[0, "status_label"] == "available"

    @patch("src.data.fetcher._fetch_json")
    def test_bootstrap_data_teams_dataframe(self, mock_fetch, mock_bootstrap_response):
        from src.data.fetcher import get_bootstrap_data
        mock_fetch.return_value = mock_bootstrap_response

        result = get_bootstrap_data()
        teams = result["teams"]
        assert isinstance(teams, pd.DataFrame)
        assert len(teams) == 1
        assert "id" in teams.columns
        assert "name" in teams.columns

    @patch("src.data.fetcher._fetch_json")
    def test_bootstrap_data_events_dataframe(self, mock_fetch, mock_bootstrap_response):
        from src.data.fetcher import get_bootstrap_data
        mock_fetch.return_value = mock_bootstrap_response

        result = get_bootstrap_data()
        events = result["events"]
        assert isinstance(events, pd.DataFrame)
        assert len(events) == 1
        assert "id" in events.columns
        assert "name" in events.columns


# ===========================================================================
# Fixtures tests
# ===========================================================================


class TestFixtures:
    """Validate fixtures data fetching."""

    @patch("src.data.fetcher._fetch_json")
    def test_get_fixtures_data_returns_dataframe(self, mock_fetch, mock_fixtures_response):
        from src.data.fetcher import get_fixtures_data
        mock_fetch.return_value = mock_fixtures_response

        result = get_fixtures_data()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "id" in result.columns
        assert "team_h" in result.columns
        assert "team_a" in result.columns
        assert "kickoff_time" in result.columns
        assert "event" in result.columns


# ===========================================================================
# Element summary tests
# ===========================================================================


class TestElementSummary:
    """Validate element summary fetching."""

    @patch("src.data.fetcher._fetch_json")
    def test_get_element_summary_returns_dataframe(self, mock_fetch, mock_element_summary_response):
        from src.data.fetcher import get_element_summary
        mock_fetch.return_value = mock_element_summary_response

        result = get_element_summary(player_id=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "player_id" in result.columns
        assert result["player_id"].iloc[0] == 1
        assert "total_points" in result.columns

    @patch("src.data.fetcher._fetch_json")
    def test_get_element_summary_empty_history(self, mock_fetch):
        from src.data.fetcher import get_element_summary
        mock_fetch.return_value = {"history": [], "history_past": [], "fixtures": []}

        result = get_element_summary(player_id=999)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "player_id" in result.columns

    @patch("src.data.fetcher.get_element_summary")
    def test_get_all_element_summaries_concatenates(self, mock_single, mock_element_summary_response):
        from src.data.fetcher import get_all_element_summaries
        mock_single.return_value = pd.DataFrame(mock_element_summary_response["history"])

        result = get_all_element_summaries([1, 2], delay=0.01)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 2 players × 2 gameweeks each

    @patch("src.data.fetcher.get_element_summary")
    def test_get_all_element_summaries_handles_failures(self, mock_single):
        from src.data.fetcher import get_all_element_summaries
        mock_single.side_effect = [
            pd.DataFrame([{"player_id": 1, "total_points": 10}]),
            Exception("API error"),
            pd.DataFrame([{"player_id": 3, "total_points": 8}]),
        ]

        result = get_all_element_summaries([1, 2, 3], delay=0.01)
        assert isinstance(result, pd.DataFrame)
        # Should have data for players 1 and 3, skipping 2
        assert len(result) == 2

    @patch("src.data.fetcher.get_element_summary")
    def test_get_all_element_summaries_all_fail(self, mock_single):
        from src.data.fetcher import get_all_element_summaries
        mock_single.side_effect = Exception("API error")

        result = get_all_element_summaries([1, 2], delay=0.01)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0  # Empty DataFrame when all fail


# ===========================================================================
# Event live tests
# ===========================================================================


class TestEventLive:
    """Validate event live data fetching."""

    @patch("src.data.fetcher._fetch_json")
    def test_get_event_live_returns_dataframe(self, mock_fetch, mock_event_live_response):
        from src.data.fetcher import get_event_live
        mock_fetch.return_value = mock_event_live_response

        result = get_event_live(gameweek=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "id" in result.columns
        # Stats should be flattened
        assert "minutes" in result.columns
        assert "goals_scored" in result.columns

    @patch("src.data.fetcher._fetch_json")
    def test_get_event_live_empty_elements(self, mock_fetch):
        from src.data.fetcher import get_event_live
        mock_fetch.return_value = {"elements": []}

        result = get_event_live(gameweek=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ===========================================================================
# Entry history tests
# ===========================================================================


class TestEntryHistory:
    """Validate entry history fetching."""

    @patch("src.data.fetcher._fetch_json")
    def test_get_entry_history_returns_dataframe(self, mock_fetch, mock_entry_history_response):
        from src.data.fetcher import get_entry_history
        mock_fetch.return_value = mock_entry_history_response

        result = get_entry_history(team_id=123)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "team_id" in result.columns
        assert result["team_id"].iloc[0] == 123
        assert "points" in result.columns
        assert "rank" in result.columns


# ===========================================================================
# Derived quantities tests
# ===========================================================================


class TestPer90Calculation:
    """Validate per-90 normalisation (§3.2.1)."""

    def test_per90_basic(self):
        from src.data.fetcher import compute_per90
        series = pd.Series([10, 20, 30], dtype=float)
        minutes = pd.Series([90, 180, 270], dtype=float)
        result = compute_per90(series, minutes)
        # Should be: 10/90*90=10, 20/180*90=10, 30/270*90=10
        assert result.tolist() == pytest.approx([10.0, 10.0, 10.0])

    def test_per90_zero_minutes_uses_clip(self):
        from src.data.fetcher import compute_per90
        series = pd.Series([10], dtype=float)
        minutes = pd.Series([0], dtype=float)
        result = compute_per90(series, minutes)
        # Should clip to 1: 10/1*90 = 900
        assert result.tolist() == pytest.approx([900.0])

    def test_per90_negative_values(self):
        from src.data.fetcher import compute_per90
        series = pd.Series([-5], dtype=float)
        minutes = pd.Series([90], dtype=float)
        result = compute_per90(series, minutes)
        assert result.tolist() == pytest.approx([-5.0])


class TestFormCalculation:
    """Validate rolling form computation (§3.2.1)."""

    def test_form_window_3(self):
        from src.data.fetcher import compute_form
        points = pd.Series([10, 5, 8, 12, 6])
        result = compute_form(points, window=3)
        # First value: 10/1=10
        # Second: (10+5)/2=7.5
        # Third: (10+5+8)/3=7.67
        # Fourth: (5+8+12)/3=8.33
        # Fifth: (8+12+6)/3=8.67
        expected = [10.0, 7.5, 7.666666666666667, 8.333333333333334, 8.666666666666666]
        assert result.tolist() == pytest.approx(expected)

    def test_form_window_1(self):
        from src.data.fetcher import compute_form
        points = pd.Series([10, 5, 8])
        result = compute_form(points, window=1)
        assert result.tolist() == pytest.approx([10.0, 5.0, 8.0])

    def test_form_with_nan_values(self):
        from src.data.fetcher import compute_form
        points = pd.Series([10, float('nan'), 8])
        result = compute_form(points, window=2)
        # Should handle NaN gracefully
        assert not result.isna().all()


class TestEnrichBootstrapPlayers:
    """Validate player enrichment with derived columns."""

    def test_enrich_adds_per90_columns(self):
        from src.data.fetcher import enrich_bootstrap_players
        df = pd.DataFrame({
            "minutes": [1800, 900],
            "goals_scored": [18, 5],
            "assists": [10, 3],
            "expected_goals": [15.5, 4.2],
            "influence": [800.0, 400.0],
        })
        result = enrich_bootstrap_players(df)
        assert "goals_scored_per_90" in result.columns
        assert "assists_per_90" in result.columns
        assert "expected_goals_per_90" in result.columns
        assert "influence_per_90" in result.columns

    def test_enrich_per90_values_correct(self):
        from src.data.fetcher import enrich_bootstrap_players
        df = pd.DataFrame({
            "minutes": [90],
            "goals_scored": [1],
        })
        result = enrich_bootstrap_players(df)
        # 1 goal in 90 minutes = 1.0 per 90
        assert result["goals_scored_per_90"].iloc[0] == pytest.approx(1.0)

    def test_enrich_handles_missing_columns(self):
        from src.data.fetcher import enrich_bootstrap_players
        df = pd.DataFrame({
            "minutes": [1800],
            # Missing most optional columns
        })
        result = enrich_bootstrap_players(df)
        # Should not raise, just won't add per-90 cols
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# Position and status map tests
# ===========================================================================


class TestPositionMap:
    """Validate position mapping constants."""

    def test_position_map_has_all_positions(self):
        from src.data.fetcher import POSITION_MAP
        assert POSITION_MAP == {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    def test_status_map_has_all_statuses(self):
        from src.data.fetcher import STATUS_MAP
        assert "a" in STATUS_MAP
        assert "i" in STATUS_MAP
        assert "u" in STATUS_MAP
        assert "d" in STATUS_MAP
        assert "s" in STATUS_MAP


# ===========================================================================
# Edge cases and error handling
# ===========================================================================


class TestErrorHandling:
    """Validate error handling in fetcher functions."""

    @patch("src.data.fetcher._fetch_json")
    def test_bootstrap_data_missing_keys_raises(self, mock_fetch):
        from src.data.fetcher import get_bootstrap_data
        mock_fetch.return_value = {"elements": []}  # Missing "teams" and "events"
        with pytest.raises(KeyError):
            get_bootstrap_data()

    @patch("src.data.fetcher._fetch_json")
    def test_element_summary_malformed_response(self, mock_fetch):
        from src.data.fetcher import get_element_summary
        mock_fetch.return_value = {}  # Missing "history" key
        result = get_element_summary(player_id=1)
        # Should return empty DataFrame, not crash
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
