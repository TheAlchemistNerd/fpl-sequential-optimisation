"""
Tests for src/data/understat.py — Understat Advanced Metrics Scraper.

Tests cover:
    - JSON extraction from HTML (§3.3.1)
    - Async fetching with mocked responses
    - Fuzzy name matching between Understat and FPL
    - Multiple season fetching
    - Edge cases and error handling
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_understat_html():
    """Sample Understat HTML with embedded JSON."""
    return """
    <html>
    <head><title>EPL 2024/25</title></head>
    <body>
        <script>
            var playersData = JSON.parse('{\\"id\\":\\"1\\",\\"player_name\\":\\"Mohamed Salah\\",\\"games\\":\\"25\\",\\"time\\":\\"2250\\",\\"goals\\":\\"18\\",\\"shots\\":\\"75\\",\\"xG\\":\\"15.5\\",\\"npxG\\":\\"14.2\\",\\"assists\\":\\"10\\",\\"xA\\":\\"8.2\\",\\"keys\\":\\"45\\",\\"yellow_cards\\":\\"2\\",\\"red_cards\\":\\"0\\"}')
        </script>
        <div>Some other content</div>
    </body>
    </html>
    """


@pytest.fixture
def sample_understat_df():
    """Sample Understat player DataFrame."""
    return pd.DataFrame({
        "player_name": ["Mohamed Salah", "Erling Haaland", "Kevin De Bruyne"],
        "games": ["25", "22", "18"],
        "time": ["2250", "1980", "1620"],
        "goals": ["18", "20", "5"],
        "shots": ["75", "65", "30"],
        "xG": ["15.5", "18.2", "4.8"],
        "npxG": ["14.2", "18.2", "4.5"],
        "assists": ["10", "3", "12"],
        "xA": ["8.2", "2.5", "10.5"],
        "keys": ["45", "15", "55"],
        "yellow_cards": ["2", "3", "1"],
        "red_cards": ["0", "0", "0"],
    })


@pytest.fixture
def fpl_player_names():
    """Sample FPL player names."""
    return [
        "Mohamed Salah",
        "Erling Haaland",
        "Kevin De Bruyne",
        "Virgil van Dijk",
        "Bukayo Saka",
    ]


# ===========================================================================
# JSON extraction tests
# ===========================================================================


class TestJSONExtraction:
    """Validate JSON extraction from Understat HTML."""

    def test_extract_json_from_html(self, sample_understat_html):
        from src.data.understat import _extract_json_from_html
        result = _extract_json_from_html(sample_understat_html, "playersData")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["player_name"] == "Mohamed Salah"
        assert result[0]["xG"] == "15.5"

    def test_extract_json_from_html_wrong_data_type(self, sample_understat_html):
        from src.data.understat import _extract_json_from_html
        with pytest.raises(ValueError, match="Could not find"):
            _extract_json_from_html(sample_understat_html, "nonexistentData")

    def test_extract_json_from_html_malformed_json(self):
        from src.data.understat import _extract_json_from_html
        html = """
        <html>
        <script>
            var playersData = JSON.parse('this is not valid json')
        </script>
        </html>
        """
        with pytest.raises(Exception):  # Should raise JSON decode error
            _extract_json_from_html(html, "playersData")

    def test_extract_json_from_html_no_script_tags(self):
        from src.data.understat import _extract_json_from_html
        html = "<html><body>No scripts here</body></html>"
        with pytest.raises(ValueError):
            _extract_json_from_html(html, "playersData")


# ===========================================================================
# Async fetch tests
# ===========================================================================


class TestAsyncFetching:
    """Validate async Understat data fetching."""

    @pytest.mark.asyncio
    async def test_fetch_understat_season(self, sample_understat_html):
        from src.data.understat import fetch_understat_season
        mock_response = MagicMock()
        mock_response.text = AsyncMock(return_value=sample_understat_html)
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        result = await fetch_understat_season(mock_session, 2024)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "player_name" in result.columns
        assert result["player_name"].iloc[0] == "Mohamed Salah"

    @pytest.mark.asyncio
    async def test_fetch_understat_numeric_conversion(self, sample_understat_html):
        from src.data.understat import fetch_understat_season
        mock_response = MagicMock()
        mock_response.text = AsyncMock(return_value=sample_understat_html)
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        result = await fetch_understat_season(mock_session, 2024)
        # Numeric columns should be converted
        assert pd.api.types.is_numeric_dtype(result["games"])
        assert pd.api.types.is_numeric_dtype(result["xG"])
        assert pd.api.types.is_numeric_dtype(result["goals"])

    @pytest.mark.asyncio
    async def test_fetch_understat_multiple_seasons(self, sample_understat_html):
        from src.data.understat import fetch_understat_multiple_seasons
        with patch("src.data.understat._fetch_html") as mock_fetch:
            mock_fetch.return_value = sample_understat_html

            result = await fetch_understat_multiple_seasons([2023, 2024])
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2  # 2 seasons
            assert "season" in result.columns

    @pytest.mark.asyncio
    async def test_fetch_understat_multiple_seasons_some_fail(self):
        from src.data.understat import fetch_understat_multiple_seasons
        with patch("src.data.understat._fetch_html") as mock_fetch:
            mock_fetch.side_effect = [
                Exception("Network error"),
                '<script>var playersData = JSON.parse(\'{"id":"1","player_name":"Salah","games":"20","time":"1800","goals":"15","shots":"60","xG":"12.0","npxG":"11.0","assists":"8","xA":"6.0","keys":"35","yellow_cards":"1","red_cards":"0"}\')</script>',
            ]
            result = await fetch_understat_multiple_seasons([2023, 2024])
            # Should only have data from successful fetch
            assert len(result) == 1
            assert result["season"].iloc[0] == 2024

    @pytest.mark.asyncio
    async def test_fetch_understat_all_fail(self):
        from src.data.understat import fetch_understat_multiple_seasons
        with patch("src.data.understat._fetch_html") as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            result = await fetch_understat_multiple_seasons([2023, 2024])
            assert len(result) == 0  # Empty DataFrame


# ===========================================================================
# Fuzzy name matching tests
# ===========================================================================


class TestNameMatching:
    """Validate fuzzy name matching between Understat and FPL."""

    def test_match_understat_to_fpl_exact(self, sample_understat_df, fpl_player_names):
        from src.data.understat import match_understat_to_fpl
        result = match_understat_to_fpl(sample_understat_df, fpl_player_names)
        assert "fpl_name" in result.columns
        # Should match exact names
        salah_row = result[result["player_name"] == "Mohamed Salah"]
        assert salah_row["fpl_name"].iloc[0] == "Mohamed Salah"

    def test_match_understat_to_fpl_fuzzy(self, fpl_player_names):
        from src.data.understat import match_understat_to_fpl
        # Slightly different name
        understat_df = pd.DataFrame({
            "player_name": ["M. Salah", "E. Haaland"],
            "games": ["25", "22"],
        })
        result = match_understat_to_fpl(understat_df, fpl_player_names)
        # Should still match with fuzzy logic
        assert result["fpl_name"].notna().sum() >= 1

    def test_match_understat_to_fpl_no_match(self, fpl_player_names):
        from src.data.understat import match_understat_to_fpl
        # Completely different names
        understat_df = pd.DataFrame({
            "player_name": ["John Smith", "Jane Doe"],
            "games": ["10", "15"],
        })
        result = match_understat_to_fpl(understat_df, fpl_player_names)
        # Should not match (below threshold)
        assert result["fpl_name"].isna().all()

    def test_match_understat_to_fpl_threshold_effects(self, fpl_player_names):
        from src.data.understat import match_understat_to_fpl
        understat_df = pd.DataFrame({
            "player_name": ["Mohamed Salah"],
            "games": ["25"],
        })
        # Low threshold: more matches
        result_low = match_understat_to_fpl(understat_df, fpl_player_names, threshold=50)
        # High threshold: fewer matches
        result_high = match_understat_to_fpl(understat_df, fpl_player_names, threshold=95)
        assert result_low["fpl_name"].notna().sum() >= result_high["fpl_name"].notna().sum()

    def test_match_understat_to_fpl_empty_understat(self, fpl_player_names):
        from src.data.understat import match_understat_to_fpl
        empty_df = pd.DataFrame(columns=["player_name"])
        result = match_understat_to_fpl(empty_df, fpl_player_names)
        assert len(result) == 0
        assert "fpl_name" in result.columns

    def test_match_understat_to_fpl_empty_fpl_names(self, sample_understat_df):
        from src.data.understat import match_understat_to_fpl
        result = match_understat_to_fpl(sample_understat_df, [])
        assert result["fpl_name"].isna().all()

    def test_match_understat_to_fpl_returns_copy(self, sample_understat_df, fpl_player_names):
        from src.data.understat import match_understat_to_fpl
        result = match_understat_to_fpl(sample_understat_df, fpl_player_names)
        assert result is not sample_understat_df


# ===========================================================================
# Integration tests
# ===========================================================================


class TestUnderstatPipeline:
    """Test Understat pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_fetch_and_match(self, sample_understat_html, fpl_player_names):
        from src.data.understat import fetch_understat_season, match_understat_to_fpl
        mock_response = MagicMock()
        mock_response.text = AsyncMock(return_value=sample_understat_html)
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        # Fetch
        df = await fetch_understat_season(mock_session, 2024)
        assert len(df) == 1

        # Match
        matched = match_understat_to_fpl(df, fpl_player_names)
        assert "fpl_name" in matched.columns


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extract_json_unicode_in_html(self):
        from src.data.understat import _extract_json_from_html
        html = """
        <html>
        <script>
            var playersData = JSON.parse('{\\"player_name\\":\\"Jos\\u00e9 Mour\\u00ednho\\",\\"games\\":\\"10\\"}')
        </script>
        </html>
        """
        result = _extract_json_from_html(html, "playersData")
        assert result[0]["player_name"] == "José Mourínho"

    def test_match_understat_to_fpl_special_characters(self, fpl_player_names):
        from src.data.understat import match_understat_to_fpl
        understat_df = pd.DataFrame({
            "player_name": ["Mohamed Salah★", "Erling Haaland©"],
            "games": ["25", "22"],
        })
        result = match_understat_to_fpl(understat_df, fpl_player_names)
        # Should handle special characters gracefully
        assert isinstance(result, pd.DataFrame)

    def test_fetch_understat_season_missing_columns(self, sample_understat_html):
        from src.data.understat import fetch_understat_season
        # HTML with missing columns
        html = """
        <html>
        <script>
            var playersData = JSON.parse('{\\"id\\":\\"1\\",\\"player_name\\":\\"Salah\\"}')
        </script>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = AsyncMock(return_value=html)
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        result = asyncio.run(fetch_understat_season(mock_session, 2024))
        # Should not crash, missing columns will be NaN
        assert "player_name" in result.columns

    def test_match_understat_to_fpl_duplicate_names(self, fpl_player_names):
        from src.data.understat import match_understat_to_fpl
        understat_df = pd.DataFrame({
            "player_name": ["Mohamed Salah", "Mohamed Salah"],  # Duplicate
            "games": ["25", "20"],
        })
        result = match_understat_to_fpl(understat_df, fpl_player_names)
        # Should handle duplicates
        assert len(result) == 2
        assert result["fpl_name"].iloc[0] == result["fpl_name"].iloc[1]
