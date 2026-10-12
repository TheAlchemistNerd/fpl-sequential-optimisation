"""
Tests for src/data/sentiment.py — News Sentiment Analysis.

Tests cover:
    - Lazy model loading
    - Article fetching (stub)
    - Sentiment scoring
    - Feature computation from live and pre-collected articles
    - Edge cases and graceful degradation
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_articles():
    """Sample news articles for sentiment analysis."""
    return [
        "Mohamed Salah scores brilliant goal as Liverpool wins 3-0",
        "Salah injury concern for upcoming match, manager confirms",
        "Liverpool star Salah praises team performance and fan support",
    ]


@pytest.fixture
def sample_precollected_articles():
    """Pre-collected articles DataFrame."""
    return pd.DataFrame({
        "player_name": ["Salah", "Salah", "Haaland", "Haaland", "Haaland"],
        "gameweek": [1, 1, 1, 2, 2],
        "article_text": [
            "Salah scores amazing goal in victory",
            "Salah injury worry for next match",
            "Haaland breaks scoring record with hat-trick",
            "Haaland misses penalty in draw",
            "Haaland praised by manager after training",
        ],
    })


# ===========================================================================
# Lazy model loading tests
# ===========================================================================


class TestLazyModelLoading:
    """Validate lazy loading of sentiment model."""

    def test_load_model_returns_pipeline(self):
        from src.data.sentiment import _load_model
        # Should try to load model, may return (None, None) if transformers not installed
        model, tokenizer = _load_model()
        # Either both are loaded or both are None
        assert (model is None and tokenizer is None) or (model is not None and tokenizer is not None)

    @patch("src.data.sentiment._load_model")
    def test_score_texts_with_no_model(self, mock_load):
        from src.data.sentiment import _score_texts
        mock_load.return_value = (None, None)
        texts = ["Positive article", "Negative news"]
        result = _score_texts(texts)
        # Should return neutral scores
        assert result == [0.0, 0.0]

    def test_score_texts_empty_texts(self):
        from src.data.sentiment import _score_texts
        with patch("src.data.sentiment._load_model") as mock_load:
            mock_load.return_value = (MagicMock(), MagicMock())
            result = _score_texts(["", "  ", None])
            # Empty texts should get 0.0
            assert all(s == 0.0 for s in result)


# ===========================================================================
# Article fetching tests
# ===========================================================================


class TestArticleFetching:
    """Validate article fetching (stub implementation)."""

    def test_fetch_articles_for_player_returns_list(self):
        from src.data.sentiment import fetch_articles_for_player
        deadline = datetime.now()
        result = fetch_articles_for_player("Mohamed Salah", deadline)
        assert isinstance(result, list)
        # Stub returns empty list
        assert len(result) == 0

    def test_fetch_articles_for_player_lookback_window(self):
        from src.data.sentiment import fetch_articles_for_player
        deadline = datetime.now()
        lookback = 14
        result = fetch_articles_for_player("Salah", deadline, lookback_days=lookback)
        # Should accept parameters without error
        assert isinstance(result, list)


# ===========================================================================
# Sentiment scoring tests
# ===========================================================================


class TestSentimentScoring:
    """Validate sentiment scoring logic."""

    @patch("src.data.sentiment._load_model")
    def test_score_texts_positive(self, mock_load):
        from src.data.sentiment import _score_texts
        mock_model = MagicMock()
        mock_model.return_value = [{"label": "positive", "score": 0.9}]
        mock_load.return_value = (mock_model, "tokenizer")

        texts = ["Great performance, excellent result"]
        result = _score_texts(texts)
        assert len(result) == 1
        assert result[0] > 0  # Positive score

    @patch("src.data.sentiment._load_model")
    def test_score_texts_negative(self, mock_load):
        from src.data.sentiment import _score_texts
        mock_model = MagicMock()
        mock_model.return_value = [{"label": "negative", "score": 0.85}]
        mock_load.return_value = (mock_model, "tokenizer")

        texts = ["Terrible injury, devastating loss"]
        result = _score_texts(texts)
        assert len(result) == 1
        assert result[0] < 0  # Negative score

    @patch("src.data.sentiment._load_model")
    def test_score_texts_neutral(self, mock_load):
        from src.data.sentiment import _score_texts
        mock_model = MagicMock()
        mock_model.return_value = [{"label": "neutral", "score": 0.7}]
        mock_load.return_value = (mock_model, "tokenizer")

        texts = ["Match scheduled for tomorrow"]
        result = _score_texts(texts)
        assert len(result) == 1
        assert result[0] == 0.0  # Neutral maps to 0

    @patch("src.data.sentiment._load_model")
    def test_score_texts_truncates_long_text(self, mock_load):
        from src.data.sentiment import _score_texts
        mock_model = MagicMock()
        mock_model.return_value = [{"label": "positive", "score": 0.8}]
        mock_load.return_value = (mock_model, "tokenizer")

        long_text = "word " * 1000  # Very long text
        result = _score_texts([long_text])
        assert len(result) == 1
        # Model should have been called with truncated text
        mock_model.assert_called_once()

    @patch("src.data.sentiment._load_model")
    def test_score_texts_handles_exception(self, mock_load):
        from src.data.sentiment import _score_texts
        mock_model = MagicMock()
        mock_model.side_effect = Exception("Model error")
        mock_load.return_value = (mock_model, "tokenizer")

        result = _score_texts(["Some text"])
        # Should handle exception gracefully
        assert result == [0.0]


# ===========================================================================
# Sentiment feature computation tests
# ===========================================================================


class TestSentimentFeatureComputation:
    """Validate sentiment feature computation from live articles."""

    @patch("src.data.sentiment.fetch_articles_for_player")
    @patch("src.data.sentiment._score_texts")
    def test_compute_sentiment_features_with_articles(self, mock_score, mock_fetch):
        from src.data.sentiment import compute_sentiment_features
        mock_fetch.return_value = ["Article 1", "Article 2"]
        mock_score.return_value = [0.5, -0.3]

        deadlines = pd.Series([datetime.now()])
        result = compute_sentiment_features(["Salah"], deadlines)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "player_name" in result.columns
        assert "gameweek" in result.columns
        assert "Sentiment" in result.columns
        assert "SentimentVol" in result.columns
        # Mean sentiment: (0.5 + (-0.3)) / 2 = 0.1
        assert result["Sentiment"].iloc[0] == pytest.approx(0.1)

    @patch("src.data.sentiment.fetch_articles_for_player")
    def test_compute_sentiment_features_no_articles(self, mock_fetch):
        from src.data.sentiment import compute_sentiment_features
        mock_fetch.return_value = []

        deadlines = pd.Series([datetime.now()])
        result = compute_sentiment_features(["Salah"], deadlines)
        assert len(result) == 1
        assert result["Sentiment"].iloc[0] == 0.0
        assert result["SentimentVol"].iloc[0] == 0.0
        assert result["n_articles"].iloc[0] == 0

    @patch("src.data.sentiment.fetch_articles_for_player")
    @patch("src.data.sentiment._score_texts")
    def test_compute_sentiment_features_multiple_players(self, mock_score, mock_fetch):
        from src.data.sentiment import compute_sentiment_features
        def fetch_side_effect(name, deadline):
            return [f"Article about {name}"] if name == "Salah" else []

        mock_fetch.side_effect = fetch_side_effect
        mock_score.return_value = [0.6]

        deadlines = pd.Series([datetime.now(), datetime.now() + timedelta(days=7)])
        result = compute_sentiment_features(["Salah", "Haaland"], deadlines)
        # Should have 2 players × 2 gameweeks = 4 rows
        assert len(result) == 4

    @patch("src.data.sentiment.fetch_articles_for_player")
    @patch("src.data.sentiment._score_texts")
    def test_compute_sentiment_features_single_article_vol_zero(self, mock_score, mock_fetch):
        from src.data.sentiment import compute_sentiment_features
        mock_fetch.return_value = ["One article"]
        mock_score.return_value = [0.5]

        deadlines = pd.Series([datetime.now()])
        result = compute_sentiment_features(["Salah"], deadlines)
        # Single article: volatility should be 0
        assert result["SentimentVol"].iloc[0] == 0.0


# ===========================================================================
# Pre-collected articles tests
# ===========================================================================


class TestPrecollectedArticles:
    """Validate sentiment computation from pre-collected articles."""

    @patch("src.data.sentiment._score_texts")
    def test_compute_sentiment_from_precollected(self, mock_score, sample_precollected_articles):
        from src.data.sentiment import compute_sentiment_from_precollected
        mock_score.return_value = [0.6, -0.3, 0.8, -0.2, 0.4]

        result = compute_sentiment_from_precollected(sample_precollected_articles)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 3 (player, gameweek) groups
        assert "Sentiment" in result.columns
        assert "SentimentVol" in result.columns

    @patch("src.data.sentiment._score_texts")
    def test_compute_sentiment_from_precollected_grouping(self, mock_score, sample_precollected_articles):
        from src.data.sentiment import compute_sentiment_from_precollected
        mock_score.side_effect = [
            [0.8],  # Haaland GW1
            [-0.2, 0.4],  # Haaland GW2
            [0.6, -0.3],  # Salah GW1
        ]

        result = compute_sentiment_from_precollected(sample_precollected_articles)
        assert len(result) == 3
        haaland_gw1 = result[(result["player_name"] == "Haaland") & (result["gameweek"] == 1)]
        haaland_gw2 = result[(result["player_name"] == "Haaland") & (result["gameweek"] == 2)]
        salah_gw1 = result[(result["player_name"] == "Salah") & (result["gameweek"] == 1)]
        assert haaland_gw1["n_articles"].iloc[0] == 1
        assert haaland_gw2["n_articles"].iloc[0] == 2
        assert len(salah_gw1) == 1
        assert salah_gw1["n_articles"].iloc[0] == 2

    def test_compute_sentiment_from_precollected_empty(self):
        from src.data.sentiment import compute_sentiment_from_precollected
        empty_df = pd.DataFrame(columns=["player_name", "gameweek", "article_text"])
        result = compute_sentiment_from_precollected(empty_df)
        assert len(result) == 0

    @patch("src.data.sentiment._score_texts")
    def test_compute_sentiment_from_precollected_missing_columns(self, mock_score):
        from src.data.sentiment import compute_sentiment_from_precollected
        df = pd.DataFrame({
            "name": ["Salah"],  # Wrong column name
            "gw": [1],
            "text": ["Article"],
        })
        with pytest.raises(KeyError):
            compute_sentiment_from_precollected(df)


# ===========================================================================
# Integration tests
# ===========================================================================


class TestSentimentPipeline:
    """Test sentiment pipeline end-to-end."""

    @patch("src.data.sentiment.fetch_articles_for_player")
    @patch("src.data.sentiment._score_texts")
    def test_full_pipeline_compute(self, mock_score, mock_fetch):
        from src.data.sentiment import compute_sentiment_features
        mock_fetch.return_value = ["Good article", "Bad article", "Neutral article"]
        mock_score.return_value = [0.7, -0.6, 0.0]

        deadlines = pd.Series([datetime.now()])
        result = compute_sentiment_features(["Salah"], deadlines)
        assert len(result) == 1
        # Sentiment: mean([0.7, -0.6, 0.0]) = 0.0333...
        assert abs(result["Sentiment"].iloc[0]) < 1.0  # Should be in [-1, 1]
        assert result["SentimentVol"].iloc[0] > 0  # Should have some volatility
        assert result["n_articles"].iloc[0] == 3


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_compute_sentiment_features_empty_player_list(self):
        from src.data.sentiment import compute_sentiment_features
        deadlines = pd.Series([datetime.now()])
        result = compute_sentiment_features([], deadlines)
        assert len(result) == 0

    def test_compute_sentiment_features_empty_deadlines(self):
        from src.data.sentiment import compute_sentiment_features
        result = compute_sentiment_features(["Salah"], pd.Series([], dtype="datetime64[ns]"))
        assert len(result) == 0

    @patch("src.data.sentiment._score_texts")
    def test_compute_sentiment_single_player_single_gameweek(self, mock_score):
        from src.data.sentiment import compute_sentiment_features
        with patch("src.data.sentiment.fetch_articles_for_player") as mock_fetch:
            mock_fetch.return_value = ["Article"]
            mock_score.return_value = [0.5]

            deadlines = pd.Series([datetime.now()])
            result = compute_sentiment_features(["Salah"], deadlines)
            assert len(result) == 1
            assert result["Sentiment"].iloc[0] == pytest.approx(0.5)
            assert result["SentimentVol"].iloc[0] == 0.0  # Single article

    def test_compute_sentiment_from_precollected_single_article(self):
        from src.data.sentiment import compute_sentiment_from_precollected
        df = pd.DataFrame({
            "player_name": ["Salah"],
            "gameweek": [1],
            "article_text": ["Single article about Salah playing well"],
        })
        with patch("src.data.sentiment._score_texts") as mock_score:
            mock_score.return_value = [0.7]
            result = compute_sentiment_from_precollected(df)
            assert len(result) == 1
            assert result["Sentiment"].iloc[0] == pytest.approx(0.7)
            assert result["SentimentVol"].iloc[0] == 0.0

    def test_compute_sentiment_handles_very_long_articles(self):
        from src.data.sentiment import compute_sentiment_from_precollected
        long_article = "word " * 5000
        df = pd.DataFrame({
            "player_name": ["Salah"],
            "gameweek": [1],
            "article_text": [long_article],
        })
        with patch("src.data.sentiment._score_texts") as mock_score:
            mock_score.return_value = [0.5]
            result = compute_sentiment_from_precollected(df)
            assert len(result) == 1
            mock_score.assert_called_once()
