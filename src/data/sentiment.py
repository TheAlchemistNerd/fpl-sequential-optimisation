"""
News Sentiment Analysis for FPL Players.

Computes sentiment features from news articles mentioning each player in the
7-day window preceding a gameweek deadline.  Uses the DeBERTa-v3-small model
fine-tuned on news sentiment, following Tamimi & Tran (2025).

Features produced per player per gameweek:
    - Sentiment_p(t): mean sentiment score in [-1, 1]
    - SentimentVol_p(t): standard deviation of sentiment scores
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model loading (only when actually called)
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_MODEL_NAME = "SamLowe/roberta-base-go_emotions"  # fallback; swap for DeBERTa if available


def _load_model():
    """Load the sentiment model lazily to avoid importing transformers at module load."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        try:
            from transformers import pipeline
        except ImportError:
            logger.warning(
                "transformers not installed. Sentiment features will return neutral (0). "
                "Install with: pip install transformers torch"
            )
            return None, None

        # Try DeBERTa first (as specified in the dissertation)
        try:
            _tokenizer = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            _model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=False,
                truncation=True,
                max_length=512,
            )
            logger.info("Loaded sentiment model: %s", _tokenizer)
        except Exception:
            logger.warning("Failed to load DeBERTa sentiment model; returning neutral scores.")
            return None, None

    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Article fetching stub
# ---------------------------------------------------------------------------


def fetch_articles_for_player(
    player_name: str,
    gameweek_deadline: datetime,
    lookback_days: int = 7,
) -> list[str]:
    """
    Stub: fetch news articles mentioning *player_name* in the lookback window.

    In production, this would query The Guardian API, GDELT, or a news RSS feed.
    For now, returns an empty list so the pipeline can run without external APIs.

    Parameters
    ----------
    player_name : str
        Full name of the player.
    gameweek_deadline : datetime
        Deadline of the gameweek.
    lookback_days : int
        Number of days before the deadline to search.

    Returns
    -------
    List of article text strings.
    """
    # TODO: Implement real article fetching via GDELT / Guardian API.
    # Example GDELT query:
    #   https://api.gdeltproject.org/api/v2/doc/doc?query=player:"{name}"&mode=artlist&format=json
    logger.debug("fetch_articles_for_player stub: %s", player_name)
    return []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_texts(texts: list[str]) -> list[float]:
    """
    Score a list of texts with the loaded sentiment pipeline.

    Returns a list of floats in [-1, 1].
    """
    model, tokenizer = _load_model()
    if model is None:
        return [0.0] * len(texts)

    scores: list[float] = []
    for text in texts:
        if not text or not text.strip():
            scores.append(0.0)
            continue
        try:
            result = model(text[:512])  # truncate to model max length
            # The roberta sentiment pipeline returns label: POSITIVE/NEGATIVE/NEUTRAL
            label = result[0]["label"]
            score_val = result[0]["score"]
            if label == "positive":
                scores.append(score_val)
            elif label == "negative":
                scores.append(-score_val)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_sentiment_features(
    player_names: list[str],
    gameweek_deadlines: pd.Series,
) -> pd.DataFrame:
    """
    Compute sentiment features for all players across all gameweeks.

    Parameters
    ----------
    player_names : list[str]
        Unique player names.
    gameweek_deadlines : pd.Series[datetime]
        Deadline timestamps for each gameweek.

    Returns
    -------
    DataFrame with columns:
        player_name, gameweek, Sentiment, SentimentVol
    """
    rows: list[dict[str, float | str | int]] = []

    for gw_idx, deadline in enumerate(gameweek_deadlines, start=1):
        for name in player_names:
            articles = fetch_articles_for_player(name, deadline)
            if not articles:
                rows.append({
                    "player_name": name,
                    "gameweek": gw_idx,
                    "Sentiment": 0.0,
                    "SentimentVol": 0.0,
                    "n_articles": 0,
                })
                continue

            scores = _score_texts(articles)
            scores_arr = np.array(scores)
            rows.append({
                "player_name": name,
                "gameweek": gw_idx,
                "Sentiment": float(scores_arr.mean()),
                "SentimentVol": float(scores_arr.std(ddof=1)) if len(scores_arr) > 1 else 0.0,
                "n_articles": len(articles),
            })

    sentiment_df = pd.DataFrame(rows)
    logger.info(
        "Sentiment features computed: %d rows, %d players, %d gameweeks",
        len(sentiment_df), len(player_names), len(gameweek_deadlines),
    )
    return sentiment_df


def compute_sentiment_from_precollected(
    articles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute sentiment from a pre-collected articles DataFrame.

    Expected columns: player_name, gameweek, article_text

    This is the preferred method when articles are collected offline.
    """
    if articles_df.empty:
        return pd.DataFrame(columns=["player_name", "gameweek", "Sentiment", "SentimentVol", "n_articles"])

    results: list[dict[str, Any]] = []

    grouped = articles_df.groupby(["player_name", "gameweek"])
    for (player, gw), group in grouped:
        texts = group["article_text"].tolist()
        scores = _score_texts(texts)
        scores_arr = np.array(scores)
        results.append({
            "player_name": player,
            "gameweek": gw,
            "Sentiment": float(scores_arr.mean()),
            "SentimentVol": float(scores_arr.std(ddof=1)) if len(scores_arr) > 1 else 0.0,
            "n_articles": len(texts),
        })

    return pd.DataFrame(results)
