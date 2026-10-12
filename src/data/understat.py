"""
Understat Advanced Metrics Scraper.

Fetches Expected Goals (xG), Expected Assists (xA), npxG, xGChain, xGBuildup
and related metrics from understat.com via asynchronous web scraping.

Player-name matching between Understat and FPL is handled via fuzzy string
matching (Levenshtein distance) with a configurable similarity threshold.
"""

import asyncio
import codecs
import json
import logging
import re
from typing import Any

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz, process as fuzzy_process

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNDERSTAT_BASE_URL: str = "https://understat.com/"
UNDERSTAT_EPL_URL: str = "https://understat.com/league/EPL/{season}"
UNDERSTAT_PATTERN: str = r"{}\s+=\s+JSON\.parse\(\'(.*?)\'\)"
FUZZY_THRESHOLD: int = 80

# Cookie required to bypass Understat's simple bot check
_UNDERSTAT_COOKIE: dict[str, str] = {"beget": "begetok"}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _fetch_html(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch the raw HTML from *url*."""
    async with session.get(url, cookies=_UNDERSTAT_COOKIE, timeout=30) as resp:
        resp.raise_for_status()
        return await resp.text()


def _extract_json_from_html(html: str, data_type: str) -> list[dict[str, Any]]:
    """
    Parse embedded JSON from a <script> tag.

    Understat pages embed data as:
        var playersData = JSON.parse('...escaped json...')
    """
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")
    pattern = re.compile(UNDERSTAT_PATTERN.format(data_type))

    for script in scripts:
        if script.string is None:
            continue
        match = pattern.search(script.string)
        if match:
            raw = match.group(1)
            decoded = codecs.escape_decode(raw)[0].decode("utf-8")
            payload = json.loads(decoded)
            if isinstance(payload, dict):
                return [payload]
            return payload  # type: ignore[no-any-return]

    raise ValueError(f"Could not find '{data_type}' in page HTML.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_understat_season(
    session: aiohttp.ClientSession,
    season: int,
) -> pd.DataFrame:
    """
    Fetch player-level Understat data for one EPL season.

    Parameters
    ----------
    session : aiohttp.ClientSession
        Active async session.
    season : int
        Season start year (e.g. 2024 for 2024/25).

    Returns
    -------
    DataFrame with columns:
        player_name, games, time, goals, shots, xG, npxG,
        assists, xA, keys, yellow_cards, red_cards
    """
    url = UNDERSTAT_EPL_URL.format(season=season)
    logger.info("Fetching Understat EPL %d/%d …", season, season + 1)
    html = await _fetch_html(session, url)
    players_data = _extract_json_from_html(html, "playersData")
    df = pd.DataFrame(players_data)
    # Numeric columns are stored as strings in the JSON
    numeric_cols = [
        "games", "time", "goals", "shots", "xG", "npxG",
        "assists", "xA", "keys", "yellow_cards", "red_cards",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Understat %d/%d: %d players", season, season + 1, len(df))
    return df


async def fetch_understat_multiple_seasons(
    seasons: list[int],
) -> pd.DataFrame:
    """Fetch Understat data for multiple seasons concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_understat_season(session, s) for s in seasons]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    frames: list[pd.DataFrame] = []
    for season, result in zip(seasons, results):
        if isinstance(result, Exception):
            logger.error("Understat season %d failed: %s", season, result)
        else:
            result["season"] = season
            frames.append(result)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------


def match_understat_to_fpl(
    understat_df: pd.DataFrame,
    fpl_names: list[str],
    threshold: int = FUZZY_THRESHOLD,
) -> pd.DataFrame:
    """
    Match Understat player names to FPL player names using fuzzy matching.

    match(p_fpl) = argmax_q similarity(p_fpl, q)  if max_similarity > threshold

    Parameters
    ----------
    understat_df : DataFrame
        DataFrame with a 'player_name' column.
    fpl_names : list[str]
        List of FPL player names (e.g. "first_name second_name").
    threshold : int
        Minimum fuzzy similarity score (0-100).

    Returns
    -------
    DataFrame with an added 'fpl_name' column containing the best match
    (or NaN if no match exceeds the threshold).
    """
    if understat_df.empty or not fpl_names:
        understat_df["fpl_name"] = pd.NA
        return understat_df

    def _best_match(name: str) -> str | None:
        result = fuzzy_process.extractOne(name, fpl_names, scorer=fuzz.token_sort_ratio)
        if result is not None and result[1] >= threshold:
            return result[0]
        return None

    understat_df = understat_df.copy()
    understat_df["fpl_name"] = understat_df["player_name"].apply(_best_match)

    matched = understat_df["fpl_name"].notna().sum()
    logger.info(
        "Name matching: %d / %d Understat players matched to FPL (threshold=%d)",
        matched, len(understat_df), threshold,
    )
    return understat_df
