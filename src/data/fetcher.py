"""
FPL API Data Acquisition Module.

Fetches data from the official Fantasy Premier League API endpoints:
- bootstrap-static/: players, teams, events
- fixtures/: all 380 Premier League matches
- element-summary/{EID}/: per-player historical data
- event/{GW}/live/: live gameweek data
- entry/{TID}/history/: manager history for benchmarking

Per-90 and form computations delegate to the canonical implementations in
``src.data.feature_engineering`` to avoid duplication.
"""

import logging
import time
from typing import Any

import pandas as pd
import requests

from src.data.feature_engineering import per90 as compute_per90

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FPL_BASE_URL: str = "https://fantasy.premierleague.com/api/"
FPL_BOOTSTRAP_URL: str = f"{FPL_BASE_URL}bootstrap-static/"
FPL_FIXTURES_URL: str = f"{FPL_BASE_URL}fixtures/"
FPL_ELEMENT_SUMMARY_URL: str = f"{FPL_BASE_URL}element-summary/{{eid}}/"
FPL_EVENT_LIVE_URL: str = f"{FPL_BASE_URL}event/{{gw}}/live/"
FPL_ENTRY_HISTORY_URL: str = f"{FPL_BASE_URL}entry/{{tid}}/history/"

# Position mapping: element_type -> position name
POSITION_MAP: dict[int, str] = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

# Status mapping
STATUS_MAP: dict[str, str] = {
    "a": "available",
    "i": "injured",
    "u": "unavailable",
    "d": "doubtful",
    "s": "suspended",
}

# HTTP session with retry logic
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return a cached requests.Session with retry adapter."""
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=requests.adapters.Retry(
                total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
            )
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
        _session.headers.update({
            "User-Agent": "FPL-Research-Agent/1.0",
            "Accept": "application/json",
        })
    return _session


def _fetch_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fetch JSON from *url* with exponential-backoff retry."""
    session = _get_session()
    try:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        raise


# ---------------------------------------------------------------------------
# Bootstrap Static
# ---------------------------------------------------------------------------


def get_bootstrap_data() -> dict[str, pd.DataFrame]:
    """
    Fetch the master bootstrap-static endpoint.

    Returns a dict with keys:
        - players:   DataFrame of all registered FPL players
        - teams:     DataFrame of all 20 Premier League clubs
        - events:    DataFrame of global gameweek summaries
    """
    logger.info("Fetching bootstrap-static data …")
    data = _fetch_json(FPL_BOOTSTRAP_URL)

    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    events = pd.DataFrame(data["events"])

    # Enrich players with readable position and status
    players["position"] = players["element_type"].map(POSITION_MAP)
    players["status_label"] = players["status"].map(STATUS_MAP)

    logger.info(
        "Bootstrap data: %d players, %d teams, %d events",
        len(players), len(teams), len(events),
    )
    return {"players": players, "teams": teams, "events": events}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def get_fixtures_data() -> pd.DataFrame:
    """
    Fetch all 380 Premier League fixtures for the current season.

    Returns a DataFrame with columns:
        id, team_h, team_a, kickoff_time, event,
        team_h_score, team_a_score, finished, …
    """
    logger.info("Fetching fixtures …")
    data = _fetch_json(FPL_FIXTURES_URL)
    fixtures = pd.DataFrame(data)
    logger.info("Fixtures fetched: %d matches", len(fixtures))
    return fixtures


# ---------------------------------------------------------------------------
# Element Summary (per-player history)
# ---------------------------------------------------------------------------


def get_element_summary(player_id: int) -> pd.DataFrame:
    """
    Fetch detailed gameweek-by-gameweek history for a single player.

    Parameters
    ----------
    player_id : int
        The FPL element ID of the player.

    Returns
    -------
    DataFrame with one row per gameweek the player has participated in.
    """
    url = FPL_ELEMENT_SUMMARY_URL.format(eid=player_id)
    data = _fetch_json(url)
    history = pd.DataFrame(data.get("history", []))
    history["player_id"] = player_id
    return history


def get_all_element_summaries(player_ids: list[int], delay: float = 0.15) -> pd.DataFrame:
    """
    Fetch element summaries for a list of player IDs.

    Parameters
    ----------
    player_ids : list[int]
        List of FPL player IDs.
    delay : float
        Seconds to sleep between requests to avoid rate-limiting.

    Returns
    -------
    DataFrame concatenating all players' gameweek histories.
    """
    all_histories: list[pd.DataFrame] = []
    total = len(player_ids)
    for idx, pid in enumerate(player_ids, 1):
        try:
            hist = get_element_summary(pid)
            all_histories.append(hist)
        except Exception as exc:
            logger.warning("Failed to fetch history for player %d: %s", pid, exc)
        if idx % 50 == 0:
            logger.info("Fetched element summaries: %d / %d", idx, total)
        time.sleep(delay)

    if not all_histories:
        logger.warning("No element summary data retrieved.")
        return pd.DataFrame()

    combined = pd.concat(all_histories, ignore_index=True)
    logger.info("Total element summary rows: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Event Live
# ---------------------------------------------------------------------------


def get_event_live(gameweek: int) -> pd.DataFrame:
    """
    Fetch live data for all players during a specific gameweek.

    Parameters
    ----------
    gameweek : int
        The gameweek number.

    Returns
    -------
    DataFrame with live stats, explain, dreamteam flag, etc.
    """
    url = FPL_EVENT_LIVE_URL.format(gw=gameweek)
    data = _fetch_json(url)
    elements = data.get("elements", [])
    live_df = pd.DataFrame(elements)
    # Flatten nested 'stats' dict if present
    if "stats" in live_df.columns:
        stats_norm = pd.json_normalize(live_df["stats"])
        live_df = pd.concat([live_df.drop(columns=["stats"]), stats_norm], axis=1)
    return live_df


# ---------------------------------------------------------------------------
# Entry History (manager benchmarking)
# ---------------------------------------------------------------------------


def get_entry_history(team_id: int) -> pd.DataFrame:
    """
    Fetch the gameweek-by-gameweek history of a specific FPL manager.

    Parameters
    ----------
    team_id : int
        The FPL entry (team) ID.

    Returns
    -------
    DataFrame with columns: event, points, total_points, rank, rank_sort,
    overall_rank, bank, value, event_transfers, event_transfers_cost, active_chip.
    """
    url = FPL_ENTRY_HISTORY_URL.format(tid=team_id)
    data = _fetch_json(url)
    history = pd.DataFrame(data.get("current", []))
    history["team_id"] = team_id
    return history


# ---------------------------------------------------------------------------
# Derived quantity helpers (delegates to canonical implementations)
# ---------------------------------------------------------------------------

# compute_per90 is re-exported from feature_engineering for backward compatibility:
#   from src.data.fetcher import compute_per90
#   compute_per90(series, minutes)  # → delegates to feature_engineering.per90


def compute_form(points_series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling average form over *window* gameweeks.

    Form_p(w) = (1/w) * sum_{k=0}^{w-1} points_p(t - k)

    Thin wrapper — the canonical implementation lives in
    ``feature_engineering.build_form_features`` for DataFrame-level usage.
    """
    return points_series.rolling(window=window, min_periods=1).mean()
