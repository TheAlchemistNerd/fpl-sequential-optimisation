"""Clean, targeted FPL data fetching with rate limiting and retry logic.

This module replaces the older ``data_fetcher.py`` script with:
* Rate-limiting between API calls
* Exponential back-off retry on transient failures
* Selective fetching (only relevant features, no noise)
* Protected endpoint access via :class:`src.api.auth_manager.FPLAuthManager`
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import aiohttp
import pandas as pd
import requests

from src.api.auth_manager import FPLAuthManager
from src.config.settings import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public (unauthenticated) endpoints
# ---------------------------------------------------------------------------
FPL_BOOTSTRAP_URL = f"{settings.FPL_BASE_URL}bootstrap-static/"
FPL_FIXTURES_URL = f"{settings.FPL_BASE_URL}fixtures/"
FPL_GAMEWEEK_LIVE_URL = f"{settings.FPL_BASE_URL}event/{{gw}}/live/"
FPL_ELEMENT_SUMMARY_URL = f"{settings.FPL_BASE_URL}element-summary/{{eid}}/"

# ---------------------------------------------------------------------------
# Protected (authenticated) endpoints
# ---------------------------------------------------------------------------
FPL_ENTRY_INFO_URL = f"{settings.FPL_BASE_URL}entry/{{tid}}/"
FPL_ENTRY_HISTORY_URL = f"{settings.FPL_BASE_URL}entry/{{tid}}/history/"
FPL_ENTRY_TRANSFERS_URL = f"{settings.FPL_BASE_URL}entry/{{tid}}/transfers/"
FPL_ENTRY_PICKS_URL = f"{settings.FPL_BASE_URL}entry/{{tid}}/event/{{gw}}/picks/"
FPL_ENTRY_LEAGUES_URL = f"{settings.FPL_BASE_URL}entry/{{tid}}/leagues/"


# ---------------------------------------------------------------------------
# Retry / rate-limit helpers
# ---------------------------------------------------------------------------

def _wait_with_backoff(attempt: int, max_retries: int) -> None:
    """Sleep with exponential back-off before retrying."""
    delay = min(2 ** attempt * 0.5, 30)  # cap at 30 s
    log.warning(
        "Retry attempt %d/%d – waiting %.1f s …", attempt, max_retries, delay
    )
    time.sleep(delay)


def _rate_limit() -> None:
    """Enforce a minimum gap between requests."""
    time.sleep(settings.API_RATE_LIMIT)


def _request_with_retry(
    url: str,
    max_retries: int | None = None,
    session: Optional[requests.Session] = None,
    **kwargs: Any,
) -> requests.Response:
    """GET *url* with retries and rate limiting.

    Returns
    -------
    requests.Response
        Raises ``requests.HTTPError`` after all retries are exhausted.
    """
    retries = max_retries or settings.API_MAX_RETRIES
    sess = session or requests.Session()

    for attempt in range(retries):
        try:
            _rate_limit()
            resp = sess.get(url, timeout=15, **kwargs)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, aiohttp.ClientError) as exc:
            log.warning("Request to %s failed: %s", url, exc)
            if attempt == retries - 1:
                raise
            _wait_with_backoff(attempt, retries)

    # Should never reach here
    raise RuntimeError(f"Unexpected fallthrough in _request_with_retry({url})")


# ---------------------------------------------------------------------------
# Public data fetchers
# ---------------------------------------------------------------------------

def get_bootstrap_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and return (players, teams) DataFrames."""
    log.info("Fetching FPL bootstrap-static data …")
    resp = _request_with_retry(FPL_BOOTSTRAP_URL)
    data = resp.json()
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    log.info("  → %d players, %d teams", len(players), len(teams))
    return players, teams


def get_fixture_data() -> pd.DataFrame:
    """Fetch the full fixture list."""
    log.info("Fetching FPL fixture data …")
    resp = _request_with_retry(FPL_FIXTURES_URL)
    fixtures = pd.DataFrame(resp.json())
    log.info("  → %d fixtures", len(fixtures))
    return fixtures


def get_gameweek_data(gameweek: int) -> pd.DataFrame:
    """Fetch live data for a single gameweek.

    Only the ``stats`` dict and basic element info are returned (no noise).
    """
    url = FPL_GAMEWEEK_LIVE_URL.format(gw=gameweek)
    log.info("Fetching gameweek %d data …", gameweek)
    resp = _request_with_retry(url)
    data = resp.json()

    elements = data.get("elements", [])
    if not elements:
        log.warning("  → No data returned for gameweek %d", gameweek)
        return pd.DataFrame()

    # Flatten: keep only the useful top-level fields
    rows = []
    for el in elements:
        row = {"id": el["id"], "gameweek": gameweek}
        stats = el.get("stats", {})
        row.update(stats)
        row["explain"] = el.get("explain", [])  # point breakdown
        row["in_dreamteam"] = el.get("in_dreamteam", False)
        row["multiplier"] = el.get("multiplier", 1)
        row["was_home"] = el.get("was_home")
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("  → %d player records", len(df))
    return df


# ---------------------------------------------------------------------------
# Protected data fetchers (require authentication)
# ---------------------------------------------------------------------------

def _protected_get(url: str, team_id: int, session: Optional[requests.Session] = None) -> dict:
    """GET a protected endpoint, authenticating if needed."""
    auth = FPLAuthManager(session=session)
    auth_session = auth.get_authorized_session()
    resp = _request_with_retry(url, session=auth_session)
    return resp.json()


def get_entry_info(team_id: Optional[int] = None) -> dict:
    """Basic info about an FPL manager's team."""
    tid = team_id or settings.FPL_TEAM_ID
    log.info("Fetching entry info for team %d …", tid)
    return _protected_get(FPL_ENTRY_INFO_URL.format(tid=tid), tid)


def get_entry_history(team_id: Optional[int] = None) -> dict:
    """Manager history: past seasons, chip plays, gameweek-by-gameweek summary."""
    tid = team_id or settings.FPL_TEAM_ID
    log.info("Fetching entry history for team %d …", tid)
    return _protected_get(FPL_ENTRY_HISTORY_URL.format(tid=tid), tid)


def get_entry_transfers(team_id: Optional[int] = None) -> list[dict]:
    """All transfers made by a manager."""
    tid = team_id or settings.FPL_TEAM_ID
    log.info("Fetching transfers for team %d …", tid)
    return _protected_get(FPL_ENTRY_TRANSFERS_URL.format(tid=tid), tid)


def get_entry_picks(gameweek: int, team_id: Optional[int] = None) -> dict:
    """Squad picks for a specific gameweek.

    Returns a dict with keys ``active_chip``, ``automatic_subs``,
    ``entry_history``, and ``picks``.
    """
    tid = team_id or settings.FPL_TEAM_ID
    url = FPL_ENTRY_PICKS_URL.format(tid=tid, gw=gameweek)
    log.info("Fetching picks for team %d / GW %d …", tid, gameweek)
    return _protected_get(url, tid)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def fetch_all_data(latest_gameweek: int) -> None:
    """Fetch and persist all data up to *latest_gameweek*.

    This is the main entry-point.  It writes CSVs to ``data/``.
    """
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -- Public data --------------------------------------------------------
    log.info("=== Public data ===")
    players, teams = get_bootstrap_data()
    players.to_csv(settings.DATA_DIR / "fpl_players.csv", index=False)
    teams.to_csv(settings.DATA_DIR / "fpl_teams.csv", index=False)

    fixtures = get_fixture_data()
    fixtures.to_csv(settings.DATA_DIR / "fpl_fixtures.csv", index=False)

    # -- Gameweek-by-gameweek data ------------------------------------------
    all_gw = []
    for gw in range(1, latest_gameweek + 1):
        gw_df = get_gameweek_data(gw)
        if not gw_df.empty:
            all_gw.append(gw_df)

    if all_gw:
        combined = pd.concat(all_gw, ignore_index=True)
        combined.to_csv(settings.DATA_DIR / "fpl_gameweek_data.csv", index=False)
        log.info("Saved %d total gameweek records.", len(combined))

    # -- Protected data (your squad) ----------------------------------------
    if settings.FPL_TEAM_ID:
        log.info("=== Protected data (team %d) ===", settings.FPL_TEAM_ID)
        try:
            info = get_entry_info()
            _save_json(info, settings.DATA_DIR / "my_entry_info.json")

            history = get_entry_history()
            _save_json(history, settings.DATA_DIR / "my_entry_history.json")

            transfers = get_entry_transfers()
            _save_json(transfers, settings.DATA_DIR / "my_transfers.json")

            # Picks for each fetched gameweek
            for gw in range(1, latest_gameweek + 1):
                picks = get_entry_picks(gw)
                if picks.get("picks"):
                    _save_json(
                        picks,
                        settings.DATA_DIR / f"my_picks_gw{gw}.json",
                    )
        except Exception:
            log.exception("Failed to fetch protected data – skipping.")

    log.info("Data fetching complete.")


def _save_json(data: Any, path: Path) -> None:
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info("  → Saved %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    GW = 5  # Adjust to your needs
    asyncio.run(fetch_all_data(GW))
