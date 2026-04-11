"""FPL API authentication for protected endpoints.

The FPL API uses session-based authentication via the Premier League
login portal.  After a successful login the session cookies must be
passed to every protected endpoint request.

Usage::

    from src.api.auth_manager import FPLAuthManager
    from src.config.settings import settings

    auth = FPLAuthManager()
    auth.login(settings.FPL_EMAIL, settings.FPL_PASSWORD)

    # Now pass auth.session to any request for protected endpoints.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from src.config.settings import settings

log = logging.getLogger(__name__)

# User-Agent header that the FPL login endpoint expects
_LOGIN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://fantasy.premierleague.com/",
    "Origin": "https://fantasy.premierleague.com",
}


class FPLAuthError(Exception):
    """Raised when FPL authentication fails."""


class FPLAuthManager:
    """Handles session-based login to the FPL API.

    Parameters
    ----------
    session : requests.Session, optional
        An existing session object.  If *None* a fresh session is created.
    """

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.session = session or requests.Session()
        self._is_authenticated = False

    # -- public API -------------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        return self._is_authenticated

    def login(self, email: str, password: str) -> None:
        """Authenticate against the Premier League login endpoint.

        On success the session cookies are populated and
        :attr:`is_authenticated` becomes ``True``.

        Raises
        ------
        FPLAuthError
            If the login request fails or the credentials are rejected.
        """
        log.info("Attempting FPL login for %s …", email)

        # Step 1 – GET the login page to obtain CSRF token and initial cookies
        page = self.session.get(
            settings.FPL_LOGIN_URL,
            headers=_LOGIN_HEADERS,
            timeout=15,
        )
        page.raise_for_status()

        # The CSRF token is embedded in the page
        csrf_token = self.session.cookies.get("csrftoken")
        if not csrf_token:
            raise FPLAuthError(
                "No CSRF token received.  The login page may have changed."
            )

        # Step 2 – POST credentials
        login_data = {
            "password": password,
            "login": email,
            "redirect_uri": "https://fantasy.premierleague.com/",
            "app": "plfpl-web",
        }
        login_headers = {
            **_LOGIN_HEADERS,
            "X-CSRFToken": csrf_token,
        }

        response = self.session.post(
            settings.FPL_LOGIN_URL,
            data=login_data,
            headers=login_headers,
            allow_redirects=False,
            timeout=15,
        )

        # The FPL login returns 302 on success, 200 with errors on failure
        if response.status_code not in (200, 302):
            raise FPLAuthError(
                f"Login request returned HTTP {response.status_code}."
            )

        # Check for error indicators in the response body
        if response.status_code == 200 and "error" in response.text.lower():
            raise FPLAuthError(
                "Login failed – check your email and password."
            )

        self._is_authenticated = True
        log.info("FPL login successful.")

    def ensure_authenticated(self) -> None:
        """Log in if not already authenticated."""
        if not self._is_authenticated:
            if not settings.FPL_EMAIL or not settings.FPL_PASSWORD:
                raise FPLAuthError(
                    "FPL_EMAIL and FPL_PASSWORD must be set in .env "
                    "to access protected endpoints."
                )
            self.login(settings.FPL_EMAIL, settings.FPL_PASSWORD)

    def get_authorized_session(self) -> requests.Session:
        """Return a session ready for protected endpoint calls."""
        self.ensure_authenticated()
        return self.session
