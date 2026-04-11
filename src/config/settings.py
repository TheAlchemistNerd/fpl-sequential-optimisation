"""Configuration management using environment variables."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parents[2]
_env_path = _project_root / ".env"
load_dotenv(_env_path)


class Settings:
    """Central configuration container."""

    # FPL API
    FPL_BASE_URL = "https://fantasy.premierleague.com/api/"
    FPL_LOGIN_URL = "https://users.premierleague.com/accounts/login/"

    # Credentials
    FPL_EMAIL: Optional[str] = os.getenv("FPL_EMAIL")
    FPL_PASSWORD: Optional[str] = os.getenv("FPL_PASSWORD")
    FPL_TEAM_ID: Optional[int] = int(os.getenv("FPL_TEAM_ID", "0")) or None
    FPL_LEAGUE_ID: Optional[int] = int(os.getenv("FPL_LEAGUE_ID", "0")) or None

    # API behaviour
    API_RATE_LIMIT: float = float(os.getenv("API_RATE_LIMIT", "1.0"))
    API_MAX_RETRIES: int = int(os.getenv("API_MAX_RETRIES", "3"))

    # Data paths
    DATA_DIR = _project_root / "data"

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing required settings."""
        missing = []
        if not cls.FPL_EMAIL:
            missing.append("FPL_EMAIL")
        if not cls.FPL_PASSWORD:
            missing.append("FPL_PASSWORD")
        if not cls.FPL_TEAM_ID:
            missing.append("FPL_TEAM_ID")
        return missing


settings = Settings()
