"""Shared pytest configuration."""

import sys
from pathlib import Path

# Always ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
