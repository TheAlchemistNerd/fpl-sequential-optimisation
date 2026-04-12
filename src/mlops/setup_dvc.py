"""
DVC Data Versioning Setup — Ch. 11 §11.3.

Initialises DVC for tracking data files (CSVs, belief states, models).
Data files are hashed and stored in a remote backend (S3, GDrive, local).
The .dvc metadata files are committed to Git for reproducibility.

Usage
-----
    python -m src.mlops.setup_dvc
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_dvc(data_dir: Path | None = None) -> None:
    """
    Initialise DVC and configure data tracking.

    Parameters
    ----------
    data_dir : Path | None
        Directory containing data files to track.
        Defaults to project root `data/`.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data"

    logger.info("Setting up DVC data versioning…")

    # 1. Check DVC is installed
    try:
        result = subprocess.run(
            ["dvc", "--version"],
            capture_output=True, text=True, check=True,
        )
        logger.info("DVC version: %s", result.stdout.strip())
    except FileNotFoundError:
        logger.error("DVC not found. Install with: pip install dvc")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logger.error("DVC initialisation failed.")
        sys.exit(1)

    # 2. Initialise DVC (if not already)
    dvc_root = data_dir.parent / ".dvc"
    if not dvc_root.exists():
        logger.info("Initialising DVC in %s", data_dir.parent)
        subprocess.run(
            ["dvc", "init"],
            cwd=data_dir.parent,
            check=True,
        )

    # 3. Track data files
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        logger.info("Tracking %d CSV files", len(csv_files))
        for f in csv_files:
            subprocess.run(
                ["dvc", "add", str(f)],
                cwd=data_dir.parent,
                check=True,
            )
    else:
        logger.warning("No CSV files found in %s", data_dir)

    # 4. Track belief states
    belief_dir = data_dir / "belief_states"
    if belief_dir.exists():
        subprocess.run(
            ["dvc", "add", str(belief_dir)],
            cwd=data_dir.parent,
            check=True,
        )

    # 5. Configure remote (default: local for development)
    remote_dir = data_dir.parent / "dvc_remote"
    remote_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["dvc", "remote", "add", "-d", "local_remote", str(remote_dir)],
        cwd=data_dir.parent,
        check=True,
    )

    logger.info("DVC setup complete. Remote: %s", remote_dir)
    logger.info("Run 'dvc push' to upload data to remote storage.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_dvc()
