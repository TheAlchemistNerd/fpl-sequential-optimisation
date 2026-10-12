"""Command-line entry point for historical challenger replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.backtesting.data import load_vaastav_merged
from src.backtesting.engine import run_backtest, sha256_file
from src.backtesting.strategies import STRATEGIES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Vaastav merged_gw.csv path")
    parser.add_argument("--season", required=True, help="Dataset season label, e.g. 2023-24")
    parser.add_argument("--output", required=True, help="Directory for CSV and metadata outputs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--max-stale-gameweeks", type=int, default=5)
    parser.add_argument(
        "--free-transfer-cap", type=int, default=2,
        help="Maximum saved free transfers (2023-24 rules default to 2)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    result = run_backtest(
        load_vaastav_merged(data_path),
        strategies=args.strategies,
        seeds=args.seeds,
        season=args.season,
        lookback=args.lookback,
        max_stale_gameweeks=args.max_stale_gameweeks,
        free_transfer_cap=args.free_transfer_cap,
        dataset_sha256=sha256_file(data_path),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.gameweeks.to_csv(output / "gameweeks.csv", index=False)
    result.summary.to_csv(output / "summary.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(result.summary.to_string(index=False))
    print(f"\nWrote backtest outputs to {output.resolve()}")


if __name__ == "__main__":
    main()
