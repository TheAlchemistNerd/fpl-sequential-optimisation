"""Command line entry point for the walk-forward research study."""

from __future__ import annotations

import argparse

from src.backtesting.data import load_vaastav_merged
from src.research.walk_forward import run_walk_forward_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Vaastav merged gameweek CSV")
    parser.add_argument("--output", required=True, help="Directory for forecast and evaluation artifacts")
    parser.add_argument("--season", default="2023-24")
    parser.add_argument("--first-gameweek", type=int, default=7)
    parser.add_argument("--last-gameweek", type=int)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--max-stale-gameweeks", type=int, default=5)
    parser.add_argument("--players-per-position", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument(
        "--max-divergence-rate",
        type=float,
        default=0.01,
        help="Abort before forecasting when a model exceeds this retained-draw divergence rate",
    )
    parser.add_argument("--truncation", type=int, default=6)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from a contiguous saved forecast prefix that passed sampler diagnostics",
    )
    args = parser.parse_args()

    gameweeks = load_vaastav_merged(args.data)
    results = run_walk_forward_study(
        gameweeks,
        data_path=args.data,
        output_dir=args.output,
        season=args.season,
        first_gameweek=args.first_gameweek,
        last_gameweek=args.last_gameweek,
        lookback=args.lookback,
        max_stale_gameweeks=args.max_stale_gameweeks,
        players_per_position=args.players_per_position,
        seed=args.seed,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        target_accept_prob=args.target_accept,
        max_divergence_rate=args.max_divergence_rate,
        truncation=args.truncation,
        resume=args.resume,
    )
    print("Player-point prediction metrics:")
    print(results["prediction_metrics"].to_string(index=False))
    print("\nConstrained squad results:")
    columns = [
        "strategy", "evaluation_net_points", "evaluation_gross_points",
        "evaluation_transfer_hits", "evaluation_transfers", "evaluation_gameweeks",
    ]
    print(results["squad_summary"].loc[:, columns].to_string(index=False))


if __name__ == "__main__":
    main()
