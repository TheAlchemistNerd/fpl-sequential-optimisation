# Sequential backtesting

## Purpose

The replay compares decision policies over a completed FPL season using a common initial squad, a chronological player-information set, a simulated bank and purchase prices, and realized gameweek points. The historical archive comes from [Vaastav Anand's Fantasy Premier League repository](https://github.com/vaastav/Fantasy-Premier-League), under `data/<season>/gws/`. The 2023–24 pilot used [`data/2023-24/gws/merged_gw.csv`](https://github.com/vaastav/Fantasy-Premier-League/blob/master/data/2023-24/gws/merged_gw.csv).

Check the selected season directory before downloading: the repository README's structure overview spells the merged filename `merged_gws.csv`, while its example path uses `merged_gw.csv`. The pilot metadata records the SHA-256 of the actual input file so the run can be tied to the exact dataset contents.

The initial challenger set is deliberately small:

| Policy | Decision rule | Status |
| --- | --- | --- |
| `hold` | Keep the initial squad and reselect the XI and captain from lagged form. | Control for transfer activity. |
| `form_milp` | Optimize the squad and XI from mean points in the prior five gameweeks. | Simple lagged prediction baseline. |
| `xg_milp` | Optimize from a prior-five-gameweek xG and xA expected-points proxy, including appearance and clean-sheet components. | Feature-based challenger, not a fitted points model. |
| `thompson_nb_milp` | Sample the existing heuristic negative-binomial point distribution, then optimize with the transfer MILP. | Repository implementation; it is not the thesis HDP/MCMC model. |
| `random_valid` | Optimize seeded random player scores under the same squad, budget, transfer, and hit constraints. | Feasible stochastic reference, not a uniform draw over all legal squads. |

The repository's current MCTS and DDQN modules are not included as historical challengers. Their current evaluation paths use simplified synthetic/static environments and do not replay FPL deadline information, legal weekly transfers, and actual season outcomes. They should be wired to this transition model before being compared.

## Data timing and accounting

Fixture rows are aggregated to one player-gameweek row. Minutes, points, xG, xA, and clean sheets are summed, retaining double-gameweek outcomes. For a decision before gameweek `t`, features are computed only from rows with `gameweek < t`; the target gameweek is used only for scoring. Rolling form treats missing player-weeks as zero after the player's first archived appearance.

The available merged archive has no official deadline availability snapshot, historical purchase price, manager squad, or bank ledger. The replay therefore uses the latest player observation strictly before the target gameweek as the candidate and price proxy, keeps current-squad players in the candidate pool when their rows go stale, initializes one randomized legal squad from GW1 metadata, and simulates purchase prices and bank from then on. The xP field is intentionally excluded because the archive README warns it can be revised after the event. Candidate freshness is configurable and defaults to five gameweeks.

The simulated 2023-24 wallet is held in tenths of a million. Sales at a profit return the purchase price plus half the profit rounded down to a tenth; losses return the latest observed market price. The solver constrains incoming purchases to bank plus sale proceeds. The season's two-free-transfer cap and four-point excess-transfer hit are applied in the weekly ledger. No chips are simulated. Starting XI score applies captain/vice-captain fallback and legal automatic substitutions. This is a reproducible engineering approximation, not a replay of actual FPL accounts.

## Run

```powershell
python -m src.backtesting `
  --data "$env:TEMP\fpl_2023_24_merged_gw.csv" `
  --season 2023-24 `
  --output reports/backtests/2023-24 `
  --seeds 42 73 101
```

The command writes `gameweeks.csv`, `summary.csv`, and `metadata.json`. Metadata includes the input SHA-256, parameters, and interpretation caveats. Add more pinned seasons before drawing general conclusions; a one-season run is an engine pilot, not evidence of out-of-sample superiority. Results also depend on the randomized initial squad, so use multiple seeds and report the distribution rather than only the best run.

## Next research steps

1. Verify the replay's accounting and leakage properties on the pinned 2023-24 archive.
2. Add multiple season archives with source URLs, retrieval dates, and checksums.
3. Add the thesis's proper probabilistic model and compare it to these simple baselines with identical information and decisions.
4. Only after the sequential transition path is validated, adapt MCTS and DDQN to the same legal state/action and scoring rules.
5. Complete the experiment design and report uncertainty across seasons and seeds before making strategy claims.
