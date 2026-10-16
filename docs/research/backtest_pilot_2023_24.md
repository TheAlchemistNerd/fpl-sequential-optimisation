# 2023–24 sequential-backtest pilot

This companion research note records the first end-to-end policy replay. The thesis remains unchanged; this document keeps implementation evidence and future experiments in a separate research-first track.

## Pilot result

All policies used the same randomly drawn GW1 squad for seed 42 and were replayed through GW38. Total net points include four-point excess-transfer hits.

| Policy | Net points | Gross points | Transfer hits | Transfers |
| --- | ---: | ---: | ---: | ---: |
| Lagged xGI proxy + MILP | 2,008 | 2,036 | 28 | 45 |
| Five-GW form + MILP | 1,937 | 2,025 | 88 | 60 |
| Heuristic NB Thompson + MILP | 1,124 | 1,712 | 588 | 185 |
| Hold squad, optimize XI/captain | 728 | 728 | 0 | 0 |
| Random-valid objective + MILP | -103 | 625 | 728 | 220 |

These are one-seed engineering-pilot results. They demonstrate that the replay produces materially different policies and accounts for transfer penalties; they do not establish that xGI is superior or that the thesis model works. The Thompson policy here samples from the existing heuristic negative-binomial implementation, not the thesis HDP/MCMC posterior. The xGI policy is a hand-specified expected-points proxy, not a trained model. The random policy is intentionally stochastic and not uniform over all legal squads. The report was refreshed after correcting the bench-substitution rule to skip a playing substitute when their entry would make the final formation illegal.

The large hit totals for the Thompson and random policies expose a useful next research question: how transfer churn changes when a policy values future free transfers and uncertainty, rather than only sampled or noisy one-week points. The comparison should be repeated with multiple seeds and additional seasons before answering it.

## Protocol and data boundary

The run used Vaastav Anand's [Fantasy Premier League repository](https://github.com/vaastav/Fantasy-Premier-League), specifically [`data/2023-24/gws/merged_gw.csv`](https://github.com/vaastav/Fantasy-Premier-League/blob/master/data/2023-24/gws/merged_gw.csv), with SHA-256 `e9c09c8856f1c86b4f920f46ddd5033af83409439dfda53be925df2a3e7c8a9e`. The replay used a five-gameweek feature window, a five-gameweek player-freshness window, a £100m starting budget, and a two-free-transfer cap. Features for GW `t` use rows strictly before GW `t`. xP is excluded. Double-gameweek fixture rows are summed before feature and score calculations. The README's directory overview spells the merged filename `merged_gws.csv`, while its example uses `merged_gw.csv`; verify the selected season directory when retrieving other seasons. The input CSV is not bundled with this report; its checksum identifies the file used for this run.

The archive has no official deadline snapshots, historical availability, actual manager squads, or purchase-price ledgers. Player eligibility and prices use the latest pre-GW archive row, while each simulated policy maintains its own bank and purchase prices after the shared initial squad. Therefore this is a controlled policy comparison over an explicit proxy state, not a reconstruction of actual 2023–24 manager decisions. See [the backtesting implementation note](../engineering/backtesting.md) and the emitted [weekly ledger](../../reports/backtests/2023-24/gameweeks.csv), [summary](../../reports/backtests/2023-24/summary.csv), and [run metadata](../../reports/backtests/2023-24/metadata.json).

The follow-on walk-forward study is documented in [walk_forward_study.md](walk_forward_study.md). Its core runner produces lagged rolling-form, HDP-GMM, and HDP-HMM forecasts; a separate Ridge pass adds a direct signed feature-to-points challenger on the same cutoff weeks. The latest GW7–GW23 prefix has four-predictor scoring and constrained replay on the same 17 target gameweeks. The 2023–24 sampler is still running, with its durable core forecast prefix at GW24; the separate four-predictor replay remains through GW23, and no full-season comparison is available yet.

## Remaining research questions

1. Do the rolling-form, direct-points Ridge, HDP-GMM, and HDP-HMM results reproduce across additional seasons, seeds, and longer sampler runs with convergence diagnostics?
2. Does the xGI policy remain ahead of form across seeds and other archived seasons under the same information and transfer rules?
3. Does a thesis-faithful posterior sampling policy improve net points or risk-adjusted outcomes after accounting for transfer hits and saved free transfers?
4. Do transfer decisions improve on hold and simple lagged baselines when player availability and price data are aligned to actual deadlines?

The next report should include paired seed/season differences, uncertainty intervals, gross points, hits, transfers, and prediction metrics. MCTS and DDQN should enter only after they use this same chronological information set and legal FPL transition path.
