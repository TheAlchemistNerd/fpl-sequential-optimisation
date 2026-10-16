# Walk-forward next-gameweek study

## Purpose and research question

This experiment builds the first reproducible research-first comparison for the sequential FPL project. It asks two separate questions:

1. Do the latent-state challengers improve forecasts of player points in the next gameweek over a lagged rolling-form baseline?
2. Do any forecast differences improve a constrained squad policy after transfers, available bank, free transfers, hits, lineup choice, captaincy, and automatic substitutions are applied?

The first question is the primary predictive evaluation. Squad points are a downstream decision result and are not a substitute for forecast accuracy. The study keeps the thesis chapters unchanged and records implementation evidence here.

## Models and their relationship to the thesis

The comparison contains three predictors:

| Predictor | Forecast construction | Role in the comparison |
| --- | --- | --- |
| `rolling_form` | Mean points over the five preceding gameweeks; unobserved weeks after first appearance contribute zero | Transparent, non-Bayesian reference |
| `hdp_gmm` | Gaussian emissions over lagged features; posterior mixture probabilities are mapped to earlier points | Independent latent-state challenger linked to [Chapter 5](../thesis/chapters/chapter_05_hdp_gmm_gaussian_wishart.md) |
| `hdp_hmm` | Gaussian emissions and shared, action-conditioned transition rows; filtered state probabilities are mapped to earlier points | Temporal latent-state challenger linked to [Chapter 6](../thesis/chapters/chapter_06_hdp_hmm_temporal_dynamics.md) |

The model priors share global stick-breaking weights with the player-specific GMM weights and HMM transition rows. In the finite GMM approximation, each player's weight vector is drawn directly as `Dirichlet(alpha * beta)`, the finite-dimensional DP marginal; this avoids an independent local-stick parameterization. The HMM draws each transition row from the same shared-base form. Both use a truncated state space, Gaussian emissions, and Wishart priors over full precision matrices. The Normal–Inverse–Wishart specification in Chapter 5 uses degrees of freedom `D + 2`; its equivalent precision-Wishart prior therefore uses `D + 2`, which is now reflected in the model code.

This alignment concerns the shared-base prior only; it does not make the current modules a complete implementation of every inference algorithm described in the thesis. Chapter 5 derives collapsed Gibbs/NIW updates, while the current `src/models/hdp_gmm.py` module uses NumPyro NUTS over continuous parameters and integrates the finite mixture assignments out exactly in the likelihood. Chapter 6 explains FFBS and mean-field extensions, while `src/models/hdp_hmm.py` uses a masked forward likelihood inside NUTS and does not implement backward state sampling or mean-field coupling. This runner adds the point-reward mapping and forecast protocol around those current model interfaces. NUTS targets an acceptance probability of 0.95 by default; each run records divergences and mean acceptance probability, which must be checked before treating a fold's forecasts as usable. The thesis remains the conceptual treatment; source behavior and this experiment report define what was run.

The Vaastav archive has no manager-action sequence. The HMM therefore receives one neutral action category. The action-conditioned interface is exercised, but action effects cannot be estimated in this study. The thesis's mean-field extension and reinforcement-learning decision layer are also outside this experiment.

Neither latent model emits FPL points directly. In each fold, posterior state probabilities on the earlier training rows are used to estimate a state-conditioned points reward. The estimate is the responsibility-weighted mean of prior points, smoothed toward the training-wide mean with a prior mass of two observations. Target predictions are the posterior state probabilities multiplied by these reward means. This is an explicit bridge from the feature-state models to the points objective; it is not a direct posterior predictive distribution for points and does not capture points uncertainty.

The GMM uses the posterior player-specific mixture for the fixed fit cohort and the global mixture for other players. The HMM uses the fitted shared transition model and filters each target player's lagged feature history; a player without a history starts from the global weights. The fit cohort is sampled by position from GW1 with a fixed seed and reused across folds. This limits NUTS runtime and means the study is a cohort pilot, rather than an all-player posterior fit.

Sequence padding is fold-local: target GW *t* uses at most *t - 2* training snapshots per player (GW2 through GW*t-1). The HMM does not scan empty future gameweeks, and the GMM likelihood contains only training snapshots, not padded future rows. Masks retain shorter player histories where needed.

## Point-in-time protocol

Let `t` be the target gameweek and `u` an earlier training gameweek. For each target fold:

1. Construct every target feature from archive rows with gameweek strictly less than `t`. No target-gameweek outcome is used to fit, scale, or predict that fold.
2. Reconstruct earlier examples as historical snapshots: the feature row for `u` uses only rows before `u`, while its label is the realized points in `u`. Training labels therefore satisfy `u < t`.
3. Refit the standard scaler on the fold's earlier training rows only. Apply that scaler to training rows, target rows, and the HMM's earlier player history.
4. Fit the GMM and HMM using the same cohort, feature columns, cutoff, and fold-specific transformation. The deterministic seed is derived from the study seed and target gameweek.
5. Append the target fold to `predictions.csv`, flush the file, and call `fsync` before advancing to the next fold. The forecast-generation function has no target-outcome lookup.
6. Only after all requested folds have been persisted does the evaluator read target outcomes and create scored predictions and metrics.

Each forecast row records its target week, feature cutoff (`t - 1`), player ID, position, price, predictor values, cohort membership, and number of earlier training rows. `candidate_fresh` marks players whose latest archive observation falls within the configured stale window. Forecasts are also generated for older archived players so the constrained replay can assign a score to a held player retained by the manager simulation; forecast-accuracy metrics use only the freshness-filtered pool, common to all three predictors.

This is an auditable chronological boundary, not proof that the archive reproduces what was visible at an official FPL deadline. Vaastav's merged gameweek rows are not dated deadline snapshots. The archive also lacks official player availability, price-at-deadline, and manager-action histories. Eligibility and features remain archive-based proxies.

## Historical data and test-fixture boundary

The study input is the real Vaastav Anand 2023-24 merged gameweek archive linked below, not generated player data. The diagnostic and planned walk-forward run read `fpl_2023_24_merged_gw.csv`; the project loader produced 28,742 aggregated player-gameweek rows across GW1–GW38 and 865 unique players. Its SHA-256 is `e9c09c8856f1c86b4f920f46ddd5033af83409439dfda53be925df2a3e7c8a9e` and is recorded in run metadata alongside the source URL. The source CSV is kept outside the repository so the research artifacts do not duplicate the archive.

Small synthetic or hand-built fixtures appear only in unit tests. They isolate contracts such as point-in-time cutoffs, posterior-array shapes, probability normalization, diagnostic-gate behavior, and figure generation. Those fixtures are not inputs to the historical forecasts, scored metrics, or squad replay. The real-data study artifacts and test-fixture checks must be reported separately.

## Feature and outcome construction

The existing point-in-time feature builder produces the following inputs from the preceding five-week window:

| Feature | Definition used in the study |
| --- | --- |
| `form_points` | Mean total points per gameweek in the window, with missing weeks after first appearance set to zero |
| `form_std` | Sample standard deviation of those weekly points |
| `xgi_points` | Expected goal/assist and clean-sheet points proxy plus recent appearance contribution |
| `appearance_probability` | Fraction of window weeks with positive minutes |
| `expected_minutes` | Mean minutes per week, including zero-minute weeks |
| Position indicators | One-hot GK, DEF, MID, and FWD columns |

Inputs are converted to finite numeric values and standardized within each fold. The rolling-form comparator uses the same `form_points` feature, while the latent-state models receive all nine numeric columns.

Vaastav fixture rows are aggregated to player-gameweek rows by summing points, minutes, expected goals, expected assists, and clean sheets. Double-gameweek totals therefore remain combined. Price and identity metadata use the final fixture row in that week. Missing player-gameweek rows contribute zero points when constructing the rolling window and training labels.

## Predictive evaluation

All predictors are scored on identical fresh-candidate player-gameweek observations. The report includes:

- mean absolute error (MAE),
- root mean squared error (RMSE),
- mean signed error (`prediction - actual`), and
- Spearman rank correlation between predicted and realized points.

Metrics are reported overall, by gameweek, and by position. MAE and RMSE quantify point error; signed error exposes systematic over- or under-prediction; Spearman correlation measures ranking rather than calibration. The script preserves raw predictions and actuals so the summary can be independently recomputed.

## Constrained squad evaluation

The three prediction streams feed the existing wallet-aware squad MILP through the same sequential replay engine:

| Policy | MILP player score |
| --- | --- |
| `form_milp` | Lagged rolling-form points |
| `hdp_gmm_milp` | Fold-specific HDP-GMM point prediction |
| `hdp_hmm_milp` | Fold-specific HDP-HMM point prediction |

For each seed, all three policies begin with the same randomly drawn legal GW1 squad and simulated bank. They hold that squad through the configured warm-up and begin optimizing in the first scored gameweek. The replay then applies the same 15-player positional rules, club cap, budget, simulated purchase/sell prices, free-transfer rollover, four-point hits, lineup and captain selection, and autosubstitution scoring to each policy. The default study starts strategy evaluation at GW7 to provide five completed gameweeks before the first target prediction; predictions may continue through GW38.

Squad results include gross and net points, hits, transfers, evaluation-window totals, and weekly squad composition. Starting from a shared squad controls the initial-roster difference within each seed. The archive-based price and availability proxies, single initial seed, and absence of chips or manager-specific ownership history limit interpretation. A higher replay total for one seed does not establish a generally superior strategy.

## Reproduction

Use the Vaastav Anand [2023-24 merged gameweek CSV](https://github.com/vaastav/Fantasy-Premier-League/blob/master/data/2023-24/gws/merged_gw.csv). Obtain that source separately; this repository stores its SHA-256 in the run metadata rather than duplicating the input archive.

From the repository root, with the project dependencies installed:

~~~powershell
python -m src.research --data "$env:TEMP\fpl_2023_24_merged_gw.csv" --output reports/research/walk_forward_2023-24 --season 2023-24 --first-gameweek 7 --warmup 25 --samples 25 --chains 1 --target-accept 0.95 --max-divergence-rate 0.01 --truncation 4 --players-per-position 6 --seed 42
~~~

The command uses a resource-constrained exploratory setting: one chain, 25 warmup and 25 posterior draws, four latent states, six sampled players per position, and a 0.95 NUTS target acceptance. These settings are not enough to support inference or claims about posterior convergence. The runner also has a screening gate: by default it stops before writing a forecast fold if a model's retained-draw divergence rate exceeds 1% or its mean acceptance probability falls below 0.5. Passing this screen is not proof of convergence. The actual attempted run stopped after GW7 because its divergence diagnostics were unacceptable; see the troubleshooting record below. The CLI accepts `--last-gameweek`, `--lookback`, and `--max-stale-gameweeks` for targeted runs. For inferential use, profile a longer multi-chain configuration, inspect effective sample sizes, split R-hat, traces and divergences, and retain enough draws after warmup.

The run metadata records the input checksum, requested folds, cutoff and feature rules, selected fit cohort, model feature list, seed, truncation, MCMC settings, and JAX device count. Each target fold derives separate GMM and HMM random keys from the seed and gameweek. For exact reproduction, retain the input archive and the dependency versions used for the run.

The longer locally measured configuration uses 200 warmup steps, 50 retained draws, one chain, target acceptance 0.99, four states, and six fit players per position. The full-season runner writes to `reports/research/walk_forward_2023-24/full_season_2023-24/`.

An interrupted run can be continued by repeating its command with `--resume`, the same source file and model settings, and the same output directory. Resume validates the source hash and coverage, fit roster, feature and sampler settings, contiguous player-week ledger, and both models' passing diagnostics before appending the next fold. It refuses a failed, partial, mismatched, or diagnostic-free prefix; use a new output directory when changing the data or configuration.

## Compute placement and troubleshooting record

The checked-out run used the repository's local Windows virtual environment at `C:\Users\Philtechber&Ent\Downloads\fpl-sequential-optimisation\venv`; it did not run on a remote ChatGPT worker. The full 2023–24 command was first attempted locally with 12 fit players per position, six latent states, two chains, and 50 warmup plus 50 retained draws. The first Bayesian fold took several CPU minutes and had not written a forecast row, so that attempt was stopped before any results were scored.

A smaller GW7 diagnostic then used six fit players per position (24 total), four latent states, one chain, 25 warmup plus 25 retained draws, and NumPyro's default 0.8 target acceptance. It wrote a 718-player GW7 forecast ledger row set, then recorded these NUTS diagnostics:

| Model | Divergent retained draws | Mean acceptance probability |
| --- | ---: | ---: |
| HDP-GMM | 21 / 25 | 0.1011 |
| HDP-HMM | 25 / 25 | 0.0115 |

The run was stopped after GW7. These forecasts are diagnostic artifacts only: their posterior draws are unreliable, they were not joined to outcomes or used for squad replay, and they must not be treated as model results. A subsequent full run should overwrite the partial ledger and set its metadata status to `complete` only after every fold, scoring, replay, and figure generation succeeds.

The diagnostic prompted two model corrections before another fold was accepted. First, Chapter 5 specifies Normal–Inverse–Wishart degrees of freedom `D + 2`; the equivalent precision-Wishart prior in the model had used `D + 1`. The source now uses `D + 2`, which gives the corresponding inverse-Wishart covariance a finite prior mean under the chapter's identity scale. Second, the finite GMM player weights now use the direct simplex marginal `Dirichlet(alpha * beta)` rather than local Beta sticks. Both GMM and HMM NUTS runners now target a 0.95 acceptance probability, and the setting is configurable and recorded.

After those changes, a GW7-only diagnostic with 100 warmup and 50 retained draws per model, one chain, four states, and the same 24-player fit cohort completed in about seven CPU minutes. The updated diagnostics were:

| Model | Divergent retained draws | Mean acceptance probability |
| --- | ---: | ---: |
| HDP-GMM | 4 / 50 | 0.9102 |
| HDP-HMM | 3 / 50 | 0.9420 |

This improved substantially over the initial run, but 4/50 and 3/50 divergences still fail the 1% screen. Those temporary forecasts were not scored or replayed.

A stricter GW7 diagnostic then used 200 warmup steps, 50 retained draws, one chain, four states, and a 0.99 target acceptance probability. It took about 16 minutes wall time on the local CPU. Both models had zero divergent retained draws and passed the configured screen:

| Model | Divergent retained draws | Mean acceptance probability | Screen |
| --- | ---: | ---: | --- |
| HDP-GMM | 0 / 50 | 0.9793 | Pass |
| HDP-HMM | 0 / 50 | 0.9878 | Pass |

The pass is only a basic screening result. One chain and 50 draws do not establish convergence, so this fold is a diagnostic pilot, not evidence that the posterior is reliable. The saved pre-outcome ledger contains 718 players. After persisting it, the study joined GW7 outcomes and evaluated all three predictors on the same 718 fresh-candidate player-weeks:

| Predictor | MAE | RMSE | Mean error | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Lagged rolling form | 1.0964 | 2.1433 | -0.0833 | 0.6449 |
| HDP-GMM | 2.1003 | 2.7316 | 0.9680 | 0.1775 |
| HDP-HMM | 2.1071 | 2.7878 | 0.8397 | 0.1280 |

On this single week, rolling form had lower point error and stronger rank correlation than either latent model. The same seed-42 initial squad was held through GW6, then each predictor fed the existing MILP for GW7:

| Policy | GW7 net points | Gross points | Transfer hits | Transfers |
| --- | ---: | ---: | ---: | ---: |
| Rolling-form MILP | 25 | 45 | 20 | 7 |
| HDP-GMM MILP | 9 | 9 | 0 | 2 |
| HDP-HMM MILP | 13 | 13 | 0 | 2 |

This one-gameweek replay is downstream evidence for the pipeline, not a strategy result. It uses the archive's proxy prices and eligibility, one initial seed, and no chips or manager-specific ownership history. The detailed forecast ledger, scored rows, metrics, replay, diagnostics, metadata, and figures are saved under [`reports/research/walk_forward_2023-24/diagnostic_pass_gw7`](../../reports/research/walk_forward_2023-24/diagnostic_pass_gw7/). The full 2023–24 walk-forward comparison remains incomplete: GW8–GW38 were not fitted, scored, or replayed. Because a single fold took about 16 minutes locally and later folds have longer histories, this CPU setup cannot support a minutes-scale full-season estimate.

The practical target for a faster full-season run is a cloud VM or notebook with more compute. A GPU is only a candidate acceleration path: JAX must use a compatible CUDA build, and the first-fold time and diagnostics must be benchmarked there before estimating total runtime. No remote machine, notebook, or GPU run has been started as part of this pilot, and no minutes-scale runtime is claimed.

## Artifacts

The output directory contains:

| File | Contents |
| --- | --- |
| `predictions.csv` | Durable pre-outcome forecast ledger, one row per player and target gameweek |
| `fit_diagnostics.csv` | Fold/model training size, fit cohort size, active states, divergences, and mean acceptance probability |
| `predictions_scored.csv` | Fresh-candidate predictions joined to realized points after forecast generation |
| `prediction_metrics.csv` | Overall error and ranking metrics by predictor |
| `prediction_week_metrics.csv` | Metrics by gameweek and predictor |
| `prediction_position_metrics.csv` | Metrics by position and predictor |
| `squad_gameweeks.csv` | Weekly sequential replay ledger, including squad, transfers, hits, and points |
| `squad_summary.csv` | Full-season and evaluation-window replay totals |
| `figures/prediction_error_by_gameweek.png` | MAE and RMSE by target week for each predictor |
| `figures/prediction_rank_correlation_by_gameweek.png` | Spearman ranking correlation by target week |
| `figures/prediction_mae_by_position.png` | MAE across GK, DEF, MID, and FWD observations |
| `figures/cumulative_squad_points.png` | Cumulative evaluation-window net points for each replay policy |
| `run_metadata.json` | Source fingerprint, run configuration, results, replay settings, and caveats |

The prediction ledger is the key audit artifact: it can be scored again without refitting, while `run_metadata.json` identifies the source and configuration. The metadata status advances from `running` to `forecasts_complete` and finally `complete` as artifacts are produced. An interrupted fold leaves a partial ledger and a `running` status, which should be treated as incomplete.

At this report revision, only the nested `diagnostic_pass_gw7/` directory contains the screened, scored, and replayed one-fold diagnostic (`status: diagnostic_complete`). The parent-directory ledger and metadata preserve the earlier 0.8-target run that failed sampling diagnostics; those forecasts were not scored and must not be combined with the passing GW7 artifacts. No full-season artifact set has been produced.

The figures are generated from the saved metrics and replay ledger after the CSV artifacts have been written; they do not fit models or fetch outcomes. Error charts show point accuracy, the rank chart shows player-order agreement, and the cumulative replay chart shows realized constrained-portfolio outcomes. Use the CSVs for exact values and the PNGs for visual comparison; a smooth or rising line does not establish a statistically reliable advantage.

## Reading results and limitations

Inspect the predictive metrics before the squad totals. A latent-state model that produces a better MILP outcome on a single replay but worse point or ranking metrics may have benefited from portfolio constraints, prices, or gameweek variance; it has not thereby demonstrated better forecasts. Compare all three columns on the same player-week panel and all squad policies on the same seed and evaluation weeks.

The inference runner records divergences and mean acceptance probability. The GMM's active-state count uses posterior assignment mass; the HMM's active-state count uses global state-weight mass, so those counts are descriptive and should not be treated as directly comparable. These are basic warning signals, not a convergence assessment. Before making statistical claims, rerun with longer chains, inspect trace plots, effective sample sizes and split R-hat, vary seeds and fit cohorts, and repeat across seasons. The initial one-season result is an engineering and research pilot only.

The current feature vector and reward mapping are practical bridge components for wiring the thesis models into a next-week exercise. They do not implement every thesis proposal: direct points likelihoods, the full action sequence, mean-field coupling, chips, richer point-in-time price/availability snapshots, multi-season validation, and decision policies using posterior uncertainty remain future research work. The thesis remains unchanged; this document records the narrower implemented experiment and its evidence.
