# Current source architecture

## Project objective and implementation status

The project aims to support sequential FPL decisions over a season. At each gameweek, the intended manager uses observations to update beliefs about player form and selects a squad or action subject to budget, squad composition, club limits, transfer costs, and other game rules. The research documents also propose season-level planning and population effects.

The source is a collection of research implementations and prototypes, not a complete end-to-end manager. The existence of a module does not establish that its method is integrated into the pipeline or validated. The dissertation remains the statement of the broader research objective; this page describes only source structure and observed wiring.

## Package map

| Package | Responsibility in the source |
|---|---|
| `src/config/` | Environment-backed settings and project paths. |
| `src/api/` | FPL HTTP/authentication code and the FastAPI inference service. `data_fetcher.py` fetches and persists FPL data; it is more than a transport-only client. |
| `src/data/` | DataFrame acquisition helpers, feature engineering, preprocessing, feature selection, EDA, Understat scraping, and sentiment features. `src/data/fetcher.py` is a second FPL-facing acquisition path with overlapping endpoints. |
| `src/pipeline/` | Coordinates fetching, feature preparation, and the current model-update hook. The update hook logs pipeline information; it does not run a complete model-training lifecycle. |
| `src/models/` | NumPyro/JAX latent-form model experiments, count models, and belief-state utilities. Model maturity and statistical integration vary. |
| `src/optimization/selection/` | Squad selection and single-gameweek decision methods: knapsack baselines, MILP, Thompson sampling, captaincy, and chip timing. |
| `src/optimization/planning/` | MCTS node, rollout, and Thompson-sampling/MCTS orchestration. The current search is a shallow prototype rather than a full multi-level POMCP implementation. |
| `src/optimization/ddqn/` | DQN components and a simplified simulation environment. This is not wired into the live FPL pipeline. |
| `src/optimization/mfg/` | A numerical mean-field-game prototype using synthetic state/reward arrays, not an FPL population model. |
| `src/mlops/` | Drift calculations, a lazy MLflow wrapper, and a DVC setup helper. These pieces are not a complete deployed monitoring/retraining system. |
| `src/utils/` | JAX kernels and sequence preparation helpers used by model experiments. |

Selection and planning modules are grouped by responsibility under `src/optimization/selection/` and `src/optimization/planning/`.

## Observed data and decision flow

1. `src.api.data_fetcher.fetch_all_data()` requests FPL bootstrap, fixture, and gameweek data and writes local data files. Optional team-specific endpoints use authentication.
2. `src.pipeline.data_pipeline.engineer_features()` reads saved player and gameweek files and builds the feature table. The pipeline does not currently connect every documented external source or every preprocessing/selection step.
3. `update_hdp_models()` is a logging hook rather than a full incremental HDP training workflow.
4. Model and optimization modules can be called independently. The FastAPI recommendation route passes `points_per_game` values into the Thompson-sampling MILP selector; it does not load HDP posterior samples or persisted beliefs. The retraining route returns `not_configured` and does not queue work.

`src.data.fetcher` is a separate FPL acquisition path; it should not be mistaken for the pipeline's only client or for a pure DataFrame transformation module. Understat and sentiment modules are also not part of the main pipeline flow.

## Decision modules

- The MILP implements a constrained squad/lineup selection formulation. Its integration and edge cases should be judged from `selection/milp_solver.py`, not from older claims of production completeness.
- Thompson sampling includes both a simple budget-only path and a MILP-backed path. The simple path does not enforce the full FPL squad rules. The API currently supplies points-per-game values as inputs rather than posterior draws from the latent-form models.
- The MCTS package samples outcomes and evaluates candidate transfers, but its current tree and rollout do not implement the full sequential search described in the thesis.
- DDQN and MFG modules are separate prototypes and are not called by the live recommendation path.

## Tests

Tests live in `tests/` and mostly use synthetic data or mocks. Run them with `python -m pytest -q` after installing the dependencies in `requirements.txt`. The test suite has dependencies beyond `pytest`; a partial run in an environment missing JAX and several project dependencies is not a full-suite result.
