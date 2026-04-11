# Codebase Architecture

> **Developer-facing reference.** This document describes the actual code structure, module
> responsibilities, layer boundaries, and design decisions. It is separate from the academic
> docs in `docs/` (which cover methodology, theory, and dissertation chapters).

---

## Overview

This project implements an autonomous decision-making agent for Fantasy Premier League (FPL).
The architecture combines Bayesian nonparametric models (Hierarchical Dirichlet Processes)
with reinforcement learning and constrained optimisation for weekly squad selection.

**Domain**: Sequential optimisation under uncertainty — each gameweek, the agent must select
a squad of 15 players (with budget, team structure, and transfer constraints) to maximise
expected total points over a planning horizon.

**Key architectural decision**: Separate the *data domain* (what information is available
about players) from the *infrastructure layer* (how we fetch it from external APIs) from the
*model layer* (how we reason about latent form states) from the *decision layer* (how we
optimise squad selection).

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Orchestration Layer  (src/pipeline/)                    │
│  Wires data → features → models → decisions             │
├─────────────────────────────────────────────────────────┤
│  Decision Layer  (src/optimization/ — future)            │
│  Thompson Sampling, dDQN, Knapsack, Budget constraints   │
├─────────────────────────────────────────────────────────┤
│  Model Layer  (src/models/)                              │
│  HDP-GMM, HDP-HMM, HDP-Poisson-Gamma, HDP-NegBin        │
├─────────────────────────────────────────────────────────┤
│  Data Domain Layer  (src/data/)                          │
│  Aggregation, feature engineering, preprocessing, EDA    │
│  External source adapters (Understat, sentiment)          │
├─────────────────────────────────────────────────────────┤
│  Infrastructure Layer  (src/api/)                        │
│  HTTP clients, auth, rate limiting, API gateways         │
├─────────────────────────────────────────────────────────┤
│  Configuration  (src/config/)                            │
│  Environment variables, settings singleton               │
└─────────────────────────────────────────────────────────┘
```

---

## Module Inventory

### Infrastructure Layer — `src/api/`

| Module | Responsibility | Key exports |
|---|---|---|
| `auth_manager.py` | FPL session-based login (CSRF cookies, credential validation). Pure infrastructure — no domain logic. | `FPLAuthManager`, `FPLAuthError` |
| `data_fetcher.py` | **FPL API Client only**. HTTP transport layer with rate limiting + exponential backoff retry. Knows nothing about features, models, or decisions. | `get_bootstrap_data()`, `get_fixture_data()`, `get_gameweek_data()`, `get_entry_info()`, `get_entry_history()`, `get_entry_transfers()`, `get_entry_picks()`, `fetch_all_data()` |
| `inference_service.py` | FastAPI HTTP service (stub). The web-facing API for external consumers. | FastAPI app, `/health`, `/recommendations` endpoints |

**Boundary rule**: `src/api/` modules never import from `src/data/` or `src/models/`. They are pure infrastructure — they speak HTTP/JSON, not domain objects.

### Data Domain Layer — `src/data/`

| Module | Responsibility | Key exports |
|---|---|---|
| `fetcher.py` | **Data aggregator**. Composes API client calls into raw DataFrames. Owns `POSITION_MAP`, `STATUS_MAP`, and the `requests.Session` with retry adapter for public endpoints. | `get_bootstrap_data()` (returns dict of DataFrames), `get_fixtures_data()`, `get_element_summary()`, `get_all_element_summaries()`, `get_event_live()`, `get_entry_history()` |
| `understat.py` | **External source adapter**. Async web scraping of Understat for advanced metrics (xG, xA, npxG). Handles embedded JSON extraction and fuzzy name matching to FPL player names. | `fetch_understat_season()`, `fetch_understat_multiple_seasons()`, `match_understat_to_fpl()` |
| `sentiment.py` | **External source adapter**. News sentiment analysis via transformer models. Lazy model loading, graceful degradation when `transformers` not installed. | `fetch_articles_for_player()`, `compute_sentiment_features()`, `compute_sentiment_from_precollected()` |
| `feature_engineering.py` | **Feature transformation**. Per-90 normalisation, rolling form windows, fixture difficulty ratings, contextual features. The canonical source for `_per90()` and `build_form_features()`. | `build_core_features()`, `build_advanced_features()`, `build_ict_features()`, `build_form_features()`, `build_fixture_features()`, `build_contextual_features()`, `assemble_feature_matrix()`, `FEATURE_GROUPS` |
| `preprocessing.py` | **Data cleaning**. Missing value imputation (structural zeros, median by position group), feature standardisation, position-based splitting for separate models. | `impute_missing_values()`, `FeatureStandardiser`, `split_by_position()`, `build_position_standardisers()` |
| `feature_selection.py` | **Dimensionality reduction**. Mutual information, correlation threshold, Lasso (L1), tree-based importance, consensus/union methods. | `mutual_information_select()`, `correlation_threshold_select()`, `lasso_select()`, `tree_importance_select()`, `select_final_features()`, `FINAL_FEATURE_LIST` |
| `eda.py` | **Exploratory data analysis**. Univariate/bivariate/time-series/position-specific plotting. All figures saved to `reports/figures/`. | `plot_points_distribution()`, `plot_xg_vs_goals()`, `plot_correlation_heatmap()`, `plot_form_trajectory()`, `run_full_eda()` |

**Boundary rule**: `src/data/` modules never import from `src/api/`. They accept DataFrames as input and produce DataFrames as output. They are agnostic to whether the data came from the FPL API, a CSV, or a database.

### Model Layer — `src/models/`

| Module | Responsibility | Key exports |
|---|---|---|
| `hdp_gmm.py` | Hierarchical Dirichlet Process — Gaussian Mixture Model with Normal-Wishart base measure. The **canonical source** for `DEFAULT_TRUNCATION` and `stick_breaking_weights`. | `hdp_gmm_model()`, `run_hdp_gmm()`, `DEFAULT_TRUNCATION`, `NORMAL_WISHART_PRIOR`, `stick_breaking_weights()` |
| `hdp_hmm.py` | HDP-Hidden Markov Model with Gaussian-Wishart emissions. Extends HDP-GMM with temporal Markov chain over latent states. | `hdp_hmm_model()`, `run_hdp_hmm()` |
| `hdp_poisson_gamma.py` | HDP-Poisson-Gamma for discrete count data (goals, assists, clean sheets). | `hdp_poisson_gamma_model()`, `run_hdp_poisson_gamma()` |
| `hdp_negbin.py` | HDP-Negative-Binomial for over-dispersed total FPL points. | `hdp_negbin_model()`, `run_hdp_negbin()` |

**Boundary rule**: Model modules import from `hdp_gmm.py` for shared constants (`DEFAULT_TRUNCATION`, `stick_breaking_weights`). They use NumPyro/Pyro for probabilistic programming and are pure functions — no side effects, no global state.

### Orchestration Layer — `src/pipeline/`

| Module | Responsibility | Key exports |
|---|---|---|
| `data_pipeline.py` | Weekly data-collection pipeline. Wires `fetch_all_data()` → `engineer_features()` → `update_hdp_models()`. Runnable standalone or via Prefect/Airflow. | `run_pipeline()`, `engineer_features()`, `update_hdp_models()` |

### Configuration — `src/config/`

| Module | Responsibility | Key exports |
|---|---|---|
| `settings.py` | Environment variable loading via `python-dotenv`. Singleton `settings` instance with validation. | `Settings`, `settings`, `validate()` |

---

## Bounded Contexts & Data Flow

### Context 1: Data Acquisition

```
External APIs (FPL, Understat, News)
    │
    ▼
src/api/auth_manager.py  (authentication only)
src/api/data_fetcher.py  (HTTP transport: rate limit, retry, JSON decode)
    │
    ▼
src/data/fetcher.py      (data aggregator: DataFrame assembly, POSITION_MAP, STATUS_MAP)
src/data/understat.py    (async scraper + fuzzy name matching)
src/data/sentiment.py    (news article fetching + transformer scoring)
    │
    ▼
Raw DataFrames (data/*.csv, data/*.json)
```

**Key decision**: `src/api/data_fetcher.py` is **infrastructure** — it handles HTTP mechanics.
`src/data/fetcher.py` is **data domain** — it knows about FPL player positions, status codes,
and how to structure raw DataFrames. The separation means you can swap the HTTP client
(e.g., add caching, change retry strategy) without touching data logic, and vice versa.

### Context 2: Feature Engineering

```
Raw DataFrames (data/*.csv)
    │
    ▼
src/data/feature_engineering.py  (per-90 rates, form, FDR, contextual features)
    │
    ▼
src/data/preprocessing.py        (imputation, standardisation, position splitting)
    │
    ▼
src/data/feature_selection.py    (MI, correlation, Lasso, tree-based selection)
    │
    ▼
Feature matrix X ∈ R^(N×D), target vector y
```

**Key decision**: The `_per90()` helper lives **only** in `feature_engineering.py`. All other
modules that need per-90 normalisation import from there. This is the canonical source of
truth — no duplication.

### Context 3: Modelling

```
Feature matrix X, target y
    │
    ▼
src/models/hdp_gmm.py         (continuous features → latent form clusters)
src/models/hdp_hmm.py         (temporal sequences → latent state trajectories)
src/models/hdp_poisson_gamma  (count data → goal-scoring rates)
src/models/hdp_negbin.py      (over-dispersed data → point distributions)
    │
    ▼
Posterior distributions, predictive densities
```

### Context 4: Orchestration

```
Weekly schedule (Prefect/Airflow/cron)
    │
    ▼
src/pipeline/data_pipeline.py
    ├── fetch_all_data()         (from src/api/data_fetcher.py)
    ├── engineer_features()      (from src/data/feature_engineering.py)
    └── update_hdp_models()      (from src/models/)
    │
    ▼
Updated posteriors, metrics logged to MLflow
```

---

## Redundancy Resolution

### What was redundant and how it was resolved

| Redundancy | Before | After |
|---|---|---|
| **Per-90 computation** | `compute_per90()` in `fetcher.py:257`, `_per90()` in `feature_engineering.py:28` — identical formula, different names | Single `_per90()` in `feature_engineering.py`. `fetcher.py` imports from there. |
| **Rolling form** | `compute_form()` in `fetcher.py:266`, `build_form_features()` in `feature_engineering.py:134` — same rolling average | Single `build_form_features()` in `feature_engineering.py`. |
| **Player enrichment** | `enrich_bootstrap_players()` in `fetcher.py:279` — computes per-90 for same metrics as `build_core/advanced/ict_features` | Deleted. Use the feature engineering pipeline instead. |
| **Understat constants** | `UNDERSTAT_BASE_URL` and `UNDERSTAT_LEAGUE_URL` in `src/api/data_fetcher.py` — **never used** (actual scraping is in `src/data/understat.py`) | Deleted from `src/api/data_fetcher.py`. |
| **`DEFAULT_TRUNCATION = 30`** | Defined independently in `hdp_gmm.py`, `hdp_negbin.py`, `hdp_poisson_gamma.py` — 3 copies | Single definition in `hdp_gmm.py`. Others import: `from src.models.hdp_gmm import DEFAULT_TRUNCATION`. |
| **Pipeline feature engineering** | `data_pipeline.py:engineer_features()` reimplements rolling averages (`_roll_3` suffix) instead of using `feature_engineering.py` (`Form_3` suffix) | Rewired to call `src/data/feature_engineering.py`. |
| **Root-level legacy files** | `data_fetcher.py` and `hdp_gmm.py` at project root — old prototypes superseded by `src/` equivalents | Deleted. |

---

## Test Architecture

All tests follow TDD conventions. Each module in `src/` has a corresponding test file in
`tests/`:

| Source module | Test file | Tests |
|---|---|---|
| `src/api/auth_manager.py` | `tests/test_api.py` | Auth flow, settings validation |
| `src/api/data_fetcher.py` | `tests/test_api_data_fetcher.py` | Rate limiting, retry, public/protected endpoints, orchestration |
| `src/data/fetcher.py` | `tests/test_fetcher.py` | Bootstrap, fixtures, element summary, per-90, form, enrichment |
| `src/data/preprocessing.py` | `tests/test_preprocessing.py` | Imputation, standardisation, position splitting |
| `src/data/feature_engineering.py` | `tests/test_feature_engineering.py` | Core, advanced, ICT, form, fixture, contextual features, assembly |
| `src/data/feature_selection.py` | `tests/test_feature_selection.py` | MI, correlation, Lasso, tree-based, consensus/union |
| `src/data/understat.py` | `tests/test_understat.py` | JSON extraction, async fetching, fuzzy name matching |
| `src/data/sentiment.py` | `tests/test_sentiment.py` | Lazy loading, scoring, pre-collected articles |
| `src/data/eda.py` | `tests/test_eda.py` | Univariate, bivariate, time series, position patterns |
| `src/pipeline/data_pipeline.py` | `tests/test_data_pipeline.py` | Pipeline orchestration, feature engineering, CLI |
| `src/models/hdp_gmm.py` | `tests/test_hdp_gmm.py` | Stick-breaking, Normal-Wishart, MCMC |
| `src/models/hdp_hmm.py` | `tests/test_hdp_hmm.py` | HDP-HMM structure and inference |
| `src/models/hdp_poisson_gamma.py` | `tests/test_hdp_other_models.py` | Poisson-Gamma, Negative-Binomial |

**Total test count**: ~350+ tests across all modules.

**Testing conventions**:
- pytest with `pytest-cov` for coverage
- `unittest.mock.MagicMock` and `patch` for API mocking
- Synthetic data fixtures for deterministic tests
- MCMC tests run short chains (100 warmup, 200 samples, 1 chain) for speed
- Async tests use `pytest-asyncio`

---

## Dependencies

### Core ML/Stats
- `numpy==2.3.4`, `pandas==2.3.3`, `scipy==1.16.3`, `scikit-learn==1.7.2`

### Probabilistic Programming
- `jax==0.8.0`, `jaxlib==0.8.0`
- `numpyro==0.19.0`, `pyro-ppl==1.9.1`

### Web/Scraping
- `requests==2.32.3`, `aiohttp==3.13.2`
- `beautifulsoup4==4.14.2`
- `fuzzywuzzy==0.18.0`, `python-Levenshtein==0.27.3`

### Production
- `fastapi>=0.115.0`, `uvicorn[standard]>=0.30.0`
- `python-dotenv>=1.0.0`
- `mlflow>=2.11.0`, `prefect>=2.14.0`, `dvc>=3.0.0`

### Testing
- `pytest>=8.0`, `pytest-cov>=5.0`, `pytest-asyncio`

---

## Design Principles

1. **Single Responsibility**: Each module has one reason to change. `auth_manager.py` changes
   when login mechanics change. `feature_engineering.py` changes when feature definitions
   change. They never change for the same reason.

2. **Dependency Inversion**: Data modules accept DataFrames (abstraction) rather than knowing
   about HTTP clients (concrete). API modules return JSON/DataFrames rather than knowing about
   features or models.

3. **No Leaky Abstractions**: `src/api/data_fetcher.py` does not do per-90 normalisation.
   `src/data/feature_engineering.py` does not make HTTP requests. Each layer stays in its lane.

4. **Testability**: Every module can be tested in isolation with mocks or synthetic data. No
   module requires a live internet connection to run its unit tests (all HTTP is mocked).

5. **Reproducibility**: Data is versioned via DVC. Settings are loaded from `.env`. Every
   experiment links to specific data, code, and model versions.
