# Implementation Plan — FPL Sequential Optimisation System

> **Status:** Active development. Last updated: April 2026.
> **Scope:** From current state (data pipeline + HDP models partially implemented) to full
> production-ready system with decision engines, MILP optimiser, and MLOps pipeline.

---

## Current State Assessment

### Completed (Phase 1 — Foundation)

| Module | Status | Notes |
|---|---|---|
| `src/api/auth_manager.py` | ✅ Complete | FPL session auth, CSRF, credential validation |
| `src/api/data_fetcher.py` | ✅ Complete | HTTP transport, rate limiting, retry, public + protected endpoints |
| `src/api/inference_service.py` | ⚠️ Stub | FastAPI app with 501 stubs — needs wiring to decision engines |
| `src/config/settings.py` | ✅ Complete | Env-var based Settings, validation |
| `src/data/fetcher.py` | ✅ Complete | Data aggregator, delegates per-90 to feature_engineering |
| `src/data/understat.py` | ✅ Complete | Async scraper, fuzzy name matching |
| `src/data/sentiment.py` | ✅ Complete | Lazy model loading, graceful degradation |
| `src/data/feature_engineering.py` | ✅ Complete | Per-90 (canonical), form, FDR, contextual, assembly |
| `src/data/preprocessing.py` | ✅ Complete | Imputation, standardisation, position splitting |
| `src/data/feature_selection.py` | ✅ Complete | MI, correlation, Lasso, tree-based, consensus, input guards |
| `src/data/eda.py` | ✅ Complete | Univariate/bivariate/time-series/position plots |
| `src/pipeline/data_pipeline.py` | ✅ Complete | Wires fetch → canonical feature_engineering → model update |
| `src/models/hdp_gmm.py` | ✅ Complete | NUTS via NumPyro, Normal-Wishart, stick-breaking |
| `src/models/hdp_hmm.py` | ✅ Complete | Markov layer, transition matrices, FFBS structure |
| `src/models/hdp_poisson_gamma.py` | ✅ Complete | Poisson-Gamma for count data, imports DEFAULT_TRUNCATION |
| `src/models/hdp_negbin.py` | ✅ Complete | Negative-Binomial for over-dispersed points, imports DEFAULT_TRUNCATION |
| `src/models/belief_state.py` | ✅ **NEW** | Belief state construction, feature→points mapper, NB posterior |
| `tests/` (14 test files) | ✅ **243/302 passing** | TDD suite — 58 failures remaining (mock/HDP API issues) |

### Not Started (Phase 2–6)

| Component | Chapters | Priority |
|---|---|---|
| HDP-GMM I²GMM Extension | Ch. 5 §5.5 | High |
| JAX/XLA Inference (Gibbs/FFBS) | Ch. 7 §7.5 | High |
| Belief-State MDP Bridge | Ch. 5 §5.7, Ch. 7 §7.9 | High |
| ASUGS Adaptive Concentration | Ch. 7 §7.8 | Medium |
| Thompson Sampling (Step A) | Ch. 8 §10.2 | High |
| TS + MCTS (Step B) | Ch. 8 §8.8 | High |
| DDQN (Step C) | Ch. 9 §9.5–9.8 | High |
| Mean-Field Games (MFG) | Ch. 9 §9.10 | High |
| MILP Constrained Optimiser | Ch. 10 §10.6 | High |
| Contextual MAB for Captaincy | Ch. 10 §10.7 | Medium |
| Chip Deployment Optimiser | Ch. 10 §10.10 | Medium |
| GenTS-Explore for Initial Squad | Ch. 10 §10.8 | Medium |
| MLOps: Drift Detection (KL-Div) | Ch. 11 | High |
| MLOps: Model Registry | Ch. 11 | Medium |
| MLOps: DVC Data Versioning | Ch. 11 | Low |

---

## Phase 2: Inference & Belief States (Weeks 1–3)

### 2.1 JAX/XLA Accelerated Inference
**Rationale**: MCMC in pure Python is prohibitively slow. JAX's XLA compiler translates recursive inference kernels (FFBS, Gibbs) into optimized machine code, hardware-agnostic across CPU/GPU.

**Files to create:**
```
src/models/hdp_gmm_gibbs.py      — JAX-native Collapsed Gibbs sampler
src/models/hdp_hmm_ffbs.py       — JAX lax.scan-powered FFBS
src/models/jax_kernels.py        — JIT-compiled MVN densities and NIW updates
```

**Implementation steps:**
1. Implement **I²GMM** cluster assignment updates with `vmap` for multi-modal form profiles.
2. Implement **FFBS** temporal loops using `jax.lax.scan` for action-conditioned transitions $\pi_k(a)$.
3. JIT-compile MVN log-density and NIW sampling kernels.
4. Add convergence diagnostics: Trace plots, R̂, and ESS using `numpyro.diagnostics`.

### 2.2 Belief-State MDP Bridge
**Rationale**: Belief states are the primary output — the probability distribution over a player's latent form states across $S$ posterior samples.

**Files to create:**
```
src/models/belief_state.py       — Belief tensor construction (N x K)
src/models/feature_to_points.py  — Negative-Binomial posterior predictive mapping
```

**Implementation steps:**
1. Compute belief vectors: $b_j^{(s)}(t)$ for all players $j$ and MCMC samples $s$.
2. Implement **Negative-Binomial parameterisation**: map features to points distributions $y \sim NB(r, p)$.
3. Construct the 757-dimension state representation for Step C (DDQN).

### 2.3 Drift Monitoring & ASUGS
**Implementation steps:**
1. Implement KL-divergence monitoring: $KL(b_t || b_{t-1})$.
2. Trigger **ASUGS online retraining** when structural population shifts exceed $\gamma_{threshold}$.

---

## Phase 3 & 4: Tactical Planning (Weeks 4–8)

### 3.1 Step A: Thompson Sampling & MILP
- **Task:** Implement **piecewise linear transfer cost encoding** in `milp_solver.py` to handle the $(f, 4.0)$ discontinuity exactly.
- **Task:** Build `thompson_sampling.py` to draw points $ỹ_j$ from Negative-Binomial posteriors and solve the myopic knapsack.

### 3.2 Step B: Multi-GW Horizon (TS+MCTS)
- **Task:** Implement **action-conditioned transition matrices** $\pi_k(a)$ for scenario planning.
- **Task:** Build the MCTS search tree with **Belief-State Pruning** (limiting branches to the top-100 candidates).
- **Task:** Integrate chip deployment logic (Wildcard, Bench Boost) into MCTS rollouts.

### 4.1 Specialized OptimizationBandits
- **Task:** Implement **LinUCB Contextual Bandit** for captaincy selection using player belief states as context.
- **Task:** implement **GenTS-Explore** for Initial Squad / Wildcard combinatorial pure exploration.

**Tests:** `tests/test_milp_solver.py` (~25 tests), `tests/test_knapsack.py` (~20 tests),
`tests/test_captaincy.py` (~15 tests), `tests/test_chip_optimizer.py` (~15 tests)

---

## Phase 5: Strategic Planning (Step C) (Weeks 9–12)

### 5.1 Step C: Double DQN (DDQN)
- **Task:** Build the **757-dimension state encoder** following Ch. 9 §9.5.3 (Squad, Budget, Beliefs, Chips, GW, FT).
- **Task:** Implement the **Soft Q-Learning** update logic for MaxEnt RL consistency.
- **Task:** Enable **LayerNorm** and Boltzmann exploration derived from the principled MaxEnt objective.

### 5.2 Mean-Field Game (MFG) Dynamics
- **Task:** Implement the **MFG Equilibrium Solver** using fixed-point iteration between HJB (value) and Fokker-Planck (population distribution).
- **Task:** Develop the mean-field price predictor driven by population transfer activity $m_t$.
- **Task:** Integrate Mean-Field Coupling into the transition matrices for strategic competitive modelling.

---

Phase 6 Upgrade: Add Shift-based Drift Detection (KL-divergence of belief states).
### 6.1 MLOps: Drift Detection & Registry
- **Task:** Implement **Population-Level Belief State KL-Divergence Monitoring** to detect structural season shifts.
- **Task:** Configure automated retraining triggers when $KL(m_t || m_{t-1}) > \gamma$.
- **Task:** Set up MLflow Model Registry for staging TS (Step A) vs DDQN (Step C) models.

### 6.2 FastAPI Service Completion

**Files to update:**
```
src/api/inference_service.py   — Replace 501 stubs with real endpoints
```

**Implementation steps:**
1. `GET /health` → already working
2. `GET /beliefs/{player_id}/{gameweek}` → return belief state vector
3. `POST /recommendations` → run TS/MCTS/DDQN, return squad + captain + chips
4. `POST /retrain` → trigger full pipeline (fetch → engineer → HDP update → log)
5. Add Pydantic models for request/response validation
6. Add rate limiting and API key authentication

### 6.3 DVC Data Versioning

**Implementation steps:**
1. Initialise DVC: `dvc init`
2. Track data files: `dvc add data/*.csv`
3. Configure remote: AWS S3 / Google Drive
4. Link data version to git commit in MLflow metadata

---

## Phase 7: Testing & Validation (Weeks 15–16)

### 7.1 Fix Remaining Test Failures

81 tests currently failing. Categorised by root cause:

| Category | Count | Fix |
|---|---|---|
| Missing `python-dotenv` | 7 | ✅ Installed |
| `__init__.py` missing exports | 31 | ✅ Fixed |
| `enrich_bootstrap_players` deleted | 3 | ✅ Rewrote tests |
| Understat async mock issues | 7 | ⚠️ Partially fixed |
| EDA edge case assertions | 6 | Fix assertion expectations |
| Feature selection edge cases | 4 | Add input validation guards |
| HDP Wishart `df` argument | 9 | Fix NumPyro API (use `concentration` not `df`) |
| Sentiment grouping logic | 1 | Fix test mock |
| Poisson-Gamma sample count | 1 | Fix assertion |

### 7.2 Integration Tests

**Files to create:**
```
tests/test_integration_pipeline.py     — End-to-end: fetch → engineer → HDP → TS → MILP
tests/test_integration_ddqn.py         — End-to-end: belief states → DDQN → season simulation
tests/test_integration_mcts.py         — End-to-end: TS + MCTS for chip timing
```

### 7.3 Empirical Validation (Chapter 12)

**Scripts to create:**
```
scripts/run_ablation_1.py      — HDP-GMM vs HDP-HMM belief quality
scripts/run_ablation_2.py      — TS+MCTS vs DDQN
scripts/run_ablation_8.py      — TS vs Boltzmann vs ε-greedy
scripts/run_chapter_12.py      — Full empirical results, tables, figures
```

---

## Dependency Graph

```
Phase 2 (Posterior Inference)
    │
    ├── belief_state.py ← hdp_gmm_gibbs.py, hdp_hmm_ffbs.py
    └── asugs.py ← belief_state.py
    │
Phase 3 (Thompson Sampling)
    │
    ├── thompson_sampling.py ← belief_state.py
    ├── ts_mcts.py ← thompson_sampling.py, hdp_hmm_ffbs.py
    └── posterior_predictive.py ← belief_state.py
    │
Phase 4 (Constrained Optimisation)
    │
    ├── milp_solver.py ← thompson_sampling.py, ts_mcts.py
    ├── knapsack.py ← milp_solver.py
    ├── captaincy.py ← milp_solver.py
    └── chip_optimizer.py ← milp_solver.py
    │
Phase 5 (DDQN)
    │
    ├── state_representation.py ← belief_state.py
    ├── q_network.py ← state_representation.py
    ├── training_loop.py ← q_network.py, fpl_environment.py
    ├── boltzmann_policy.py ← training_loop.py
    ├── fpl_environment.py ← belief_state.py
    └── mfg/ ← training_loop.py, hdp_hmm_ffbs.py
    │
Phase 6 (MLOps)
    │
    ├── mlflow_tracker.py ← ALL (logs everything)
    ├── inference_service.py ← thompson_sampling.py, ts_mcts.py, ddqn, milp_solver.py
    └── monitoring.py ← mlflow_tracker.py
    │
Phase 7 (Testing & Validation)
    │
    └── Integration tests + Ablation scripts
```

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| FPL API changes breaking data fetcher | High | Version-locked API contracts, fallback to cached data |
| NumPyro Wishart API incompatibility | Medium | Use `LKJ` prior as alternative, or manual Cholesky parameterisation |
| MCTS computational explosion | High | Aggressive action pruning, parallel rollout execution |
| DDQN training instability | Medium | LayerNorm, gradient clipping, target network, extensive hyperparameter search |
| Insufficient historical data | Medium | Data augmentation via simulation, transfer learning from prior seasons |
| MILP solver timeout on large instances | Low | Time limit with best-known solution fallback, warm-start from greedy |

---

## Module Map (Final Target State)

```
src/
├── api/                          # Infrastructure Layer
│   ├── auth_manager.py
│   ├── data_fetcher.py
│   └── inference_service.py      # ← Complete (replace stubs)
├── config/
│   └── settings.py
├── data/                         # Data Domain Layer
│   ├── fetcher.py
│   ├── understat.py
│   ├── sentiment.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   └── eda.py
├── models/                       # Model Layer
│   ├── hdp_gmm.py                # NUTS implementation
│   ├── hdp_gmm_gibbs.py          # ← NEW: JAX-native I²GMM
│   ├── hdp_hmm_ffbs.py           # ← NEW: JAX lax.scan trajectory sampling
│   ├── jax_kernels.py            # ← NEW: JIT MVN/NIW samplers
│   ├── belief_state.py           # ← NEW: Belief-State MDP bridge
│   ├── feature_to_points.py      # ← NEW: NB Points posterior mapping
│   ├── asugs.py                  # ← NEW: Adaptive drift updates
├── optimization/                  # Decision Layer
│   ├── thompson_sampling.py      # ← NEW: Step A
│   ├── posterior_predictive.py   # ← NEW: Points posterior
│   ├── ts_mcts.py                # ← NEW: Step B
│   ├── mcts_node.py              # ← NEW
│   ├── rollout.py                # ← NEW
│   ├── milp_solver.py            # ← NEW: MILP constraints
│   ├── knapsack.py               # ← NEW: DP, B&B, Greedy, GA
│   ├── captaincy.py              # ← NEW: Contextual MAB
│   ├── chip_optimizer.py         # ← NEW: Chip timing bandit
│   ├── ddqn/                     # ← NEW: Step C
│   │   ├── state_representation.py
│   │   ├── q_network.py
│   │   ├── replay_buffer.py
│   │   ├── action_space.py
│   │   ├── training_loop.py
│   │   ├── boltzmann_policy.py
│   │   ├── fpl_environment.py
│   │   └── evaluation.py
│   └── mfg/                      # ← NEW: Mean-field
│       ├── mean_field_game.py
│       └── mean_field_control.py
├── pipeline/
│   └── data_pipeline.py
└── mlops/                        # ← NEW: MLOps Layer
    ├── mlflow_tracker.py
    ├── model_registry.py
    └── monitoring.py

tests/                            # Test Suite (~600+ tests target)
├── test_api.py
├── test_api_data_fetcher.py
├── test_fetcher.py
├── test_preprocessing.py
├── test_feature_engineering.py
├── test_feature_selection.py
├── test_understat.py
├── test_sentiment.py
├── test_eda.py
├── test_data_pipeline.py
├── test_hdp_gmm.py
├── test_hdp_hmm.py
├── test_hdp_other_models.py
├── test_hdp_gmm_gibbs.py         # ← NEW
├── test_hdp_hmm_ffbs.py          # ← NEW
├── test_belief_state.py          # ← NEW
├── test_thompson_sampling.py     # ← NEW
├── test_ts_mcts.py               # ← NEW
├── test_milp_solver.py           # ← NEW
├── test_knapsack.py              # ← NEW
├── test_captaincy.py             # ← NEW
├── test_chip_optimizer.py        # ← NEW
├── test_ddqn.py                  # ← NEW
├── test_fpl_environment.py       # ← NEW
├── test_integration_pipeline.py  # ← NEW
├── test_integration_ddqn.py      # ← NEW
└── test_integration_mcts.py      # ← NEW
```

---

## Execution Order (Immediate Next Steps)

1. **Fix 81 failing tests** (unblock CI/CD pipeline)
2. **Implement belief_state.py** (the bridge that every decision engine needs)
3. **Implement thompson_sampling.py** (Step A — simplest decision engine, depends only on belief states)
4. **Implement milp_solver.py** (constraint layer that TS needs)
5. **Implement feature_to_points.py** (maps HDP outputs → point predictions)
6. **Implement posterior_predictive.py** (Negative-Binomial parameterisation)

These six steps deliver a **working end-to-end pipeline**: data → features → HDP → belief states → point predictions → Thompson Sampling → MILP → optimal squad. Everything else (MCTS, DDQN, MFG, MLOps) extends this foundation.
