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
| `src/data/fetcher.py` | ✅ Complete | Data aggregator, POSITION_MAP, STATUS_MAP, element summaries |
| `src/data/understat.py` | ✅ Complete | Async scraper, fuzzy name matching |
| `src/data/sentiment.py` | ✅ Complete | Lazy model loading, graceful degradation |
| `src/data/feature_engineering.py` | ✅ Complete | Per-90, form, FDR, contextual, assembly |
| `src/data/preprocessing.py` | ✅ Complete | Imputation, standardisation, position splitting |
| `src/data/feature_selection.py` | ✅ Complete | MI, correlation, Lasso, tree-based, consensus |
| `src/data/eda.py` | ✅ Complete | Univariate/bivariate/time-series/position plots |
| `src/pipeline/data_pipeline.py` | ✅ Complete | Wires fetch → engineer → model update |
| `src/models/hdp_gmm.py` | ✅ Complete | NUTS via NumPyro, Normal-Wishart, stick-breaking |
| `src/models/hdp_hmm.py` | ✅ Complete | Markov layer, transition matrices, FFBS structure |
| `src/models/hdp_poisson_gamma.py` | ✅ Complete | Poisson-Gamma for count data |
| `src/models/hdp_negbin.py` | ✅ Complete | Negative-Binomial for over-dispersed points |
| `tests/` (13 test files) | ✅ ~192/273 passing | TDD suite — 81 failures to fix (mock issues, edge cases, missing deps) |

### Not Started (Phase 2–6)

| Component | Chapters | Priority |
|---|---|---|
| HDP-GMM Collapsed Gibbs Sampler (non-NUTS) | Ch. 5 §5.4, Ch. 7 §7.2 | High |
| Belief State Construction | Ch. 5 §5.7, Ch. 7 §7.9 | High |
| ASUGS Adaptive Concentration | Ch. 7 §7.8 | Medium |
| Thompson Sampling (Step A) | Ch. 8 §8.1–8.7 | High |
| TS + MCTS (Step B) | Ch. 8 §8.8 | High |
| DDQN (Step C) | Ch. 9 §9.5–9.8 | High |
| Mean-Field Games / Control | Ch. 9 §9.10–9.12 | Low |
| MILP Constrained Optimiser | Ch. 10 §10.6 | High |
| Knapsack Solvers (DP, B&B, GA) | Ch. 10 §10.3–10.5 | Medium |
| Contextual MAB for Captaincy | Ch. 10 §10.7 | Medium |
| Chip Deployment Optimiser | Ch. 10 §10.10 | Medium |
| GenTS-Explore for Initial Squad | Ch. 10 §10.8 | Medium |
| MLOps: MLflow Experiment Tracking | Ch. 11 | Medium |
| MLOps: Model Registry & Monitoring | Ch. 11 | Medium |
| MLOps: DVC Data Versioning | Ch. 11 | Low |

---

## Phase 2: Posterior Inference & Belief States (Weeks 1–3)

### 2.1 Collapsed Gibbs Sampler for HDP-GMM

**Rationale**: The current NumPyro NUTS implementation is excellent for small-to-medium datasets but doesn't exploit conjugacy. A collapsed Gibbs sampler (Ch. 7 §7.2) provides exact asymptotic posterior samples and is 10–30x faster for purely conjugate models.

**Files to create:**
```
src/models/hdp_gmm_gibbs.py      — Collapsed Gibbs sampler
src/models/concentration_updates.py — Auxiliary variable method for γ, α
src/models/numba_accelerated.py  — JIT-compiled cluster assignment updates
```

**Implementation steps:**
1. Implement full conditional for cluster assignments: `P(z_{j,i} = k | ...) ∝ π_{j,k} · N(x | μ_k, Σ_k)`
2. Implement NIW posterior sampling: `(μ_k, Σ_k) | data ~ NIW(m_k, κ_k, Ψ_k, ν_k)`
3. Implement stick-breaking weight updates: `β'_k ~ Beta(1 + n_k, γ + Σ_{ℓ>k} n_ℓ)`
4. Implement auxiliary variable method for γ (Escobar & West, 1995)
5. Numba-accelerate the cluster assignment loop (10–30x speedup target)
6. Add convergence diagnostics: trace plots, Gelman-Rubin R̂, ESS, Geweke z-score
7. Write comprehensive tests (synthetic data, known cluster recovery)

**Tests:** `tests/test_hdp_gmm_gibbs.py` (~30 tests)

### 2.2 FFBS for HDP-HMM

**Rationale**: The current HDP-HMM uses NUTS, which doesn't exploit the Markov structure. FFBS (Ch. 6 §6.3) is O(T·K²) per player — optimal for trajectory sampling.

**Files to create:**
```
src/models/hdp_hmm_ffbs.py       — Forward-Filtering Backward-Sampling
src/models/transition_updates.py — Dirichlet posterior for transition matrices
```

**Implementation steps:**
1. Implement forward filter: `α_t(ℓ) ∝ Σ_k α_{t-1}(k) · π_{k,ℓ} · N(x_t | μ_ℓ, Σ_ℓ)`
2. Implement backward sample: `z_t ~ Categorical(α_t(k) · π_{k, z_{t+1}})`
3. Implement transition matrix updates: `π_k | data ~ Dir(αβ + n_{k→·})`
4. Wire to existing HDP-HMM model (replace NUTS trajectory sampling)
5. Test on synthetic Markov sequences with known transition matrices

**Tests:** `tests/test_hdp_hmm_ffbs.py` (~25 tests)

### 2.3 Belief State Construction

**Rationale**: Belief states are the mathematical bridge between the HDP models and the decision engines. They are the primary output of the inference pipeline.

**Files to create:**
```
src/models/belief_state.py       — Belief state computation, mapping to points
src/models/feature_to_points.py  — Linear/GBM mapping from features → FPL points
```

**Implementation steps:**
1. Compute belief vectors: `b_j^{(k)}(t) = (1/S) Σ_s I(z_{j,t}^{(s)} = k)`
2. Implement feature-to-points mapping: `E[points] = w^T Σ_k b_j^{(k)}(t) · μ_k`
3. Train mapping model on historical data (linear regression + Gradient Boosting)
4. Negative-Binomial parameterisation: `y ~ NB(r, p)` with `E[y] = μ̃`
5. Posterior predictive sampling: `{ỹ_j^{(s)}}_{s=1}^S`
6. Belief state persistence API: `get_belief_state(player_id, gameweek) → dict`

**Tests:** `tests/test_belief_state.py` (~20 tests)

### 2.4 ASUGS Adaptive Concentration

**Files to create:**
```
src/models/asugs.py              — Adaptive SUGS for online γ adaptation
```

**Implementation steps:**
1. Implement online γ update: `γ_n = γ_{n-1} + η_n · ∂/∂γ log p(z_n | z_{1:n-1}, γ)`
2. Implement drift detection: trigger ASUGS when `KL(b_t || b_{t-1}) > threshold`
3. Wire to belief state pipeline

---

## Phase 3: Thompson Sampling & MCTS (Weeks 4–6)

### 3.1 Thompson Sampling (Step A)

**Rationale**: The simplest decision engine. Samples from posterior, solves knapsack. Foundation for all subsequent methods.

**Files to create:**
```
src/optimization/thompson_sampling.py  — Core TS algorithm
src/optimization/posterior_predictive.py — Points posterior from belief states
```

**Implementation steps:**
1. Implement posterior construction: for each MCMC sample s, compute `μ̃_j^{(s)}`
2. Sample one point prediction per player: `ỹ_j ~ NB(r_j, p_j(μ̃_j))`
3. Implement the "simulated world" reduction: deterministic knapsack given `ỹ`
4. Wire to MILP solver (Phase 4) for constraint handling
5. Implement temperature scaling: `τ_t` annealing over season
6. Implement GenTS-Explore for combinatorial settings (initial squad / Wildcard)
7. Bayes regret computation and logging

**Tests:** `tests/test_thompson_sampling.py` (~25 tests)

### 3.2 TS + MCTS (Step B)

**Rationale**: Multi-gameweek lookahead for chip timing, transfer saving, double-GW preparation.

**Files to create:**
```
src/optimization/ts_mcts.py      — TS coupled with Monte Carlo Tree Search
src/optimization/mcts_node.py    — MCTS tree node with UCB1
src/optimization/rollout.py      — Forward simulation under sampled world
```

**Implementation steps:**
1. Implement MCTS node class: `(state, action, visit_count, value, children)`
2. Implement UCB1 selection: `UCB1(s,a) = Q(s,a) + c·√(log N(s) / N(s,a))`
3. Implement rollout: simulate T gameweeks forward using HDP-HMM transitions
4. Implement backpropagation: cumulative discounted reward up the tree
5. Implement world sampling: draw one MCMC sample to fix the "simulated world"
6. Action space pruning: restrict to top-100 candidates by expected points
7. Temperature-annealed Boltzmann action selection at leaf nodes
8. Wire to chip timing logic (Wildcard, Bench Boost, Triple Captain)

**Tests:** `tests/test_ts_mcts.py` (~30 tests)

---

## Phase 4: Constrained Optimisation (Weeks 7–8)

### 4.1 MILP Squad Optimiser

**Rationale**: Every decision engine outputs candidate selections. The MILP layer enforces FPL's rigid constraints (budget, formation, per-team limits, captaincy).

**Files to create:**
```
src/optimization/milp_solver.py      — PuLP/CBC MILP formulation
src/optimization/knapsack.py         — DP, Branch-and-Bound, Greedy, GA
src/optimization/captaincy.py        — Contextual MAB (LinUCB, TS)
src/optimization/chip_optimizer.py   — Multi-armed bandit over remaining GWs
```

**Implementation steps:**
1. Implement base 0/1 knapsack with Dynamic Programming (Ch. 10 §10.3)
2. Implement Branch-and-Bound with fractional knapsack upper bound (Ch. 10 §10.5)
3. Implement Greedy baseline (value-per-weight, fast warm-start)
4. Implement full MILP formulation with all FPL constraints (Ch. 10 §10.6):
   - Decision variables: `x_i` (squad), `s_i` (starting XI), `c_i` (captain), `v_i` (vice-captain)
   - Constraints: squad size (15), positional (2-5-5-3), budget, per-team (≤3), formation, captaincy
   - Transfer cost encoding: piecewise linear via auxiliary variable
5. Implement Genetic Algorithm for large instances / initial squad (Ch. 10 §10.4.3)
6. Contextual Thompson Sampling for captaincy (Ch. 10 §10.7.2)
7. LinUCB alternative for captaincy (Ch. 10 §10.7.3)
8. Chip deployment optimiser: 4-armed bandit over remaining gameweeks
9. Double-gameweek strategy: expected reward ≈ 2× for BB/TC

**Tests:** `tests/test_milp_solver.py` (~25 tests), `tests/test_knapsack.py` (~20 tests),
`tests/test_captaincy.py` (~15 tests), `tests/test_chip_optimizer.py` (~15 tests)

---

## Phase 5: DDQN (Step C) (Weeks 9–12)

### 5.1 State Representation

**Files to create:**
```
src/optimization/ddqn/
├── __init__.py
├── state_representation.py    — Encode squad + budget + beliefs + chips → tensor
├── q_network.py               — Feedforward NN (757 → 512 → 256 → 128 → |A|)
├── replay_buffer.py           — Experience replay with prioritised sampling
└── action_space.py            — Candidate action generation + pruning
```

**Implementation steps:**
1. Encode state vector: squad (binary N-vector) + budget + belief vectors + chips + metadata
2. Build Q-network: Dense(512) → ReLU → LayerNorm → Dropout(0.1) → Dense(256) → ... → Dense(|A|)
3. Implement candidate action pruning: top-100 players by expected points → ~8,775 actions → ~100
4. Implement experience replay buffer with uniform sampling
5. Implement target network with periodic hard/soft updates

### 5.2 Training Loop

**Files to create:**
```
src/optimization/ddqn/
├── training_loop.py           — Full DDQN training with HDP-HMM belief updates
├── boltzmann_policy.py        — Temperature-annealed softmax action selection
├── fpl_environment.py         — Gym-like env: step() → next_state, reward, done
└── evaluation.py              — Season simulation, metrics collection
```

**Implementation steps:**
1. Implement FPL environment: `env.step(action) → (next_squad, reward, done)`
2. Implement Boltzmann policy with temperature annealing (τ: 2.0 → 0.1 over 38 GW)
3. Implement DDQN update: decouple action selection (online) from evaluation (target)
4. Implement unified HDP-HMM + DDQN training loop (Ch. 9 §9.6.3)
5. Implement season simulation for training data generation
6. Implement evaluation metrics: cumulative points, transfer efficiency, rank trajectory

**Tests:** `tests/test_ddqn.py` (~30 tests), `tests/test_fpl_environment.py` (~20 tests)

### 5.3 Mean-Field Games / Control

**Files to create:**
```
src/optimization/mfg/
├── __init__.py
├── mean_field_game.py         — MFG formulation for population-level strategy
└── mean_field_control.py      — MFC for single-agent vs population
```

**Implementation steps:**
1. Implement population distribution dynamics: `m_t^{(k)} = (1/J) Σ_j P(z_{j,t} = k)`
2. Implement mean-field coupling: `π_k(m_t) = (1-ω)·π_k^ind + ω·m_t`
3. Implement MFG equilibrium solver (fixed-point iteration)
4. Implement MFC for single agent optimising against population average

---

## Phase 6: MLOps & Production (Weeks 13–14)

### 6.1 MLflow Experiment Tracking

**Files to create:**
```
src/mlops/
├── __init__.py
├── mlflow_tracker.py          — Log params, metrics, artifacts, models
├── model_registry.py           — Model versioning, staging, promotion
└── monitoring.py               — Drift detection, performance monitoring
```

**Implementation steps:**
1. Implement MLflow logging: params (hyperparameters), metrics (cumulative points, rank), artifacts (trace plots, transition heatmaps)
2. Implement model registry: tag models as "staging" → "production" → "archived"
3. Implement drift detection: monitor `KL(pop_t || pop_{t-1})` and trigger retraining
4. Wire to `data_pipeline.py` → log every pipeline run

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
│   ├── hdp_gmm_gibbs.py          # ← NEW: Collapsed Gibbs
│   ├── hdp_hmm.py                # NUTS implementation
│   ├── hdp_hmm_ffbs.py           # ← NEW: FFBS trajectory sampling
│   ├── hdp_poisson_gamma.py
│   ├── hdp_negbin.py
│   ├── belief_state.py           # ← NEW: Belief state construction
│   ├── feature_to_points.py      # ← NEW: Feature → points mapping
│   ├── concentration_updates.py  # ← NEW: Auxiliary variable method
│   ├── numba_accelerated.py      # ← NEW: JIT-compiled Gibbs
│   └── asugs.py                  # ← NEW: Adaptive concentration
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
