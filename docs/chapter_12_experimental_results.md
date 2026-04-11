# Chapter 12: Experimental Results and Evaluation

## 12.1 Introduction

<!-- TODO: Brief overview of experimental goals — validate each component of the pipeline and the end-to-end system. Reference the ablation-heavy design philosophy. -->

## 12.2 Experimental Setup

### 12.2.1 Data

<!-- TODO: 
- Seasons used: e.g. 2021/22, 2022/23, 2023/24
- Train/validation/test split strategy (e.g. train on seasons 1-2, test on season 3; or rolling walk-forward)
- Number of players, gameweeks, total observations
- Feature set summary (reference Ch3)
-->

### 12.2.2 Baselines

<!-- TODO: Define all baseline strategies:
1. Naive Highest-xG: Select 15 players with highest xG per 90, solve MILP
2. Rolling Form: Select based on rolling 5-game average points
3. FPL Average Manager: Average score across all ~10M managers (from API)
4. Top 10k Benchmark: Average score of top 10,000 managers
5. Matthews & Ramchurn (2012) Bayesian Q-learning agent (if reproducible)
6. Random Valid Squad: Uniformly random valid squad each week
-->

### 12.2.3 Evaluation Metrics

<!-- TODO:
- Cumulative season points
- Overall rank (and percentile)
- Per-gameweek MAE/RMSE of player point predictions
- Transfer efficiency: points gained per transfer
- Captaincy hit rate: % of GWs where captain was highest-scoring starter
- Chip timing value: points gained from chip vs average chip timing
-->

### 12.2.4 Computational Environment

<!-- TODO: Hardware (CPU/GPU), software versions (Python, JAX, NumPyro, PuLP), MCMC configuration (num_warmup, num_samples, num_chains) -->

## 12.3 HDP-GMM / HDP-HMM Convergence Diagnostics

<!-- TODO:
- Trace plots for key parameters (γ, α, μ_k, Σ_k diagonal elements)
- Gelman-Rubin r̂ values across 4 chains (target: < 1.01 for all parameters)
- Effective Sample Size (ESS) per parameter (target: > 400)
- Geweke diagnostic z-scores
- Table: Parameter | r̂ | ESS | Geweke z
-->

## 12.4 Cluster Interpretability

<!-- TODO:
- Number of active clusters discovered (per position group: GK, DEF, MID, FWD)
- Table of cluster means: Cluster | Label | xG/90 | xA/90 | ICT/90 | Minutes | Form(5)
- Heatmap of transition matrix π_k for each position
- Example player trajectories: plot z_{j,t} over gameweeks for 3-4 well-known players (e.g. Salah, Haaland, Saka)
- Discussion of form persistence (diagonal dominance) and transition asymmetries
-->

## 12.5 Ablation Studies

### 12.5.1 Ablation 1: HDP-GMM vs HDP-HMM

<!-- TODO:
- Compare belief states from iid HDP-GMM vs temporal HDP-HMM
- Metric: prediction MAE, cluster stability, belief state entropy
- Hypothesis: HDP-HMM produces smoother, more predictive belief states due to form persistence
- Table: Model | MAE | RMSE | Avg Belief Entropy | Active Clusters
-->

### 12.5.2 Ablation 2: Thompson Sampling + MCTS vs DDQN

<!-- TODO:
- Compare season-long cumulative points
- Pure TS is myopic and excels at single-GW optimisation. Coupling TS with Monte Carlo Tree Search (MCTS) enables multi-week rollout planning (D2NG-POMCP style).
- DDQN expected to excel at learned generalized long-horizon value, while TS+MCTS excels at explicit scenario lookahead for chip timing and transfer saving.
- Table: Agent | Total Points | Rank | Transfer Efficiency | Chip Value
-->

### 12.5.3 Ablation 3: With vs Without Sentiment Features

<!-- TODO:
- Run full pipeline with and without DeBERTa sentiment features
- Metric: prediction MAE improvement, cumulative points difference
- Reference Tamimi & Tran (2025) MSE reduction from 6.27 to 5.63
- Table: Features | MAE | Points | Δ Points
-->

### 12.5.4 Ablation 4: With vs Without Understat Advanced Metrics

<!-- TODO:
- Run with FPL API data only vs FPL + Understat (xG, xA, npxG, xGChain, xGBuildup)
- Metric: cluster quality (silhouette score on belief states), prediction accuracy
- Table: Data Source | MAE | RMSE | Silhouette | Active Clusters
-->

### 12.5.5 Ablation 5: Number of Truncation Components K

<!-- TODO:
- Sweep K ∈ {5, 10, 15, 20, 30}
- Metric: log-likelihood, number of active clusters, prediction MAE
- Identify the point of diminishing returns
- Table: K | Log-Likelihood | Active Clusters | MAE | Runtime (s)
-->

### 12.5.6 Ablation 6: Concentration Parameter Sensitivity

<!-- TODO:
- Compare fixed γ ∈ {0.5, 1.0, 2.0, 5.0} vs inferred γ (Gamma hyperprior)
- Metric: number of active clusters, prediction stability
- Table: γ Setting | E[γ] | Active Clusters | MAE
-->

### 12.5.7 Ablation 7: MILP vs Greedy vs Genetic Algorithm

<!-- TODO:
- Compare squad selection quality across optimisation methods
- Metric: optimality gap, runtime, cumulative points
- MILP expected to dominate on quality; GA useful as warm start reference
- Table: Method | Points | Optimality Gap (%) | Runtime (ms)
-->

### 12.5.8 Ablation 8: Principled Boltzmann (Max-Entropy RL) vs $\epsilon$-Greedy

<!-- TODO:
- Evaluate the performance of the principled Boltzmann policy (derived from Maximum Entropy RL) against a standard $\epsilon$-greedy heuristic.
- Metric: cumulative points, exploration diversity (unique players selected), and risk-adjusted returns.
- Hypothesis: $\epsilon$-greedy is overly greedy and explores arbitrarily, leading to suboptimal transfers in FPL's stochastic environment. The principled Boltzmann policy optimally balances reward maximization with entropy (hedging), providing superior performance in navigating multi-modal posterior beliefs.
- Table: Strategy | Total Points | Unique Players Selected | Rank
-->

### 12.5.9 Ablation 9: Position-Specific vs Unified HDP-HMM

<!-- TODO:
- Compare separate HDP-HMM per position (GK/DEF/MID/FWD) vs single unified model
- Metric: cluster coherence, prediction accuracy per position
- Table: Model | GK MAE | DEF MAE | MID MAE | FWD MAE | Overall MAE
-->

### 12.5.10 Ablation 10: Mean-Field Coupling Strength ω

<!-- TODO:
- Sweep ω ∈ {0, 0.1, 0.3, 0.5, 0.8}
- ω=0 is standard HDP-HMM; higher ω pulls transitions toward population average
- Metric: prediction accuracy, belief state KL divergence from population
- Table: ω | MAE | KL_pop | Active Clusters
-->

## 12.6 End-to-End Backtesting Results

<!-- TODO:
- Full pipeline results across 2-3 seasons
- Comparison table: Agent | Season | Total Points | Rank | Percentile | vs Avg Manager | vs Top 10k
- Per-gameweek point trajectory plot (agent vs average manager vs top 10k average)
- Analysis of best/worst gameweeks — what went right/wrong
-->

## 12.7 Decision Analysis

### 12.7.1 Transfer Analysis

<!-- TODO:
- Net points gained from transfers per season
- Examples of high-value transfers (bought player who subsequently hauled) and poor transfers
- Comparison: agent transfer quality vs O'Brien et al. (2021) skill benchmarks
-->

### 12.7.2 Captaincy Analysis

<!-- TODO:
- Captaincy hit rate (% of GWs where captain was highest-scoring starter)
- Comparison with naive "always captain highest xG" baseline
- Points gained from captaincy vs random captain selection
-->

### 12.7.3 Chip Deployment Analysis

<!-- TODO:
- When did the agent deploy each chip?
- Points gained vs season-average chip deployment timing
- Comparison with O'Brien et al. (2021) finding that 79.4% of top 10k use Bench Boost in optimal DGW
-->

## 12.8 Drift Detection and Model Monitoring

<!-- TODO:
- KL divergence between consecutive gameweek belief states
- Identify any gameweeks where drift was detected and retraining triggered
- Plot: KL_pop(t) over the season with threshold line
-->

## 12.9 Computational Performance

<!-- TODO:
- Table: Component | Runtime | Hardware
  - Data ingestion: X seconds
  - HDP-HMM MCMC (4 chains, 500 warmup, 1000 samples): X minutes on GPU
  - Thompson Sampling (posterior sampling + MILP): X milliseconds
  - DDQN forward pass: X milliseconds
  - Full weekly pipeline: X minutes total
-->

## 12.10 Summary

<!-- TODO: Summarise key findings:
1. HDP-HMM outperforms HDP-GMM by X% on prediction MAE
2. Sentiment features improve points by X per season
3. Thompson Sampling achieves top X% rank; DDQN achieves top Y%
4. The agent achieves human-competitive performance (comparison with Matthews & Ramchurn 2012)
5. All ablations confirm the contribution of each component
-->
