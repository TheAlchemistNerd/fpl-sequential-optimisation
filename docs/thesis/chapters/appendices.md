# Appendices

## Appendix A: Hyperparameter Settings

<!-- TODO: Complete table of all hyperparameters used in experiments -->

| Component | Parameter | Value | Justification |
|-----------|-----------|-------|---------------|
| HDP-GMM | $K$ (truncation) | 20 | $> 30\gamma$ for $\gamma \leq 5$ |
| HDP-GMM | $m_0$ (prior mean) | $\mathbf{0}$ | Features are standardised |
| HDP-GMM | $\kappa_0$ (prior precision) | 1.0 | Weakly informative |
| HDP-GMM | $\nu_0$ (prior df) | $D + 2$ | Minimum for finite $\mathbb{E}[\Sigma]$ |
| HDP-GMM | $\Psi_0$ (prior scale) | $I_D$ | Identity (standardised features) |
| HDP-HMM | $\alpha$ prior | $\text{Gamma}(1, 1)$ | Weakly informative |
| HDP-HMM | $\gamma$ prior | $\text{Gamma}(1, 1)$ | Weakly informative |
| MCMC/NUTS | num_warmup | 500 | <!-- TODO --> |
| MCMC/NUTS | num_samples | 1000 | <!-- TODO --> |
| MCMC/NUTS | num_chains | 4 | For Gelman-Rubin diagnostic |
| MCMC/NUTS | dense_mass | True | Captures parameter correlations |
| DDQN | Hidden layers | 512-256-128 | <!-- TODO --> |
| DDQN | Learning rate $\alpha$ | <!-- TODO --> | <!-- TODO --> |
| DDQN | Discount $\gamma$ | 0.99 | Near-future points equally valuable |
| DDQN | Target update $C$ | 1000 steps | <!-- TODO --> |
| DDQN | Replay buffer size | <!-- TODO --> | <!-- TODO --> |
| DDQN | Batch size | <!-- TODO --> | <!-- TODO --> |
| Boltzmann | $\tau_{\text{init}}$ | 2.0 | High initial exploration |
| Boltzmann | $\tau_{\text{final}}$ | 0.1 | Near-greedy exploitation |
| MILP | Solver | CBC | Open-source, sub-second |
| Feature Eng | Rolling windows | 3, 5, 10 | Short/medium/long-term form |
| Feature Eng | Fuzzy match threshold | 80 | Understat name matching |
| Sentiment | Model | DeBERTa-v3-small | Following Tamimi & Tran (2025) |
| Sentiment | Window | 7 days | Pre-deadline articles |

## Appendix B: Feature Set

<!-- TODO: Complete list of all D features used, grouped by category, with description and source -->

| # | Feature | Category | Source | Type |
|---|---------|----------|--------|------|
| 1 | goals_per_90 | Core | FPL API | Continuous |
| 2 | assists_per_90 | Core | FPL API | Continuous |
| 3 | clean_sheets_per_90 | Core | FPL API | Continuous |
| ... | ... | ... | ... | ... |
| D | injured | Contextual | FFS / API | Binary |

## Appendix C: Full Algorithm Listings

<!-- TODO: Include complete pseudocode for:
1. Full HDP-GMM Gibbs Sampler (expanded from Chapter 5, §5.4.3)
2. Full HDP-HMM Gibbs Sampler with FFBS (expanded from Chapter 6, §6.3.4)
3. Complete DDQN Training Loop (expanded from Chapter 10, §10.6.3)
4. Thompson Sampling Pipeline (expanded from Chapter 9, §9.4.2)
5. MILP Solver Configuration (expanded from Chapter 8, §8.6.2)
6. Data Ingestion Pipeline (expanded from Chapter 11, §11.3.1)
-->

## Appendix D: Additional Convergence Diagnostics

<!-- TODO: Include full trace plots, autocorrelation plots, and posterior density plots for all key parameters across all 4 MCMC chains. These are too voluminous for the main text. -->

## Appendix E: Per-Gameweek Results

<!-- TODO: Include a complete table of per-gameweek results for the backtested seasons:
| GW | Agent Points | Avg Manager | Top 10k Avg | Captain | Captain Points | Transfers Made | Transfer Cost |
-->

## Appendix F: Discovered Cluster Profiles

<!-- TODO: For each position group (GK, DEF, MID, FWD), include:
- Full mean vector μ_k for each active cluster
- Covariance matrix Σ_k (or at least diagonal elements and top correlations)
- Interpretive label and description
- Number of observations assigned to each cluster
- Example players predominantly in each cluster
-->
