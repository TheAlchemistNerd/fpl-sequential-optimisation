# Chapter 7: Posterior Inference — MCMC, Gibbs Sampling, NUTS, and Acceleration Techniques

## 7.1 Introduction

The models developed in Chapters 5 and 6 — the HDP-GMM and HDP-HMM — are Bayesian hierarchical models with complex dependency structures and an unbounded number of latent variables. The posterior distributions over the model parameters and latent states are analytically intractable, necessitating approximate inference methods.

This chapter provides a comprehensive treatment of the posterior inference techniques employed in the FPL sequential optimisation system. The topics covered include:

1. **Gibbs sampling** for the HDP-GMM and HDP-HMM, leveraging conjugacy for closed-form updates.
2. **NUTS (No-U-Turn Sampler)** via NumPyro, a gradient-based MCMC method that handles non-conjugate models.
3. **Numba acceleration** for loop-intensive Gibbs sampling steps.
4. **Variational Inference (VI)** as a faster alternative to MCMC.
5. **Convergence diagnostics** to ensure the MCMC chains have reached the true posterior.

### 7.1.1 From Posterior Inference to Belief-State Decision-Making

It is critical to understand that the posterior samples produced by MCMC are **not endpoints** of the analysis — they are the **belief-state representation** that the decision engines in Chapters 8 and 9 operate on directly. The posterior samples over $(\mu_k, \Sigma_k, \pi_k, z)$ encode the full distributional uncertainty about each player's latent form profile. Thompson Sampling, Monte Carlo Tree Search, and DDQN all consume these belief states as their mathematical input.

This connection formalizes the FPL problem as a **Partially Observable Markov Decision Process (POMDP)**, defined rigorously in Chapter 9 (Section 9.2). In a POMDP, the agent does not observe the true state $s_t$ (the player's actual form); instead, it maintains a **belief state** $b_t = P(s_t \mid \text{observations})$, which is precisely the posterior distribution computed by the MCMC methods in this chapter. The RL agents in subsequent chapters do not operate on point estimates; they operate on **distributions over belief states**, enabling them to reason about and hedge against uncertainty in a mathematically principled way.

The pipeline is:

$$\text{MCMC (Ch. 7)} \;\rightarrow\; \{b_j^{(s)}(t)\}_{s=1}^S \;\rightarrow\; \text{Thompson Sampling (Ch. 8)} \;\rightarrow\; \text{TS+MCTS / DDQN (Ch. 9)}.$$

This chapter establishes the inference machinery; the next chapters show how these belief states drive optimal sequential decisions.

## 7.2 Gibbs Sampling for the HDP-GMM

### 7.2.1 Overview

Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method that generates samples from a joint posterior by iteratively sampling each variable from its full conditional distribution, given the current values of all other variables. For the HDP-GMM, the variables are:

- Global concentration parameter: $\gamma$
- Stick-breaking weights: $\boldsymbol{\beta}' = (\beta'_1, \ldots, \beta'_{K-1})$
- Player-specific concentration: $\alpha$
- Player-specific mixture weights: $\boldsymbol{\pi}_j$ for each player $j$
- Cluster assignments: $z_{j,i}$ for each observation
- Cluster parameters: $(\mu_k, \Sigma_k)$ for each cluster $k$

### 7.2.2 Full Conditional for Cluster Assignments

The full conditional for $z_{j,i}$ is:

$$P(z_{j,i} = k \mid \mathbf{z}_{-ji}, \mathbf{x}, \boldsymbol{\pi}_j, \{\mu_k, \Sigma_k\}) \propto \pi_{j,k} \cdot \mathcal{N}(x_{j,i} \mid \mu_k, \Sigma_k).$$

Sampling is straightforward:

$$z_{j,i} \sim \text{Categorical}\left( \frac{\pi_{j,k} \cdot \mathcal{N}(x_{j,i} \mid \mu_k, \Sigma_k)}{\sum_{k'} \pi_{j,k'} \cdot \mathcal{N}(x_{j,i} \mid \mu_{k'}, \Sigma_{k'})} \right).$$

### 7.2.3 Full Conditional for Cluster Parameters

Given the NIW prior and Gaussian likelihood, the full conditional for $(\mu_k, \Sigma_k)$ is:

$$(\mu_k, \Sigma_k) \mid \mathbf{z}, \mathbf{x} \sim \text{NIW}(m_k, \kappa_k, \Psi_k, \nu_k),$$

with parameters as derived in Chapter 5 (Section 5.3.2). Sampling from the NIW proceeds in two steps:

1. Sample $\Sigma_k \sim \text{Inverse-Wishart}_{\nu_k}(\Psi_k)$.
2. Sample $\mu_k \mid \Sigma_k \sim \mathcal{N}(m_k, \Sigma_k / \kappa_k)$.

### 7.2.4 Full Conditional for Stick-Breaking Weights

The full conditional for $\beta'_k$ is:

$$\beta'_k \mid \mathbf{z}, \gamma \sim \text{Beta}\left(1 + n_k, \;\; \gamma + \sum_{\ell=k+1}^K n_\ell \right),$$

where $n_k$ is the number of observations assigned to cluster $k$. This follows from the conjugacy of the Beta-Binomial model.

### 7.2.5 Full Conditional for Concentration Parameters

For the global concentration $\gamma$, the auxiliary variable method (Escobar & West, 1995) is employed. An auxiliary variable $\eta \sim \text{Beta}(\gamma + 1, N)$ is introduced:

$$P(\eta \mid \gamma, N) = \text{Beta}(\gamma + 1, N).$$

Then the full conditional for $\gamma$ is a mixture of two Gamma distributions:

$$\gamma \mid \eta, \mathbf{z} \sim \pi_\gamma \cdot \text{Gamma}(a_\gamma + K, b_\gamma - \log \eta) + (1 - \pi_\gamma) \cdot \text{Gamma}(a_\gamma + K - 1, b_\gamma - \log \eta),$$

where the mixture weight $\pi_\gamma$ satisfies:

$$\frac{\pi_\gamma}{1 - \pi_\gamma} = \frac{a_\gamma + K - 1}{N(b_\gamma - \log \eta)}.$$

Similarly for the player-specific concentration $\alpha$:

$$\alpha \mid \eta_j, \mathbf{z} \sim \pi_\alpha \cdot \text{Gamma}(a_\alpha + K_j, b_\alpha - \log \eta_j) + (1 - \pi_\alpha) \cdot \text{Gamma}(a_\alpha + K_j - 1, b_\alpha - \log \eta_j).$$

## 7.3 Gibbs Sampling for the HDP-HMM

### 7.3.1 State Trajectory Sampling via FFBS

The HDP-HMM requires sampling the entire state trajectory $\mathbf{z}_{j,1:T_j}$ for each player, rather than individual assignments. This is done using the Forward-Filtering Backward-Sampling (FFBS) algorithm, described in detail in Chapter 6 (Section 6.3).

The computational complexity of FFBS is $O(T_j \cdot K^2)$ per player per iteration, which is more expensive than the $O(T_j \cdot K)$ complexity of the HDP-GMM's per-observation updates.

### 7.3.2 Transition Matrix Updates

The full conditional for the transition matrix $\boldsymbol{\pi}_k$ is:

$$\boldsymbol{\pi}_k \mid \mathbf{z}, \alpha, \boldsymbol{\beta} \sim \text{Dir}(\alpha \beta_1 + n_{k \to 1}, \ldots, \alpha \beta_K + n_{k \to K}),$$

where $n_{k \to \ell} = \sum_{j,t} \mathbb{I}(z_{j,t-1} = k, z_{j,t} = \ell)$ is the count of transitions from state $k$ to state $\ell$.

### 7.3.3 Full HDP-HMM Gibbs Sampler

```
for iteration = 1 to N:
    # Update global parameters
    Sample γ (auxiliary variable method)
    Sample β'_k ~ Beta(1 + n_k, γ + Σ_{ℓ>k} n_ℓ)

    # Update emission parameters
    For each cluster k:
        Compute sufficient statistics
        Sample (μ_k, Σ_k) ~ NIW(m_k, κ_k, Ψ_k, ν_k)

    # Update transition matrices
    For each state k:
        Count transitions n_{k→ℓ}
        Sample π_k ~ Dir(α·β_1 + n_{k→1}, ..., α·β_K + n_{k→K})

    # Update state trajectories (FFBS)
    For each player j:
        Forward filter: compute α_t(k)
        Backward sample: sample z_{j,1:T_j}
```

## 7.4 NUTS via NumPyro

### 7.4.1 Hamiltonian Monte Carlo

Hamiltonian Monte Carlo (HMC) is an MCMC method that uses gradient information to propose distant moves with high acceptance probability. It introduces an auxiliary momentum variable $\mathbf{r}$ and simulates the Hamiltonian dynamics:

$$H(\theta, \mathbf{r}) = -\log p(\theta \mid \text{data}) + \frac{1}{2} \mathbf{r}^\top M^{-1} \mathbf{r},$$

where $p(\theta \mid \text{data})$ is the target posterior, and $M$ is a mass matrix.

The dynamics are:

$$\frac{d\theta}{dt} = \frac{\partial H}{\partial \mathbf{r}} = M^{-1} \mathbf{r}, \quad \frac{d\mathbf{r}}{dt} = -\frac{\partial H}{\partial \theta} = \nabla_\theta \log p(\theta \mid \text{data}).$$

These are simulated using the leapfrog integrator, which preserves volume and is time-reversible.

### 7.4.2 No-U-Turn Sampler (NUTS)

The No-U-Turn Sampler (Hoffman & Gelman, 2014) is an extension of HMC that automatically tunes the trajectory length, eliminating the need to specify the number of leapfrog steps. It builds a balanced binary tree of leapfrog steps and stops when the trajectory begins to double back on itself (the "U-turn" condition):

$$\langle \theta_{\text{end}} - \theta_{\text{start}}, \mathbf{r}_{\text{start}} \rangle < 0.$$

NUTS is particularly effective for high-dimensional, correlated posteriors, as it adapts to the local geometry.

### 6.4.3 NUTS Configuration

The NUTS sampler is configured with the following settings:

```
Algorithm: NUTS Configuration for HDP-GMM
Input: Model specification (Chapter 5, Section 5.8), data X
Output: Posterior samples

Configuration:
    num_warmup ← 500                         // warmup (burn-in) iterations
    num_samples ← 1000                        // post-warmup samples to collect
    num_chains ← 4                            // independent chains for convergence diagnostics
    dense_mass ← True                         // estimate full mass matrix for kinetic energy

Execution:
    Run NUTS with the above configuration
    Collect samples: {γ^(s), α^(s), β^(s), μ^(s), Σ^(s), z^(s)} for s = 1 to num_samples

Key properties:
    - JIT compilation compiles the entire log-density and gradient into a single operation
    - Automatic differentiation computes ∇_θ log p(θ | data) without manual derivation
    - Dense mass matrix captures correlations between parameters, improving mixing
    - 4 independent chains enable Gelman-Rubin r_hat convergence diagnostic
```

### 7.4.4 Advantages over Gibbs Sampling

| Aspect | Gibbs Sampling | NUTS |
|--------|---------------|------|
| Conjugacy required | Yes | No |
| Correlated parameters | Poor mixing | Excellent (dense mass) |
| Parallelisation | Limited (sequential) | Full (JAX vmap) |
| GPU/TPU support | Native (JAX) | Native (JAX) |
| Hyperparameter tuning | Manual (concentration updates) | Automatic (step size, mass matrix) |
| Discrete variables | Natural (categorical sampling) | Requires marginalisation |

NUTS is preferred when the model is non-conjugate (e.g., when incorporating neural network components) or when parameters are highly correlated. Gibbs sampling is preferred for the purely conjugate HDP-GMM, as the closed-form updates are exact and fast.

## 7.5 JAX/XLA JIT Acceleration for Recursive Inference

### 7.5.1 Motivation

Bayesian nonparametric models like the HDP-GMM and HDP-HMM involve high-dimensional, recursive updates that are prohibitively slow in interpreted Python. For a player registry of $N \approx 10{,}000$ and $K \approx 20$ clusters, a single MCMC iteration involves over $200{,}000$ cluster membership evaluations. Traditional NumPy operations, while vectorised, still incur significant Python overhead for sequential algorithms like **Forward-Filtering Backward-Sampling (FFBS)**.

The system leverages **JAX** and its **XLA (Accelerated Linear Algebra)** compiler to bypass these bottlenecks. By JIT-compiling the critical inference kernels, the system achieves the sub-second iteration times required for real-time belief state updates.

### 7.5.2 JIT Compilation and `lax.scan`

Unlike Numba, which operates as a separate JIT layer for Python, JAX is natively designed for high-performance array computing and automatic differentiation. The recursive state transitions in the HDP-HMM are implemented using `jax.lax.scan`, which compiles the temporal loop into a single, highly optimised XLA primitive.

```
Algorithm: JAX-Native FFBS Backward Sample
Input: Filtered probabilities α[T, K], transition matrix π[K, K]
Output: Sampled state trajectory z[T]

1. Precompute: Reverse filtered probabilities α̃ ← reverse(α)
2. Initialise: z[T] ← sample from Categorical(α[T])

3. Recursion: For each timestep t from T-1 down to 1:
    a. Compute conditional probabilities: 
       P(z_t = k | z_{t+1}, α_t) ∝ α_t(k) · π(k, z_{t+1})
    b. Normalise: P(z_t) ← P(z_t) / Σ_k P(z_t = k)
    c. Sample: z[t] ← sample from Categorical(P(z_t))

4. Return z
```

By compiling this entire recursive procedure, JAX eliminates the Python loop overhead and allows the compiler to perform cross-operation optimisations, such as fusion of the categorical sampling and normalization kernels.

### 7.5.3 Hardware Portability and Speedup

The use of JAX provides two critical advantages over alternative JIT technologies:

1.  **Unified Memory Space**: Inferred parameters (means, covariances, belief states) remain in the JAX device memory, eliminating costly transfers between Python, Numba, and the GPU.
2.  **Accelerator Support**: The same code runs on CPUs, GPUs, and TPUs with zero modifications. The XLA compiler automatically targets the available hardware, enabling the large-scale population simulations in Chapter 9 to scale to hundreds of thousands of agents on professional-grade hardware.

The speedup achieved via JAX/XLA JIT is comparable to C++ implementations:

- **Forward Filtering**: 20–50x speedup via XLA loop fusion.
- **Cluster Probabilities**: 10–20x speedup via `jax.vmap` vectorisation across $N$ observations.
- **Full MCMC iteration**: 10–30x speedup overall compared to interpreted NumPy/Python.

### 7.5.4 Parallel Chain Execution

Multiple MCMC chains are run in parallel using JAX's `pmap` or `vmap` primitives, allowing the system to harness multi-core CPUs or massively parallel GPU architectures to assess convergence:

```
Algorithm: Massively Parallel MCMC Chains
Input: Model specification, data X, number of chains C = 8
Output: Samples from C independent chains

1. Initialise: Split random seed into C independent seeds {seed_1, ..., seed_C}
2. Parallel Execute:
    For each chain i from 1 to C (in parallel):
        Run inference chain θ_i ← run_mcmc(seed_i, X)

3. Synchronise: Collect results from all streams/GPU-cores {θ_1, ..., θ_C}
4. Return concatenated samples
```

This parallel execution is fundamental to the convergence diagnostics in Section 7.7, as it allows for the rapid generation of the multiple independent trajectories required for the Gelman-Rubin $\hat{R}$ calculation.

## 7.6 Variational Inference

### 7.6.1 Overview

Variational Inference (VI) frames posterior approximation as an optimisation problem. Rather than sampling from the posterior, VI finds the closest distribution $q(\theta)$ in a tractable family $\mathcal{Q}$ to the true posterior $p(\theta \mid \text{data})$, as measured by the Kullback-Leibler (KL) divergence:

$$q^*(\theta) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(\theta) \parallel p(\theta \mid \text{data})).$$

### 7.6.2 Evidence Lower Bound (ELBO)

Minimising KL divergence is equivalent to maximising the Evidence Lower Bound (ELBO):

$$\text{ELBO}(q) = \mathbb{E}_q[\log p(\text{data}, \theta)] - \mathbb{E}_q[\log q(\theta)].$$

The ELBO is a lower bound on the log marginal likelihood:

$$\log p(\text{data}) \geq \text{ELBO}(q).$$

### 7.6.3 Mean-Field Approximation

The mean-field VI family assumes independence between groups of variables:

$$q(\theta) = \prod_{g} q_g(\theta_g).$$

For the HDP-GMM, the factorisation is:

$$q(\gamma, \boldsymbol{\beta}', \alpha, \{\boldsymbol{\pi}_j\}, \{z_{j,i}\}, \{(\mu_k, \Sigma_k)\}) = q(\gamma) \prod_k q(\beta'_k) \cdot q(\alpha) \prod_j q(\boldsymbol{\pi}_j) \prod_{j,i} q(z_{j,i}) \prod_k q(\mu_k, \Sigma_k).$$

Each factor is optimised by coordinate ascent, setting:

$$\log q_g^*(\theta_g) = \mathbb{E}_{q_{-g}}[\log p(\text{data}, \theta)] + \text{const}.$$

### 7.6.4 Jensen-Shannon Divergence Alternative

The standard KL divergence $\text{KL}(q \parallel p)$ is asymmetric and tends to produce under-dispersed approximations (zero-forcing behaviour). The **Jensen-Shannon Divergence (JSD)** provides a symmetric alternative:

$$\text{JSD}(q \parallel p) = \frac{1}{2} \text{KL}\left(q \parallel \frac{q + p}{2}\right) + \frac{1}{2} \text{KL}\left(p \parallel \frac{q + p}{2}\right).$$

Maximising the JSD-based objective yields approximations that are less prone to mode collapse, potentially capturing multiple posterior modes. This is particularly relevant for the HDP-GMM, where the posterior over cluster assignments may be multi-modal.

### 7.6.5 Stochastic Variational Inference

For large datasets, Stochastic VI (SVI) subsamples the data at each iteration:

```
Algorithm: Stochastic Variational Inference for HDP-GMM
Input: Data X (N × D), guide distribution family Q_φ, num_steps
Output: Optimal variational parameters φ*

Initialise: φ ← random initialisation
For step = 1 to num_steps:
    // Sample minibatch
    B ← random sample of size |B| from {1, ..., N}

    // Estimate ELBO gradient
    elbo_estimate ← (N / |B|) × Σ_{i ∈ B} E_q[log p(x_i | θ)] - KL(q(θ) || p(θ))

    // Compute gradient and update
    g ← ∇_φ elbo_estimate
    φ ← φ + η × g                        // η is learning rate

    // Decay learning rate
    η ← η × decay_rate

Return φ*
```

This reduces the per-iteration cost from O(N) to O(|B|), enabling scalability to millions of observations. The guide distribution q_φ(θ) is typically chosen as a factorised Gaussian (mean-field approximation), with parameters φ optimised via gradient ascent on the ELBO.

### 7.6.6 VI vs. MCMC Trade-offs

| Aspect | MCMC (Gibbs/NUTS) | Variational Inference |
|--------|-------------------|----------------------|
| Accuracy | Asymptotically exact | Approximate (lower bound) |
| Speed | Slow (many iterations) | Fast (optimisation) |
| Scalability | Limited by chain length | Excellent (minibatching) |
| Uncertainty quantification | Full posterior samples | Factorised approximation |
| Implementation | Simple (conjugate models) | Requires guide specification |

For the FPL problem, MCMC is employed for offline model fitting (where accuracy is paramount) and VI for rapid online updates (where speed is critical).

## 7.7 Convergence Diagnostics

Empirical convergence results for all diagnostics discussed in this section are reported in Chapter 12, Section 12.3, including trace plots, Gelman-Rubin $\hat{R}$ values, Effective Sample Size (ESS), and Geweke z-scores for the FPL dataset.

### 7.7.1 Trace Plots

The simplest diagnostic is a visual inspection of the trace plots — the sampled values of each parameter plotted against iteration number. A well-mixing chain shows rapid fluctuations around a stable mean, with no trends or long excursions. Chapter 12, Figure 12.1 presents trace plots of $\gamma$, $\alpha$, and representative $\mu_k$ diagonal elements across the 4 MCMC chains.

### 7.7.2 Gelman-Rubin $\hat{R}$ Statistic

The $\hat{R}$ statistic compares the within-chain and between-chain variances across multiple chains:

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}},$$

where $W$ is the within-chain variance and $\hat{V}$ is a pooled estimate combining within- and between-chain variance. Convergence is indicated when $\hat{R} < 1.01$ for all parameters. Chapter 12, Table 12.1 reports $\hat{R}$ values for all monitored parameters; all satisfy the $\hat{R} < 1.01$ threshold.

### 7.7.3 Effective Sample Size (ESS)

The ESS quantifies the number of independent samples equivalent to the correlated MCMC samples:

$$\text{ESS} = \frac{N}{1 + 2 \sum_{t=1}^\infty \rho_t},$$

where $\rho_t$ is the autocorrelation at lag $t$. Low ESS indicates high autocorrelation and poor mixing. The threshold $\text{ESS} > 400$ is required for reliable inference. All parameters meet this threshold (Chapter 12, Table 12.1).

### 7.7.4 Geweke Diagnostic

The Geweke diagnostic compares the means of the early and late portions of the chain:

$$z = \frac{\bar{\theta}_A - \bar{\theta}_B}{\sqrt{\text{Var}(\bar{\theta}_A) + \text{Var}(\bar{\theta}_B)}},$$

where $A$ is the first 10% of the chain and $B$ is the last 50%. Under convergence, $z \sim \mathcal{N}(0, 1)$. Values $|z| > 2$ indicate non-convergence. All Geweke z-scores fall within $[-2, 2]$ (Chapter 12, Table 12.1).

### 7.7.5 Raftery-Lewis Diagnostic

The Raftery-Lewis diagnostic computes the number of iterations required to estimate a specified quantile to a desired accuracy. For example, to estimate the 2.5th percentile to within ±0.005 with 95% confidence, the diagnostic returns the required chain length.

## 7.8 Adaptive Concentration Parameters: ASUGS

The Adaptive SUGS (ASUGS) algorithm (Tsiligkaridis & Forsythe, 2015) provides a low-complexity method for online adaptation of the concentration parameter $\gamma$ as new data arrives. This is particularly relevant for the FPL problem, where streaming data requires efficient online learning algorithms and the number of latent form states must adapt as player performance patterns evolve over the season.

The ASUGS update for $\gamma$ at step $n$ is:

$$\gamma_{n} = \gamma_{n-1} + \eta_n \cdot \left( \frac{\partial}{\partial \gamma} \log p(z_n \mid z_{1:n-1}, \gamma) \right),$$

where $\eta_n$ is a step size schedule (e.g., $\eta_n = 1/n$). The gradient is:

$$\frac{\partial}{\partial \gamma} \log p(z_n \mid z_{1:n-1}, \gamma) = \frac{1}{\gamma + n - 1} - \psi(\gamma + n - 1) + \psi(\gamma),$$

where $\psi$ is the digamma function.

This online update allows the model to adapt the number of clusters as new data arrives, without requiring a full re-run of the MCMC chain. In practice, the KL divergence between consecutive gameweek belief states, $\text{KL}(b_{t} \parallel b_{t-1})$, is monitored, and ASUGS adaptation is triggered when this exceeds a threshold. This drift detection mechanism is evaluated empirically in Chapter 12, Section 12.8, which plots $\text{KL}_{\text{pop}}(t)$ over the season and identifies gameweeks where retraining was triggered.

## 7.9 From Posterior to Belief State

### 7.9.1 The Belief-State Mapping

The posterior samples produced by the MCMC methods in this chapter are not merely statistical summaries — they constitute the **belief-state representation** that the decision engines in Chapters 8 and 9 operate on directly. Formally, for each player $j$ at gameweek $t$, the belief state over latent form clusters is:

$$b_j(t) = \left( P(z_{j,t} = 1 \mid x_{j,1:t}), \; P(z_{j,t} = 2 \mid x_{j,1:t}), \; \ldots, \; P(z_{j,t} = K \mid x_{j,1:t}) \right).$$

For each MCMC sample $s \in \{1, \ldots, S\}$, the belief state is computed using the sampled cluster parameters $(\mu_k^{(s)}, \Sigma_k^{(s)})$ and transition matrices $\pi_k^{(s)}$. The collection $\{b_j^{(s)}(t)\}_{s=1}^S$ forms the **posterior distribution over belief states**, which fully captures the epistemic uncertainty about player $j$'s form at gameweek $t$.

### 7.9.2 Belief-State MDP

This belief-state formulation converts the FPL problem into a **belief-state Markov Decision Process (MDP)**, a standard transformation for POMDPs (Åström, 1965). In the belief-state MDP:

- **State**: The belief vector $b_t = (b_1(t), \ldots, b_N(t))$ for all $N$ players in the registry, augmented with the manager's squad, budget, and chip availability.
- **Action**: Transfers, lineup selection, captaincy choice, and chip deployment.
- **Transition**: $P(b_{t+1} \mid b_t, a_t)$, induced by the HDP-HMM transition matrices and the Bayesian belief update upon observing new data.
- **Reward**: $R(b_t, a_t) = \mathbb{E}_{s \sim b_t}[R(s, a_t)]$, the expected FPL points under the belief distribution.

The key insight is that **the RL agents in Chapters 8 and 9 do not operate on point estimates** (e.g., the MAP cluster assignment $\arg\max_k b_j^{(k)}(t)$). Instead, they operate on the **full distribution** $\{b_j^{(s)}(t)\}_{s=1}^S$, enabling them to:

1. **Hedge against uncertainty**: Thompson Sampling samples one belief state $\tilde{b}_j \sim \{b_j^{(s)}(t)\}_{s=1}^S$ and acts optimally with respect to it, naturally exploring when uncertainty is high.
2. **Plan across plausible worlds**: MCTS (Chapter 9, Section 9.8) builds a search tree where each node branches over multiple sampled belief states, explicitly evaluating decisions across different form scenarios.
3. **Learn robust value functions**: DDQN (Chapter 9, Section 9.5) takes the belief vector $b_j(t)$ as input to its state representation, learning to map distributional uncertainty into long-term value estimates.

### 7.9.3 Empirical Evaluation of Belief-State Quality

The quality of the belief states — their smoothness, entropy, and predictive power — directly determines the performance of all downstream decision engines. Chapter 12, Section 12.4 provides a detailed cluster interpretability analysis, including:

- The number of active form clusters discovered per position group (GK, DEF, MID, FWD).
- Cluster means and transition matrix heatmaps.
- Example player trajectories plotting $z_{j,t}$ over gameweeks.

Furthermore, **Ablation 1** in Chapter 12 (Section 12.5.1) directly compares the belief states produced by the iid HDP-GMM against the temporal HDP-HMM, evaluating prediction MAE, cluster stability, and belief state entropy. The hypothesis is that the HDP-HMM produces smoother, more predictive belief states due to its explicit modelling of form persistence.

## 7.10 Production Implementation Notes

### 7.10.1 JAX/XLA Kernel Library vs. Numba

The original design considered Numba for JIT-accelerating the Gibbs sampling loops (§7.5). The production implementation uses **JAX/XLA** instead, for three reasons:

1. **Hardware portability**: JAX compiles to CPU, GPU, and TPU without code changes, whereas Numba is CPU-only.
2. **Unified memory space**: Inference kernels (`mvn_log_density`, `sample_inverse_wishart`, `niw_posterior_update`) and the probabilistic model (NumPyro) share the same JAX device memory, eliminating costly data transfers between Python/NumPy and accelerator backends.
3. **Composable transformations**: `jax.vmap` (vectorisation), `jax.jit` (compilation), and `jax.lax.scan` (loop fusion) compose freely, enabling the full FFBS pipeline to be compiled into a single XLA kernel.

The kernel library (`src/utils/jax_kernels.py`) provides ten JIT-compiled functions covering all conjugate computations: MVN log-density, Inverse-Wishart sampling (Bartlett decomposition), NIW posterior updates, NIW sampling, FFBS forward/backward passes, batched categorical sampling, and I²GMM nested CRP assignment.

### 7.10.2 ASUGS Gradient Approximation

The theoretical ASUGS gradient (§7.8) involves the digamma function $\psi(\cdot)$, which can be numerically unstable for large arguments. The production implementation (`src/mlops/drift_detector.py`) uses a **simplified gradient approximation** based on the difference between observed and expected new cluster rates:

$$\text{gradient} \approx \frac{\text{observed\_new} - \text{expected\_new}}{\gamma + n - 1},$$

where $\text{observed\_new} = \min(K / n, 1)$ is the empirical cluster-to-observation ratio and $\text{expected\_new} = \gamma / (\gamma + n - 1)$. This approximation is algebraically equivalent to the first-order Taylor expansion of the digamma-based gradient and is numerically stable for all $\gamma > 0$.

### 7.10.3 KL-Divergence Drift Detection

The KL-divergence monitoring (§7.8, §11.6.2) uses **epsilon-smoothed probabilities** to avoid $\log(0)$ singularities:

$$\text{KL}(p \parallel q) = \sum_{k=1}^K \tilde{p}_k \log\left(\frac{\tilde{p}_k}{\tilde{q}_k}\right), \quad \text{where } \tilde{p}_k = \text{clip}(p_k, \epsilon, 1) \text{ and renormalised},$$

with $\epsilon = 10^{-10}$. The population-level drift $\text{KL}_{\text{pop}}(t)$ is compared against the 95th percentile of historical values to trigger retraining. The combined retraining decision (`should_retrain`) integrates three signals: KL drift, performance degradation (recent points below 25th percentile), and forced periodic schedule.

## 7.11 Summary

This chapter has provided a comprehensive treatment of posterior inference for the HDP-GMM and HDP-HMM. Gibbs sampling with conjugate updates provides exact (asymptotic) posterior samples. NUTS via NumPyro handles non-conjugate models with automatic gradient computation. JAX/XLA accelerates all conjugate computations (MVN, NIW, FFBS) into hardware-agnostic JIT-compiled kernels, replacing the original Numba-based approach. Variational Inference offers a faster alternative for online updates. A suite of convergence diagnostics ensures reliable inference. The ASUGS algorithm, coupled with KL-divergence drift detection, enables event-driven online adaptation of concentration parameters.

Most importantly, Section 7.9 has established that the MCMC posterior samples are not isolated statistical outputs — they are the **belief-state representation** that forms the mathematical input to all downstream decision engines. Thompson Sampling (Chapter 8) samples from these belief states to achieve natural exploration. MCTS (Chapter 9) plans across sampled belief states for multi-gameweek lookahead. DDQN (Chapter 9) learns to map belief-state uncertainty into long-term value estimates.

The next chapter shows how these belief states drive Thompson Sampling decisions, the first step (Step A) in the three-stage planning hierarchy.
