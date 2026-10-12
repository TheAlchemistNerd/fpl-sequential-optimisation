# Chapter 5: HDP-Gaussian Mixture Model with Gaussian-Wishart Base Measure

## 5.1 Introduction

The Hierarchical Dirichlet Process Gaussian Mixture Model (HDP-GMM) is the core latent state estimator in the FPL sequential optimisation framework. It combines the nonparametric clustering power of the HDP (Chapter 4) with a multivariate Gaussian emission model, where each cluster (form profile) is characterised by a mean vector and a covariance matrix. The Gaussian-Wishart (more precisely, Normal-Inverse-Wishart) base measure provides conjugacy, enabling efficient closed-form posterior updates.

The choice of the Normal-Inverse-Wishart base measure is motivated by the structure of the engineered feature space. As detailed in Chapter 3, raw performance counts are transformed into continuous, per-90 normalised rates (e.g., xG/90, ICT/90) and standardised. For such continuous, multivariate data, the Gaussian emission provides a natural likelihood, while the Wishart component captures the full covariance structure between features—modelling, for instance, the correlation between a player's goal threat and their involvement in buildup play. Furthermore, the NIW is the conjugate prior for the multivariate Gaussian, enabling efficient closed-form posterior updates that are critical for the scalability of the Gibbs sampler. While alternative base measures (e.g., Poisson-Gamma for raw counts, Negative-Binomial for over-dispersed points) are theoretically applicable to specific data modalities, the Gaussian-Wishart serves as the principled default for the continuous feature representation central to this framework.

This chapter provides a detailed mathematical treatment of the HDP-GMM, including the NIW prior specification, conjugate posterior derivations, the modelling of multi-modal player form profiles, the I²GMM extension for within-cluster multi-modality, and the construction of belief states for downstream decision-making.

## 5.2 Model Specification

### 5.2.1 Generative Process

The HDP-GMM for FPL player performance data is specified as follows. Let $J$ denote the number of players, and for each player $j$, let $T_j$ denote the number of gameweek observations. Each observation $x_{j,i} \in \mathbb{R}^D$ is a $D$-dimensional feature vector (as defined in Chapter 3).

**Global level:**

$$\begin{aligned}
\gamma &\sim \text{Gamma}(a_\gamma, b_\gamma), \\
\beta'_k &\stackrel{\text{i.i.d.}}{\sim} \text{Beta}(1, \gamma), \quad k = 1, 2, \ldots, \\
\beta_k &= \beta'_k \prod_{l=1}^{k-1} (1 - \beta'_l), \\
(\mu_k, \Sigma_k) &\stackrel{\text{i.i.d.}}{\sim} \text{NIW}(m_0, \kappa_0, \Psi_0, \nu_0), \quad k = 1, 2, \ldots
\end{aligned}$$

**Player level:**

$$\begin{aligned}
\alpha &\sim \text{Gamma}(a_\alpha, b_\alpha), \\
\pi_{j,k} &= \beta'_j \text{ stick-breaking with base weights } \boldsymbol{\beta}, \text{ concentration } \alpha.
\end{aligned}$$

**Observation level:**

$$\begin{aligned}
z_{j,i} \mid \boldsymbol{\pi}_j &\sim \text{Categorical}(\boldsymbol{\pi}_j), \\
x_{j,i} \mid z_{j,i} = k, \mu_k, \Sigma_k &\sim \mathcal{N}(\mu_k, \Sigma_k).
\end{aligned}$$

### 5.2.2 The Normal-Inverse-Wishart Prior

The Normal-Inverse-Wishart (NIW) distribution is a joint prior over the mean $\mu$ and covariance $\Sigma$ of a multivariate Gaussian. It is the conjugate prior for the multivariate Gaussian with unknown mean and covariance.

A random pair $(\mu, \Sigma)$ follows an NIW distribution with parameters $(m_0, \kappa_0, \Psi_0, \nu_0)$, written $(\mu, \Sigma) \sim \text{NIW}(m_0, \kappa_0, \Psi_0, \nu_0)$, if:

$$\begin{aligned}
\Sigma &\sim \text{Inverse-Wishart}_{\nu_0}(\Psi_0), \\
\mu \mid \Sigma &\sim \mathcal{N}(m_0, \Sigma / \kappa_0).
\end{aligned}$$

The density is:

$$\begin{aligned}
&p(\mu, \Sigma \mid m_0, \kappa_0, \Psi_0, \nu_0) \\
&\quad = \frac{|\Psi_0|^{\nu_0/2}}{2^{\nu_0 D/2} \pi^{D/2} \Gamma_D(\nu_0/2)} \cdot \frac{1}{\kappa_0^{D/2}} \cdot |\Sigma|^{-(\nu_0 + D + 2)/2} \\
&\quad \cdot \exp\left( -\frac{1}{2} \operatorname{tr}(\Psi_0 \Sigma^{-1}) - \frac{\kappa_0}{2} (\mu - m_0)^\top \Sigma^{-1} (\mu - m_0) \right),
\end{aligned}$$

where $\Gamma_D$ is the multivariate Gamma function.

### 5.2.3 Hyperparameter Elicitation

The NIW hyperparameters have clear interpretations:

- **$m_0$ (prior mean)**: Since features are standardised (Chapter 3), $m_0 = \mathbf{0}$ is set.
- **$\kappa_0$ (prior precision on mean)**: Controls the prior confidence in $m_0$. A small value (e.g., $\kappa_0 = 1$) yields a weakly informative prior, allowing the data to dominate.
- **$\nu_0$ (prior degrees of freedom)**: Must satisfy $\nu_0 \geq D$ for the Inverse-Wishart to be proper. $\nu_0 = D + 2$ is set, the minimum for a finite prior mean of $\Sigma$.
- **$\Psi_0$ (prior scale matrix)**: Controls the prior expected covariance. Since features are standardised, $\Psi_0 = I_D$ is set, the identity matrix. The prior expected covariance is:

  $$\mathbb{E}[\Sigma] = \frac{\Psi_0}{\nu_0 - D - 1} = \frac{I_D}{(D + 2) - D - 1} = I_D.$$

These settings yield a weakly informative prior that is centred at the standardised feature distribution but allows the data to shift the posterior substantially.

## 5.3 Conjugate Posterior Updates

### 5.3.1 Sufficient Statistics

For a cluster $k$ with $n_k$ assigned observations $\{x_{j,i} : z_{j,i} = k\}$, the sufficient statistics are:

$$\begin{aligned}
n_k &= \sum_{j,i} \mathbb{I}(z_{j,i} = k), \\
\bar{x}_k &= \frac{1}{n_k} \sum_{j,i: z_{j,i} = k} x_{j,i}, \\
S_k &= \sum_{j,i: z_{j,i} = k} (x_{j,i} - \bar{x}_k)(x_{j,i} - \bar{x}_k)^\top.
\end{aligned}$$

### 5.3.2 Posterior Parameters

Given the NIW prior and the Gaussian likelihood, the posterior is also NIW:

$$(\mu_k, \Sigma_k) \mid \{x_{j,i} : z_{j,i} = k\} \sim \text{NIW}(m_k, \kappa_k, \Psi_k, \nu_k),$$

with updated parameters:

$$\begin{aligned}
\kappa_k &= \kappa_0 + n_k, \\
m_k &= \frac{\kappa_0 m_0 + n_k \bar{x}_k}{\kappa_0 + n_k}, \\
\nu_k &= \nu_0 + n_k, \\
\Psi_k &= \Psi_0 + S_k + \frac{\kappa_0 n_k}{\kappa_0 + n_k} (\bar{x}_k - m_0)(\bar{x}_k - m_0)^\top.
\end{aligned}$$

The posterior mean of $\mu_k$ is a weighted average of the prior mean and the sample mean:

$$\mathbb{E}[\mu_k \mid \text{data}] = m_k = \frac{\kappa_0}{\kappa_0 + n_k} m_0 + \frac{n_k}{\kappa_0 + n_k} \bar{x}_k.$$

The posterior expected covariance is:

$$\mathbb{E}[\Sigma_k \mid \text{data}] = \frac{\Psi_k}{\nu_k - D - 1}.$$

### 5.3.3 Predictive Distribution

The posterior predictive distribution for a new observation from cluster $k$ is a multivariate Student-t distribution:

$$p(x_{\text{new}} \mid \{x_{j,i} : z_{j,i} = k\}) = \text{Student-t}_{\nu_k - D + 1}\left(m_k, \frac{(\kappa_k + 1) \Psi_k}{\kappa_k (\nu_k - D + 1)}\right).$$

This predictive distribution is used in the Gibbs sampling update for cluster assignments.

## 5.4 Gibbs Sampling for the HDP-GMM

### 5.4.1 Collapsed Gibbs Sampler

A collapsed Gibbs sampler is employed that integrates out $(\mu_k, \Sigma_k)$ and samples the cluster assignments $z_{j,i}$ directly. The update for $z_{j,i}$ is:

$$P(z_{j,i} = k \mid \mathbf{z}_{-ji}, \mathbf{x}) \propto \begin{cases} \frac{n_{k, -ji}}{n - 1 + \gamma} \cdot p(x_{j,i} \mid \mathbf{x}_{-ji}^{(k)}) & \text{if } k \text{ is occupied}, \\ \frac{\gamma}{n - 1 + \gamma} \cdot p(x_{j,i} \mid H) & \text{if } k \text{ is new}. \end{cases}$$

Here, $p(x_{j,i} \mid \mathbf{x}_{-ji}^{(k)})$ is the predictive density under cluster $k$ (Student-t), and $p(x_{j,i} \mid H)$ is the prior predictive density (also Student-t, with prior parameters).

### 5.4.2 Truncated Stick-Breaking

In practice, the stick-breaking is truncated at $K$ components. The truncated approximation introduces a residual weight $\pi_{K+1} = \prod_{l=1}^K (1 - \beta'_l)$, which is typically negligible for $K > 30\gamma$. In the implementation, $K = 20$ is used as a default, which is sufficient for $\gamma \leq 5$.

### 5.4.3 Full Gibbs Sampling Algorithm

```
Input: Data {x_{j,i}}, truncation K, hyperparameters (m_0, κ_0, Ψ_0, ν_0)
Initialise: cluster assignments z_{j,i}, global weights β, player weights π_j

for iteration = 1 to N_iterations:
    # Update global concentration parameter
    Sample γ from its full conditional (Gamma posterior)

    # Update stick-breaking weights
    for k = 1 to K-1:
        Sample β'_k from its full conditional (Beta posterior)
    Compute β_k from β'_k

    # Update player-specific concentration
    Sample α from its full conditional (Gamma posterior)

    # Update player-specific mixture weights
    for player j:
        Sample π_j from its full conditional (Dirichlet posterior)

    # Update cluster assignments
    for each observation (j, i):
        For each cluster k = 1, ..., K:
            Compute prior: P(z_{j,i} = k | π_j) = π_{j,k}
            Compute likelihood: p(x_{j,i} | z_{j,i} = k, data_{-ji}^{(k)})
        Sample z_{j,i} from the categorical posterior

    # Update cluster parameters (μ_k, Σ_k)
    for each cluster k:
        Compute sufficient statistics (n_k, x̄_k, S_k)
        Sample (μ_k, Σ_k) from NIW(m_k, κ_k, Ψ_k, ν_k)
```

## 5.5 Multi-Modal Player Form Profiles

### 5.5.1 The Challenge of Within-Cluster Multi-Modality

The HDP-GMM assigns a single Gaussian to each cluster. This means each discovered form profile is unimodal and ellipsoidal in shape. However, FPL performance data often exhibits **within-cluster multi-modality**. For example:

- An "Elite Form" state may contain two sub-populations: "Elite at Home" and "Elite Away."
- A midfielder's "Creative Form" may manifest as either high-key-pass/low-goal or low-key-pass/high-goal performances.
- A defender's "Solid Form" may be either "Clean Sheet + No Attacking Contribution" or "Clean Sheet + High Involvement."

A single Gaussian cannot capture these sub-structures, which may lead the model to fragment a single interpretable form state into multiple small, hard-to-distinguish clusters.

### 5.5.2 The I²GMM Extension

The **Infinite Mixture of Infinite Gaussian Mixtures (I²GMM)** (Yerebakan et al., 2014) replaces each single-Gaussian cluster with its own Gaussian DP mixture. Its higher-level DP draws meta-cluster parameters, and each meta-cluster defines the base measure for a lower-level DP over component means:

$$G \sim \mathrm{DP}(\gamma,H), \qquad H=\mathrm{NIW}(\mu_0,\Sigma_0,\kappa_0,m), \qquad (\mu_k,\Sigma_k) \sim G.$$

For each meta-cluster $k$, define $H_k=\mathcal{N}(\mu_k,\Sigma_k/\kappa_1)$ and draw $G_k\sim\mathrm{DP}(\eta,H_k)$. An observation assigned to meta-cluster $k$ draws a component mean $\mu_i\sim G_k$ and then $x_i\sim\mathcal{N}(\mu_i,\Sigma_k)$. Thus component means vary within the cluster while its covariance is shared; $\kappa_1$ controls within-cluster spread and $\kappa_0$ controls between-cluster spread.

Equivalently, $G=\sum_k\beta_k\delta_{(\mu_k,\Sigma_k)}$ with $\beta\sim\mathrm{GEM}(\gamma)$, while $G_k=\sum_m\pi_{k,m}\delta_{\mu_{k,m}}$ with $\pi_k\sim\mathrm{GEM}(\eta)$ and $\mu_{k,m}\sim H_k$. The generative assignments are $z_i\sim\mathrm{Categorical}(\beta)$, $c_i\sim\mathrm{Categorical}(\pi_{z_i})$, and $x_i\sim\mathcal{N}(\mu_{z_i,c_i},\Sigma_{z_i})$. The earlier generative pseudocode uses $\alpha$ for the within-profile concentration; here it is denoted $\eta$ to match the DP definition.

### 5.5.3 Collapsed Gibbs Sampler for I²GMM

The collapsed Gibbs sampler for I²GMM has an additional layer:

1. Sample each observation's within-cluster component assignment from the CRP associated with its cluster-specific DP.
2. Sample component-to-meta-cluster assignments under the higher-level DP, allowing new clusters when supported by the posterior predictive probability.
3. Update component means and shared meta-cluster covariance parameters from their conjugate posterior.

The partial parallelisability of the collapsed Gibbs sampler (Yerebakan et al., 2014) enables scaling to large datasets, which is essential for the FPL problem with thousands of player-gameweek observations.

## 5.6 Alternative Base Measures

While the Gaussian-Wishart base measure is well-suited to the continuous, multivariate feature vectors produced by the engineering pipeline, the HDP framework is agnostic to the choice of likelihood. Alternative base measures can be deployed to model specific characteristics of the underlying data that deviate from Gaussian assumptions. These extensions allow the model to handle raw discrete counts, heavy-tailed distributions, or complex mixtures of data types without altering the core nonparametric clustering mechanism.

### 5.6.1 HDP-Poisson-Gamma Model

For discrete count data (goals, assists, clean sheets), the Poisson likelihood is natural:

$$x_{j,i}^{(e)} \mid \lambda_k^{(e)} \sim \text{Poisson}(\lambda_k^{(e)}),$$

with Gamma prior:

$$\lambda_k^{(e)} \sim \text{Gamma}(\alpha_0, \beta_0).$$

The conjugate posterior is:

$$\lambda_k^{(e)} \mid \text{data} \sim \text{Gamma}\left(\alpha_0 + \sum x_{j,i}^{(e)}, \; \beta_0 + n_k\right).$$

This model captures the underlying "rate" at which a player produces discrete events in each form state.

### 5.6.2 HDP-Negative-Binomial Model

For total FPL points, which exhibit over-dispersion (variance > mean), the Negative Binomial is preferred:

$$x_{j,i} \mid r_k, p_k \sim \text{Negative-Binomial}(r_k, p_k),$$

with factorised prior:

$$p_k \sim \text{Beta}(a_0, b_0), \quad r_k \sim \text{Gamma}(c_0, d_0).$$

The posterior for $p_k$ is conjugate:

$$p_k \mid \text{data} \sim \text{Beta}\left(a_0 + \sum x_{j,i}, \; b_0 + n_k r_k\right).$$

The posterior for $r_k$ has no closed form and requires Metropolis-Hastings.

### 5.6.3 Product Base Measure

A comprehensive model can combine all base measures:

$$H = \text{NIW} \times \text{Gamma}^{\otimes E} \times \text{Beta} \times \text{Gamma},$$

where $E$ is the number of discrete event types. Each hidden state then represents a complete player profile with continuous stats, discrete event rates, and total points distribution.

## 5.7 Belief State Construction

The output of the HDP-GMM for player $j$ at gameweek $t$ is a **belief state** — a probability distribution over the $K$ latent profiles:

$$b_j(t) = \left( P(z_{j,t} = 1 \mid x_{j,1:t}), \; P(z_{j,t} = 2 \mid x_{j,1:t}), \; \ldots, \; P(z_{j,t} = K \mid x_{j,1:t}) \right).$$

This belief vector is computed from the posterior samples of the Gibbs sampler. For each MCMC sample $s = 1, \ldots, S$, a cluster assignment $z_{j,t}^{(s)}$ is obtained. The belief state is the empirical frequency:

$$b_j^{(k)}(t) = \frac{1}{S} \sum_{s=1}^S \mathbb{I}(z_{j,t}^{(s)} = k).$$

The belief state is a rich, interpretable representation of player form. For example:

- $b_{\text{Salah}}(t) = (0.85, 0.10, 0.05, 0, \ldots)$ indicates 85% probability of "Elite Form."
- $b_{\text{Player X}}(t) = (0.1, 0.3, 0.4, 0.2, \ldots)$ indicates uncertainty — the player's recent performances are consistent with multiple profiles.

This belief vector becomes a primary input to the decision engine (Thompson Sampling, DDQN, MFG, MFC), replacing point estimates with a full probabilistic description of form.

## 5.8 Model Specification for Inference

The HDP-GMM is specified for inference using the truncated stick-breaking construction. The following pseudocode describes the model structure passed to the NUTS sampler:

```
Algorithm: HDP-GMM Model Specification
Input: Feature matrix X (N × D), player assignment vector player_ids (N), truncation K = 20
Output: Posterior samples over {γ, α, β, μ, Σ, π, z}

Stage 1: Sample Hyperparameters
    γ ← sample from Gamma(1.0, 1.0)          // global concentration
    α ← sample from Gamma(1.0, 1.0)           // player-specific concentration

Stage 2: Construct Global Mixture Weights via Stick-Breaking
    For k = 1 to K - 1:
        β'_k ← sample from Beta(1.0, γ)
    β ← stick_breaking(β')                     // β_k = β'_k × ∏_{l<k}(1 - β'_l)

Stage 3: Sample Cluster Parameters (NIW Prior)
    For k = 1 to K:
        Σ_k ← sample from Inverse-Wishart(df = D + 2, scale = I_D)
        μ_k ← sample from MultivariateNormal(mean = 0, cov = Σ_k / 1.0)

Stage 4: Sample Player-Specific Mixture Weights
    For each player j = 1 to num_players:
        For k = 1 to K - 1:
            π'_{j,k} ← sample from Beta(α × β_k, α × (1 - β_k))
        π_j ← stick_breaking(π'_j)

Stage 5: Define Likelihood
    For each observation i = 1 to N:
        j ← player_ids[i]                       // which player does this observation belong to?
        z_i ← sample from Categorical(π_j)      // cluster assignment
        x_i ~ MultivariateNormal(mean = μ_{z_i}, cov = Σ_{z_i})
```

The model is executed using the NUTS sampler, leveraging just-in-time compilation for efficiency. The NUTS sampler automatically handles the gradient computations via automatic differentiation, eliminating the need for manual posterior derivations.

## 5.9 Covariance Structure and Feature Correlations

The Gaussian-Wishart base measure captures not just the mean performance level of each form profile, but also the **covariance structure** — the correlations between different performance metrics within that profile.

For example, an "Elite Attacking Form" profile for a forward may exhibit:

- High mean xG per 90 and high mean shots per 90.
- Positive correlation between xG and goals (the more chances, the more goals).
- Low variance in both (consistent output).

A "Creative Form" profile for a midfielder may exhibit:

- High mean xA and high mean key passes.
- Negative correlation between xA and goals (focuses on creating rather than finishing).
- Moderate variance.

The Inverse-Wishart prior on $\Sigma_k$ regularises the covariance estimates, shrinking them towards the identity matrix when data is sparse. This prevents overfitting and ensures numerical stability.

## 5.10 Advanced Representational Considerations: I²GMM

The complete I²GMM treatment—including its motivation, two-level DP construction, stick-breaking representation, and collapsed Gibbs updates—is in Sections 5.5.1–5.5.3.

## 5.11 Production Implementation Notes

This section documents how the theoretical framework above is realised in the codebase, including design decisions, library choices, and the relationship between conjugate and gradient-based inference methods.

### 5.11.1 NUTS vs. Collapsed Gibbs Sampling

The collapsed Gibbs sampler described in Section 5.4 exploits the NIW-Gaussian conjugacy to produce closed-form posterior updates. This is the **conceptual algorithm** — the one that makes the statistical structure transparent. In practice, the production implementation uses the **No-U-Turn Sampler (NUTS)** via NumPyro (§5.8), which treats the full joint posterior (stick-breaking weights, cluster parameters, assignments) as a single differentiable computation graph.

| Aspect | Collapsed Gibbs (§5.4) | NUTS (§5.8) |
|---|---|---|
| Conjugacy exploited | Yes — closed-form NIW updates | No — automatic differentiation |
| Discrete assignments | Explicitly sampled (categorical) | Marginalised or enumerated |
| GPU support | Requires manual porting (Numba/JAX) | Native via JAX/XLA |
| Implementation | Imperative loop over observations | Declarative model function |
| Diagnostics | Trace plots, R̂, ESS | R̂, ESS, divergences, tree depth |

The NUTS approach is preferred for the **initial model development** because it requires no manual derivation of full conditionals and automatically handles the truncated stick-breaking construction. The collapsed Gibbs sampler remains available as a **JAX-accelerated kernel library** (`src/utils/jax_kernels.py`) for scenarios where conjugacy provides a significant speedup or where discrete assignments must be explicitly sampled.

### 5.11.2 JAX/XLA Accelerated Conjugate Kernels

While the theoretical derivation in Section 5.4 presents the Gibbs sampler as a loop over observations, the production implementation (`src/utils/jax_kernels.py`) leverages **JAX/XLA** to compile these updates into efficient, hardware-agnostic kernels. This replaces the Numba-accelerated approach originally envisaged, providing native GPU/TPU support.

The library provides the following JIT-compiled kernels:

1.  **MVN Log-Density**: `mvn_log_density(x, mean, precision)` computes the multivariate normal log-likelihood for all clusters simultaneously using `jax.vmap`. This eliminates the Python interpreter overhead of the Mahalanobis distance loop.
2.  **Bartlett Decomposition**: The Wishart sampling for the precision matrix is implemented via the Bartlett decomposition, which generates lower-triangular matrices with $\chi^2$ diagonals. This is numerically stable and fully differentiable, replacing the less stable Cholesky-based approaches.
3.  **Vectorised NIW Updates**: The sufficient statistics ($n_k, \bar{x}_k, S_k$) are computed using `jax.vmap` across all clusters simultaneously. This allows the posterior updates to be computed in parallel rather than sequentially.
4.  **Parallel I²GMM Assignment**: The nested CRP assignment for the I²GMM extension (§5.5) uses `jax.vmap` to parallelise the sampling of sub-component assignments across all players.

These kernels allow the conjugate steps to run on GPU/TPU, significantly reducing inference time for large player registries ($N > 10{,}000$). Profiling benchmarks comparing these conjugate kernels to NUTS are available in `tests/test_jax_profiling.py`.

This chapter has presented the HDP-GMM with Gaussian-Wishart base measure in full mathematical detail. The NIW prior provides conjugacy for efficient posterior updates. The truncated stick-breaking construction enables practical implementation. The I²GMM extension addresses within-cluster multi-modality. Alternative base measures (Poisson-Gamma, Negative-Binomial) handle discrete count data and over-dispersed points. The belief state output provides a rich, probabilistic representation of player form. The NumPyro implementation leverages JAX's acceleration for scalable inference.

The next chapter extends the HDP-GMM with a Hidden Markov Model layer, capturing the temporal dynamics of player form — not just which state a player is in, but how they transition between states over gameweeks.
