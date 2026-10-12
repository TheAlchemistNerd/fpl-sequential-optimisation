# Chapter 4: Bayesian Nonparametrics — The Chinese Restaurant Process, Dirichlet Process, and Hierarchical Dirichlet Process

## 4.1 Motivation: Why Nonparametric Bayesian Methods for FPL

The central modelling hypothesis of this dissertation is that a player's observable FPL performance is driven by a **latent form state** that evolves over the season. The critical question is: *how many distinct form states exist?*

A parametric approach would require fixing the number of states $K$ in advance — for example, assuming that every player cycles through exactly three form states (Hot, Normal, Cold). This is unsatisfactory for several reasons:

1. **Unknown cardinality**: There is no theoretical or empirical basis for fixing $K$ a priori. Some players may exhibit only two distinguishable states (fit vs. injured), while others may exhibit five or more (elite home, elite away, good, average, poor, fatigued).

2. **Heterogeneity across positions**: Goalkeepers, defenders, midfielders, and forwards have fundamentally different performance distributions. A single $K$ cannot capture this heterogeneity.

3. **Season evolution**: New form states may emerge over the course of a season (e.g., a player transitions to a new club, changes role, or recovers from a long-term injury). A fixed $K$ cannot adapt to such structural changes.

**Bayesian nonparametric methods** resolve these issues by placing a prior distribution over the *number* of clusters (form states), allowing the data to determine the appropriate complexity. The term "nonparametric" is slightly misleading — these models do have parameters, but the number of parameters grows with the data, potentially without bound.

The foundational object in Bayesian nonparametrics is the **Dirichlet Process (DP)**, a stochastic process whose realisations are probability distributions. The DP serves as a prior over distributions, enabling mixture models with an unbounded number of components. The **Hierarchical Dirichlet Process (HDP)** extends the DP to grouped data, allowing multiple groups (players) to share a common set of clusters (form profiles) while retaining group-specific variation.

This chapter develops the theoretical foundations of the DP and HDP from first principles, progressing from the Chinese Restaurant Process to the stick-breaking construction, and culminating in the Chinese Restaurant Franchise representation of the HDP.

## 4.2 The Dirichlet Distribution

Before defining the Dirichlet Process, the finite-dimensional **Dirichlet distribution** is reviewed, which serves as its finite analogue.

### 4.2.1 Definition

The Dirichlet distribution is a distribution over probability vectors (simplexes). A random vector $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$ follows a Dirichlet distribution with concentration parameters $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)$, written $\boldsymbol{\pi} \sim \text{Dir}(\boldsymbol{\alpha})$, if its density is:

$$p(\boldsymbol{\pi} \mid \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K \pi_k^{\alpha_k - 1}, \quad \text{where } \sum_{k=1}^K \pi_k = 1, \; \pi_k \geq 0,$$

and $B(\boldsymbol{\alpha})$ is the multivariate Beta function:

$$B(\boldsymbol{\alpha}) = \frac{\prod_{k=1}^K \Gamma(\alpha_k)}{\Gamma\left(\sum_{k=1}^K \alpha_k\right)}.$$

### 4.2.2 Properties

- **Mean**: $\mathbb{E}[\pi_k] = \frac{\alpha_k}{\alpha_0}$, where $\alpha_0 = \sum_{k=1}^K \alpha_k$.
- **Variance**: $\text{Var}[\pi_k] = \frac{\alpha_k (\alpha_0 - \alpha_k)}{\alpha_0^2 (\alpha_0 + 1)}$.
- **Concentration**: The scalar $\alpha_0$ controls the concentration of the distribution. When $\alpha_0$ is small (e.g., $\alpha_0 < 1$), the distribution concentrates near the corners of the simplex (sparse vectors). When $\alpha_0$ is large, it concentrates near the center (uniform vectors).

### 4.2.3 Dirichlet-Multinomial Conjugacy

The Dirichlet distribution is the conjugate prior for the Multinomial distribution. If $\boldsymbol{\pi} \sim \text{Dir}(\boldsymbol{\alpha})$ and $\mathbf{n} \mid \boldsymbol{\pi} \sim \text{Multinomial}(N, \boldsymbol{\pi})$, then the posterior is:

$$\boldsymbol{\pi} \mid \mathbf{n} \sim \text{Dir}(\boldsymbol{\alpha} + \mathbf{n}).$$

This conjugacy is essential for Gibbs sampling, as it enables closed-form posterior updates.

## 4.3 The Dirichlet Process

### 4.3.1 Formal Definition

A **Dirichlet Process** is a stochastic process whose realisations are probability distributions. Formally, a Dirichlet Process with concentration parameter $\gamma > 0$ and base distribution $G_0$, denoted $\text{DP}(\gamma, G_0)$, is defined by the property that for any finite measurable partition $(A_1, \ldots, A_r)$ of the sample space $\Theta$:

$$(G(A_1), \ldots, G(A_r)) \sim \text{Dir}(\gamma G_0(A_1), \ldots, \gamma G_0(A_r)).$$

Here, $G$ is a random distribution drawn from the DP, and $G_0$ is the **base distribution** (the mean of the DP: $\mathbb{E}[G] = G_0$). The concentration parameter $\gamma$ controls how closely $G$ concentrates around $G_0$: small $\gamma$ yields sparse, discrete distributions; large $\gamma$ yields distributions close to $G_0$.

### 4.3.2 Almost Sure Discreteness

A crucial property of the Dirichlet Process is that its realisations are **almost surely discrete**. That is, if $G \sim \text{DP}(\gamma, G_0)$, then with probability 1, $G$ is a discrete distribution with countably infinite support:

$$G = \sum_{k=1}^\infty \pi_k \delta_{\theta_k},$$

where $\delta_{\theta}$ is a point mass at $\theta$, the atoms $\theta_k \stackrel{\text{i.i.d.}}{\sim} G_0$, and the weights $\pi_k$ are random and sum to 1.

This discreteness is what enables clustering: drawing parameters $\theta_i \sim G$ for each data point $i$ means that multiple data points may share the same $\theta_k$ (because $G$ is discrete), which defines a cluster.

### 4.3.3 Stick-Breaking Construction

The **stick-breaking construction** (Sethuraman, 1994) provides an explicit representation of the weights $\pi_k$ and atoms $\theta_k$ of a draw from the Dirichlet Process.

Let $\beta'_k \stackrel{\text{i.i.d.}}{\sim} \text{Beta}(1, \gamma)$ for $k = 1, 2, \ldots$. Define the weights recursively:

$$\pi_k = \beta'_k \prod_{j=1}^{k-1} (1 - \beta'_j).$$

The atoms are drawn independently from the base distribution:

$$\theta_k \stackrel{\text{i.i.d.}}{\sim} G_0.$$

Then the random distribution:

$$G = \sum_{k=1}^\infty \pi_k \delta_{\theta_k}$$

is a realisation of $\text{DP}(\gamma, G_0)$.

The metaphor is intuitive: start with a stick of unit length. Break off a fraction $\beta'_1$ — this is $\pi_1$. From the remaining stick of length $(1 - \beta'_1)$, break off a fraction $\beta'_2$ — this is $\pi_2 = \beta'_2 (1 - \beta'_1)$. Continue indefinitely. The resulting weights sum to 1:

$$\sum_{k=1}^\infty \pi_k = 1 \quad \text{almost surely}.$$

The distribution over weights $(\pi_1, \pi_2, \ldots)$ is known as the **Griffiths-Engen-McCloskey (GEM) distribution** with parameter $\gamma$, denoted $\text{GEM}(\gamma)$.

### 4.3.4 Truncated Approximation

In practice, working with infinitely many components is infeasible. A common approximation is to truncate the stick-breaking after $K$ terms, setting:

$$\pi_k = \beta'_k \prod_{j=1}^{k-1} (1 - \beta'_j) \quad \text{for } k = 1, \ldots, K-1,$$
$$\pi_K = \prod_{j=1}^{K-1} (1 - \beta'_j).$$

This ensures $\sum_{k=1}^K \pi_k = 1$. The truncation error decreases exponentially with $K$. For concentration parameter $\gamma$, a safe choice is $K > 30 \gamma$, which captures over 99% of the mass.

## 4.4 The Chinese Restaurant Process

### 4.4.1 The Metaphor

The **Chinese Restaurant Process (CRP)** provides an intuitive, constructive definition of the clustering distribution induced by a Dirichlet Process. As described in the lecture notes of Blei et al. (2007), consider a restaurant with an infinite number of tables, labelled $k = 1, 2, \ldots$. Customers (data points) arrive sequentially and choose a table according to the following rule:

- The first customer sits at table 1.
- The $n$-th customer chooses:
  - An **occupied table** $k$ with probability proportional to the number of customers already seated there:
    $$P(z_n = k \mid z_1, \ldots, z_{n-1}) = \frac{n_k}{n - 1 + \gamma},$$
    where $n_k$ is the number of customers at table $k$.
  - A **new, previously unoccupied table** with probability:
    $$P(z_n = K_{\text{new}} \mid z_1, \ldots, z_{n-1}) = \frac{\gamma}{n - 1 + \gamma}.$$

Here, $\gamma > 0$ is the concentration parameter, and $K_{\text{new}}$ is the index of the next available table.

### 4.4.2 The Rich-Get-Richer Property

The CRP exhibits a "rich-get-richer" property: tables with more customers are more likely to attract new customers. This leads to a power-law distribution of table sizes: a few tables are large, while many tables have only a few customers. This is empirically appropriate for FPL form states: a few dominant form profiles (e.g., "Standard Performance") will contain many observations, while specialised profiles (e.g., "Elite Under Pressure") may contain only a few.

### 4.4.3 Exchangeability

The CRP is **exchangeable**: the probability of any partition of customers into tables is invariant to the order in which customers arrive. This follows from the fact that the joint distribution of $(z_1, \ldots, z_n)$ depends only on the counts $\{n_k\}$, not on the ordering.

By de Finetti's theorem, exchangeability implies the existence of a latent random measure (the Dirichlet Process) such that, conditioned on this measure, the $z_i$ are i.i.d. This establishes the equivalence between the CRP and the DP.

### 4.4.4 Expected Number of Clusters

An important property of the CRP is the expected number of occupied tables (clusters) after $n$ customers:

$$\mathbb{E}[K_n] = \sum_{i=1}^n \frac{\gamma}{\gamma + i - 1} \approx \gamma \log\left(1 + \frac{n}{\gamma}\right) \approx \gamma \log n \quad \text{for large } n.$$

This **logarithmic growth** implies that the number of clusters grows slowly with the data size. For $\gamma = 1$ and $n = 10{,}000$ player-gameweek observations, the expected number of clusters is approximately $\log(10{,}000) \approx 9.2$.

This is a key result for FPL: even with thousands of observations per player, the number of latent form states remains manageable (single digits to low double digits).

### 4.4.5 CRP Mixture Models

The CRP can be used to define a **mixture model** by endowing each table with a parameter. Specifically:

1. Customers choose tables according to the CRP.
2. Each table $k$ has a parameter $\theta_k \sim G_0$.
3. Customer $i$, seated at table $z_i$, generates observation $x_i \sim F(\theta_{z_i})$.

The marginal distribution over observations is:

$$p(x_1, \ldots, x_n) = \int \cdots \int \prod_{i=1}^n F(\theta_{z_i}) \prod_{k=1}^{K_n} G_0(d\theta_k) \cdot P(z_1, \ldots, z_n).$$

When $F$ is a Gaussian distribution and $G_0$ is a Normal-Inverse-Wishart distribution, this yields the **DP Gaussian Mixture Model (DP-GMM)**, which is the foundation for the HDP-GMM.

## 4.5 Dirichlet Process Mixture Models

### 4.5.1 The Model

A Dirichlet Process Mixture Model (DPMM) is defined as:

$$\begin{aligned}
G &\sim \text{DP}(\gamma, G_0), \\
\theta_i \mid G &\stackrel{\text{i.i.d.}}{\sim} G, \\
x_i \mid \theta_i &\sim F(\theta_{z_i}).
\end{aligned}$$

Equivalently, using the stick-breaking construction:

$$\begin{aligned}
\beta'_k &\stackrel{\text{i.i.d.}}{\sim} \text{Beta}(1, \gamma), \quad k = 1, 2, \ldots, \\
\pi_k &= \beta'_k \prod_{j=1}^{k-1} (1 - \beta'_j), \\
\theta_k &\stackrel{\text{i.i.d.}}{\sim} G_0, \\
z_i \mid \boldsymbol{\pi} &\sim \text{Categorical}(\boldsymbol{\pi}), \\
x_i \mid z_i = k, \theta_k &\sim F(\theta_k).
\end{aligned}$$

### 4.5.2 Collapsed Gibbs Sampling

For posterior inference, $G$ is typically integrated out and the cluster assignments $z_i$ are sampled directly. The collapsed Gibbs sampler iteratively updates each $z_i$ conditional on all other assignments $\mathbf{z}_{-i}$ and the data:

$$P(z_i = k \mid \mathbf{z}_{-i}, \mathbf{x}) \propto \begin{cases} \frac{n_{k, -i}}{n - 1 + \gamma} \cdot F(x_i \mid \theta_k) & \text{if } k \text{ is occupied}, \\ \frac{\gamma}{n - 1 + \gamma} \cdot \int F(x_i \mid \theta) G_0(d\theta) & \text{if } k \text{ is new}. \end{cases}$$

The first term is the prior probability of joining table $k$ (CRP), and the second term is the likelihood of the data under the cluster's parameters (marginalised over the prior). When $F$ and $G_0$ are conjugate, the integral has a closed form.

## 4.6 The Hierarchical Dirichlet Process

### 4.6.1 Motivation: Grouped Data

The DPMM assumes that all data points come from a single group. In FPL, however, data is **grouped by player**: each player $j$ has a sequence of gameweek performances $\{x_{j,1}, \ldots, x_{j,T_j}\}$. The objectives are to:

1. Discover a **global set of form profiles** shared across all players.
2. Allow each player to have their own **distribution over profiles** (some players spend more time in "Elite" states, others in "Average" states).

A single DPMM would pool all players' data together, ignoring player-specific variation. Separate DPMMs per player would fail to share statistical strength. The **Hierarchical Dirichlet Process** (Teh et al., 2005) resolves this tension.

### 4.6.2 The Chinese Restaurant Franchise

The HDP is best understood through the **Chinese Restaurant Franchise (CRF)** metaphor. Consider a franchise of restaurants (players), each with its own infinite tables (player-specific clusters). All restaurants share a **global menu of dishes** (latent form profiles).

The generative process is:

1. **Global level**: A global menu of dishes is drawn from a Dirichlet Process:
   $$G_0 \sim \text{DP}(\gamma, H),$$
   where $H$ is the base measure over dish parameters (form profile parameters), and $\gamma$ is the global concentration parameter.

2. **Restaurant level**: Each restaurant (player) $j$ has its own distribution over dishes, drawn from a Dirichlet Process with base measure $G_0$:
   $$G_j \sim \text{DP}(\alpha, G_0),$$
   where $\alpha$ is the player-specific concentration parameter.

3. **Customer level**: Each customer (gameweek performance) $i$ in restaurant $j$ chooses a dish (latent profile):
   $$\theta_{j,i} \sim G_j, \quad x_{j,i} \sim F(\theta_{j,i}).$$

Because $G_0$ is discrete, the $G_j$ share atoms (dishes). This means that while each player has their own distribution over profiles, the profiles themselves are shared globally.

### 4.6.3 Mathematical Formulation

The HDP is formally defined as:

$$\begin{aligned}
G_0 &\sim \text{DP}(\gamma, H), \\
G_j \mid G_0 &\sim \text{DP}(\alpha, G_0), \quad j = 1, 2, \ldots, J, \\
\theta_{j,i} \mid G_j &\sim G_j, \\
x_{j,i} \mid \theta_{j,i} &\sim F(\theta_{j,i}).
\end{aligned}$$

Here, $J$ is the number of players, and $i = 1, \ldots, T_j$ indexes gameweek observations for player $j$. This hierarchical construction — first introduced by Teh et al. (2005) — allows groups of data (each player's gameweek performances) to share statistical strength by sharing clusters, while still allowing for group-specific variations.

In the FPL application, the base measure $H$ is the **Normal-Inverse-Wishart (NIW)** distribution:

$$H = \text{NIW}(m_0, \kappa_0, \Psi_0, \nu_0),$$

so that each atom $\theta_k = (\mu_k, \Sigma_k)$ represents the mean vector and covariance matrix of a latent form profile. The NIW is the conjugate prior for the multivariate Gaussian, enabling efficient closed-form posterior updates as detailed in Chapter 5, Section 5.3. The analogy to document modelling, as noted by Teh et al. (2005), is instructive: players are "documents" and their gameweek performances are "words."

### 4.6.4 Stick-Breaking Representation

Using the stick-breaking construction, the HDP can be written explicitly. At the global level:

$$\begin{aligned}
\beta'_k &\stackrel{\text{i.i.d.}}{\sim} \text{Beta}(1, \gamma), \\
\beta_k &= \beta'_k \prod_{j=1}^{k-1} (1 - \beta'_j), \\
\theta_k &\stackrel{\text{i.i.d.}}{\sim} H, \\
G_0 &= \sum_{k=1}^\infty \beta_k \delta_{\theta_k}.
\end{aligned}$$

At the player level, for each player $j$:

$$\begin{aligned}
\pi'_{j,k} &\stackrel{\text{i.i.d.}}{\sim} \text{Beta}(\alpha \beta_k, \alpha (1 - \beta_k)), \\
\pi_{j,k} &= \pi'_{j,k} \prod_{l=1}^{k-1} (1 - \pi'_{j,l}), \\
G_j &= \sum_{k=1}^\infty \pi_{j,k} \delta_{\theta_k}.
\end{aligned}$$

The key insight is that the global weights $\beta_k$ influence the expected player-specific weights: $\mathbb{E}[\pi_{j,k}] = \beta_k$. This ensures that popular global profiles are also likely to appear in each player's mixture.

The observed data for player $j$ at gameweek $i$ is then:

$$\begin{aligned}
z_{j,i} \mid \boldsymbol{\pi}_j &\sim \text{Categorical}(\boldsymbol{\pi}_j), \\
x_{j,i} \mid z_{j,i} = k, \theta_k &\sim F(\theta_k).
\end{aligned}$$

### 4.6.5 Intuition for FPL

In the FPL context:

- **Dishes (global profiles)**: The $\theta_k$ represent distinct form profiles, such as "Elite Attacking Output," "Defensive Solidarity," "Creative Involvement," "Fatigued/Low Output." These are discovered from the data.
- **Restaurant menus (player-specific distributions)**: Each player $j$ has their own distribution $\boldsymbol{\pi}_j$ over these profiles. For example, Mohamed Salah's $\boldsymbol{\pi}_j$ may assign high probability to "Elite Attacking Output," while a backup goalkeeper's $\boldsymbol{\pi}_j$ may assign high probability to "Low Output, High Saves."
- **Customer assignments (gameweek states)**: Each gameweek observation $x_{j,i}$ is assigned to a profile $z_{j,i}$. The sequence of assignments over time reveals a player's form trajectory.

### 4.6.6 Posterior Distribution

The posterior distribution of interest is:

$$\begin{aligned}
&P(\{z_{j,i}\}, \{\theta_k\}, \boldsymbol{\beta}, \{\boldsymbol{\pi}_j\} \mid \{x_{j,i}\}) \\
&\quad \propto P(\{x_{j,i}\} \mid \{z_{j,i}\}, \{\theta_k\}) \cdot P(\{z_{j,i}\} \mid \{\boldsymbol{\pi}_j\}) \\
&\quad \cdot P(\{\boldsymbol{\pi}_j\} \mid \boldsymbol{\beta}, \alpha) \cdot P(\boldsymbol{\beta} \mid \gamma) \cdot P(\{\theta_k\} \mid H).
\end{aligned}$$

Posterior inference is intractable in closed form and requires approximate methods (MCMC or variational inference), which are covered in Chapter 7.

## 4.7 The Concentration Parameters

### 4.7.1 Global Concentration $\gamma$

The global concentration parameter $\gamma$ controls the number of global profiles (dishes). From Section 4.4.4, the expected number of global profiles is:

$$\mathbb{E}[K] \approx \gamma \log N,$$

where $N = \sum_j T_j$ is the total number of player-gameweek observations. A larger $\gamma$ permits more profiles, increasing model flexibility but also the risk of overfitting.

### 4.7.2 Player-Specific Concentration $\alpha$

The player-specific concentration parameter $\alpha$ controls how spread out each player's distribution over profiles is. A small $\alpha$ means each player concentrates on a few profiles (specialisation); a large $\alpha$ means each player samples more uniformly from the global menu (diversification).

In the stick-breaking construction, $\alpha$ influences the expected number of distinct profiles per player:

$$\mathbb{E}[K_j] \approx \alpha \log T_j.$$

### 4.7.3 Adaptive Concentration Parameters

Rather than fixing $\gamma$ and $\alpha$ as hyperparameters, hyperpriors are placed on them and they are inferred from the data:

$$\gamma \sim \text{Gamma}(a_\gamma, b_\gamma), \quad \alpha \sim \text{Gamma}(a_\alpha, b_\alpha).$$

The Adaptive SUGS (ASUGS) algorithm (Tsiligkaridis & Forsythe, 2015) provides a low-complexity method for online adaptation of $\gamma$ as new data arrives, motivated by large-sample asymptotics. This is particularly relevant for the FPL problem, where streaming data requires efficient online learning.

## 4.8 Relation to Finite Mixture Models

The HDP can be understood as the infinite limit of a finite hierarchical mixture model. Consider a finite mixture model with $K$ components:

$$\begin{aligned}
\boldsymbol{\beta} &\sim \text{Dir}(\gamma/K, \ldots, \gamma/K), \\
\boldsymbol{\pi}_j &\sim \text{Dir}(\alpha \beta_1, \ldots, \alpha \beta_K), \\
z_{j,i} &\sim \text{Categorical}(\boldsymbol{\pi}_j), \\
x_{j,i} \mid z_{j,i} = k &\sim F(\theta_k).
\end{aligned}$$

As $K \to \infty$, this converges to the HDP. This limit provides a constructive way to approximate the HDP by truncating at a large but finite $K$, which is the approach used in the NumPyro implementation.

## 4.9 Summary

This chapter has established the theoretical foundations of Bayesian nonparametrics required for modelling latent player form in FPL. The Chinese Restaurant Process provides an intuitive, constructive definition of the clustering distribution induced by a Dirichlet Process. The Dirichlet Process, defined via its stick-breaking construction, serves as a prior over distributions with countably infinite support, enabling mixture models with unbounded components. The Hierarchical Dirichlet Process extends this to grouped data, allowing players to share a global set of form profiles while retaining player-specific distributions.

The key properties — exchangeability, logarithmic cluster growth, rich-get-richer dynamics, and conjugacy — make the HDP ideally suited for discovering latent form states from FPL performance data. The next chapter applies this framework to the specific case of the HDP-GMM with a Gaussian-Wishart base measure, detailing the conjugate posterior updates and the modelling of multi-modal player form profiles.
