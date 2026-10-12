# Chapter 9: Thompson Sampling and Bayesian Decision Theory — Step A: Myopic Optimisation

## 9.1 Introduction

Thompson Sampling is a Bayesian approach to sequential decision-making that naturally balances exploration and exploitation by acting according to the posterior distribution over unknown parameters. Unlike value-function-based methods (e.g., DDQN, covered in Chapter 10), Thompson Sampling does not learn an explicit Q-value function. Instead, it maintains a posterior distribution over the quantities of interest (player point expectations), samples from this posterior at each decision point, and acts optimally with respect to the sampled values.

This chapter develops the Thompson Sampling framework for FPL decision-making, building on the HDP-GMM and HDP-HMM posterior inference methods from Chapters 5–7. The topics covered include the algorithm, the construction of player point posteriors from the HDP belief states, the reduction to a deterministic knapsack optimisation problem, the exploration-exploitation trade-off, the GenTS-Explore algorithm for combinatorial settings, the extension to multi-gameweek planning via Monte Carlo Tree Search (MCTS), and the theoretical Bayes regret bounds.

### 9.1.1 Thompson Sampling as Step A in the Planning Hierarchy

It is important to understand Thompson Sampling's role in the overall decision architecture. TS constitutes the **first step (Step A)** in a three-stage planning hierarchy:

- **Step A (Sections 9.2–9.7)**: Thompson Sampling optimises **for the next gameweek**. It samples from the posterior belief states produced by Chapter 7's MCMC and solves a deterministic knapsack problem. It is myopic — it does not plan across multiple gameweeks — but is interpretable and can handle FPL constraints via MILP (Chapter 8).
- **Step B (Section 9.8)**: Thompson Sampling coupled with Monte Carlo Tree Search (TS+MCTS) extends the horizon by simulating explicit, multi-gameweek transfer sequences. This is used for chip timing, transfer saving, and double-gameweek preparation.
- **Step C (Chapter 10)**: When tree-search becomes computationally explosive (e.g., over a 38-game horizon), Double Deep Q-Networks (DDQN) approximate long-term FPL values using neural networks, learning a generalised value function in $O(1)$ forward pass time.

Thompson Sampling is the foundation upon which the more sophisticated planning methods are built. Understanding its mechanics is essential for appreciating the extensions in Chapter 10.

## 9.2 The Thompson Sampling Algorithm

### 9.2.1 The Multi-Armed Bandit Problem

The Thompson Sampling algorithm was originally developed for the multi-armed bandit (MAB) problem. In the MAB, an agent faces $K$ arms, each with an unknown reward distribution. At each round $t$, the agent chooses an arm $a_t$, observes a reward $r_t \sim P(\cdot \mid a_t)$, and seeks to maximise cumulative reward over $T$ rounds.

The performance metric is **regret**:

$$\text{Regret}(T) = \sum_{t=1}^T \left( \mu^* - \mu_{a_t} \right),$$

where $\mu^* = \max_k \mu_k$ is the expected reward of the best arm, and $\mu_{a_t}$ is the expected reward of the chosen arm.

### 9.2.2 Thompson Sampling for MAB

The Thompson Sampling algorithm for MAB is:

1. **Maintain a posterior** $P(\theta_k \mid \mathcal{D}_t)$ over the parameters of each arm $k$, where $\mathcal{D}_t$ is the history of actions and rewards up to round $t$.
2. **Sample** $\tilde{\theta}_k \sim P(\theta_k \mid \mathcal{D}_t)$ for each arm $k$.
3. **Act optimally**: choose $a_t = \arg\max_k \mathbb{E}[r \mid \tilde{\theta}_k]$.
4. **Observe reward** $r_t$ and update the posterior.

The key insight is that the sampling step introduces exploration naturally: arms with high uncertainty (wide posterior) will occasionally be sampled high, leading to their selection. This is more efficient than $\epsilon$-greedy exploration, which explores uniformly at random.

### 9.2.3 The Relationship Between Thompson Sampling, HDP-GMM, and MCMC

An important conceptual point deserves emphasis: **Thompson Sampling does not inherently require a specific sampling method** like Gibbs sampling or NUTS. The key requirement is to draw a sample from the posterior distribution of the model's parameters. However, in complex Bayesian models like the HDP-GMM, directly sampling from the posterior is computationally intractable. This is where MCMC methods become essential.

The relationship can be understood through an analogy:

- **Thompson Sampling is the strategy** for picking a winning lottery ticket — the strategy is to "buy a ticket that's been drawn from the box of all possible winning tickets."
- **The HDP-GMM is the box** of all possible winning tickets — it is the full distribution of what could be true about the players, encoding all plausible form profiles and their probabilities.
- **Gibbs Sampling (or NUTS) is the mechanical arm** that reaches into that incredibly complex box and draws out one single, plausible "winning ticket" (a sample from the posterior). Without that arm, you cannot actually perform the Thompson Sampling strategy on a complex model.

Gibbs Sampling works by iteratively sampling each parameter from its conditional distribution given the current values of all other parameters. This process generates a sequence of samples that, after a burn-in period, can be considered as draws from the true joint posterior distribution. For Thompson Sampling, **one** of those posterior samples is selected at random to represent the "simulated world" for the upcoming gameweek.

Posterior sampling from a model's parameters to make decisions is the very essence of Thompson Sampling. The HDP-GMM's posterior distribution over the player's latent form is a perfect fit for this. The MCMC sampler — whether Gibbs or NUTS — is simply the computational mechanism that makes the posterior sampling feasible.

### 9.2.4 Thompson Sampling for FPL

For FPL, the "arms" are the set of all possible actions (transfers, captaincy choices, chip deployments). However, the action space is combinatorially large, making direct Thompson Sampling intractable. The problem is decomposed as follows:

1. **Maintain a posterior** over each player's expected points for the upcoming gameweek.
2. **Sample** one set of predicted points for all players.
3. **Solve the optimisation problem**: find the best team (transfers, lineup, captain) given the sampled points.
4. **Observe real points** and update the posterior.

The decomposition transforms the combinatorial Thompson Sampling into a sequence of per-player posterior sampling followed by a deterministic optimisation step.

## 9.3 Posterior Construction from HDP-GMM Belief States

### 9.3.1 From Belief States to Point Predictions

Chapter 7 (Section 7.9) established that the MCMC posterior samples over $(\mu_k, \Sigma_k, \pi_k)$ define a belief state $b_j(t) = (b_j^{(1)}(t), \ldots, b_j^{(K)}(t))$ for each player $j$ at gameweek $t$, where $b_j^{(k)}(t) = P(z_{j,t} = k \mid x_{j,1:t})$. This belief state is the mathematical input to the Thompson Sampling decision process.

Given the belief state and the emission parameters $(\mu_k, \Sigma_k)$ for each cluster, the **predictive distribution** for player $j$'s feature vector at gameweek $t+1$ is:

$$p(x_{j,t+1} \mid x_{j,1:t}) = \sum_{k=1}^K b_j^{(k)}(t) \cdot \mathcal{N}(x_{j,t+1} \mid \mu_k, \Sigma_k).$$

This is a Gaussian mixture distribution with $K$ components.

### 9.3.2 The Belief-State-to-Reward Pipeline

The full transformation from MCMC posterior samples to Thompson Sampling decisions proceeds through the following pipeline:

$$\underbrace{\{\mu_k^{(s)}, \Sigma_k^{(s)}, \pi_k^{(s)}\}_{s=1}^S}_{\text{MCMC samples (Ch. 7)}}
\;\xrightarrow{\text{Bayes rule}}\;
\underbrace{\{b_j^{(s)}(t)\}_{s=1}^S}_{\text{Belief-state posterior}}
\;\xrightarrow{w^\top \mathbb{E}[\cdot]}_{\text{Feature-to-points}}
\;\underbrace{\{\tilde{\mu}_j^{(s)}\}_{s=1}^S}_{\text{Points posterior}}
\;\xrightarrow{\text{sample}}\;
\underbrace{\tilde{y}_j}_{\text{Thompson sample}}
\;\xrightarrow{\text{knapsack}}\;
\underbrace{a^*}_{\text{Optimal action}}.$$

Each MCMC sample $s$ induces a specific belief state $b_j^{(s)}(t)$, which in turn induces a specific expected point prediction $\tilde{\mu}_j^{(s)}$. The collection $\{\tilde{\mu}_j^{(s)}\}_{s=1}^S$ forms the posterior over expected points. Drawing one sample from this posterior yields the "simulated world" in which the optimal action is computed.

### 9.3.3 Mapping Features to Points

To convert the feature vector prediction into a point prediction, a mapping function $f: \mathbb{R}^D \to \mathbb{R}$ is learned that maps features to expected FPL points. This can be a simple linear model:

$$\mathbb{E}[\text{points}_{j,t+1}] = w^\top \mathbb{E}[x_{j,t+1}] = w^\top \sum_{k=1}^K b_j^{(k)}(t) \mu_k,$$

or a more complex model (e.g., Gradient Boosting, Neural Network) trained on historical data.

The **posterior over expected points** for player $j$ is induced by the posterior over $(\mu_k, \Sigma_k)$ and $b_j^{(k)}(t)$. For each MCMC sample $s$:

$$\tilde{\mu}_j^{(s)} = w^\top \sum_{k=1}^K b_j^{(k)}(t) \mu_k^{(s)}.$$

The collection $\{\tilde{\mu}_j^{(s)}\}_{s=1}^S$ forms the posterior samples of expected points.

### 9.3.4 Points Distribution

The distribution of actual FPL points $y_{j,t+1}$ (which are discrete and over-dispersed) is modelled as a Negative Binomial distribution parameterised by the expected points:

$$y_{j,t+1} \mid \tilde{\mu}_j^{(s)} \sim \text{Negative-Binomial}(r_j, p_j^{(s)}),$$

where $p_j^{(s)}$ is chosen such that $\mathbb{E}[y_{j,t+1}] = \tilde{\mu}_j^{(s)}$. The dispersion parameter $r_j$ is learned from historical data.

The **full posterior predictive distribution** for player $j$'s points is:

$$p(y_{j,t+1} \mid \text{data}) = \frac{1}{S} \sum_{s=1}^S \text{Negative-Binomial}(y_{j,t+1} \mid r_j, p_j^{(s)}).$$

### 9.3.5 Posterior Sampling

For Thompson Sampling, one sample is drawn from the posterior predictive distribution for each player:

$$\tilde{y}_j^{(s)} \sim p(y_{j,t+1} \mid \text{data}).$$

This yields a vector of sampled points $\tilde{\mathbf{y}}^{(s)} = (\tilde{y}_1^{(s)}, \ldots, \tilde{y}_N^{(s)})$ for all $N$ players in the FPL registry.

## 9.4 The "Simulated World" Reduction

### 9.4.1 From Stochastic to Deterministic

The key insight of Thompson Sampling is that once a set of predicted points $\tilde{\mathbf{y}}^{(s)}$ has been sampled, the decision problem becomes **deterministic**:

$$\max_{a \in \mathcal{A}} \; \sum_{j \in \text{team}(a)} \tilde{y}_j^{(s)} + 2 \cdot \tilde{y}_{\text{captain}(a)}^{(s)} - \text{cost}(a),$$

subject to the FPL constraints (budget, formation, per-team limit, chip availability). This is a **0/1 Knapsack Problem with side constraints**, whose full MILP formulation is developed in Chapter 8, Section 8.6; the basic knapsack and its solution methods are introduced in Section 8.2.

### 9.4.2 Algorithm Summary

The full Thompson Sampling algorithm for FPL is:

```
For each gameweek t = 1, ..., 38:
    1. Update HDP-HMM with latest data → belief states b_j(t)
    2. For each player j:
        a. Compute posterior over expected points: {μ̃_j^(s)}_{s=1}^S
        b. Sample one expected point: μ̃_j ~ Posterior
        c. Sample one realised point: ỹ_j ~ NB(r_j, p_j(μ̃_j))
    3. Solve knapsack: max Σ_{j∈team} ỹ_j + 2·ỹ_captain - cost
       subject to budget, formation, 3-per-team constraints
    4. Execute optimal action (transfers, lineup, captain)
    5. Observe real points → update model
```

### 9.4.3 Computational Complexity

The complexity per gameweek is:

- HDP-HMM update: $O(J \cdot T \cdot K^2)$ for $J$ players, $T$ gameweeks, $K$ clusters.
- Posterior sampling: $O(N)$ for $N$ players (one sample each).
- Knapsack solving: $O(N \cdot W)$ for Dynamic Programming (budget $W$), or exponential in the worst case for MILP.
- Total: dominated by HDP-HMM update.

For $N \approx 600$ and $W \approx 1000$ (in £0.1m units), Section 12.9 is planned to report measured knapsack and HDP-HMM runtimes. The HDP-HMM update may be the bottleneck, which is why MCMC acceleration (Chapter 7) is considered; the runtime has not yet been established.

## 9.5 Exploration-Exploitation Trade-off

### 9.5.1 Natural Exploration

Thompson Sampling's exploration is a natural consequence of posterior uncertainty. A player with a wide posterior (high variance) has a non-negligible probability of being sampled high, leading to their selection. This is fundamentally different from $\epsilon$-greedy exploration, which explores uniformly at random regardless of uncertainty.

**Example**: A new signing has limited historical data, so their points posterior has high variance. Thompson Sampling will occasionally sample a high value for this player, leading to their selection (exploration). As more data is collected, the posterior narrows, and the player is selected only if their mean is genuinely high (exploitation).

### 9.5.2 Quantifying Exploration

The expected number of times a suboptimal arm (player) is selected under Thompson Sampling is bounded. For the standard MAB with Gaussian rewards, the regret bound is:

$$\mathbb{E}[\text{Regret}(T)] \leq O\left( \sum_{j: \Delta_j > 0} \frac{\log T}{\Delta_j} \right),$$

where $\Delta_j = \mu^* - \mu_j$ is the suboptimality gap. This is asymptotically optimal (matching the Lai-Robbins lower bound).

### 9.5.3 Temperature Scaling

In practice, the posterior may be overconfident (narrower than the true uncertainty), leading to insufficient exploration. A **temperature parameter** $\tau > 0$ can be introduced to inflate the posterior variance:

$$\tilde{y}_j \sim \text{Negative-Binomial}(r_j, p_j(\tilde{\mu}_j / \tau)).$$

When $\tau > 1$, the posterior is flattened, increasing exploration. When $\tau < 1$, it is sharpened, increasing exploitation. The temperature can be annealed over the season:

$$\tau_t = \tau_0 \cdot \left( 1 - \frac{t}{38} \right) + \tau_{\min} \cdot \frac{t}{38}.$$

### 9.5.4 Comparison with Boltzmann and $\epsilon$-Greedy Exploration

Thompson Sampling's posterior-driven exploration stands in sharp contrast to two commonly used alternatives:

- **$\epsilon$-greedy**: Explores uniformly at random with probability $\epsilon$. This is a purely arbitrary heuristic — it does not adapt to uncertainty structure and may explore clearly suboptimal actions (e.g., transferring in a low-quality player) with the same probability as exploring promising but uncertain alternatives.
- **Boltzmann (softmax) exploration** (Chapter 10, Section 10.8): Explores according to a softmax distribution over Q-values, derived from the Maximum Entropy RL objective. This is more principled than $\epsilon$-greedy, as it favours high-value actions while maintaining entropy-based hedging.

Thompson Sampling differs from both in that exploration is **driven by posterior uncertainty**: actions are selected proportional to their probability of being optimal, not proportional to their estimated value or an arbitrary random chance. This makes TS particularly well-suited for FPL, where some players (e.g., new signings, returning-from-injury) genuinely have wide posteriors, while others (established premium assets) have narrow posteriors. TS explores the former more than the latter, while $\epsilon$-greedy treats both identically.

Chapter 12, Ablation 8 (Section 12.5.8) is planned to compare TS posterior sampling, Boltzmann, and $\epsilon$-greedy using cumulative points, exploration diversity (unique players selected), risk-adjusted returns, and rank.

## 9.6 GenTS-Explore for Combinatorial Settings

### 9.6.1 The Challenge

Standard Thompson Sampling is designed for the single-arm selection problem. In FPL, the action is a **combinatorial structure** (a team of 15 players with constraints). The interaction between players (budget trade-offs, positional requirements) means that the optimal team is not simply the 15 players with the highest sampled points.

### 9.6.2 GenTS-Explore Algorithm

Nakamura and Sugiyama (2024) introduce GenTS-Explore for real-valued combinatorial pure exploration. Their method uses an optimization oracle to find an action for a sampled reward vector, avoiding enumeration of the full action set; its objective is identifying the best combinatorial action through exploration. As the literature review explains (Chapter 2, Section 2.5), this is a relevant comparison for FPL, not a drop-in solution to its stateful, multi-gameweek objective.

For the FPL adaptation, maintain posteriors over player points. With independent marginals, sample $\tilde{\mu}_i \sim P(\mu_i \mid \mathcal{D})$ for each player; where dependence is represented, sample the joint vector $\tilde{\mathbf{y}} \sim P(\mathbf{y} \mid \mathcal{D})$. Then solve $\max_{S \in \mathcal{F}} \sum_{i \in S} \tilde{y}_i$, where $\mathcal{F}$ encodes the budget, formation, and per-team constraints; execute the selected team, observe rewards, and update the posterior. This retains the player-wise and joint-posterior forms stated in the earlier draft without treating them as separate algorithms.

<!-- TODO: Verify the two distinct regret expressions in the earlier drafts, O(N log T / Δ_min^2) with N players and a minimum expected-points gap, and O(m log T / Δ_min) with m semi-bandit arms, against their primary sources and assumptions before including a bound. Chapter 2 identifies problem-dependent sample-complexity results for Nakamura and Sugiyama (2024), but does not support either expression as a regret bound. -->

### 9.6.3 Application to Wildcard Optimisation

The initial squad selection (or Wildcard deployment) is a pure combinatorial problem: choose 15 players from ~600 to maximise expected cumulative points over the remaining season. The proposed adaptation uses the FPL feasible set for this budget- and formation-constrained choice; its stateful, season-long performance remains to be evaluated.

## 9.7 Bayes Regret Bounds

### 9.7.1 Thompson Sampling with Mixture Prior

Hong et al. (2022) studied Thompson Sampling with a mixture prior (MixTS), showing that the Bayes regret depends on the number of mixture components and their widths. For the HDP-GMM prior, the mixture components are the latent form profiles discovered by the hierarchical clustering process, and the regret bound is:

$$\mathbb{E}[\text{Regret}(T)] \leq O\left( K \cdot \sqrt{T \log T} \right),$$

where $K$ is the number of form profiles. This is sublinear in $T$, ensuring that the average per-round regret vanishes as $T \to \infty$.

### 9.7.2 Implications for FPL

For $K \approx 10$ form profiles and $T = 38$ gameweeks, the regret expression evaluates asymptotically to approximately $O(10 \cdot \sqrt{38 \log 38}) \approx O(60)$. This is not a finite-sample points guarantee for FPL; applying the bound to this setting would require establishing its assumptions for the constrained, sequential decision problem. The comparison with global league ranks also requires season-specific results.



## 9.8 Thompson Sampling and Monte Carlo Tree Search (MCTS) — Step B: Explicit Lookahead

While pure Thompson Sampling (Step A, Sections 9.2–9.7) provides an elegant Bayesian approach to single-gameweek optimisation, it is inherently **myopic**: it optimises for the immediate gameweek without considering the long-term consequences of decisions. In FPL, however, many decisions have intertemporal effects — transfer timing, chip deployment, price changes, and budget management all require **long-horizon planning**.

Coupling Thompson Sampling with **Monte Carlo Tree Search (MCTS)** creates **Step B** in the planning hierarchy. This approach, conceptually aligned with algorithms like D2NG-POMCP (Partially Observable Monte Carlo Planning), enables explicit multi-gameweek scenario planning by simulating transfer sequences over a finite horizon and evaluating their cumulative consequences.

### 9.8.1 Belief-State Pruning

A critical challenge in applying MCTS to FPL is the combinatorially large action space (~8,775 possible single transfers). To make the search tractable, the implementation employs **Belief-State Pruning**:

1.  **Candidate Selection**: Instead of considering all players, the action space is restricted to the **top-K candidates** (e.g., $K=100$) ranked by expected points derived from their belief states.
2.  **Action Encoding**: The MCTS tree expands only these candidate transfer actions. This reduces the branching factor $b$ from $\approx 720 \times 15$ to $\approx 100 \times 15$, making the $O(b^d)$ search feasible for depths $d \in [3, 5]$.
3.  **Transfer Cost Penalisation**: The rollout rewards are penalised by the actual transfer costs (4 points per hit), ensuring the search prefers minimal changes unless the expected gain is substantial.

### 9.8.2 The TS+MCTS Algorithm

The TS+MCTS algorithm proceeds in four stages:

**Stage 1 — Sample a World:** One sample is drawn from the HDP-HMM posterior: $\tilde{b}_j \sim \{b_j^{(s)}(t)\}_{s=1}^S$ for each player $j$. This fixes the latent form state distribution for all players in the sampled world. The sampled world is a deterministic scenario: player $j$ is believed to be in form cluster $k$ with probability $\tilde{b}_j^{(k)}$.

**Stage 2 — Build the Search Tree:** An MCTS tree is constructed where:
- **Nodes** represent $(b_t, S_t)$ pairs: the belief state over player forms and the current squad configuration.
- **Edges** represent transfer actions $a_t \in \mathcal{A}$ (swap player $i$ out, player $j$ in; or "no transfer").
- **Rollouts** simulate $T$ gameweeks forward from each node. At each simulated step $t' \in \{t, \ldots, t+T-1\}$:
  1. Player performances are sampled from the HDP-HMM emission distributions: $\tilde{x}_{j,t'+1} \sim \mathcal{N}(\mu_{\tilde{z}_{j,t'}}, \Sigma_{\tilde{z}_{j,t'}})$.
  2. Features are mapped to points: $\tilde{y}_{j,t'+1} = w^\top \tilde{x}_{j,t'+1}$.
  3. The squad's total points minus transfer costs are computed: $r_{t'} = \sum_{j \in S_{t'}} \tilde{y}_{j,t'+1} - \lambda \cdot c(a_{t'})$.
  4. Belief states are updated via the HDP-HMM transition: $\tilde{b}_{j,t'+1} \sim \pi_{\tilde{z}_{j,t'}} \cdot \text{BayesUpdate}(\tilde{x}_{j,t'+1})$.

**Stage 3 — Tree Expansion and Backpropagation:** The UCB1 (Upper Confidence Bound) selection rule is applied at each node:

$$\text{UCB1}(s, a) = Q(s, a) + c \cdot \sqrt{\frac{\log N(s)}{N(s, a)}},$$

where $Q(s, a)$ is the average rollout reward from action $a$, $N(s)$ is the visit count of state $s$, $N(s, a)$ is the visit count of action $a$, and $c$ controls the exploration bonus. After each rollout, the cumulative discounted reward $\sum_{t'=t}^{t+T-1} \gamma^{t'-t} r_{t'}$ is backpropagated up the tree.

**Stage 4 — Act:** After $N_{\text{sim}}$ simulations, the action with the highest visit count at the root node is selected: $a^* = \arg\max_a N(s_{\text{root}}, a)$.

### 9.8.3 When to Use TS+MCTS

TS+MCTS is particularly valuable for **high-stakes tactical decisions** where the consequences unfold over multiple gameweeks:

- **Chip timing**: Deciding when to deploy the Wildcard, Bench Boost, or Triple Captain. The MCTS tree explicitly simulates fixture difficulty, double-gameweeks, and price changes across the planning horizon, identifying the deployment window that maximises cumulative returns.
- **Transfer saving**: Deciding whether to take a -4 point hit now for an immediate upgrade, or save the transfer for a more favourable fixture next week. MCTS evaluates both sequences and selects the higher-cumulative-reward path.
- **Double-gameweek preparation**: Building a squad with players who have two fixtures in a single gameweek requires planning 2–3 weeks ahead to ensure budget availability and optimal squad structure.

For routine single-gameweek captaincy and transfer decisions, pure Thompson Sampling (Step A) is sufficient and computationally cheaper. For full-season squad building, DDQN (Step C, Chapter 10) is more appropriate due to its $O(1)$ inference time.

### 9.8.4 Computational Complexity and the Motivation for DDQN

The computational complexity of MCTS is $O(b^d)$, where $b$ is the branching factor (number of candidate transfer actions) and $d$ is the planning depth (number of gameweeks). For FPL:

- $b \approx 100$ (top-ranked candidates by expected points).
- $d = 5$ for tactical chip-timing decisions → $100^5 = 10^{10}$ leaf nodes, tractable with pruning and UCB-guided search (~$10^4$–$10^5$ simulations).
- $d = 38$ for full-season planning → $100^{38} = 10^{76}$ leaf nodes, computationally intractable.

This exponential scaling is the fundamental motivation for **Step C (DDQN)**: the neural network in Chapter 10 learns a generalised value function that approximates the optimal long-horizon policy without explicit tree expansion. DDQN trades off the interpretability and exactness of MCTS for scalability to the full 38-game season.

## 9.9 Production Implementation Notes

### 9.9.1 Greedy Knapsack vs. MILP Optimiser

The Thompson Sampling selection algorithm (§9.4) solves a deterministic knapsack problem on the sampled point predictions. The production implementation provides **two tiers** of knapsack solving:

1. **Greedy budget-feasible selection** (`thompson_sampling_select` in `src/optimization/selection/thompson_sampling.py`): Players are sorted by sampled points in descending order and selected greedily until the budget is exhausted. This is fast ($O(N \log N)$) and produces a feasible solution, but may be suboptimal when budget constraints create non-trivial trade-offs (e.g., one premium player vs. three mid-priced players).

2. **MILP exact solution** (`solve_squad_optimisation` in `src/optimization/selection/milp_solver.py`): The full constrained optimisation problem is formulated using PuLP with the CBC solver, enforcing all FPL rules (formation 2-5-5-3, per-team limit $\leq 3$, starting XI validity, captaincy). The configured solve limit is 30 seconds; observed runtime and solution quality are planned for evaluation in Chapter 12, Section 12.9.

The greedy tier is used for **rapid prototyping and single-gameweek decisions** where the budget slack is typically large enough that the greedy solution is near-optimal. The MILP tier is used for **production decisions** and **transfer-aware optimisation** (where the piecewise linear transfer cost encoding requires exact constraint handling).

### 9.9.2 Transfer Cost Encoding

The MILP solver implements the piecewise linear transfer cost function (§8.9.2) using **auxiliary binary variables**. For each player $i$, a binary variable $t_i$ indicates whether the player's squad membership has changed:

$$t_i \geq x_i - x_i^{\text{prev}}, \quad t_i \geq x_i^{\text{prev}} - x_i,$$

where $x_i$ is the new squad selection and $x_i^{\text{prev}}$ is the previous squad membership. The total number of transfers is $k = \frac{1}{2}\sum_i t_i$, and the transfer cost is encoded as:

$$\text{cost} \geq 0, \quad \text{cost} \geq 4 \cdot (k - f),$$

where $f$ is the number of free transfers available. This encoding correctly handles the discontinuity at $k = f$ (zero cost for transfers within the free allowance, 4 points per transfer beyond).

### 9.9.3 Temperature Annealing Schedule

The temperature parameter $\tau$ (Section 9.5.3) controls the exploration-exploitation trade-off by scaling the posterior variance. The production implementation uses a **linear annealing schedule**:

$$\tau_t = \tau_0 \cdot \left(1 - \frac{t}{T}\right) + \tau_{\min} \cdot \frac{t}{T},$$

with default values $\tau_0 = 2.0$ and $\tau_{\min} = 0.1$. Early gameweeks ($t \ll T$) use high temperature to encourage exploration of uncertain players (new signings, returning from injury), while late gameweeks ($t \approx T$) use low temperature to exploit well-established form assessments.

### 9.9.4 GenTS-Explore Position-Aware Selection

The GenTS-Explore algorithm (§9.6) for combinatorial squad selection implements **position-aware greedy**: players are grouped by position (GK, DEF, MID, FWD) and the required formation (2-5-5-3) is satisfied within each group before moving to the next. This ensures that the budget allocation across positions respects the FPL rules, rather than selecting the 15 highest-scoring players regardless of position (which would often produce invalid squads).

## 9.10 Summary

This chapter has developed the first two stages of the FPL planning hierarchy: **Step A (Thompson Sampling)** and **Step B (TS+MCTS)**. Grounded in the HDP-GMM/HDP-HMM belief states from Chapter 7, Thompson Sampling provides a principled Bayesian framework for single-gameweek decisions, balancing exploration and exploitation through posterior sampling. By coupling TS with Monte Carlo Tree Search, the framework extends to medium-horizon tactical decisions, enabling explicit lookahead for chip deployment and transfer timing. Together, these methods form a comprehensive toolkit for Bayesian decision-making in the highly stochastic FPL environment, with Step A handling immediate squad selection and Step B addressing medium-term scenario planning.

However, the exponential scaling of MCTS limits its utility for full-season strategic planning. The next chapter introduces **Step C (DDQN)**, which employs neural function approximation to learn long-horizon values, providing a scalable solution for the entire 38-game season.
