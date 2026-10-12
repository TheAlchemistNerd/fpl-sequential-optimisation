# Chapter 8: Constrained Optimisation — Knapsack, MILP, and Captaincy Selection

## 8.1 Introduction

The outputs of the decision methods developed in Chapters 9 and 10 — whether sampled point predictions (Thompson Sampling), Q-values (DDQN), or equilibrium strategies (MFG/MFC) — must be translated into **actionable FPL decisions**: which players to transfer in/out, which 11 to start, who to captain, and whether to deploy a chip. This translation is a **constrained combinatorial optimisation problem**.

The core problem is a **0/1 Knapsack Problem**: select a subset of items (players) with maximum total value (predicted points), subject to a weight constraint (budget). However, the FPL problem introduces additional constraints — positional requirements, per-team limits, formation validity, chip availability — that transform the basic knapsack into a **Mixed-Integer Linear Program (MILP)**.

Crucially, this MILP layer operates as a **regularizing post-process**. The DDQN and MCTS agents compute the optimal unconstrained or soft-constrained value of candidate transfers. The MILP solver then intercepts these suggested actions, enforcing the rigid real-world FPL rules to guarantee that the final executed squad is strictly valid. This separates the difficult problem of learning generalized long-horizon FPL values (handled by RL) from the rigid problem of rule enforcement (handled by MILP).

This chapter provides a comprehensive treatment of the optimisation methods used to solve the FPL squad selection problem, including Dynamic Programming for the basic knapsack, Branch-and-Bound for exact solutions, MILP formulations for the full constraint set, and Contextual Multi-Armed Bandits for the specialised captaincy sub-problem.

## 8.2 The 0/1 Knapsack Problem

### 8.2.1 Formal Definition

Given $N$ items (players), each with a value $v_i$ (predicted points) and a weight $w_i$ (price in £0.1m units), and a knapsack capacity $W$ (total budget in £0.1m units), the 0/1 Knapsack Problem is:

$$\max_{x \in \{0, 1\}^N} \; \sum_{i=1}^N v_i x_i \quad \text{subject to} \quad \sum_{i=1}^N w_i x_i \leq W.$$

In the FPL context, $x_i = 1$ means player $i$ is selected for the squad, and $x_i = 0$ means they are not.

### 8.2.2 Thompson Sampling Reduction

Under Thompson Sampling (Chapter 9), the values $v_i$ are the sampled points $\tilde{y}_i^{(s)}$ from the posterior predictive distribution. The weights $w_i$ are the player prices. The capacity $W$ is the total budget (1000, corresponding to £100.0m).

The reduction is exact: once the values are sampled, the stochastic FPL problem becomes a deterministic knapsack problem.

### 8.2.3 DDQN Reduction

Under DDQN (Chapter 10), the values $v_i$ are the Q-values $Q(s, a_i; \theta)$ for the action of selecting player $i$. The formulation is identical, but the values represent long-term expected rewards rather than single-gameweek predictions.

### 8.2.4 Why Thompson Sampling Reduces Complexity

The genius of Thompson Sampling lies in its elegant **separation of uncertainty from decision-making**:

- **Learning is probabilistic.** The core complexity of the problem is handled by the probabilistic model that defines the posterior distribution. This model continuously learns and updates its beliefs about the underlying reality as new data comes in. The HDP-GMM learns a player's latent form and influences the shape of their points posterior.

- **Decision is deterministic.** Once the sampling is done, the decision-making process is made trivial. The randomness and uncertainty are incorporated into the sampling step, *not* the optimisation step. The optimisation problem — maximising team points given a budget, formation, and per-team constraints — is solved exactly by the MILP layer (Section 8.6).

This contrasts sharply with DDQN, where the complexity is intertwined: a DDQN's neural network must learn a complex value function that *simultaneously* handles long-term value and uncertainty, which can be computationally intensive and difficult to interpret. In Thompson Sampling, the two concerns are cleanly decoupled.

## 8.3 Dynamic Programming Solution

### 8.3.1 Bellman Recursion

The Dynamic Programming (DP) approach solves the knapsack problem by building a table $V[i, w]$, representing the maximum value achievable using a subset of the first $i$ items with total weight at most $w$:

$$V[i, w] = \begin{cases} V[i-1, w] & \text{if } w_i > w, \\ \max\left( V[i-1, w], \; v_i + V[i-1, w - w_i] \right) & \text{if } w_i \leq w. \end{cases}$$

The base case is $V[0, w] = 0$ for all $w$.

### 8.3.2 Algorithm

```
function knapsack_DP(values, weights, W):
    N = len(values)
    V = zeros(N + 1, W + 1)

    for i = 1 to N:
        for w = 0 to W:
            if weights[i] > w:
                V[i, w] = V[i-1, w]
            else:
                V[i, w] = max(V[i-1, w], values[i] + V[i-1, w - weights[i]])

    # Reconstruct solution
    x = zeros(N)
    w = W
    for i = N down to 1:
        if V[i, w] != V[i-1, w]:
            x[i] = 1
            w -= weights[i]

    return V[N, W], x
```

### 8.3.3 Complexity

- **Time**: $O(N \cdot W)$, where $N$ is the number of items and $W$ is the capacity.
- **Space**: $O(N \cdot W)$ for the full table, or $O(W)$ with space optimisation (rolling array).

For FPL, $N \approx 600$ and $W \approx 1000$, yielding $600{,}000$ operations — solved in milliseconds.

### 8.3.4 Limitations

The DP approach handles only the basic budget constraint. It cannot naturally incorporate:
- Positional requirements (2 GKs, 5 DEFs, 5 MIDs, 3 FWDs).
- Per-team limits (max 3 players per club).
- Captaincy selection (double points for one player).
- Transfer costs (penalty for each transfer beyond the free allowance).

These require the MILP formulation.

## 8.4 Alternative Knapsack Methods

Before presenting the MILP formulation, additional knapsack solution methods are surveyed that are relevant in different operational regimes.

### 8.4.1 Brute-Force Enumeration

The brute-force method enumerates all $2^N$ possible subsets of players and selects the highest-value feasible subset. For $N \approx 600$, this is $2^{600} \approx 10^{181}$ — utterly infeasible. Brute-force is included only for completeness; it is never used in practice.

### 8.4.2 Greedy Approach

The greedy approach sorts players by value-per-weight ratio (predicted points per £0.1m) and selects them in descending order until the budget is exhausted.

**Complexity:** $O(N \log N)$ for sorting.

**Limitation:** The greedy approach does **not** guarantee optimality for the 0/1 knapsack problem. It cannot handle FPL-specific constraints (formation, per-team limits, captaincy). Despite its suboptimality, the greedy solution is useful as a **fast baseline** for comparison and as a **warm start** for the MILP solver.

### 8.4.3 Genetic Algorithms

Genetic Algorithms (GAs) are evolutionary heuristics useful for large knapsack instances where exact methods are too slow.

**Algorithm:**

1. **Initialisation:** Generate a population of $P$ random feasible squad selections satisfying budget and formation constraints.
2. **Evaluation:** Compute fitness as predicted total points for each squad.
3. **Selection:** Select the fittest squads for reproduction via tournament selection.
4. **Crossover:** Exchange player subsets between parent squads to produce offspring.
5. **Mutation:** Randomly swap players in/out with a small probability, respecting budget and formation constraints.
6. **Termination:** Repeat for $G$ generations. Return the best squad found.

**Advantages:**
- Naturally parallelisable (fitness evaluation is embarrassingly parallel).
- Can handle arbitrary additional constraints via penalty functions in the fitness evaluation.

**Limitations:**
- No optimality guarantee.
- May converge to local optima, especially with complex constraint landscapes.

GAs are most useful in the **initial squad selection** and **Wildcard optimisation** settings, where the action space is largest and a good approximate solution found quickly is more valuable than an exact solution found slowly.

## 8.5 Branch-and-Bound

### 8.5.1 Overview

Branch-and-Bound is a tree-search algorithm that systematically explores the space of feasible solutions, pruning branches that cannot lead to an optimal solution.

### 8.5.2 Algorithm

1. **Root node**: All variables are free. Compute an upper bound (e.g., the fractional knapsack solution).
2. **Branching**: Select a variable $x_i$ and create two child nodes: $x_i = 1$ and $x_i = 0$.
3. **Bounding**: For each node, compute an upper bound on the best possible solution in that subtree.
4. **Pruning**: If a node's upper bound is worse than the best known feasible solution, prune it.
5. **Termination**: When all nodes are pruned or solved, the best known solution is optimal.

### 8.5.3 Upper Bound: Fractional Knapsack

The upper bound for a node is computed by solving the **fractional knapsack relaxation**: allow $x_i \in [0, 1]$ (partial item selection). The optimal solution is greedy: sort items by value-to-weight ratio $v_i / w_i$ and fill the knapsack in descending order, taking the last item fractionally if needed.

```
function fractional_knapsack_upper_bound(values, weights, W, fixed_vars):
    # Fix variables as specified
    remaining_capacity = W - sum(weights[i] for i where x_i = 1)
    current_value = sum(values[i] for i where x_i = 1)

    # Sort remaining items by value/weight ratio
    remaining = [(v_i/w_i, v_i, w_i) for i not in fixed_vars]
    remaining.sort(reverse=True)

    for (ratio, v, w) in remaining:
        if w <= remaining_capacity:
            current_value += v
            remaining_capacity -= w
        else:
            current_value += ratio * remaining_capacity  # Fractional
            break

    return current_value
```

### 8.5.4 Complexity

In the worst case, Branch-and-Bound explores $O(2^N)$ nodes. However, with good bounding and pruning, the practical performance is often much better. For the FPL knapsack with $N = 600$, Branch-and-Bound typically solves in seconds.

## 8.6 Mixed-Integer Linear Programming (MILP)

### 8.6.1 Full MILP Formulation

The full FPL squad selection problem is formulated as a MILP with the following decision variables, objective, and constraints.

**Decision Variables:**

- $x_i \in \{0, 1\}$: 1 if player $i$ is in the squad, 0 otherwise.
- $s_i \in \{0, 1\}$: 1 if player $i$ is in the starting XI, 0 otherwise.
- $c_i \in \{0, 1\}$: 1 if player $i$ is captain, 0 otherwise.
- $v_i \in \{0, 1\}$: 1 if player $i$ is vice-captain, 0 otherwise.

**Objective:**

$$\max \; \sum_{i=1}^N v_i^{\text{pred}} \cdot s_i + v_{c}^{\text{pred}} \cdot c_c + v_{v}^{\text{pred}} \cdot c_v - 4 \cdot \max(0, T - f),$$

where $v_i^{\text{pred}}$ is the predicted points for player $i$, $c_c$ and $c_v$ are the captain and vice-captain indices, $T$ is the number of transfers, and $f$ is the number of free transfers.

**Constraints:**

1. **Squad size**: $\sum_{i=1}^N x_i = 15$.

2. **Positional requirements**:
   - $\sum_{i \in \text{GK}} x_i = 2$.
   - $\sum_{i \in \text{DEF}} x_i = 5$.
   - $\sum_{i \in \text{MID}} x_i = 5$.
   - $\sum_{i \in \text{FWD}} x_i = 3$.

3. **Budget**: $\sum_{i=1}^N \text{price}_i \cdot x_i \leq \text{budget}$.

4. **Per-team limit**: $\sum_{i \in \text{Club}(c)} x_i \leq 3$ for each club $c$.

5. **Starting XI size**: $\sum_{i=1}^N s_i = 11$.

6. **Starters are from squad**: $s_i \leq x_i$ for all $i$.

7. **Formation validity**:
   - $\sum_{i \in \text{GK}} s_i = 1$.
   - $\sum_{i \in \text{DEF}} s_i \geq 3$.
   - $\sum_{i \in \text{FWD}} s_i \geq 1$.

8. **Captaincy**: $\sum_{i=1}^N c_i = 1$, $c_i \leq s_i$ for all $i$.

9. **Vice-captaincy**: $\sum_{i=1}^N v_i = 1$, $v_i \leq s_i$ for all $i$, $v_i + c_i \leq 1$ for all $i$.

10. **Transfer constraint**: $\sum_{i=1}^N |x_i - x_i^{\text{prev}}| / 2 \leq f + T_{\text{paid}}$, where $x_i^{\text{prev}}$ is the previous squad.

### 8.6.2 Solver Configuration

The MILP is solved using a standard solver (e.g., CBC, Gurobi, SCIP). The formulation is passed to the solver as follows:

```
Algorithm: MILP Solver Configuration
Input: Player predictions v_pred[N], prices price[N], budget B,
       position masks GK[], DEF[], MID[], FWD[], club assignments club[N]
Output: Optimal squad x[15], starting XI s[11], captain c, vice-captain v

Stage 1: Define Decision Variables
    x_i ∈ {0, 1} for i = 1 to N          // squad selection
    s_i ∈ {0, 1} for i = 1 to N          // starting XI
    c_i ∈ {0, 1} for i = 1 to N          // captain
    v_i ∈ {0, 1} for i = 1 to N          // vice-captain

Stage 2: Define Objective
    Maximise: Σ_i v_pred[i] × s_i + v_pred[captain] × c_captain - transfer_cost

Stage 3: Define Constraints
    (1)  Σ_i x_i = 15                     // squad size
    (2)  Σ_{i ∈ GK} x_i = 2               // goalkeepers
    (3)  Σ_{i ∈ DEF} x_i = 5              // defenders
    (4)  Σ_{i ∈ MID} x_i = 5              // midfielders
    (5)  Σ_{i ∈ FWD} x_i = 3              // forwards
    (6)  Σ_i price[i] × x_i ≤ B            // budget
    (7)  Σ_{i ∈ Club(c)} x_i ≤ 3 for each club c  // per-team limit
    (8)  Σ_i s_i = 11                      // starting XI size
    (9)  s_i ≤ x_i for all i               // starters from squad
    (10) Σ_{i ∈ GK} s_i = 1                // GK in starting XI
    (11) Σ_{i ∈ DEF} s_i ≥ 3               // minimum DEF in starting XI
    (12) Σ_{i ∈ FWD} s_i ≥ 1               // minimum FWD in starting XI
    (13) Σ_i c_i = 1, c_i ≤ s_i            // exactly one captain from starters
    (14) Σ_i v_i = 1, v_i ≤ s_i, v_i + c_i ≤ 1  // vice-captain

Stage 4: Solve
    result ← solve(MILP, solver=CBC/Gurobi/SCIP)

Return result.x, result.s, result.c, result.v
```

### 8.6.3 Computational Performance

For $N \approx 600$ variables and $\approx 50$ constraints, modern MILP solvers find the optimal solution in **sub-second** time. This is more than fast enough for the weekly decision cycle.

## 8.7 Captaincy Selection as a Contextual Multi-Armed Bandit

### 8.7.1 Problem Formulation

The captaincy choice is a myopic decision (affects only the current gameweek's score), making it a natural candidate for the **Contextual Multi-Armed Bandit (MAB)** formulation.

- **Arms**: The 11 players in the starting lineup.
- **Context**: For each arm $a$, a feature vector $x_{t,a}$ containing:
  - The player's belief state $b_a(t)$ from the HDP-HMM.
  - Fixture difficulty (FDR) for the upcoming gameweek.
  - Home/away status.
  - Rolling form (3, 5, 10 game averages).
  - Opponent defensive strength.
  - Sentiment score.
- **Reward**: The points scored by the chosen captain (doubled).

### 8.7.2 Contextual Thompson Sampling

The Contextual Thompson Sampling algorithm (Agrawal & Goyal, 2013) maintains a posterior over the reward function parameters:

$$r_{t,a} = x_{t,a}^\top \theta_a + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2).$$

The algorithm is:

1. **Maintain posterior**: $\theta_a \sim \mathcal{N}(\hat{\mu}_a, \hat{\Sigma}_a)$ for each arm $a$.
2. **Sample**: $\tilde{\theta}_a \sim \mathcal{N}(\hat{\mu}_a, \hat{\Sigma}_a)$.
3. **Select**: $a_t = \arg\max_a x_{t,a}^\top \tilde{\theta}_a$.
4. **Observe reward** $r_t$ and update the posterior for the selected arm.

The posterior update for arm $a$ is:

$$\hat{\Sigma}_a = \left( \Sigma_0^{-1} + \frac{1}{\sigma^2} \sum_{t: a_t = a} x_{t,a} x_{t,a}^\top \right)^{-1},$$
$$\hat{\mu}_a = \hat{\Sigma}_a \left( \Sigma_0^{-1} \mu_0 + \frac{1}{\sigma^2} \sum_{t: a_t = a} x_{t,a} r_t \right).$$

### 8.7.3 LinUCB Alternative

The **LinUCB** (Linear Upper Confidence Bound) algorithm provides an alternative:

$$a_t = \arg\max_a \left( x_{t,a}^\top \hat{\theta}_a + \alpha \cdot \sqrt{x_{t,a}^\top \hat{\Sigma}_a^{-1} x_{t,a}} \right),$$

where $\alpha > 0$ controls the exploration bonus. The second term is the uncertainty (width of the confidence interval) for arm $a$.

### 8.7.4 Active Inference Approach

**Active Inference** (Friston et al.) provides a unified framework for exploration and exploitation via expected free energy minimisation:

$$a_t = \arg\min_a \mathbb{E}_{P(o \mid a)}\left[ \underbrace{-\log P(o = \text{high score} \mid a)}_{\text{Pragmatic value (exploitation)}} + \underbrace{\text{KL}(P(s \mid o, a) \parallel P(s \mid a))}_{\text{Epistemic value (exploration)}} \right].$$

The first term encourages actions that lead to high scores (exploitation). The second term encourages actions that reduce uncertainty about the state (exploration). This provides a principled unification of exploration and exploitation without ad-hoc parameters like $\epsilon$ or $\tau$.

## 8.8 GenTS-Explore for Initial Squad Selection

The GenTS-Explore method and its initial-squad/Wildcard application are developed in Chapter 9, Sections 9.6.1–9.6.3. Chapter 2, Section 2.5 distinguishes its combinatorial pure-exploration objective from the full stateful FPL task.

## 8.9 Handling Transfer Costs

### 8.9.1 Net Gain Formulation

When making transfers, the objective is not simply to maximise the predicted points of the new squad, but to maximise the **net gain** over the current squad, accounting for transfer costs:

$$\max \; \left( \sum_{i \in S_{\text{new}}} v_i - \sum_{i \in S_{\text{old}}} v_i \right) - 4 \cdot \max(0, |S_{\text{new}} \setminus S_{\text{old}}| - f).$$

This is equivalent to:

$$\max \; \sum_{i \in S_{\text{new}} \setminus S_{\text{old}}} v_i - \sum_{i \in S_{\text{old}} \setminus S_{\text{new}}} v_i - 4 \cdot \max(0, k - f),$$

where $k = |S_{\text{new}} \setminus S_{\text{old}}|$ is the number of transfers.

### 8.9.2 MILP Encoding

The transfer cost is a piecewise linear function, which can be encoded in the MILP using auxiliary variables.

To handle the non-linearity of the cost function (0 points for transfers $\leq f$, and 4 points per transfer thereafter), we introduce auxiliary binary variables. Specifically, for each player $i$ in the potential squad, we define a binary variable $t_i$ indicating whether the player's membership has changed:

$$t_i \geq x_i - x_i^{\text{prev}}, \quad t_i \geq x_i^{\text{prev}} - x_i, \quad t_i \in \{0, 1\},$$

where $x_i$ is the new squad selection variable and $x_i^{\text{prev}}$ is the indicator of the player's presence in the previous squad. Since $x_i, x_i^{\text{prev}} \in \{0, 1\}$, these constraints ensure that $t_i = |x_i - x_i^{\text{prev}}|$, i.e., $t_i = 1$ if and only if the player's squad membership has changed.

The total number of transfers is then $k = \frac{1}{2} \sum_{i} t_i$. The transfer cost is modelled as an auxiliary variable $C$ constrained by:

$$C \geq 0, \quad C \geq 4 \cdot (k - f).$$

The objective then becomes:

$$\max \; \sum_{i \in S_{\text{new}}} v_i \cdot s_i + v_c \cdot c_c - C.$$

This encoding correctly handles the discontinuity at $k = f$ (zero cost for transfers within the free allowance, 4 points per transfer beyond) and allows the solver to optimise the trade-off between immediate squad improvement and future point penalties.

### 8.10 Chip Deployment Optimisation

### 8.10.1 Chip Effects

Each chip modifies the scoring rules for one gameweek:

- **Wildcard**: Unlimited free transfers for one gameweek (resets the squad without cost).
- **Free Hit**: Replace the entire squad for one gameweek, reverting to the previous squad the next gameweek.
- **Bench Boost**: Points from all 4 bench players are added to the total.
- **Triple Captain**: The captain's points are tripled (instead of doubled).

### 8.10.2 Chip Timing as a Multi-Armed Bandit

The decision of when to deploy each chip is a **4-armed bandit problem**, where the arms are the remaining gameweeks. The reward for deploying a chip at gameweek $t$ is:

- **Wildcard**: The improvement in squad quality from unlimited transfers.
- **Free Hit**: The improvement from temporarily accessing premium players.
- **Bench Boost**: The sum of bench players' points.
- **Triple Captain**: The captain's additional points (beyond the standard doubling).

The optimal timing is found by evaluating the expected reward for each remaining gameweek and selecting the maximum.

### 8.10.3 Double-Gameweek Strategy

The most common strategy for Bench Boost and Triple Captain is to deploy them during a **double-gameweek** (when a team plays twice in one gameweek). The expected reward is approximately doubled:

$$\mathbb{E}[\text{Bench Boost}_{\text{double}}] \approx 2 \cdot \mathbb{E}[\text{Bench Boost}_{\text{single}}].$$

The MILP can be extended to include the chip decision:

$$\max \; \sum_{t=1}^{38} \gamma^{t-1} \left( R_t + \mathbb{I}(\chi_t = \text{BB}) \cdot R^{\text{bench}}_t + \mathbb{I}(\chi_t = \text{TC}) \cdot R^{\text{captain}}_t \right),$$

subject to the constraint that each chip is used at most once.

## 8.11 Production Implementation Notes

### 8.11.1 PuLP and CBC Solver

The MILP formulation (§8.6) is implemented using **PuLP**, a Python linear programming modelling library, with the **CBC (Coin-or Branch and Cut)** solver as the default backend. CBC is an open-source MILP solver included with PuLP, requiring no additional installation or licensing.

The solver is configured with a 30-second time limit, which is sufficient for the typical problem size ($N \approx 600$ players, $\approx 50$ constraints). If the solver does not find the optimal solution within the time limit, it returns the best feasible solution found so far. In practice, the MILP typically solves in 1–5 seconds for standard squad selection problems.

### 8.11.2 Auxiliary Variable Encoding for Transfer Costs

The piecewise linear transfer cost function (§8.9.2) is encoded using **auxiliary binary variables** $t_i$ for each player:

$$t_i \geq x_i - x_i^{\text{prev}}, \quad t_i \geq x_i^{\text{prev}} - x_i, \quad t_i \in \{0, 1\}.$$

Since $x_i, x_i^{\text{prev}} \in \{0, 1\}$, this encoding ensures $t_i = |x_i - x_i^{\text{prev}}|$, i.e., $t_i = 1$ if and only if the player's squad membership has changed. The total number of transfers is then $k = \frac{1}{2}\sum_i t_i$ (the factor of $\frac{1}{2}$ accounts for the fact that each transfer involves one player in and one player out, so $t_i = 1$ for both).

The production implementation (`src/optimization/selection/milp_solver.py`) solves the **net gain** objective:

$$\max \left(\sum_{j \in S_{\text{new}}} v_j \cdot s_j + v_c - \text{cost}\right),$$

where the cost term correctly handles the discontinuity at the free transfer allowance.

## 8.12 Summary

This chapter has provided a comprehensive treatment of the constrained optimisation layer in the FPL decision pipeline. The 0/1 Knapsack Problem captures the core budget-squad trade-off, solvable exactly via Dynamic Programming. The full FPL problem — with positional, per-team, formation, captaincy, and transfer cost constraints — is formulated as a MILP, solvable in sub-second time by modern solvers. The captaincy sub-problem is addressed via Contextual Thompson Sampling and LinUCB. GenTS-Explore for initial squad selection is discussed in Chapter 9, Section 9.6. Chip deployment is optimised via a multi-armed bandit over remaining gameweeks.

The next chapter develops Thompson Sampling for FPL decision-making, using the constrained optimisation methods introduced here.
