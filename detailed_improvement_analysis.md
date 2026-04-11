# Improvement Analysis: FPL Sequential Optimisation Implementation

## 1. Introduction
This report provides a detailed critique and roadmap for improving the [Implementation Plan](file:///c:/Users/Nevo/Downloads/Predicting%20FPL%20using%20Sequential%20Optimisation/IMPLEMENTATION_PLAN.md) and associated task lists. Having completed a thorough audit of Chapters 4 through 10 of the dissertation documentation, it is evident that several critical theoretical nuances described in the methodology are either missing or under-specified in the current implementation roadmap.

The following analysis categorises improvements by theoretical alignment, computational efficiency, and strategic execution.

---

## 2. Theoretical Alignment: Bridging Methodology and Code

### 2.1 Implementing I²GMM for Multi-Modal Form Profiles
**Theoretical Context (Ch. 5 §5.5):** The HDP-GMM as currently planned assumes a single Gaussian per cluster. However, FPL performance data often exhibits within-cluster multi-modality (e.g., "Elite at Home" vs "Elite Away"). 
**Improvement:** The implementation plan should explicitly include the **Infinite Mixture of Infinite Gaussian Mixtures (I²GMM)** extension. 
- **Task Addition:** "Extend code in `src/models/hdp_gmm_gibbs.py` to support two-level hierarchical cluster assignments via nested CRPs."
- **Rationale:** Without this, the model will likely over-fragment interpretable form states into multiple small, hard-to-distinguish clusters, reducing the interpretability of belief states.

### 2.2 Action-Conditioned Transition Dynamics
**Theoretical Context (Ch. 6 §6.4):** The documentation defines the HDP-HMM transitions as action-conditioned: $P(z_{t+1} | z_t, a_t)$. 
**Improvement:** The current `data_pipeline.py` and model stubs assume static transitions.
- **Task Addition:** "Update `run_hdp_hmm` to accept action vectors $a_t$ and learn transition matrices $\pi_k(a)$."
- **Rationale:** This is the core of the "Belief-State MDP". If transitions are not action-conditioned, the agent cannot learn the effect of its own decisions (e.g., transfer hits or benching) on future form trajectories.

### 2.3 Mean-Field Price Coupling
**Theoretical Context (Ch. 9 §9.10):** Price changes are driven by the aggregate actions of the population (the mean-field).
- **Improvement:** The implementation plan for DDQN and MCTS should include a **Mean-Field price predictor**.
- **Task Addition:** "Develop a population-level price change model driven by the aggregate transfer intensity $m_t$."
- **Rationale:** Without this, the agent is "blind" to price rises, missing the strategic "buy-early" arbitrage opportunities central to high-level FPL play.

---

## 3. Computational Strategy: Efficiency and Convergence

### 3.1 JAX/XLA JIT-Accelerated Collapsed Gibbs & FFBS
**Theoretical Context (Ch. 7 §7.5):** MCMC in Python is prohibitively slow for the $N \approx 10,000, K \approx 20$ regime.
- **Improvement:** The implementation plan should prioritize JAX-native acceleration over Numba to maintain hardware portability (GPU/TPU) and a unified codebase.
- **Task Detail:** "Develop JIT-compiled kernels in JAX/NumPyro for: (1) Multivariate Normal log-density, (2) Categorical sampling for FFBS backward pass using `jax.lax.scan`."
- **Target:** These are the bottlenecks. JAX's `lax.scan` translates sequential FFBS operations into optimized XLA primitives, matching Numba's performance while staying within the JAX memory space.

### 3.2 Drift-Triggered ASUGS Retraining
**Theoretical Context (Ch. 7 §7.8):** Concentration parameters should adapt to streaming data shifts.
- **Improvement:** Use the **KL-divergence of belief states** as a trigger for retraining, rather than a fixed schedule.
- **Task Addition:** "Implement $KL(b_t || b_{t-1})$ monitoring for the population. Trigger ASUGS online update when $KL > \threshold$."
- **Rationale:** This ensures the model adapts $K$ (number of clusters) only when structural performance shifts occur, saving significant compute during "stable" weeks.

---

## 4. Strategic Execution: The Planning Hierarchy (A, B, C)

### 4.1 Step A: MILP Regularization for Transfer Costs
**Theoretical Context (Ch. 10 §10.9):** Transfer costs are piecewise linear ($f$ free, then 4 per transfer).
- **Improvement:** The MILP solver in Step A must use auxiliary variables to encode this discontinuity exactly.
- **Task Detail:** "Implement piecewise linear cost constraints in `milp_solver.py` using PuLP's boolean variables."

### 4.2 Step B: Scenario Planning with MCTS Pruning
**Theoretical Context (Ch. 8 §8.8):** MCTS builds scenario trees over sampled worlds.
- **Improvement:** Implement **Belief-State Pruning** to handle the branching factor.
- **Task Addition:** "Prune MCTS action space to the top-N players by expected points from the Thompson Sampled world."
- **Rationale:** Without pruning, MCTS is intractable. Pruning to the top-100 candidates makes Step B feasible for tactical lookahead.

### 4.3 Step C: MaxEnt RL alignment for DDQN
**Theoretical Context (Ch. 9 §9.8):** The Boltzmann policy is optimal for the Maximum Entropy RL objective.
- **Improvement:** Frame the DDQN training not just as "points maximization" but as "Entropy-regularized Optimal Control."
- **Implementation Detail:** In `training_loop.py`, explicitly include the $-\tau \log \pi$ term in the soft-Bellman update.
- **Rationale:** This ensures the agent "hedges" its bets across multimodal form clusters rather than committing prematurely to a single noisy estimate.

---

## 5. Revised Implementation Roadmap (Detailed Tasks)

### Phase 2: Posterior Inference & Belief States (New Focus)
- [/] **Task 2.1.1:** Implement **I²GMM** kernels using JAX `vmap` and `jit`.
- [/] **Task 2.1.2:** Implement **FFBS** using `jax.lax.scan` for action-conditioned transitions $\pi_k(a)$.
- [ ] **Task 2.1.3:** Develop **Feature-to-Points mapping** using Negative-Binomial posterior predictive distributions (NB parameters per cluster).
- [ ] **Task 2.1.4:** Construct the **Belief-State MDP Bridge**: Export posterior samples as 757-dimension state tensors for Phase 5.

### Phase 3 & 4: Tactical Engines (TS + MCTS + MILP)
- [ ] **Task 3.1.1:** Build `src/optimization/milp_solver.py` with full formation and transfer cost constraints.
- [ ] **Task 4.1.1:** Implement **TS+MCTS** with world-sampling (D2NG-POMCP style).
- [ ] **Task 4.1.2:** Implement **Captaincy LinUCB** bandit with HDP-HMM belief states as context.

### Phase 5: Strategic Engine (DDQN + MFG)
- [ ] **Task 5.1.1:** Build the 757-dim **757-Dim State Encoder** including chips and beliefs.
- [ ] **Task 5.2.1:** Implement **Mean-Field Game Equilibrium Solver** (Fixed-point iteration between HJB and FP).
- [ ] **Task 5.3.1:** Implement **Boltzmann Policy** with MaxEnt RL soft-Q updates.

---

## 6. Validation & Ablation Plan (Ch. 12 Context)

The implementation plan should explicitly schedule the following **Ablation Studies** described in the docs:
1. **Ablation 1:** IID HDP-GMM vs Temporal HDP-HMM (Predictive power of persistence).
2. **Ablation 2:** TS+MCTS vs DDQN (Lookahead efficiency vs Generalization).
3. **Ablation 8:** Thompson vs Boltzmann vs E-greedy (Exploration quality).
4. **Ablation 10:** Mean-Field Coupling Strength $\omega$ (Strategic stability).

---

## 7. Conclusion
By shifting the focus from "Generic ML Implementation" to "Deep Bayesian Optimal Control," the task list becomes significantly more robust. The core improvements revolve around:
1. **Representational Fidelity:** Capturing multimodality (I²GMM) and persistence (HDP-HMM).
2. **Strategic Depth:** Combining exact MILP constraints with long-horizon neural Q-learning (Step C).
3. **Population Dynamics:** Incorporating price changes via Mean-Field Games.

Adopting this revised plan will ensure the final system is not just a predictor, but a sophisticated Bayesian agent capable of navigating the complex, partially observable environment of the FPL.
