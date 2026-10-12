# Chapter 13: Conclusion and Future Work

## 13.1 Summary of Contributions

<!-- TODO: Restate each contribution with reference to evidence from Chapter 12:

1. **HDP-HMM for latent player form modelling** (Chapters 4-6): First application of Bayesian nonparametric methods to fantasy sports. The HDP-HMM with Gaussian-Wishart base measure discovers [X] interpretable form profiles per position and captures form persistence with diagonal-dominant transition matrices (avg π_{k,k} ≈ [X]).

2. **Mean-Field Game analysis of classical FPL leagues** (Chapter 10, §10.10): Derived the coupled HJB-FP system characterising Nash equilibrium behaviour. The MFG framework reveals [strategic insight from experiments].

3. **Mean-Field Control for draft and H2H leagues** (Chapter 10, §10.11): Formulated optimal policies accounting for resource scarcity and opponent strength distributions.

4. **Unified decision framework** (Chapters 8-10): Integrated Thompson Sampling (myopic, interpretable) with DDQN (long-horizon, strategic) and exact MILP constraint satisfaction. Ablations in Chapter 12 show [key finding].

5. **End-to-end MLOps pipeline** (Chapter 11): Production-grade automated pipeline for data ingestion, model updating, inference, and monitoring.
-->

## 13.2 Key Findings

<!-- TODO: Discuss the most important experimental results:
- Which decision engine performed best overall? In which contexts?
- How much did the nonparametric approach (HDP-HMM) improve over parametric alternatives?
- What was the impact of external data sources (sentiment, Understat)?
- How did the agent compare with human managers?
-->

## 13.3 Limitations

<!-- TODO: Be honest about limitations:

1. **Simulated backtesting only**: The agent was evaluated on historical data, not deployed in a live FPL season. Live performance may differ due to:
   - Price change dynamics (the agent's transfers would affect prices in reality)
   - API latency and data staleness
   - Model-reality gap in transition dynamics

2. **Computational cost**: The HDP-HMM MCMC inference requires [X] minutes/hours per gameweek. This may be a bottleneck for real-time deployment.

3. **Mean-field approximation**: The MFG/MFC assumes a continuum of agents. With ~10M discrete managers, the mean-field limit is a reasonable but imperfect approximation.

4. **Action space simplification**: The DDQN restricts the action space to top-100 candidate transfers. The full combinatorial space is too large for direct neural network output.

5. **No multi-season transfer learning**: Each season's model is trained independently. Cross-season transfer of learned form profiles is not implemented.

6. **Sentiment data availability**: News sentiment data may not be available for all players, especially lower-profile ones.
-->

## 13.4 Future Work

<!-- TODO: Propose concrete future directions:

1. **Live deployment**: Deploy the agent in a real FPL season (38-week longitudinal study). Compare live performance with backtested predictions.

2. **Hierarchical RL**: Replace the flat DDQN with a hierarchical RL framework where high-level options (chip timing, squad strategy) are learned separately from low-level actions (individual transfers).

3. **Graph Neural Networks for team interactions**: Model player-team-opponent interactions as a graph, using GNNs to capture positional relationships and tactical dependencies (extending Gollan et al. 2020).

4. **Active Inference**: Implement the Active Inference framework (Marković et al. 2021) as a third decision engine, leveraging its superior performance in non-stationary environments.

5. **Cross-season transfer learning**: Develop methods to transfer learned form profiles across seasons, accounting for player transfers, aging, and tactical evolution.

6. **Multi-agent draft league**: Extend the MFC framework to a full multi-agent simulation for draft leagues, where multiple agents compete for player acquisition.

7. **Attention-based temporal models**: Replace the HDP-HMM with a transformer-based temporal model that can capture long-range dependencies in form trajectories.

8. **Causal inference for transfers**: Use causal inference methods to distinguish between genuine form transitions and confounded effects (e.g., fixture difficulty masking true form).

9. An untested extension is to estimate weekly price changes from aggregate transfer intensity, then evaluate whether this population-level signal changes the value or timing of early transfers. This is a proposed coupling between mean-field dynamics and price forecasting; it is not implemented or validated in the current system.
-->

## 13.5 Closing Remarks

<!-- TODO: 1-2 paragraphs reflecting on the broader significance:
- FPL as a microcosm of real-world sequential decision-making under uncertainty
- The modularity of the framework (each component can be improved independently)
- The potential for applying this framework to other fantasy sports, portfolio optimisation, or resource allocation problems
-->
