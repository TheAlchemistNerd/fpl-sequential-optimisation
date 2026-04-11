# Chapter 2: Literature Review

## 2.1 Overview

<!-- TODO: Write 1-2 paragraph overview positioning this work at the intersection of Bayesian nonparametrics, reinforcement learning, and sports analytics. Identify the gap: no existing work combines HDP-HMM latent form modelling with RL-based sequential decision-making for fantasy sports. -->

## 2.2 Fantasy Sports as a Decision-Making Domain

<!-- TODO: Synthesise from: Matthews & Ramchurn (2012), O'Brien et al. (2021), Gollan et al. (2020)

Key themes to cover:
- Matthews & Ramchurn: first belief-state MDP for FPL, Bayesian Q-learning, top 1.1-2.5% rank → establishes FPL as a legitimate AI benchmark
- O'Brien et al.: quantitative proof that FPL is a game of skill (not luck), identifies transfer quality, captaincy, chip timing as key skill components, "template team" phenomenon
- Gollan et al.: contextual factors (opposition quality > home advantage) influence team styles → motivates context-aware features in the proposed model
- GAP: These works use parametric models with fixed state spaces; none employ nonparametric Bayesian methods for latent form discovery
-->

## 2.3 Player Performance Prediction

<!-- TODO: Synthesise from: Tamimi & Tran (2025), Smit et al. (2020)

Key themes:
- Tamimi & Tran: integrating news sentiment + injury data reduces MSE from 6.27 to 5.63; DeBERTa-v3 sentiment model; CNN architecture outperforms Ridge, LightGBM, LSTM
- Smit et al.: dynamic Bayesian hierarchical model with Skellam distribution for EPL match prediction; time-varying team strengths via autoregressive processes
- GAP: Existing prediction models are discriminative (predict points directly); the proposed approach models the generative process via latent form states, enabling uncertainty quantification and principled exploration
-->

## 2.4 Bayesian Nonparametric Methods

<!-- TODO: Synthesise from: Teh et al. (2005), Blei et al. (2007), Yerebakan et al. (2014), Tsiligkaridis & Forsythe (2015), Bruno et al. (2013)

Key themes:
- Teh et al. (2005): foundational HDP paper — sharing clusters across groups via Chinese Restaurant Franchise
- Blei et al. (2007): CRP intuition — rich-get-richer, exchangeability, logarithmic cluster growth
- Yerebakan et al. (2014): I²GMM — infinite mixture of infinite Gaussian mixtures for within-cluster multi-modality
- Tsiligkaridis & Forsythe (2015): ASUGS — adaptive concentration parameter for streaming data, logarithmic growth bound
- Bruno et al. (2013): DP-based multi-optima policy search in RL
- GAP: None of these apply HDP/HDP-HMM to sports performance data; this work represents the first application of Bayesian nonparametrics for latent player form in fantasy sports
-->

## 2.5 Thompson Sampling and Bandit Methods

<!-- TODO: Synthesise from: Agrawal & Goyal (2012), Hong et al. (2022), Nakamura & Sugiyama (2024), Bai et al. (2014)

Key themes:
- Agrawal & Goyal (2012): first logarithmic regret bound for TS, matching Lai-Robbins lower bound — theoretical justification
- Hong et al. (2022): MixTS — TS with mixture prior, Bayes regret bounds depend on component count/widths → directly applicable to HDP-GMM posterior
- Nakamura & Sugiyama (2024): GenTS-Explore — combinatorial pure exploration without action enumeration, near-optimal sample complexity, Knapsack application
- Bai et al. (2014): D2NG-POMCP — integrating TS into Monte-Carlo tree search for POMDPs with conjugate priors
- GAP: No existing work applies combinatorial TS (GenTS-Explore) to fantasy sports squad selection under nonparametric Bayesian uncertainty
-->

## 2.6 Reinforcement Learning under Partial Observability

<!-- TODO: Synthesise from: Blau et al. (2022), O'Donoghue (2021), Kurniawati (2021), Abdulsamad et al. (2025), Peralez et al. (2025), Marković et al. (2021)

Key themes:
- Blau et al. (2022): SED as MDP — transforming sequential design problems into MDPs solvable with deep RL; handles discrete actions and black-box models
- O'Donoghue (2021): K-values — compressing expected rewards + epistemic uncertainty; Boltzmann policy over K-values achieves O(L^{3/2}√(SAT)) Bayes regret
- Kurniawati (2021): POMDP survey — curse of dimensionality/history, sampling-based solvers (PBVI, SARSOP, POMCP)
- Abdulsamad et al. (2025): nested SMC for continuous POMDPs — inherent information gathering without exploration bonuses
- Peralez et al. (2025): Dec-POMDP sequential central planning — polynomial complexity reduction
- Marković et al. (2021): Active Inference outperforms TS in non-stationary "switching" bandits → relevant for form state transitions
- GAP: No existing work formulates FPL as a belief-state MDP with HDP-HMM belief states and solves it with DDQN + MFG/MFC integration
-->

## 2.7 Convergence Diagnostics for MCMC

<!-- TODO: Synthesise from: Ozturk & Cengiz (2016), Hoffman & Gelman (2014)

Key themes:
- Ozturk & Cengiz: multi-diagnostic approach (Geweke, Raftery-Lewis, ESS, trace plots), MCSE/SD most sensitive
- Hoffman & Gelman (2014): NUTS — automatic trajectory length tuning, U-turn condition
- Relevance: ensuring HDP-HMM posterior samples are reliable for downstream decision-making
-->

## 2.8 Summary and Research Gap

<!-- TODO: Write 2-3 paragraphs summarising the gap this dissertation fills:

1. Existing FPL work uses parametric models with fixed state spaces (Matthews & Ramchurn 2012) or purely predictive models (Tamimi & Tran 2025). No work models latent player form nonparametrically.

2. Bayesian nonparametric methods (HDP, HDP-HMM) have been applied to document modelling and speech recognition but never to sports analytics or fantasy sports.

3. Thompson Sampling and DDQN have been studied independently but never unified within a single framework that also incorporates MFG/MFC population dynamics and exact MILP constraint satisfaction.

4. This dissertation addresses all three gaps simultaneously.
-->
