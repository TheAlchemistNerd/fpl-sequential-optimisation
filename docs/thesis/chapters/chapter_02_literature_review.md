# Chapter 2: Literature Review

## 2.1 Overview

This review brings together work on fantasy-sports decision-making, football performance prediction,
Bayesian nonparametrics, bandit methods, and reinforcement learning under partial observability. The
literature provides several relevant pieces: a direct precedent for sequential FPL team formation,
statistical evidence about manager skill and contextual football performance, methods for representing
uncertainty and evolving latent states, and algorithms for selecting actions over time.

These contributions are related, but they do not by themselves establish that one method can solve the
full FPL problem. Results from prediction, bandit, robotics, and experimental-design settings have
different objectives and assumptions. The discussion therefore distinguishes direct FPL evidence from
methods that offer a possible modelling or algorithmic connection. The research question emerging from
this review is whether latent player-form beliefs can support constrained, season-long decisions, and how
that combination performs empirically.

## 2.2 Fantasy Sports as a Decision-Making Domain

Matthews and Ramchurn model FPL team formation as a sequential decision problem with partial
observability. Their belief-state MDP represents uncertainty about player characteristics, while Bayesian
Q-learning and value of perfect information guide decisions. They use a multidimensional knapsack
formulation to generate candidate teams under squad constraints. The study reports that its manager
ranked within the top 1.1–2.5 percent of approximately 2.5 million participants. This is a direct
precedent for belief-based sequential FPL management; its result belongs to that agent and evaluation
setting, rather than establishing expected performance for a different system [@matthews2012fpl].

O’Brien, Gleeson, and O’Sullivan examine skill across historical FPL seasons and analyze the decisions of
nearly one million managers in 2018–19. Their measures include transfer quality, captaincy, team-value
growth, and chip use. They report persistence in manager performance across seasons and an association
between experience and final points, estimated at about 22 additional points per year of experience. In
their chip analysis, 79.4% of top-10k managers used Bench Boost in an optimal Double Gameweek, compared
with 28.9% of other managers. These are findings from the study’s dataset and operational definitions;
they motivate evaluating these decisions separately rather than treating final points as the only measure
of decision quality [@obrien2021fplskill].

The same study identifies a dynamic “template team”: high-ranking managers’ squads show greater
similarity to one another, measured using Jaccard similarity, with patterns varying across the season.
This suggests an empirical comparison between following widely selected players and choosing
differentials, but does not imply that any particular exploration policy will outperform the template
[@obrien2021fplskill].

Gollan, Bellenger, and Norton analyze 380 matches from the 2015–16 Premier League season using 96 OptaPro
performance metrics. Their clustering identifies three recurring team styles; their contextual analysis
finds opposition quality more influential than venue in predicting style, while also observing a home
advantage in some comparisons. The result supports including opponent and venue context in
football-performance analysis. It concerns team-level styles, however, and is not direct evidence that
latent player-form states have the same structure [@gollan2020contextual].

Together, these studies establish a decision-making context for the project: FPL choices recur over a
season, managers differ in measurable decisions, and football performance varies with context. The
reviewed studies use different state representations and objectives. The proposed work investigates
whether nonparametric player-form beliefs can complement these decision formulations; that extension
requires evaluation rather than assumption.

## 2.3 Player Performance Prediction

Tamimi and Tran study player-performance prediction using statistical features alongside news sentiment
and injury information. Their first setup uses a short, one-month EPL sample with 70 engineered features,
Google News articles, and Pysentimiento sentiment analysis; it predicts both participation and
performance. Their second setup spans two seasons, combines Guardian and GDELT news with injury data from
WorldfootballR, and evaluates Ridge regression, LightGBM, a one-dimensional CNN, and an LSTM using a
pretrained DeBERTa news-sentiment model. They report that adding external information reduced the CNN’s
MSE from 6.27 to 5.63; CatBoost performed best on the first setup’s classification task. They also
identify limited data and API access as practical constraints, particularly for deep models. These
findings motivate testing external features, but the reported improvement is specific to the study’s data
and evaluation and does not establish that sentiment will improve this project’s predictions
[@tamimi2025fpl].

Smit, Ravazzolo, and Rossini model EPL goal differences with a dynamic Bayesian hierarchical model and
the Skellam distribution. Team attack and defence strengths vary over time through autoregressive
processes, with a home-advantage term and hierarchical priors. For promoted teams, the model initializes
parameters using those of relegated teams. The paper reports approximately 60% accuracy for match
outcomes and discusses shrinkage effects. This work offers a precedent for dynamic football-strength
modelling, but it predicts team match outcomes rather than individual FPL points; adapting its structure
to player-level rewards would require a separate model and evaluation [@smit2020skellam].

The two studies address complementary questions. Tamimi and Tran evaluate player predictions with
external news and injury features; Smit and colleagues model time-varying team strengths. Neither result
alone determines how predictive uncertainty should affect transfers, captaincy, or chip timing. That
connection motivates the decision methods reviewed below.

## 2.4 Bayesian Nonparametric Methods

The Chinese Restaurant Process (CRP) gives an exchangeable distribution over partitions: an observation
joins an occupied cluster with probability proportional to its membership or starts a new cluster
according to a concentration parameter. The lecture notes by Blei, Frazier, and Mukherjee use this
construction to explain the Dirichlet Process and its logarithmic expected growth in occupied clusters.
Teh and colleagues develop the Hierarchical Dirichlet Process (HDP), in which grouped Dirichlet processes
share a discrete global base measure and therefore share cluster atoms. The stick-breaking and Chinese
Restaurant Franchise representations provide complementary accounts of this sharing structure
[@blei2007crp; @teh2005hierarchical].

This framework supplies a way to represent an unknown number of latent profiles while sharing statistical
strength across groups. It does not guarantee that the inferred clusters correspond to interpretable
football “forms”; that interpretation depends on the observations, model specification, and empirical
diagnostics.

Yerebakan, Rajwa, and Dundar introduce the Infinite Mixture of Infinite Gaussian Mixtures (I²GMM), using
an additional mixture level to represent multimodality within broader clusters. Their work motivates
considering substructure where a single Gaussian component would be restrictive. The examples in the FPL
research notes—such as a player profile varying across home and away fixtures—are hypotheses for this
domain, not findings established by the I²GMM paper. Its inference method and performance also cannot be
assumed to transfer unchanged to the project’s HDP models [@yerebakan2014i2gmm].

Tsiligkaridis and Forsythe present Adaptive SUGS (ASUGS), a low-complexity sequential inference procedure
for Dirichlet Process Gaussian mixtures that adapts the concentration parameter as observations arrive.
Their analysis gives a logarithmic upper bound on class growth and studies asymptotic behavior of the
conditional likelihood; they report experiments on synthetic and real datasets. This provides a relevant
reference for online adaptation when the number of clusters is unknown. Whether its assumptions and
update rule are suitable for seasonally evolving player data remains to be established
[@tsiligkaridis2015asugs].

Bruno, Calinon, and Caldwell propose DIPOLE, combining a Dirichlet Process mixture over policy parameters
with the PoWER policy-search algorithm to discover multiple candidate policies. Their experiments include
Cart-Pole and Mountain Car. The work offers an example of nonparametric modelling within reinforcement
learning, but it concerns robot policy search rather than FPL and does not validate the use of multiple
FPL strategies [@bruno2013dipole].

These papers supply foundations and candidate extensions at different levels: CRP/HDP for shared
clusters, I²GMM for within-cluster mixtures, ASUGS for sequential concentration adaptation, and DIPOLE
for multiple policies. They should not be collapsed into one method: each has distinct assumptions,
inference procedures, and objectives.

## 2.5 Thompson Sampling and Bandit Methods

Agrawal and Goyal analyze Thompson Sampling (TS) in stochastic multi-armed bandits and derive logarithmic
expected-regret bounds under their stated assumptions. This supplies theoretical motivation for posterior
sampling as an exploration strategy. Those guarantees are problem-specific; they do not automatically
apply to constrained, combinatorial FPL decisions or season-long rewards [@agrawal2012thompson].

Hong and colleagues study Mixture Thompson Sampling (MixTS). The method samples a latent mixture
component and then model parameters conditional on that component. Their analysis covers linear bandits
and finite-horizon reinforcement learning, with regret bounds depending on properties of the mixture
prior. A mixture prior is conceptually relevant to uncertainty over player profiles, but using an HDP
posterior inside MixTS would require a compatible decision model and a separate analysis
[@hong2022mixts].

Nakamura and Sugiyama study real-valued combinatorial pure exploration and introduce GenTS-Explore. The
method uses an optimization oracle to find actions for sampled reward vectors, avoiding enumeration of an
exponentially large action set; the paper gives problem-dependent sample-complexity results and evaluates
the approach on tasks including knapsack. Its objective is identifying a best combinatorial action
through exploration. FPL also involves squad constraints, but weekly and season-long management
accumulates rewards and carries state across gameweeks. GenTS-Explore is therefore a relevant comparison,
not a drop-in solution or evidence that a full FPL squad problem has been solved
[@nakamura2024gentsexplore].

Bai, Wu, and Chen propose D2NG-POMCP, a Monte Carlo planning method for POMDPs that incorporates
Thompson-sampling ideas into tree search. Their reward modelling uses conjugate priors, including
Dirichlet models for immediate rewards and Normal-Gamma models for returns, and particle filtering to
represent beliefs. They report comparisons on benchmark domains including RockSample and PocMan. This
work links posterior uncertainty with online planning, but its benchmark results do not establish
performance in FPL. The proposed FPL use of tree search should be described according to its actual
action-selection and rollout implementation, rather than treated as an implementation of D2NG-POMCP
[@bai2014d2ng].

Taken together, these studies cover posterior sampling, mixture priors, combinatorial pure exploration,
and POMDP planning. Their guarantees and empirical results depend on different problem formulations. For
FPL, evaluation must account for the actual squad constraints, information revealed between gameweeks,
transfer costs, and cumulative season objective.

## 2.6 Reinforcement Learning under Partial Observability

Blau and colleagues reformulate sequential experimental design as an MDP and use deep reinforcement
learning to learn policies for discrete and black-box design problems. They report state-of-the-art
results on their experimental-design benchmarks and faster deployment than repeated non-amortized design.
The connection is methodological: both settings involve repeated decisions that affect future
information. Experimental design and FPL team management have different actions and rewards, so the
reported performance does not transfer directly [@blau2022sed].

O’Donoghue’s K-learning represents expected return and epistemic uncertainty through “K-values.” A
risk-seeking exponential utility leads to a soft Bellman operator and a Boltzmann policy; the paper gives
a sublinear Bayes-regret bound of order (O(L^{3/2}sqrt{SAT})) under its assumptions. This provides a
Bayesian account of entropy-regularized exploration. It is related to, but distinct from, a standard DDQN
objective; applying its bound or update to the FPL model would require matching assumptions
[@odonoghue2021kvalues].

Kurniawati reviews POMDPs, belief-state planning, and approximate solvers in robotics. The review
describes the continuous belief-space and growing-history challenges, and discusses methods including
point-based value iteration, SARSOP, and POMCP. This gives general context for why planning under partial
observability is difficult and why approximate methods are used. It does not prescribe a particular FPL
solver [@kurniawati2021pomdp].

Abdulsamad, Iqbal, and Särkkä formulate policy optimization for continuous POMDPs through a Feynman–Kac
model and nested sequential Monte Carlo. Their approach uses outer trajectory particles and inner belief
estimates to optimize a history-dependent policy, with information gathering represented in the inference
objective. It is a possible alternative or comparison for continuous partially observed control; the
review does not establish that it can be directly inserted into a DDQN pipeline [@abdulsamad2025smc].

Peralez and colleagues study Dec-POMDPs using sequential central planning, with a reported reduction in
the complexity of Bellman backups and an oSARSAseq algorithm evaluated in multi-agent domains. This
concerns decentralized multi-agent decision-making. It may inform a future extension involving
interacting managers, but the current single-manager FPL formulation is not itself a Dec-POMDP
[@peralez2025decpomdp].

Marković and colleagues compare Active Inference with Bayesian UCB and optimistic Thompson Sampling in
stationary and switching bandits. Their reported results differ by setting: Active Inference is
eventually outperformed in the stationary case, while it performs better in their switching-bandit
experiments. This makes it a candidate comparison for non-stationary decisions, not a generally superior
method or evidence about FPL performance [@markovic2021activeinference].

The cited work thus offers several distinct ways to address uncertainty, planning horizon, and changing
environments. It supports comparing methods under a common FPL simulator and objective; it does not
justify treating deep RL, active inference, sequential Monte Carlo, and mean-field models as
interchangeable or already validated components.

## 2.7 Convergence Diagnostics for MCMC

Ozturk and Cengiz compare MCMC convergence diagnostics in Bayesian logistic random-effects models. They
discuss Geweke, Raftery–Lewis, Heidelberger–Welch, effective sample size, trace plots, and
autocorrelation plots, and emphasize using multiple diagnostics rather than relying on one test. They
also report that the Monte Carlo standard error to standard deviation ratio is a sensitive indicator in
their study. These diagnostics are relevant to assessing posterior computation, though the cited
application is not an HDP model or FPL data [@ozturk2016mcmc].

Hoffman and Gelman introduce the No-U-Turn Sampler (NUTS), which adapts Hamiltonian Monte Carlo
trajectory lengths using a U-turn criterion. It reduces the need to choose a fixed path length manually;
it does not remove the need to assess convergence and effective sampling for a particular model
[@hoffman2014nuts].

For this project, these references motivate reporting sampler configuration and convergence evidence
alongside downstream decisions. A sampler’s availability or successful execution alone is not evidence
that its posterior estimates are reliable.

## 2.8 Summary and Research Gap

The reviewed FPL studies demonstrate that team formation can be treated as a sequential decision problem
and that manager decisions show measurable differences. Football and player-prediction studies also
provide evidence for contextual and time-varying performance modelling. These findings motivate a
framework that represents uncertainty and evaluates transfers, captaincy, squad composition, and
longer-term consequences. They do not identify a single best decision algorithm.

Bayesian nonparametric methods offer tools for shared latent clusters and for adapting model complexity;
TS and related bandit methods address exploration; POMDP and reinforcement-learning methods address
sequential action under partial observability. The literature reviewed here examines these elements
across different tasks. It does not report an evaluated system that combines the project’s proposed
latent player-form representation with FPL squad constraints and season-level decision-making. This is a
gap within the reviewed set, not a claim that no such work exists elsewhere.

The resulting research task is empirical: specify which components are implemented, compare them against
appropriate baselines, and test whether the inferred beliefs improve decisions over a season. Theoretical
extensions such as I²GMM, alternative policy-learning methods, and multi-agent planning should remain
identified as proposals until their assumptions, integrations, and results are demonstrated.
