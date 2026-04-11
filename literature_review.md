# Literature Review

## Players' Performance Prediction for Fantasy Premier League, Using Transformer-based Sentiment Analysis on News and Statistical Data

*   **Authors:** Tamimi, M., Tran, T.
*   **Year:** 2025
*   **Source:** International Journal of Computer Science in Sport

### Key Contributions:

This paper presents an innovative approach to enhance FPL analysis and performance prediction by integrating news sentiment and players' injury with statistical data sources. The authors demonstrate that leveraging diverse data sources can lead to more accurate predictions and informed decision-making in FPL.

### Methodology:

The authors propose two methods:

1.  **Method 1:**
    *   **Objective:** Predict player performance (regression) and participation (classification).
    *   **Dataset:** A small portion of the EPL dataset (one month) with 70 engineered features.
    *   **External Data:** News articles from the Google News API for the week preceding each match.
    *   **Sentiment Analysis:** A pretrained transformer-based toolkit (Pysentimiento) was used to extract sentiment from news articles.
    *   **Models:** Boosting algorithms including Gradient Boosting (GB), CatBoost, and XGBoost.

2.  **Method 2:**
    *   **Objective:** Predict the points each player would gain in each game (regression).
    *   **Dataset:** A larger EPL dataset spanning two seasons (2022/23 and 2023/24).
    *   **External Data:** News articles from The Guardian and GDELT APIs, and player injury data from the `WorldfootballR` package.
    *   **Sentiment Analysis:** A pretrained Sentiment Analysis Pipeline from Hugging Face (DeBERTa-v3-small-ft-news-sentiment-analisys).
    *   **Models:** Ridge Regression, LightGBM, 1-dimensional Convolutional Neural Networks (CNN), and Long Short-Term Memory (LSTM) architectures.

### Key Findings:

*   **Impact of External Data:** Integrating news sentiment and injury data improved model performance. The CNN architecture with these additional features achieved a reduction in Mean Squared Error (MSE) from 6.27 to 5.63, outperforming the state-of-the-art model.
*   **Algorithm Performance:**
    *   In the classification task, the CatBoost algorithm outperformed other boosting algorithms.
    *   In the regression task, the CNN architecture generally yielded better results compared to other models.
*   **Challenges:** The study faced challenges with data access due to API restrictions and the limited size of the dataset, which posed challenges for training deep learning models like LSTM.

### Relevance to Dissertation:

This paper is highly relevant as it directly addresses the problem of FPL performance prediction. It provides a strong case for incorporating external data sources, particularly news sentiment and injury data. The methodologies explored, including the use of transformer-based sentiment analysis, CNNs, and LSTMs, are directly applicable to the proposed dissertation research. The challenges identified in this paper, such as data access and dataset size, are also important considerations for the dissertation.

## Thompson Sampling with a Mixture Prior

*   **Authors:** Hong, J., Kveton, B., Zaheer, M., & Boutilier, C.
*   **Year:** 2022
*   **Source:** Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)

### Key Contributions:

This paper studies Thompson Sampling (TS) in online decision-making where the environment is sampled from a mixture distribution. This is particularly relevant in multi-task learning scenarios where an agent faces different classes of problems. The authors propose a general algorithm called Mixture Thompson Sampling (MixTS) that incorporates this structure by initializing TS with a mixture prior.

### Methodology:

*   **Algorithm:** MixTS first samples a latent state from its posterior and then samples model parameters conditioned on that state.
*   **Analysis:** The authors develop a novel and general proof technique for analyzing the concentration of mixture distributions. They use this technique to prove Bayes regret bounds for MixTS in both linear bandits and finite-horizon reinforcement learning.
*   **Bounds:** The derived regret bounds capture the structure of the prior, depending on the number of mixture components and their widths.

### Key Findings:

*   **Improved Performance:** The paper demonstrates the empirical effectiveness of MixTS in synthetic and real-world experiments, showing that it outperforms standard TS with a less-informative unimodal prior.
*   **Theoretical Guarantees:** The paper provides the first Bayes regret bounds for TS with a mixture prior that are sublinear in the number of rounds and depend on how informative the prior is.
*   **Generalizability:** The analysis outline can be applied to any model by specifying the confidence intervals, demonstrating the modularity and generality of the approach.

### Relevance to Dissertation:

This paper is highly relevant to the proposed dissertation, particularly for the Thompson Sampling approach to FPL team selection. The concept of using a mixture prior to model different classes of problems (e.g., different player types or form states) is directly applicable. The MixTS algorithm and its theoretical analysis provide a strong foundation for developing and analyzing a sophisticated Thompson Sampling agent for FPL. The paper's focus on regret bounds and the impact of prior information is also highly relevant to the dissertation's experimental and theoretical contributions.

## Variational Bayesian Reinforcement Learning with Regret Bounds

*   **Author:** O'Donoghue, B.
*   **Year:** 2021
*   **Source:** 35th Conference on Neural Information Processing Systems (NeurIPS)

### Key Contributions:

This paper introduces a new approach to Bayesian reinforcement learning by deriving a new Bellman operator and a fixed point called 'knowledge values' (K-values). These K-values compress both the expected future rewards and the epistemic uncertainty into a single value, allowing the agent to balance exploration and exploitation in a principled manner.

### Methodology:

*   **K-learning:** The proposed algorithm, K-learning, endows the agent with a risk-seeking exponential utility function. This leads to a new optimistic Bellman operator that replaces the standard max operator with a soft-max and incorporates a 'bonus' for state-action visitation.
*   **Boltzmann Policy:** The agent follows a Boltzmann policy over the K-values, with a carefully chosen temperature schedule to ensure a sub-linear Bayes regret bound.
*   **Regret Analysis:** The paper provides a theoretical analysis of the Bayes regret for K-learning, showing that it matches the best-known bound for Thompson sampling up to logarithmic factors.

### Key Findings:

*   **Principled Exploration:** The paper shows that soft-max operators and maximum entropy policies in RL can be derived from a principled Bayesian perspective by endowing the agent with a risk-seeking utility function.
*   **Sub-linear Regret:** The K-learning algorithm is shown to achieve a sub-linear Bayes regret bound of O(L^(3/2) * sqrt(SAT)), where L is the time horizon, S is the number of states, A is the number of actions, and T is the total number of elapsed timesteps.
*   **Connections to Other Methods:** The paper highlights the deep connections between K-learning, soft Q-learning, and maximum-entropy reinforcement learning.

### Relevance to Dissertation:

This paper is highly relevant as it provides a theoretical foundation for balancing exploration and exploitation in a Bayesian RL setting. The concept of K-values and the optimistic Bellman operator offers a novel way to think about exploration bonuses. The regret bounds analysis provides a strong theoretical justification for the proposed approach. The connections to soft Q-learning and maximum entropy RL are also relevant, as these are popular and effective techniques in deep RL. The use of a Boltzmann policy over the K-values is also a practical and effective exploration strategy that can be implemented in the FPL agent.

## Adaptive Low-Complexity Sequential Inference for Dirichlet Process Mixture Models

*   **Authors:** Tsiligkaridis, T., Forsythe, K. W.
*   **Year:** 2015
*   **Source:** 29th Conference on Neural Information Processing Systems (NIPS)

### Key Contributions:

This paper develops a sequential low-complexity inference procedure for Dirichlet process mixture models (DPMM) of Gaussians for online clustering and parameter estimation when the number of clusters is unknown a-priori. The authors propose a novel adaptive low-complexity design for the Dirichlet process concentration parameter.

### Methodology:

*   **Algorithm:** The paper proposes an adaptive non-Bayesian approach for adapting the concentration parameter, motivated by large-sample asymptotics, and calls the resulting algorithm ASUGS (Adaptive SUGS).
*   **Concentration Parameter Adaptation:** The main contribution is a novel low-complexity stable method for choosing the concentration parameter adaptively as new data arrive. This method is data-driven and has a theoretical guarantees.
*   **Asymptotic Analysis:** The paper proves that the conditional likelihood is asymptotically Gaussian in the large-sample limit, implying that the clustering part of ASUGS asymptotically behaves as a Gaussian classifier.

### Key Findings:

*   **Logarithmic Growth of Classes:** The authors derive an upper bound on the number of classes, showing that it grows at most at a logarithmic rate with the number of samples.
*   **Stability of Concentration Parameter:** The paper analytically studies the evolution and stability of the adapted sequence of concentration parameters in the online learning setting.
*   **Superior Performance:** Experiments on synthetic and real data sets demonstrate that the proposed approach is superior to other online state-of-the-art methods for learning DPMMs.

### Relevance to Dissertation:

This paper is highly relevant to the dissertation as it addresses the problem of online clustering with an unknown number of clusters, which is a core component of the HDP-GMM approach for modeling latent player forms. The proposed ASUGS algorithm and the adaptive method for choosing the concentration parameter are directly applicable to the FPL problem. The theoretical analysis of the growth rate of the number of classes and the asymptotic normality of the conditional likelihood provides a strong theoretical foundation for the use of DPMMs in this context. The paper's focus on low-complexity sequential inference is also highly relevant, as the FPL problem involves streaming data and requires efficient online learning algorithms.

## The Infinite Mixture of Infinite Gaussian Mixtures

*   **Authors:** Yerebakan, H. Z., Rajwa, B., & Dundar, M.
*   **Year:** 2014
*   **Source:** Advances in Neural Information Processing Systems 27 (NIPS 2014)

### Key Contributions:

This paper introduces the **Infinite Mixture of Infinite Gaussian Mixtures (I²GMM)**, a novel nonparametric Bayesian generative model that extends the standard Dirichlet Process Mixture of Gaussians (DPMG) to handle complex, skewed, or multi-modal cluster distributions. Standard DPMG models assign a single Gaussian component per cluster, which often results in extraneous clusters when modeling real-world data with complex shapes. I²GMM solves this by replacing each single Gaussian with an entire DPMG, enabling rich, multi-modal representations of each cluster without pre-specifying the number of clusters or components.

### Methodology:

*   **Hierarchical DP Prior Structure:** The I²GMM employs a two-level hierarchical Dirichlet process prior. At the top level, a global Dirichlet process defines a distribution over an infinite number of clusters. At the cluster level, each cluster is itself modeled as a DPMG, with its base distribution centered at the atoms of the higher-level DP. This creates a structure where each top-level cluster is an infinite mixture of infinite Gaussian components.
*   **Collapsed Gibbs Sampling:** The authors develop an efficient collapsed Gibbs sampler for posterior inference. The sampler integrates out certain parameters analytically, reducing variance and improving convergence. The design of the sampler supports partial parallelization, enabling the model to scale to larger datasets.
*   **Flexibility in Cluster Shapes:** By replacing single Gaussians with infinite Gaussian mixtures per cluster, I²GMM can capture skewed, elongated, or multi-modal cluster shapes that a standard DPMG would fragment into many smaller clusters.

### Key Findings:

*   **Superior Clustering Accuracy:** Experiments on both synthetic and real-world datasets demonstrate that I²GMM predicts cluster assignments more accurately than existing DPMG implementations based on variational Bayes and standard Gibbs sampling.
*   **Reduced Extraneous Clusters:** The model requires fewer total clusters to explain complex data distributions, as each cluster can capture multi-modal structure internally.
*   **Scalability:** The partially parallelizable collapsed Gibbs sampler enables I²GMM to handle datasets that are computationally prohibitive for standard DPMG implementations.

### Relevance to Dissertation:

This paper is highly relevant to the FPL dissertation, particularly for enhancing the **Hierarchical Dirichlet Process Gaussian Mixture Model (HDP-GMM)** used to model latent player "form."

*   **Multi-Modal Player Form Profiles:** In FPL, a single latent "form" state may itself be multi-modal. For example, an "Elite Form" state might contain sub-profiles such as "Elite at Home" versus "Elite Away," or "Elite Against Weak Defenses" versus "Elite Against Top Defenses." The I²GMM's ability to model each top-level cluster as an infinite mixture of Gaussians directly addresses this, allowing each discovered form state to have its own internal substructure.
*   **Improved Modeling of Complex Distributions:** FPL player performance data is often skewed (many low-scoring games, occasional high-scoring hauls) and multi-modal (different performance patterns against different types of opponents). The I²GMM provides a more flexible modeling framework that can capture these complex distributions without fragmenting them into many small, hard-to-interpret clusters.
*   **Hierarchical Extension of HDP-GMM:** The I²GMM's two-level hierarchical DP structure complements the existing HDP framework. While the HDP shares clusters across players (the "Chinese Restaurant Franchise"), the I²GMM adds an additional layer of flexibility within each cluster. This suggests a natural extension: an HDP-I²GMM where each shared form profile is itself an infinite Gaussian mixture, enabling even richer modeling of player form dynamics.
*   **Efficient Inference:** The collapsed Gibbs sampler with partial parallelization is directly applicable to the FPL problem, where the number of player-gameweek observations is large. This inference approach can be adapted to the NumPyro/Pyro implementation of the HDP-GMM, improving scalability and convergence.
*   **Connection to HDP-HMM:** When combined with the HDP-HMM framework for modeling temporal dynamics, the I²GMM's multi-modal cluster structure provides even richer state representations. Each hidden Markov state can have its own internal mixture, allowing the model to capture the fact that a player's "form" may manifest differently depending on contextual factors (opponent quality, home/away, fixture congestion).

In summary, the I²GMM paper provides a powerful methodological extension that directly enhances the HDP-GMM approach for modeling latent player form in FPL. Its ability to capture multi-modal, complex cluster shapes within a nonparametric Bayesian framework makes it an ideal candidate for improving the fidelity and interpretability of player form profiles.

## Hierarchical Dirichlet Processes

*   **Authors:** Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M.
*   **Year:** 2005

### Key Contributions:

This paper introduces the Hierarchical Dirichlet Process (HDP), a nonparametric Bayesian approach for modeling grouped data. The HDP allows groups of data to share statistical strength by sharing clusters, while still allowing for group-specific variations. This is particularly useful when the number of clusters is unknown a priori and needs to be inferred from the data.

### Methodology:

*   **Hierarchical Model:** The HDP is a hierarchical model where the base measure for a set of Dirichlet processes is itself drawn from a Dirichlet process. This forces the base measure to be discrete, which in turn forces the child Dirichlet processes to share atoms, thus allowing for the sharing of clusters between groups.
*   **Representations:** The paper discusses three representations of HDPs:
    1.  **Stick-breaking construction:** Provides an explicit construction of the random measures.
    2.  **Chinese restaurant franchise:** A generalization of the Chinese restaurant process that provides a metaphor for the clustering properties of the HDP.
    3.  **Infinite limit of finite mixture models:** Shows that the HDP can be derived as the limit of a sequence of finite mixture models.
*   **Inference:** The paper presents Markov chain Monte Carlo (MCMC) algorithms for posterior inference in HDP mixtures.

### Key Findings:

*   **Sharing Clusters:** The HDP provides a principled way to share clusters among related groups of data.
*   **Nonparametric:** The HDP is a nonparametric model, meaning that the number of clusters does not need to be specified in advance and can be inferred from the data.
*   **Flexibility:** The HDP is a flexible framework that can be extended to multiple hierarchical levels and applied to a wide range of problems, including information retrieval, text modeling, and genetics.

### Relevance to Dissertation:

This paper is a foundational text for the proposed dissertation. The HDP is the core of the proposed approach for modeling latent player forms in FPL. This paper provides the theoretical background and justification for using HDPs in this context. The different representations of the HDP, particularly the Chinese restaurant franchise and the stick-breaking construction, are essential for understanding and implementing the model. The MCMC algorithms for posterior inference are also directly relevant to the proposed methodology. The applications of HDPs to document modeling are particularly relevant, as the problem of modeling player performance can be seen as analogous to modeling documents, where players are documents and their performances are words.

## Optimizing Sequential Experimental Design with Deep Reinforcement Learning

*   **Authors:** Blau, T., Bonilla, E., Dezfouli, A., & Chades, I.
*   **Year:** 2022

### Key Contributions:

This paper proposes a method for transforming sequential experimental design (SED) problems into Markov Decision Processes (MDPs), such that they can be solved using standard deep reinforcement learning (RL) algorithms. This approach addresses several limitations of existing methods for Bayesian optimal experimental design (BOED), such as the need for differentiable models and continuous design spaces.

### Methodology:

*   **SED as an MDP:** The authors show that the problem of optimizing a policy for sequential experimental design can be reduced to solving an MDP. They define the state, action, reward, and transition dynamics of this equivalent MDP.
*   **Deep RL for SED:** By formulating SED as an MDP, the authors are able to apply modern deep RL techniques to learn design policies offline. These policies can then be used for rapid online experiment design.
*   **Advantages of the RL Formulation:**
    *   **Discrete Design Spaces:** RL algorithms can handle discrete action spaces, making the approach applicable to a wider range of problems.
    *   **Black-box Models:** The RL approach does not require gradients of the likelihood function, allowing it to be used with black-box probabilistic models.
    *   **Improved Exploration:** Modern RL algorithms have sophisticated exploration mechanisms that can help find better solutions in complex design spaces.

### Key Findings:

*   **State-of-the-Art Performance:** The proposed RL approach achieves state-of-the-art performance on both continuous and discrete experimental design problems, outperforming existing methods like deep adaptive design (DAD).
*   **Computational Efficiency:** The learned policies are computationally efficient at deployment time, achieving inference times orders of magnitude faster than non-amortized methods.
*   **Superior Exploration:** The experiments show that the RL policies learn to propose more diverse designs compared to other methods, leading to better performance in challenging exploration problems.

### Relevance to Dissertation:

This paper is highly relevant as it provides a novel and powerful framework for optimizing sequential decision-making problems, which is the core of the FPL problem. The idea of formulating the FPL team selection problem as an MDP and solving it with deep RL is a key component of the proposed dissertation. This paper provides a strong theoretical and empirical justification for this approach. The advantages of the RL formulation, such as the ability to handle discrete action spaces and black-box models, are directly applicable to the FPL problem, where the action space is discrete and the underlying model of player performance is complex and potentially non-differentiable. The paper's emphasis on exploration is also highly relevant, as exploration is a key challenge in FPL.

## Bayesian Nonparametrics: The Chinese Restaurant Process

*   **Authors:** Blei, D., Frazier, P., Mukherjee, I.
*   **Year:** 2007
*   **Source:** COS 597C: Bayesian nonparametrics, Lecture # 1

### Key Contributions:

These lecture notes provide a clear and intuitive introduction to the Chinese Restaurant Process (CRP), a fundamental concept in Bayesian nonparametrics. The notes explain how the CRP can be used to define a distribution over an infinite number of clusters, and how it relates to the Dirichlet Process (DP). This provides a foundation for understanding more complex models like the Hierarchical Dirichlet Process (HDP).

### Methodology:

The notes introduce the CRP through a simple metaphor:

*   **The Metaphor:** Customers (data points) enter a restaurant with an infinite number of tables (clusters). The first customer sits at the first table. The nth customer chooses an occupied table with a probability proportional to the number of customers already there, or chooses a new, unoccupied table with a probability proportional to a concentration parameter, α.
*   **Properties:** The notes highlight two key properties of the CRP:
    1.  **Exchangeability:** The order in which customers arrive does not affect the probability of the final seating arrangement (partition).
    2.  **Logarithmic Growth:** The expected number of occupied tables (clusters) grows logarithmically with the number of customers. This "rich get richer" phenomenon ensures that a few large clusters will dominate, while still allowing for the creation of new clusters.
*   **CRP Mixture Models:** The notes explain how to construct a CRP mixture model. Each table is endowed with a parameter (e.g., a mean and variance for a Gaussian mixture model). When a customer sits at a table, they draw their data from the distribution associated with that table. This allows for clustering data without pre-specifying the number of clusters.
*   **Connection to Dirichlet Processes:** The notes explain that the CRP provides a constructive and intuitive way to understand the Dirichlet Process. The CRP describes the distribution of partitions of data points that arises from a Dirichlet Process mixture model.

### Relevance to Dissertation:

This document is highly relevant to the dissertation as it provides a clear and concise explanation of the foundational concepts behind the Hierarchical Dirichlet Process Gaussian Mixture Model (HDP-GMM) proposed for modeling latent player "form".

*   **Foundation for HDP:** The CRP is the building block for the HDP. The "Chinese Restaurant Franchise" model, which is a common way to explain the HDP, is a direct extension of the CRP. A solid understanding of the CRP is therefore essential for understanding and implementing the HDP.
*   **Non-Parametric Clustering:** The FPL project aims to model latent player "form" without knowing in advance how many "form" states exist. The CRP, and by extension the HDP, provides a principled Bayesian framework for this kind of non-parametric clustering. The ability to automatically infer the number of clusters from the data is a key advantage of this approach.
*   **Intuitive Understanding:** The CRP metaphor provides a powerful and intuitive way to think about the clustering process. This is invaluable for developing and debugging the HDP-GMM model.

In summary, these lecture notes provide an essential introduction to the concepts that underpin a core component of the proposed dissertation research. They bridge the gap between the high-level concept of non-parametric Bayesian modeling and the concrete implementation details of models like the HDP.

## An Empirical Evaluation of Active Inference in Multi-Armed Bandits

*   **Authors:** Marković, D., Stojić, H., Schwöbel, S., & Kiebel, S. J.
*   **Year:** 2021
*   **Source:** arXiv:2101.08699v4 [cs.LG]

### Key Contributions:

This paper provides an empirical comparison of Active Inference with two state-of-the-art bandit algorithms, Bayesian Upper Confidence Bound (UCB) and optimistic Thompson Sampling. The evaluation is performed on both stationary and, more importantly, non-stationary "switching" bandit problems. The paper introduces an approximate but scalable Active Inference algorithm and demonstrates its effectiveness, particularly in dynamic environments.

### Methodology:

*   **Active Inference:** The paper frames decision-making as a process of minimizing expected free energy. This objective function combines two key elements:
    1.  **Extrinsic Value (Exploitation):** The drive to achieve preferred outcomes or rewards.
    2.  **Intrinsic Value (Exploration):** The drive to gain information and reduce uncertainty about the environment. This is also referred to as epistemic value or novelty.
    This formulation provides a unified, principled approach to the exploration-exploitation dilemma, where every action is chosen to optimize both objectives simultaneously.
*   **Bandit Problems:** The algorithms are tested on two types of multi-armed bandit problems:
    1.  **Stationary Bandit:** Reward probabilities are fixed throughout the experiment.
    2.  **Switching Bandit:** The best arm changes over time, creating a non-stationary environment. This is a more realistic model for many real-world scenarios where conditions are not static.

### Key Findings:

*   **Stationary Environments:** In the classic stationary bandit problem, Active Inference performs well initially but is asymptotically outperformed by Thompson Sampling and Bayesian UCB. The authors suggest this is due to the fixed precision parameter that balances exploration and exploitation in their Active Inference agent, which does not adapt over time.
*   **Dynamic (Switching) Environments:** In the more challenging switching bandit problem, Active Inference demonstrates substantially better performance than both Thompson Sampling and Bayesian UCB. The paper attributes this to Active Inference's more sophisticated and efficient strategy for information gathering. When the environment is volatile, the ability to effectively reduce uncertainty becomes paramount, and Active Inference excels in this regard.

### Relevance to Dissertation:

This paper is highly relevant to the FPL project, as it directly compares different strategies for sequential decision-making under uncertainty in dynamic environments.

*   **Modeling Player Form Dynamics:** The "switching bandit" scenario is an excellent analogue for the FPL problem, where player "form" is a latent, non-stationary variable. A player who is performing well one week may not be the next. The superior performance of Active Inference in this setting strongly suggests it could be a powerful and effective model for the FPL decision-making agent.
*   **A New Direction for the Agent:** The preliminary research document proposes DDQN and Thompson Sampling as the primary methods for the decision-making agent. This paper introduces Active Inference as a compelling third alternative, which may be better suited to the specific challenges of the FPL domain. It provides a clear rationale for exploring Active Inference as part of the dissertation research.
*   **Connection to POMDPs:** The paper explicitly links Active Inference to the broader framework of Partially Observable Markov Decision Processes (POMDPs), which is the chosen formulation for the FPL problem in the preliminary research. This reinforces the theoretical coherence of applying Active Inference in this context.

In conclusion, this paper provides strong empirical evidence that Active Inference is a highly promising framework for decision-making in dynamic environments like FPL. It challenges the default assumption that standard bandit algorithms like Thompson Sampling are always the best choice and opens up a new and exciting avenue for the dissertation research.

## Partially Observable Markov Decision Processes (POMDPs) and Robotics

*   **Author:** Kurniawati, H.
*   **Year:** 2021
*   **Source:** arXiv:2107.07599v1 [cs.RO]

### Key Contributions:

This paper provides a comprehensive review of the Partially Observable Markov Decision Process (POMDP) framework, with a focus on its application in robotics. It details the computational challenges that historically made POMDPs impractical and surveys the modern, sampling-based approximate solvers that have enabled their successful application to complex, real-world problems. The paper serves as an excellent primer on the theory and practice of POMDPs.

### The POMDP Framework:

The paper formally defines a POMDP as a 6-tuple: (S, A, O, T, Z, R), representing the state space, action space, observation space, transition function, observation function, and reward function. The core of the POMDP is that the agent does not know its true state. Instead, it maintains a **belief state**, which is a probability distribution over all possible states. The goal is to find an optimal policy that maps beliefs to actions to maximize the expected cumulative reward.

### Computational Challenges and Solutions:

The author highlights the main reasons for the notorious difficulty of solving POMDPs:

*   **Curse of Dimensionality:** The belief space is continuous and its dimensionality grows with the number of states, making exhaustive search impossible.
*   **Curse of History:** The number of possible belief states to explore grows exponentially with the planning horizon.

To overcome these challenges, the paper focuses on **sampling-based approximate solvers**. These algorithms, such as Point-Based Value Iteration (PBVI), SARSOP (Successive Approximations of the Reachable Space under Optimal Policies), and online planners like POMCP (Partially Observable Monte Carlo Planning), have been instrumental in making POMDPs practical. The key idea is to focus computation on a limited set of reachable and representative belief points, rather than the entire belief space. This makes it possible to find high-quality approximate solutions for problems with very large or even continuous state spaces.

### Relevance to Dissertation:

This paper is of fundamental importance to the FPL dissertation, as it provides the theoretical and practical justification for the entire problem formulation.

*   **Formalizing the FPL Problem:** The FPL decision-making process is explicitly framed as a POMDP in the preliminary research. This paper provides the formal definition and a deep dive into the mechanics of the POMDP framework, solidifying the theoretical underpinnings of the project. The concept of the belief state is particularly crucial, as the HDP-GMM in the FPL project is precisely a model for this belief over a player's latent "form".
*   **Justifying the Solution Approach:** The paper explains *why* the FPL problem, when framed as a POMDP, is computationally hard. It then explains *how* modern methods can solve it. The proposed solution for the FPL agent—combining a belief-state model (HDP-GMM) with a sophisticated decision-making component (DDQN, Thompson Sampling, or Active Inference)—is a form of approximate POMDP solution. This paper provides the context and rationale for why this is a sensible and state-of-the-art approach.
*   **Practical Guidance:** The paper discusses various solvers and software toolkits available for POMDPs. This is valuable practical information for the implementation phase of the dissertation. It also provides insights into the challenges of applying POMDPs to real-world problems, such as model creation and the trade-offs between online and offline planning.

In essence, this paper acts as a bridge between the high-level problem formulation in the preliminary research and the specific algorithms and techniques required to solve it. It confirms that framing the FPL problem as a POMDP is not only theoretically sound but also practically feasible with modern approximate solution methods.

## Sequential Monte Carlo for Policy Optimization in Continuous POMDPs

*   **Authors:** Abdulsamad, H., Iqbal, S., & Särkkä, S.
*   **Year:** 2025
*   **Source:** arXiv:2505.16732v1 [cs.LG]

### Key Contributions:

This paper introduces a novel policy optimization framework specifically designed for continuous Partially Observable Markov Decision Processes (POMDPs). The core contribution lies in casting policy learning as a probabilistic inference problem within a non-Markovian Feynman-Kac model. This approach inherently integrates the value of information gathering into the policy optimization process, eliminating the need for explicit exploration bonuses or handcrafted heuristics. The authors propose a nested Sequential Monte Carlo (SMC) algorithm to efficiently estimate a history-dependent policy gradient, which is then used for policy optimization.

### Methodology:

*   **Feynman-Kac Models and Control-as-Inference:** The paper leverages the connection between optimization and probabilistic inference by formulating the POMDP policy learning problem within a Feynman-Kac model. This model naturally captures the adaptive nature of decision-making and the value of future observations. The policy optimization is framed as maximum likelihood estimation within this model.
*   **Nested Sequential Monte Carlo (SMC):** To address the computational challenges of continuous POMDPs and the entanglement of inference and control, a nested SMC algorithm is developed. This algorithm uses an outer particle filter to target the trajectory-level distribution defined by the Feynman-Kac model and an inner particle filter to maintain estimates of the belief states. This nested structure allows for efficient sampling from the optimal trajectory distribution and estimation of the policy gradient.
*   **Policy Gradient Estimation:** The policy gradient is estimated using a novel method that benefits from lower variance compared to traditional REINFORCE-like estimators. This is achieved by sampling trajectories directly from a reward-weighted posterior distribution, which concentrates probability mass on high-reward trajectories.

### Key Findings:

*   **Principled Information Gathering:** The framework inherently incorporates the value of information gathering, leading to deliberate and intentional exploration without the need for ad-hoc exploration strategies.
*   **Scalability to Continuous Spaces:** The method is designed to handle continuous state, action, and observation spaces, which are prevalent in many real-world applications.
*   **Empirical Effectiveness:** The proposed algorithm demonstrates effectiveness across standard continuous POMDP benchmarks, outperforming existing methods that often struggle with uncertainty and require problematic simplifications (e.g., QMDP approximation).

### Relevance to Dissertation:

This paper is highly relevant to the FPL project, particularly for enhancing the Deep Reinforcement Learning (DDQN) component and strengthening the overall POMDP formulation.

*   **Advanced Policy Optimization for DDQN:** The DDQN agent in the preliminary research aims to learn an optimal policy. This paper offers a sophisticated and principled policy optimization framework that can be directly applied or adapted to improve the DDQN's learning process, especially in handling continuous state spaces and ensuring effective exploration.
*   **Handling Continuous Player States:** The FPL problem involves continuous player statistics and latent "form" states. This paper's focus on continuous POMDPs provides a robust methodology for dealing with such continuous variables within the reinforcement learning framework.
*   **Explicit Exploration in POMDPs:** The paper's core idea of inherently integrating information gathering into policy optimization is crucial for the FPL agent. An FPL manager needs to explore different players and strategies to discover optimal long-term team compositions. This framework provides a principled way to achieve this, moving beyond simple exploration bonuses.
*   **Connection to MCMC/VI:** The use of Sequential Monte Carlo (SMC) for policy optimization aligns with the proposed MCMC and Variational Inference methods for posterior inference in the HDP-GMM. SMC is a powerful tool for Bayesian inference in dynamic systems, and its application here demonstrates its utility in policy learning.

In conclusion, this paper offers a cutting-edge approach to continuous POMDPs that directly addresses key challenges relevant to the FPL project. Its principled method for policy optimization and information gathering, coupled with its scalability to continuous spaces, makes it a valuable resource for developing a robust and intelligent FPL agent.

## Thompson Sampling for Real-Valued Combinatorial Pure Exploration of Multi-Armed Bandit

*   **Authors:** Nakamura, S., & Sugiyama, M.
*   **Year:** 2024
*   **Source:** The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)

### Key Contributions:

This paper addresses the Real-Valued Combinatorial Pure Exploration Multi-Armed Bandit (R-CPE-MAB) problem, a challenging variant where the action set can be exponentially large. The authors introduce a novel algorithm, **GenTS-Explore (Generalized Thompson Sampling Explore)**, which is designed to efficiently identify the optimal action (a combination of arms) even in such high-dimensional settings. They also provide novel problem-dependent sample complexity lower bounds and demonstrate that GenTS-Explore achieves optimal sample complexity up to a constant factor.

### Methodology:

*   **R-CPE-MAB Problem Formulation:** The problem involves `d` stochastic arms, each with an unknown mean reward. The goal is to find the optimal action `π*` from a real-valued action set `A` (which can be exponentially large) with the minimum number of arm pulls. This is a "pure exploration" setting, focusing on identification rather than cumulative reward maximization.
*   **Limitations of Existing Methods:** Traditional CPE-MAB algorithms often require enumerating all feasible actions, which becomes intractable for large action spaces. GenTS-Explore overcomes this by not requiring explicit enumeration.
*   **GenTS-Explore Algorithm:** This algorithm extends Thompson Sampling Explore to handle real-valued combinatorial actions. It leverages an "offline oracle" that can compute the optimal action for a given mean reward vector. The algorithm iteratively pulls arms based on a sampling strategy that balances exploration (to reduce uncertainty about arm rewards) and exploitation (to identify the best action).
*   **Hardness Measures and Bounds:** The paper introduces a new hardness measure, `H`, which generalizes the `G-gap` notion from previous CPE-CPE-MAB literature. This measure quantifies the difficulty of distinguishing the optimal action from suboptimal ones. The theoretical analysis shows that GenTS-Explore achieves near-optimal sample complexity, providing strong guarantees for its efficiency.
*   **Application to Knapsack Problem:** The paper demonstrates the applicability of R-CPE-MAB to the Knapsack problem, where items have unknown values and the goal is to select a subset of items to maximize total value under a weight constraint. This serves as a concrete example of a real-valued combinatorial optimization problem under uncertainty.

### Key Findings:

*   **Scalability to Large Action Spaces:** GenTS-Explore is the first algorithm capable of efficiently solving R-CPE-MAB problems with exponentially large action sets, a significant advancement over prior methods.
*   **Theoretical Optimality:** The algorithm achieves optimal sample complexity, providing strong theoretical backing for its performance.
*   **Empirical Validation:** Experiments on problems like the Knapsack problem demonstrate the practical effectiveness of GenTS-Explore.

### Relevance to Dissertation:

This paper is highly relevant to the FPL project, particularly for the **Thompson Sampling** approach and the **Knapsack Problem** formulation for team selection.

*   **Enhanced Thompson Sampling:** The preliminary research outlines Thompson Sampling as a key decision-making agent. GenTS-Explore provides a sophisticated extension that directly addresses the combinatorial nature and potentially vast action space of FPL team selection. This can significantly improve the efficiency and effectiveness of the Thompson Sampling agent.
*   **Solving the FPL Knapsack Problem:** The FPL team selection is explicitly formulated as a 0/1 Knapsack Problem. This paper's application of R-CPE-MAB to the Knapsack problem, where item values are uncertain and need to be explored, is a direct parallel. GenTS-Explore offers a robust and theoretically sound method for solving this problem in the FPL context, especially when player points (values) are unknown and need to be estimated through observation.
*   **Managing Large Action Spaces:** The number of possible FPL team combinations is enormous. The ability of GenTS-Explore to handle exponentially large action sets is crucial for making the FPL agent computationally tractable and effective in exploring the vast space of potential teams.
*   **Pure Exploration for Initial Team Selection:** While FPL is an ongoing sequential decision problem, there are instances where a "pure exploration" phase is critical, such as at the beginning of a season or after a wildcard chip. GenTS-Explore provides a principled way to efficiently identify the optimal initial team composition under uncertainty.

In conclusion, this paper offers a cutting-edge algorithm and theoretical framework that directly addresses the combinatorial challenges and large action spaces inherent in the FPL problem. It provides a strong foundation for developing a highly effective Thompson Sampling-based agent for FPL team selection, particularly when combined with the Knapsack optimization.

## Optimally Solving Simultaneous-Move Dec-POMDPs: The Sequential Central Planning Approach

*   **Authors:** Peralez, J., Delage, A., Castellini, J., Cunha, R. F., & Dibangoye, J. S.
*   **Year:** 2025
*   **Source:** The Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)

### Key Contributions:

This paper introduces a novel paradigm for solving Decentralized Partially Observable Markov Decision Processes (Dec-POMDPs) called "sequential-move centralized training for decentralized execution." This approach aims to overcome the significant scalability issues of traditional simultaneous-move Dec-POMDP solvers by transforming the problem into a sequential one. The core idea is to disentangle decision variables, allowing for a drastic reduction in computational complexity while maintaining optimality.

### Methodology:

*   **Dec-POMDP Challenges:** The paper first highlights the inherent difficulties in solving Dec-POMDPs, particularly the "silent coordination problem" (agents cannot communicate or see each other's states) and the "decision entanglement problem" (simultaneous decisions lead to double-exponential complexity in Bellman backups).
*   **Sequential-Move Central Planning:** The proposed solution involves a central planner that selects joint decision rules sequentially, one private decision rule at a time, even though agents execute actions simultaneously. This paradigm shift allows for a more tractable solution.
*   **Theoretical Properties:** The authors prove several key properties of this new approach:
    *   **Complexity Reduction:** The complexity of Bellman's backups is reduced from double exponential to polynomial.
    *   **Sufficient Statistics:** The central planner can reason about sufficient sequential-move statistics, simplifying the problem.
    *   **Value Function Properties:** ε-optimal value functions are shown to be piecewise linear and convex in these sequential-move statistics.
    *   **Single-Agent Algorithm Integration:** The framework facilitates the use of efficient single-agent reinforcement learning algorithms (like SARSA) while preserving convergence guarantees.

### Key Findings:

*   **Superior Scalability:** The sequential central planning approach significantly improves the scalability of Dec-POMDP solvers, making it feasible to tackle problems with larger numbers of agents and longer planning horizons.
*   **Empirical Outperformance:** Experiments demonstrate that the proposed `oSARSAseq` algorithm (an enhanced SARSA within this framework) outperforms existing ε-optimal simultaneous-move solvers across various multi-agent domains.
*   **Addressing Credit Assignment:** The paradigm also implicitly addresses the multi-agent credit assignment problem by breaking down the optimal value functions among agents.

### Relevance to Dissertation:

This paper is highly relevant to the FPL project, especially concerning the overall **POMDP formulation** and potential **multi-agent extensions**.

*   **Scalability of FPL POMDP:** The FPL problem, framed as a POMDP, inherently faces scalability challenges due to the large number of players and decision options. This paper provides a powerful framework for addressing such computational complexity, making the FPL agent more efficient and capable of handling a broader scope of decisions.
*   **Multi-Agent FPL Scenarios:** While the preliminary research focuses on a single FPL manager, the FPL problem can be extended to consider multiple interacting agents (e.g., multiple managers in a mini-league, or even different sub-agents managing transfers, captaincy, and chips). This paper offers a theoretically sound and empirically validated approach to solving such Dec-POMDPs, providing a strong foundation for future multi-agent FPL research.
*   **Enhancing Deep Reinforcement Learning (DDQN):** The paper demonstrates how single-agent RL algorithms can be integrated and enhanced within this sequential planning framework. This could provide valuable insights and methodologies for improving the DDQN component of the FPL agent, particularly in how it handles complex decision interdependencies and scales to larger problems.
*   **Theoretical Justification for Complex FPL:** The theoretical guarantees and complexity reductions presented in this paper provide a robust foundation for tackling more intricate FPL scenarios, ensuring that the proposed solutions are not only practical but also theoretically sound.

In conclusion, this paper offers a groundbreaking approach to solving Dec-POMDPs that directly addresses the scalability and complexity issues inherent in multi-agent decision-making under uncertainty. Its insights are invaluable for the FPL project, providing a pathway to develop more sophisticated and scalable agents, especially if multi-agent considerations are introduced or if the single-agent POMDP needs to be scaled to even larger state/action spaces.

## Thompson Sampling Based Monte-Carlo Planning in POMDPs

*   **Authors:** Bai, A., Wu, F., & Chen, X.
*   **Year:** 2014
*   **Source:** Proceedings of the Twenty-Fourth International Conference on Automated Planning and Scheduling

### Key Contributions:

This paper introduces D2NG-POMCP (Dirichlet-Dirichlet-NormalGamma based Partially Observable Monte-Carlo Planning), a novel online planning algorithm designed for solving large Partially Observable Markov Decision Processes (POMDPs). The algorithm effectively integrates Thompson Sampling (TS) with Monte-Carlo Tree Search (MCTS) to balance exploration and exploitation, particularly in scenarios with large state and action spaces. It leverages Bayesian modeling with conjugate priors to represent and update uncertainties about rewards.

### Methodology:

*   **POMDPs and MCTS:** The paper addresses the challenges of online planning in POMDPs, where the true state is unknown and decisions must be made under uncertainty. MCTS is used as a framework for building a best-first search tree to find near-optimal policies.
*   **Thompson Sampling Integration:** Thompson Sampling is incorporated into the MCTS framework to guide action selection. At each decision node in the search tree, TS is used to stochastically select actions based on their posterior probabilities of being optimal. This approach is noted for its empirical effectiveness in balancing exploration and exploitation.
*   **Bayesian Modeling with Conjugate Priors:** D2NG-POMCP employs a Bayesian approach to model uncertainties:
    *   Immediate rewards are modeled as a categorical distribution with a Dirichlet prior.
    *   Accumulated rewards are modeled as a mixture of Normal distributions with a NormalGamma prior.
    This use of conjugate priors simplifies the posterior inference process.
*   **Particle Filter for Belief State:** A particle filter is utilized to approximate and update the belief state (probability distribution over true states) at each step, which is crucial for handling partial observability.

### Key Findings:

*   **Superior Performance:** D2NG-POMCP demonstrates superior performance compared to state-of-the-art online POMDP solvers on various benchmark problems, including RockSample and PocMan.
*   **Effective Exploration-Exploitation:** The integration of Thompson Sampling within MCTS provides an effective mechanism for balancing exploration (gathering information about uncertain rewards) and exploitation (choosing actions with high expected rewards).
*   **Lower Simple Regret:** The paper observes that Thompson Sampling yields lower simple regret, especially in problems with large action spaces, which is beneficial for quickly identifying optimal actions.
*   **Technical Soundness:** The algorithm is shown to converge, providing theoretical justification for its effectiveness.

### Relevance to Dissertation:

This paper is highly relevant to the FPL project, particularly for the **Thompson Sampling** approach, the **POMDP formulation**, and the **MCMC/VI inference methods**.

*   **Advanced Thompson Sampling for FPL:** The preliminary research proposes Thompson Sampling as a decision-making agent. D2NG-POMCP provides a concrete, advanced algorithm that integrates TS into an online planning framework suitable for large POMDPs. This offers a robust blueprint for implementing the Thompson Sampling agent in the FPL context.
*   **Online Decision-Making in FPL:** FPL is an inherently online decision-making problem, requiring weekly adjustments based on new information. D2NG-POMCP's focus on online planning directly addresses this aspect, providing a framework for real-time decision support.
*   **Handling Large FPL State and Action Spaces:** The FPL problem involves a vast number of players, team configurations, and potential actions. D2NG-POMCP's demonstrated ability to solve large POMDPs is crucial for making the FPL agent computationally tractable and effective in such a complex environment.
*   **Bayesian Inference for Player Form:** The paper's use of Dirichlet and NormalGamma distributions for Bayesian modeling and inference aligns perfectly with the project's proposed HDP-GMM for modeling latent player form. This provides practical guidance on how to implement and update belief states within a Bayesian framework.
*   **Balancing FPL Objectives:** The algorithm's ability to balance cumulative and simple regrets is important for FPL, where both maximizing total points over the season (cumulative) and quickly identifying in-form players (simple) are key objectives.

In conclusion, this paper offers a sophisticated and empirically validated approach to online POMDP planning that directly enhances the Thompson Sampling component of the FPL project. Its integration of TS with MCTS and Bayesian inference provides a powerful framework for developing an intelligent FPL agent capable of navigating the complex, partially observable, and dynamic environment of Fantasy Premier League.

## Competing with Humans at Fantasy Football: Team Formation in Large Partially-Observable Domains

*   **Authors:** Matthews, T., & Ramchurn, S. D.
*   **Year:** 2012
*   **Source:** Proceedings of the Twenty-Sixth AAAI Conference on Artificial Intelligence

### Key Contributions:

This paper presents a pioneering work that models the Fantasy Premier League (FPL) team formation problem as a belief-state Markov Decision Process (MDP). It introduces an automated FPL manager that uses a Bayesian Q-learning algorithm to make sequential optimal team formation decisions under uncertainty. The paper demonstrates that this AI agent can achieve a top percentile rank when competing against millions of human players, establishing FPL as a real-world benchmark for complex sequential decision-making problems.

### Methodology:

*   **FPL Problem Formulation:** The FPL problem is described as a sequential decision-making task where a manager selects a team of 15 players each gameweek, subject to budget, positional, and club constraints, with the goal of maximizing total points over a season. The inherent challenges include an exponentially large action space (over 10^25 team combinations), uncertainty in player performance, and the need to consider long-term consequences of transfers.
*   **Belief-State MDP Modeling:** The problem is formally modeled as a belief-state MDP to account for the partial observability of player "form" and abilities. A generative belief model is used to represent and update the manager's beliefs about player characteristics (e.g., probability of starting, scoring goals, assisting) using Bayesian inference based on observed gameweek outcomes.
*   **Bayesian Q-Learning:** A Bayesian Q-learning algorithm is employed to solve the belief-state MDP. This approach incorporates uncertainty directly into the Q-value estimates and uses the concept of **Value of Perfect Information (VPI)** to guide exploration. VPI helps the agent prioritize actions that are expected to yield the most valuable information, thus balancing exploration (trying new players/strategies) and exploitation (sticking with known good performers).
*   **Action Generation via Knapsack Problem:** To handle the vast action space of team selections, the problem of generating high-quality team candidates is framed as a multi-dimensional knapsack packing problem. This allows the agent to efficiently identify optimal team compositions under various constraints.

### Key Findings:

*   **Human-Competitive Performance:** The automated FPL manager achieved a rank within the top 1.1% to 2.5% percentile among 2.5 million human players, demonstrating the effectiveness of the AI approach in a complex real-world setting.
*   **Importance of Belief Modeling and Exploration:** The study highlights that a robust belief model for player abilities and a principled exploration strategy (guided by VPI) are crucial for achieving high performance in FPL.
*   **Scalability and Tractability:** The combination of belief-state MDPs, Bayesian Q-learning, and knapsack-based action generation provides a computationally tractable solution for the FPL problem despite its large state and action spaces.

### Relevance to Dissertation:

This paper is exceptionally relevant to the FPL project, serving as a foundational work that directly addresses the core problem and validates many of the proposed methodologies.

*   **Direct FPL Application and Benchmark:** This paper provides a direct precedent for framing the FPL problem as a sequential decision-making task under partial observability. Its success in competing with human players establishes FPL as a viable and challenging real-world benchmark for the dissertation.
*   **Justification for POMDP/Belief-State MDP:** The paper strongly supports the preliminary research's choice to model the FPL problem as a Partially Observable Markov Decision Process (or belief-state MDP). It demonstrates the practical utility and effectiveness of this formulation for FPL team management.
*   **Blueprint for DDQN Agent:** The Bayesian Q-learning approach presented here offers a detailed blueprint and theoretical justification for the **Double Deep Q-Network (DDQN)** component of the FPL agent. The methods for incorporating uncertainty, using VPI for exploration, and handling large action spaces are directly transferable and can inform the design of the DDQN.
*   **Knapsack Problem for Team Optimization:** The paper's use of the multi-dimensional knapsack problem for generating optimal team selections is directly applicable to the **Optimization** section of the preliminary research, which also formulates team selection as a knapsack problem.
*   **Modeling Latent Player Form:** The generative belief model for player abilities aligns with the proposed **Hierarchical Dirichlet Process Gaussian Mixture Model (HDP-GMM)** for modeling latent player form, providing a concrete example of how such belief models are integrated into a decision-making framework.

In conclusion, this paper is a cornerstone for the FPL project, offering both a successful application and a detailed methodological framework. Its findings validate the core assumptions and proposed approaches of the dissertation, providing invaluable insights and a strong foundation for developing a competitive FPL agent.

## Comparison of Markov Chain Monte Carlo Convergence Diagnostic Tests for Bayesian Logistic Random Effect Models

*   **Authors:** Ozturk, Z., & Cengiz, M. A.
*   **Year:** 2016
*   **Source:** International Journal of Research In Medical and Health Sciences

### Key Contributions:

This paper provides a comparative study of various convergence diagnostic tests for Markov Chain Monte Carlo (MCMC) methods, specifically applied to Bayesian Logistic Random Effect Models. It addresses the critical need to rigorously assess whether MCMC chains have converged to their stationary distribution, which is essential for ensuring the validity of Bayesian inference. The paper evaluates tests such as Geweke, Raftery and Lewis, Heidelberger and Welch, Effective Sample Size (ESS), and visual trace and autocorrelation plots.

### Methodology:

*   **MCMC in Bayesian Inference:** The paper contextualizes MCMC methods (Gibbs sampling, Metropolis-Hastings) as indispensable tools for drawing samples from complex posterior distributions in Bayesian modeling when analytical solutions are intractable.
*   **Convergence Problem:** The fundamental challenge is to determine if the MCMC chain has sampled enough from the target posterior distribution to be considered representative. Non-convergence leads to unreliable parameter estimates and statistical inferences.
*   **Diagnostic Test Analysis:** The authors apply and compare a suite of established MCMC convergence diagnostics, including statistical tests and visual inspection methods. They emphasize that while some informal methods are common, formal diagnostics are necessary to confirm convergence.

### Key Findings:

*   **Multi-Diagnostic Approach:** The study concludes that relying on a single diagnostic test is insufficient, as each test evaluates different aspects of convergence. A comprehensive assessment requires the use of multiple diagnostic tools.
*   **Sensitivity of MCSE/SD:** The ratio of Monte Carlo Standard Error to Standard Deviation (MCSE/SD) was identified as one of the most sensitive indicators of convergence problems.
*   **Importance of Visual Inspection:** Trace plots and autocorrelation plots are highlighted as valuable visual tools for quickly identifying mixing issues and assessing chain behavior.
*   **Practical Guidance:** The paper provides practical insights into the application of these diagnostics, including the importance of "burn-in" (discarding initial samples) and "thinning" (reducing autocorrelation between samples) for improving chain efficiency and reliability.

### Relevance to Dissertation:

This paper is highly relevant to the FPL project, specifically for the **MCMC-based posterior inference** within the **Hierarchical Dirichlet Process Gaussian Mixture Model (HDP-GMM)**.

*   **Ensuring MCMC Reliability for HDP-GMM:** The preliminary research proposes using Gibbs Sampling (a type of MCMC) for posterior inference in the HDP-GMM to model latent player form. The reliability of these inferred player states directly depends on the convergence of the MCMC chains. This paper provides the essential knowledge and tools to ensure the MCMC process for the HDP-GMM is robust and converged, thereby validating the belief state of the POMDP.
*   **Practical Framework for MCMC Implementation:** The detailed discussion and comparison of various diagnostic tests offer a practical guide for implementing and monitoring the MCMC component. This includes understanding when a chain has converged sufficiently, how to identify mixing problems, and the impact of settings like burn-in and thinning.
*   **Validating Bayesian Inference:** By providing a clear methodology for assessing MCMC convergence, this paper strengthens the overall Bayesian inference aspect of the dissertation. It ensures that the player form parameters derived from the HDP-GMM via MCMC are statistically sound and can be confidently used by the decision-making agents.

In conclusion, this paper serves as an indispensable resource for the FPL project's MCMC implementation. It provides the critical understanding and tools necessary to ensure the robust and reliable operation of the Bayesian inference components central to modeling dynamic player form.

## Analysis of Thompson Sampling for the Multi-armed Bandit Problem

*   **Authors:** Agrawal, S., & Goyal, N.
*   **Year:** 2012
*   **Source:** JMLR: Workshop and Conference Proceedings vol 23 (2012) 39.1-39.26

### Key Contributions:

This paper provides a rigorous theoretical analysis of Thompson Sampling (TS) for the multi-armed bandit (MAB) problem, building upon previous empirical successes and addressing the lack of strong theoretical guarantees. It establishes, for the first time, logarithmic regret bounds for TS in both two-armed and N-armed stochastic bandit problems, demonstrating its asymptotic optimality. The paper also details the algorithm's adaptation for general stochastic bandits from the Bernoulli case.

### Methodology:

*   **Multi-Armed Bandit Problem:** The paper frames the problem as a sequential decision-making task where an agent chooses from `N` arms, each yielding a random real-valued reward from an unknown distribution. The objective is to maximize cumulative reward over time, balancing exploration (trying new arms) and exploitation (choosing the best-known arm).
*   **Thompson Sampling Algorithm:** TS is presented as a natural Bayesian algorithm. For Bernoulli bandits, it maintains Beta distributions as priors for the success probabilities of each arm. At each time step, it samples from these posteriors and plays the arm with the highest sampled mean. For general stochastic bandits, it adapts by performing a Bernoulli trial with the observed reward as the success probability.
*   **Regret Analysis:** The primary metric for evaluating performance is "regret," defined as the difference between the expected reward of an optimal policy and the algorithm's actual expected reward. The paper focuses on bounding this regret.
*   **Theoretical Proofs:** The core of the paper involves detailed mathematical proofs to derive finite-time logarithmic regret bounds for TS. This includes handling the complexities of the randomized nature of TS and the dependencies between arm plays in the N-armed setting.

### Key Findings:

*   **Logarithmic Regret:** The paper proves that Thompson Sampling achieves logarithmic expected regret, which is asymptotically optimal and matches the lower bound established by Lai and Robbins (1985).
*   **Asymptotic Optimality:** This theoretical result provides strong justification for the use of TS, confirming its efficiency in balancing exploration and exploitation.
*   **Robustness:** The paper implicitly supports the empirical observation that TS is robust to delayed feedback and efficient to implement.

### Relevance to Dissertation:

This paper is of fundamental importance to the FPL project, particularly for the **Thompson Sampling** approach to decision-making.

*   **Theoretical Foundation for TS:** The preliminary research proposes Thompson Sampling as a key methodology. This paper provides the rigorous theoretical backing for this choice, ensuring that the FPL agent's decision-making process is not only empirically sound but also theoretically justified with strong performance guarantees.
*   **Principled Exploration-Exploitation:** The FPL problem inherently involves an exploration-exploitation trade-off (e.g., trying out new players vs. sticking with established ones). The paper's analysis of TS demonstrates its effectiveness in managing this trade-off in a principled, Bayesian manner.
*   **Handling Uncertainty in Player Performance:** The Bayesian nature of TS, which maintains and updates posterior distributions over arm rewards, aligns perfectly with the FPL project's goal of modeling latent player form and dealing with uncertainty in player performance. This paper shows how TS can be applied to general stochastic rewards, which is relevant for FPL points.
*   **Comparison with Other Bandit Algorithms:** The paper contrasts TS with UCB algorithms, highlighting TS's empirical advantages and robustness. This context is valuable for understanding the strengths and weaknesses of different bandit strategies in the FPL context.

In conclusion, this paper is a cornerstone for the FPL project's Thompson Sampling component. Its rigorous theoretical analysis and proof of asymptotic optimality provide a strong scientific basis for employing Thompson Sampling as a core decision-making mechanism for the FPL agent, ensuring efficient and effective management of the exploration-exploitation dilemma.

## Dynamic Bayesian forecasting of English Premier League match results with the Skellam distribution

*   **Authors:** Smit, R. C., Ravazzolo, F., & Rossini, L.
*   **Year:** 2020
*   **Source:** BEMPS – Bozen Economics & Management Paper Series, NO 72/2020

### Key Contributions:

This paper introduces a dynamic Bayesian hierarchical model for forecasting English Premier League (EPL) match results. The model utilizes the Skellam distribution to model goal differences, capturing the time-varying nature of team attacking and defensive strengths. A novel aspect is the method for handling promotion and relegation, which allows for continuous use of team parameters and provides informed priors for newly promoted teams. The model is fitted using STAN and validated on real EPL data, demonstrating its predictive capabilities.

### Methodology:

*   **Skellam Distribution for Goal Difference:** The core of the model is the use of the Skellam distribution to represent the goal difference between two teams. This distribution is derived from the difference of two independent Poisson random variables, each modeling the goals scored by a team. This approach directly models the outcome of interest (goal difference) rather than individual team scores.
*   **Dynamic Bayesian Hierarchical Model:** The model incorporates several key features:
    *   **Time-Varying Team Strengths:** Team attacking (`α_i,t`) and defensive (`β_i,t`) strengths are modeled as dynamic, time-varying parameters using autoregressive processes. This allows the model to adapt to changes in team performance throughout the season.
    *   **Home Ground Advantage:** A home ground advantage parameter (`δ`) is included, acknowledging its significant impact on match outcomes.
    *   **Hierarchical Structure and Priors:** The model employs a hierarchical Bayesian structure with priors specified for the autoregressive parameters and variance components. This enables the sharing of statistical strength across teams and seasons, improving inference, especially for teams with limited data.
*   **Handling Promotion and Relegation:** A unique method is proposed where newly promoted teams are assigned the parameters of the relegated teams they replace. This provides an informed prior for promoted teams, assuming their performance will be similar to the teams they replaced, thus avoiding the need for uninformative priors.
*   **Model Fitting:** The model is implemented and fitted using STAN, a probabilistic programming language, which facilitates Bayesian inference.

### Key Findings:

*   **Accurate Match Outcome Prediction:** The model accurately predicts EPL match outcomes (win, loss, or draw) approximately 60% of the time.
*   **Dynamic Adaptation:** The time-varying nature of team strengths allows the model to adapt to changes in team performance over the season.
*   **Effective Handling of Promotion/Relegation:** The proposed method for dealing with promoted and relegated teams provides a practical solution for incorporating new teams into the model with informed priors.
*   **Shrinkage Effects:** The paper notes that hierarchical models can exhibit shrinkage, which can sometimes lead to under-prediction for top-performing teams and over-prediction for under-performing teams.

### Relevance to Dissertation:

This paper is highly relevant to the FPL project, particularly for the **Hierarchical Dirichlet Process Gaussian Mixture Model (HDP-GMM)** component and the overall **Bayesian framework** for modeling player form.

*   **Modeling Dynamic Player Form:** The paper's dynamic Bayesian hierarchical model for team strengths provides a strong methodological blueprint for modeling the latent, time-varying "form" of individual FPL players. The use of autoregressive processes to capture changes over time is directly applicable.
*   **Skellam Distribution for Player Contributions:** The Skellam distribution, used here for goal differences, could be adapted to model differences in FPL points or other player contributions, offering a robust statistical tool for player performance modeling.
*   **Handling New Players/Transfers:** The paper's innovative approach to managing promoted/relegated teams can inform strategies for incorporating new FPL players or players transferring between clubs, providing more informed initial estimates of their latent form.
*   **Bayesian Hierarchical Modeling:** The entire modeling approach aligns perfectly with the dissertation's emphasis on Bayesian hierarchical models for sharing statistical strength and inferring latent variables.
*   **Probabilistic Programming:** The use of STAN for model fitting reinforces the choice of probabilistic programming frameworks (NumPyro/Pyro) for the FPL project's inference mechanisms.

In conclusion, this paper offers valuable insights and a robust methodological framework for modeling dynamic performance in football. Its dynamic Bayesian hierarchical approach, particularly the use of the Skellam distribution and the handling of team changes, provides a strong foundation and inspiration for developing the HDP-GMM component of the FPL agent to accurately track and predict latent player form.

## Bayesian Nonparametric Multi-optima Policy Search in Reinforcement Learning

*   **Authors:** Bruno, D., Calinon, S., & Caldwell, D. G.
*   **Year:** 2013
*   **Source:** Proceedings of the Twenty-Seventh AAAI Conference on Artificial Intelligence

### Key Contributions:

This paper introduces a Bayesian nonparametric approach to multi-optima policy search within the framework of Reinforcement Learning (RL). The core idea is to use Dirichlet Processes (DP) to model a variable number of policies, allowing the system to automatically discover and represent multiple optimal solutions for a given task. This approach enhances robot adaptability by providing alternative policies that can be used when environmental changes render a primary solution unfeasible.

### Methodology:

*   **Multi-Optima Policy Search:** The paper addresses the problem of learning multiple, distinct optimal policies for a task. This is motivated by the need for robots to adapt to unpredictable environments, where having diverse solutions can prevent the need for complete relearning.
*   **Dirichlet Processes for Policy Modeling:** Dirichlet Processes are employed as a prior for a Gaussian Mixture Model (GMM). This nonparametric Bayesian approach allows the number of policies (components in the GMM) to be inferred directly from the data, rather than being fixed a priori. Each component of the GMM represents a distinct policy.
*   **Integration with RL (PoWER Algorithm):** The DP-based GMM is integrated with the Policy learning by Weighting Exploration with the Returns (PoWER) algorithm. This RL algorithm is used to update the parameters of the GMM components based on evaluated rewards, effectively guiding the search towards promising local optima in the policy parameter space.
*   **Adaptive Exploration:** The model automatically tunes the exploration covariance, leading to an adaptive exploration strategy that efficiently discovers multiple optimal policies.

### Key Findings:

*   **Automatic Discovery of Multiple Policies:** The DP-based approach successfully identifies and represents a variable number of optimal policies, which is a significant advantage over methods that assume a fixed number of solutions.
*   **Enhanced Adaptability:** By learning diverse policies, the system gains robustness and adaptability to environmental changes, as it can switch to an alternative optimal policy if needed.
*   **Competitive Performance:** Experiments on standard benchmarks (e.g., Cart-Pole, Mountain Car) demonstrate that the proposed DIPOLE (Dirichlet Process Policy Learning) algorithm achieves performance comparable to or better than single-policy RL algorithms, particularly in its ability to adapt exploration.

### Relevance to Dissertation:

This paper is highly relevant to the FPL project, particularly for the **Hierarchical Dirichlet Process Gaussian Mixture Model (HDP-GMM)** component and the **Deep Reinforcement Learning (DDQN)** component.

*   **HDP-GMM for Latent Player Form:** The paper's use of Dirichlet Processes to model a variable number of components in a GMM directly informs the design of the HDP-GMM for modeling latent player "form." The HDP, a hierarchical extension of the DP, is designed to infer the number of latent states (player forms) from data, aligning perfectly with this paper's methodology.
*   **Multi-Strategy FPL Agent:** The concept of learning multiple optimal policies is highly applicable to FPL. An FPL manager might benefit from having diverse strategies (e.g., aggressive vs. conservative transfer policies, different captaincy selection rules) that can be adapted based on game week conditions or overall league dynamics. This paper provides a framework for developing an FPL agent capable of managing such multi-optima strategies.
*   **Enhancing DDQN Exploration and Policy Search:** The insights from integrating DP-based GMMs with RL for policy search can be used to enhance the DDQN component of the FPL agent. This could lead to more adaptive exploration strategies and the ability to learn a richer set of policies for team management.
*   **Nonparametric Bayesian RL:** The paper demonstrates the power of nonparametric Bayesian methods in RL, reinforcing their utility for the FPL project, which also relies heavily on Bayesian approaches to handle uncertainty and model complexity.

In conclusion, this paper provides a valuable framework for integrating Bayesian nonparametric models with reinforcement learning to discover and manage multiple optimal policies. Its methodologies offer significant insights for the FPL project, particularly for the HDP-GMM and DDQN components, enabling the development of a more adaptive, robust, and intelligent FPL agent capable of navigating the complex and dynamic FPL environment.

## Thompson Sampling for Real-Valued Combinatorial Pure Exploration of Multi-Armed Bandit

*   **Authors:** Nakamura, S., & Sugiyama, M.
*   **Year:** 2024
*   **Source:** The Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)

### Key Contributions:

This paper introduces a novel algorithm, **GenTS-Explore (Generalized Thompson Sampling Explore)**, to solve the Real-Valued Combinatorial Pure Exploration Multi-Armed Bandit (R-CPE-MAB) problem. This setting is crucial for real-world applications where the action space is combinatorially large, often growing exponentially with the number of base items, making traditional methods that require enumerating all actions computationally infeasible. The paper's primary contribution is an algorithm that efficiently identifies the optimal combinatorial action without needing to iterate through the entire action space. Furthermore, it provides a rigorous theoretical foundation by introducing novel problem-dependent sample complexity lower bounds and proving that GenTS-Explore achieves near-optimal performance, matching these bounds up to a constant factor. This work bridges a significant gap by extending pure exploration bandit frameworks to large-scale, real-valued combinatorial domains.

### Methodology:

The R-CPE-MAB problem is formulated as a pure exploration task. Given a set of `d` arms with unknown mean rewards, the objective is to identify the optimal action `π*` from a vast, real-valued action set `A ⊆ R^d` with the minimum number of arm pulls. An "action" `π` is a vector that represents a combination of the base arms (e.g., a selection of items in a knapsack).

The GenTS-Explore algorithm operates on a principled Bayesian framework:
1.  **Bayesian Posterior:** It maintains a posterior distribution (e.g., Gaussian) for the mean reward of each of the `d` arms.
2.  **Thompson Sampling:** In each round, it draws multiple full reward vectors `θ` from the current posterior distributions of the arms. This step captures the uncertainty about the true rewards.
3.  **Oracle-Based Action Selection:** For each sampled reward vector `θ_k`, the algorithm queries an "offline oracle" to find the optimal action `π_k = Oracle(θ_k)`. This oracle is a solver for the underlying combinatorial problem (e.g., a knapsack solver) assuming `θ_k` were the true rewards. This is a key feature, as it outsources the combinatorial complexity, avoiding enumeration of `A`.
4.  **Stopping Condition:** The algorithm compares the set of optimal actions found in the sampling step `{π_k}` to the action `π_hat` that is optimal for the current *mean* of the posteriors. If all sampled actions `π_k` are identical to `π_hat`, the algorithm has converged with high confidence and terminates, returning `π_hat` as the solution.
5.  **Information-Driven Exploration:** If the algorithm does not stop, it must select the next arm to pull. It identifies the action `π_{k*}` from the sampled set that is most "confusing" or "competitive" with the current best `π_hat`. The arm to pull is then chosen based on which one provides the most information to disambiguate between `π_hat` and `π_{k*}`. The paper proposes and analyzes two selection strategies for this, with the "R-CPE-MAB" strategy being the more sophisticated.

### Key Findings:

*   **Scalability:** GenTS-Explore is the first algorithm demonstrated to be effective for R-CPE-MAB problems with exponentially large action sets, a significant leap over previous methods that required action enumeration.
*   **Theoretical Optimality:** The paper establishes that GenTS-Explore is asymptotically optimal in terms of sample complexity. Its performance is proven to be within a constant factor of the newly derived theoretical lower bound for the problem, providing strong guarantees of its efficiency.
*   **Empirical Validation:** The algorithm's effectiveness is demonstrated on the Knapsack problem, a classic NP-hard combinatorial problem. The experiments show that GenTS-Explore significantly outperforms other applicable algorithms and that its more sophisticated arm selection strategy is superior to a naive approach.

### Relevance to Dissertation:

This paper is of paramount importance to the FPL dissertation, offering a state-of-the-art, theoretically robust solution that directly addresses the core challenges of FPL team selection.

*   **Direct Application to FPL Knapsack Problem:** The dissertation formulates FPL team selection as a 0/1 Knapsack Problem, where players are "items" with unknown "values" (future points) and known "weights" (prices). GenTS-Explore is tailor-made for this scenario. It provides a principled way to solve this exact problem under the uncertainty of player performance, moving beyond simple heuristics.
*   **Advanced and Scalable Thompson Sampling Agent:** The dissertation proposes a Thompson Sampling (TS) agent. This paper provides a direct, powerful, and necessary extension of TS to the combinatorial nature of FPL. Standard TS is designed for selecting a single best arm, whereas FPL requires selecting a *portfolio* of arms (a full 15-player squad). GenTS-Explore provides the framework for this combinatorial selection. Its ability to handle the astronomically large action space of possible FPL teams is its most critical feature, ensuring the agent is computationally feasible.
*   **"Wildcard" and "Free Hit" Chip Optimization:** The FPL game includes chips like the "Wildcard" and "Free Hit," which allow for a complete team overhaul. These scenarios are perfect instances of a "pure exploration" problem: the goal is not to maximize points in the current week, but to identify the single best squad for the foreseeable future. GenTS-Explore provides a formal and optimal method for making these high-stakes decisions.
*   **Theoretical Foundation:** By incorporating this work, the dissertation can ground its methodology in rigorous, peer-reviewed theoretical analysis. The paper's concepts of "hardness measure" and "sample complexity" provide a formal language to analyze and discuss the difficulty of the FPL problem and the efficiency of the proposed agent, adding significant academic depth. It elevates the proposed agent from a collection of heuristics to a near-optimally efficient algorithm for a well-defined mathematical problem.

## Contextual Factors Impact Styles of Play in the English Premier League

*   **Authors:** Gollan, S., Bellenger, C., & Norton, K.
*   **Year:** 2020
*   **Source:** Journal of Sports Science and Medicine

### Key Contributions:

This paper provides a quantitative analysis of how situational or "contextual" factors—specifically match location, opposition quality, and competing play styles—influence the tactical "game styles" adopted by teams in the English Premier League (EPL). The study successfully clusters team behavior into three distinct, data-driven styles and demonstrates that the choice of style is not random but is systematically influenced by the match context. Crucially, it establishes a hierarchy among these factors, showing that opposition quality has a more significant impact on tactical decisions than home-field advantage. This provides a clear, evidence-based framework for understanding tactical adaptability in elite soccer.

### Methodology:

The study analyzed all 380 matches from the 2015-16 EPL season using 96 performance metrics from OptaPro. The core of the methodology involved a multi-stage process:
1.  **"Moments of Play" Framework:** Team actions were categorized into five distinct phases: established offence (EO), established defence (ED), transition into offence (TO), transition into defence (TD), and set pieces (SP). This allowed for a granular analysis of team behavior in different situations.
2.  **Game Style Clustering:** Using the performance data within these moments, the authors employed cluster analysis to identify three dominant, recurring game styles across the league:
    *   **Style 1 (Defensive):** Characterized by moderate strength in established defence but relatively poor performance in attacking and transitional phases.
    *   **Style 2 (Transitional):** Defined by dominance in the transition phases (both TO and TD), indicating a fast, counter-attacking style.
    *   **Style 3 (Attacking/Possession):** Marked by high performance in established offence and on set pieces, reflecting a more patient, possession-based attacking approach.
3.  **Contextual Analysis:** The researchers then used logistic regression and odds ratios to investigate how the probability of a team adopting one of these styles changed based on three contextual variables: **Match Venue** (Home vs. Away), **Opposition Quality** (Top 10 vs. Bottom 10 finish), and **Competing Styles** (the style adopted by the opponent).

### Key Findings:

*   **Hierarchy of Influence:** The study's most significant finding is that opposition quality is a stronger predictor of game style than match venue. While teams tend to be more attacking at home (favoring Style 3), this tendency is overridden by the strength of the opponent. Against a top-10 team, a home team is more likely to adopt a more cautious, defensive style (Style 1).
*   **Dominance of Strong Teams:** Stronger teams are better able to impose their preferred style, especially against weaker opposition. They are highly likely to play an attacking Style 3 against bottom-10 teams, regardless of whether they are home or away.
*   **Forced Adaptability of Weaker Teams:** Weaker teams are often forced to play a reactive, defensive style (Style 1) against top-tier opponents, effectively having their own tactical approach neutralized.
*   **Impact on Match Outcomes:** The adopted styles were shown to correlate with match results. Matches where both teams played defensively (Style 1 vs. Style 1) were less likely to be high-scoring. Conversely, matches where neither team played Style 1 were more likely to result in a win for one of the teams, suggesting that defensive styles are more associated with draws.

### Relevance to Dissertation:

This paper is fundamentally important for the FPL dissertation, as it provides a crucial empirical link between abstract modeling and real-world football dynamics. Its relevance lies in informing and validating the core components of the proposed POMDP and HDP-GMM framework.

*   **Empirical Validation for Latent States:** The dissertation proposes an HDP-GMM to model latent player "form." This paper provides powerful, data-driven evidence that distinct, quantifiable tactical states (the three game styles) genuinely exist in the EPL. These styles serve as a concrete, real-world analogue for the latent states the HDP-GMM is intended to discover, justifying the entire modeling approach.
*   **Informing the POMDP State Dynamics:** The FPL problem is framed as a POMDP, where an agent acts based on its belief about the partially observable state of the world (i.e., player form). This paper provides critical insights into the transition dynamics of this POMDP. It demonstrates that the "state" of a player is not static but is heavily conditioned by context. The FPL agent's belief-update mechanism must account for:
    *   **Opposition Quality:** This is the most critical variable. The agent's expectation of a player's performance should be significantly adjusted based on the strength of the upcoming opponent. A striker facing a bottom-tier defense is in a fundamentally different, higher-potential state.
    *   **Match Venue:** The home/away factor is also significant and must be included in the model that determines a player's likely performance.
*   **Justifying a Dynamic, Context-Aware Model:** This research makes a compelling case against using simple, static player averages for prediction. It proves that player and team performance is a dynamic process governed by context and tactical interactions. This directly validates the dissertation's choice of sophisticated, dynamic models (POMDPs, HDP-GMMs) that can capture this complexity, rather than simpler, static predictive algorithms.
*   **Enriching the Belief Model:** The HDP-GMM aims to infer a belief over a player's form. This paper suggests that this belief could be multi-dimensional, reflecting the tactical style the player's team is likely to adopt. For instance, the agent could maintain separate beliefs about a player's effectiveness in "possession-based attacks" versus "counter-attacks," and weigh these beliefs differently based on the upcoming match context. This would allow for much more nuanced and accurate FPL predictions.

## Identification of skill in an online game: The case of Fantasy Premier League

*   **Authors:** O'Brien, J. D., Gleeson, J. P., & O'Sullivan, D. J. P.
*   **Year:** 2021
*   **Source:** PLOS ONE

### Key Contributions:

This paper provides a large-scale, quantitative validation that Fantasy Premier League (FPL) is a game of skill rather than luck. Analyzing the performance and decisions of approximately one million managers, the authors identify a clear and persistent skill gradient. The study's main contributions are threefold: it establishes the long-term consistency of manager performance, it pinpoints the specific, repeatable behaviors that constitute "skill" in the FPL context (such as transfer and captaincy choices), and it uncovers the collective phenomenon of the "template team," where skilled managers independently converge on a common set of optimal players. This work serves as a foundational piece that scientifically legitimizes FPL as a complex decision-making problem domain suitable for AI research.

### Methodology:

The study was based on a massive dataset of FPL managers, including historical performance data across 13 seasons and detailed in-game actions for nearly one million managers during the 2018-19 season. The methodology was structured to first establish the presence of skill and then to identify its components:
1.  **Skill Persistence Analysis:** To test the skill vs. luck hypothesis, the authors calculated the Pearson correlation of managers' final ranks and total points across consecutive seasons. A consistent positive correlation would indicate that performance is not random but is a persistent trait.
2.  **Tier-Based Behavioral Analysis:** Managers from the 2018-19 season were segmented into disjoint tiers based on their final rank (e.g., top 1k, 1k-10k, 10k-100k, etc.). This allowed for a comparative analysis of decision-making between high-performing and low-performing groups.
3.  **Decision-Making Metrics:** The study quantified several key in-game actions:
    *   **Transfer Quality:** Assessed by the net points gained from transfers and by comparing the chosen player against all other valid, cheaper, or equally-priced options.
    *   **Captaincy Success:** Measured by the total points gained from the captaincy choice over the season.
    *   **Financial Management:** Tracked the growth of a manager's team value throughout the season.
    *   **Chip Strategy:** Analyzed the timing and point returns from the use of single-use chips, particularly the "Bench Boost."
4.  **Team Composition Analysis:** To investigate the "template team" phenomenon, the authors used the **Jaccard similarity index** to measure the pairwise overlap in squads between managers. They also employed co-citation matrices and hierarchical clustering to identify the core groups of players that formed the template at different stages of the season.

### Key Findings:

*   **Skill is a Dominant Factor:** The analysis revealed a consistent, statistically significant positive correlation in manager performance year-over-year. High-ranking managers tend to stay high-ranking, providing strong evidence that success is not primarily luck-driven. The study found that each additional year of experience was worth, on average, an extra 22 points.
*   **Skilled Decisions are Quantifiable:** Top-tier managers consistently and significantly outperformed lower-tier managers across all measured decision vectors. They made better transfers, picked more effective captains, were more adept at increasing their team value, and, most strikingly, showed a clear, coordinated strategy in deploying their chips to maximize returns (e.g., 79.4% of the top 10k used their Bench Boost in an optimal Double Gameweek, compared to only 28.9% of others).
*   **Emergence of a "Template Team":** The study found that the teams of high-ranking managers are more similar to each other than to the teams of lower-ranking managers. This suggests that skilled managers, likely using similar data and reasoning, converge on a common set of optimal players—a "template." This similarity is dynamic, peaking at key moments like the start of the season and around double gameweeks.

### Relevance to Dissertation:

This paper is arguably the most critical piece of literature for the FPL dissertation, as it provides the entire scientific justification for the project.

*   **Validates the Problem Domain:** The dissertation aims to build an intelligent agent for a game of skill. This paper delivers the definitive, data-driven proof that FPL is that game. It establishes FPL as a legitimate and challenging benchmark for research in AI, sequential decision-making, and computational intelligence, thereby validating the dissertation's entire premise.
*   **Provides a Blueprint for Agent Design:** The study deconstructs "skill" into a set of concrete, optimizable actions. This serves as a clear and invaluable roadmap for designing the FPL agent. The agent's architecture must include modules or logic dedicated to excelling at these specific tasks:
    *   **Transfer and Knapsack Optimization:** The agent's core decision-making component (DDQN/Thompson Sampling) must be trained to optimize the selection of players, directly addressing the "transfer quality" finding.
    *   **Captaincy and Chip Sub-routines:** The agent needs dedicated logic for the weekly captaincy pick and, crucially, a long-term strategic planner within the POMDP framework to determine the optimal timing for chip deployment.
    *   **Financial Awareness:** The agent should ideally incorporate player price changes into its state representation and reward function to learn effective financial management.
*   **Informs the Reward Function:** The dissertation's reinforcement learning agent requires a well-designed reward function. This paper shows that while final rank is the ultimate goal, intermediate metrics like net transfer points, captaincy returns, and team value growth are strong proxies for skill. These can be used as components of a shaped reward function to guide the agent's learning process more effectively.
*   **Defines the Exploration-Exploitation Landscape:** The discovery of the "template team" is a crucial insight. It defines the "meta" or the current exploitation strategy (the wisdom of the crowd). This creates a fascinating and complex exploration-exploitation dilemma for the proposed agent: should it stick to the high-performing template (exploit) or should it seek out differential players to gain a unique advantage (explore)? The agent's ability to navigate this trade-off will be a key measure of its intelligence and a central theme for the dissertation's analysis.

## Thompson Sampling based Monte-Carlo planning in POMDPs

*   **Authors:** Bai, A., Wu, F., Zhang, Z., & Chen, X.
*   **Year:** 2014
*   **Source:** Proceedings of the Twenty-Fourth International Conference on Automated Planning and Scheduling (ICAPS)

### Key Contributions:

This paper introduces **D2NG-POMCP** (Dirichlet-Dirichlet-NormalGamma based Partially Observable Monte-Carlo Planning), a sophisticated online planning algorithm designed to solve large Partially Observable Markov Decision Processes (POMDPs). The primary contribution is the novel integration of **Thompson Sampling (TS)** into the core of a Monte-Carlo Tree Search (MCTS) framework. This replaces the commonly used UCB1 heuristic, providing a more robust and efficient mechanism for balancing the exploration-exploitation trade-off. The algorithm is further distinguished by its use of a full Bayesian approach to model reward uncertainty, employing conjugate priors (Dirichlet and Normal-Gamma distributions) to make the inference process tractable. The authors demonstrate that this TS-based approach leads to faster convergence and superior performance compared to state-of-the-art online POMDP solvers on challenging benchmark problems.

### Methodology:

The algorithm is designed for online planning in POMDPs, where an agent must make decisions in real-time under uncertainty about the true state of the world. It maintains a belief over the state using a particle filter.

1.  **MCTS Framework:** The algorithm uses MCTS to build a search tree representing possible future action-observation sequences. Each node in the tree corresponds to a history of interactions, and the goal is to identify the most promising path.
2.  **Thompson Sampling for Action Selection:** This is the core innovation. Instead of using a deterministic heuristic like UCB1 to choose which action to explore within the search tree, D2NG-POMCP uses Thompson Sampling. It maintains a full posterior probability distribution over the expected long-term value (Q-value) of each action. To choose an action, it draws one sample of the Q-value from each action's posterior distribution and selects the action with the highest sampled value. This probabilistic approach naturally balances exploring uncertain but potentially high-reward actions with exploiting actions that are already known to be good.
3.  **Hierarchical Bayesian Reward Model:** To support the Thompson Sampling, the algorithm models reward uncertainty in a hierarchical Bayesian manner:
    *   Immediate (one-step) rewards are modeled as a categorical distribution with a **Dirichlet** prior.
    *   Accumulated long-term rewards (the returns from simulations) are modeled as a mixture of **Normal** distributions with a **Normal-Gamma** prior. Using these conjugate priors is crucial, as it allows for efficient, closed-form updates to the posterior distributions as new simulation results are gathered, avoiding costly numerical integration.
4.  **Online Planning Loop:** In each decision step, the algorithm runs the MCTS-with-TS process for a fixed number of iterations or a time limit, building out the search tree. It then executes the action that is currently believed to be the best, observes the outcome, updates its belief state, and repeats the process.

### Key Findings:

*   **Superior Performance on Large Problems:** The empirical results show that D2NG-POMCP consistently outperforms the standard UCB1-based POMCP algorithm on benchmark problems (RockSample, PocMan). The performance advantage becomes more pronounced as the problem size and complexity increase, highlighting the scalability of the Thompson Sampling approach.
*   **Improved Sample Efficiency:** The algorithm converges to a high-quality policy with significantly fewer simulations (tree traversals) than POMCP. This means it can "think" more efficiently and find better solutions in the same amount of time.
*   **Effectiveness of Thompson Sampling for Simple Regret:** The paper provides empirical backing for the idea that Thompson Sampling is particularly effective at minimizing *simple regret* (identifying the best single action from a set), which is often the true goal in online planning, as opposed to minimizing *cumulative regret* (the sum of missed rewards over all steps).

### Relevance to Dissertation:

This paper is of profound technical relevance to the dissertation, offering a complete, state-of-the-art algorithmic blueprint for the most ambitious parts of the proposed FPL agent.

*   **The Missing Link Between POMDP and TS:** The dissertation proposes framing FPL as a POMDP and using a Thompson Sampling agent. This paper provides the critical missing link: a concrete, tested algorithm for how to actually **use Thompson Sampling to solve a POMDP**. It provides the formal methodology for integrating these two cornerstone concepts.
*   **A Blueprint for the FPL Agent's "Brain":** The D2NG-POMCP algorithm serves as a direct, implementable model for the FPL agent's decision-making engine. The weekly task of picking a transfer and captain can be formulated as an online planning problem where the agent runs D2NG-POMCP for a set time limit to determine the best action(s), using the HDP-GMM as part of its belief model.
*   **Handles FPL's Core Challenges:** The FPL domain is a large POMDP with significant uncertainty. This paper's focus on **online planning** for **large POMDPs** using a robust method for handling **uncertainty** (Thompson Sampling) makes it perfectly suited to the problem. The algorithm's superior scalability is particularly crucial for handling the vast FPL action space.
*   **Practical Guide for Bayesian Modeling:** The paper's use of Dirichlet and Normal-Gamma priors to model reward uncertainty provides a direct, practical example of how the dissertation's HDP-GMM (which models player form) can be integrated into a value-based planning framework. The HDP-GMM provides the belief, and D2NG-POMCP shows how to plan and act based on that belief.
*   **Strong Theoretical and Empirical Justification:** Adopting this framework would ground the dissertation's agent in a state-of-the-art algorithm that has been rigorously tested and shown to outperform its predecessors. It provides a powerful combination of theoretical elegance (principled Bayesian inference) and empirical success.
