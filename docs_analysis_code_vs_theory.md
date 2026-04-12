# Code vs. Theory: Implementation Audit & Documentation Analysis

> **Date:** 12 April 2026  
> **Scope:** Chapters 3–11 of the dissertation (docs/) versus all 31 source files (5,387 lines) and 21 test files (6,268 lines).  
> **Purpose:** Identify where the implementation has surpassed the docs, where docs need rewriting, and assess reusability of novel code.

---

## 1. Executive Summary

This project has evolved from a dissertation methodology document into a partially-implemented production system. Of the ~9 dissertation chapters covering technical content, **4 chapters have implementations that now exceed or diverge from the docs**, requiring theoretical updates. **5 chapters are still fully aligned** (no code exists yet for those chapters). **One novel contribution** — the JAX-accelerated inference kernel library — has been written with zero dissertation coverage and represents significant standalone value.

The key finding: **the code is now ahead of the docs in implementation detail, but the docs are ahead of the code in theoretical depth**. The implementation has made pragmatic choices (JAX over Numba, PuLP over custom solvers, belief-state assembly as a bridge) that the dissertation does not yet reflect.

---

## 2. Chapter-by-Chapter Audit

### Chapter 3: Data Acquisition & Feature Engineering

**Docs status:** Comprehensive and accurate (§3.1–§3.9 covers FPL API, Understat, sentiment, feature engineering, preprocessing, EDA, feature selection).

**Implementation status:** ✅ **Complete and aligned.** All 7 data modules exist (`fetcher.py`, `understat.py`, `sentiment.py`, `feature_engineering.py`, `preprocessing.py`, `feature_selection.py`, `eda.py`) with 130/130 tests passing. The implementation adds one refinement the docs don't mention: **delegation architecture** where `fetcher.py` delegates `compute_per90` to `feature_engineering.py` to avoid duplication. This is a code quality improvement, not a theoretical change.

**Docs rewriting needed:** ❌ None. Chapter 3 accurately describes the system. Minor addition: note the canonical `per90()` function location and the input validation guards in `feature_selection.py`.

---

### Chapter 4: Bayesian Nonparametrics

**Docs status:** Pure theory — CRP, DP, HDP, stick-breaking, conjugacy. No code expectations.

**Implementation status:** ✅ **Foundational, no code divergence.** This chapter provides the mathematical foundation for Chapters 5–7. The implementation (`hdp_gmm.py`, `hdp_hmm.py`) follows the stick-breaking construction and DP priors exactly as described.

**Docs rewriting needed:** ❌ None. Chapter 4 is purely theoretical and remains correct.

---

### Chapter 5: HDP-GMM with Gaussian-Wishart

**Docs status:** Describes NIW prior, conjugate posterior updates, I²GMM extension (§5.5), alternative base measures (§5.6), belief state construction (§5.7).

**Implementation status:** ⚠️ **Partially implemented, with additions not in docs.**

The `hdp_gmm.py` implementation uses NumPyro NUTS (not the collapsed Gibbs sampler described in §5.4). This is a **significant divergence**: the docs present Gibbs sampling with closed-form NIW updates, but the code uses gradient-based NUTS via automatic differentiation. Both are valid, but they produce different computational characteristics:

| Aspect | Docs (Ch. 5) | Code |
|---|---|---|
| Inference method | Collapsed Gibbs (closed-form) | NUTS (gradient-based) |
| Conjugacy exploited | Yes (NIW posterior updates) | No (black-box AD) |
| Discrete variables | Natural (categorical sampling) | Requires marginalisation |
| GPU support | Manual (Numba) | Native (JAX/XLA) |

Additionally, the **I²GMM extension** (§5.5) is partially implemented — the `nested_crp_assignment()` kernel exists in `jax_kernels.py`, but the full two-level hierarchical model is not wired into `hdp_gmm.py`. The docs describe I²GMM as addressing "within-cluster multi-modality" (e.g., "Elite at Home" vs "Elite Away"), but the kernel is currently a standalone utility.

**Belief state construction** (§5.7) is **fully implemented** in `belief_state.py` (538 lines), but goes beyond the docs: the code includes JAX-native belief vector computation (`compute_belief_vecs_jax`), the 757-dimensional state assembler (`assemble_757_state`), and the `FeatureToPointsMapper` (Ridge + GBM) — none of which are described in Chapter 5.

**Docs rewriting needed:** ✅ **Yes — Section 5.4 and §5.7.**
- §5.4 should be updated to acknowledge **NUTS as the primary inference method**, with collapsed Gibbs as an alternative (implemented via JAX kernels but not yet wired).
- §5.7 needs a new subsection on the **757-dimensional belief-state MDP bridge**, describing how belief vectors are computed from MCMC samples and assembled into the strategic state representation.
- §5.5 should note that I²GMM kernels are JIT-compiled but not yet integrated into the full model.

---

### Chapter 6: HDP-HMM Temporal Dynamics

**Docs status:** Describes HDP-HMM generative process (§6.2), FFBS algorithm (§6.3), action-conditioned transitions (§6.4), mean-field coupling (§6.5).

**Implementation status:** ⚠️ **Structurally implemented, but inference method differs.**

The `hdp_hmm.py` model correctly implements the action-conditioned transition structure (`actions: (P, T, A)` tensor, `trans_probs: (A, T, T)` transition matrices). However, like the HDP-GMM, it uses **NUTS for inference** rather than the FFBS algorithm described in §6.3. The FFBS kernels exist in `jax_kernels.py` (`ffbs_forward_pass`, `ffbs_backward_sample`, `ffbs_full`) and are fully tested (24/24 tests), but they are **not yet wired into `hdp_hmm.py`**.

The `jax_helpers.py` module provides `prepare_hmm_sequences()` which converts tabular data into padded 3D tensors `(P, T_max, D)` — this is a practical necessity for JAX-nativity that the docs don't mention.

**Mean-field coupling** (§6.5) is **not implemented**. The code has the infrastructure for action-conditioned transitions but not the population-level `m_t` coupling.

**Docs rewriting needed:** ✅ **Yes — Section 6.3 and §6.8.**
- §6.3 should describe FFBS as a **JAX-accelerated kernel** (using `jax.lax.scan`) rather than a pure algorithm. The implementation uses scan for the forward pass and backward sample, which is the modern equivalent of the recursive algorithm described.
- §6.8 (Model Specification) should be updated to note that the current NUTS implementation handles the full model jointly (transitions + emissions + stick-breaking), while FFBS is available as a separate accelerated kernel for posterior trajectory sampling.
- New subsection needed on **sequence padding and masking**: the 3D tensor conversion with dynamic-length sequences and boolean masks.

---

### Chapter 7: Posterior Inference — MCMC, Gibbs, NUTS

**Docs status:** Comprehensive coverage of Gibbs sampling (§7.2), NUTS (§7.4), Numba acceleration (§7.5), VI alternatives (§7.6), convergence diagnostics (§7.7), ASUGS (§7.8).

**Implementation status:** ⚠️ **Partially implemented, with a novel contribution not in docs.**

The NUTS configuration described in §7.4.3 matches the actual implementation (`num_warmup=500`, `num_samples=1000`, `num_chains=4`, `dense_mass=True`). However, the **Numba acceleration** described in §7.5 has been **replaced entirely by JAX/XLA**. The `jax_kernels.py` module (465 lines) provides:

1. **JIT-compiled MVN log-density** — replaces the Numba-accelerated Mahalanobis distance loop
2. **Inverse-Wishart sampling via Bartlett decomposition** — not described anywhere in the docs
3. **NIW posterior updates** — conjugate closed-form updates in JAX
4. **FFBS via `jax.lax.scan`** — replaces the Numba-accelerated forward-backward loops
5. **I²GMM nested CRP** — vmap-accelerated sub-component assignment
6. **Batched categorical sampling** — JIT-compiled

This is a **major novel contribution**: the project now has a hardware-agnostic (CPU/GPU/TPU) inference kernel library that is **not described in Chapter 7 at all**. The docs discuss Numba (§7.5) but the implementation has moved to JAX, which is a qualitatively different approach (functional programming + XLA compilation vs. imperative JIT compilation).

The **ASUGS algorithm** (§7.8) is now implemented in `drift_detector.py` (366 lines) with both `asugs_gamma_update()` and `asugs_alpha_update()`, but it is **triggered by KL-divergence drift detection** rather than the online gradient formula in §7.8. The implementation uses a simplified gradient approximation based on observed vs. expected new cluster rates, which is a practical simplification of the digamma-based gradient in the docs.

**Docs rewriting needed:** ✅ **Yes — Section 7.5 and §7.8.**
- §7.5 should be **rewritten as "JAX/XLA Acceleration"** rather than "Numba Acceleration". Describe the 6 kernel categories, the `jax.lax.scan` approach for sequential operations, and the hardware portability advantage.
- §7.8 should note that ASUGS is now **coupled with drift detection** (KL-divergence monitoring), making it event-driven rather than purely online. The gradient approximation is simplified for numerical stability.
- New subsection needed on the **Bartlett decomposition for Inverse-Wishart sampling**, which is the computational backbone of the JAX kernels.

---

### Chapter 8: Thompson Sampling

**Docs status:** Describes TS algorithm (§8.2), posterior construction from HDP belief states (§8.3), simulated world reduction (§8.4), exploration-exploitation (§8.5), GenTS-Explore (§8.6), Bayes regret bounds (§8.7), TS+MCTS (§8.8).

**Implementation status:** ⚠️ **Step A complete, Step B (MCTS) not implemented.**

The `thompson_sampling.py` module (237 lines) implements:
- **Posterior predictive sampling** via `NBPointsPosterior` (delegated to `belief_state.py`)
- **Temperature annealing** (`anneal_temperature()` with linear schedule)
- **Thompson Sampling selection** (`thompson_sampling_select()` with budget-constrained greedy selection)
- **GenTS-Explore** (`gents_explore()` with position-aware combinatorial selection)
- **Bayes regret computation** (`compute_bayes_regret()`)

All of this matches the docs. However, **TS+MCTS (§8.8) is not implemented** — there is no `ts_mcts.py`, no `mcts_node.py`, no `rollout.py`. The planning hierarchy is incomplete: Step A exists, Step B is a gap.

The implementation makes a practical choice that the docs don't mention: the Thompson Sampling selection uses a **greedy budget-feasible algorithm** rather than the MILP optimiser. The MILP solver (`milp_solver.py`, 348 lines) exists but is not yet wired into Thompson Sampling. This means the current TS implementation approximates the knapsack solution, while the docs (§8.4.1) describe the exact MILP reduction.

**Docs rewriting needed:** ✅ **Yes — Section 8.4 and new subsection needed.**
- §8.4 should clarify that the **simulated world reduction produces a greedy knapsack solution** as a first approximation, with MILP as the exact solver.
- New subsection needed on **temperature annealing schedule** (the docs mention it in §8.5.3 but don't specify the linear schedule implementation).
- §8.6 should note that GenTS-Explore uses **position-aware greedy selection** (2-5-5-3 formation) rather than full combinatorial optimisation.

---

### Chapter 9: DDQN, Bellman, MFG

**Docs status:** Describes POMDP formulation (§9.3), latent-state MDP (§9.4), DQN/DDQN (§9.5–9.6), network architecture (§9.7), Boltzmann policy (§9.8), entropy-regularised control (§9.9), Mean-Field Games (§9.10).

**Implementation status:** ❌ **Not implemented.** No DDQN, no MFG, no Boltzmann policy, no Q-network, no replay buffer. The `belief_state.py` module provides the 757-dimensional state assembler (which is the prerequisite for DDQN), but the neural network components don't exist.

**Docs rewriting needed:** ❌ None needed yet. Chapter 9 is prescriptive (describing what should be built) rather than descriptive (describing what exists). It remains an accurate blueprint.

---

### Chapter 10: Constrained Optimisation

**Docs status:** Describes 0/1 knapsack (§10.2), Dynamic Programming (§10.3), Branch-and-Bound (§10.5), MILP formulation (§10.6), captaincy as contextual MAB (§10.7), GenTS-Explore (§10.8), transfer costs (§10.9), chip deployment (§10.10).

**Implementation status:** ⚠️ **Partially implemented.**

The `milp_solver.py` module (348 lines) implements the **full MILP formulation** from §10.6 with all constraints (squad size, formation, budget, per-team, starting XI, captaincy, vice-captaincy) plus the **piecewise linear transfer cost encoding** from §10.9.2 using auxiliary variables. This is a complete and correct implementation of the docs.

However, **Dynamic Programming** (§10.3), **Branch-and-Bound** (§10.5), **Genetic Algorithms** (§10.4.3), **captaincy MAB** (§10.7), and **chip deployment** (§10.10) are **not implemented**. The knapsack baselines described as "alternative methods" in the docs don't exist as code.

**Docs rewriting needed:** ✅ **Minor — Section 10.6.**
- §10.6 should note that the MILP is solved using **PuLP with CBC solver** (open-source), not Gurobi or SCIP as the docs might imply. The 30-second time limit is specified.
- New note needed on the **auxiliary variable encoding** for transfer costs: `t_i ≥ x_i - x_i_prev`, `t_i ≥ x_i_prev - x_i`, `cost ≥ 4·(k/2 - f)`.

---

### Chapter 11: MLOps

**Docs status:** Describes data ingestion (§11.3), model training (§11.4), inference service (§11.5), monitoring and drift detection (§11.6), automated retraining (§11.7), GPU/TPU acceleration (§11.8).

**Implementation status:** ⚠️ **Partially implemented.**

The `drift_detector.py` module (366 lines) implements **KL-divergence drift detection** (§11.6.2) with per-player and population-level computations, percentile-based thresholds, and the `should_retrain()` decision function that combines drift + performance + schedule signals. The **ASUGS online updates** (§11.7.2) are implemented as `asugs_gamma_update()` and `asugs_alpha_update()`.

However, **MLflow experiment tracking**, **model registry**, **DVC data versioning**, and the **FastAPI service completion** are not implemented. The `inference_service.py` stub still returns 501 for all business logic endpoints.

**Docs rewriting needed:** ✅ **Minor — Section 11.6.**
- §11.6.2 should note that the KL-divergence implementation uses **epsilon-smoothed probabilities** (`clip(p, 1e-10, 1.0)`) for numerical stability, and renormalises after clipping.
- §11.7.2 should note that ASUGS uses a **simplified gradient approximation** (observed vs. expected new cluster rates) rather than the full digamma-based gradient, for numerical stability.

---

## 3. Novel Contributions Not in the Docs

### 3.1 JAX/XLA Inference Kernel Library

`src/utils/jax_kernels.py` (465 lines) is a **standalone, reusable library** that has no counterpart in the dissertation. It provides:

- **`mvn_log_density(x, mean, precision)`**: JIT-compiled multivariate normal log-density
- **`mvn_log_density_batch(X, mean, precision)`**: Batched version via `jax.vmap`
- **`sample_inverse_wishart(key, df, scale)`**: Bartlett decomposition for Inverse-Wishart sampling
- **`niw_posterior_update(m_0, κ_0, ν_0, Ψ_0, n_k, x̄_k, S_k)`**: Conjugate NIW closed-form updates in JAX
- **`sample_niw(key, m_0, κ_0, ν_0, Ψ_0)`**: Joint (μ, Σ) sampling from NIW
- **`ffbs_forward_pass(data, trans_probs, means, covs, pi_0)`**: Forward filtering via `jax.lax.scan`
- **`ffbs_backward_sample(key, alphas, trans_probs)`**: Backward sampling (JAX-traceable, no boolean branching)
- **`ffbs_full(key, data, trans_probs, means, covs, pi_0)`**: Complete FFBS pipeline
- **`batched_categorical_sample(key, logits)`**: JIT-compiled categorical sampling
- **`nested_crp_assignment(key, obs, existing_counts, K, η)`**: I²GMM nested CRP via `jax.vmap`

**This is publishable as a standalone library.** It is the kind of infrastructure that would be useful for any Bayesian nonparametric modelling framework, not just FPL.

### 3.2 757-Dimensional Belief-State Assembler

`assemble_757_state()` in `belief_state.py` is a novel engineering contribution that bridges probabilistic modelling and deep reinforcement learning. The docs describe the components (belief vectors, squad encoding, budget, chips) but not the concrete assembly function that produces the exact tensor.

### 3.3 Piecewise Linear Transfer Cost Encoding

The MILP implementation of transfer costs using auxiliary binary variables (`t_i`) is a correct and complete encoding of the piecewise linear function described in §10.9.2. The docs describe the mathematics but not the PuLP implementation details.

---

## 4. Library Packaging Assessment

### Can the kernels be packaged as a library?

**Yes, absolutely.** The following modules have zero FPL-specific dependencies and could be extracted into a standalone package:

| Module | FPL-specific? | Reusable? |
|---|---|---|
| `jax_kernels.py` | ❌ No | ✅ Yes — general Bayesian inference kernels |
| `jax_helpers.py` | ❌ No | ✅ Yes — sequence padding for any HMM |
| `belief_state.py` (partial) | ⚠️ Partially | ✅ `BeliefState`, `kl_divergence`, `NBPointsPosterior` are general |
| `drift_detector.py` | ❌ No | ✅ Yes — KL drift detection for any time-series model |
| `milp_solver.py` | ⚠️ Partially | ⚠️ FPL-specific constraints, but the MILP pattern is reusable |
| `thompson_sampling.py` | ⚠️ Partially | ✅ `anneal_temperature`, `compute_bayes_regret` are general |

### Proposed package structure:

```
bayesian-inference-kernels/
├── pyproject.toml
├── src/
│   └── bik/
│       ├── mvn.py           # MVN log-density (JIT)
│       ├── wishart.py        # Inverse-Wishart sampling (Bartlett)
│       ├── niw.py            # NIW posterior + sampling
│       ├── ffbs.py           # FFBS via lax.scan
│       ├── categorical.py    # Batched categorical sampling
│       ├── crp.py            # Nested CRP assignment
│       └── drift.py          # KL-divergence drift detection
└── tests/
```

This would require:
1. Creating `pyproject.toml` with dependencies (`jax`, `jaxlib`, `scipy`, `numpy`)
2. Moving `jax_kernels.py` → `bik/` subpackage
3. Moving `drift_detector.py` → `bik/drift.py`
4. Adding type stubs and documentation

The remaining FPL-specific code (`milp_solver.py`, `thompson_sampling.py`, `belief_state.py` assembly functions) stays in the main project.

---

## 5. MCTS Implementation Status

**MCTS is NOT implemented.** There is no `ts_mcts.py`, no `mcts_node.py`, no `rollout.py`, no UCB1 selection logic, no belief-state pruning. The only references to MCTS are docstring mentions in `belief_state.py` and `optimization/__init__.py`.

This is the **single largest gap** between the plan and the implementation. Chapter 8 §8.8 describes TS+MCTS as "Step B" in the planning hierarchy, and it is a critical component for multi-gameweek tactical decisions (chip timing, transfer saving, double-GW preparation). Without MCTS, the system can only make myopic (single-gameweek) decisions.

---

## 6. Recommendations

### 6.1 Docs Rewriting Priority

| Chapter | Sections to Rewrite | Priority | Estimated Words |
|---|---|---|---|
| Ch. 5 | §5.4 (NUTS vs Gibbs), §5.5 (I²GMM status), §5.7 (757-dim bridge) | High | ~800 |
| Ch. 6 | §6.3 (JAX FFBS), §6.8 (NUTS model spec), new: padding/masking | High | ~600 |
| Ch. 7 | §7.5 (JAX replaces Numba), §7.8 (ASUGS + drift coupling), new: Bartlett decomposition | High | ~700 |
| Ch. 8 | §8.4 (greedy vs MILP), new: temperature schedule, §8.6 (GenTS-Explore position-aware) | Medium | ~400 |
| Ch. 10 | §10.6 (PuLP/CBC details), new: auxiliary variable encoding | Low | ~200 |
| Ch. 11 | §11.6.2 (epsilon smoothing), §11.7.2 (simplified ASUGS gradient) | Low | ~200 |

**Total estimated rewriting:** ~2,900 words across 6 chapters.

### 6.2 Library Extraction Priority

1. **`jax_kernels.py`** → standalone package `bayesian-inference-kernels` (highest value, zero FPL coupling)
2. **`drift_detector.py`** → add to same package (general-purpose drift detection)
3. **`jax_helpers.py`** → add to same package (sequence utilities)
4. **`belief_state.py`** partial → extract `BeliefState`, `NBPointsPosterior`, `kl_divergence` as a second package `belief-state-bridge`

### 6.3 Implementation Gaps (in priority order)

1. **MCTS (Step B)** — `ts_mcts.py`, `mcts_node.py`, `rollout.py` — ~800 lines, ~30 tests
2. **DDQN (Step C)** — `ddqn/` subpackage — ~1,500 lines, ~50 tests
3. **Knapsack baselines** — `knapsack.py` (DP, B&B, Greedy) — ~300 lines, ~20 tests
4. **Captaincy MAB** — `captaincy.py` (LinUCB + Contextual TS) — ~200 lines, ~15 tests
5. **MLflow integration** — `mlops/mlflow_tracker.py` — ~300 lines, ~15 tests
6. **Mean-Field Games** — `mfg/` subpackage — ~500 lines, ~15 tests

---

## 7. Conclusion

The project has reached a significant milestone: **229 tests passing across 10 core modules**, covering the complete data pipeline, feature engineering, HDP modelling, belief-state construction, Thompson Sampling, MILP optimisation, and drift detection. The code is now ahead of the docs in several areas (JAX kernels, belief-state assembly, MILP encoding), requiring approximately 2,900 words of theoretical updating across 6 chapters.

The **JAX inference kernel library** (`jax_kernels.py`, 465 lines) is the most significant novel contribution — it is a general-purpose, hardware-agnostic Bayesian inference toolkit that could be published independently of this dissertation. The **MILP solver** (348 lines) is a complete and correct implementation of the Ch. 10 formulation, including the piecewise linear transfer cost encoding.

The **single largest remaining gap** is MCTS (Step B of the planning hierarchy), which is essential for multi-gameweek tactical decisions. Without it, the system can only make myopic decisions. The DDQN (Step C) and Mean-Field Games are also significant gaps, but they depend on MCTS being operational first (as the intermediate step between myopic and learned long-horizon planning).
