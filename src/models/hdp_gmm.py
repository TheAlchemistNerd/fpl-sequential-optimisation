"""HDP-Gaussian Mixture Model with **Gaussian-Wishart** (Normal-Wishart) base
measure.

This is the canonical Bayesian nonparametric construction of an **Infinite
Gaussian Mixture Model (IGMM)** via a Hierarchical Dirichlet Process.

The base measure is a **Normal-Wishart** distribution — the conjugate prior
for a multivariate Gaussian with *unknown mean* AND *unknown full covariance*:

    Λ_k ~ Wishart(V, ν)                     # precision matrix (full D×D)
    μ_k | Λ_k ~ N(μ₀, (κ₀ · Λ_k)⁻¹)         # mean conditioned on precision
    x | μ_k, Λ_k ~ N(μ_k, Λ_k⁻¹)            # likelihood

The **Gaussian component** (μ) models the **center** of a performance state
(e.g. mean xG/90, mean xA/90, mean ICT/90).
The **Wishart component** (Λ) models the **shape / precision** — how much a
player's stats fluctuate and, crucially, how those stats *co-vary* within a
form state (e.g. xG and xA rise together for an "Elite" state).

This is **Variant A** of the three HDP model variants documented in
``methodology.md`` §3.2.

Usage
-----
>>> from src.models.hdp_gmm import run_hdp_gmm
>>> samples, summary = run_hdp_gmm(data, player_ids, num_players)
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters for the Normal-Wishart (Gaussian-Wishart) base measure
# ---------------------------------------------------------------------------
NORMAL_WISHART_PRIOR = {
    "mu_0": 0.0,        # prior mean μ₀ (data should be standardised)
    "kappa_0": 1.0,     # prior precision scaling on the mean
    "wishart_df_factor": 1.0,  # ν = D + wishart_df_factor  (must be > D-1)
    "wishart_scale": 1.0,      # scale matrix V = scale · I_D
}

# Default truncation level (max number of mixture components)
DEFAULT_TRUNCATION = 30


# ---------------------------------------------------------------------------
# Stick-breaking helper
# ---------------------------------------------------------------------------

def stick_breaking_weights(beta: jnp.ndarray) -> jnp.ndarray:
    """Convert raw Beta-distributed stick-breaking variables to weights.

    Given ``beta`` of shape ``(..., T-1)``, returns weights of shape
    ``(..., T)`` that sum to (approximately) 1.
    """
    pad_shape = [(0, 0)] * (beta.ndim - 1) + [(0, 1)]
    beta_padded = jnp.pad(beta, pad_shape, "constant", constant_values=1.0)

    remaining = jnp.cumprod(1.0 - beta_padded, axis=-1)
    prev_remaining = jnp.pad(
        remaining[..., :-1],
        [(0, 0)] * (remaining.ndim - 1) + [(1, 0)],
        "constant",
        constant_values=1.0,
    )
    return beta_padded * prev_remaining


# ---------------------------------------------------------------------------
# NumPyro model — Gaussian-Wishart (Normal-Wishart) base measure
# ---------------------------------------------------------------------------

def hdp_gmm_model(
    data: jnp.ndarray,
    player_ids: jnp.ndarray,
    num_players: int,
    truncation: int = DEFAULT_TRUNCATION,
) -> None:
    """Hierarchical DP Gaussian Mixture Model with Gaussian-Wishart base.

    **Normal-Wishart prior** per cluster *k*:

    .. math::
        Λ_k  ~  text{Wishart}(V, ν)
        μ_k | Λ_k  ~  mathcal{N}(μ₀, (κ₀ Λ_k)^{-1})

    The **Gaussian component** μ_k defines the *center* of performance
    state *k* (mean xG, xA, ICT).  The **Wishart component** Λ_k defines
    the *precision* (inverse covariance) — the shape and correlation
    structure of that state.

    Parameters
    ----------
    data : jnp.ndarray (N, D)
        Standardised per-90 feature vectors.
    player_ids : jnp.ndarray (N,)
        Integer-encoded player identifier for each observation.
    num_players : int
        Number of unique players.
    truncation : int
        Upper bound on the number of global mixture components.
    """
    N, D = data.shape

    mu_0 = NORMAL_WISHART_PRIOR["mu_0"]
    kappa_0 = NORMAL_WISHART_PRIOR["kappa_0"]
    wishart_df = D + NORMAL_WISHART_PRIOR["wishart_df_factor"]
    wishart_scale = NORMAL_WISHART_PRIOR["wishart_scale"]
    V = wishart_scale * jnp.eye(D)

    # -- Global DP -----------------------------------------------------------
    gamma = numpyro.sample("gamma", dist.Gamma(1.0, 1.0))
    with numpyro.plate("beta_plate", truncation - 1):
        beta_raw = numpyro.sample("beta_raw", dist.Beta(1.0, gamma))

    # -- Player-specific DPs -------------------------------------------------
    alpha = numpyro.sample("alpha", dist.Gamma(1.0, 1.0))
    with numpyro.plate("player_plate", num_players):
        with numpyro.plate("pi_plate", truncation - 1):
            pi_raw = numpyro.sample("pi_raw", dist.Beta(1.0, alpha))

    # -- Gaussian-Wishart base measure (H) -----------------------------------
    # Lambda_k ~ Wishart(nu, V)          — full DxD precision matrix
    # mu_k | Lambda_k ~ N(mu_0, (kappa_0 * Lambda_k)^-1) — mean conditioned on precision
    with numpyro.plate("components", truncation):
        # Precision matrix Lambda ~ Wishart(concentration, scale_matrix)
        precision = numpyro.sample(
            "precision",
            dist.Wishart(concentration=wishart_df, scale_matrix=V),
        )  # (T, D, D)

        # Covariance for mean prior: (kappa_0 * Lambda_k)^-1 = Lambda_k^-1 / kappa_0
        cov_for_mean = jnp.linalg.inv(kappa_0 * precision)  # (T, D, D)

        mu = numpyro.sample(
            "mu",
            dist.MultivariateNormal(
                loc=jnp.full(D, mu_0),
                covariance_matrix=cov_for_mean,
            ),
        )  # (T, D)

    # -- Observation likelihood (indexed mixture) ----------------------------
    # We use numpyro.factor with explicit log_prob to support arbitrary
    # cluster assignments with full precision matrices.
    with numpyro.plate("data", N):
        pid = player_ids[jnp.arange(N)]
        w = stick_breaking_weights(pi_raw[pid])  # (N, T)
        z = numpyro.sample("z", dist.CategoricalProbs(probs=w))

        # Index parameters by assigned cluster
        obs_mu = mu[z]               # (N, D)
        obs_precision = precision[z]  # (N, D, D)
        obs_cov = jnp.linalg.inv(obs_precision + 1e-6 * jnp.eye(D))

        # Log-probability under the assigned Gaussian component
        log_prob = dist.MultivariateNormal(
            loc=obs_mu, covariance_matrix=obs_cov
        ).log_prob(data)
        numpyro.factor("obs_log_prob", log_prob)


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_hdp_gmm(
    data: np.ndarray,
    player_ids: np.ndarray,
    num_players: int,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    truncation: int = DEFAULT_TRUNCATION,
    rng_key: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run MCMC inference for the HDP-GMM.

    Parameters
    ----------
    data : np.ndarray (N, D)
        Standardised features.
    player_ids : np.ndarray (N,)
        Integer player IDs.
    num_players : int
        Number of unique players.
    num_warmup : int
        Warm-up / burn-in steps.
    num_samples : int
        Post-warm-up posterior samples.
    num_chains : int
        Parallel chains (for convergence diagnostics).
    truncation : int
        Maximum number of mixture components.
    rng_key : int
        Seed for reproducibility.

    Returns
    -------
    samples : dict[str, np.ndarray]
        Raw posterior samples from MCMC.
    summary : dict[str, Any]
        Key statistics (mean cluster assignments, concentration params, etc.).
    """
    jax_data = jnp.array(data, dtype=jnp.float32)
    jax_pids = jnp.array(player_ids, dtype=jnp.int32)

    log.info(
        "Running HDP-GMM (Gaussian-Wishart): N=%d, D=%d, players=%d, T=%d",
        len(jax_data),
        jax_data.shape[1],
        num_players,
        truncation,
    )

    kernel = NUTS(hdp_gmm_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    rng = jax.random.PRNGKey(rng_key)
    mcmc.run(
        rng,
        jax_data,
        jax_pids,
        num_players,
        truncation=truncation,
    )

    samples = mcmc.get_samples()
    mcmc.print_summary()

    # -- Derive summary statistics ------------------------------------------
    global_weights_mean = np.array(
        stick_breaking_weights(samples["beta_raw"].mean(axis=0))
    )
    active_clusters = int((global_weights_mean > 0.05).sum())

    precision_means = np.array(samples["precision"].mean(axis=0))  # (T, D, D)
    mu_means = np.array(samples["mu"].mean(axis=0))  # (T, D)

    summary = {
        "gamma_mean": float(np.array(samples["gamma"].mean())),
        "alpha_mean": float(np.array(samples["alpha"].mean())),
        "active_clusters": active_clusters,
        "global_weights": global_weights_mean,
        "mu_means": mu_means,              # (T, D) — Gaussian component
        "precision_means": precision_means,  # (T, D, D) — Wishart component
    }

    log.info(
        "Active clusters (mass > 5%%): %d | mu shape %s, Lambda shape %s",
        active_clusters,
        mu_means.shape,
        precision_means.shape,
    )

    return samples, summary
