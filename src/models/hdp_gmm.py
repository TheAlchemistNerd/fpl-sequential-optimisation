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
from scipy.special import logsumexp

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters for the Normal-Wishart (Gaussian-Wishart) base measure
# ---------------------------------------------------------------------------
NORMAL_WISHART_PRIOR = {
    "mu_0": 0.0,        # prior mean μ₀ (data should be standardised)
    "kappa_0": 1.0,     # prior precision scaling on the mean
    "wishart_df_factor": 2.0,  # ν = D + 2 gives the thesis's finite IW covariance mean
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


def _posterior_assignment_weights(
    data: np.ndarray,
    player_ids: np.ndarray,
    player_weights: np.ndarray,
    means: np.ndarray,
    precisions: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    """Estimate occupied-state mass from posterior mixture responsibilities."""
    data = np.asarray(data, dtype=float)
    player_ids = np.asarray(player_ids, dtype=int)
    player_weights = np.asarray(player_weights, dtype=float)
    means = np.asarray(means, dtype=float)
    precisions = np.asarray(precisions, dtype=float)
    observed = np.ones(len(data), dtype=bool) if mask is None else np.asarray(mask) > 0
    if not observed.any():
        raise ValueError("At least one observed row is required to summarize assignments")

    assignment_mass = np.zeros(means.shape[1], dtype=float)
    dimension = data.shape[1]
    for draw in range(len(player_weights)):
        draw_means = means[draw]
        draw_precisions = precisions[draw]
        signs, log_determinants = np.linalg.slogdet(draw_precisions)
        if np.any(signs <= 0):
            raise ValueError("HDP posterior contains a non-positive precision matrix")
        differences = data[:, None, :] - draw_means[None, :, :]
        quadratic = np.einsum(
            "nkd,kde,nke->nk", differences, draw_precisions, differences
        )
        log_density = (
            0.5 * log_determinants[None, :]
            - 0.5 * dimension * np.log(2.0 * np.pi)
            - 0.5 * quadratic
        )
        log_joint = log_density + np.log(
            np.clip(player_weights[draw, player_ids], 1e-12, None)
        )
        responsibilities = np.exp(
            log_joint - logsumexp(log_joint, axis=1, keepdims=True)
        )
        assignment_mass += responsibilities[observed].sum(axis=0)
    assignment_mass /= len(player_weights)
    return assignment_mass / assignment_mass.sum()


# ---------------------------------------------------------------------------
# NumPyro model — Gaussian-Wishart (Normal-Wishart) base measure
# ---------------------------------------------------------------------------

def hdp_gmm_model(
    data: jnp.ndarray,
    player_ids: jnp.ndarray,
    num_players: int,
    mask: jnp.ndarray | None = None,
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
    beta = stick_breaking_weights(beta_raw)
    local_concentration = jnp.maximum(alpha * beta, 1e-5)
    with numpyro.plate("player_plate", num_players):
        player_weights = numpyro.sample(
            "player_weights",
            dist.Dirichlet(local_concentration),
        )

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

    # -- Observation likelihood (marginalized mixture) -----------------------
    # Integrate the independent categorical assignments out exactly. This is
    # the same finite-mixture likelihood, without discrete sites for NUTS to
    # enumerate for every observation.
    if mask is None:
        mask = jnp.ones(N)

    with numpyro.plate("data", N):
        pid = player_ids[jnp.arange(N)]
        w = player_weights[pid]  # (N, T)
        component_log_prob = dist.MultivariateNormal(
            loc=mu, precision_matrix=precision
        ).log_prob(data[:, None, :])  # (N, T)
        mixture_log_prob = jax.scipy.special.logsumexp(
            jnp.log(jnp.clip(w, 1e-12, 1.0)) + component_log_prob,
            axis=-1,
        )
        numpyro.factor("obs_log_prob", mixture_log_prob * mask)


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
    target_accept_prob: float = 0.95,
    truncation: int = DEFAULT_TRUNCATION,
    rng_key: int = 0,
    mask: np.ndarray | None = None,
    progress_bar: bool = True,
    print_summary: bool = True,
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
    jax_mask = None if mask is None else jnp.asarray(mask, dtype=jnp.float32)
    if jax_mask is not None and jax_mask.shape != (len(jax_data),):
        raise ValueError("mask must have one value per data row")

    log.info(
        "Running HDP-GMM (Gaussian-Wishart): N=%d, D=%d, players=%d, T=%d",
        len(jax_data),
        jax_data.shape[1],
        num_players,
        truncation,
    )

    if not 0.5 < target_accept_prob < 1.0:
        raise ValueError("target_accept_prob must be between 0.5 and 1.0")
    kernel = NUTS(hdp_gmm_model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    rng = jax.random.PRNGKey(rng_key)
    mcmc.run(
        rng,
        jax_data,
        jax_pids,
        num_players,
        mask=jax_mask,
        truncation=truncation,
        extra_fields=("accept_prob",),
    )

    samples = mcmc.get_samples()
    extra_fields = mcmc.get_extra_fields()
    if print_summary:
        mcmc.print_summary()

    # -- Derive summary statistics ------------------------------------------
    global_weights_mean = np.asarray(
        stick_breaking_weights(samples["beta_raw"]).mean(axis=0)
    )
    assignment_weights = _posterior_assignment_weights(
        data,
        player_ids,
        samples["player_weights"],
        samples["mu"],
        samples["precision"],
        mask,
    )
    active_clusters = int((assignment_weights > 0.05).sum())

    precision_means = np.array(samples["precision"].mean(axis=0))  # (T, D, D)
    mu_means = np.array(samples["mu"].mean(axis=0))  # (T, D)

    summary = {
        "gamma_mean": float(np.array(samples["gamma"].mean())),
        "alpha_mean": float(np.array(samples["alpha"].mean())),
        "active_clusters": active_clusters,
        "global_weights": global_weights_mean,
        "assignment_weights": assignment_weights,
        "mu_means": mu_means,              # (T, D) — Gaussian component
        "precision_means": precision_means,  # (T, D, D) — Wishart component
        "num_divergences": int(np.asarray(extra_fields.get("diverging", [])).sum()),
        "mean_accept_prob": float(np.asarray(extra_fields.get("accept_prob", [np.nan])).mean()),
    }

    log.info(
        "Active clusters (mass > 5%%): %d | mu shape %s, Lambda shape %s",
        active_clusters,
        mu_means.shape,
        precision_means.shape,
    )

    return samples, summary
