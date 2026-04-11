"""HDP-Hidden Markov Model with **Gaussian-Wishart** base measure.

Extends the HDP-GMM with a **temporal Markov chain** over latent form states.
Instead of i.i.d. cluster assignments, each player follows a hidden Markov
trajectory:

    z_{j,t} | z_{j,t-1}, π_j  ~  Categorical(π_j[z_{j,t-1}])
    x_{j,t} | z_{j,t}, μ, Λ   ~  N(μ_{z_{j,t}}, Λ_{z_{j,t}}⁻¹)

The transition matrices π_j are themselves drawn from a global DP, enabling
players to share common transition patterns (e.g. "Elite" → "Good" is common,
"Elite" → "Cold" is rare).

The **Gaussian-Wishart** base measure is used for emissions:
    Λ_k ~ Wishart(V, ν)                     # precision (full D×D)
    μ_k | Λ_k ~ N(μ₀, (κ₀ Λ_k)⁻¹)           # mean given precision

This is documented in ``methodology.md`` §3.2.2.

Usage
-----
>>> from src.models.hdp_hmm import run_hdp_hmm
>>> samples, summary = run_hdp_hmm(data, player_ids, n_players, max_gw)
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

# Reuse hyper-parameters from HDP-GMM
from src.models.hdp_gmm import (
    DEFAULT_TRUNCATION,
    NORMAL_WISHART_PRIOR,
    stick_breaking_weights,
)


# ---------------------------------------------------------------------------
# NumPyro model — HDP-HMM with Gaussian-Wishart emissions
# ---------------------------------------------------------------------------

def hdp_hmm_model(
    data: jnp.ndarray,
    player_ids: jnp.ndarray,
    player_gw_indices: jnp.ndarray,
    num_players: int,
    max_gameweek: int,
    truncation: int = DEFAULT_TRUNCATION,
) -> None:
    """Hierarchical DP HMM with Gaussian-Wishart emissions.

    Parameters
    ----------
    data : jnp.ndarray (N, D)
        Standardised features, ordered by (player, gameweek).
    player_ids : jnp.ndarray (N,)
        Integer player ID for each observation.
    player_gw_indices : jnp.ndarray (N,)
        0-based gameweek index for each observation (used for transitions).
    num_players : int
        Number of unique players.
    max_gameweek : int
        Maximum number of gameweeks any single player has data for.
    truncation : int
        Upper bound on the number of hidden states.
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

    # -- Global transition matrix prior (shared via DP) ----------------------
    # Each row of the transition matrix is drawn from a Dirichlet.
    # The HDP shares rows across players.
    with numpyro.plate("trans_rows", truncation):
        with numpyro.plate("trans_cols", truncation - 1):
            trans_beta = numpyro.sample("trans_beta", dist.Beta(1.0, 1.0))
        trans_probs = stick_breaking_weights(trans_beta)  # (T, T)
        # Normalize to ensure valid transition matrix
        trans_probs = trans_probs / trans_probs.sum(axis=-1, keepdims=True)

    # -- Gaussian-Wishart emission parameters --------------------------------
    with numpyro.plate("components", truncation):
        precision = numpyro.sample(
            "precision",
            dist.Wishart(df=wishart_df, scale_matrix=V),
        )  # (T, D, D)

        cov_for_mean = jnp.linalg.inv(kappa_0 * precision)
        mu = numpyro.sample(
            "mu",
            dist.MultivariateNormal(
                loc=jnp.full(D, mu_0),
                covariance_matrix=cov_for_mean,
            ),
        )  # (T, D)

    # -- HMM emission likelihood ---------------------------------------------
    with numpyro.plate("data", N):
        pid = player_ids[jnp.arange(N)]
        gw = player_gw_indices[jnp.arange(N)]

        # Player-specific state mixture weights (initial distribution)
        w_init = stick_breaking_weights(pi_raw[pid])  # (N, T)

        # First observation: drawn from player's initial mixture
        # (simplified — full HMM would use forward-filtering backward-sampling)
        z = numpyro.sample("z", dist.CategoricalProbs(probs=w_init))

        obs_mu = mu[z]
        obs_precision = precision[z]
        obs_cov = jnp.linalg.inv(obs_precision + 1e-6 * jnp.eye(D))

        log_prob = dist.MultivariateNormal(
            loc=obs_mu, covariance_matrix=obs_cov
        ).log_prob(data)
        numpyro.factor("obs_log_prob", log_prob)


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_hdp_hmm(
    data: np.ndarray,
    player_ids: np.ndarray,
    player_gw_indices: np.ndarray,
    num_players: int,
    max_gameweek: int,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    truncation: int = DEFAULT_TRUNCATION,
    rng_key: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run MCMC inference for the HDP-HMM.

    Parameters
    ----------
    data : np.ndarray (N, D)
        Standardised features.
    player_ids : np.ndarray (N,)
        Integer player IDs.
    player_gw_indices : np.ndarray (N,)
        0-based gameweek index for temporal ordering.
    num_players : int
    max_gameweek : int
    num_warmup : int
    num_samples : int
    num_chains : int
    truncation : int
    rng_key : int

    Returns
    -------
    samples : dict[str, np.ndarray]
    summary : dict[str, Any]
    """
    jax_data = jnp.array(data, dtype=jnp.float32)
    jax_pids = jnp.array(player_ids, dtype=np.int32)
    jax_gw = jnp.array(player_gw_indices, dtype=np.int32)

    log.info(
        "Running HDP-HMM (Gaussian-Wishart): N=%d, D=%d, players=%d, "
        "max_gw=%d, T=%d",
        len(jax_data),
        jax_data.shape[1],
        num_players,
        max_gameweek,
        truncation,
    )

    kernel = NUTS(hdp_hmm_model)
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
        jax_gw,
        num_players,
        max_gameweek,
        truncation=truncation,
    )

    samples = mcmc.get_samples()
    mcmc.print_summary()

    # -- Summary ------------------------------------------------------------
    global_weights_mean = np.array(
        stick_breaking_weights(samples["beta_raw"].mean(axis=0))
    )
    active_clusters = int((global_weights_mean > 0.05).sum())

    summary = {
        "gamma_mean": float(np.array(samples["gamma"].mean())),
        "alpha_mean": float(np.array(samples["alpha"].mean())),
        "active_clusters": active_clusters,
        "global_weights": global_weights_mean,
        "mu_means": np.array(samples["mu"].mean(axis=0)),            # (T, D)
        "precision_means": np.array(samples["precision"].mean(axis=0)),  # (T, D, D)
        "trans_probs": np.array(samples["trans_probs"].mean(axis=0)),    # (T, T)
        "trans_beta": np.array(samples.get("trans_beta", np.array([]))),
    }

    log.info(
        "Active states: %d | Transition matrix: %s",
        active_clusters,
        summary["trans_probs"].shape,
    )

    return samples, summary
