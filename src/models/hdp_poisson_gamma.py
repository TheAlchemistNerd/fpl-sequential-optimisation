"""HDP-Poisson-Gamma model for discrete FPL events.

Models count-based performance metrics (goals, assists, clean sheets,
saves, bonus points) using a Poisson likelihood with a Gamma-distributed
base measure.  This is **Variant B** documented in ``methodology.md`` §3.2.

The model discovers shared "event-rate profiles" across players.  For
example, one cluster might represent a high goal-rate state (λ ≈ 0.5 per
game), another a low rate (λ ≈ 0.05).

Usage
-----
>>> from src.models.hdp_poisson_gamma import run_hdp_poisson_gamma
>>> samples, summary = run_hdp_poisson_gamma(count_data, player_ids, n_players)
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from src.models.hdp_gmm import stick_breaking_weights

log = logging.getLogger(__name__)

DEFAULT_TRUNCATION = 30

# Gamma prior hyper-parameters: Gamma(alpha_0, beta_0)
GAMMA_PRIOR = {"alpha_0": 1.0, "beta_0": 1.0}


def hdp_poisson_gamma_model(
    counts: jnp.ndarray,
    player_ids: jnp.ndarray,
    num_players: int,
    truncation: int = DEFAULT_TRUNCATION,
) -> None:
    """HDP mixture of Poisson distributions with Gamma base measure.

    Parameters
    ----------
    counts : jnp.ndarray (N, E)
        Non-negative integer counts for each event type (goals, assists,
        clean sheets, etc.).  Each column is modelled with its own rate.
    player_ids : jnp.ndarray (N,)
        Integer player identifier.
    num_players : int
        Number of unique players.
    truncation : int
        Maximum number of mixture components.
    """
    N, E = counts.shape
    alpha_0 = GAMMA_PRIOR["alpha_0"]
    beta_0 = GAMMA_PRIOR["beta_0"]

    # -- Global DP -----------------------------------------------------------
    gamma = numpyro.sample("gamma", dist.Gamma(1.0, 1.0))

    with numpyro.plate("beta_plate", truncation - 1):
        beta_raw = numpyro.sample("beta_raw", dist.Beta(1.0, gamma))

    # -- Player-specific DPs -------------------------------------------------
    alpha = numpyro.sample("alpha", dist.Gamma(1.0, 1.0))

    with numpyro.plate("player_plate", num_players):
        with numpyro.plate("pi_plate", truncation - 1):
            pi_raw = numpyro.sample("pi_raw", dist.Beta(1.0, alpha))

    # -- Poisson rate parameters (Gamma base measure) ------------------------
    with numpyro.plate("components", truncation):
        with numpyro.plate("events", E):
            lam = numpyro.sample(
                "lambda",
                dist.Gamma(alpha_0, beta_0),
            )  # (T, E)

    # -- Observations --------------------------------------------------------
    with numpyro.plate("data", N):
        pid = player_ids[jnp.arange(N)]
        w = stick_breaking_weights(pi_raw[pid])  # (N, T)
        z = numpyro.sample("z", dist.CategoricalProbs(probs=w))

        # Poisson likelihood for each event type
        # We model each event type independently given the cluster
        for e in range(E):
            numpyro.sample(
                f"obs_event_{e}",
                dist.Poisson(lam[z, e]),
                obs=counts[jnp.arange(N), e],
            )


def run_hdp_poisson_gamma(
    counts: np.ndarray,
    player_ids: np.ndarray,
    num_players: int,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    truncation: int = DEFAULT_TRUNCATION,
    rng_key: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run MCMC inference for the HDP-Poisson-Gamma model.

    Parameters
    ----------
    counts : np.ndarray (N, E)
        Non-negative integer count data.
    player_ids : np.ndarray (N,)
        Integer player IDs.
    num_players : int
        Number of unique players.

    Returns
    -------
    samples : dict[str, np.ndarray]
        Posterior samples.
    summary : dict
        Key statistics including per-cluster event rates.
    """
    jax_counts = jnp.array(counts, dtype=jnp.float32)
    jax_pids = jnp.array(player_ids, dtype=jnp.int32)

    log.info(
        "Running HDP-Poisson-Gamma: N=%d, events=%d, players=%d",
        len(jax_counts),
        jax_counts.shape[1],
        num_players,
    )

    kernel = NUTS(hdp_poisson_gamma_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    rng = jax.random.PRNGKey(rng_key)
    mcmc.run(rng, jax_counts, jax_pids, num_players, truncation=truncation)

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
        "lambda_means": np.array(samples["lambda"].mean(axis=0)),  # (T, E)
    }

    log.info("Active clusters: %d", active_clusters)
    return samples, summary
