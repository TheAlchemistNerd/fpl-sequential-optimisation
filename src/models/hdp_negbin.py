"""HDP-Negative-Binomial model for non-negative integer outcomes.

This is **Variant C** documented in ``methodology.md`` §3.2. The likelihood
has support only on non-negative integers. It therefore cannot model raw FPL
``total_points`` when the target includes negative scores. Use a signed-target
regressor for a direct point-mean challenger, or specify a different
signed-support likelihood before treating points as an HDP outcome emission.

The base measure factorises as:
    p  ~ Beta(a_0, b_0)    # success probability
    r  ~ Gamma(c_0, d_0)   # dispersion / stop parameter

Usage
-----
>>> from src.models.hdp_negbin import run_hdp_negbin
>>> samples, summary = run_hdp_negbin(points, player_ids, n_players)
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from src.models.hdp_gmm import stick_breaking_weights, DEFAULT_TRUNCATION

log = logging.getLogger(__name__)

NB_PRIOR = {
    "a_0": 2.0,  # Beta prior on p (success prob)
    "b_0": 5.0,
    "c_0": 2.0,  # Gamma prior on r (dispersion)
    "d_0": 0.5,
}


def hdp_negbin_model(
    points: jnp.ndarray,
    player_ids: jnp.ndarray,
    num_players: int,
    truncation: int = DEFAULT_TRUNCATION,
) -> None:
    """HDP mixture of Negative Binomial distributions.

    Parameters
    ----------
    points : jnp.ndarray (N,)
        Non-negative integer outcome for each player-gameweek observation.
        This low-level NumPyro model assumes the support has been validated.
    player_ids : jnp.ndarray (N,)
        Integer player identifier.
    num_players : int
        Number of unique players.
    truncation : int
        Maximum number of mixture components.
    """
    N = points.shape[0]
    a_0 = NB_PRIOR["a_0"]
    b_0 = NB_PRIOR["b_0"]
    c_0 = NB_PRIOR["c_0"]
    d_0 = NB_PRIOR["d_0"]

    # -- Global DP -----------------------------------------------------------
    gamma = numpyro.sample("gamma", dist.Gamma(1.0, 1.0))

    with numpyro.plate("beta_plate", truncation - 1):
        beta_raw = numpyro.sample("beta_raw", dist.Beta(1.0, gamma))

    # -- Player-specific DPs -------------------------------------------------
    alpha = numpyro.sample("alpha", dist.Gamma(1.0, 1.0))

    with numpyro.plate("player_plate", num_players):
        with numpyro.plate("pi_plate", truncation - 1):
            pi_raw = numpyro.sample("pi_raw", dist.Beta(1.0, alpha))

    # -- NB parameters -------------------------------------------------------
    with numpyro.plate("components", truncation):
        p = numpyro.sample("p", dist.Beta(a_0, b_0))        # success prob
        r = numpyro.sample("r", dist.Gamma(c_0, d_0))       # dispersion

    # -- Observations --------------------------------------------------------
    with numpyro.plate("data", N):
        pid = player_ids[jnp.arange(N)]
        w = stick_breaking_weights(pi_raw[pid])  # (N, T)
        z = numpyro.sample("z", dist.CategoricalProbs(probs=w))

        numpyro.sample(
            "obs",
            dist.NegativeBinomial2(mean=p[z] * r[z], concentration=r[z]),
            obs=points,
        )


def _validate_nb_inputs(
    points: np.ndarray,
    player_ids: np.ndarray,
    num_players: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate count support and player-index alignment before JAX tracing."""
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must be finite, non-negative integer values.") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("points must be a non-empty one-dimensional array.")
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values != np.floor(values))
    ):
        raise ValueError(
            "points must be finite, non-negative integers; a Negative Binomial "
            "cannot represent signed raw FPL total_points."
        )

    try:
        ids = np.asarray(player_ids, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("player_ids must contain integer indices.") from exc
    if ids.ndim != 1 or ids.size != values.size:
        raise ValueError("player_ids must be one-dimensional and match points length.")
    if not np.isfinite(ids).all() or np.any(ids != np.floor(ids)):
        raise ValueError("player_ids must contain finite integer indices.")
    if not isinstance(num_players, (int, np.integer)) or num_players <= 0:
        raise ValueError("num_players must be a positive integer.")
    if np.any(ids < 0) or np.any(ids >= num_players):
        raise ValueError("player_ids must fall within [0, num_players).")

    return values.astype(np.float32), ids.astype(np.int32)


def run_hdp_negbin(
    points: np.ndarray,
    player_ids: np.ndarray,
    num_players: int,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    truncation: int = DEFAULT_TRUNCATION,
    rng_key: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run MCMC inference for the HDP-Negative-Binomial model.

    Parameters
    ----------
    points : np.ndarray (N,)
        Non-negative integer outcomes. Raw signed FPL ``total_points`` are not
        valid inputs unless an explicitly documented non-negative transform is
        applied before fitting.
    player_ids : np.ndarray (N,)
        Integer player IDs.
    num_players : int
        Number of unique players.

    Returns
    -------
    samples : dict[str, np.ndarray]
        Posterior samples.
    summary : dict
        Key statistics including per-cluster mean points and dispersion.
    """
    validated_points, validated_player_ids = _validate_nb_inputs(
        points, player_ids, num_players
    )
    jax_points = jnp.array(validated_points, dtype=jnp.float32)
    jax_pids = jnp.array(validated_player_ids, dtype=jnp.int32)

    log.info(
        "Running HDP-Negative-Binomial: N=%d, players=%d",
        len(jax_points),
        num_players,
    )

    kernel = NUTS(hdp_negbin_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    rng = jax.random.PRNGKey(rng_key)
    mcmc.run(rng, jax_points, jax_pids, num_players, truncation=truncation)

    samples = mcmc.get_samples()
    mcmc.print_summary()

    # -- Summary ------------------------------------------------------------
    global_weights_mean = np.array(
        stick_breaking_weights(samples["beta_raw"].mean(axis=0))
    )
    active_clusters = int((global_weights_mean > 0.05).sum())

    # Per-cluster expected points: mean = p * r  (NB parameterisation used)
    p_mean = np.array(samples["p"].mean(axis=0))
    r_mean = np.array(samples["r"].mean(axis=0))
    expected_points = p_mean * r_mean  # (T,)

    summary = {
        "gamma_mean": float(np.array(samples["gamma"].mean())),
        "alpha_mean": float(np.array(samples["alpha"].mean())),
        "active_clusters": active_clusters,
        "global_weights": global_weights_mean,
        "p_means": p_mean,
        "r_means": r_mean,
        "expected_points": expected_points,
    }

    log.info("Active clusters: %d", active_clusters)
    return samples, summary
