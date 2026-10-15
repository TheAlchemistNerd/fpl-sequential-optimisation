"""HDP-Hidden Markov Model with **Gaussian-Wishart** base measure.

Extends the HDP-GMM with a **temporal Markov chain** over latent form states.
Instead of i.i.d. cluster assignments, each player follows a hidden Markov
trajectory:

    z_{j,t} | z_{j,t-1}, a_{j,t-1}, Π  ~  Categorical(Π[a_{j,t-1}, z_{j,t-1}])
    x_{j,t} | z_{j,t}, μ, Λ   ~  N(μ_{z_{j,t}}, Λ_{z_{j,t}}⁻¹)

Transition rows are shared across players and drawn from the global DP base
measure. An action category can select a transition matrix; the neutral-action
research study uses one category. This module does not implement the thesis's
separate mean-field coupling extension.

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
    data: jnp.ndarray,    # (num_players, max_T, D)
    actions: jnp.ndarray, # (num_players, max_T, A)
    mask: jnp.ndarray,    # (num_players, max_T)
    truncation: int = DEFAULT_TRUNCATION,
) -> None:
    """HDP-HMM with shared-base transitions and masked Gaussian emissions."""
    P, max_T, D = data.shape
    A = actions.shape[2]

    mu_0 = NORMAL_WISHART_PRIOR["mu_0"]
    kappa_0 = NORMAL_WISHART_PRIOR["kappa_0"]
    wishart_df = D + NORMAL_WISHART_PRIOR["wishart_df_factor"]
    wishart_scale = NORMAL_WISHART_PRIOR["wishart_scale"]
    V = wishart_scale * jnp.eye(D)

    # -- Global DP -----------------------------------------------------------
    gamma = numpyro.sample("gamma", dist.Gamma(1.0, 1.0))
    alpha = numpyro.sample("alpha", dist.Gamma(1.0, 1.0))

    with numpyro.plate("beta_plate", truncation - 1):
        beta_raw = numpyro.sample("beta_raw", dist.Beta(1.0, gamma))
    beta = stick_breaking_weights(beta_raw)

    # -- Action-Conditioned HDP Transition Matrices --------------------------
    # Each transition row shares the global state weights, as in thesis Ch. 6.
    trans_probs = numpyro.sample(
        "trans_probs",
        dist.Dirichlet(alpha * beta).expand((A, truncation)).to_event(2),
    )

    # -- Gaussian-Wishart Emission Parameters --------------------------------
    with numpyro.plate("components", truncation):
        precision = numpyro.sample("precision", dist.Wishart(concentration=wishart_df, scale_matrix=V))
        cov_for_mean = jnp.linalg.inv(kappa_0 * precision)
        mu = numpyro.sample(
            "mu",
            dist.MultivariateNormal(loc=jnp.full(D, mu_0), covariance_matrix=cov_for_mean)
        )

    # -- Trajectory Likelihood (Padded & Masked Forward Pass) ----------------
    
    def log_lik_player(p_data, p_actions, p_mask):
        # Emission log-probs for all states for this player
        def compute_emissions(x):
            def log_prob_k(k):
                return dist.MultivariateNormal(loc=mu[k], precision_matrix=precision[k]).log_prob(x)
            return jax.vmap(log_prob_k)(jnp.arange(truncation))
            
        log_e = jax.vmap(compute_emissions)(p_data) # (max_T, truncation)

        # Initial distribution
        pi_0 = beta
        
        def scan_fn(alpha_prev, inputs):
            log_e_t, action_t, mask_t = inputs
            
            # Action-conditioned transition: sum_a action_t[a] * trans_probs[a, t, k]
            # We explicitly broadcast action_t to match trans_probs
            # trans_probs shape is (A, T, T)
            pi_t = (action_t.reshape(-1, 1, 1) * trans_probs).sum(axis=0)
            
            # Forward step in log space
            alpha_raw = log_e_t + jax.scipy.special.logsumexp(alpha_prev[:, None] + jnp.log(pi_t + 1e-10), axis=0)
            
            # Apply mask: if masked out, stay at alpha_prev (or neutral)
            alpha_t = jnp.where(mask_t > 0, alpha_raw, alpha_prev)
            return alpha_t, alpha_t

        init_alpha = jnp.log(pi_0 + 1e-10) + log_e[0]
        # We handle t=0 separately if needed, but here we assume max_T >= 1
        _, alphas = jax.lax.scan(scan_fn, init_alpha, (log_e[1:], p_actions[:-1], p_mask[1:]))
        
        # Log-likelihood is the logsumexp of the final REAL (not padded) alpha
        # We find the last index where mask == 1.0
        final_alpha = jnp.where(p_mask.sum() == 1, init_alpha, alphas[-1])
        return jax.scipy.special.logsumexp(final_alpha)

    # Vmap over the entire player batch (no dynamic slicing!)
    total_log_lik_vector = jax.vmap(log_lik_player)(data, actions, mask)
    numpyro.factor("trajectory_log_lik", jnp.sum(total_log_lik_vector))


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_hdp_hmm(
    data: np.ndarray,
    player_ids: np.ndarray,
    actions: np.ndarray,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    target_accept_prob: float = 0.95,
    truncation: int = DEFAULT_TRUNCATION,
    rng_key: int = 0,
    max_timesteps: int | None = None,
    progress_bar: bool = True,
    print_summary: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run MCMC inference for the upgraded HDP-HMM.

    Parameters
    ----------
    data : np.ndarray (N, D)
        Standardised features in tabular format.
    player_ids : np.ndarray (N,)
    actions : np.ndarray (N, A)
        Action vectors in tabular format.
    """
    from src.utils.jax_helpers import prepare_hmm_sequences

    # Reshape tabular data into 3D tensors for JAX-nativity
    p_data, p_actions, p_mask = prepare_hmm_sequences(
        data, player_ids, actions, max_timesteps=max_timesteps
    )
    
    jax_data = jnp.array(p_data, dtype=jnp.float32)
    jax_actions = jnp.array(p_actions, dtype=jnp.float32)
    jax_mask = jnp.array(p_mask, dtype=jnp.float32)

    log.info(
        "Running upgraded HDP-HMM (3D Tensor): players=%d, T_max=%d, D=%d, A=%d",
        jax_data.shape[0], jax_data.shape[1], jax_data.shape[2], jax_actions.shape[2]
    )

    if not 0.5 < target_accept_prob < 1.0:
        raise ValueError("target_accept_prob must be between 0.5 and 1.0")
    kernel = NUTS(hdp_hmm_model, target_accept_prob=target_accept_prob)
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
        jax_actions,
        jax_mask,
        truncation=truncation,
        extra_fields=("accept_prob",),
    )

    samples = mcmc.get_samples()
    extra_fields = mcmc.get_extra_fields()
    if print_summary:
        mcmc.print_summary()

    # -- Summary ------------------------------------------------------------
    global_weights_mean = np.asarray(
        stick_breaking_weights(samples["beta_raw"]).mean(axis=0)
    )
    active_clusters = int((global_weights_mean > 0.05).sum())

    trans_probs_mean = np.asarray(samples["trans_probs"].mean(axis=0))
    summary = {
        "gamma_mean": float(np.array(samples["gamma"].mean())),
        "alpha_mean": float(np.array(samples["alpha"].mean())),
        "active_clusters": active_clusters,
        "global_weights": global_weights_mean,
        "mu_means": np.array(samples["mu"].mean(axis=0)),            # (T, D)
        "precision_means": np.array(samples["precision"].mean(axis=0)),  # (T, D, D)
        "trans_probs_mean": trans_probs_mean,                       # (A, T, T)
        "num_divergences": int(np.asarray(extra_fields.get("diverging", [])).sum()),
        "mean_accept_prob": float(np.asarray(extra_fields.get("accept_prob", [np.nan])).mean()),
    }

    log.info(
        "Active states: %d | Action-conditioned transition shape: %s",
        active_clusters,
        summary["trans_probs_mean"].shape,
    )

    return samples, summary
