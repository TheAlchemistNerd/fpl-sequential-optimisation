"""
JAX/XLA Accelerated Inference Kernels.

JIT-compiled computational kernels for HDP-GMM and HDP-HMM inference:
    - Multivariate Normal log-density (batched, JIT)
    - Inverse-Wishart sampling via Bartlett decomposition
    - Normal-Inverse-Wishart conjugate posterior updates
    - FFBS (Forward-Filtering Backward-Sampling) via jax.lax.scan
    - Batched categorical sampling
    - I²GMM nested CRP assignments via vmap

All kernels are hardware-agnostic (CPU/GPU/TPU) via JAX's XLA compiler.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# Multivariate Normal log-density
# ---------------------------------------------------------------------------


@jax.jit
def mvn_log_density(
    x: jnp.ndarray,
    mean: jnp.ndarray,
    precision: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute log N(x | mean, precision⁻¹) using the precision matrix.

    log p(x) = -0.5 * (D·log(2π) + log|Σ| + (x-μ)ᵀ Λ (x-μ))

    where Λ = Σ⁻¹ is the precision matrix.

    Parameters
    ----------
    x : (D,)
    mean : (D,)
    precision : (D, D)

    Returns
    -------
    Scalar log-density.
    """
    D = x.shape[0]
    diff = x - mean
    # Mahalanobis distance: (x-μ)ᵀ Λ (x-μ)
    mahal = diff @ precision @ diff
    # log|Σ| = -log|Λ|
    _, log_det_prec = jnp.linalg.slogdet(precision)
    log_det_sigma = -log_det_prec

    return -0.5 * (D * jnp.log(2 * jnp.pi) + log_det_sigma + mahal)


@jax.jit
def mvn_log_density_batch(
    X: jnp.ndarray,
    mean: jnp.ndarray,
    precision: jnp.ndarray,
) -> jnp.ndarray:
    """
    Batched MVN log-density.

    Parameters
    ----------
    X : (N, D)
    mean : (D,)
    precision : (D, D)

    Returns
    -------
    (N,) log-densities.
    """
    return jax.vmap(lambda x: mvn_log_density(x, mean, precision))(X)


# ---------------------------------------------------------------------------
# Inverse-Wishart sampling via Bartlett decomposition
# ---------------------------------------------------------------------------


def sample_inverse_wishart(
    key: jnp.ndarray,
    df: float,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """
    Sample from Inverse-Wishart(df, scale).

    Uses the relationship: if W ~ Wishart(df, I), then W⁻¹ ~ IW(df, I).
    For general scale Ψ: X = L @ W⁻¹ @ L.T ~ IW(df, Ψ) where Ψ = L @ L.T.

    Parameters
    ----------
    key : JAX PRNG key
    df : float — degrees of freedom (must be > D - 1)
    scale : (D, D) — scale matrix (positive definite)

    Returns
    -------
    Sample from Inverse-Wishart(df, scale).
    """
    D = scale.shape[0]
    L_scale = jnp.linalg.cholesky(scale)

    # Bartlett: A·Aᵀ ~ Wishart(df, I)
    key, subkey = jax.random.split(key)
    A = _bartlett_decomposition(subkey, df, D)
    W = A @ A.T  # ~ Wishart(df, I)

    # W⁻¹ ~ IW(df, I), then X = L @ W⁻¹ @ L.T ~ IW(df, Ψ=LLᵀ)
    W_inv = jnp.linalg.inv(W)
    return L_scale @ W_inv @ L_scale.T


def _bartlett_decomposition(
    key: jnp.ndarray,
    df: float,
    D: int,
) -> jnp.ndarray:
    """
    Bartlett decomposition: A such that A·Aᵀ ~ Wishart(df, I_D).

    A is lower triangular with:
    - A[i,i] ~ sqrt(Chi²(df - i)) for i = 0, ..., D-1
    - A[i,j] ~ N(0, 1) for i > j
    """
    # Need D diagonal + D*(D-1)/2 off-diagonal = D*(D+1)/2 keys
    n_keys = D * (D + 1) // 2
    keys = jax.random.split(key, n_keys)

    A = jnp.zeros((D, D))
    key_idx = 0

    for i in range(D):
        # Off-diagonal (lower triangle)
        for j in range(i):
            A = A.at[i, j].set(jax.random.normal(keys[key_idx]))
            key_idx += 1
        # Diagonal: sqrt of Chi-squared
        chi2 = jax.random.gamma(keys[key_idx], (df - i) / 2.0) * 2.0
        A = A.at[i, i].set(jnp.sqrt(jnp.maximum(chi2, 1e-10)))
        key_idx += 1

    return A


# ---------------------------------------------------------------------------
# Normal-Inverse-Wishart posterior updates
# ---------------------------------------------------------------------------


def niw_posterior_update(
    m_0: jnp.ndarray,
    kappa_0: float,
    nu_0: float,
    Psi_0: jnp.ndarray,
    n_k: int,
    x_bar: jnp.ndarray,
    S_k: jnp.ndarray,
) -> tuple[jnp.ndarray, float, float, jnp.ndarray]:
    """
    Conjugate NIW posterior update.

    Given prior NIW(m_0, κ_0, ν_0, Ψ_0) and data with sufficient statistics
    (n_k, x̄_k, S_k), compute posterior NIW(m_k, κ_k, ν_k, Ψ_k).

    Parameters
    ----------
    m_0 : (D,) — prior mean
    kappa_0 : float — prior precision on mean
    nu_0 : float — prior degrees of freedom
    Psi_0 : (D, D) — prior scale matrix
    n_k : int — number of observations in cluster
    x_bar : (D,) — sample mean
    S_k : (D, D) — scatter matrix Σ(x - x̄)(x - x̄)ᵀ

    Returns
    -------
    (m_k, kappa_k, nu_k, Psi_k)
    """
    kappa_k = kappa_0 + n_k
    m_k = (kappa_0 * m_0 + n_k * x_bar) / kappa_k
    nu_k = nu_0 + n_k
    diff = x_bar - m_0
    Psi_k = Psi_0 + S_k + (kappa_0 * n_k / kappa_k) * jnp.outer(diff, diff)
    return m_k, kappa_k, nu_k, Psi_k


# ---------------------------------------------------------------------------
# NIW sampling
# ---------------------------------------------------------------------------


def sample_niw(
    key: jnp.ndarray,
    m_0: jnp.ndarray,
    kappa_0: float,
    nu_0: float,
    Psi_0: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample (μ, Σ) from NIW(m_0, κ_0, ν_0, Ψ_0).

    Σ ~ Inverse-Wishart(ν_0, Ψ_0)
    μ | Σ ~ N(m_0, Σ / κ_0)

    Parameters
    ----------
    key : JAX PRNG key
    m_0 : (D,)
    kappa_0 : float
    nu_0 : float
    Psi_0 : (D, D)

    Returns
    -------
    (mu, Sigma)
    """
    D = m_0.shape[0]
    key_sigma, key_mu = jax.random.split(key)

    # Sample Σ from Inverse-Wishart
    Sigma = sample_inverse_wishart(key_sigma, nu_0, Psi_0)

    # Sample μ | Σ ~ N(m_0, Σ / κ_0)
    cov_mu = Sigma / kappa_0
    mu = jax.random.multivariate_normal(key_mu, m_0, cov_mu)

    return mu, Sigma


# ---------------------------------------------------------------------------
# FFBS: Forward pass
# ---------------------------------------------------------------------------


def ffbs_forward_pass(
    data: jnp.ndarray,
    trans_probs: jnp.ndarray,
    means: jnp.ndarray,
    covs: jnp.ndarray,
    pi_0: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward filtering for HDP-HMM.

    Computes filtered state probabilities α_t(k) = P(z_t = k | x_{1:t}).

    Parameters
    ----------
    data : (T, D)
    trans_probs : (K, K) — transition matrix
    means : (K, D) — emission means
    covs : (K, D, D) — emission covariances
    pi_0 : (K,) — initial distribution

    Returns
    -------
    alphas : (T, K) — unnormalized log forward variables
    log_norms : (T,) — normalising constants at each step
    """
    T, D = data.shape
    K = means.shape[0]

    def compute_emissions(x):
        def log_prob_k(k):
            prec = jnp.linalg.inv(covs[k])
            return mvn_log_density(x, means[k], prec)
        return jax.vmap(log_prob_k)(jnp.arange(K))

    log_e = jax.vmap(compute_emissions)(data)  # (T, K)

    def scan_fn(alpha_prev, inputs):
        log_e_t = inputs
        # Predict step: α_t(ℓ) = Σ_k α_{t-1}(k) · π_{k,ℓ}
        log_pred = jax.scipy.special.logsumexp(
            alpha_prev[:, None] + jnp.log(trans_probs + 1e-10),
            axis=0,
        )
        # Update step: multiply by emission
        alpha_raw = log_e_t + log_pred
        # Normalise (for numerical stability)
        log_norm = jax.scipy.special.logsumexp(alpha_raw)
        alpha_norm = alpha_raw - log_norm
        return alpha_norm, (alpha_norm, log_norm)

    # Initial step
    init_alpha = jnp.log(pi_0 + 1e-10) + log_e[0]
    log_norm_0 = jax.scipy.special.logsumexp(init_alpha)
    init_alpha_norm = init_alpha - log_norm_0

    # Scan over t = 1, ..., T-1
    _, (alphas_norm, log_norms_rest) = lax.scan(
        scan_fn, init_alpha_norm, log_e[1:]
    )

    # Combine
    alphas_norm_full = jnp.vstack([init_alpha_norm.reshape(1, -1), alphas_norm])
    log_norms_full = jnp.concatenate([jnp.array([log_norm_0]), log_norms_rest])

    return alphas_norm_full, log_norms_full


# ---------------------------------------------------------------------------
# FFBS: Backward sample
# ---------------------------------------------------------------------------


def ffbs_backward_sample(
    key: jnp.ndarray,
    alphas: jnp.ndarray,
    trans_probs: jnp.ndarray,
) -> jnp.ndarray:
    """
    Backward sampling for HDP-HMM.

    Sample z_T ~ Categorical(α_T), then backwards:
    P(z_t = k | z_{t+1} = ℓ, x_{1:t}) ∝ α_t(k) · π_{k,ℓ}

    Parameters
    ----------
    key : JAX PRNG key
    alphas : (T, K) — normalised log forward variables
    trans_probs : (K, K)

    Returns
    -------
    states : (T,) — sampled state trajectory
    """
    T, K = alphas.shape

    # Sample z_T from alphas[T-1]
    key, subkey = jax.random.split(key)
    z_T = jax.random.categorical(subkey, alphas[T - 1])

    def backward_step(key_and_z_next, t):
        key, z_next = key_and_z_next
        # P(z_t = k | z_{t+1}) ∝ α_t(k) · π_{k, z_{t+1}}
        log_probs = alphas[t] + jnp.log(trans_probs[:, z_next] + 1e-10)
        key, subkey = jax.random.split(key)
        z_t = jax.random.categorical(subkey, log_probs)
        return (key, z_t), z_t

    # Scan backwards from T-2 to 0
    init = (key, z_T)
    ts = jnp.arange(T - 2, -1, -1)
    _, states_reversed = lax.scan(backward_step, init, ts)

    # Prepend z_T and reverse to get forward order
    return jnp.concatenate([jnp.array([z_T]), states_reversed[::-1]])


def ffbs_full(
    key: jnp.ndarray,
    data: jnp.ndarray,
    trans_probs: jnp.ndarray,
    means: jnp.ndarray,
    covs: jnp.ndarray,
    pi_0: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Full FFBS: forward filter + backward sample.

    Parameters
    ----------
    key : JAX PRNG key
    data : (T, D)
    trans_probs : (K, K)
    means : (K, D)
    covs : (K, D, D)
    pi_0 : (K,)

    Returns
    -------
    states : (T,) — sampled state trajectory
    log_likelihood : scalar — log P(data | params)
    """
    alphas, log_norms = ffbs_forward_pass(data, trans_probs, means, covs, pi_0)
    log_likelihood = jnp.sum(log_norms)

    key, subkey = jax.random.split(key)
    states = ffbs_backward_sample(subkey, alphas, trans_probs)

    return states, log_likelihood


# ---------------------------------------------------------------------------
# Batched categorical sampling
# ---------------------------------------------------------------------------


@jax.jit
def batched_categorical_sample(
    key: jnp.ndarray,
    logits: jnp.ndarray,
) -> jnp.ndarray:
    """
    Sample from categorical distributions given logits.

    Parameters
    ----------
    key : JAX PRNG key
    logits : (N, K) or (K,) — unnormalised log-probabilities

    Returns
    -------
    (N,) or scalar sample indices.
    """
    return jax.random.categorical(key, logits)


# ---------------------------------------------------------------------------
# I²GMM nested CRP assignment
# ---------------------------------------------------------------------------


def nested_crp_assignment(
    key: jnp.ndarray,
    obs_in_cluster: jnp.ndarray,
    existing_sub_counts: jnp.ndarray,
    K: int,
    eta: float,
) -> jnp.ndarray:
    """
    Sample sub-component assignments for observations in a top-level cluster
    under the I²GMM nested CRP.

    For each observation i, the probability of joining sub-component w is:
        P(w_i = w | ...) ∝ n_w  (if w exists)
        P(w_i = w_new | ...) ∝ η  (new sub-component)

    Parameters
    ----------
    key : JAX PRNG key
    obs_in_cluster : (n,) — dummy indices (not used directly, just counts n)
    existing_sub_counts : (M,) — counts of existing sub-components
    K : int — number of top-level clusters (used for sizing)
    eta : float — sub-component concentration parameter

    Returns
    -------
    sub_assignments : (n,) — sub-component index for each observation
    """
    n = obs_in_cluster.shape[0]
    M = existing_sub_counts.shape[0]

    def assign_one(key_i, i):
        """Assign observation i to a sub-component."""
        # Probabilities: proportional to existing counts + eta for new
        probs = jnp.concatenate([
            existing_sub_counts.astype(jnp.float32),
            jnp.array([eta]),
        ])
        probs = probs / probs.sum()
        return jax.random.categorical(key_i, jnp.log(probs + 1e-10))

    keys = jax.random.split(key, n)
    return jax.vmap(assign_one)(keys, jnp.arange(n))
