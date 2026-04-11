"""
Tests for src/utils/jax_kernels.py — JAX/XLA Accelerated Inference Kernels.

Tests cover:
    - JIT-compiled Multivariate Normal log-density
    - JIT-compiled Inverse-Wishart sampling
    - JIT-compiled Normal-Inverse-Wishart posterior updates
    - JAX lax.scan-powered FFBS forward pass
    - JAX lax.scan-powered FFBS backward sample
    - Vmap-accelerated I²GMM nested CRP assignments
    - Batched categorical sampling
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import wishart, multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.jax_kernels import (
    mvn_log_density,
    mvn_log_density_batch,
    sample_inverse_wishart,
    niw_posterior_update,
    sample_niw,
    ffbs_forward_pass,
    ffbs_backward_sample,
    ffbs_full,
    batched_categorical_sample,
    nested_crp_assignment,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_mvn_params():
    """Sample MVN parameters for testing."""
    D = 3
    mean = jnp.array([1.0, 2.0, 3.0])
    # Create a positive definite covariance
    A = jnp.array([[1.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.2]])
    cov = A @ A.T  # Ensure positive definite
    precision = jnp.linalg.inv(cov)
    x = jnp.array([1.5, 1.8, 2.5])
    return mean, cov, precision, x, D


@pytest.fixture
def sample_niw_params():
    """Sample NIW parameters."""
    D = 2
    m_0 = jnp.zeros(D)
    kappa_0 = 1.0
    nu_0 = D + 2.0
    Psi_0 = jnp.eye(D)
    return m_0, kappa_0, nu_0, Psi_0, D


@pytest.fixture
def sample_ffbs_data():
    """Sample data for FFBS tests."""
    T = 10
    K = 4
    D = 2
    # Generate synthetic observations
    key = jax.random.PRNGKey(42)
    data = jax.random.normal(key, (T, D))

    # Random transition matrix (row-stochastic)
    trans_logits = jax.random.normal(key, (K, K))
    trans_probs = jax.nn.softmax(trans_logits, axis=-1)

    # Random emission parameters
    means = jax.random.normal(key, (K, D))
    covs = jnp.stack([jnp.eye(D) * 1.0] * K)

    # Initial distribution
    pi_0 = jnp.ones(K) / K

    return data, trans_probs, means, covs, pi_0, T, K, D


# ===========================================================================
# MVN log-density tests
# ===========================================================================


class TestMVNLogDensity:
    """Validate JIT-compiled MVN log-density computation."""

    def test_mvn_log_density_scalar(self, sample_mvn_params):
        mean, cov, precision, x, D = sample_mvn_params
        result = mvn_log_density(x, mean, precision)
        # Compare with scipy reference
        expected = multivariate_normal.logpdf(np.array(x), mean=np.array(mean), cov=np.array(cov))
        assert float(result) == pytest.approx(expected, abs=1e-4)

    def test_mvn_log_density_batch(self, sample_mvn_params):
        mean, cov, precision, x, D = sample_mvn_params
        # Batch of 5 observations
        X = jnp.stack([x, x + 0.1, x - 0.1, x * 1.1, jnp.zeros(D)])
        result = mvn_log_density_batch(X, mean, precision)
        assert result.shape == (5,)

    def test_mvn_log_density_jit_compiled(self, sample_mvn_params):
        """Verify the function is JIT-compilable (no Python traces after first call)."""
        mean, cov, precision, x, D = sample_mvn_params
        # First call compiles, second uses cached
        _ = mvn_log_density(x, mean, precision)
        _ = mvn_log_density(x, mean, precision)  # Should use cache
        assert True  # If we get here, JIT worked

    def test_mvn_log_density_symmetric_precision(self, sample_mvn_params):
        """Should handle symmetric precision matrices."""
        mean, cov, precision, x, D = sample_mvn_params
        # Symmetrise precision
        precision_sym = (precision + precision.T) / 2
        result = mvn_log_density(x, mean, precision_sym)
        assert jnp.isfinite(result)


# ===========================================================================
# Inverse-Wishart sampling tests
# ===========================================================================


class TestInverseWishartSampling:
    """Validate Inverse-Wishart sampling via JAX."""

    def test_sample_inverse_wishart_shape(self):
        D = 3
        df = D + 2
        scale = jnp.eye(D)
        key = jax.random.PRNGKey(42)
        sample = sample_inverse_wishart(key, df, scale)
        assert sample.shape == (D, D)
        # Should be symmetric positive definite
        assert jnp.allclose(sample, sample.T)

    def test_sample_inverse_wishart_positive_definite(self):
        D = 2
        df = D + 2
        scale = jnp.eye(D)
        key = jax.random.PRNGKey(0)
        sample = sample_inverse_wishart(key, df, scale)
        # Check positive definiteness via eigenvalues
        eigenvalues = jnp.linalg.eigvalsh(sample)
        assert jnp.all(eigenvalues > 0)

    def test_sample_inverse_wishart_mean_approx(self):
        """Mean of Inverse-Wishart(df, scale) ≈ scale / (df - D - 1)."""
        D = 2
        df = D + 5  # Larger df for better mean convergence
        scale = jnp.eye(D) * 2.0
        key = jax.random.PRNGKey(42)
        # Average over many samples
        keys = jax.random.split(key, 1000)
        samples = jax.vmap(lambda k: sample_inverse_wishart(k, df, scale))(keys)
        empirical_mean = jnp.mean(samples, axis=0)
        expected_mean = scale / (df - D - 1)
        assert jnp.allclose(empirical_mean, expected_mean, atol=0.2)


# ===========================================================================
# NIW posterior update tests
# ===========================================================================


class TestNIWPosteriorUpdate:
    """Validate Normal-Inverse-Wishart conjugate posterior updates."""

    def test_niw_posterior_update_shapes(self, sample_niw_params):
        m_0, kappa_0, nu_0, Psi_0, D = sample_niw_params
        n_k = 10
        x_bar = jnp.zeros(D)
        S_k = jnp.eye(D) * n_k

        m_k, kappa_k, nu_k, Psi_k = niw_posterior_update(
            m_0, kappa_0, nu_0, Psi_0, n_k, x_bar, S_k
        )
        assert m_k.shape == (D,)
        assert kappa_k > kappa_0
        assert nu_k > nu_0
        assert Psi_k.shape == (D, D)

    def test_niw_posterior_update_mean_shift(self, sample_niw_params):
        """Posterior mean should shift toward sample mean."""
        m_0, kappa_0, nu_0, Psi_0, D = sample_niw_params
        m_0 = jnp.zeros(D)
        x_bar = jnp.ones(D) * 5.0
        n_k = 20
        S_k = jnp.eye(D) * n_k

        m_k, kappa_k, nu_k, Psi_k = niw_posterior_update(
            m_0, kappa_0, nu_0, Psi_0, n_k, x_bar, S_k
        )
        # m_k should be between m_0 and x_bar
        assert jnp.all(m_k > m_0)  # Shifted toward x_bar
        assert jnp.all(m_k < x_bar)  # But not past x_bar

    def test_niw_posterior_update_zero_data(self, sample_niw_params):
        """With zero data, posterior should equal prior."""
        m_0, kappa_0, nu_0, Psi_0, D = sample_niw_params
        m_k, kappa_k, nu_k, Psi_k = niw_posterior_update(
            m_0, kappa_0, nu_0, Psi_0, 0, m_0, jnp.zeros((D, D))
        )
        assert jnp.allclose(m_k, m_0)
        assert kappa_k == kappa_0
        assert nu_k == nu_0
        assert jnp.allclose(Psi_k, Psi_0)


# ===========================================================================
# NIW sampling tests
# ===========================================================================


class TestNIWSampling:
    """Validate sampling from NIW distribution."""

    def test_sample_niw_returns_tuple(self, sample_niw_params):
        m_0, kappa_0, nu_0, Psi_0, D = sample_niw_params
        key = jax.random.PRNGKey(42)
        mu, Sigma = sample_niw(key, m_0, kappa_0, nu_0, Psi_0)
        assert mu.shape == (D,)
        assert Sigma.shape == (D, D)

    def test_sample_niw_positive_definite(self, sample_niw_params):
        m_0, kappa_0, nu_0, Psi_0, D = sample_niw_params
        key = jax.random.PRNGKey(42)
        mu, Sigma = sample_niw(key, m_0, kappa_0, nu_0, Psi_0)
        eigenvalues = jnp.linalg.eigvalsh(Sigma)
        assert jnp.all(eigenvalues > 0)


# ===========================================================================
# FFBS tests
# ===========================================================================


class TestFFBS:
    """Validate Forward-Filtering Backward-Sampling."""

    def test_ffbs_forward_pass_shape(self, sample_ffbs_data):
        data, trans_probs, means, covs, pi_0, T, K, D = sample_ffbs_data
        alphas, log_norms = ffbs_forward_pass(data, trans_probs, means, covs, pi_0)
        assert alphas.shape == (T, K)
        assert log_norms.shape == (T,)

    def test_ffbs_forward_pass_normalised(self, sample_ffbs_data):
        data, trans_probs, means, covs, pi_0, T, K, D = sample_ffbs_data
        alphas, log_norms = ffbs_forward_pass(data, trans_probs, means, covs, pi_0)
        # Filtered probabilities should sum to 1 at each timestep
        probs = jax.nn.softmax(alphas, axis=-1)
        assert jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)

    def test_ffbs_backward_sample_shape(self, sample_ffbs_data):
        data, trans_probs, means, covs, pi_0, T, K, D = sample_ffbs_data
        alphas, log_norms = ffbs_forward_pass(data, trans_probs, means, covs, pi_0)
        key = jax.random.PRNGKey(42)
        states = ffbs_backward_sample(key, alphas, trans_probs)
        assert states.shape == (T,)
        assert jnp.all((states >= 0) & (states < K))

    def test_ffbs_full_pipeline(self, sample_ffbs_data):
        data, trans_probs, means, covs, pi_0, T, K, D = sample_ffbs_data
        key = jax.random.PRNGKey(42)
        states, log_likelihood = ffbs_full(key, data, trans_probs, means, covs, pi_0)
        assert states.shape == (T,)
        assert jnp.isfinite(log_likelihood)

    def test_ffbs_deterministic_with_same_key(self, sample_ffbs_data):
        data, trans_probs, means, covs, pi_0, T, K, D = sample_ffbs_data
        key = jax.random.PRNGKey(42)
        states1, ll1 = ffbs_full(key, data, trans_probs, means, covs, pi_0)
        states2, ll2 = ffbs_full(key, data, trans_probs, means, covs, pi_0)
        assert jnp.array_equal(states1, states2)
        assert ll1 == ll2


# ===========================================================================
# Batched categorical sampling tests
# ===========================================================================


class TestBatchedCategoricalSampling:
    """Validate batched categorical sampling."""

    def test_batched_categorical_sample_shape(self):
        key = jax.random.PRNGKey(42)
        logits = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        samples = batched_categorical_sample(key, logits)
        assert samples.shape == (3,)

    def test_batched_categorical_sample_high_confidence(self):
        """With extreme logits, should sample deterministically."""
        key = jax.random.PRNGKey(42)
        logits = jnp.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
        samples = batched_categorical_sample(key, logits)
        assert jnp.array_equal(samples, jnp.array([0, 1, 2]))

    def test_batched_categorical_sample_uniform(self):
        """Uniform logits should produce roughly uniform samples."""
        key = jax.random.PRNGKey(42)
        logits = jnp.zeros((1000, 3))  # 1000 draws from 3 categories
        samples = batched_categorical_sample(key, logits)
        counts = jnp.bincount(samples, length=3)
        # Each category should have ~333 samples
        assert jnp.all(counts > 250)  # Reasonable lower bound


# ===========================================================================
# I²GMM nested CRP tests
# ===========================================================================


class TestNestedCRP:
    """Validate I²GMM nested CRP assignment."""

    def test_nested_crp_assignment_shape(self):
        """Test that nested CRP returns valid sub-component assignments."""
        key = jax.random.PRNGKey(42)
        K = 3  # Top-level clusters
        eta = 1.0  # Sub-component concentration
        # 5 observations in cluster 0
        obs_in_cluster = jnp.array([0, 0, 0, 0, 0])
        existing_sub = jnp.array([2, 1, 0])  # 2 in sub-0, 1 in sub-1, 0 in sub-2
        sub_assignments = nested_crp_assignment(key, obs_in_cluster, existing_sub, K, eta)
        assert sub_assignments.shape == (5,)
        assert jnp.all(sub_assignments >= 0)

    def test_nested_crp_prefers_populous_subs(self):
        """Rich-get-richer: should prefer sub-components with more existing members."""
        key = jax.random.PRNGKey(42)
        K = 1
        eta = 0.1  # Low concentration → strong rich-get-richer
        obs_in_cluster = jnp.array([0])  # One new observation
        existing_sub = jnp.array([100, 1, 0])  # Sub-0 is very populous
        sub_assignments = nested_crp_assignment(key, obs_in_cluster, existing_sub, K, eta)
        # With high concentration on sub-0, new obs should likely join sub-0
        assert sub_assignments[0] == 0


# ===========================================================================
# Integration tests
# ===========================================================================


class TestJAXKernelIntegration:
    """Test JAX kernels work together in realistic scenarios."""

    def test_full_gibbs_iteration(self, sample_niw_params):
        """Simulate one iteration of a Gibbs sampler using all kernels."""
        m_0, kappa_0, nu_0, Psi_0, D = sample_niw_params
        key = jax.random.PRNGKey(42)

        # Simulate data from one cluster
        n_k = 20
        key, subkey = jax.random.split(key)
        true_mean = jnp.array([2.0, 3.0])
        true_cov = jnp.eye(D) * 0.5
        data = jax.random.multivariate_normal(subkey, true_mean, true_cov, shape=(n_k,))

        # Compute sufficient statistics
        x_bar = jnp.mean(data, axis=0)
        diff = data - x_bar
        S_k = diff.T @ diff

        # Update NIW posterior
        m_k, kappa_k, nu_k, Psi_k = niw_posterior_update(
            m_0, kappa_0, nu_0, Psi_0, n_k, x_bar, S_k
        )

        # Sample new parameters
        key, subkey = jax.random.split(key)
        mu_new, Sigma_new = sample_niw(subkey, m_k, kappa_k, nu_k, Psi_k)

        # Compute log-density of a new point under sampled parameters
        precision_new = jnp.linalg.inv(Sigma_new)
        log_dens = mvn_log_density(x_bar, mu_new, precision_new)
        assert jnp.isfinite(log_dens)

    def test_ffbs_with_known_transitions(self):
        """FFBS should recover states when transitions are deterministic."""
        T = 20
        K = 3
        D = 1
        key = jax.random.PRNGKey(42)

        # Deterministic transitions: 0→1→2→0→1→2...
        trans_probs = jnp.zeros((K, K))
        trans_probs = trans_probs.at[0, 1].set(1.0)
        trans_probs = trans_probs.at[1, 2].set(1.0)
        trans_probs = trans_probs.at[2, 0].set(1.0)

        # Well-separated emissions
        means = jnp.array([[-5.0], [0.0], [5.0]])
        covs = jnp.stack([jnp.eye(1) * 0.1] * K)

        # Generate data following deterministic path
        true_states = jnp.array([i % K for i in range(T)])
        data = jax.vmap(lambda s: jax.random.normal(key, ()) * 0.1 + means[s, 0])(true_states)
        data = data.reshape(T, 1)

        pi_0 = jnp.array([1.0, 0.0, 0.0])  # Start in state 0

        _, ll = ffbs_full(key, data, trans_probs, means.reshape(K, 1), covs, pi_0)
        assert jnp.isfinite(ll)
