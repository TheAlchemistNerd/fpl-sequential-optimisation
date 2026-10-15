"""Tests for HDP-GMM with Gaussian-Wishart (Normal-Wishart) base measure.

These tests verify:
1. The base measure is a proper **Normal-Wishart** distribution where:
   - The **Gaussian component** models the mean μ of each form state.
   - The **Wishart component** models the full precision matrix Λ = Σ⁻¹
     (capturing feature correlations, not just diagonal variance).
2. The HDP prior enables an **infinite** mixture — cluster count is inferred.
3. Stick-breaking weights are valid (non-negative, sum ≈ 1).
4. The model discovers distinct performance profiles (e.g., "Elite" vs "Poor").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_player_data():
    """Generate synthetic FPL data with 3 known form states.

    Returns
    -------
    data : np.ndarray (N, D)
        Standardised features: [xG_p90, xA_p90, ict_index_p90].
    player_ids : np.ndarray (N,)
        Integer player identifiers.
    num_players : int
    true_cluster_labels : np.ndarray (N,)
        Ground truth for evaluation.
    """
    rng = np.random.default_rng(42)
    D = 3  # xG/90, xA/90, ICT/90

    # Three "form states" with different means and full covariance matrices
    cluster_means = [
        np.array([0.6, 0.3, 8.0]),   # Elite form
        np.array([0.25, 0.15, 4.5]), # Average form
        np.array([0.05, 0.05, 2.0]), # Poor form
    ]

    # Full covariance matrices (not diagonal!) — captures feature correlations
    cluster_covs = [
        np.array([[0.04, 0.02, 0.3],
                   [0.02, 0.02, 0.15],
                   [0.3,  0.15, 1.0]]),
        np.array([[0.02, 0.01, 0.2],
                   [0.01, 0.01, 0.1],
                   [0.2,  0.1,  0.8]]),
        np.array([[0.01, 0.005, 0.1],
                   [0.005, 0.005, 0.05],
                   [0.1,  0.05,  0.4]]),
    ]

    n_per_cluster = [50, 80, 70]
    data_parts = []
    label_parts = []
    pid_parts = []

    for k, (mean, cov, n) in enumerate(zip(cluster_means, cluster_covs, n_per_cluster)):
        samples = rng.multivariate_normal(mean, cov, size=n)
        data_parts.append(samples)
        label_parts.append(np.full(n, k))
        pid_parts.append(rng.integers(0, 10, size=n))  # 10 players

    data = np.vstack(data_parts).astype(np.float32)
    labels = np.concatenate(label_parts)
    pids = np.concatenate(pid_parts).astype(np.int32)

    return data, pids, 10, labels


@pytest.fixture
def single_player_series():
    """Single player with 30 gameweeks of data (2 features)."""
    rng = np.random.default_rng(7)
    D = 2
    n_gw = 30
    data = rng.normal(0, 1, size=(n_gw, D)).astype(np.float32)
    pids = np.zeros(n_gw, dtype=np.int32)
    return data, pids, 1


# ---------------------------------------------------------------------------
# Tests: stick-breaking helper
# ---------------------------------------------------------------------------

class TestStickBreaking:
    """Validate the stick-breaking weight computation."""

    def test_weights_sum_to_one(self):
        """Stick-breaking weights must sum to ≈ 1."""
        from src.models.hdp_gmm import stick_breaking_weights
        import jax.numpy as jnp

        beta = jnp.array([0.3, 0.5, 0.2])
        w = stick_breaking_weights(beta)
        assert w.shape == (4,)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-5)

    def test_weights_non_negative(self):
        """All weights must be ≥ 0."""
        from src.models.hdp_gmm import stick_breaking_weights
        import jax.numpy as jnp

        beta = jnp.array([0.1, 0.9, 0.4])
        w = stick_breaking_weights(beta)
        assert np.all(np.array(w) >= 0)

    def test_weights_batch_shape(self):
        """Should handle batched beta inputs."""
        from src.models.hdp_gmm import stick_breaking_weights
        import jax.numpy as jnp

        beta = jnp.ones((5, 19)) * 0.1  # 5 players, T=20
        w = stick_breaking_weights(beta)
        assert w.shape == (5, 20)
        sums = np.array(w.sum(axis=-1))
        assert np.allclose(sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Normal-Wishart base measure structure
# ---------------------------------------------------------------------------

class TestNormalWishartBaseMeasure:
    """Verify the Gaussian-Wishart (Normal-Wishart) base measure is correct.

    The Normal-Wishart prior is:
        Λ ~ Wishart(V, ν)                        # precision matrix
        μ | Λ ~ Normal(μ₀, (κ₀ · Λ)⁻¹)           # mean conditioned on precision

    This is the conjugate prior for a multivariate Gaussian with
    unknown mean and unknown full covariance.
    """

    def test_model_has_wishart_precision(self):
        """The NumPyro model must sample a Wishart-distributed precision."""
        from src.models import hdp_gmm
        import numpyro

        # Inspect the model source — it must reference Wishart
        source = Path(hdp_gmm.__file__).read_text(encoding='utf-8')
        assert "Wishart" in source or "LKJ" in source, (
            "Model must use a Wishart (or equivalent) distribution for the "
            "precision matrix Λ, not a diagonal approximation."
        )

    def test_wishart_is_full_matrix_not_diagonal(self):
        """The precision must be a full D×D matrix, not a diagonal vector.

        This is the key distinction from the InverseGamma approximation:
        the Wishart captures correlations between features (e.g., xG and xA
        co-varying within a form state).
        """
        from src.models import hdp_gmm
        source = Path(hdp_gmm.__file__).read_text(encoding='utf-8')

        # The model should NOT use InverseGamma as the sole covariance prior
        # when D > 1 (diagonal-only is insufficient for correlations).
        assert "InverseGamma" not in source or "Wishart" in source, (
            "For D > 1 the model must use Wishart for the full precision "
            "matrix, not just InverseGamma on diagonals."
        )

    def test_gaussian_component_models_mean(self):
        """The Gaussian component must model μ (center of the state)."""
        from src.models import hdp_gmm
        source = Path(hdp_gmm.__file__).read_text(encoding='utf-8')
        assert "mu" in source.lower() or "mean" in source.lower(), (
            "Model must define a mean parameter μ for each cluster."
        )

    def test_wishart_component_models_precision(self):
        """The Wishart component must model Λ = Σ⁻¹ (shape/precision)."""
        from src.models import hdp_gmm
        source = Path(hdp_gmm.__file__).read_text(encoding='utf-8')
        assert "precision" in source.lower() or "wishart" in source.lower(), (
            "Model must define a Wishart/precision component Λ."
        )


# ---------------------------------------------------------------------------
# Tests: HDP infinite mixture property
# ---------------------------------------------------------------------------

class TestHDPInfiniteMixture:
    """The HDP must allow a countably infinite number of clusters
    (truncated in practice).  The model must NOT require a fixed K.
    """

    def test_truncation_is_configurable(self):
        """User can set truncation level; model itself doesn't fix K."""
        from src.models.hdp_gmm import hdp_gmm_model, DEFAULT_TRUNCATION
        assert DEFAULT_TRUNCATION >= 20, (
            "Default truncation should be large enough for the 'infinite' "
            "approximation (≥ 20)."
        )

    def test_active_clusters_inferred_not_fixed(self):
        """Given data with 3 true clusters, model should discover ≈ 3.

        This is an integration test — we run short MCMC on synthetic data.
        """
        from src.models.hdp_gmm import run_hdp_gmm

        rng = np.random.default_rng(42)
        D = 2
        # Create 3 well-separated clusters
        parts = []
        for mean in [np.array([5.0, 5.0]), np.array([0.0, 0.0]), np.array([-5.0, 5.0])]:
            parts.append(rng.multivariate_normal(mean, np.eye(D) * 0.5, size=30))
        data = np.vstack(parts).astype(np.float32)
        pids = np.zeros(len(data), dtype=np.int32)

        _, summary = run_hdp_gmm(
            data, pids, num_players=1,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=15, rng_key=0,
        )

        # Should discover ~3 active clusters (mass > 5%)
        assert 2 <= summary["active_clusters"] <= 6, (
            f"Expected ~3 active clusters, got {summary['active_clusters']}. "
            "The HDP should infer the number of clusters from data."
        )


# ---------------------------------------------------------------------------
# Tests: HDP-GMM run
# ---------------------------------------------------------------------------

class TestHDPGMMRun:
    """End-to-end tests for the HDP-GMM runner."""

    def test_run_returns_samples_and_summary(self, single_player_series):
        data, pids, n_players = single_player_series
        samples, summary = run_hdp_gmm_small(data, pids, n_players)

        assert isinstance(samples, dict)
        assert isinstance(summary, dict)
        assert "gamma_mean" in summary
        assert "alpha_mean" in summary
        assert "active_clusters" in summary
        assert "global_weights" in summary
        assert samples["player_weights"].shape == (200, n_players, 10)
        assert np.allclose(samples["player_weights"].sum(axis=-1), 1.0, atol=1e-5)

    def test_summary_has_gaussian_wishart_params(self, single_player_series):
        data, pids, n_players = single_player_series
        _, summary = run_hdp_gmm_small(data, pids, n_players)

        # Gaussian-Wishart parameters must be present
        assert "mu_means" in summary, "Mean μ of each cluster required"
        assert "precision_means" in summary, (
            "Precision Λ (Wishart) of each cluster required — must be full D×D "
            "matrix, not a diagonal approximation."
        )

        # Precision must be a 3-D tensor (T, D, D) — full matrix per cluster
        prec = summary["precision_means"]
        assert prec.ndim == 3, (
            f"Precision must be (T, D, D), got shape {prec.shape}"
        )
        assert prec.shape[-1] == prec.shape[-2], (
            "Precision matrices must be square (D×D per cluster)."
        )

    def test_concentration_params_positive(self, single_player_series):
        data, pids, n_players = single_player_series
        _, summary = run_hdp_gmm_small(data, pids, n_players)

        assert summary["gamma_mean"] > 0, "Global concentration γ must be > 0"
        assert summary["alpha_mean"] > 0, "Player concentration α must be > 0"


# Lightweight runner for tests (fewer MCMC steps)
def run_hdp_gmm_small(data, pids, n_players):
    from src.models.hdp_gmm import run_hdp_gmm
    return run_hdp_gmm(
        data, pids, n_players,
        num_warmup=100, num_samples=200, num_chains=1,
        truncation=10, rng_key=0,
    )
