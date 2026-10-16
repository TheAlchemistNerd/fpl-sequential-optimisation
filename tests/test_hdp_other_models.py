"""Tests for HDP-Poisson-Gamma and HDP-Negative-Binomial models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# HDP-Poisson-Gamma
# ===========================================================================

@pytest.fixture
def count_data():
    """Synthetic count data: 50 observations, 3 event types (goals, assists, cs)."""
    rng = np.random.default_rng(42)
    # Poisson rates: [goals, assists, clean_sheets]
    rates = np.array([
        [0.5, 0.3, 0.1],   # Elite
        [0.15, 0.1, 0.05],  # Average
        [0.02, 0.02, 0.01], # Poor
    ])
    labels = rng.choice(3, size=60, p=[0.3, 0.4, 0.3])
    counts = np.column_stack([
        rng.poisson(rates[l, 0]) for l in labels
    ])
    # Create multi-event data
    data = np.zeros((60, 3), dtype=np.float32)
    for i, l in enumerate(labels):
        data[i, 0] = rng.poisson(rates[l, 0])
        data[i, 1] = rng.poisson(rates[l, 1])
        data[i, 2] = rng.poisson(rates[l, 2])
    pids = np.zeros(60, dtype=np.int32)
    return data, pids, 1


class TestHDPPoissonGamma:
    """Validate HDP-Poisson-Gamma model."""

    def test_model_uses_poisson_likelihood(self):
        from src.models import hdp_poisson_gamma
        source = Path(hdp_poisson_gamma.__file__).read_text()
        assert "Poisson" in source, "Model must use Poisson likelihood."

    def test_model_uses_gamma_prior(self):
        from src.models import hdp_poisson_gamma
        source = Path(hdp_poisson_gamma.__file__).read_text()
        assert "Gamma" in source, "Base measure must be Gamma for Poisson rate."

    def test_run_returns_lambda_means(self, count_data):
        from src.models.hdp_poisson_gamma import run_hdp_poisson_gamma
        data, pids, n_players = count_data

        samples, summary = run_hdp_poisson_gamma(
            data, pids, n_players,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        assert "lambda_means" in summary
        # lambda_means should be (T, E) — per-cluster per-event rates
        lam = summary["lambda_means"]
        assert lam.ndim == 2
        assert lam.shape[-1] == 3  # 3 event types

    def test_rates_are_non_negative(self, count_data):
        from src.models.hdp_poisson_gamma import run_hdp_poisson_gamma
        data, pids, n_players = count_data

        _, summary = run_hdp_poisson_gamma(
            data, pids, n_players,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        lam = summary["lambda_means"]
        assert np.all(lam >= 0), "Poisson rates must be non-negative."


# ===========================================================================
# HDP-Negative-Binomial
# ===========================================================================

@pytest.fixture
def points_data():
    """Synthetic FPL points: over-dispersed count data."""
    rng = np.random.default_rng(42)
    # Negative Binomial: mixture of high-variance states
    # State 1: mean=8, high dispersion (elite, volatile)
    # State 2: mean=4, low dispersion (average, consistent)
    # State 3: mean=1, low dispersion (poor, consistent)
    n1, n2, n3 = 30, 40, 30
    pts1 = rng.negative_binomial(n=3, p=0.3, size=n1)  # mean ≈ 7
    pts2 = rng.negative_binomial(n=5, p=0.5, size=n2)  # mean ≈ 5
    pts3 = rng.negative_binomial(n=8, p=0.85, size=n3)  # mean ≈ 1.3
    points = np.concatenate([pts1, pts2, pts3]).astype(np.float32)
    pids = np.zeros(len(points), dtype=np.int32)
    return points, pids, 1


class TestHDPNegativeBinomial:
    """Validate HDP-Negative-Binomial model."""

    def test_run_rejects_negative_points_before_sampling(self):
        from src.models.hdp_negbin import run_hdp_negbin

        with pytest.raises(ValueError, match="cannot represent signed raw FPL"):
            run_hdp_negbin(
                np.array([2, -1]), np.array([0, 0]), 1,
                num_warmup=1, num_samples=1,
            )

    def test_run_rejects_noninteger_points_before_sampling(self):
        from src.models.hdp_negbin import run_hdp_negbin

        with pytest.raises(ValueError, match="non-negative integers"):
            run_hdp_negbin(
                np.array([2.5, 1.0]), np.array([0, 0]), 1,
                num_warmup=1, num_samples=1,
            )

    def test_model_uses_negbin_likelihood(self):
        from src.models import hdp_negbin
        source = Path(hdp_negbin.__file__).read_text()
        assert "NegativeBinomial" in source or "NegBin" in source, (
            "Model must use Negative Binomial likelihood."
        )

    def test_model_uses_beta_gamma_priors(self):
        from src.models import hdp_negbin
        source = Path(hdp_negbin.__file__).read_text()
        assert "Beta" in source, "Prior on p must be Beta."
        assert "Gamma" in source, "Prior on r must be Gamma."

    def test_run_returns_expected_points(self, points_data):
        from src.models.hdp_negbin import run_hdp_negbin
        points, pids, n_players = points_data

        samples, summary = run_hdp_negbin(
            points, pids, n_players,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        assert "expected_points" in summary
        assert "p_means" in summary
        assert "r_means" in summary

    def test_expected_points_are_positive(self, points_data):
        from src.models.hdp_negbin import run_hdp_negbin
        points, pids, n_players = points_data

        _, summary = run_hdp_negbin(
            points, pids, n_players,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        ep = summary["expected_points"]
        assert np.all(ep >= 0), "Expected points must be non-negative."
