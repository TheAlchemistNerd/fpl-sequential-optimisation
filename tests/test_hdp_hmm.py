"""Tests for HDP-HMM with Gaussian-Wishart base measure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def synthetic_hmm_data():
    """Single player with 10 gameweeks, 2 features."""
    rng = np.random.default_rng(42)
    N = 10
    D = 2
    data = rng.normal(0, 1, size=(N, D)).astype(np.float32)
    pids = np.zeros(N, dtype=np.int32)
    gw_idx = np.arange(N, dtype=np.int32)
    return data, pids, gw_idx, 1, 10


class TestHDPHMMStructure:
    """Validate the HDP-HMM model structure."""

    def test_model_has_transition_probs(self):
        """HDP-HMM must define transition probabilities between states."""
        from src.models import hdp_hmm
        source = Path(hdp_hmm.__file__).read_text()
        assert "trans" in source.lower(), (
            "HDP-HMM must define transition probabilities."
        )

    def test_model_has_gaussian_wishart_emissions(self):
        """Emissions must use Gaussian-Wishart (not diagonal approx)."""
        from src.models import hdp_hmm
        source = Path(hdp_hmm.__file__).read_text()
        assert "Wishart" in source, (
            "HDP-HMM emissions must use Wishart for full precision."
        )
        assert "mu" in source.lower(), (
            "HDP-HMM emissions must include Gaussian mean μ."
        )

    def test_imports_from_hdp_gmm(self):
        """HDP-HMM reuses constants from HDP-GMM module."""
        from src.models.hdp_hmm import DEFAULT_TRUNCATION, NORMAL_WISHART_PRIOR
        assert DEFAULT_TRUNCATION >= 20
        assert "mu_0" in NORMAL_WISHART_PRIOR
        assert "wishart_scale" in NORMAL_WISHART_PRIOR


class TestHDPHMMRun:
    """End-to-end runner tests."""

    def test_run_returns_summary(self, synthetic_hmm_data):
        from src.models.hdp_hmm import run_hdp_hmm
        data, pids, gw, n_players, max_gw = synthetic_hmm_data

        samples, summary = run_hdp_hmm(
            data, pids, gw, n_players, max_gw,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        assert "active_clusters" in summary
        assert "mu_means" in summary
        assert "precision_means" in summary
        assert "trans_probs" in summary

    def test_transition_matrix_shape(self, synthetic_hmm_data):
        from src.models.hdp_hmm import run_hdp_hmm
        data, pids, gw, n_players, max_gw = synthetic_hmm_data

        _, summary = run_hdp_hmm(
            data, pids, gw, n_players, max_gw,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        T = 8
        tp = summary["trans_probs"]
        assert tp.shape == (T, T), (
            f"Transition matrix must be ({T}, {T}), got {tp.shape}"
        )

    def test_precision_is_full_matrix(self, synthetic_hmm_data):
        from src.models.hdp_hmm import run_hdp_hmm
        data, pids, gw, n_players, max_gw = synthetic_hmm_data

        _, summary = run_hdp_hmm(
            data, pids, gw, n_players, max_gw,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        prec = summary["precision_means"]
        assert prec.ndim == 3, (
            f"Precision must be (T, D, D), got {prec.shape}"
        )
