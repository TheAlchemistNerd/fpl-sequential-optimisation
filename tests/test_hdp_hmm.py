"""Tests for HDP-HMM with Gaussian-Wishart base measure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def synthetic_hmm_data():
    """Single player with 10 gameweeks, 2 features, 2 discrete actions."""
    rng = np.random.default_rng(42)
    N = 10
    D = 2
    A = 2
    data = rng.normal(0, 1, size=(N, D)).astype(np.float32)
    pids = np.zeros(N, dtype=np.int32)
    gw_idx = np.arange(N, dtype=np.int32)
    # Binary 1-hot actions (e.g. [1,0] = No action, [0,1] = Transfer)
    actions = np.zeros((N, A), dtype=np.float32)
    actions[:, 0] = 1.0 # default to neutral action
    return data, pids, gw_idx, actions, 1, 10


class TestHDPHMMStructure:
    """Validate the HDP-HMM model structure."""

    def test_model_has_transition_probs(self):
        """HDP-HMM must define transition probabilities between states."""
        from src.models import hdp_hmm
        source = Path(hdp_hmm.__file__).read_text(encoding="utf-8")
        assert "trans" in source.lower(), (
            "HDP-HMM must define transition probabilities."
        )

    def test_model_has_gaussian_wishart_emissions(self):
        """Emissions must use Gaussian-Wishart (not diagonal approx)."""
        from src.models import hdp_hmm
        source = Path(hdp_hmm.__file__).read_text(encoding="utf-8")
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
        data, pids, gw, actions, n_players, max_gw = synthetic_hmm_data

        samples, summary = run_hdp_hmm(
            data, pids, actions,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        assert "active_clusters" in summary
        assert "mu_means" in summary
        assert "precision_means" in summary
        assert "trans_probs_mean" in summary
        assert "alpha_mean" in summary

    def test_transition_matrix_shape(self, synthetic_hmm_data):
        from src.models.hdp_hmm import run_hdp_hmm
        data, pids, gw, actions, n_players, max_gw = synthetic_hmm_data

        _, summary = run_hdp_hmm(
            data, pids, actions,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        T = 8
        A = 2
        transitions = summary["trans_probs_mean"]
        assert transitions.shape == (A, T, T), (
            f"Transition probabilities must be ({A}, {T}, {T}), got {transitions.shape}"
        )
        assert np.allclose(transitions.sum(axis=-1), 1.0, atol=1e-5)

    def test_precision_is_full_matrix(self, synthetic_hmm_data):
        from src.models.hdp_hmm import run_hdp_hmm
        data, pids, gw, actions, n_players, max_gw = synthetic_hmm_data

        _, summary = run_hdp_hmm(
            data, pids, actions,
            num_warmup=100, num_samples=200, num_chains=1,
            truncation=8, rng_key=0,
        )

        prec = summary["precision_means"]
        assert prec.ndim == 3, (
            f"Precision must be (T, D, D), got {prec.shape}"
        )


def test_prepare_hmm_sequences_supports_fixed_padding():
    from src.utils.jax_helpers import prepare_hmm_sequences

    data = np.arange(10, dtype=np.float32).reshape(5, 2)
    player_ids = np.array([0, 0, 1, 1, 1], dtype=np.int32)
    actions = np.ones((5, 1), dtype=np.float32)
    padded, padded_actions, mask = prepare_hmm_sequences(
        data, player_ids, actions, max_timesteps=5
    )
    assert padded.shape == (2, 5, 2)
    assert padded_actions.shape == (2, 5, 1)
    assert mask.sum(axis=1).tolist() == [2.0, 3.0]
    with pytest.raises(ValueError, match="smaller than observed"):
        prepare_hmm_sequences(data, player_ids, actions, max_timesteps=2)
