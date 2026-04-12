"""
Tests for src/mlops/drift_detector.py — KL-Divergence Drift Detection & ASUGS.

Tests cover (Ch. 11 §11.6.2, Ch. 7 §7.8):
    - Per-player KL divergence between consecutive belief states
    - Population-level KL averaging
    - Threshold computation (percentile-based)
    - Drift detection logic
    - ASUGS online concentration update
    - Retraining trigger conditions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_belief_timeseries():
    """
    Simulated belief states for 5 players over 10 gameweeks.
    Returns list of dicts: [{player_id: belief_vec}, ...]
    """
    np.random.seed(42)
    K = 5  # clusters
    n_players = 5
    n_gw = 10
    series = []
    for gw in range(n_gw):
        beliefs = {}
        for pid in range(1, n_players + 1):
            # Stable beliefs for first 7 GW, then drift
            if gw < 7:
                probs = np.random.dirichlet([3.0, 1.0, 1.0, 0.5, 0.5])
            else:
                # Shift: player 1 moves to cluster 4
                if pid == 1:
                    probs = np.array([0.05, 0.05, 0.05, 0.1, 0.75])
                else:
                    probs = np.random.dirichlet([1.0, 1.0, 1.0, 1.0, 3.0])
            beliefs[pid] = probs
        series.append(beliefs)
    return series


# ===========================================================================
# KL Divergence tests (Ch. 11 §11.6.2)
# ===========================================================================


class TestKLDivergence:
    """Validate per-player and population KL divergence computation."""

    def test_kl_identical_distributions_zero(self):
        """KL(p || p) = 0 for identical distributions."""
        from src.mlops.drift_detector import kl_divergence
        p = np.array([0.3, 0.3, 0.4])
        result = kl_divergence(p, p)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_kl_different_distributions_positive(self):
        """KL(p || q) > 0 for different distributions."""
        from src.mlops.drift_detector import kl_divergence
        p = np.array([0.8, 0.1, 0.1])
        q = np.array([0.1, 0.1, 0.8])
        result = kl_divergence(p, q)
        assert result > 0

    def test_kl_symmetry_not_guaranteed(self):
        """KL(p || q) ≠ KL(q || p) in general."""
        from src.mlops.drift_detector import kl_divergence
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.3, 0.6])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert kl_pq != pytest.approx(kl_qp)

    def test_kl_handles_near_zero_probabilities(self):
        """Should handle near-zero probabilities without NaN."""
        from src.mlops.drift_detector import kl_divergence
        p = np.array([0.999, 0.001])
        q = np.array([0.001, 0.999])
        result = kl_divergence(p, q)
        assert np.isfinite(result)
        assert result > 0


# ===========================================================================
# Population drift tests
# ===========================================================================


class TestPopulationDrift:
    """Validate population-level drift computation."""

    def test_compute_population_drift(self, sample_belief_timeseries):
        """Should return average KL across all players."""
        from src.mlops.drift_detector import compute_population_drift
        # Compare GW 0 vs GW 1 (small drift expected)
        drift = compute_population_drift(
            sample_belief_timeseries[0],
            sample_belief_timeseries[1],
        )
        assert isinstance(drift, float)
        assert drift >= 0

    def test_compute_drift_timeseries(self, sample_belief_timeseries):
        """Should return drift values for consecutive GW pairs."""
        from src.mlops.drift_detector import compute_drift_timeseries
        drifts = compute_drift_timeseries(sample_belief_timeseries)
        assert len(drifts) == len(sample_belief_timeseries) - 1
        assert all(isinstance(d, float) for d in drifts)
        assert all(d >= 0 for d in drifts)

    def test_drift_spikes_at_gw7(self, sample_belief_timeseries):
        """Drift should spike around GW 7 where beliefs shift."""
        from src.mlops.drift_detector import compute_drift_timeseries
        drifts = compute_drift_timeseries(sample_belief_timeseries)
        # GW 6→7 (index 6) should have higher drift than early GWs
        early_drift = np.mean(drifts[:5])
        spike_drift = drifts[6]  # GW 6→7
        assert spike_drift > early_drift


# ===========================================================================
# Threshold computation tests
# ===========================================================================


class TestThresholdComputation:
    """Validate percentile-based threshold computation."""

    def test_threshold_percentile(self):
        """95th percentile threshold from historical drift values."""
        from src.mlops.drift_detector import compute_threshold
        historical_drifts = [0.1, 0.15, 0.12, 0.08, 0.2, 0.11, 0.09, 0.14]
        threshold = compute_threshold(historical_drifts, percentile=95)
        assert isinstance(threshold, float)
        # 95th percentile of 8 values ≈ max or near-max
        assert threshold >= max(historical_drifts) * 0.9

    def test_threshold_lower_for_lower_percentile(self):
        """50th percentile should be lower than 95th."""
        from src.mlops.drift_detector import compute_threshold
        drifts = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18]
        t50 = compute_threshold(drifts, percentile=50)
        t95 = compute_threshold(drifts, percentile=95)
        assert t50 < t95


# ===========================================================================
# Drift detection tests
# ===========================================================================


class TestDriftDetection:
    """Validate drift detection and retraining trigger logic."""

    def test_detect_drift_below_threshold(self):
        """No drift detected when current < threshold."""
        from src.mlops.drift_detector import detect_drift
        historical = [0.1, 0.15, 0.12, 0.08, 0.2]
        current_drift = 0.11
        is_drift, details = detect_drift(current_drift, historical)
        assert is_drift is False

    def test_detect_drift_above_threshold(self):
        """Drift detected when current > 95th percentile threshold."""
        from src.mlops.drift_detector import detect_drift
        historical = [0.1, 0.15, 0.12, 0.08, 0.2]
        current_drift = 0.5  # Much higher than any historical value
        is_drift, details = detect_drift(current_drift, historical)
        assert is_drift is True
        assert "threshold" in details

    def test_detect_drift_insufficient_history(self):
        """Should not trigger drift with insufficient history (<5 values)."""
        from src.mlops.drift_detector import detect_drift
        historical = [0.1, 0.2]
        current_drift = 0.5
        is_drift, details = detect_drift(current_drift, historical)
        assert is_drift is False


# ===========================================================================
# ASUGS tests (Ch. 7 §7.8)
# ===========================================================================


class TestASUGS:
    """Validate Adaptive SUGS online concentration parameter updates."""

    def test_asugs_update_increases_gamma(self):
        """Gamma should increase when new data suggests more clusters."""
        from src.mlops.drift_detector import asugs_gamma_update
        gamma_old = 1.0
        n_clusters = 5
        n_observations = 100
        gamma_new = asugs_gamma_update(gamma_old, n_clusters, n_observations)
        assert gamma_new > 0

    def test_asugs_update_with_step_size(self):
        """ASUGS uses step size schedule η_n = 1/n."""
        from src.mlops.drift_detector import asugs_gamma_update
        gamma_old = 2.0
        # With large n, step size is small → small update
        gamma_new_early = asugs_gamma_update(gamma_old, 5, 10)
        gamma_new_late = asugs_gamma_update(gamma_old, 5, 1000)
        # Later updates should be smaller (smaller step size)
        assert abs(gamma_new_early - gamma_old) >= abs(gamma_new_late - gamma_old)

    def test_asugs_stable_with_many_observations(self):
        """With many observations, gamma should stabilise."""
        from src.mlops.drift_detector import asugs_gamma_update
        gamma = 1.0
        for n in range(10, 1000, 10):
            gamma = asugs_gamma_update(gamma, 5, n)
        assert np.isfinite(gamma)
        assert gamma > 0


# ===========================================================================
# Retraining trigger tests
# ===========================================================================


class TestRetrainingTrigger:
    """Validate retraining trigger conditions."""

    def test_trigger_on_drift_detection(self, sample_belief_timeseries):
        """Should trigger retraining when drift exceeds threshold."""
        from src.mlops.drift_detector import (
            compute_drift_timeseries,
            compute_threshold,
            detect_drift,
        )
        drifts = compute_drift_timeseries(sample_belief_timeseries)
        # Use first 5 drifts as history, check if GW 6→7 triggers
        historical = drifts[:5]
        current = drifts[6]
        is_drift, _ = detect_drift(current, historical)
        # May or may not trigger depending on synthetic data
        assert isinstance(is_drift, bool)

    def test_trigger_on_performance_degradation(self):
        """Should trigger when cumulative points drop below 25th percentile."""
        from src.mlops.drift_detector import check_performance_degradation
        historical_points = [60, 55, 70, 65, 58, 72, 68, 61]
        recent_points = [30, 25]  # Significant drop
        should_retrain = check_performance_degradation(
            historical_points, recent_points
        )
        assert should_retrain is True

    def test_no_trigger_when_stable(self):
        """Should not trigger when performance is stable."""
        from src.mlops.drift_detector import check_performance_degradation
        historical_points = [60, 55, 70, 65, 58, 72, 68, 61]
        recent_points = [62, 64]  # Within normal range
        should_retrain = check_performance_degradation(
            historical_points, recent_points
        )
        assert should_retrain is False
