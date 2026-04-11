"""
Tests for src/models/belief_state.py — Belief State Construction.

Tests cover:
    - BeliefState dataclass (probabilities, entropy, mode, concentration)
    - build_belief_states_from_mcmc
    - build_belief_states_from_posterior_predictive
    - FeatureToPointsMapper (linear + GBM)
    - NBPointsPosterior (sampling, PMF, dispersion fitting)
    - BeliefStateStore (persistence)
    - Convenience API (get_top_form_players, get_high_uncertainty_players)
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_mcmc_assignments():
    """Synthetic MCMC cluster assignments: 100 samples, 20 observations."""
    np.random.seed(42)
    S, N = 100, 20
    # Player 1 (obs 0-9): concentrated on cluster 0
    # Player 2 (obs 10-19): spread across clusters 0, 1, 2
    assignments = np.zeros((S, N), dtype=int)
    for s in range(S):
        # Player 1: 90% cluster 0, 10% cluster 1
        assignments[s, :10] = np.random.choice([0, 1], size=10, p=[0.9, 0.1])
        # Player 2: uniform across 0, 1, 2
        assignments[s, 10:] = np.random.choice([0, 1, 2], size=10)
    return assignments


@pytest.fixture
def sample_player_ids():
    """Player IDs for 20 observations: 10 each for players 1 and 2."""
    return np.array([1]*10 + [2]*10)


@pytest.fixture
def sample_gameweeks():
    """Gameweeks: obs 0-4 = GW1, obs 5-9 = GW2, etc."""
    return np.array([1]*5 + [2]*5 + [1]*5 + [2]*5)


@pytest.fixture
def sample_cluster_means():
    """Cluster means: 3 clusters, 4 features each."""
    return np.array([
        [2.0, 1.5, 0.5, 0.3],  # Cluster 0: Elite
        [1.0, 0.8, 0.3, 0.2],  # Cluster 1: Average
        [0.2, 0.1, 0.1, 0.05],  # Cluster 2: Cold
    ])


# ===========================================================================
# BeliefState dataclass tests
# ===========================================================================


class TestBeliefState:
    """Validate BeliefState dataclass properties."""

    def test_mode_cluster(self):
        from src.models.belief_state import BeliefState
        bs = BeliefState(
            player_id=1, gameweek=1,
            probabilities=np.array([0.7, 0.2, 0.1]),
        )
        assert bs.mode_cluster == 0

    def test_entropy_uniform(self):
        from src.models.belief_state import BeliefState
        bs = BeliefState(
            player_id=1, gameweek=1,
            probabilities=np.array([1/3, 1/3, 1/3]),
        )
        # Maximum entropy for 3 clusters: log(3) ≈ 1.099
        assert bs.entropy == pytest.approx(np.log(3), abs=0.01)

    def test_entropy_concentrated(self):
        from src.models.belief_state import BeliefState
        bs = BeliefState(
            player_id=1, gameweek=1,
            probabilities=np.array([0.99, 0.005, 0.005]),
        )
        assert bs.entropy < 0.1  # Very low entropy

    def test_is_concentrated_true(self):
        from src.models.belief_state import BeliefState
        bs = BeliefState(
            player_id=1, gameweek=1,
            probabilities=np.array([0.85, 0.1, 0.05]),
        )
        assert bs.is_concentrated is True

    def test_is_concentrated_false(self):
        from src.models.belief_state import BeliefState
        bs = BeliefState(
            player_id=1, gameweek=1,
            probabilities=np.array([0.4, 0.35, 0.25]),
        )
        assert bs.is_concentrated is False

    def test_expected_features_computed_automatically(self, sample_cluster_means):
        from src.models.belief_state import BeliefState
        probs = np.array([0.7, 0.2, 0.1])
        bs = BeliefState(
            player_id=1, gameweek=1,
            probabilities=probs,
            cluster_means=sample_cluster_means,
        )
        # E[x] = 0.7*[2,1.5,0.5,0.3] + 0.2*[1,0.8,0.3,0.2] + 0.1*[0.2,0.1,0.1,0.05]
        expected = 0.7 * sample_cluster_means[0] + 0.2 * sample_cluster_means[1] + 0.1 * sample_cluster_means[2]
        assert bs.expected_features == pytest.approx(expected, abs=1e-6)

    def test_to_dict(self):
        from src.models.belief_state import BeliefState
        bs = BeliefState(
            player_id=42, gameweek=5,
            probabilities=np.array([0.6, 0.3, 0.1]),
        )
        d = bs.to_dict()
        assert d["player_id"] == 42
        assert d["gameweek"] == 5
        assert len(d["probabilities"]) == 3
        assert "entropy" in d
        assert "mode_cluster" in d


# ===========================================================================
# Build belief states from MCMC tests
# ===========================================================================


class TestBuildBeliefStatesFromMCMC:
    """Validate belief state construction from MCMC assignments."""

    def test_returns_dict(self, sample_mcmc_assignments, sample_player_ids, sample_gameweeks):
        from src.models.belief_state import build_belief_states_from_mcmc
        result = build_belief_states_from_mcmc(
            sample_mcmc_assignments, sample_player_ids, sample_gameweeks,
        )
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_correct_keys(self, sample_mcmc_assignments, sample_player_ids, sample_gameweeks):
        from src.models.belief_state import build_belief_states_from_mcmc
        result = build_belief_states_from_mcmc(
            sample_mcmc_assignments, sample_player_ids, sample_gameweeks,
        )
        # Players 1,2 × GWs 1,2 = up to 4 belief states
        assert (1, 1) in result
        assert (1, 2) in result
        assert (2, 1) in result
        assert (2, 2) in result

    def test_player1_concentrated(self, sample_mcmc_assignments, sample_player_ids, sample_gameweeks):
        from src.models.belief_state import build_belief_states_from_mcmc
        result = build_belief_states_from_mcmc(
            sample_mcmc_assignments, sample_player_ids, sample_gameweeks,
        )
        # Player 1 should be concentrated on cluster 0
        bs = result[(1, 1)]
        assert bs.mode_cluster == 0
        assert bs.probabilities[0] > 0.5

    def test_player2_higher_entropy(self, sample_mcmc_assignments, sample_player_ids, sample_gameweeks):
        from src.models.belief_state import build_belief_states_from_mcmc
        result = build_belief_states_from_mcmc(
            sample_mcmc_assignments, sample_player_ids, sample_gameweeks,
        )
        bs_p1 = result[(1, 1)]
        bs_p2 = result[(2, 1)]
        # Player 2 has uniform assignments → higher entropy
        assert bs_p2.entropy > bs_p1.entropy

    def test_probabilities_sum_to_one(self, sample_mcmc_assignments, sample_player_ids, sample_gameweeks):
        from src.models.belief_state import build_belief_states_from_mcmc
        result = build_belief_states_from_mcmc(
            sample_mcmc_assignments, sample_player_ids, sample_gameweeks,
        )
        for bs in result.values():
            assert bs.probabilities.sum() == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# Feature-to-Points Mapper tests
# ===========================================================================


class TestFeatureToPointsMapper:
    """Validate feature-to-points mapping."""

    @pytest.fixture
    def sample_features(self):
        np.random.seed(42)
        return pd.DataFrame({
            "xG_per_90": np.random.uniform(0, 1, 100),
            "Form_5": np.random.uniform(0, 15, 100),
            "FDR": np.random.uniform(2, 5, 100),
        })

    @pytest.fixture
    def sample_points(self, sample_features):
        # Linear relationship + noise
        return pd.Series(
            5 * sample_features["xG_per_90"]
            + 0.5 * sample_features["Form_5"]
            - 0.3 * sample_features["FDR"]
            + np.random.randn(100) * 0.5
        )

    def test_linear_fit_predict(self, sample_features, sample_points):
        from src.models.belief_state import FeatureToPointsMapper
        mapper = FeatureToPointsMapper(method="linear")
        mapper.fit(sample_features, sample_points)
        preds = mapper.predict(sample_features.head(10))
        assert len(preds) == 10
        assert mapper._is_fit is True

    def test_gbm_fit_predict(self, sample_features, sample_points):
        from src.models.belief_state import FeatureToPointsMapper
        mapper = FeatureToPointsMapper(method="gbm", n_estimators=20)
        mapper.fit(sample_features, sample_points)
        preds = mapper.predict(sample_features.head(10))
        assert len(preds) == 10

    def test_predict_without_fit_raises(self, sample_features):
        from src.models.belief_state import FeatureToPointsMapper
        mapper = FeatureToPointsMapper(method="linear")
        with pytest.raises(RuntimeError, match="must be fitted"):
            mapper.predict(sample_features)

    def test_predict_from_belief_state(self, sample_features, sample_points, sample_cluster_means):
        from src.models.belief_state import FeatureToPointsMapper, BeliefState
        mapper = FeatureToPointsMapper(method="linear")
        mapper.fit(sample_features, sample_points)

        bs = BeliefState(
            player_id=1, gameweek=1,
            probabilities=np.array([0.7, 0.2, 0.1]),
            cluster_means=sample_cluster_means[:, :3],  # Match 3 features
        )
        # Need to set feature_names manually for this test
        mapper.feature_names = ["xG_per_90", "Form_5", "FDR"]
        pred = mapper.predict_from_belief_state(bs)
        assert isinstance(pred, float)
        assert pred > 0


# ===========================================================================
# NBPointsPosterior tests
# ===========================================================================


class TestNBPointsPosterior:
    """Validate Negative-Binomial points posterior."""

    def test_p_computation(self):
        from src.models.belief_state import NBPointsPosterior
        nb = NBPointsPosterior(player_id=1, r=5.0, expected_points=10.0)
        # p = r / (r + μ) = 5 / 15 = 0.333
        assert nb.p == pytest.approx(5 / 15, abs=1e-4)

    def test_sample_returns_non_negative_int(self):
        from src.models.belief_state import NBPointsPosterior
        nb = NBPointsPosterior(player_id=1, r=5.0, expected_points=10.0)
        rng = np.random.default_rng(42)
        sample = nb.sample(rng)
        assert isinstance(sample, int)
        assert sample >= 0

    def test_mean_equals_expected_points(self):
        from src.models.belief_state import NBPointsPosterior
        nb = NBPointsPosterior(player_id=1, r=5.0, expected_points=10.0)
        # Mean ≈ r(1-p)/p = 5*(10/15)/(5/15) = 10
        assert nb.mean() == pytest.approx(10.0, abs=0.1)

    def test_variance_positive(self):
        from src.models.belief_state import NBPointsPosterior
        nb = NBPointsPosterior(player_id=1, r=5.0, expected_points=10.0)
        assert nb.variance() > 0

    def test_fit_nb_dispersion_overdispersed(self):
        from src.models.belief_state import fit_nb_dispersion
        # Over-dispersed data: var > mean
        data = np.array([0, 1, 2, 3, 5, 8, 12, 15, 2, 1, 0, 3, 7, 10, 4])
        r = fit_nb_dispersion(data)
        assert r > 0

    def test_fit_nb_dispersion_underdispersed_fallback(self):
        from src.models.belief_state import fit_nb_dispersion
        # Under-dispersed: var < mean
        data = np.array([5, 5, 5, 5, 5])
        r = fit_nb_dispersion(data)
        assert r == 100.0  # Fallback to Poisson


# ===========================================================================
# BeliefStateStore tests
# ===========================================================================


class TestBeliefStateStore:
    """Validate belief state persistence."""

    def test_save_and_load(self, tmp_path):
        from src.models.belief_state import BeliefStateStore, BeliefState
        store = BeliefStateStore(base_dir=tmp_path / "belief_states")
        belief_states = {
            (1, 1): BeliefState(1, 1, np.array([0.7, 0.2, 0.1])),
            (2, 1): BeliefState(2, 1, np.array([0.3, 0.4, 0.3])),
        }
        path = store.save(belief_states, gameweek=1)
        assert path.exists()

        loaded = store.load(gameweek=1)
        assert len(loaded) == 2
        assert (1, 1) in loaded
        assert (2, 1) in loaded

    def test_load_nonexistent_file(self, tmp_path):
        from src.models.belief_state import BeliefStateStore
        store = BeliefStateStore(base_dir=tmp_path / "belief_states")
        result = store.load(gameweek=99)
        assert result == {}


# ===========================================================================
# Convenience API tests
# ===========================================================================


class TestConvenienceAPI:
    """Test convenience functions."""

    def test_get_belief_state_found(self):
        from src.models.belief_state import BeliefState, get_belief_state
        belief_states = {
            (1, 1): BeliefState(1, 1, np.array([0.7, 0.2, 0.1])),
        }
        result = get_belief_state(1, 1, belief_states)
        assert result is not None
        assert result.player_id == 1

    def test_get_belief_state_not_found(self):
        from src.models.belief_state import BeliefState, get_belief_state
        belief_states = {
            (1, 1): BeliefState(1, 1, np.array([0.7, 0.2, 0.1])),
        }
        result = get_belief_state(99, 1, belief_states)
        assert result is None

    def test_get_top_form_players(self):
        from src.models.belief_state import BeliefState, get_top_form_players
        belief_states = {
            (1, 1): BeliefState(1, 1, np.array([0.9, 0.05, 0.05])),  # mode=0, max_prob=0.9
            (2, 1): BeliefState(2, 1, np.array([0.1, 0.8, 0.1])),   # mode=1, max_prob=0.8
            (3, 1): BeliefState(3, 1, np.array([0.05, 0.05, 0.9])),  # mode=2, max_prob=0.9
        }
        result = get_top_form_players(gameweek=1, belief_states=belief_states, top_k=2)
        assert len(result) == 2
        # All have high concentration; player 1 has mode=0 (highest prob cluster)
        assert result[0].mode_cluster in [0, 2]  # Players 1 or 3 have highest mode_cluster index

    def test_get_high_uncertainty_players(self):
        from src.models.belief_state import BeliefState, get_high_uncertainty_players
        belief_states = {
            (1, 1): BeliefState(1, 1, np.array([0.99, 0.005, 0.005])),  # Low entropy
            (2, 1): BeliefState(2, 1, np.array([0.33, 0.34, 0.33])),   # High entropy
        }
        result = get_high_uncertainty_players(gameweek=1, belief_states=belief_states, top_k=1)
        assert len(result) == 1
        assert result[0].player_id == 2  # Highest entropy


# ===========================================================================
# Posterior predictive tests
# ===========================================================================


class TestPosteriorPredictive:
    """Test belief state construction from posterior predictive samples."""

    def test_build_from_posterior_predictive(self):
        from src.models.belief_state import build_belief_states_from_posterior_predictive
        np.random.seed(42)
        S, N = 50, 10
        samples = np.random.normal(5, 2, size=(S, N))
        player_ids = np.array([1]*5 + [2]*5)
        gameweeks = np.array([1]*5 + [2]*5)

        result = build_belief_states_from_posterior_predictive(
            samples, player_ids, gameweeks,
        )
        assert isinstance(result, dict)
        assert len(result) > 0
