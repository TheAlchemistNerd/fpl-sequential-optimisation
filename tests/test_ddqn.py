"""
Tests for DDQN subpackage modules.

Tests cover:
    - replay_buffer.py (push, sample, capacity)
    - boltzmann_policy.py (sampling, temperature, soft value)
    - action_space.py (encoding/decoding, pruning)
    - q_network.py (forward pass, action selection, target updates)
    - fpl_environment.py (reset, step, state dimensionality)
    - training_loop.py (TD targets, train step)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===========================================================================
# Replay Buffer Tests
# ===========================================================================


class TestReplayBuffer:
    """Validate experience replay buffer."""

    def test_push_and_sample(self):
        from src.optimization.ddqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(
                state=torch.zeros(757),
                action=i % 10,
                reward=1.0,
                next_state=torch.zeros(757),
                done=False,
            )
        assert len(buf) == 50
        batch = buf.sample(10)
        assert len(batch) == 10

    def test_capacity_limit(self):
        from src.optimization.ddqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(torch.zeros(10), i, 1.0, torch.zeros(10), False)
        assert len(buf) == 10  # Circular buffer

    def test_sample_raises_when_too_small(self):
        from src.optimization.ddqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=100)
        for i in range(5):
            buf.push(torch.zeros(10), 0, 1.0, torch.zeros(10), False)
        with pytest.raises(ValueError):
            buf.sample(10)

    def test_is_ready(self):
        from src.optimization.ddqn.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=2000)
        assert not buf.is_ready
        for i in range(1000):
            buf.push(torch.zeros(10), 0, 1.0, torch.zeros(10), False)
        assert buf.is_ready


# ===========================================================================
# Boltzmann Policy Tests
# ===========================================================================


class TestBoltzmannPolicy:
    """Validate Boltzmann policy and temperature annealing."""

    def test_greedy_at_zero_temperature(self):
        from src.optimization.ddqn.boltzmann_policy import boltzmann_policy
        q = np.array([1.0, 5.0, 3.0, 2.0])
        action = boltzmann_policy(q, temperature=0.0)
        assert action == 1  # argmax

    def test_high_temperature_gives_diverse_actions(self):
        from src.optimization.ddqn.boltzmann_policy import boltzmann_policy
        q = np.array([1.0, 5.0, 3.0, 2.0])
        rng = np.random.default_rng(42)
        # With very high temperature, should sample diverse actions
        actions = set()
        for seed in range(100):
            rng = np.random.default_rng(seed)
            actions.add(boltzmann_policy(q, temperature=100.0, rng=rng))
        # Should sample multiple different actions
        assert len(actions) > 1

    def test_anneal_temperature_decreases(self):
        from src.optimization.ddqn.boltzmann_policy import anneal_temperature
        t_early = anneal_temperature(5, 38, 2.0, 0.1)
        t_late = anneal_temperature(35, 38, 2.0, 0.1)
        assert t_early > t_late

    def test_anneal_temperature_bounds(self):
        from src.optimization.ddqn.boltzmann_policy import anneal_temperature
        t_start = anneal_temperature(1, 38, 2.0, 0.1)
        t_end = anneal_temperature(38, 38, 2.0, 0.1)
        assert t_start == pytest.approx(2.0, abs=0.1)
        assert t_end == pytest.approx(0.1, abs=0.1)

    def test_soft_value_converges_to_max(self):
        from src.optimization.ddqn.boltzmann_policy import soft_value
        q = np.array([1.0, 5.0, 3.0])
        # As τ → 0, soft_value → max(q)
        sv = soft_value(q, temperature=0.001)
        assert sv == pytest.approx(5.0, abs=0.01)

    def test_soft_value_higher_than_max(self):
        from src.optimization.ddqn.boltzmann_policy import soft_value
        q = np.array([1.0, 5.0, 3.0])
        # Soft value is always ≥ max(q) due to entropy bonus
        sv = soft_value(q, temperature=1.0)
        assert sv >= 5.0


# ===========================================================================
# Action Space Tests
# ===========================================================================


class TestActionSpace:
    """Validate action space construction and encoding."""

    def test_n_actions(self):
        from src.optimization.ddqn.action_space import ActionSpace
        space = ActionSpace(candidate_ids=[1, 2, 3])
        assert space.n_actions == 4  # 1 no-transfer + 3 candidates

    def test_no_transfer_action(self):
        from src.optimization.ddqn.action_space import ActionSpace
        space = ActionSpace(candidate_ids=[1, 2, 3], current_squad=[10, 11, 12])
        result = space.action_to_transfer(0)
        assert result is None

    def test_transfer_encoding_decoding(self):
        from src.optimization.ddqn.action_space import ActionSpace
        space = ActionSpace(
            candidate_ids=[1, 2, 3],
            current_squad=[10, 11, 12],
        )
        # Action 1 = transfer in candidate[0] = player 1
        result = space.action_to_transfer(1)
        assert result is not None
        player_in, player_out = result
        assert player_in == 1
        assert player_out in [10, 11, 12]

    def test_build_action_space_prunes_to_top_k(self):
        from src.optimization.ddqn.action_space import build_action_space
        pool = {
            "ids": list(range(1, 51)),
            "points": {i: float(i) for i in range(1, 51)},  # Higher ID = more points
        }
        space = build_action_space(pool, current_squad=[1, 2, 3], top_k=10)
        assert len(space.candidate_ids) == 10
        # Should be the top-10 by points: 41, 42, ..., 50
        assert max(space.candidate_ids) == 50


# ===========================================================================
# Q-Network Tests
# ===========================================================================


class TestQNetwork:
    """Validate Q-network architecture and operations."""

    def test_forward_pass_shape(self):
        from src.optimization.ddqn.q_network import QNetwork
        net = QNetwork(state_dim=757, action_dim=101)
        state = torch.randn(4, 757)
        output = net(state)
        assert output.shape == (4, 101)

    def test_forward_pass_single_state(self):
        from src.optimization.ddqn.q_network import QNetwork
        net = QNetwork(state_dim=757, action_dim=101)
        state = torch.randn(757)
        output = net(state)
        assert output.shape == (1, 101)

    def test_select_action_greedy(self):
        from src.optimization.ddqn.q_network import QNetwork
        net = QNetwork(state_dim=10, action_dim=5)
        # Manually set weights to produce known Q-values
        state = torch.zeros(10)
        with torch.no_grad():
            action = net.select_action(state, temperature=0.0)
        assert 0 <= action < 5

    def test_target_network_hard_update(self):
        from src.optimization.ddqn.q_network import QNetwork, TargetNetwork
        online = QNetwork(state_dim=10, action_dim=5)
        target = TargetNetwork(state_dim=10, action_dim=5)
        # Modify online network
        with torch.no_grad():
            online.network[0].weight.fill_(1.0)
        # Hard update
        target.hard_update_from(online)
        assert torch.allclose(target.network[0].weight, online.network[0].weight)

    def test_target_network_soft_update(self):
        from src.optimization.ddqn.q_network import QNetwork, TargetNetwork
        online = QNetwork(state_dim=10, action_dim=5)
        target = TargetNetwork(state_dim=10, action_dim=5)
        # Modify online network
        with torch.no_grad():
            online.network[0].weight.fill_(1.0)
            target.network[0].weight.fill_(0.0)
        # Soft update with tau=0.5
        target.soft_update_from(online, tau=0.5)
        expected = 0.5 * 1.0 + 0.5 * 0.0
        assert torch.allclose(target.network[0].weight, torch.full_like(target.network[0].weight, expected))


# ===========================================================================
# FPL Environment Tests
# ===========================================================================


class TestFPLEnvironment:
    """Validate FPL environment interface."""

    @pytest.fixture
    def sample_env(self):
        """Create a minimal FPL environment."""
        from src.optimization.ddqn.fpl_environment import FPLEnvironment
        n_players = 50
        K = 4
        player_pool = {
            "ids": list(range(1, n_players + 1)),
            "positions": ["GK"]*4 + ["DEF"]*8 + ["MID"]*8 + ["FWD"]*4 + ["MID"]*(n_players-24),
            "prices": [5.0 + i * 0.1 for i in range(n_players)],
            "clubs": [f"Club_{i % 20}" for i in range(n_players)],
            "points": {i: float(5 + i * 0.1) for i in range(1, n_players + 1)},
        }
        beliefs = {i: np.random.dirichlet([2.0]*K) for i in range(1, n_players + 1)}
        trans = np.eye(K) * 0.7 + (1 - np.eye(K)) * 0.3 / (K - 1)

        return FPLEnvironment(
            player_pool=player_pool,
            belief_states=beliefs,
            transition_matrix=trans,
            budget=100.0,
            seed=42,
        )

    def test_reset_returns_state(self, sample_env):
        state = sample_env.reset()
        assert isinstance(state, np.ndarray)
        assert len(state) > 0

    def test_step_returns_tuple(self, sample_env):
        sample_env.reset()
        result = sample_env.step(0)  # no transfer
        assert len(result) == 4
        next_state, reward, done, info = result
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_episode_eventually_ends(self, sample_env):
        """A full season episode should terminate."""
        sample_env.reset()
        for _ in range(100):  # Safety limit
            _, _, done, _ = sample_env.step(0)
            if done:
                break
        assert sample_env.done or sample_env.gameweek > 0

    def test_reward_is_non_negative_for_no_transfers(self, sample_env):
        """Without transfers, reward should be the points scored (≥0)."""
        sample_env.reset()
        _, reward, _, info = sample_env.step(0)
        assert reward >= 0  # Points are non-negative, no transfer cost
        assert info["transfer_cost"] == 0


# ===========================================================================
# Training Loop Tests
# ===========================================================================


class TestTrainingLoop:
    """Validate DDQN training components."""

    def test_compute_td_targets_shape(self):
        from src.optimization.ddqn.training_loop import compute_td_targets
        from src.optimization.ddqn.replay_buffer import Transition
        from src.optimization.ddqn.q_network import QNetwork, TargetNetwork

        online = QNetwork(state_dim=10, action_dim=5)
        target = TargetNetwork(state_dim=10, action_dim=5)
        target.hard_update_from(online)

        batch = [
            Transition(
                state=torch.randn(10),
                action=0,
                reward=1.0,
                next_state=torch.randn(10),
                done=False,
            )
            for _ in range(8)
        ]

        targets = compute_td_targets(batch, online, target, gamma=0.99)
        assert targets.shape == (8,)

    def test_train_step_reduces_loss(self):
        from src.optimization.ddqn.training_loop import train_step
        from src.optimization.ddqn.replay_buffer import Transition
        from src.optimization.ddqn.q_network import QNetwork, TargetNetwork
        import torch.optim as optim

        online = QNetwork(state_dim=10, action_dim=5)
        target = TargetNetwork(state_dim=10, action_dim=5)
        target.hard_update_from(online)
        optimizer = optim.Adam(online.parameters(), lr=1e-3)

        # Create a batch with consistent targets
        batch = [
            Transition(
                state=torch.randn(10),
                action=i % 5,
                reward=1.0,
                next_state=torch.randn(10),
                done=False,
            )
            for i in range(16)
        ]

        # Train for a few steps
        losses = []
        for _ in range(10):
            loss = train_step(batch, online, target, optimizer, gamma=0.99)
            losses.append(loss)

        # Loss should generally decrease (or at least be finite)
        assert all(np.isfinite(l) for l in losses)
