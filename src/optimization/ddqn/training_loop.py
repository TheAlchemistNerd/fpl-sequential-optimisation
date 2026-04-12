"""
DDQN Training Loop — Ch. 9 §9.6.3.

Implements the unified HDP-HMM + DDQN training algorithm:
    1. Run HDP-HMM inference → belief states
    2. Construct state from beliefs + squad + chips
    3. Select action via Boltzmann policy
    4. Execute action, observe reward and next state
    5. Store transition in replay buffer
    6. Sample minibatch and perform DDQN update:
        a. Online network selects best action
        b. Target network evaluates it
        c. Compute TD error and backpropagate
    7. Update target network periodically

The DDQN decouples action selection from evaluation:
    a* = argmax_a' Q_online(s', a'; θ)        # online selects
    y = r + γ · Q_target(s', a*; θ') · (1-done)  # target evaluates
    loss = (y - Q_online(s, a; θ))²
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.optimization.ddqn.q_network import QNetwork, TargetNetwork
from src.optimization.ddqn.replay_buffer import ReplayBuffer, Transition
from src.optimization.ddqn.boltzmann_policy import boltzmann_policy, anneal_temperature

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_td_targets(
    batch: list[Transition],
    online_net: QNetwork,
    target_net: TargetNetwork,
    gamma: float = 0.99,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute DDQN TD targets with decoupled action selection/evaluation.

    y = r + γ · Q_target(s', argmax_a' Q_online(s', a'; θ); θ') · (1 - done)

    Parameters
    ----------
    batch : list[Transition]
        Minibatch of transitions.
    online_net : QNetwork
        Online (current) Q-network.
    target_net : TargetNetwork
        Target Q-network.
    gamma : float
        Discount factor.
    device : str

    Returns
    -------
    torch.Tensor — TD targets, shape (batch_size,).
    """
    states = torch.stack([t.next_state for t in batch]).to(device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

    # Online network selects best action
    q_online_next = online_net(states)
    best_actions = q_online_next.argmax(dim=1)  # (batch,)

    # Target network evaluates selected action
    q_target_next = target_net(states)
    q_next = q_target_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)

    # TD targets
    targets = rewards + gamma * q_next * (1 - dones)
    return targets


def train_step(
    batch: list[Transition],
    online_net: QNetwork,
    target_net: TargetNetwork,
    optimizer: optim.Optimizer,
    gamma: float = 0.99,
    device: str = "cpu",
) -> float:
    """
    Perform a single DDQN training step.

    Parameters
    ----------
    batch : list[Transition]
        Minibatch of transitions.
    online_net : QNetwork
    target_net : TargetNetwork
    optimizer : optim.Optimizer
    gamma : float
    device : str

    Returns
    -------
    float — loss value.
    """
    states = torch.stack([t.state for t in batch]).to(device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device).unsqueeze(1)
    targets = compute_td_targets(batch, online_net, target_net, gamma, device)

    # Forward pass
    q_values = online_net(states)
    q_selected = q_values.gather(1, actions).squeeze(1)

    # Compute loss (Huber loss for robustness)
    loss = nn.SmoothL1Loss()(q_selected, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())


def ddqn_training_loop(
    env: Any,  # FPLEnvironment
    online_net: QNetwork,
    target_net: TargetNetwork,
    replay_buffer: ReplayBuffer,
    n_episodes: int = 100,
    batch_size: int = 64,
    gamma: float = 0.99,
    learning_rate: float = 1e-4,
    target_update_freq: int = 1000,
    tau: float = 0.001,  # soft update rate
    min_buffer_size: int = 1000,
    train_freq: int = 4,
    tau_init: float = 2.0,
    tau_final: float = 0.1,
    device: str = "cpu",
) -> dict[str, list]:
    """
    Full DDQN training loop.

    Parameters
    ----------
    env : FPLEnvironment
    online_net : QNetwork
    target_net : TargetNetwork
    replay_buffer : ReplayBuffer
    n_episodes : int
        Number of training episodes.
    batch_size : int
        Minibatch size for training.
    gamma : float
        Discount factor.
    learning_rate : float
    target_update_freq : int
        Steps between hard target updates.
    tau : float
        Soft update rate (if using soft updates).
    min_buffer_size : int
        Minimum buffer size before training starts.
    train_freq : int
        Training frequency (every N steps).
    tau_init : float
        Initial Boltzmann temperature.
    tau_final : float
        Final Boltzmann temperature.
    device : str

    Returns
    -------
    dict with training metrics: losses, rewards, episode_lengths.
    """
    optimizer = optim.Adam(online_net.parameters(), lr=learning_rate)
    target_net.hard_update_from(online_net)

    metrics: dict[str, list] = {
        "losses": [],
        "rewards": [],
        "episode_lengths": [],
    }

    step_count = 0

    for episode in range(n_episodes):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        episode_reward = 0.0
        episode_length = 0

        while not env.done:
            # Select action via Boltzmann policy
            temperature = anneal_temperature(
                env.gameweek + 1, 38, tau_init, tau_final
            )
            with torch.no_grad():
                q_vals = online_net(state_tensor).squeeze(0).cpu().numpy()
            action = boltzmann_policy(q_vals, temperature, np.random.default_rng())

            # Execute action
            next_state, reward, done, info = env.step(action)
            next_state_tensor = torch.tensor(
                next_state, dtype=torch.float32, device=device
            )

            # Store transition
            replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)

            # Train
            if len(replay_buffer) >= min_buffer_size and step_count % train_freq == 0:
                batch = replay_buffer.sample(batch_size)
                loss = train_step(batch, online_net, target_net, optimizer, gamma, device)
                metrics["losses"].append(loss)

            # Update target network (soft update every step, hard every N)
            target_net.soft_update_from(online_net, tau)
            if step_count % target_update_freq == 0:
                target_net.hard_update_from(online_net)
                logger.info(
                    "Episode %d, Step %d: target network updated",
                    episode, step_count,
                )

            state_tensor = next_state_tensor
            episode_reward += reward
            episode_length += 1
            step_count += 1

        metrics["rewards"].append(episode_reward)
        metrics["episode_lengths"].append(episode_length)

        if episode % 10 == 0:
            logger.info(
                "Episode %d/%d: reward=%.1f, length=%d, buffer=%d",
                episode, n_episodes, episode_reward, episode_length, len(replay_buffer),
            )

    return metrics
