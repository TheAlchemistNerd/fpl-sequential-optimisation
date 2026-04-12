"""
Experience Replay Buffer — Ch. 9 §9.5.5.

Stores transitions (s_t, a_t, r_t, s_{t+1}, done) and samples uniformly
for DDQN training. Breaks correlation between consecutive samples.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Transition:
    """A single experience transition."""
    state: Any       # 757-dim state vector
    action: int      # action index
    reward: float    # immediate reward
    next_state: Any  # next state vector
    done: bool       # whether episode ended


class ReplayBuffer:
    """
    Fixed-size circular buffer for experience replay.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        """
        Sample a batch of transitions uniformly at random.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        List of Transition objects.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}"
            )
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """Whether the buffer has enough samples for meaningful training."""
        return len(self.buffer) >= 1000  # Minimum samples for first training
