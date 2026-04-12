"""
Captaincy MAB — Ch. 10 §10.7.

Contextual Multi-Armed Bandit for captaincy selection.
Maintains a posterior over reward function parameters for each candidate,
and selects the captain using either LinUCB or Contextual Thompson Sampling.

The context vector for each arm includes:
    - Player belief state (K-dim distribution over latent form clusters)
    - Fixture difficulty (FDR)
    - Home/away status
    - Rolling form (3, 5, 10 game averages)
    - Sentiment score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LinUCBArm:
    """
    A single arm in the LinUCB contextual bandit.

    Maintains the posterior parameters for the linear reward model:
        r = x^T θ + ε,  ε ~ N(0, σ²)

    Posterior: θ ~ N(μ, A⁻¹)
    """
    d: int  # context dimension
    A: np.ndarray = field(default=None)  # d×d precision matrix
    b: np.ndarray = field(default=None)  # d×1 reward-weighted context sum
    mu: np.ndarray = field(default=None)  # d×1 posterior mean

    def __post_init__(self) -> None:
        if self.A is None:
            self.A = np.eye(self.d)
        if self.b is None:
            self.b = np.zeros(self.d)
        if self.mu is None:
            self.mu = np.zeros(self.d)

    def update(self, context: np.ndarray, reward: float) -> None:
        """
        Update posterior after observing a reward.

        A ← A + x·x^T
        b ← b + r·x
        μ ← A⁻¹ b
        """
        x = context.reshape(-1)
        self.A += np.outer(x, x)
        self.b += reward * x
        self.mu = np.linalg.solve(self.A, self.b)

    def ucb_score(self, context: np.ndarray, alpha: float = 1.0) -> float:
        """
        Compute LinUCB score: x^T μ + α · √(x^T A⁻¹ x)

        Parameters
        ----------
        context : np.ndarray (d,)
        alpha : float
            Exploration parameter.

        Returns
        -------
        float — UCB score.
        """
        x = context.reshape(-1)
        A_inv = np.linalg.inv(self.A)
        expected = x @ self.mu
        uncertainty = alpha * np.sqrt(x @ A_inv @ x)
        return float(expected + uncertainty)

    def sample_theta(self, context: np.ndarray, rng: np.random.Generator) -> float:
        """
        Sample from posterior and compute predicted reward.

        θ ~ N(μ, A⁻¹)
        r̂ = x^T θ

        Used for Contextual Thompson Sampling.
        """
        x = context.reshape(-1)
        A_inv = np.linalg.inv(self.A)
        # Sample from multivariate normal
        theta = rng.multivariate_normal(self.mu, A_inv)
        return float(x @ theta)


class CaptaincyBandit:
    """
    Contextual bandit for captaincy selection.

    Each arm corresponds to a candidate player. The context vector encodes
    the player's belief state, fixture difficulty, form, and sentiment.

    Parameters
    ----------
    n_candidates : int
        Number of candidate players (arms).
    context_dim : int
        Dimension of the context vector.
    alpha : float
        LinUCB exploration parameter.
    """

    def __init__(
        self,
        n_candidates: int,
        context_dim: int,
        alpha: float = 1.0,
    ) -> None:
        self.n_candidates = n_candidates
        self.context_dim = context_dim
        self.alpha = alpha
        self.arms = [LinUCBArm(context_dim) for _ in range(n_candidates)]
        self.total_pulls = 0

    def select_linucb(self, contexts: np.ndarray) -> int:
        """
        Select captain using LinUCB (upper confidence bound).

        Parameters
        ----------
        contexts : np.ndarray (n_candidates, context_dim)

        Returns
        -------
        int — index of selected captain.
        """
        scores = [
            arm.ucb_score(contexts[i], self.alpha)
            for i, arm in enumerate(self.arms)
        ]
        return int(np.argmax(scores))

    def select_thompson(self, contexts: np.ndarray, rng: np.random.Generator) -> int:
        """
        Select captain using Contextual Thompson Sampling.

        Parameters
        ----------
        contexts : np.ndarray (n_candidates, context_dim)
        rng : np.random.Generator

        Returns
        -------
        int — index of selected captain.
        """
        scores = [
            arm.sample_theta(contexts[i], rng)
            for i, arm in enumerate(self.arms)
        ]
        return int(np.argmax(scores))

    def update(self, captain_index: int, context: np.ndarray, reward: float) -> None:
        """
        Update the posterior for the selected captain.

        Parameters
        ----------
        captain_index : int
            Index of the selected captain.
        context : np.ndarray (context_dim,)
            Context vector at the time of selection.
        reward : float
            Observed reward (captain's doubled points).
        """
        self.arms[captain_index].update(context, reward)
        self.total_pulls += 1

    def build_context(
        self,
        belief_state: np.ndarray,
        fdr: float,
        is_home: bool,
        form_3: float,
        form_5: float,
        sentiment: float = 0.0,
    ) -> np.ndarray:
        """
        Build the context vector for a candidate player.

        Parameters
        ----------
        belief_state : np.ndarray (K,)
            Belief distribution over latent form clusters.
        fdr : float
            Fixture difficulty rating.
        is_home : bool
            Whether the player is at home.
        form_3 : float
            3-game rolling average.
        form_5 : float
            5-game rolling average.
        sentiment : float
            News sentiment score.

        Returns
        -------
        np.ndarray — context vector.
        """
        return np.concatenate([
            belief_state,
            np.array([fdr, float(is_home), form_3, form_5, sentiment]),
        ])
