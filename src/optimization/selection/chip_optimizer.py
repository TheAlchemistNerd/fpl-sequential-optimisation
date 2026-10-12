"""
Chip Deployment Optimiser — Ch. 10 §10.10.

Multi-armed bandit over remaining gameweeks for chip timing:
    - Wildcard: Unlimited free transfers (reset squad without cost)
    - Free Hit: Replace entire squad for one GW, revert next GW
    - Bench Boost: Points from all 4 bench players added to total
    - Triple Captain: Captain's points tripled (instead of doubled)

Each chip is a separate 4-armed bandit where arms = remaining gameweeks.
The reward for deploying a chip at GW t is evaluated by simulating the
squad's performance with and without the chip effect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Chip types
CHIP_WILDCARD = "wildcard"
CHIP_FREE_HIT = "free_hit"
CHIP_BENCH_BOOST = "bench_boost"
CHIP_TRIPLE_CAPTAIN = "triple_captain"
ALL_CHIPS = [CHIP_WILDCARD, CHIP_FREE_HIT, CHIP_BENCH_BOOST, CHIP_TRIPLE_CAPTAIN]


@dataclass
class ChipReward:
    """Reward estimate for deploying a specific chip at a specific gameweek."""
    chip: str
    gameweek: int
    expected_gain: float  # Additional points compared to no-chip baseline


class ChipDeploymentOptimiser:
    """
    Optimiser for chip deployment timing.

    Models each chip type as a separate bandit over remaining gameweeks.
    Uses Thompson Sampling to select the optimal deployment gameweek.

    Parameters
    ----------
    n_gameweeks : int
        Number of remaining gameweeks to consider.
    """

    def __init__(self, n_gameweeks: int = 38) -> None:
        self.n_gameweeks = n_gameweeks
        # Posterior parameters for each chip at each gameweek
        # θ_{chip, gw} ~ N(μ, 1/τ)
        self.mu: dict[str, np.ndarray] = {}
        self.tau: dict[str, np.ndarray] = {}  # Precision (1/variance)
        self.n_observations: dict[str, np.ndarray] = {}

        for chip in ALL_CHIPS:
            self.mu[chip] = np.zeros(n_gameweeks)
            self.tau[chip] = np.ones(n_gameweeks)  # Prior precision = 1
            self.n_observations[chip] = np.zeros(n_gameweeks)

        # Track which chips are still available
        self.available_chips: set[str] = set(ALL_CHIPS)

    def select_chip(self, rng: np.random.Generator) -> ChipReward | None:
        """
        Select the best chip and gameweek using Thompson Sampling.

        For each available chip, sample from the posterior at each gameweek
        and select the (chip, gw) pair with the highest sampled reward.

        Parameters
        ----------
        rng : np.random.Generator

        Returns
        -------
        ChipReward or None if no chips available.
        """
        if not self.available_chips:
            return None

        best_chip = None
        best_gw = -1
        best_reward = -float("inf")

        for chip in self.available_chips:
            # Sample from posterior at each gameweek
            sigma = 1.0 / np.sqrt(self.tau[chip] + 1e-10)
            sampled_rewards = rng.normal(self.mu[chip], sigma)

            # Find best gameweek for this chip
            gw = int(np.argmax(sampled_rewards))
            reward = sampled_rewards[gw]

            if reward > best_reward:
                best_reward = reward
                best_chip = chip
                best_gw = gw

        if best_chip is None:
            return None

        return ChipReward(
            chip=best_chip,
            gameweek=best_gw,
            expected_gain=float(best_reward),
        )

    def update(
        self,
        chip: str,
        gameweek: int,
        observed_gain: float,
    ) -> None:
        """
        Update the posterior after observing the actual gain from a chip.

        Uses conjugate Normal-Normal updating:
            τ ← τ + 1
            μ ← (τ·μ + r) / (τ + 1)

        Parameters
        ----------
        chip : str
            The chip that was deployed.
        gameweek : int
            The gameweek it was deployed at.
        observed_gain : float
            The actual additional points gained vs. no-chip baseline.
        """
        if chip not in self.mu:
            raise ValueError(f"Unknown chip: {chip}")
        if gameweek < 0 or gameweek >= self.n_gameweeks:
            raise ValueError(f"Invalid gameweek: {gameweek}")

        # Conjugate update
        self.tau[chip][gameweek] += 1.0
        self.mu[chip][gameweek] = (
            (self.tau[chip][gameweek] - 1.0) * self.mu[chip][gameweek] + observed_gain
        ) / self.tau[chip][gameweek]
        self.n_observations[chip][gameweek] += 1

        logger.info(
            "Chip update: %s at GW %d, gain=%.1f, μ=%.2f, τ=%.2f",
            chip, gameweek, observed_gain,
            self.mu[chip][gameweek], self.tau[chip][gameweek],
        )

    def mark_chip_used(self, chip: str) -> None:
        """
        Mark a chip as used (no longer available for deployment).
        """
        if chip in self.available_chips:
            self.available_chips.remove(chip)

    def estimate_double_gw_reward(
        self,
        chip: str,
        gameweek: int,
        base_points: float,
        bench_points: float = 0.0,
        captain_points: float = 0.0,
    ) -> float:
        """
        Estimate the reward for deploying a chip at a double gameweek.

        Parameters
        ----------
        chip : str
            The chip being considered.
        gameweek : int
            The gameweek index.
        base_points : float
            Expected points without the chip.
        bench_points : float
            Expected points from bench players (for Bench Boost).
        captain_points : float
            Expected captain points (for Triple Captain).

        Returns
        -------
        float — estimated additional gain.
        """
        if chip == CHIP_BENCH_BOOST:
            return bench_points  # Gain = bench points
        elif chip == CHIP_TRIPLE_CAPTAIN:
            return captain_points  # Gain = additional captain points (3x instead of 2x)
        elif chip == CHIP_FREE_HIT:
            # Gain from accessing premium players temporarily
            return base_points * 0.2  # Rough estimate: 20% improvement
        elif chip == CHIP_WILDCARD:
            # Gain from squad reset (harder to estimate)
            return base_points * 0.15  # Rough estimate: 15% improvement
        return 0.0
